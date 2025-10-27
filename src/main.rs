mod config;
mod generator;

use std::collections::HashSet;
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use atty::Stream;
use chrono::{DateTime, Local};
use clap::{ArgAction, CommandFactory, Parser, Subcommand, ValueEnum};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{seq::SliceRandom, thread_rng};
use rpassword::read_password;
use tracing::{info, warn};

use crate::config::{load as load_config, save_default_env};
use crate::generator::{CommandConfidence, GeneratedCommand, generate_command};

#[derive(Parser, Debug)]
#[command(
    name = "task",
    author,
    version,
    about = "Generate safe shell commands from natural language prompts",
    long_about = "task is a CLI assistant that converts natural language descriptions into shell commands using OpenAI-backed intelligence.",
    after_help = "EXAMPLES:\n  task gen \"list large files\" --shell zsh -v\n  echo \"list staged changes\" | task gen --verbose\n\nCONFIG:\n  ~/.task.toml    Default configuration file.\n  --config         Override configuration path.\n\nENVIRONMENT:\n  OPENAI_API_KEY           Required for live command generation\n  TASK_SH_FAKE_RESPONSE    Optional testing override.",
    propagate_version = true
)]
struct Cli {
    /// Optional config file path
    #[arg(long, value_name = "PATH")]
    config: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate a shell command from a description
    Gen {
        /// Natural language description of the desired task
        description: Option<String>,

        /// Target shell flavor for the generated command
        #[arg(long)]
        shell: Option<Shell>,

        /// Include verbose explanation of the suggested command
        #[arg(short, long, action = ArgAction::SetTrue)]
        verbose: bool,

        /// Override the system prompt sent to the model
        #[arg(long, value_name = "PROMPT")]
        system_prompt: Option<String>,

        /// Override the model name used for generation
        #[arg(long, value_name = "MODEL")]
        model: Option<String>,

        /// Disable progress spinner even if enabled in config
        #[arg(long, action = ArgAction::SetFalse)]
        spinner: Option<bool>,
    },

    /// Generate shell autocompletion scripts
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Shell {
    Bash,
    Zsh,
}

impl Shell {
    fn as_str(&self) -> &'static str {
        match self {
            Shell::Bash => "bash",
            Shell::Zsh => "zsh",
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    init_tracing();

    let cli = Cli::parse();

    if needs_api_key(&cli) {
        ensure_required_env()?;
    }

    let config_path = cli.config.as_ref().map(|p| p.into());
    let app_config = load_config(config_path)?;

    let result = match cli.command {
        Commands::Gen {
            description,
            shell,
            verbose,
            system_prompt,
            model,
            spinner,
        } => {
            let effective_verbose = verbose || app_config.verbose.unwrap_or(false);
            handle_generate(
                description,
                shell.or_else(|| {
                    app_config
                        .default_shell
                        .as_deref()
                        .and_then(Shell::from_str_case_insensitive)
                }),
                effective_verbose,
                system_prompt.or(app_config.system_prompt.clone()),
                model.or(app_config.model.clone()),
                spinner.unwrap_or_else(|| app_config.spinner.unwrap_or(true)),
            )
            .await
        }
        Commands::Completions { shell } => {
            generate_completions(shell);
            Ok(())
        }
    };

    match result {
        Ok(()) => Ok(()),
        Err(err) => {
            eprintln!("{}", format!("Error: {:#}", err).red());
            Err(err)
        }
    }
}

async fn handle_generate(
    description: Option<String>,
    shell: Option<Shell>,
    verbose: bool,
    system_prompt: Option<String>,
    model: Option<String>,
    spinner_enabled: bool,
) -> Result<()> {
    let prompt = match description {
        Some(desc) if !desc.trim().is_empty() => desc,
        Some(_) | None => {
            let stdin_value = read_stdin()?;
            match stdin_value {
                Some(value) => {
                    info!("Read description from stdin");
                    value
                }
                None => {
                    warn!("No description provided via argument or stdin");
                    String::new()
                }
            }
        }
    };

    let shell = shell.unwrap_or(Shell::Bash);

    let spinner = if spinner_enabled {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("{spinner} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        pb.set_message("Generating command...");
        pb.enable_steady_tick(Duration::from_millis(120));
        Some(pb)
    } else {
        None
    };

    let GeneratedCommand {
        cmd,
        explanation,
        raw_response,
        confidence,
        alternatives,
    } = generate_command(
        prompt.trim(),
        shell.as_str(),
        system_prompt.as_deref(),
        model.as_deref(),
    )
    .await
    .with_context(|| format!("Failed to generate command for description: {prompt}"))?;

    if let Some(pb) = spinner {
        pb.finish_and_clear();
    }

    println!(
        "{}",
        format!("Suggested command ({}):", shell.as_str()).green()
    );
    let is_guidance_only = cmd.trim_start().starts_with('#');
    let cmd_output = if is_guidance_only {
        cmd.yellow()
    } else {
        cmd.bold().green()
    };
    println!("{}", cmd_output);

    if verbose {
        if let Some(raw) = raw_response {
            println!("\n{}", "Raw response:".yellow());
            println!("{}", raw.yellow());
        }

        println!("\n{}", "Explanation:".green());
        println!("{}", explanation.green());
    }

    let mut seen_commands: HashSet<String> = HashSet::new();
    let mut command_options: Vec<String> = Vec::new();

    if let Some(primary_cmd) = executable_command(&cmd) {
        if seen_commands.insert(primary_cmd.clone()) {
            command_options.push(primary_cmd);
        }
    }

    alternatives
        .iter()
        .filter_map(|alt| executable_command(alt))
        .filter(|cmd_option| seen_commands.insert(cmd_option.clone()))
        .for_each(|cmd_option| command_options.push(cmd_option));

    if command_options.is_empty() {
        if is_guidance_only {
            println!(
                "{}",
                "The assistant provided guidance only; no command will be executed.".yellow()
            );
        }
        println!(
            "{}",
            "No runnable commands were produced. The request may be unclear—try adding more detail.".yellow()
        );
        return Ok(());
    }

    if command_options.len() > 1 {
        println!("\n{}", "Command options:".yellow());
        for (idx, option) in command_options.iter().enumerate() {
            println!("  {}. {}", idx + 1, option);
        }
        println!(
            "{}",
            "Multiple possible commands detected. Choose one to run:".bright_yellow()
        );
        if let Some(choice) = prompt_for_command_selection(&command_options)? {
            confirm_and_execute(&choice, shell.as_str())?;
        } else {
            println!("{}", "No command selected; exiting.".yellow());
        }
    } else {
        let primary_cmd = &command_options[0];
        if matches!(confidence, CommandConfidence::NeedsConfirmation) {
            println!(
                "{}",
                "AI is unsure about this command; review carefully before running.".bright_yellow()
            );
        }
        confirm_and_execute(primary_cmd, shell.as_str())?;
    }

    Ok(())
}

fn read_stdin() -> Result<Option<String>> {
    if atty::is(atty::Stream::Stdin) {
        return Ok(None);
    }

    let mut buffer = String::new();
    io::stdin()
        .read_to_string(&mut buffer)
        .context("Failed to read from stdin")?;

    if buffer.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(buffer))
    }
}

fn init_tracing() {
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .finish();

    let _ = tracing::subscriber::set_global_default(subscriber);
}

fn needs_api_key(cli: &Cli) -> bool {
    matches!(cli.command, Commands::Gen { .. })
}

fn ensure_required_env() -> Result<()> {
    const VAR: &str = "OPENAI_API_KEY";
    const FAKE_VAR: &str = "TASK_SH_FAKE_RESPONSE";

    if matches!(std::env::var(VAR), Ok(ref v) if !v.trim().is_empty()) {
        return Ok(());
    }

    if let Ok(_fake) = std::env::var(FAKE_VAR) {
        unsafe {
            std::env::set_var(VAR, "sk-test-placeholder");
        }
        println!(
            "{}",
            format!("Using {} for deterministic output.", FAKE_VAR).bright_black()
        );
        return Ok(());
    }

    if !atty::is(Stream::Stdin) {
        return Err(anyhow!(
            "OPENAI_API_KEY is not set. Provide it via environment, .env, or use TASK_SH_FAKE_RESPONSE for testing."
        ));
    }

    println!(
        "{}",
        "task.sh hasn’t been connected to OpenAI yet. Let’s add your API key.".cyan()
    );
    println!(
        "{}",
        "You can generate one at https://platform.openai.com/api-keys".bright_black()
    );

    let key = prompt_for_api_key()?;
    let trimmed = key.trim();
    if trimmed.is_empty() {
        println!(
            "{}",
            "No key entered. Please rerun once you have a key.".red()
        );
        std::process::exit(1);
    }

    save_default_env(VAR, trimmed)?;
    unsafe {
        std::env::set_var(VAR, trimmed);
    }
    println!("{}", "API key saved to .env".green());
    Ok(())
}

fn prompt_for_api_key() -> Result<String> {
    print!("{}", "API key: ".bright_blue());
    io::stdout().flush().ok();
    let key = read_password().context("Failed to read API key")?;
    Ok(key)
}

fn maybe_execute(command: &str, shell: &str, _force_interactive: bool) -> Result<()> {
    if command.trim().is_empty() {
        return Ok(());
    }

    if !atty::is(atty::Stream::Stdin) {
        println!(
            "{}",
            "Non-interactive session detected; skipping execution.".yellow()
        );
        return Ok(());
    }

    let is_running = Arc::new(AtomicBool::new(true));
    let animation_handle = spawn_execution_animation(command.to_string(), is_running.clone());

    let output = Command::new(shell)
        .arg("-c")
        .arg(command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("Failed to execute command")?;

    is_running.store(false, Ordering::SeqCst);
    if let Some(handle) = animation_handle {
        let _ = handle.join();
    }

    println!();

    if !output.stdout.is_empty() {
        let resolved = enrich_find_output(command, &output.stdout)?;
        io::stdout().write_all(resolved.as_bytes())?;
        if !resolved.ends_with('\n') {
            println!();
        }
    }

    if !output.stderr.is_empty() {
        io::stderr().write_all(&output.stderr)?;
        if !output.stderr.ends_with(b"\n") {
            eprintln!();
        }
    }

    if output.status.success() {
        println!("{}", "Command completed successfully.".green());
    } else {
        println!(
            "{}",
            format!("Command exited with status: {}", output.status).red()
        );
    }

    Ok(())
}

fn spawn_execution_animation(
    command: String,
    is_running: Arc<AtomicBool>,
) -> Option<thread::JoinHandle<()>> {
    if !atty::is(atty::Stream::Stderr) {
        println!("{}", format!("Running: {}", command).cyan());
        return None;
    }

    Some(thread::spawn(move || {
        static QUIPS: &[&str] = &[
            "Asking the kernel nicely, again...",
            "Sacrificing a goat to the CI gods...",
            "Turning it off and on again...",
            "Did you mean to run rm -rf? No? Good.",
            "Reading the manual so you don't have to...",
            "Checking if coffee supply is adequate...",
            "Dropping packets like it's 1999...",
            "Politely bullying prod into behaving...",
            "Threatening the CI with a stern email...",
            "Bribing the load balancer with donuts...",
            "Performing ritual log sacrifice...",
            "Convincing cron this isn't personal...",
            "Telling Kubernetes it's not the chosen one...",
            "Convincing the firewall to chill...",
        ];

        static SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

        let mut rng = thread_rng();
        let mut quip = QUIPS
            .choose(&mut rng)
            .copied()
            .unwrap_or("Executing command...");
        let mut spin_idx: usize = 0;
        let mut pulse_pos: f32 = -6.0;
        let mut last_quip_update = Instant::now();

        while is_running.load(Ordering::SeqCst) {
            let frame = SPINNER[spin_idx % SPINNER.len()];
            let rendered = format!("{} {}", frame, quip);
            let colored = render_gradient(&rendered, pulse_pos);
            let mut stderr = io::stderr();
            let _ = write!(stderr, "\r\x1b[2K{}", colored);
            let _ = stderr.flush();
            spin_idx = spin_idx.wrapping_add(1);
            thread::sleep(Duration::from_millis(35));

            pulse_pos += 0.85;
            let span = rendered.chars().count() as f32 + 6.0;
            if pulse_pos > span {
                pulse_pos = -6.0;
            }

            if last_quip_update.elapsed() > Duration::from_secs(6) {
                quip = QUIPS.choose(&mut rng).copied().unwrap_or(quip);
                last_quip_update = Instant::now();
            }
        }

        let mut stderr = io::stderr();
        let _ = write!(stderr, "\r\x1b[2K");
        let _ = stderr.flush();
    }))
}

fn render_gradient(text: &str, pulse_pos: f32) -> String {
    if text.is_empty() {
        return String::new();
    }

    let sigma = 3.5_f32;
    let base = 150_f32;
    let amplitude = 100_f32;
    let mut out = String::with_capacity(text.len());
    for (idx, ch) in text.chars().enumerate() {
        let dist = idx as f32 - pulse_pos;
        let mut intensity = base + amplitude * (-(dist * dist) / (2.0 * sigma * sigma)).exp();
        intensity = intensity.min(255.0).max(base);
        let intensity = intensity as u8;
        out.push_str(
            &ch.to_string()
                .truecolor(intensity, intensity, intensity)
                .to_string(),
        );
    }
    out
}

fn generate_completions(shell: Shell) {
    use clap_complete::{generate, shells};
    use std::io;

    let mut cmd = Cli::command();
    match shell {
        Shell::Bash => generate(shells::Bash, &mut cmd, "task", &mut io::stdout()),
        Shell::Zsh => generate(shells::Zsh, &mut cmd, "task", &mut io::stdout()),
    }
}

impl Shell {
    fn from_str_case_insensitive(value: &str) -> Option<Self> {
        match value.to_lowercase().as_str() {
            "bash" => Some(Shell::Bash),
            "zsh" => Some(Shell::Zsh),
            _ => None,
        }
    }
}

fn enrich_find_output(command: &str, stdout: &[u8]) -> Result<String> {
    if !command.trim_start().starts_with("find") {
        return Ok(String::from_utf8_lossy(stdout).into_owned());
    }

    let output_str = String::from_utf8_lossy(stdout);
    let mut enriched = String::new();

    for line in output_str.lines() {
        let path = line.trim();
        if path.is_empty() {
            continue;
        }

        let metadata = match fs::metadata(Path::new(path)) {
            Ok(meta) => meta,
            Err(_) => {
                enriched.push_str(path);
                enriched.push('\n');
                continue;
            }
        };

        let size = metadata.len();
        let modified = metadata.modified().ok().map(|time| {
            DateTime::<Local>::from(time)
                .format("%Y-%m-%d %H:%M:%S")
                .to_string()
        });
        let display_size = format_size(size);
        let mut entry = format!("{}  {}", display_size, path);
        if let Some(ts) = modified {
            entry.push_str(&format!("  (modified {})", ts));
        }
        enriched.push_str(&entry);
        enriched.push('\n');
    }

    Ok(enriched)
}

fn format_size(bytes: u64) -> String {
    let mut bytes = bytes;
    let mut unit = "B";
    if bytes >= 1_000_000_000 {
        bytes /= 1_000_000_000;
        unit = "GB";
    } else if bytes >= 1_000_000 {
        bytes /= 1_000_000;
        unit = "MB";
    } else if bytes >= 1_000 {
        bytes /= 1_000;
        unit = "KB";
    }
    format!("{} {}", bytes, unit)
}

fn prompt_for_command_selection(commands: &[String]) -> Result<Option<String>> {
    if commands.is_empty() {
        return Ok(None);
    }

    loop {
        println!("{}", "Select a command to run:".cyan());
        for (idx, command) in commands.iter().enumerate() {
            println!("  {}) {}", idx + 1, command);
        }
        println!("  0) Cancel");
        print!("Enter choice (default 0): ");
        io::stdout().flush().context("Failed to flush stdout")?;
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .context("Failed to read selection")?;
        let trimmed = input.trim();
        if trimmed.is_empty() || trimmed == "0" {
            return Ok(None);
        }
        if let Some(idx) = trimmed
            .parse::<usize>()
            .ok()
            .filter(|idx| (1..=commands.len()).contains(idx))
        {
            return Ok(Some(commands[idx - 1].clone()));
        }
        println!("{}", "Invalid selection, please try again.".yellow());
    }
}

fn executable_command(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn confirm_and_execute(command: &str, shell: &str) -> Result<()> {
    println!(
        "\n{}",
        "The following command will be executed:".bright_blue()
    );
    println!("{}", format!("{} -c \"{}\"", shell, command).bold());

    println!("{}", "Proceed with execution? [y/N] ".bright_blue());
    io::stdout().flush().context("Failed to flush stdout")?;

    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .context("Failed to read confirmation input")?;

    if matches!(answer.trim().to_lowercase().as_str(), "y" | "yes") {
        maybe_execute(command, shell, true)?;
    } else {
        println!("{}", "Command not executed.".yellow());
    }

    Ok(())
}
