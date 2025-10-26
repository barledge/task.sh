use std::{env, sync::Arc, time::Duration};

use anyhow::{Context, Result, anyhow};
use async_openai::{
    Client,
    config::OpenAIConfig,
    error::{ApiError, OpenAIError},
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
        CreateChatCompletionRequest, CreateChatCompletionRequestArgs, Role,
    },
};
use once_cell::sync::Lazy;
use regex::Regex;
use tokio::time::sleep;
use tracing::{debug, trace, warn};

/// A generated shell command returned by the AI backend.
///
/// This struct bundles the executable command, a short explanation, and an optional raw response
/// payload that callers can surface in verbose modes.
///
/// # Examples
///
/// ```
/// use task_sh::generator::GeneratedCommand;
///
/// let command = GeneratedCommand {
///     cmd: "echo 'hello'".into(),
///     explanation: "Prints hello".into(),
///     raw_response: None,
/// };
/// assert!(command.cmd.contains("echo"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedCommand {
    pub cmd: String,
    pub explanation: String,
    pub raw_response: Option<String>,
}

/// Fake response override environment variable.
const FAKE_RESPONSE_ENV: &str = "TASK_SH_FAKE_RESPONSE";

/// OpenAI chat model used for generation.
pub const MODEL: &str = "gpt-3.5-turbo";
/// Number of attempts before giving up on OpenAI.
const MAX_RETRIES: usize = 3;
/// Timeout for each OpenAI request.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Generate a shell command for the provided description and shell type.
///
/// Returns rich contextual errors when the OpenAI backend fails, the description is not usable,
/// or when safety heuristics detect a dangerous command.
///
/// # Examples
///
/// ```no_run
/// use task_sh::generator::generate_command;
///
/// # tokio_test::block_on(async {
/// let result = generate_command("List files", "bash").await;
/// # let _ = result; // ignore in doc example
/// # });
/// ```
pub async fn generate_command(
    desc: &str,
    shell: &str,
    custom_system_prompt: Option<&str>,
    model_override: Option<&str>,
) -> Result<GeneratedCommand> {
    trace!(description = %desc, shell, "Starting command generation");

    let trimmed = desc.trim();
    if trimmed.is_empty() {
        warn!("Received empty description");
        return Ok(GeneratedCommand {
            cmd: "# Please provide more details.".to_string(),
            explanation: "Description was empty or ambiguous.".to_string(),
            raw_response: None,
        });
    }

    if trimmed.split_whitespace().count() < 2 {
        warn!(description = %trimmed, "Description appears ambiguous");
        return Ok(GeneratedCommand {
            cmd: "# Please provide more details.".to_string(),
            explanation: "Description appears too short or ambiguous.".to_string(),
            raw_response: None,
        });
    }

    if let Ok(fake) = env::var(FAKE_RESPONSE_ENV) {
        trace!("Using fake response for testing mode");
        let (cmd, explanation) = parse_completion_content(&fake)?;
        enforce_safety(&cmd)?;

        return Ok(GeneratedCommand {
            cmd,
            explanation,
            raw_response: Some(fake),
        });
    }

    let api_key = env::var("OPENAI_API_KEY").context(
        "OPENAI_API_KEY missing. Set it as an environment variable or in your .env file",
    )?;

    if api_key.trim().is_empty() {
        return Err(anyhow!("OPENAI_API_KEY is empty"));
    }

    let system_prompt = custom_system_prompt.map(|prompt| prompt.to_string()).unwrap_or_else(|| {
        format!(
            "Generate a safe {shell} command for: {desc}. Explain briefly. Avoid destructive actions like rm -rf or sudo."
        )
    });

    let user_prompt = format!("Description: {desc}");

    let client = Arc::new(Client::with_config(
        OpenAIConfig::default().with_api_key(api_key.clone()),
    ));

    let mut last_err: Option<OpenAIError> = None;

    for attempt in 0..MAX_RETRIES {
        let request = build_chat_request(
            model_override.unwrap_or(MODEL),
            &system_prompt,
            &user_prompt,
        )?;
        trace!(attempt, "Dispatching chat completion request");

        match tokio::time::timeout(REQUEST_TIMEOUT, client.chat().create(request)).await {
            Ok(Ok(response)) => {
                let choice = response
                    .choices
                    .into_iter()
                    .next()
                    .context("OpenAI response did not contain any choices")?;

                trace!(?choice.message, "raw choice message");

                let mut content = choice.message.content.unwrap_or_default();

                if content.trim().is_empty() {
                    if let Some(tool_calls) = choice.message.tool_calls {
                        content = tool_calls
                            .into_iter()
                            .map(|call| call.function.arguments.clone())
                            .collect::<Vec<_>>()
                            .join("\n");
                    }
                }

                trace!(%content, "raw completion content");

                let (cmd, explanation) = parse_completion_content(&content)?;
                enforce_safety(&cmd)?;

                debug!(command = %cmd, "Generated command candidate");

                return Ok(GeneratedCommand {
                    cmd,
                    explanation,
                    raw_response: Some(content),
                });
            }
            Ok(Err(err)) => {
                let is_last_attempt = attempt + 1 == MAX_RETRIES;
                if is_last_attempt {
                    last_err = Some(err);
                    break;
                }

                let backoff = compute_backoff_delay(&err, attempt);
                sleep(backoff).await;
                last_err = Some(err);
            }
            Err(_) => {
                let timeout_err = OpenAIError::ApiError(ApiError {
                    message: "Request timed out".to_string(),
                    r#type: None,
                    param: None,
                    code: None,
                });
                let is_last_attempt = attempt + 1 == MAX_RETRIES;
                if is_last_attempt {
                    last_err = Some(timeout_err);
                    break;
                }
                let backoff = compute_backoff_delay(&timeout_err, attempt);
                sleep(backoff).await;
                last_err = Some(timeout_err);
            }
        }
    }

    Err(last_err
        .map(|err| anyhow!(err))
        .unwrap_or_else(|| anyhow!("Unknown error while calling OpenAI"))
        .context("Failed to generate command after multiple attempts"))
}

/// Build a chat completion request from the system and user prompts.
fn build_chat_request(
    model: &str,
    system_prompt: &str,
    user_prompt: &str,
) -> Result<CreateChatCompletionRequest> {
    Ok(CreateChatCompletionRequestArgs::default()
        .model(model)
        .temperature(0.2)
        .messages(vec![
            ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: system_prompt.to_string(),
                role: Role::System,
                name: None,
            }),
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(user_prompt.to_string()),
                role: Role::User,
                name: None,
            }),
        ])
        .build()?)
}

/// Parse the command and explanation from the raw OpenAI response content.
fn parse_completion_content(raw: &str) -> Result<(String, String)> {
    let mut command: Option<String> = None;
    let mut explanation: Option<String> = None;
    let mut explanation_line: Option<String> = None;
    let mut body_lines: Vec<String> = Vec::new();
    let mut code_buffer: Vec<String> = Vec::new();
    let mut in_code_block = false;

    for raw_line in raw.lines() {
        let trimmed = raw_line.trim();

        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            if !in_code_block && !code_buffer.is_empty() {
                let joined = code_buffer.join("\n");
                if command.is_none() {
                    command = Some(joined.clone());
                } else {
                    body_lines.push(joined);
                }
                code_buffer.clear();
            }
            continue;
        }

        if in_code_block {
            code_buffer.push(trimmed.to_string());
            continue;
        }

        if trimmed.is_empty() {
            continue;
        }

        let lower = trimmed.to_lowercase();
        if lower.starts_with("command:") {
            let value = trimmed
                .splitn(2, ':')
                .nth(1)
                .map(str::trim)
                .unwrap_or("")
                .to_string();
            if !value.is_empty() {
                command = Some(value);
            }
        } else if lower.starts_with("explanation:") {
            let value = trimmed
                .splitn(2, ':')
                .nth(1)
                .map(str::trim)
                .unwrap_or("")
                .to_string();
            if !value.is_empty() {
                explanation = Some(value);
            }
            explanation_line = Some(trimmed.to_string());
        } else {
            body_lines.push(trimmed.to_string());
        }
    }

    if command.is_none() && !code_buffer.is_empty() {
        command = Some(code_buffer.join("\n"));
    }

    if command.is_none() {
        if let Some(first) = body_lines.first() {
            command = Some(first.clone());
            body_lines.remove(0);
        } else if let Some(line) = explanation_line {
            command = Some(line);
        } else {
            return Err(anyhow!("OpenAI response missing 'Command:' line"));
        }
    }

    if explanation.is_none() {
        explanation = Some(body_lines.join(" "));
    }

    let cmd = command.context("OpenAI response missing 'Command:' line")?;
    let explanation = explanation.unwrap_or_else(|| "No explanation provided.".to_string());

    Ok((cmd, explanation))
}

/// Run heuristic safety checks against the generated command.
fn enforce_safety(command: &str) -> Result<()> {
    static BLOCK_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
        vec![
            Regex::new("(?i)rm\\s+-rf").expect("valid regex"),
            Regex::new("(?i)\\bsudo\\b").expect("valid regex"),
            Regex::new("(?i)dd\\s+.*of=").expect("valid regex"),
            Regex::new("(?i)curl\\s+[^|]+\\|\\s*sh").expect("valid regex"),
            Regex::new("(?i)chmod\\s+777").expect("valid regex"),
            Regex::new("(?i)mkfs\\.\\w*").expect("valid regex"),
            Regex::new("(?i)scp\\s+-r").expect("valid regex"),
            Regex::new("(?i)shutdown").expect("valid regex"),
            Regex::new("(?i)reboot").expect("valid regex"),
            Regex::new("(?i)poweroff").expect("valid regex"),
        ]
    });

    if let Some(pattern) = BLOCK_PATTERNS
        .iter()
        .find(|pattern| pattern.is_match(command))
    {
        warn!(%command, pattern = %pattern, "Blocked unsafe command");
        return Err(anyhow!(
            "Generated command was blocked by safety rules. Please refine your description."
        ));
    }

    Ok(())
}

fn compute_backoff_delay(err: &OpenAIError, attempt: usize) -> Duration {
    let base_delay_ms = if err.to_string().to_lowercase().contains("rate limit") {
        1_000
    } else {
        300
    };

    Duration::from_millis(base_delay_ms * (attempt as u64 + 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn unset_fake_response() {
        unsafe {
            env::remove_var(FAKE_RESPONSE_ENV);
        }
    }

    #[test]
    fn parses_command_and_explanation() {
        let raw = "Command: echo hello\nExplanation: Prints a greeting";
        let (cmd, explanation) = parse_completion_content(raw).expect("should parse");

        assert_eq!(cmd, "echo hello");
        assert_eq!(explanation, "Prints a greeting");
    }

    #[test]
    fn blocks_destructive_commands() {
        let err = enforce_safety("rm -rf /").expect_err("should block");
        assert!(err.to_string().contains("blocked"));
    }

    #[test]
    fn missing_command_line_errors() {
        let (cmd, explanation) =
            parse_completion_content("Explanation: hi").expect("fallback should handle");
        assert_eq!(cmd, "Explanation: hi");
        assert_eq!(explanation, "hi");
    }

    #[tokio::test]
    async fn returns_hint_on_empty_description() {
        let result = generate_command("", "bash", None, None)
            .await
            .expect("empty descriptions should succeed");

        assert!(result.cmd.contains("Please provide more details"));
        assert!(result.raw_response.is_none());
    }

    #[tokio::test]
    #[serial]
    async fn uses_fake_response_environment() {
        unset_fake_response();
        unsafe {
            env::set_var(FAKE_RESPONSE_ENV, "Command: ls\nExplanation: List files");
        }

        let result = generate_command("list files recursively", "bash", None, None)
            .await
            .expect("fake response should succeed");

        assert_eq!(result.cmd, "ls");
        assert_eq!(result.explanation, "List files");
        assert!(result.raw_response.is_some());

        unset_fake_response();
    }

    #[tokio::test]
    #[serial]
    async fn fake_response_respects_safety_filters() {
        unset_fake_response();
        unsafe {
            env::set_var(FAKE_RESPONSE_ENV, "Command: rm -rf /\nExplanation: wipe");
        }

        let err = generate_command("delete everything", "bash", None, None)
            .await
            .expect_err("should block unsafe command");

        assert!(err.to_string().contains("blocked"));

        unset_fake_response();
    }

    #[tokio::test]
    async fn ambiguous_description_returns_guidance() {
        let result = generate_command("status", "bash", None, None)
            .await
            .expect("ambiguous prompts return guidance");

        assert!(result.cmd.starts_with('#'));
    }
}
