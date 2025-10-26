use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use dirs::home_dir;
use serde::Deserialize;

const ENV_FILE: &str = ".env";

pub fn save_default_env(key: &str, value: &str) -> Result<()> {
    let env_path = PathBuf::from(ENV_FILE);
    let mut contents = if env_path.exists() {
        fs::read_to_string(&env_path)
            .with_context(|| format!("Failed to read {}", env_path.display()))?
    } else {
        String::new()
    };

    let assignment = format!("{}=\"{}\"\n", key, value);
    if contents.contains(key) {
        contents = contents
            .lines()
            .filter(|line| !line.starts_with(&format!("{}=", key)))
            .chain(std::iter::once(assignment.trim_end()))
            .map(|line| format!("{}\n", line))
            .collect();
    } else {
        contents.push_str(&assignment);
    }

    fs::write(&env_path, contents)
        .with_context(|| format!("Failed to write {}", env_path.display()))
}

#[derive(Debug, Deserialize, Default)]
pub struct FileConfig {
    pub default_shell: Option<String>,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub verbose: Option<bool>,
    pub spinner: Option<bool>,
}

#[derive(Debug, Default, Clone)]
pub struct AppConfig {
    pub default_shell: Option<String>,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub verbose: Option<bool>,
    pub spinner: Option<bool>,
}

pub fn load(user_path: Option<PathBuf>) -> Result<AppConfig> {
    let mut cfg = AppConfig::default();

    if let Some(path) = user_path.clone() {
        load_from_path(&mut cfg, &path)?;
        if cfg.is_populated() {
            return Ok(cfg);
        }
    }

    if let Some(default_path) = default_path() {
        load_from_path(&mut cfg, &default_path)?;
    }

    Ok(cfg)
}

fn load_from_path(cfg: &mut AppConfig, path: &PathBuf) -> Result<()> {
    if path.exists() {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file at {}", path.display()))?;
        let file_cfg: FileConfig = toml::from_str(&contents)
            .with_context(|| format!("Failed to parse config file at {}", path.display()))?;
        cfg.apply(file_cfg);
    }
    Ok(())
}

impl AppConfig {
    fn apply(&mut self, file: FileConfig) {
        if self.default_shell.is_none() {
            self.default_shell = file.default_shell;
        }
        if self.model.is_none() {
            self.model = file.model;
        }
        if self.system_prompt.is_none() {
            self.system_prompt = file.system_prompt;
        }
        if self.verbose.is_none() {
            self.verbose = file.verbose;
        }
        if self.spinner.is_none() {
            self.spinner = file.spinner;
        }
    }

    fn is_populated(&self) -> bool {
        self.default_shell.is_some()
            || self.model.is_some()
            || self.system_prompt.is_some()
            || self.verbose.is_some()
            || self.spinner.is_some()
    }
}

fn default_path() -> Option<PathBuf> {
    home_dir().map(|mut dir| {
        dir.push(".task.toml");
        dir
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn loads_user_provided_path() {
        let mut tmp = NamedTempFile::new().unwrap();
        writeln!(
            tmp,
            "default_shell = \"zsh\"\nmodel = \"gpt-4o-mini\"\nverbose = true\nspinner = false"
        )
        .unwrap();

        let cfg = load(Some(tmp.path().to_path_buf())).unwrap();
        assert_eq!(cfg.default_shell.as_deref(), Some("zsh"));
        assert_eq!(cfg.model.as_deref(), Some("gpt-4o-mini"));
        assert_eq!(cfg.verbose, Some(true));
        assert_eq!(cfg.spinner, Some(false));
    }
}
