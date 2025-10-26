# task.sh

[![Crates.io](https://img.shields.io/crates/v/task-sh.svg)](https://crates.io/crates/task-sh)
[![Downloads](https://img.shields.io/crates/d/task-sh.svg)](https://crates.io/crates/task-sh)
[![CI](https://github.com/barledge/task.sh/actions/workflows/rust.yml/badge.svg)](https://github.com/barledge/task.sh/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`task.sh` turns natural language task descriptions into safe shell command suggestions using OpenAI. The CLI is designed for cross-platform usage (Linux, macOS) and emphasizes safety, observability, and customization.

## Features

- `task gen` subcommand with `--shell` selection (`bash`, `zsh`).
- Verbose mode prints raw AI output and explanations.
- Regex-based command safety filters.
- Optional `TASK_SH_FAKE_RESPONSE` environment variable for deterministic tests.

## Getting Started

### Prerequisites

- Rust toolchain (via [rustup](https://rustup.rs/)).
- OpenAI API key with access to GPT-3.5 (or compatible) models.

### Installation

```bash
git clone https://github.com/barledge/task.sh.git
cd task.sh
cargo install --path .
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` to your secret key.

### Usage

```bash
# Generate a bash command with explanation
task gen "list large files" --shell bash -v

# Pipe input via stdin
echo "list staged changes" | task gen --verbose

# Use the fake response mode for testing/demos
TASK_SH_FAKE_RESPONSE=$'Command: ls\nExplanation: list files' task gen "anything"
```

Inspect all options:

```bash
sh --help
```

## Environment

- `OPENAI_API_KEY`: Required for live generation.
- `TASK_SH_FAKE_RESPONSE`: Optional string that substitutes the OpenAI response for testing.

## Development

```bash
cargo fmt
cargo clippy --all-targets
cargo test
```

## Build Scripts

```bash
scripts/build-release.sh
scripts/publish.sh
```

For cross-compilation (requires `cargo-zigbuild`):

```bash
scripts/build-cross.sh x86_64-unknown-linux-gnu
scripts/build-cross.sh x86_64-pc-windows-gnu
scripts/build-cross.sh aarch64-apple-darwin
```

## Documentation

The mdBook-based docs live in `docs/`.

```bash
cargo install mdbook
mdbook build docs
```

Publish to `task.sh/docs` or serve locally with `mdbook serve docs`.

## Contributing

1. Fork the repository and create a feature branch.
2. Add tests covering new functionality.
3. Run `cargo fmt`, `cargo clippy`, and `cargo test` before submitting a PR.
4. Submit a PR with a concise description of the changes.

## License

Licensed under the MIT License.

