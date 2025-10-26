# Usage

## Basic generation

```bash
task gen "list large files" --shell bash -v
```

## Stdin pipelines

```bash
echo "summarize git status" | task gen --shell zsh
```

## Flags

- `--shell <bash|zsh>`: specify the output shell.
- `--verbose / -v`: include explanations and raw API payload.

When verbose mode is on, the CLI prints raw OpenAI output in yellow and explanations in green.
