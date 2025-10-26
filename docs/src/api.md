# Command Reference

## `task gen`

| Option | Description |
| ------ | ----------- |
| `description` | Positional natural-language prompt. |
| `--shell <bash|zsh>` | Target shell for the generated command. |
| `-v`, `--verbose` | Emit raw response and explanation details. |

## Environment Variables

| Variable | Purpose |
| -------- | ------- |
| `OPENAI_API_KEY` | Required for contacting OpenAI APIs. |
| `TASK_SH_FAKE_RESPONSE` | Optional test hook that overrides the API response. |

## Exit Codes

- `0`: success.
- non-zero: failure to generate/validate command or missing configuration.
