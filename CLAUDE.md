# CLAUDE.md

## What is this?

A Claude Code **plugin** that implements an autonomous experiment loop. Port of [pi-autoresearch](https://github.com/davebcn87/pi-autoresearch) — no MCP server, pure skill + hooks.

## Project structure

```
.claude-plugin/plugin.json     # Plugin manifest
skills/autoresearch/SKILL.md   # Core skill: setup, JSONL protocol, run/log/loop logic
commands/autoresearch.md       # /autoresearch slash command (start, resume, off)
hooks/hooks.json               # Hook definitions (plugin format)
hooks/autoresearch-context.sh  # UserPromptSubmit hook — injects context when active
examples/                      # Fastball velocity prediction demo files
experiments/                   # Gitignored — experiment worklogs go here
```

## Key conventions

- **SKILL.md is the source of truth** for all behavior. The original 3 MCP tools (`init_experiment`, `run_experiment`, `log_experiment`) are encoded as instructions the agent follows using Bash/Read/Write.
- **JSONL format** in `autoresearch.jsonl` is the state format. Config headers start segments, result lines track experiments. See SKILL.md for exact JSON schemas.
- **Git commits on keep** use a `Result: {...}` trailer in the commit message body.
- **Dashboard** is written to `autoresearch-dashboard.md` (file-based, not TUI).
- **Worklog** is written to `experiments/worklog.md` — narrative log of experiments and insights, survives context compactions.
- The hook script must output to stdout (that's how Claude Code hooks inject context).
- All experiment artifacts (`autoresearch.jsonl`, `autoresearch-dashboard.md`, `autoresearch.md`, `autoresearch.sh`, `experiments/`, `plots/`) are gitignored.

## Editing tips

- If changing the JSONL schema, update both the "JSONL State Protocol" and "Logging Results" sections in SKILL.md — they must stay in sync.
- The command file uses `$ARGUMENTS` which Claude Code substitutes with the user's slash command arguments.
- Hook scripts run in the user's cwd, not the repo directory.
- `hooks/hooks.json` defines hooks in plugin format. The shell script path uses `${CLAUDE_PLUGIN_ROOT}` which resolves at runtime.
