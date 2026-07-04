# CLAUDE.md

## What is this?

A Claude Code skill/plugin that implements an autonomous experiment loop. Port of [pi-autoresearch](https://github.com/davebcn87/pi-autoresearch) — no MCP server, pure skill + hooks. Installable as a plugin (`claude --plugin-dir .`) or via manual symlinks (`./install.sh`).

## Project structure

```
.claude-plugin/plugin.json          # Plugin manifest
skills/autoresearch/SKILL.md        # Core skill: setup, JSONL protocol, run/log/loop logic
skills/autoresearch/scripts/ar-log.sh  # Appends valid-JSON result lines (jq w/ Python fallback)
commands/autoresearch.md            # /autoresearch (start, resume, status, report, off)
hooks/hooks.json                    # Hook definitions (plugin format)
hooks/autoresearch-stop.sh          # Stop hook — loop engine + budget valve (the real "never stop")
hooks/autoresearch-precompact.sh    # PreCompact hook — snapshots state before compaction
hooks/autoresearch-sessionstart.sh  # SessionStart hook — rehydrates active loop on resume/compact
hooks/autoresearch-context.sh       # UserPromptSubmit hook — context + user steers
install.sh / uninstall.sh           # Manual symlink install (alternative to plugin)
examples/                           # Demo: fastball velocity prediction + model zoo
  train.py                          # Training orchestrator with rich TUI output (AR_SEED-aware)
  models.py                         # Model registry (19 models, GPU detection)
  pyproject.toml                    # uv project config with dependency groups
experiments/                        # Gitignored — experiment worklogs go here
```

## Key conventions

- **SKILL.md is the source of truth** for all behavior. The original 3 MCP tools (`init_experiment`/`run_experiment`/`log_experiment`) are encoded as instructions the agent follows with Bash/Read/Write.
- **`autoresearch.jsonl` is the state format.** Config headers start segments and carry the budget/noise contract; result lines track experiments. Exact schemas → SKILL.md "JSONL State Protocol". If you change the schema, update both that section and "Logging Results" — they must stay in sync.
- **Never hand-build result JSON with `echo`** — a quote/apostrophe in a description corrupts the file. Use `scripts/ar-log.sh` (or `jq -nc`).
- **The four hooks are the loop's spine** — `Stop` enforces continuation, `PreCompact`/`SessionStart` survive compaction, `UserPromptSubmit` carries steers. Continuation MUST use JSON `decision:block`, never `exit 2` (broken for plugin hooks, anthropics/claude-code#10412). → SKILL.md "Loop enforcement".
- **Hook stdout must be clean** — JSON-only where parsed, all diagnostics to stderr; a stray echo corrupts parsing. All four honor the `.autoresearch-off` sentinel.
- **The eval harness is locked.** `autoresearch.sh` and metric-emitting code are Off Limits to experiments — an agent that can edit its own scorer will game it. Keep/discard is gated on the **noise floor**, not raw improvement.
- **Git commits on keep** use a `Result: {...}` trailer in the commit body. Dashboard → `autoresearch-dashboard.md`; narrative worklog → `experiments/worklog.md`.
- All session artifacts (`autoresearch.jsonl`, `autoresearch-dashboard.md`, `autoresearch.md`, `autoresearch.sh`, `experiments/`, `plots/`, `brain/`) are gitignored.

## Package management & output

- **Use `uv` for all package management**, never `pip`. The example uses `pyproject.toml` with optional dependency groups (`uv sync`, `uv sync --extra torch`, `uv sync --extra all`).
- **The example uses `rich` for terminal output** with graceful fallback. Rich output goes to stderr via `Console(stderr=True)`; `METRIC name=value` lines always go to stdout as plain text (autoresearch parses them).
- Models in `examples/models.py` use **lazy imports** for optional deps (torch, catboost, lightgbm, tabpfn, pytorch-tabnet). Missing deps produce clear error messages, not crashes.

## Editing tips

- The command file uses `$ARGUMENTS`, which Claude Code substitutes with the slash-command arguments.
- **Hook scripts run in the user's cwd, not the repo** — resolve state files relative to the payload's `cwd`, not `$0`.
- `hooks/hooks.json` uses plugin format; script paths use `"${CLAUDE_PLUGIN_ROOT}"` (quoted — resolves at runtime, may contain spaces).

<!-- AGENT-MANAGED SECTION — Claude and other agents may add learnings below this line. -->
<!-- Lifecycle: (1) write the why/history into SKILL.md or a notes file FIRST, (2) add a one-line pointer here, -->
<!-- (3) once stable, graduate it into the matching section above and delete it here. Keep this an inbox, not an archive. -->

## Discovered Patterns

_(none yet)_
