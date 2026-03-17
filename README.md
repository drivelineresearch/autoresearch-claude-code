# autoresearch-claude-code

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Claude Code Plugin](https://img.shields.io/badge/Claude%20Code-Plugin-blueviolet)](https://docs.anthropic.com/en/docs/claude-code)

Autonomous experiment loop for [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Port of [pi-autoresearch](https://github.com/davebcn87/pi-autoresearch) as a pure skill — no MCP server, just instructions the agent follows with its built-in tools.

Runs experiments, measures results, keeps winners, discards losers, loops forever.

## Install

### Quick start (development / testing)

```bash
claude --plugin-dir /path/to/autoresearch-claude-code
```

This loads the plugin for the current session only.

### Permanent install

Clone the repo and point Claude Code at it:

```bash
git clone https://github.com/obra/autoresearch-claude-code.git ~/autoresearch-claude-code
```

Then add to your `~/.claude/settings.json`:

```json
{
  "plugins": ["~/autoresearch-claude-code"]
}
```

### Toggle on/off

```bash
claude plugin disable autoresearch   # disable (hooks stop firing too)
claude plugin enable autoresearch    # re-enable
```

Or use `/plugin` inside Claude Code for an interactive manager.

## Usage

```
/autoresearch:autoresearch optimize test suite runtime
/autoresearch:autoresearch                              # resume existing loop
/autoresearch:autoresearch off                          # pause (in-session)
```

The agent creates a branch, writes a session doc + benchmark script, runs a baseline, then loops autonomously. Send messages mid-loop to steer the next experiment.

## Example: Fastball Velocity Prediction

Included in `examples/` — uses the [Driveline OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics) dataset to predict fastball velocity from biomechanical POI metrics.

![Experiment Progress](imgs/experiment_progress.png)

22 autonomous experiments took R² from **0.44 to 0.78** (+78%), predicting a new player's fastball velocity within ~2 mph from biomechanics alone.

| Metric | Baseline | Best | Change |
|--------|----------|------|--------|
| R² | 0.440 | 0.783 | +78% |
| RMSE | 3.53 mph | 2.20 mph | -38% |

To run it yourself:

```bash
mkdir -p third_party
git clone https://github.com/drivelineresearch/openbiomechanics.git third_party/openbiomechanics
python3 -m venv .venv && source .venv/bin/activate
pip install xgboost scikit-learn pandas numpy matplotlib
cp examples/train.py examples/autoresearch.sh .
.venv/bin/python train.py
```

See [`examples/obp-autoresearch.md`](examples/obp-autoresearch.md) for the session config and [`experiments/worklog.md`](experiments/worklog.md) for the full experiment narrative.

## How it works

| pi-autoresearch (MCP) | This port (Plugin) |
|---|---|
| `init_experiment` tool | Agent writes config to `autoresearch.jsonl` |
| `run_experiment` tool | Agent runs `./autoresearch.sh` with timing |
| `log_experiment` tool | Agent appends result JSON, `git commit` on keep |
| TUI dashboard | `autoresearch-dashboard.md` |
| `before_agent_start` hook | `UserPromptSubmit` hook injects context |

State lives in `autoresearch.jsonl`. Session artifacts (`*.jsonl`, dashboard, session doc, benchmark script, ideas backlog, worklog) are gitignored.

## Plugin structure

```
.claude-plugin/plugin.json     # Plugin manifest
skills/autoresearch/SKILL.md   # Core skill: setup, JSONL protocol, run/log/loop logic
commands/autoresearch.md       # /autoresearch slash command (start, resume, off)
hooks/hooks.json               # Hook definitions (plugin format)
hooks/autoresearch-context.sh  # UserPromptSubmit hook — injects context when active
examples/                      # Fastball velocity prediction demo
```

## License

[MIT](LICENSE)
