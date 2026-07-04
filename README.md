# autoresearch-claude-code

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Claude Code Plugin](https://img.shields.io/badge/Claude%20Code-Plugin-blueviolet)](https://docs.anthropic.com/en/docs/claude-code)

Autonomous experiment loop for [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Give it a goal, a benchmark, and files to modify — it loops forever: try ideas, measure results, keep winners, discard losers.

Port of [pi-autoresearch](https://github.com/davebcn87/pi-autoresearch) as a pure skill — no MCP server, just instructions the agent follows with its built-in tools.

## Install

### Option A: Let Claude do it (easiest)

```bash
git clone https://github.com/drivelineresearch/autoresearch-claude-code.git ~/autoresearch-claude-code
claude -p "Install the autoresearch plugin from ~/autoresearch-claude-code"
```

Claude will read the repo, run `install.sh`, and configure everything.

### Option B: Plugin flag

```bash
# One-session test drive
claude --plugin-dir /path/to/autoresearch-claude-code

# Permanent — add to ~/.claude/settings.json:
# { "plugins": ["~/autoresearch-claude-code"] }

# Toggle on/off
claude plugin disable autoresearch
claude plugin enable autoresearch
```

### Option C: Manual symlinks

```bash
git clone https://github.com/drivelineresearch/autoresearch-claude-code.git ~/autoresearch-claude-code
cd ~/autoresearch-claude-code && ./install.sh
```

To remove: `./uninstall.sh`

## Quick Start

```
/autoresearch optimize test suite runtime
/autoresearch                              # resume existing loop
/autoresearch status                       # read-only: dashboard + best result so far
/autoresearch report                       # write a final summary report
/autoresearch off                          # pause the loop
```

The agent creates a branch, writes a session doc + benchmark script, measures a noise floor, runs a baseline, then loops autonomously. Send messages mid-loop to steer the next experiment.

## Guardrails (why it doesn't fool itself)

An autonomous "keep whatever wins" loop can quietly lie to you — banking seed noise as progress, gaming its own scorer, or running up unbounded spend. This port borrows the safeguards from Karpathy's [nanochat autoresearch agent](https://github.com/karpathy/nanochat) and the ML-experimentation literature:

- **The loop is mechanically enforced.** A **Stop hook** vetoes the agent ending its turn until a real budget boundary is hit — "never stop" is a mechanism, not a hope. (Uses the JSON-`decision:block` form; exit-code-2 continuation is broken for plugin hooks, [#10412](https://github.com/anthropics/claude-code/issues/10412).)
- **Noise floor.** The baseline is run several times at setup to measure metric variance; a change is only *kept* if it beats the best by **more than the noise floor**. Borderline wins are re-run on multiple seeds and compared by mean.
- **Locked eval harness.** `autoresearch.sh` and the metric-emitting code are off-limits to experiments, so the agent can't "improve" the score by editing the scorer.
- **Budget cap.** `maxRuns` / `maxSeconds` / `targetMetric` in the config header stop the loop cleanly — no unbounded overnight token burn.
- **Correctness gate.** An optional `checks.sh` runs after the benchmark; a faster-but-wrong change can't be committed (`checks_failed` status).
- **Survives compaction.** **PreCompact** snapshots state to the worklog; **SessionStart** rehydrates the objective + best result so a resumed agent continues instead of restarting.
- **Structured search.** Draft several diverse approaches before greedily refining, track experiments as a tree (`parent` pointers) to backtrack out of local optima, and cap debug attempts so it can't rabbit-hole.

## What Can You Optimize?

Anything with a measurable metric:

- **ML models** — R², RMSE, accuracy, F1 (see the [OpenBiomechanics example](#example-fastball-velocity-prediction))
- **Code performance** — runtime, memory usage, throughput
- **Build systems** — bundle size, compile time, dependency count
- **Frontend** — Lighthouse score, load time, CLS
- **Prompt engineering** — eval scores, parameter-golf
- **Any script** that outputs `METRIC name=number` to stdout

The only requirement: a bash command that runs your benchmark and prints `METRIC name=number` lines.

## Example: Fastball Velocity Prediction

Included in `examples/` — predicts fastball velocity from biomechanical data using the [Driveline OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics) dataset and a [model zoo of 19 algorithms](#model-zoo).

![Experiment Progress](imgs/experiment_progress.png)

22 autonomous experiments took R² from **0.44 to 0.78** (+78%), predicting a new player's velocity within ~2 mph from biomechanics alone.

| Metric | Baseline | Best | Change |
|--------|----------|------|--------|
| R² | 0.440 | 0.783 | +78% |
| RMSE | 3.53 mph | 2.20 mph | -38% |

### Setup

```bash
# Clone data
mkdir -p third_party
git clone https://github.com/drivelineresearch/openbiomechanics.git third_party/openbiomechanics

# Install dependencies with uv (https://docs.astral.sh/uv/)
cd examples
uv sync                    # core deps (xgboost, sklearn, rich, etc.)
uv sync --extra all        # all model backends (PyTorch, CatBoost, LightGBM, TabPFN, TabNet)

# Copy example files to working directory and run
cd ..
cp examples/train.py examples/models.py examples/autoresearch.sh .
uv run python train.py
```

See [`examples/obp-autoresearch.md`](examples/obp-autoresearch.md) for the session config and [`experiments/worklog.md`](experiments/worklog.md) for the full experiment narrative.

## Model Zoo

The example ships with 19 models the agent can swap between. All use a common interface — change `MODEL_TYPE` in `train.py` to switch.

| Category | Models | GPU | Extra Deps |
|----------|--------|-----|------------|
| **Boosting** | xgboost, catboost, lightgbm, histgb | xgb/catboost/lgbm | catboost, lightgbm |
| **Neural** | pytorch_mlp, mc_dropout, ft_transformer, tabpfn, tabnet, mlp | torch-based | torch, tabpfn, pytorch-tabnet |
| **Linear** | ridge, elasticnet, lasso, huber | — | — |
| **Bayesian** | bayesian_ridge, gp | — | — |
| **Other** | svr, knn | — | — |
| **Ensemble** | stacking | — | — |

Models use lazy imports — missing optional deps produce clear error messages, not crashes. Install what you need:

```bash
uv sync                    # core (xgboost, sklearn, rich)
uv sync --extra torch      # + PyTorch/CUDA models
uv sync --extra boost      # + CatBoost, LightGBM
uv sync --extra all        # everything
```

GPU is auto-detected. When CUDA is available, XGBoost/CatBoost/LightGBM/PyTorch models use it automatically.

## How It Works

| pi-autoresearch (MCP) | This port (Plugin) |
|---|---|
| `init_experiment` tool | Agent writes config to `autoresearch.jsonl` (via `jq`) |
| `run_experiment` tool | Agent runs `./autoresearch.sh` with timing |
| `log_experiment` tool | Agent appends result via `scripts/ar-log.sh`, `git commit` on keep |
| TUI dashboard | `autoresearch-dashboard.md` |
| iteration cap in `config.json` | `maxRuns`/`maxSeconds`/`targetMetric` in the config header |
| — | **Stop hook** enforces the loop (never stops on its own) |
| — | **PreCompact** + **SessionStart** hooks survive context compaction |
| `before_agent_start` hook | `UserPromptSubmit` hook injects context + carries steers |

State lives in `autoresearch.jsonl`. Session artifacts (`*.jsonl`, dashboard, session doc, benchmark script, ideas backlog, worklog) are gitignored.

## Project Structure

```
.claude-plugin/plugin.json          # Plugin manifest
skills/autoresearch/SKILL.md        # Core skill: setup, JSONL protocol, run/log/loop logic
skills/autoresearch/scripts/ar-log.sh  # Appends valid-JSON result lines (jq, Python fallback)
commands/autoresearch.md            # /autoresearch (start, resume, status, report, off)
hooks/hooks.json                    # Hook definitions (plugin format)
hooks/autoresearch-stop.sh          # Stop hook — the loop engine + budget valve
hooks/autoresearch-precompact.sh    # PreCompact hook — snapshots state before compaction
hooks/autoresearch-sessionstart.sh  # SessionStart hook — rehydrates an active loop on resume
hooks/autoresearch-context.sh       # UserPromptSubmit hook — context + user steers
install.sh / uninstall.sh           # Manual symlink install (alternative to plugin)
examples/                           # Demo: fastball velocity prediction
  train.py                          # Training script with rich TUI output (AR_SEED-aware)
  models.py                         # Model registry (19 models, GPU detection)
  pyproject.toml                    # uv project config with dependency groups
  obp-autoresearch.md               # Session config for the OBP demo
  autoresearch.sh                   # Benchmark runner (takes optional SEED arg)
```

## License

[MIT](LICENSE)
