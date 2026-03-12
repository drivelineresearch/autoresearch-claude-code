# autoresearch-claude-code

Autonomous experiment loop for [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Port of [pi-autoresearch](https://github.com/davebcn87/pi-autoresearch) as a pure skill (no MCP server).

Continuously optimizes any measurable target (test speed, bundle size, training loss, model accuracy, etc.) by running experiments, measuring results, keeping winners, discarding losers, and looping forever until interrupted.

## Install

Clone the repo and tell Claude Code to install it:

```bash
git clone https://github.com/drivelineresearch/autoresearch-claude-code.git ~/autoresearch-claude-code
```

Then in Claude Code, point it at the repo and ask it to install:

```
Install autoresearch from ~/autoresearch-claude-code
```

Claude will run `install.sh`, which symlinks the skill and command into `~/.claude/` and copies the hook script. It will also guide you through adding the hook to your `~/.claude/settings.json`.

### Manual install

If you prefer to install manually:

```bash
cd ~/autoresearch-claude-code
./install.sh
```

Then add the hook to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {"type": "command", "command": "~/.claude/hooks/autoresearch-context.sh"}
        ]
      }
    ]
  }
}
```

## Usage

In Claude Code:

```
/autoresearch optimize test suite runtime
```

This will:
1. Ask/infer: goal, command, metric, files in scope, constraints
2. Create a git branch `autoresearch/<goal>-<date>`
3. Write `autoresearch.md` (session doc), `autoresearch.sh` (benchmark script), and `experiments/worklog.md` (narrative log)
4. Run a baseline, then loop forever optimizing

### Commands

- `/autoresearch <goal>` — Start a new experiment loop
- `/autoresearch` — Resume an existing loop (if `autoresearch.md` exists)
- `/autoresearch off` — Pause autoresearch mode

### Steering

Send a message while an experiment is running to steer the next experiment. The agent will finish the current experiment, log it, then incorporate your idea.

### Ideas backlog

The agent writes promising but complex ideas to `autoresearch.ideas.md`. On resume, it reads this file for inspiration.

## How it works

The original pi-autoresearch used 3 registered MCP tools. This port encodes all that logic as skill instructions that tell Claude to use its built-in Bash/Read/Write tools:

| Original | Claude Code |
|---|---|
| `init_experiment` tool | Agent writes config header to `autoresearch.jsonl` |
| `run_experiment` tool | Agent runs `bash -c "./autoresearch.sh"` with timing |
| `log_experiment` tool | Agent appends result JSON, runs `git commit` if keep |
| TUI dashboard | `autoresearch-dashboard.md` file |
| `before_agent_start` hook | `UserPromptSubmit` hook injects context |
| Steer queueing | Skill instructions |

### State format

All state lives in `autoresearch.jsonl`:

```jsonl
{"type":"config","name":"optimize tests","metricName":"duration_s","metricUnit":"s","bestDirection":"lower"}
{"run":1,"commit":"abc1234","metric":42.3,"metrics":{},"status":"keep","description":"baseline","timestamp":1234567890,"segment":0}
{"run":2,"commit":"def5678","metric":39.1,"metrics":{},"status":"keep","description":"parallelize setup","timestamp":1234567891,"segment":0}
```

### Experiment artifacts

During a session, several files are created in your working directory. These are all gitignored by default:

| File | Purpose |
|---|---|
| `autoresearch.jsonl` | Machine-readable experiment state |
| `autoresearch-dashboard.md` | Human-readable results table |
| `autoresearch.md` | Session config (objective, constraints, scope) |
| `autoresearch.sh` | Benchmark runner script |
| `autoresearch.ideas.md` | Ideas backlog for future experiments |
| `experiments/worklog.md` | Narrative log of all experiments and insights |

## Uninstall

```bash
cd ~/autoresearch-claude-code
./uninstall.sh
```

Then remove the hook entry from `~/.claude/settings.json`.

## License

[MIT](LICENSE)
