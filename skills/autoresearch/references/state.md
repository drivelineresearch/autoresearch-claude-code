# State protocol

Run commands from the experiment workspace. `AR_SCRIPTS` is the absolute path to
this skill's `scripts/` directory. Python 3.10+ is required; no jq or packages are
needed by the state helper.

## Configuration

```bash
python3 "$AR_SCRIPTS/ar_state.py" init \
  --name 'Optimize benchmark' --metric-name runtime --metric-unit s \
  --direction lower --noise-floor 0.03 --max-runs 50 --max-seconds 3600
```

The first nonblank line must be a config object. To change the evaluation contract,
append a new config with the same command plus `--new-segment`; remeasure its
baseline and noise floor. An existing file is never silently overwritten.

```json
{"type":"config","name":"Optimize benchmark","metricName":"runtime","metricUnit":"s","bestDirection":"lower","noiseFloor":0.03,"maxRuns":50,"maxSeconds":3600,"targetMetric":null,"startedAt":1788800000}
```

| Field | Meaning |
|---|---|
| `name`, `metricName`, `metricUnit` | Session label, primary metric identifier, display unit |
| `bestDirection` | `lower` or `higher` |
| `noiseFloor` | Nonnegative finite baseline sample standard deviation; default 0 |
| `maxRuns` | Current-segment result cap, including baseline/crashes/check failures; default 200 |
| `maxSeconds` | Wall time since `startedAt`, including time paused; null means unset |
| `targetMetric` | Stop when the best **kept** value reaches the target; null means unset |
| `startedAt` | Unix epoch seconds; required for a time budget |

`--max-runs none` is supported for compatibility, but use a finite cap for an
unattended session. The Codex supervisor also has independent turn/time limits.
These budgets do not measure API tokens or dollars. Claude's Stop hook checks
budgets between turns and cannot interrupt a stuck benchmark; set its own timeout.

## Results

```bash
"$AR_SCRIPTS/ar-log.sh" auto abc1234 1.25 keep auto \
  'avoid duplicate parsing' n_seeds=3 memory_mb=42
```

`auto` resolves the next global run/current segment under the writer lock; explicit
numbers remain supported and must match the next run/current segment.

```json
{"run":1,"commit":"abc1234","metric":1.25,"metrics":{"n_seeds":3,"memory_mb":42},"status":"keep","description":"avoid duplicate parsing","timestamp":1788800030,"segment":0}
```

| Field | Meaning |
|---|---|
| `run` | Positive sequential integer across every segment |
| `commit` | 7–64 hexadecimal characters: git commit after keep, or starting HEAD after discard/failure |
| `metric` | Finite primary value; use 0 for an unmeasurable crash |
| `metrics` | Secondary metric names mapped to finite numbers |
| `status` | `keep`, `discard`, `crash`, or `checks_failed` |
| `description` | Experiment summary; arbitrary quotes/newlines are safely encoded |
| `timestamp` | Unix epoch seconds |
| `segment` | Zero-based config-header index; rows must follow their segment's header |
| `op` | Optional `draft`, `improve`, or `debug` |
| `parent` | Optional earlier run number in the current segment, or null |

For search-tree metadata, use `ar-log.sh --op improve --parent 1 ...` (options precede
positional arguments). Include secondary metrics consistently after introducing
them within a segment; represent failures explicitly without presenting a zero as
a measured observation. The helper validates data shape and history, not whether
the agent actually ran the benchmark, satisfied the noise rule, or committed the
claimed code.

`AR_JSONL=/path/to/state.jsonl` selects another file for `ar-log.sh`. Python helper
commands accept `--path FILE` after the subcommand. Hooks and the Codex supervisor
read the workspace's `autoresearch.jsonl`.

## Status and recovery

```bash
python3 "$AR_SCRIPTS/ar_state.py" status
```

Success returns JSON with `valid`, `config`, `segment`, `total_runs`, `segment_runs`,
`next_run`, `total_kept`, `segment_kept`, `best`, `budget_reached`, and
`budget_reason`. Invalid/missing state exits 2 with a diagnostic. Counts and best
selection operate on parsed objects, independent of whitespace formatting.

Writers validate under `.autoresearch.jsonl.lock`, reject symlink destinations,
and publish an atomic replacement. This protects cooperating local writers;
run one experiment agent per worktree. Do not delete an active lock file or edit
JSONL by hand while a writer runs. These locks require a filesystem supporting
POSIX advisory locks. Writers wait up to five seconds for the lock, then fail with
a diagnostic and leave the state unchanged; retry only after checking the other
writer's progress. A completed in-flight result may still be logged after its
time or target boundary; controllers check the budget before starting another run.

Claude PreCompact makes a snapshot at
`experiments/autoresearch.jsonl.precompact.bak` and appends a checkpoint. It is a
recovery aid, not proof the current unlogged experiment finished. On corruption,
pause, retain the original file and git diff, inspect the snapshot/benchmark logs,
and reconcile verified rows before resuming. Never erase malformed lines merely
to satisfy the parser or reset an exhausted budget.
