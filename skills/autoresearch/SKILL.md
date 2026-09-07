---
name: autoresearch
description: Set up, resume, inspect, or pause a measured experiment loop in Claude Code or Codex. Use when asked to run autoresearch or iteratively optimize a named benchmark. Reviewing this plugin alone does not start an experiment loop.
---

# Autoresearch

Try a scoped change, measure it against a fixed benchmark, keep a supported improvement,
and record what happened. Continue within the user's authorization and a finite budget.
User interruptions and scope changes take precedence over continuation instructions.

## Host and mode

Resolve `AR_SCRIPTS` to the `scripts` directory alongside this skill's actual file.
The helpers require Python 3.10+; Bash wrappers and locking support Linux/macOS.

- **Claude Code:** invoke `/autoresearch` for manual installation, or
  `/autoresearch:autoresearch` when loaded as a plugin. The four registered Claude
  hooks handle continuation and compaction. Complete one experiment per turn.
- **Codex:** invoke `$autoresearch` in CLI/IDE, or select this skill in the app.
  Use the same protocol with Codex's available read/edit/exec tools. This package's
  Claude hook registration does not install Codex hooks. For unattended continuation,
  use `scripts/codex_loop.py` as described in [references/codex.md](references/codex.md).
  A supervisor turn must complete exactly one experiment and then end. Never nest
  a supervisor inside a supervised turn.
- **Status:** read state/dashboard/worklog and report; run no experiments and do not
  unpause. `python3 "$AR_SCRIPTS/ar_state.py" status` is a read-only status command.
- **Report:** write `autoresearch-report.md` with objective, baseline, best, winning
  changes, evidence limitations, failures, and remaining ideas; run no experiments.
- **Off/pause:** create `.autoresearch-off` and stop. During a running benchmark,
  cancel safely if possible; preserve partial work and report any unfinished process.

For Claude status/report turns in an active experiment, first create
`.autoresearch-inspect` in the experiment workspace, including when this skill
was loaded directly. The Stop hook consumes it to let the inspection turn end.
For status, this control-file write is the only write; it does not log or run an
experiment. Codex status remains read-only and creates no inspection sentinel.

## Setup

1. Establish the goal, benchmark command, primary metric and direction, exact files
   in scope, fixed scorer/data/splits, correctness checks, and budget. Infer these
   from the request when possible. Authorization to optimize code does not imply
   permission to push, deploy, contact people, install dependencies, or rent compute.
2. Inspect git status, including untracked files and the index. Use an experiment
   branch in a clean dedicated worktree when other work is present; preserve that
   work without stashing or cleaning it away. Require a committed starting point.
3. Read the workload. Write `autoresearch.md` using the outline below and create
   `experiments/worklog.md`. Prepare `autoresearch.sh` and, when needed, `checks.sh`.
   Commit only explicitly selected initial code/harness files; keep session artifacts
   local. Check ignores in the target repository: this plugin's `.gitignore` is not
   inherited by another project. Add anchored artifact patterns to that project's
   local git exclude file (locate it with `git rev-parse --git-path info/exclude`):
   `/autoresearch.jsonl`, `/autoresearch.md`, `/autoresearch-dashboard.md`,
   `/autoresearch-report.md`, `/autoresearch.ideas.md`, `/experiments/`,
   `/.autoresearch-off`, `/.autoresearch-inspect`, `/.autoresearch.jsonl.lock`,
   `/.autoresearch-codex.lock`. Preserve existing excludes. Track the benchmark or
   explicitly ignore `/autoresearch.sh` if it is session-local.
4. Lock the evaluation definition **before optimization**. Record scorer hashes,
   dataset/split identity, dependencies, command, seed schedule, and hardware where
   relevant. Keep benchmark and metric-emitting code outside editable scope. A
   legitimate scorer fix requires a new segment and a new baseline.
5. Validate the harness with an appropriate trivial/input-independent control.
   Run the unchanged baseline 3–5 times to estimate noise. Hold evaluation data and
   folds fixed; vary only training seeds or repeat timing measurements. Record the
   raw values, mean, and sample standard deviation. A zero measured deviation is
   not proof of zero uncertainty.
6. Initialize the state with the measured floor, then log the baseline as the first
   keep (an unchanged baseline need not create an empty commit). Use a finite
   `maxRuns` (default 200), and a practical wall-time budget. Setup/calibration runs
   are outside the logged experiment count: record and budget them separately.
7. Continue until a budget, target, user pause, or a real execution failure prevents
   progress. On failure, preserve evidence, pause, and explain the blocker.

### Session outline: `autoresearch.md`

```markdown
# Autoresearch: <goal>
## Objective
Workload, expected outcome, and what was learned from initial inspection.
## Metrics
Primary name/unit/direction, secondary metrics, measured noise floor and seed values.
## Budget
maxRuns, maxSeconds, optional targetMetric, per-benchmark timeout, setup cost.
## How to Run
Exact benchmark/check commands, working directory, runtime and dependencies.
## Files in Scope
Exact files the experiment may edit.
## Off Limits
Scorer, metric code, data, split/seed definition; record hashes/identity here.
## Constraints
Correctness requirements, resource/network limits, and allowed actions.
## What's Been Tried
Results, failed ideas, insights, and next candidates. Update every 5–10 runs.
```

## Run one experiment

1. Read the current segment, best kept result, session rules, and recent worklog.
   Check the budget **before launching**. Do not reset a budget to continue.
2. Record starting HEAD and clean status. Choose one hypothesis and its precise
   editable file list; keep scorer and evaluation data fixed. Use diverse drafts
   before refining a promising approach. Cap debugging one idea at three attempts.
3. Run the benchmark with a timeout and a unique log file. Capture stdout and stderr
   separately; parse `METRIC name=number` from stdout only. Check exit status even
   when a metric was printed. Reject missing, duplicate, or non-finite primary
   metrics. A failing benchmark is `crash`; passing benchmark plus failing checks
   is `checks_failed`. Both consume a run.
4. Inspect only metric lines and bounded log excerpts; keep full output on disk.
   Do not use a shared `/tmp/autoresearch-output.txt` or parse `tee`'s exit status as
   the benchmark status. Use Python's monotonic clock or an available timeout tool;
   GNU `date +%s%N` and `timeout` are not portable to stock macOS.
5. Verify locked files against the initial hashes **before accepting or committing**
   a result, including ignored harness files. An instruction to lock a harness is
   not an access-control boundary. The Codex supervisor checks configured hashes
   between turns; inspect a detected violation and re-baseline before trusting it.
6. Apply the decision rule, keep or restore the experiment's scoped changes, then
   append exactly one result and update the dashboard/worklog. Complete cleanup
   before ending a supervisor turn.

### Decision rule

- `keep`: a finite primary metric beats the current best by **strictly more than**
  `noiseFloor` in the right direction, and correctness checks pass.
- `discard`: worse, equal, or an improvement less than or equal to the floor.
- `crash`: failed/timed-out benchmark or invalid/missing primary metric; use metric
  `0` as a placeholder. This value is never eligible as a best result.
- `checks_failed`: benchmark passed but a correctness check failed; cannot be kept.

Compare means over the same seed schedule for a borderline improvement (within
about 2× the floor), rerunning both incumbent and candidate when needed. Log
`n_seeds` and the measured mean. A noise threshold is a practical heuristic, not a
significance guarantee across hundreds of adaptive trials. Equal-performance code
simplification requires a separately declared objective/acceptance rule, not an
exception silently applied to this metric contract.

For ML, fit imputation, scaling, feature selection, and early stopping exclusively
inside training folds. Fix groups/splits before searching. Use validation for
selection; reserve a final untouched test set. Repeated test-set monitoring that
steers subsequent ideas also leaks information. Changing sample aggregation, CV
folds, or the target definition starts a new segment; scores across these changes
are not measured improvements under one benchmark.

### Git operations

Never stage the entire repository or restore/clean the entire working tree.
Inspect the diff and index. On a keep, stage **only the explicit experiment files**
using `git add -- <files>`, check `git diff --cached`, and commit with a description
and a valid `Result: {...}` JSON trailer. Record the resulting HEAD hash.

On discard/crash/checks_failed, use `git restore --source=<starting-HEAD> -- <tracked-experiment-files>`
and remove only exact untracked files created by this experiment after inspecting
those paths. Preserve pre-existing files and artifacts. Do not use blanket
`git checkout -- .`, `git clean`, or `git reset --hard`. If ownership is unclear,
pause with the diff intact. Record starting HEAD for discarded results.

Backtracking requires a clean experiment workspace and an explicit recorded parent;
restore the intended scoped code from that parent. Compare against the segment's
best metric, and leave the best code in place when pausing. Do not silently detach
HEAD or change branches under the Codex supervisor.

## State, logging, and reporting

Read [references/state.md](references/state.md) for config/result schemas and helper
commands. `autoresearch.jsonl` is the authoritative append-only logical history.
The helper validates the entire history and atomically replaces it under a lock;
never hand-build JSON or overwrite prior segments. A malformed state is a reason
to pause and repair from evidence, not to silently skip rows or restart the count.

After each result, update `autoresearch-dashboard.md` with the current segment's
run budget, status counts, baseline/best, floor, and a table of **all** current
segment runs (run, commit, metric, delta, status, description). Flag undefined
percentages when the baseline is zero; avoid interpreting R² ratios as accuracy.

Append a worklog entry containing run/time, hypothesis, exact change, metrics,
keep/discard explanation, log path, insight, and next idea. Refresh the session
summary and `autoresearch.ideas.md` periodically. At a boundary, summarize verified
results, remaining budget, and partial work. An empty ideas file does not prove
research is complete.

On resume, read session, state, worklog, and git status/log first. Reconcile any
unlogged benchmark, unfinished diff, or commit-without-result before another run;
do not fabricate missing results or blindly repeat a possibly completed job.
Remove `.autoresearch-off` only for an explicit resume within the current budget.
An exhausted budget requires the user's extension or a deliberately new experiment
contract, not merely removing the sentinel.
