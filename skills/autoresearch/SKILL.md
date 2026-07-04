---
name: autoresearch
description: Set up and run an autonomous experiment loop for any optimization target. Use when asked to "run autoresearch", "optimize X in a loop", "set up autoresearch for X", or "start experiments".
---

# Autoresearch

Autonomous experiment loop: try ideas, keep what works, discard what doesn't, never stop.

## Loop enforcement (how "never stop" actually works)

Four plugin hooks turn "please keep looping" into mechanism. You don't call them — they fire automatically — but you must produce the state they read (`autoresearch.md`, `autoresearch.jsonl`, `experiments/worklog.md`):

- **Stop hook** — the loop engine. On every turn-end it vetoes the stop and feeds you the next instruction, *unless* a budget boundary is hit (`maxRuns`/`maxSeconds`/`targetMetric` in the config header) or `.autoresearch-off` exists. This is why you never have to ask "should I continue?" — you can't accidentally end the loop; only a real budget or `/autoresearch off` ends it. (Read-only `/autoresearch status`/`report` turns set a one-shot `.autoresearch-inspect` sentinel so they can end cleanly without triggering an experiment.)
- **PreCompact hook** — snapshots `autoresearch.jsonl` and writes a checkpoint to the worklog before context compaction, so nothing is lost.
- **SessionStart hook** — on resume/compaction, re-injects the objective + best result + recent worklog so a fresh agent continues instead of restarting.
- **UserPromptSubmit hook** — reinjects context and carries user *steers* mid-loop.

Because the Stop hook enforces continuation, your job is to make each turn *one complete experiment* (run → log → commit-or-revert → update dashboard + worklog), then let the hook bounce you into the next one.

## Setup

0. **Refuse to start outside a git repo.** The loop's keep/discard depends on `git commit`/`git checkout` to save winners and revert losers. If `git rev-parse --git-dir` fails, stop and tell the user to `git init` first (or point the loop at a repo). No git → no autoresearch.
1. Ask (or infer): **Goal**, **Command**, **Metric** (+ direction), **Files in scope**, **Constraints**.
2. `git checkout -b autoresearch/<goal>-<date>`
3. **Become one with the workload.** Read the source files and study the data/workload — distributions, outliers, where time or error actually goes. Summarize it in `autoresearch.md`. The best experiments come from understanding, not random variation (Karpathy's "Recipe").
4. `mkdir -p experiments` then write `autoresearch.md`, `autoresearch.sh`, and `experiments/worklog.md` (see below). Commit all three.
5. **Lock the eval harness.** `autoresearch.sh` and any code that *emits the metric* are OFF LIMITS to experiments — an agent that can edit its own scorer will eventually game it (Karpathy: "the eval is the product"). List the harness under "Off Limits" in `autoresearch.md`, and make `autoresearch.sh` fix a random seed so runs are reproducible. If the benchmark supports it, have `autoresearch.sh` accept a `SEED` argument.
6. **Sanity gate + noise floor (before run 1):**
   - *Sanity:* run a trivial/input-independent baseline (predict the mean, shuffle labels, empty change) and confirm the metric gets *worse*. If a broken model scores well, the harness is wrong — fix it before optimizing anything.
   - *Noise floor:* run the unchanged baseline 3–5× (varying the seed) and compute the stddev of the primary metric with `awk`. Patch it into the config header's `noiseFloor`. This is the bar every future "improvement" must clear.
7. Initialize experiment (write config header to `autoresearch.jsonl`) → run baseline → log result → start looping immediately.

### `autoresearch.md`

This is the heart of the session. A fresh agent with no context should be able to read this file and run the loop effectively. Invest time making it excellent.

```markdown
# Autoresearch: <goal>

> The metric is the sole arbiter. "This looks better" never drives a keep/discard —
> only a measured improvement that clears the noise floor does.

## Objective
<Specific description of what we're optimizing and the workload — including what you
learned "becoming one with the data": distributions, outliers, where error/time goes.>

## Metrics
- **Primary**: <name> (<unit>, lower/higher is better)
- **Secondary**: <name>, <name>, ...
- **Noise floor**: <stddev> (deltas smaller than this are noise, not progress)

## Budget
- maxRuns: <N> | maxSeconds: <N|none> | targetMetric: <value|none>
- Per-experiment wall-clock cap: <e.g. 5 min> (kill + mark crash past this)

## How to Run
`./autoresearch.sh [SEED]` — outputs `METRIC name=number` lines. Fixes a random seed for reproducibility.

## Files in Scope
<Every file the agent may modify, with a brief note on what it does.>

## Off Limits
- `autoresearch.sh` and any metric-emitting code — **the eval harness is locked.** Never edit it to change what/how the metric is measured. (If it genuinely needs a fix, do it as a deliberate, separately-noted step, then re-baseline.)
- <anything else that must not be touched.>

## Constraints
<Hard rules: tests must pass, no new deps, correctness checks (checks.sh) must pass, etc.>

## What's Been Tried
<Update this section as experiments accumulate. Note key wins, dead ends,
and architectural insights so the agent doesn't repeat failed approaches.
Every ~10 runs, add a meta-review: which *classes* of change win vs lose.>
```

Update `autoresearch.md` periodically — especially the "What's Been Tried" section — so resuming agents have full context.

### `autoresearch.sh`

Bash script (`set -euo pipefail`) that: pre-checks fast (syntax errors in <1s), fixes a random seed (accept an optional `SEED` arg for reproducibility re-runs), runs the benchmark, and outputs `METRIC name=number` lines. Keep it fast — every second is multiplied by hundreds of runs, so also enforce a per-experiment wall-clock cap (e.g. `timeout 300`) that kills and marks a crash.

**This is the locked eval harness — do NOT edit it during the loop.** An agent that can change how the metric is measured will eventually game its own scorer. Get it right at setup. If it genuinely needs a fix mid-session, treat that as a deliberate re-baseline step (note it in the worklog, re-measure the noise floor), not a routine experiment.

---

## JSONL State Protocol

All experiment state lives in `autoresearch.jsonl`. This is the source of truth for resuming across sessions.

### Config Header

The first line (and any re-initialization line) is a config header:

```json
{"type":"config","name":"<session name>","metricName":"<primary metric name>","metricUnit":"<unit>","bestDirection":"lower|higher","noiseFloor":<stddev>,"maxRuns":<N|null>,"maxSeconds":<N|null>,"targetMetric":<value|null>,"startedAt":<epoch seconds>}
```

Fields beyond the original four are the **budget and noise contract the Stop hook reads** (see "Loop enforcement" above):
- `noiseFloor`: stddev of the primary metric across repeated baseline runs (see Setup step 6). Keep/discard compares deltas against this. `0` if unmeasured.
- `maxRuns`: cap on runs **in the current segment** before the loop auto-pauses. **Always set this** (default 200) so an unattended overnight run can't burn unbounded metered budget. `null` = unlimited (only for cheap benchmarks you're watching). Re-initializing (a new segment) gives a fresh count, so prior segments never instantly cap a new target.
- `maxSeconds`: wall-clock budget from this segment's `startedAt`. `null` = none.
- `targetMetric`: stop once the current segment's best kept metric reaches this (respecting `bestDirection`). `null` = keep optimizing.
- `startedAt`: Unix epoch when the segment began (`date +%s`), used for `maxSeconds`.

Rules:
- First line of the file is always a config header.
- Each subsequent config header (re-init) starts a new **segment**. Segment index increments with each config header.
- The baseline for a segment is the first result line after the config header.
- Re-init resets `startedAt` and may change budgets, but **never change `metricName`/`bestDirection` mid-segment** — moving the goalposts invalidates every prior comparison.

### Result Lines

Each experiment result is appended as a JSON line:

```json
{"run":1,"commit":"abc1234","metric":42.3,"metrics":{"secondary_metric":123},"status":"keep","description":"baseline","op":"draft","parent":null,"timestamp":1234567890,"segment":0}
```

Fields:
- `run`: sequential run number (1-indexed, across all segments)
- `commit`: 7-char git short hash (the commit hash AFTER the auto-commit for keeps, or current HEAD for discard/crash/checks_failed)
- `metric`: primary metric value (0 for crashes)
- `metrics`: object of secondary metric values — **once you start tracking a secondary metric, include it in every subsequent result**. Also the home for `n_seeds`, `test_<metric>`, etc.
- `status`: `keep` | `discard` | `crash` | `checks_failed`
- `description`: short description of what this experiment tried
- `op`: `draft` | `improve` | `debug` — the operator this experiment applied (see "Search strategy"). Optional but recommended.
- `parent`: the `run` number this experiment branched from (the code state it started from), or `null` for a draft-from-scratch. Makes the run history a **tree**, not just a linear chain — enables backtracking out of local optima.
- `timestamp`: Unix epoch seconds
- `segment`: current segment index

The `ar-log.sh` helper doesn't take `op`/`parent` positionally — if you track them, append with `jq -nc` (see below) or add them to the object. They're optional; the loop works without them, but they make backtracking and meta-review far easier.

### Initialization (equivalent of `init_experiment`)

To initialize, write the config header to `autoresearch.jsonl` (build it with `jq -nc` so it's valid JSON, then measure the noise floor and patch it in during Setup step 6):

```bash
jq -nc --arg name "<name>" --arg metricName "<metric>" --arg metricUnit "<unit>" \
  --arg bestDirection "<lower|higher>" --argjson maxRuns 200 --argjson startedAt "$(date +%s)" \
  '{type:"config",name:$name,metricName:$metricName,metricUnit:$metricUnit,bestDirection:$bestDirection,noiseFloor:0,maxRuns:$maxRuns,maxSeconds:null,targetMetric:null,startedAt:$startedAt}' \
  > autoresearch.jsonl
```

To re-initialize (change optimization target), **append** a new config header (same command with `>>`), which starts a new segment and resets `startedAt`.

---

## Running Experiments (equivalent of `run_experiment`)

Run the benchmark command, capturing timing and output:

```bash
START_TIME=$(date +%s%N)
./autoresearch.sh > /tmp/autoresearch-output.txt 2>&1
EXIT_CODE=$?
END_TIME=$(date +%s%N)
DURATION=$(awk "BEGIN{printf \"%.3f\", ($END_TIME - $START_TIME)/1000000000}")
cat /tmp/autoresearch-output.txt
echo "Duration: ${DURATION}s, Exit code: ${EXIT_CODE}"
```

> **Do not pipe the benchmark into `tee`.** `EXIT_CODE=$?` after a pipe captures `tee`'s exit code (always 0), so real crashes would silently register as successes. Redirect to a file and `cat` it, or use `${PIPESTATUS[0]}` if you must pipe. Duration uses `awk` (not `bc`, which isn't installed everywhere).

After running:
- Parse `METRIC name=number` lines: `grep '^METRIC ' /tmp/autoresearch-output.txt`
- If exit code != 0 → this is a crash
- **Don't `cat` the whole output into context.** A verbose training job floods the window across hundreds of runs and accelerates compaction. Read the `METRIC` lines plus a bounded slice — `tail -c 4000 /tmp/autoresearch-output.txt` (and `head` if a crash traceback is at the top). The full log stays on disk if you need it.

### Correctness gate (optional `checks.sh`)

If a `checks.sh` exists, run it *after* a passing benchmark. It encodes correctness invariants the primary metric doesn't capture (held-out accuracy didn't regress, output still valid, no NaNs). If it exits non-zero, the result's status is **`checks_failed`** and the change **cannot be kept** — a faster-but-wrong change must never be committed. This separates "is it better?" (the metric) from "is it still correct?" (the checks).

---

## Logging Results (equivalent of `log_experiment`)

After each experiment run, follow this exact protocol:

### 1. Determine status

- **keep**: primary metric improved *by more than the noise floor* (`|new − best| > noiseFloor`, in the right direction), AND `checks.sh` passed (if present).
- **discard**: primary metric worse, equal, or improved by *less than the noise floor* (a delta inside the noise band is not signal — log it as discard with a note, don't bank it).
- **crash**: command failed (non-zero exit code).
- **checks_failed**: benchmark passed but `checks.sh` failed. Treated like discard (revert), but recorded distinctly so a recurring correctness regression is visible.

**Confirm borderline wins.** If a candidate beats best but by an amount *near* the noise floor (say within ~2×), don't trust one run — re-run it 2–3× with different seeds and keep only if the **mean** still beats best. Single-run luck permanently corrupts the baseline; over hundreds of unattended runs this is the difference between real progress and banking noise. Record `n_seeds` in `metrics` when you do this.

**Held-out test set (ML targets).** The loop optimizes on validation only. If you reserved a test set at setup, evaluate it *occasionally as monitoring* (log it in `metrics`, e.g. `test_r2`), **never** as a keep/discard input. If validation keeps improving while held-out test flatlines or worsens, you're overfitting the split — flag it in the worklog and dashboard.

Secondary metrics are for monitoring only — they almost never affect keep/discard decisions. Only discard a primary improvement if a secondary metric degraded catastrophically, and explain why in the description.

### 2. Git operations

**If keep:**
```bash
git add -A
git diff --cached --quiet && echo "nothing to commit" || git commit -m "<description>

Result: {\"status\":\"keep\",\"<metricName>\":<value>,<secondary metrics>}"
```

Then get the new commit hash:
```bash
git rev-parse --short=7 HEAD
```

**If discard, crash, or checks_failed:**
```bash
git checkout -- .
git clean -fd
```

> **Warning:** Never use `git clean -fdx` — the `-x` flag deletes gitignored files including JSONL state, dashboards, and experiment artifacts.

Use the current HEAD hash (before revert) as the commit field.

### 3. Append result to JSONL

**Never hand-build the JSON with `echo`** — a quote or apostrophe in the description (`don't scale y`, `n_estimators="auto"`) produces invalid JSON and silently corrupts the state file. Use the shipped helper, which builds valid JSON with `jq` (or a Python fallback):

```bash
# scripts/ar-log.sh <run> <commit> <metric> <status> <segment> <description> [k=v secondary metrics...]
"$AR_SCRIPTS/ar-log.sh" 12 abc1234 40.1 keep 0 "optimize hot loop" rmse=2.20 mean_uncertainty=0.31
```

`$AR_SCRIPTS` points at the plugin's `scripts/` dir (resolve it once at setup: it is `<skill dir>/scripts` — the skill lives at `${CLAUDE_PLUGIN_ROOT}/skills/autoresearch/` under the plugin, or `~/.claude/skills/autoresearch/` for manual installs). If the helper is somehow unavailable, fall back to `jq -nc` — still never raw `echo`:

```bash
jq -nc --argjson run 12 --arg commit abc1234 --argjson metric 40.1 \
  --arg status keep --argjson segment 0 --arg description "optimize hot loop" \
  --argjson metrics '{"rmse":2.20}' --argjson ts "$(date +%s)" \
  '{run:$run,commit:$commit,metric:$metric,metrics:$metrics,status:$status,description:$description,timestamp:$ts,segment:$segment}' \
  >> autoresearch.jsonl
```

### 4. Update dashboard

After every log, regenerate `autoresearch-dashboard.md` (see Dashboard section below).

### 5. Append to worklog

After every experiment, append a concise entry to `experiments/worklog.md`. This file survives context compactions and crashes, giving any resuming agent (or the user) a complete narrative of the session. Format:

```markdown
### Run N: <short description> — <primary_metric>=<value> (<STATUS>)
- Timestamp: YYYY-MM-DD HH:MM
- What changed: <1-2 sentences describing the code/config change>
- Result: <metric values>, <delta vs best>
- Insight: <what was learned, why it worked/failed>
- Next: <what to try next based on this result>
```

Also update the "Key Insights" and "Next Ideas" sections at the bottom of the worklog when you learn something new.

**On setup**, create `experiments/worklog.md` with the session header, data summary, and baseline result. **On resume**, read `experiments/worklog.md` to recover context.

### 6. Secondary metric consistency

Once you start tracking a secondary metric, you MUST include it in every subsequent result. Parse the JSONL to discover which secondary metrics have been tracked and ensure all are present.

If you want to add a new secondary metric mid-session, that's fine — but from that point forward, always include it.

---

## Dashboard

After each experiment, regenerate `autoresearch-dashboard.md`:

```markdown
# Autoresearch Dashboard: <name>

**Runs:** 12 / 200 | **Kept:** 8 | **Discarded:** 3 | **Crashed:** 1 | **Checks-failed:** 0
**Baseline:** <metric_name>: <value><unit> (#1)
**Best:** <metric_name>: <value><unit> (#8, -26.2%)
**Noise floor:** ±<stddev><unit> — improvements smaller than this are noise
**Val/test gap:** <if tracking a held-out set: val best vs test; widening = overfitting alarm>

| # | commit | <metric_name> | Δ vs best | status | op | description |
|---|--------|---------------|-----------|--------|-----|-------------|
| 1 | abc1234 | 42.3s | — | keep | draft | baseline |
| 2 | def5678 | 40.1s (-5.2%) | -2.2 (>floor ✓) | keep | improve | optimize hot loop |
| 3 | abc1234 | 43.0s (+1.7%) | +0.7 | discard | improve | try vectorization |
| 4 | def5678 | 40.0s (-0.2%) | -0.1 (<floor) | discard | improve | reorder branches (noise) |
...
```

Include delta percentages vs baseline, and a "Δ vs best" column flagging whether a keep cleared the noise floor. Show ALL runs in the current segment (not just recent ones).

---

## Loop Rules

**LOOP FOREVER** (within budget). Never ask "should I continue?" — the user expects autonomous work.

- **Primary metric is king — but only past the noise floor.** Improved by `> noiseFloor` → `keep`. Worse, equal, or within the noise band → `discard`. Secondary metrics rarely affect this.
- **One atomic change per experiment.** Each run alters exactly one lever (one hyperparameter, one code path, one idea). If a hypothesis needs several coupled changes, split it across runs — otherwise you can't attribute the result. This makes the worklog a clean ablation.
- **Lock the harness.** After every keep, `git diff <commit-before> -- autoresearch.sh` (and any metric-emitting file). If the harness changed, you (or a bad edit) moved the goalposts — revert the harness and re-run before trusting the number.
- **Don't be a hero.** Before inventing a novel change, search for and copy the simplest known-good approach (existing implementation, prior art). Diverge only after the baseline-good version works. Note in the description whether you copied or invented.
- **Simpler is better.** Removing code for equal perf = keep. Ugly complexity for a tiny gain = discard even if the metric ticked up inside the noise.
- **Crashes and debug:** fix if trivial. Cap consecutive debug attempts on one broken idea at **3** — past that, abandon the path (revert to its parent) and try something else. Don't rabbit-hole.
- **Think longer when stuck.** Re-read source files, study the profiling data, reason about what's actually happening. The best ideas come from deep understanding, not random variation.
- **Resuming:** if `autoresearch.md` exists, read it + `autoresearch.jsonl` + `experiments/worklog.md` + git log, continue looping. (The SessionStart hook injects a summary automatically.)

### Search strategy (avoid greedy local optima)

A pure "always improve the current best" chain gets stuck. Structure the search:

1. **Draft first (explore).** Spend the first 3–5 experiments on *structurally different* approaches (`op:"draft"`, `parent:null`) before greedily refining — e.g. for an ML target, try boosting vs. a neural net vs. a linear model, not five variants of one. Committing to the first thing that beats baseline is how you land in a shallow local optimum.
2. **Improve (exploit).** Once you have a strong draft, make atomic improvements to it (`op:"improve"`, `parent:<its run>`).
3. **Backtrack.** If the best node hasn't improved in ~K runs, `git checkout` a *promising non-best* ancestor and branch a different direction from there. The `parent` field makes the history a tree you can navigate.
4. **Meta-review every ~10 runs.** Write a short block in the worklog distilling which *classes* of change consistently win or lose, and let it steer the next batch.

**NEVER STOP on your own.** The Stop hook keeps you looping until a real budget boundary (`maxRuns`/`maxSeconds`/`targetMetric` in the config header) or the user runs `/autoresearch off`. The user may be away for hours — keep going until interrupted or the budget is genuinely exhausted.

## Ideas Backlog

When you discover complex but promising optimizations that you decide not to pursue right now, **append them as bullet points to `autoresearch.ideas.md`**. Don't let good ideas get lost.

If the loop stops (context limit, crash, etc.) and `autoresearch.ideas.md` exists, you'll be asked to:
1. Read the ideas file and use it as inspiration for new experiment paths
2. Prune ideas that are duplicated, already tried, or clearly bad
3. Create experiments based on the remaining ideas
4. If nothing is left, try to come up with your own new ideas
5. If all paths are exhausted, delete `autoresearch.ideas.md` and write a final summary report

When there is no `autoresearch.ideas.md` file and the loop ends, the research is complete.

## User Steers

User messages sent while an experiment is running should be noted and incorporated into the NEXT experiment. Finish your current experiment first — don't stop or ask for confirmation. Incorporate the user's idea in the next experiment.

## Updating autoresearch.md

Periodically update `autoresearch.md` — especially the "What's Been Tried" section — so that a fresh agent resuming the loop has full context on what worked, what didn't, and what architectural insights have been gained. Do this every 5-10 experiments or after any significant breakthrough.
