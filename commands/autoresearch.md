---
description: Start, resume, or inspect autonomous experiment loop
argument-hint: [off | status | report | goal description]
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
  - Skill
---

# Autoresearch Command

You are starting, resuming, or inspecting an autonomous experiment loop.

## Handle arguments

Arguments: $ARGUMENTS

### If arguments = "off"

Create a `.autoresearch-off` sentinel file in the current directory:
```bash
touch .autoresearch-off
```
Then tell the user autoresearch mode is paused. It can be resumed by running `/autoresearch` again (which will delete the sentinel). The Stop hook honors this sentinel and will let the loop end.

### If arguments = "status"

Read-only. Do NOT run experiments or change state. Print a concise status:
1. `cat autoresearch-dashboard.md` if it exists (the pre-rendered table).
2. Otherwise reconstruct from `autoresearch.jsonl`: total runs vs `maxRuns`, kept/discarded/crashed/checks_failed counts, baseline, current best (metric + which run + Δ%), noise floor, and whether `.autoresearch-off` is set (paused) or the loop is live.
3. Show the last 3 worklog entries from `experiments/worklog.md`.
Then stop — this is a report, not a resume.

### If arguments = "report"

Read-only final report. Write `autoresearch-report.md` summarizing the session: objective, baseline → best (with %), the winning configuration/diff summary, what classes of change worked vs. failed (from the worklog meta-reviews), and any open ideas from `autoresearch.ideas.md`. Then print its path. Do NOT resume the loop.

### If `autoresearch.md` exists in the current directory (resume)

This is a resume. Do the following:

1. Delete `.autoresearch-off` if it exists
2. Read `autoresearch.md` to understand the objective, constraints, and what's been tried
3. Read `autoresearch.jsonl` to reconstruct state:
   - Count total runs, kept, discarded, crashed
   - Find baseline metric (first result in current segment)
   - Find best metric and which run achieved it
   - Identify which secondary metrics are being tracked
4. Read recent git log: `git log --oneline -20`
5. If `autoresearch.ideas.md` exists, read it for experiment inspiration
6. Continue the loop from where it left off — pick up the next experiment

### If `autoresearch.md` does NOT exist (fresh start)

1. **Verify you're in a git repo** (`git rev-parse --git-dir`). If not, stop and tell the user to `git init` first — the loop needs git to keep/revert experiments.
2. Delete `.autoresearch-off` if it exists
3. Invoke the `autoresearch` skill to set up the experiment from scratch
4. If arguments were provided (other than a subcommand), use them as the goal description to skip/answer the setup questions
