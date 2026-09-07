---
description: Start, resume, inspect, report, or pause a measured experiment loop
argument-hint: [off | status | report | goal description]
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
---

# Autoresearch command

Arguments: $ARGUMENTS

Read `${CLAUDE_PLUGIN_ROOT}/skills/autoresearch/SKILL.md` and follow its protocol.
Claude substitutes that plugin path in this command's text; it need not exist as
an environment variable in Bash. For a manual install, where the placeholder is
not expanded, read `~/.claude/skills/autoresearch/SKILL.md` instead. Resolve
`AR_SCRIPTS` to the `scripts` directory beside the file you read. Do not invoke
`autoresearch` through the Skill tool again: this adapter shares that name and
can shadow the shared skill. If neither known path exists, pause and report the
missing installation rather than searching outside those locations.

Handle the requested mode before any resume or setup action:

- **off:** create `.autoresearch-off` in the experiment workspace, preserve partial
  work, and stop. Do not launch another experiment.
- **status:** read state/dashboard/recent worklog and show current segment budget,
  baseline/best, counts, and pause status. Do not resume or change experiment data.
  For a live Claude session, first create `.autoresearch-inspect` so the Stop hook
  permits this inspection turn to end. This small control write is the only write.
- **report:** for a live Claude session create `.autoresearch-inspect`; write
  `autoresearch-report.md` with results, evidence limits, winning changes, failures,
  and remaining ideas. Do not run experiments or unpause. This mode writes a report.
- **resume (session exists):** read session/state/worklog and git status/log. Resolve
  partial/unlogged work first and check the budget. Remove `.autoresearch-off` only
  when explicitly resuming within budget; do not reset the config to gain runs.
  An exhausted budget needs a user extension or deliberate new experiment contract.
- **fresh goal (no session):** verify git and preserve existing work, then follow
  skill setup. Infer goal/metric/scope from supplied arguments where possible.

A user stop, cancellation, or new scope overrides continuation. Complete one
experiment per autonomous turn; the Claude Stop hook handles the next turn when
valid state and budget permit it. Never treat a plugin code review as a request to
start an optimization session.
