# Claude Code plugin integration verification

Verified on 2026-09-07 with Claude Code **2.1.263**, loading this repository through
`--plugin-dir`. A refreshed OAuth session was required; no API key or alternate
provider was introduced. Personal settings/hooks and MCP servers were excluded
from the test so the observed hook events came from this plugin.

## Adapter correction

The first authenticated test exposed a command/skill name collision:
`Skill("autoresearch:autoresearch")` loaded `commands/autoresearch.md`, which
previously asked Claude to load the same-named skill without a file path. The
shared protocol was never injected. An ordinary Bash tool also had no
`CLAUDE_PLUGIN_ROOT` environment variable.

The adapter now explicitly reads the shared skill using the plugin root
substituted into command text. Only a manual installation with an unresolved
placeholder uses `~/.claude/skills/autoresearch/SKILL.md`. It does not recursively
invoke the same command. The rerun's native event stream confirms the substituted
absolute path, the Read of `SKILL.md`, and execution of the shared `ar-log.sh`.
This follows Claude's documented [plugin path substitution](https://code.claude.com/docs/en/plugins-reference#environment-variables).

## Full hook-driven cycle: passed

The fresh fixture was `/tmp/autoresearch-claude-final-iyh10j0u`. Its deterministic
scorer echoes `candidate.json`'s integer value; the correctness check accepts only
1 or 2. The baseline was 1, target 2, and the run budget was 2 including baseline.
Only `candidate.json` was authorized for candidate edits.

Claude ran in noninteractive mode with `--permission-mode dontAsk`, explicit
Read/Write/Edit/Bash/Glob/Grep/Skill tool permissions, `--max-budget-usd 5`, no
session persistence, and a separate 300-second process timeout. The prompt asked
it to acknowledge readiness first, then perform one candidate only if the actual
Stop hook requested continuation. It could not manually simulate a hook or create
the pause sentinel as part of the test instructions.

Native hook events and independent checks confirmed:

- SessionStart restored the fixture context; UserPromptSubmit reported active mode.
- The first Stop returned `decision: block`; Claude then loaded the shared skill.
- Exactly one candidate commit, `1ba0d5ebcadcae908c064536a361da376ff80de0`, changed
  only `candidate.json` from 1 to 2.
- Exactly one appended result: run 2, segment 0, `keep`, metric 2, full commit hash.
- Independently rerun scorer emitted `METRIC score=2`; correctness check exited 0.
- Original config/baseline bytes remained the ledger prefix; scorer, checks, and
  `.gitignore` hashes were unchanged. Dashboard/worklog updated; worktree clean.
- The final Stop created `.autoresearch-off` and reported the 2/2 run cap on
  stderr, with empty stdout allowing the turn to finish. Both Stop hooks exited 0.
- Claude exited 0 with a successful result and no permission denials.

Local evidence is retained in the fixture's `experiments/plugin-smoke/events.jsonl`
and `/tmp/autoresearch-claude-final-result.json`; it is not required to install or
run the plugin. This checks integration and control flow, not ML accuracy. It
does not establish independent scorer attestation or a Claude no-progress
watchdog; those limits remain in [the review](review.md#5-remaining-work-in-order).
