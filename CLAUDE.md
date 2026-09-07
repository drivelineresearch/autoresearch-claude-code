# Claude Code contributor guidance

Read [AGENTS.md](AGENTS.md) for the shared repository instructions.

`commands/autoresearch.md` uses Claude's `$ARGUMENTS`. Plugin commands in
`hooks/hooks.json` use a quoted `${CLAUDE_PLUGIN_ROOT}`. Manual installation links
these scripts and registers their quoted absolute paths. Hook workspaces come
from validated payload `cwd`; never use a failed `cd` as permission to operate in
the hook process's unrelated working directory.

Continuation uses JSON `{"decision":"block","reason":"..."}` on Stop stdout,
with a successful hook exit. Do not import this event contract into Codex's runner.
