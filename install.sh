#!/usr/bin/env bash
# Install autoresearch-claude-code into ~/.claude/
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CLAUDE_DIR="$HOME/.claude"

echo "Installing autoresearch for Claude Code..."

# Symlink skill
mkdir -p "$CLAUDE_DIR/skills"
ln -sfn "$REPO_DIR/skills/autoresearch" "$CLAUDE_DIR/skills/autoresearch"
echo "  Linked skill: ~/.claude/skills/autoresearch"

# Symlink command
mkdir -p "$CLAUDE_DIR/commands"
ln -sfn "$REPO_DIR/commands/autoresearch.md" "$CLAUDE_DIR/commands/autoresearch.md"
echo "  Linked command: ~/.claude/commands/autoresearch.md"

# Symlink hook scripts (symlink, not copy, so `git pull` updates them in place)
mkdir -p "$CLAUDE_DIR/hooks"
for hook in autoresearch-context.sh autoresearch-stop.sh autoresearch-precompact.sh autoresearch-sessionstart.sh; do
  ln -sfn "$REPO_DIR/hooks/$hook" "$CLAUDE_DIR/hooks/$hook"
  echo "  Linked hook: ~/.claude/hooks/$hook"
done

# Register the four hooks in settings.json (idempotent merge — never clobbers existing settings).
SETTINGS="$CLAUDE_DIR/settings.json"
if command -v python3 >/dev/null 2>&1; then
  python3 - "$SETTINGS" <<'PY'
import json, os, sys
path = sys.argv[1]
try:
    with open(path) as f:
        cfg = json.load(f)
except FileNotFoundError:
    cfg = {}

wants = {
    "Stop":            ("",                     "autoresearch-stop.sh"),
    "PreCompact":      ("",                     "autoresearch-precompact.sh"),
    "SessionStart":    ("startup|resume|compact","autoresearch-sessionstart.sh"),
    "UserPromptSubmit":("",                     "autoresearch-context.sh"),
}
hooks = cfg.setdefault("hooks", {})
changed = False
for event, (matcher, script) in wants.items():
    cmd = "~/.claude/hooks/" + script
    groups = hooks.setdefault(event, [])
    if any(cmd in h.get("command", "") for g in groups for h in g.get("hooks", [])):
        continue
    entry = {"hooks": [{"type": "command", "command": cmd}]}
    if matcher:
        entry["matcher"] = matcher
    groups.append(entry)
    changed = True

if changed:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("  Registered autoresearch hooks in settings.json")
else:
    print("  Hooks already configured in settings.json")
PY
else
  echo "  python3 not found — add these to $SETTINGS 'hooks' manually:"
  echo '    Stop / PreCompact / UserPromptSubmit → ~/.claude/hooks/autoresearch-{stop,precompact,context}.sh'
  echo '    SessionStart (matcher "startup|resume|compact") → ~/.claude/hooks/autoresearch-sessionstart.sh'
fi

echo ""
echo "Done! Run /autoresearch in Claude Code to start."
