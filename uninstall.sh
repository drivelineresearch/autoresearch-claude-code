#!/usr/bin/env bash
# Uninstall autoresearch-claude-code from ~/.claude/
set -euo pipefail

CLAUDE_DIR="$HOME/.claude"

echo "Uninstalling autoresearch for Claude Code..."

# Remove skill symlink
if [ -L "$CLAUDE_DIR/skills/autoresearch" ] || [ -d "$CLAUDE_DIR/skills/autoresearch" ]; then
  rm -rf "$CLAUDE_DIR/skills/autoresearch"
  echo "  Removed skill: ~/.claude/skills/autoresearch"
fi

# Remove command symlink
if [ -L "$CLAUDE_DIR/commands/autoresearch.md" ] || [ -f "$CLAUDE_DIR/commands/autoresearch.md" ]; then
  rm -f "$CLAUDE_DIR/commands/autoresearch.md"
  echo "  Removed command: ~/.claude/commands/autoresearch.md"
fi

# Remove hook scripts (symlinks or copies)
for hook in autoresearch-context.sh autoresearch-stop.sh autoresearch-precompact.sh autoresearch-sessionstart.sh; do
  if [ -e "$CLAUDE_DIR/hooks/$hook" ] || [ -L "$CLAUDE_DIR/hooks/$hook" ]; then
    rm -f "$CLAUDE_DIR/hooks/$hook"
    echo "  Removed hook: ~/.claude/hooks/$hook"
  fi
done

# Remove the four autoresearch hook entries from settings.json (leaves other settings intact).
SETTINGS="$CLAUDE_DIR/settings.json"
if [ -f "$SETTINGS" ] && command -v python3 >/dev/null 2>&1; then
  python3 - "$SETTINGS" <<'PY'
import json, sys
path = sys.argv[1]
try:
    cfg = json.load(open(path))
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit()
hooks = cfg.get("hooks", {})
for event in ("Stop", "PreCompact", "SessionStart", "UserPromptSubmit"):
    groups = hooks.get(event)
    if not groups:
        continue
    kept = [g for g in groups
            if not any("autoresearch-" in h.get("command", "") for h in g.get("hooks", []))]
    if kept:
        hooks[event] = kept
    else:
        hooks.pop(event, None)
if not hooks:
    cfg.pop("hooks", None)
json.dump(cfg, open(path, "w"), indent=2)
print("  Removed autoresearch hook entries from settings.json")
PY
else
  echo "  If you configured hooks manually, remove the autoresearch-* entries from ~/.claude/settings.json."
fi

echo ""
echo "Done!"
