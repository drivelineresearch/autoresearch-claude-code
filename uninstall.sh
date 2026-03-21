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

# Remove hook script
if [ -f "$CLAUDE_DIR/hooks/autoresearch-context.sh" ]; then
  rm -f "$CLAUDE_DIR/hooks/autoresearch-context.sh"
  echo "  Removed hook: ~/.claude/hooks/autoresearch-context.sh"
fi

echo ""
echo "Done! Remember to also remove the hook entry from ~/.claude/settings.json if present."
