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

# Copy hook script
mkdir -p "$CLAUDE_DIR/hooks"
cp "$REPO_DIR/hooks/autoresearch-context.sh" "$CLAUDE_DIR/hooks/autoresearch-context.sh"
chmod +x "$CLAUDE_DIR/hooks/autoresearch-context.sh"
echo "  Copied hook: ~/.claude/hooks/autoresearch-context.sh"

# Check if hook is already configured in settings.json
SETTINGS="$CLAUDE_DIR/settings.json"
if [ -f "$SETTINGS" ]; then
  if grep -q "autoresearch-context.sh" "$SETTINGS" 2>/dev/null; then
    echo "  Hook already configured in settings.json"
  else
    echo ""
    echo "Add the hook to $SETTINGS in the UserPromptSubmit hooks array:"
    echo '  {"type": "command", "command": "~/.claude/hooks/autoresearch-context.sh"}'
    echo ""
    echo "Example (add to existing UserPromptSubmit hooks):"
    echo '  "UserPromptSubmit": [{"hooks": [...existing hooks..., {"type": "command", "command": "~/.claude/hooks/autoresearch-context.sh"}]}]'
  fi
else
  echo ""
  echo "Create $SETTINGS with:"
  echo '{'
  echo '  "hooks": {'
  echo '    "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "~/.claude/hooks/autoresearch-context.sh"}]}]'
  echo '  }'
  echo '}'
fi

echo ""
echo "Done! Run /autoresearch in Claude Code to start."
