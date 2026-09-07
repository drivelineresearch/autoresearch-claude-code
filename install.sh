#!/usr/bin/env bash
# Install shared autoresearch skills for Claude Code and/or Codex.
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
if ! command -v python3 >/dev/null 2>&1; then
  echo "autoresearch: python3 is required; no installation changes made." >&2
  exit 1
fi
exec python3 "$REPO_DIR/scripts/install.py" install "$@"
