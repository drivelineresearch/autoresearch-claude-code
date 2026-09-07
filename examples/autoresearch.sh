#!/usr/bin/env bash
set -euo pipefail

# Resolve paths from this script so `./examples/autoresearch.sh` also works from
# the repository root. Run dependencies and syntax checks with the same runtime.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR"
export AR_SEED="${1:-${AR_SEED:-42}}"
if [[ $# -gt 1 || ! "$AR_SEED" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 [nonnegative-integer-seed]" >&2
    exit 2
fi
if command -v uv >/dev/null 2>&1; then
    runtime=(uv run --project "$SCRIPT_DIR" python)
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    runtime=("$SCRIPT_DIR/.venv/bin/python")
else
    echo "Install uv (https://docs.astral.sh/uv/) and run: cd '$SCRIPT_DIR' && uv sync" >&2
    exit 1
fi
"${runtime[@]}" -m py_compile train.py models.py candidate.py
exec "${runtime[@]}" train.py
