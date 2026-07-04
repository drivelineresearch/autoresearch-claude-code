#!/usr/bin/env bash
set -euo pipefail

# Optional SEED arg → AR_SEED env, so the loop can re-run with different seeds to
# measure the noise floor and confirm borderline wins. Defaults to train.py's own.
export AR_SEED="${1:-${AR_SEED:-42}}"

# Quick syntax checks (use same Python runtime as training to avoid version mismatch)
if command -v uv &>/dev/null; then
    uv run python -c "import py_compile; py_compile.compile('train.py', doraise=True)" 2>&1 || { echo "Syntax error in train.py"; exit 1; }
    uv run python -c "import py_compile; py_compile.compile('models.py', doraise=True)" 2>&1 || { echo "Syntax error in models.py"; exit 1; }
else
    python3 -c "import py_compile; py_compile.compile('train.py', doraise=True)" 2>&1 || { echo "Syntax error in train.py"; exit 1; }
    python3 -c "import py_compile; py_compile.compile('models.py', doraise=True)" 2>&1 || { echo "Syntax error in models.py"; exit 1; }
fi

# Run training (prefer uv, fall back to .venv/bin/python)
if command -v uv &>/dev/null; then
    uv run python train.py
elif [ -x .venv/bin/python ]; then
    echo "Warning: uv not found, using .venv/bin/python" >&2
    .venv/bin/python train.py
else
    echo "Error: neither uv nor .venv/bin/python found. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
