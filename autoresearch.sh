#!/usr/bin/env bash
set -euo pipefail

# Quick syntax check
python3 -c "import py_compile; py_compile.compile('train.py', doraise=True)" 2>&1 || { echo "Syntax error in train.py"; exit 1; }

# Run training
.venv/bin/python train.py
