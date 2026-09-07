#!/usr/bin/env bash
# ar-log.sh [--op draft|improve|debug] [--parent null|RUN]
#   <run|auto> <commit> <metric> <status> <segment|auto> <description> [key=number ...]
# AR_JSONL or --path selects state; Python 3 is the only dependency.
set -euo pipefail
exec python3 -c 'import pathlib,runpy,sys; p=pathlib.Path(sys.argv[1]).resolve().with_name("ar_state.py"); sys.argv=[str(p),"append",*sys.argv[2:]]; runpy.run_path(str(p),run_name="__main__")' "$0" "$@"
