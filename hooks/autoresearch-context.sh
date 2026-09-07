#!/usr/bin/env bash
# Resolve manual-install symlinks and preserve stdin for the event payload.
set -euo pipefail
exec python3 -c 'import pathlib,runpy,sys; p=pathlib.Path(sys.argv[1]).resolve().parent.parent/"skills/autoresearch/scripts/ar_state.py"; sys.argv=[str(p),"hook","context"]; runpy.run_path(str(p),run_name="__main__")' "$0"
