#!/usr/bin/env bash
# Autoresearch PreCompact hook — flush durable state before context is lost.
#
# Compaction is exactly when the agent forgets what it was doing. This hook does
# the durable write itself (it does not rely on the model): it snapshots the
# JSONL state and drops a timestamped checkpoint into experiments/worklog.md so a
# post-compaction agent (rehydrated by the SessionStart hook) can continue.
#
# Input on stdin: {"session_id","transcript_path","cwd","hook_event_name":"PreCompact","trigger":"manual"|"auto"}
set -uo pipefail

input=$(cat)
cwd=$(printf '%s' "$input" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("cwd",""))' 2>/dev/null)
[ -n "$cwd" ] && cd "$cwd" 2>/dev/null || true

[ -f "autoresearch.md" ] || exit 0
[ -f ".autoresearch-off" ] && exit 0

ts=$(date '+%Y-%m-%d %H:%M:%S')

# Snapshot the state file so a corrupted/lost JSONL can be recovered.
if [ -f "autoresearch.jsonl" ]; then
  mkdir -p experiments
  cp autoresearch.jsonl "experiments/autoresearch.jsonl.precompact.bak"

  runs=$(grep -c '"run"' autoresearch.jsonl 2>/dev/null || echo 0)
  kept=$(grep -c '"status":"keep"' autoresearch.jsonl 2>/dev/null || echo 0)
  {
    echo ""
    echo "### ⟳ Compaction checkpoint — $ts"
    echo "- Context is about to compact. State snapshot: experiments/autoresearch.jsonl.precompact.bak"
    echo "- Runs so far: $runs | kept: $kept"
    echo "- On resume: read autoresearch.md, autoresearch.jsonl, and this worklog, then CONTINUE the loop (do not restart)."
  } >> experiments/worklog.md
fi

# stdout is surfaced to the user; keep it to a single clean line.
echo "[autoresearch] flushed state to worklog before compaction ($ts)."
exit 0
