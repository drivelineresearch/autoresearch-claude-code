#!/usr/bin/env bash
# Autoresearch SessionStart hook — rehydrate an active loop after resume/compaction.
#
# Without this, an agent that starts fresh (or wakes from compaction) in a
# directory with a live autoresearch session asks "what was I doing?" instead of
# continuing. Plain stdout from a SessionStart hook is injected as context, so we
# just print the objective + current best + recent worklog and tell it to resume.
#
# Matcher: resume | compact | startup (see hooks.json).
# Input on stdin: {"session_id","cwd","hook_event_name":"SessionStart","source":"startup"|"resume"|"compact"}
set -uo pipefail

input=$(cat)
cwd=$(printf '%s' "$input" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("cwd",""))' 2>/dev/null)
[ -n "$cwd" ] && cd "$cwd" 2>/dev/null || true

# Only speak up for a real, non-paused session.
[ -f "autoresearch.md" ] || exit 0
[ -f ".autoresearch-off" ] && exit 0

echo "## Autoresearch session detected in this directory"
echo "An autonomous experiment loop is active here. Resume it — do not start over."
echo ""

if [ -f "autoresearch.jsonl" ]; then
  runs=$(grep -c '"run"' autoresearch.jsonl 2>/dev/null || echo 0)
  kept=$(grep -c '"status":"keep"' autoresearch.jsonl 2>/dev/null || echo 0)
  echo "State: $runs runs logged, $kept kept. Full protocol + best result are in autoresearch.jsonl."
fi

# Objective (first ~15 lines of the session doc) and the tail of the worklog narrative.
echo ""
echo "### Objective (autoresearch.md)"
sed -n '1,15p' autoresearch.md 2>/dev/null

if [ -f "experiments/worklog.md" ]; then
  echo ""
  echo "### Recent worklog"
  tail -n 25 experiments/worklog.md 2>/dev/null
fi

echo ""
echo "→ Read autoresearch.md fully, reconstruct state from autoresearch.jsonl, and continue the loop (run the next experiment). The Stop hook will keep you going until the run budget is hit or /autoresearch off is set."
exit 0
