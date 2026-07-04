#!/usr/bin/env bash
# Autoresearch Stop hook — the actual loop engine.
#
# UserPromptSubmit can only nudge when the *user* types; during autonomous
# operation there are no user turns, so nothing stops the agent from ending its
# turn. This Stop hook is what makes "LOOP FOREVER" mechanical: on every
# turn-end it vetoes the stop with {"decision":"block","reason":...} (stdout,
# exit 0), feeding `reason` back as the agent's next instruction.
#
# Why JSON-on-stdout and not `exit 2`: exit-code-2 Stop continuation is broken
# for plugin-installed hooks (anthropics/claude-code#10412). The JSON form works.
#
# Safety valves (so it isn't a literal infinite trap):
#   - `.autoresearch-off` sentinel (written by `/autoresearch off`) → allow stop
#   - budget fields in the JSONL config header: maxRuns / maxSeconds / targetMetric
#   - the user pressing Esc always interrupts, regardless of this hook
#
# stdout MUST contain only the JSON object (or nothing). All diagnostics → stderr.
set -uo pipefail

input=$(cat)

# Resolve cwd from the hook payload (fall back to $PWD).
cwd=$(printf '%s' "$input" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("cwd",""))' 2>/dev/null)
[ -n "$cwd" ] && cd "$cwd" 2>/dev/null || true

# Not an autoresearch session, or explicitly paused → allow the stop.
[ -f "autoresearch.md" ] || exit 0
[ -f ".autoresearch-off" ] && exit 0
[ -f "autoresearch.jsonl" ] || exit 0

# Ask Python for a decision: "stop <reason>" to allow termination, or "" to continue.
decision=$(python3 - "$input" <<'PY'
import json, sys, time

def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None

try:
    lines = [l for l in open("autoresearch.jsonl") if l.strip()]
except OSError:
    print("")  # can't read state → don't trap the agent
    sys.exit()

cfg = {}
all_results = []
seg_index = -1
for l in lines:
    try: obj = json.loads(l)
    except json.JSONDecodeError: continue
    if obj.get("type") == "config":
        cfg = obj  # last config header wins (current segment)
        seg_index += 1
    elif "run" in obj:
        all_results.append(obj)

# Budget/target apply to the CURRENT segment only — a re-init starts a fresh
# optimization target, so prior segments' runs must not instantly cap it.
results = [r for r in all_results if r.get("segment", 0) == seg_index] if seg_index >= 0 else all_results

max_runs = num(cfg.get("maxRuns"))
max_secs = num(cfg.get("maxSeconds"))
target   = num(cfg.get("targetMetric"))
started  = num(cfg.get("startedAt"))
direction = cfg.get("bestDirection", "higher")

n = len(results)
if max_runs is not None and n >= max_runs:
    print(f"stop budget reached: {n} runs >= maxRuns {int(max_runs)}"); sys.exit()

if max_secs is not None and started is not None and (time.time() - started) >= max_secs:
    print(f"stop budget reached: elapsed >= maxSeconds {int(max_secs)}"); sys.exit()

if target is not None and results:
    vals = [num(r.get("metric")) for r in results if r.get("status") == "keep"]
    vals = [v for v in vals if v is not None]
    if vals:
        best = max(vals) if direction == "higher" else min(vals)
        hit = best >= target if direction == "higher" else best <= target
        if hit:
            print(f"stop target reached: best {best:g} vs targetMetric {target:g}"); sys.exit()

print("")  # keep looping
PY
)

if [ -n "$decision" ]; then
  # Budget/target hit: pause the loop so the next turn ends cleanly, and allow the stop.
  touch .autoresearch-off
  echo "[autoresearch] $decision — pausing loop (.autoresearch-off written). Resume with /autoresearch." >&2
  exit 0
fi

# Continue the loop. reason becomes the agent's next instruction.
cat <<'JSON'
{"decision":"block","reason":"Autoresearch is active and the run budget is not exhausted. Do NOT stop. Run the next experiment now: read autoresearch.md (objective, constraints, What's Been Tried) and experiments/worklog.md, pick the single most promising untried change (one atomic change per experiment), edit only files in scope, run ./autoresearch.sh, then log the result (keep/discard/crash), update the dashboard and worklog, and continue. If you are out of ideas, consult autoresearch.ideas.md or generate a new diverse draft."}
JSON
exit 0
