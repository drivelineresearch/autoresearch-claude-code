#!/usr/bin/env bash
# ar-log.sh — append one result line to autoresearch.jsonl as valid JSON.
#
# Usage:
#   ar-log.sh <run> <commit> <metric> <status> <segment> <description> [key=value ...]
#
# The trailing key=value pairs become the "metrics" object (secondary metrics).
# Values that parse as numbers are emitted as JSON numbers; otherwise as strings.
# This exists so the agent never hand-builds JSON with echo (which breaks the
# moment a description contains a quote or apostrophe).
set -euo pipefail

if [ "$#" -lt 6 ]; then
  echo "usage: ar-log.sh <run> <commit> <metric> <status> <segment> <description> [k=v ...]" >&2
  exit 2
fi

run="$1"; commit="$2"; metric="$3"; status="$4"; segment="$5"; description="$6"
shift 6

out="${AR_JSONL:-autoresearch.jsonl}"
ts="$(date +%s)"

if command -v jq >/dev/null 2>&1; then
  # Fold k=v lines into a JSON object, numeric-coercing where possible.
  # Note: `tonumber? // .value` keeps the string on parse failure — a plain
  # `try tonumber catch .` would bind `.` to jq's error message, not the value.
  metrics_obj=$(
    printf '%s\n' "$@" | jq -R -s -c '
      split("\n") | map(select(length > 0)) | map(
        (index("=")) as $i | {key: .[0:$i], value: .[$i+1:]}
        | .value |= (tonumber? // .)
      ) | from_entries'
  )
  [ -n "$metrics_obj" ] || metrics_obj='{}'
  jq -nc \
    --argjson run "$run" --arg commit "$commit" --argjson metric "$metric" \
    --arg status "$status" --argjson segment "$segment" --arg description "$description" \
    --argjson metrics "$metrics_obj" --argjson ts "$ts" \
    '{run:$run,commit:$commit,metric:$metric,metrics:$metrics,status:$status,description:$description,timestamp:$ts,segment:$segment}' \
    >> "$out"
else
  # jq absent: fall back to Python (stdlib json handles all escaping correctly).
  python3 - "$run" "$commit" "$metric" "$status" "$segment" "$description" "$ts" "$out" "$@" <<'PY'
import json, sys
run, commit, metric, status, segment, description, ts, out = sys.argv[1:9]
metrics = {}
for kv in sys.argv[9:]:
    k, _, v = kv.partition("=")
    try:
        metrics[k] = int(v)
    except ValueError:
        try:
            metrics[k] = float(v)
        except ValueError:
            metrics[k] = v
def num(x):
    try:
        return int(x)
    except ValueError:
        return float(x)
row = {
    "run": num(run), "commit": commit, "metric": num(metric),
    "metrics": metrics, "status": status, "description": description,
    "timestamp": int(ts), "segment": num(segment),
}
with open(out, "a") as f:
    f.write(json.dumps(row) + "\n")
PY
fi
