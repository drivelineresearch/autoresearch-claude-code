#!/usr/bin/env python3
"""Validated experiment state shared by Claude hooks and the Codex runner.

Only the standard library is needed. Writers must use init/append to participate
in the advisory lock and atomic replacement protocol.
"""
import argparse
import contextlib
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time


class StateError(ValueError):
    """State is missing, malformed, or violates the experiment protocol."""


def _number(value, label, *, minimum=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateError(f"{label} must be a finite {'integer' if integer else 'number'}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (integer and not isinstance(value, int)):
        raise StateError(f"{label} must be a finite {'integer' if integer else 'number'}")
    if minimum is not None and value < minimum:
        raise StateError(f"{label} must be >= {minimum}")
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise StateError(f"non-finite JSON number: {value}")


def _json(text):
    try:
        result = json.loads(text, object_pairs_hook=_object, parse_constant=_reject_constant)
        json.dumps(result, allow_nan=False)  # catches overflowing exponent notation
        return result
    except (ValueError, OverflowError, RecursionError) as exc:
        raise StateError(f"invalid JSON: {exc}") from exc


def _config(row):
    cfg = dict(row)
    for key in ("name", "metricName", "metricUnit"):
        if not isinstance(cfg.get(key), str):
            raise StateError(f"config.{key} must be a string")
    if not cfg["name"].strip() or not cfg["metricName"].strip():
        raise StateError("config.name and config.metricName must not be empty")
    if cfg.get("bestDirection") not in ("lower", "higher"):
        raise StateError("config.bestDirection must be lower or higher")
    # Legacy four-field headers get a bounded default, not unlimited runs.
    cfg.setdefault("noiseFloor", 0)
    cfg.setdefault("maxRuns", 200)
    cfg.setdefault("maxSeconds", None)
    cfg.setdefault("targetMetric", None)
    _number(cfg["noiseFloor"], "config.noiseFloor", minimum=0)
    if cfg["maxRuns"] is not None:
        _number(cfg["maxRuns"], "config.maxRuns", minimum=1, integer=True)
    if cfg["maxSeconds"] is not None:
        _number(cfg["maxSeconds"], "config.maxSeconds", minimum=0)
        _number(cfg.get("startedAt"), "config.startedAt", minimum=0)
    elif "startedAt" in cfg:
        _number(cfg["startedAt"], "config.startedAt", minimum=0)
    if cfg["targetMetric"] is not None:
        _number(cfg["targetMetric"], "config.targetMetric")
    return cfg


def _result(row, expected_run, segment, previous):
    _number(row.get("run"), "result.run", minimum=1, integer=True)
    if row["run"] != expected_run:
        raise StateError(f"expected run {expected_run}, got {row['run']}")
    _number(row.get("segment"), "result.segment", minimum=0, integer=True)
    if row["segment"] != segment:
        raise StateError(f"expected segment {segment}, got {row['segment']}")
    commit = row.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{7,64}", commit):
        raise StateError("result.commit must be a 7-64 character hexadecimal git hash")
    _number(row.get("metric"), "result.metric")
    _number(row.get("timestamp"), "result.timestamp", minimum=0)
    if row.get("status") not in ("keep", "discard", "crash", "checks_failed"):
        raise StateError("invalid result.status")
    if not isinstance(row.get("description"), str):
        raise StateError("result.description must be a string")
    if not isinstance(row.get("metrics"), dict):
        raise StateError("result.metrics must be an object of finite numbers")
    for key, value in row["metrics"].items():
        if not key:
            raise StateError("secondary metric names must not be empty")
        _number(value, f"result.metrics.{key}")
    if "op" in row and row["op"] not in ("draft", "improve", "debug"):
        raise StateError("result.op must be draft, improve, or debug")
    if row.get("parent") is not None:
        parent = _number(row["parent"], "result.parent", minimum=1, integer=True)
        if parent >= expected_run or previous[parent - 1]["segment"] != segment:
            raise StateError("result.parent must refer to an earlier run in this segment")


def _parse(text, now=None):
    cfg = None
    segment = -1
    results = []
    current = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = _json(line)
            if not isinstance(row, dict):
                raise StateError("each line must be an object")
            if row.get("type") == "config":
                cfg = _config(row)
                segment += 1
                current = []
            else:
                if cfg is None:
                    raise StateError("first record must be a config header")
                if "type" in row:
                    raise StateError("unknown record type")
                _result(row, len(results) + 1, segment, results)
                results.append(row)
                current.append(row)
        except StateError as exc:
            raise StateError(f"line {line_number}: {exc}") from exc
    if cfg is None:
        raise StateError("state has no config header")
    kept = [row for row in current if row["status"] == "keep"]
    best = None
    if kept:
        choose = min if cfg["bestDirection"] == "lower" else max
        best = choose(kept, key=lambda row: row["metric"])
    reason = None
    now = time.time() if now is None else _number(now, "now", minimum=0)
    if cfg["maxRuns"] is not None and len(current) >= cfg["maxRuns"]:
        reason = f"run budget reached: {len(current)} >= {cfg['maxRuns']}"
    elif cfg["maxSeconds"] is not None and now - cfg["startedAt"] >= cfg["maxSeconds"]:
        reason = f"time budget reached: elapsed >= {cfg['maxSeconds']:g} seconds"
    elif cfg["targetMetric"] is not None and best is not None:
        target = cfg["targetMetric"]
        hit = best["metric"] <= target if cfg["bestDirection"] == "lower" else best["metric"] >= target
        if hit:
            reason = f"target reached: best {best['metric']:g} vs {target:g}"
    return {
        "valid": True, "config": cfg, "segment": segment,
        "total_runs": len(results), "segment_runs": len(current),
        "next_run": len(results) + 1,
        "total_kept": sum(row["status"] == "keep" for row in results),
        "segment_kept": len(kept), "best": best,
        "budget_reached": reason is not None, "budget_reason": reason,
    }


def _read_text(path):
    try:
        if not Path(path).is_file():
            raise StateError(f"not a regular readable file: {path}")
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise StateError(f"cannot read {path}: {exc}") from exc


def read_state(path="autoresearch.jsonl", now=None):
    """Return validated counts, current config/best result, and budget status."""
    return parse_state(_read_text(path), now=now)


def parse_state(text, now=None):
    """Validate an already-read UTF-8 text snapshot without rereading the file."""
    return _parse(text, now=now)


def _check_destination(path):
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise StateError(f"refusing to write non-regular or symlink state: {path}")


@contextlib.contextmanager
def _write_lock(path):
    path = Path(path).absolute()
    lock = path.with_name(f".{path.name}.lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise StateError(f"lock is not a regular file: {lock}")
        deadline = time.monotonic() + 5
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise StateError(f"timed out waiting for state writer lock: {lock}")
                time.sleep(0.05)
        _check_destination(path)
        yield path
    finally:
        os.close(fd)


def _atomic_write(path, text):
    _check_destination(path)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            os.fchmod(output.fileno(), mode)
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        _check_destination(path)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_text(text, row):
    return text.rstrip("\n") + "\n" + json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"


def init_state(path, config, new_segment=False):
    """Create state or explicitly append a fresh segment under the writer lock."""
    cfg = _config({**config, "type": "config"})
    with _write_lock(path) as destination:
        if destination.exists():
            if not new_segment:
                raise StateError("state already exists; use --new-segment to append a config")
            old = _read_text(destination)
            _parse(old)
            updated = _append_text(old, cfg)
        else:
            if new_segment:
                raise StateError("cannot append a segment to missing state")
            updated = json.dumps(cfg, allow_nan=False, separators=(",", ":")) + "\n"
        summary = _parse(updated)
        _atomic_write(destination, updated)
    return summary


def append_result(path, *, run, commit, metric, status, segment, description,
                  metrics=None, op=None, parent=None):
    """Append one validated result; auto allocates run/segment while locked."""
    with _write_lock(path) as destination:
        old = _read_text(destination)
        state = _parse(old)
        row = {
            "run": state["next_run"] if run == "auto" else run,
            "commit": commit, "metric": metric, "status": status,
            "segment": state["segment"] if segment == "auto" else segment,
            "description": description, "metrics": {} if metrics is None else metrics,
            "timestamp": int(time.time()), "parent": parent,
        }
        if op is not None:
            row["op"] = op
        updated = _append_text(old, row)
        _parse(updated)
        _atomic_write(destination, updated)
    return row


def _diagnostic(message):
    print(f"[autoresearch] {message}", file=sys.stderr)


def _hook_directory():
    payload = _json(sys.stdin.read())
    if not isinstance(payload, dict):
        raise StateError("hook input must be an object")
    cwd = payload.get("cwd")
    if not isinstance(cwd, str) or not cwd or not Path(cwd).is_absolute():
        raise StateError("hook input needs an absolute cwd; ignoring hook")
    path = Path(cwd).resolve(strict=True)
    if not path.is_dir():
        raise StateError("hook cwd is not a directory; ignoring hook")
    return path


def run_hook(event):
    """Hooks fail open for stopping, and never substitute the process cwd."""
    try:
        cwd = _hook_directory()
        if not (cwd / "autoresearch.md").is_file() or os.path.lexists(cwd / ".autoresearch-off"):
            return
        inspect = cwd / ".autoresearch-inspect"
        if event == "stop" and os.path.lexists(inspect):
            inspect.unlink()
            return
        path = cwd / "autoresearch.jsonl"
        if not path.exists():
            return
        if event == "precompact":
            with _write_lock(path) as destination:
                contents = _read_text(destination)
                state = _parse(contents)
                directory = cwd / "experiments"
                directory.mkdir(exist_ok=True)
                if directory.is_symlink() or not directory.is_dir():
                    raise StateError("refusing checkpoint through a symlink/non-directory experiments path")
                _atomic_write(directory / "autoresearch.jsonl.precompact.bak", contents)
                ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
                worklog = directory / "worklog.md"
                _check_destination(worklog)
                with worklog.open("a", encoding="utf-8") as output:
                    output.write(
                        f"\n### Compaction checkpoint — {ts}\n"
                        "- State snapshot: experiments/autoresearch.jsonl.precompact.bak\n"
                        f"- Runs so far: {state['total_runs']} | kept: {state['total_kept']}\n"
                        "- On resume: read autoresearch.md, autoresearch.jsonl, and this worklog; honor budgets and user stop requests.\n"
                    )
            _diagnostic("saved state and worklog checkpoint before compaction")
            return
        state = read_state(path)
        if state["budget_reached"]:
            if event == "stop":
                fd = os.open(cwd / ".autoresearch-off", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(fd)
            _diagnostic(f"{state['budget_reason']}; loop paused")
            return
        if event == "stop":
            print(json.dumps({
                "decision": "block",
                "reason": "Autoresearch is active within its budget. Run one complete next experiment: read autoresearch.md and experiments/worklog.md, edit only authorized files, run the locked harness, log keep/discard/crash/checks_failed, and update the dashboard and worklog. Honor user requests to stop or pause by creating .autoresearch-off; do not start another experiment after such a request. If blocked or unable to make progress, create .autoresearch-off and explain why.",
            }))
        elif event == "sessionstart":
            print("## Autoresearch session detected")
            print(f"State: {state['total_runs']} runs logged, {state['total_kept']} kept; current segment {state['segment']}: {state['segment_runs']} runs.")
            if state["best"] is not None:
                print(f"Current best {state['config']['metricName']}: {state['best']['metric']} (run {state['best']['run']}).")
            print("\n### Objective (autoresearch.md)")
            print("\n".join(_read_text(cwd / "autoresearch.md").splitlines()[:15]))
            worklog = cwd / "experiments/worklog.md"
            if worklog.is_file():
                print("\n### Recent worklog")
                print("\n".join(_read_text(worklog).splitlines()[-25:]))
            print("\nRead the full session documents and resume within the configured budgets and the user's current instructions.")
        elif event == "context":
            print("## Autoresearch Mode (ACTIVE)\nRead autoresearch.md and autoresearch.jsonl for the objective, authorized scope, and remaining budget. Complete one experiment at a time, log results, and update the worklog. User requests to stop, pause, inspect, or change scope take priority over continuation; do not treat them as mere experiment suggestions.")
    except (StateError, OSError, UnicodeError) as exc:
        _diagnostic(f"{exc}; automatic continuation disabled for this hook")


def _cli_number(value):
    return _number(_json(value), "argument")


class _ArgumentParser(argparse.ArgumentParser):
    def _parse_optional(self, argument):
        # argparse otherwise mistakes valid negative scientific metrics for flags.
        if re.fullmatch(r"-\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", argument):
            return None
        return super()._parse_optional(argument)


def main(argv=None):
    parser = _ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="validate state and print current budget/status")
    status.add_argument("--path", default=os.environ.get("AR_JSONL", "autoresearch.jsonl"))
    status.add_argument("--now", type=float)
    init = commands.add_parser("init", help="create state or append a new segment without overwriting history")
    init.add_argument("--path", default=os.environ.get("AR_JSONL", "autoresearch.jsonl"))
    init.add_argument("--name", required=True)
    init.add_argument("--metric-name", required=True)
    init.add_argument("--metric-unit", default="")
    init.add_argument("--direction", choices=("lower", "higher"), required=True)
    init.add_argument("--noise-floor", default="0")
    init.add_argument("--max-runs", default="200")
    init.add_argument("--max-seconds")
    init.add_argument("--target-metric")
    init.add_argument("--new-segment", action="store_true")
    append = commands.add_parser("append", help="log with explicit or auto run/segment indices")
    append.add_argument("--path", default=os.environ.get("AR_JSONL", "autoresearch.jsonl"))
    append.add_argument("--op", choices=("draft", "improve", "debug"))
    append.add_argument("--parent", default="null")
    for name in ("run", "commit", "metric", "status", "segment", "description"):
        append.add_argument(name)
    append.add_argument("metrics", nargs="*")
    hook = commands.add_parser("hook", help="internal hook entrypoint; reads event payload from stdin")
    hook.add_argument("event", choices=("stop", "precompact", "sessionstart", "context"))
    args = parser.parse_args(argv)
    if args.command == "hook":
        run_hook(args.event)
        return 0
    try:
        if args.command == "status":
            result = read_state(args.path, now=args.now)
        elif args.command == "init":
            cfg = {
                "name": args.name, "metricName": args.metric_name,
                "metricUnit": args.metric_unit, "bestDirection": args.direction,
                "noiseFloor": _cli_number(args.noise_floor),
                "maxRuns": None if args.max_runs.lower() in ("none", "null") else _cli_number(args.max_runs),
                "maxSeconds": _cli_number(args.max_seconds) if args.max_seconds is not None else None,
                "targetMetric": _cli_number(args.target_metric) if args.target_metric is not None else None,
                "startedAt": int(time.time()),
            }
            result = init_state(args.path, cfg, args.new_segment)
        else:
            metrics = {}
            for pair in args.metrics:
                key, separator, value = pair.partition("=")
                if not separator or not key or key in metrics:
                    raise StateError("secondary metrics must be unique nonempty key=number pairs")
                metrics[key] = _cli_number(value)
            result = append_result(
                args.path, run="auto" if args.run == "auto" else _cli_number(args.run),
                commit=args.commit, metric=_cli_number(args.metric), status=args.status,
                segment="auto" if args.segment == "auto" else _cli_number(args.segment),
                description=args.description, metrics=metrics, op=args.op,
                parent=None if args.parent == "null" else _cli_number(args.parent),
            )
        print(json.dumps(result, allow_nan=False, separators=(",", ":")))
        return 0
    except (StateError, OSError, ValueError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, allow_nan=False))
        _diagnostic(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
