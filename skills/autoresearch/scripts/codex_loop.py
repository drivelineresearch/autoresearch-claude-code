#!/usr/bin/env python3
"""Bounded Codex CLI supervisor for an initialized autoresearch workspace."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
from ar_state import StateError, parse_state  # noqa: E402


class LoopError(Exception):
    pass


class LoopInterrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


def interrupt(signum, _frame):
    raise LoopInterrupted(signum)


def state_snapshot(root):
    path = root / "autoresearch.jsonl"
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise LoopError("state must be a regular file, without a symlink")
        contents = source.read()
    return parse_state(contents.decode("utf-8")), contents


def is_paused(root):
    # Even a dangling symlink or an unexpected directory means stop.
    return os.path.lexists(root / ".autoresearch-off")


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if result.returncode:
        raise LoopError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def preflight(root):
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise LoopError("--workspace must be the repository/worktree root")
    branch = git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    if branch in {"main", "master"}:
        raise LoopError("create an experiment branch before running")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise LoopError("workspace has uncommitted files; preserve them before starting")
    for name in ("autoresearch.md", "autoresearch.sh"):
        if not (root / name).is_file():
            raise LoopError(f"initialize the session first: missing {name}")
    return branch


def git_metadata_roots(root):
    """Grant Git's metadata directories, including a linked worktree's common dir."""
    root = Path(root).resolve()
    directories = []
    for options in (("--absolute-git-dir",), ("--path-format=absolute", "--git-common-dir")):
        path = Path(git(root, "rev-parse", *options))
        if not path.is_absolute() or not path.is_dir():
            raise LoopError("Git metadata must resolve to an existing absolute directory")
        path = path.resolve()
        if root.is_relative_to(path):
            raise LoopError("refusing to grant Git metadata access to the workspace or its ancestors")
        if path not in directories:
            directories.append(path)
    return directories


def protected_hashes(root, paths):
    hashes = {}
    for name in paths:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise LoopError(f"protected file must be inside the workspace, without a symlink: {name}")
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            hashes[name] = None
            continue
        with os.fdopen(fd, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise LoopError(f"protected path must be a regular file: {name}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            hashes[name] = digest.hexdigest()
    return hashes


def pause(root, reason):
    path = root / ".autoresearch-off"
    # Never follow a symlink for a control-file write.
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        pass
    except OSError as exc:
        print(f"[autoresearch] {reason}; unable to create pause file: {exc}", file=sys.stderr)
        return
    else:
        os.close(fd)
    print(f"[autoresearch] {reason}; paused (.autoresearch-off)", file=sys.stderr)


def terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # Still reap the direct child below, even if its group disappeared.
    # The CLI can exit before descendants, including children ignoring SIGTERM.
    # Keep ownership of the process group until it disappears or grace expires.
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        except PermissionError:
            # Darwin can report EPERM for a group containing only zombies if a
            # child exits between poll() and this probe. Retry after reaping;
            # a real permission failure still reaches SIGKILL at the deadline.
            pass
        time.sleep(0.05)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired as exc:
        raise LoopError("Codex process did not exit after SIGKILL; inspect its process group") from exc


def run_turn(command, prompt, root, log_dir, timeout):
    with (log_dir / "events.jsonl").open("wb") as stdout, (log_dir / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=root, stdin=subprocess.PIPE,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            deadline = time.monotonic() + timeout
            pending_input = prompt.encode()
            while True:
                if is_paused(root):
                    raise LoopError("pause requested during a turn; inspect partial work before resuming")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LoopError("turn/time budget expired; inspect partial work before resuming")
                try:
                    process.communicate(input=pending_input, timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pending_input = None
            if process.returncode:
                raise LoopError(f"codex exited {process.returncode}; inspect {log_dir}")
        finally:
            # Also stop background descendants after a successful CLI exit.
            try:
                terminate(process)
            finally:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path.cwd())
    p.add_argument("--max-turns", type=int, default=20, help="maximum Codex invocations (default: 20)")
    p.add_argument("--max-seconds", type=positive, default=3600, help="supervisor wall time (default: 3600)")
    p.add_argument("--turn-timeout", type=positive, default=900, help="seconds per Codex invocation (default: 900)")
    p.add_argument("--model", help="optional model override; otherwise use Codex configuration")
    p.add_argument("--protect", action="append", default=[], metavar="PATH", help="additional locked scorer/data/split file (repeatable)")
    p.add_argument("--status", action="store_true", help="read state and exit without writes or Codex")
    p.add_argument("--dry-run", action="store_true", help="validate and print command without writes or Codex")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_turns <= 0:
        parser().error("--max-turns must be greater than zero")
    root = args.workspace.resolve()
    lock = None
    running = False
    previous_sigterm = signal.signal(signal.SIGTERM, interrupt)
    try:
        state, history = state_snapshot(root)
        if args.status:
            print(json.dumps({**state, "paused": is_paused(root)}, indent=2, allow_nan=False))
            return 0
        branch = preflight(root)
        metadata_roots = git_metadata_roots(root)
        paths = list(dict.fromkeys(["autoresearch.sh", "checks.sh", *args.protect]))
        for name in args.protect:
            if not (root / name).is_file():
                raise LoopError(f"explicit --protect path must be an existing regular file: {name}")
        hashes = protected_hashes(root, paths)
        config = state["config"]
        if is_paused(root):
            raise LoopError("session is paused; review its budget and remove .autoresearch-off to resume")
        if state["budget_reached"]:
            raise LoopError(state["budget_reason"])
        command = ["codex", "exec", "--cd", str(root), "--sandbox", "workspace-write",
                   "-c", 'approval_policy="never"', "--ephemeral", "--json"]
        # workspace-write protects .git by default. Keeps need the index, objects,
        # and refs writable; never widen this grant to a parent checkout or home.
        for path in metadata_roots:
            command += ["--add-dir", str(path)]
        if args.model:
            command += ["--model", args.model]
        if args.dry_run:
            print(json.dumps({"command": [*command, "--output-last-message", "<turn-log>/final.md", "-"],
                              "branch": branch, "state": state, "protected": paths}, indent=2))
            return 0
        lock = os.fdopen(os.open(root / ".autoresearch-codex.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "a")
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise LoopError("supervisor lock must be a regular file")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LoopError("another Codex supervisor owns this workspace") from None
        running = True
        # State may have advanced while acquiring ownership.
        if state_snapshot(root)[1] != history:
            raise LoopError("state changed during startup; review it before retrying")
        deadline = time.monotonic() + args.max_seconds
        logs = root / "experiments" / "codex"
        if logs.is_symlink() or (root / "experiments").is_symlink():
            raise LoopError("experiment log directories must not be symlinks")
        logs.mkdir(parents=True, exist_ok=True)
        for _ in range(args.max_turns):
            before, before_history = state_snapshot(root)
            if before_history != history:
                raise LoopError("state changed outside the supervisor turn; review its history")
            if before["config"] != config or before["segment"] != state["segment"]:
                raise LoopError("the segment/config changed; do not reset budgets during a run")
            if before["budget_reached"]:
                pause(root, before["budget_reason"])
                return 0
            if is_paused(root):
                return 0
            if preflight(root) != branch:
                raise LoopError("experiment branch changed")
            if git_metadata_roots(root) != metadata_roots:
                raise LoopError("Git metadata directories changed")
            if protected_hashes(root, paths) != hashes:
                raise LoopError("locked harness changed; review and re-baseline")
            remaining = deadline - time.monotonic()
            if config.get("maxSeconds") is not None:
                remaining = min(remaining, config["startedAt"] + config["maxSeconds"] - time.time())
            if remaining <= 0:
                pause(root, "wall-time budget reached")
                return 0
            if logs.is_symlink() or (root / "experiments").is_symlink():
                raise LoopError("experiment log directories must not be symlinks")
            log_dir = Path(tempfile.mkdtemp(prefix=f"run-{before['next_run']}-", dir=logs))
            prompt = (
                f"Use the autoresearch skill at {SKILL / 'SKILL.md'}. "
                "This is one bounded supervisor turn. Read autoresearch.md, autoresearch.jsonl, "
                "and experiments/worklog.md. Complete exactly ONE experiment: change only scoped files, "
                "run the locked benchmark and correctness checks, keep or restore only your experiment's "
                "changes, append exactly one valid result with ar-log.sh, update the dashboard/worklog, "
                "then end this turn. Do not start a supervisor or run a second experiment. "
                "Do not edit the config, reset budgets, switch branches, alter gitignore, or edit the "
                "runner/skill. Do not install software, contact people, push, deploy, or acquire compute "
                "unless explicitly authorized in the session. "
                f"Locked files: {json.dumps(paths)}. Expected result run: {before['next_run']}, "
                f"segment: {before['segment']}. If blocked, explain and end without inventing a result."
            )
            print(f"[autoresearch] starting run {before['next_run']}; logs: {log_dir}", flush=True)
            run_turn([*command, "--output-last-message", str(log_dir / "final.md"), "-"],
                     prompt, root, log_dir, min(args.turn_timeout, remaining))
            after, after_history = state_snapshot(root)
            if not after_history.startswith(before_history.rstrip(b"\n") + b"\n"):
                raise LoopError("Codex rewrote existing state history; recorded results require review")
            if after["config"] != config or after["segment"] != state["segment"]:
                raise LoopError("Codex changed the segment/config")
            if after["total_runs"] != before["total_runs"] + 1:
                raise LoopError("expected exactly one new result; refusing blind retries")
            if protected_hashes(root, paths) != hashes:
                raise LoopError("locked harness changed; recorded result requires review")
            if preflight(root) != branch:
                raise LoopError("experiment branch changed")
            if git_metadata_roots(root) != metadata_roots:
                raise LoopError("Git metadata directories changed")
            history = after_history
            if after["budget_reached"]:
                pause(root, after["budget_reason"])
                return 0
        pause(root, "maximum Codex turns reached")
        return 0
    except (LoopError, StateError, OSError, ValueError) as exc:
        if running:
            pause(root, str(exc))
        else:
            print(f"[autoresearch] {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, LoopInterrupted) as exc:
        if running:
            pause(root, "interrupted; inspect partial work before resuming")
        return 128 + (exc.signum if isinstance(exc, LoopInterrupted) else signal.SIGINT)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    sys.exit(main())
