"""Exercise supervisor boundaries without calling a model or modifying user config."""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock

RUNNER = Path(__file__).resolve().parents[1] / "skills" / "autoresearch" / "scripts" / "codex_loop.py"

FAKE_CODEX = '''#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys, time
root = pathlib.Path.cwd()
mode = os.environ.get("AR_TEST_MODE", "ok")
(root / "invocations").open("a").write(json.dumps(sys.argv[1:]) + "\\n")
(root / ".autoresearch-test-parent.pid").write_text(str(os.getpid()))
prompt = sys.stdin.read()
if "exactly ONE experiment" not in prompt:
    sys.exit(9)
if mode == "error":
    sys.exit(7)
if mode == "none":
    sys.exit(0)
if mode == "sleep":
    time.sleep(30)
if mode == "pause":
    (root / ".autoresearch-off").touch()
    time.sleep(30)
if mode == "descendant_exit":
    code = "import os, pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path('.autoresearch-test-child.pid').write_text(str(os.getpid())); time.sleep(30)"
    subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while not (root / ".autoresearch-test-child.pid").exists():
        time.sleep(0.01)
path = root / "autoresearch.jsonl"
rows = [json.loads(line) for line in path.read_text().splitlines()]
if mode == "config":
    rows[0]["maxRuns"] = 999
if mode == "harness":
    (root / "autoresearch.sh").write_text("changed scorer")
if mode == "dirty":
    (root / "code.txt").write_text("unfinished experiment")
if mode == "history":
    rows[1]["description"] = "rewritten previous experiment"
for _ in range(2 if mode == "double" else 1):
    rows.append({"run": len(rows), "commit": "abc1234", "metric": 1.0,
                 "metrics": {}, "status": "discard", "segment": 0,
                 "description": "fake experiment", "timestamp": time.time()})
path.write_text("".join(json.dumps(row) + "\\n" for row in rows))
'''


class CodexLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ar loop test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        fake = self.bin / "codex"
        fake.write_text(FAKE_CODEX)
        fake.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}")
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        self.git("checkout", "-b", "experiment")
        (self.root / ".gitignore").write_text("bin/\nautoresearch.*\n.autoresearch*\nexperiments/\ninvocations\n")
        (self.root / "code.txt").write_text("original")
        self.git("add", ".gitignore", "code.txt")
        self.git("commit", "-qm", "baseline")
        (self.root / "autoresearch.md").write_text("Test session; code.txt in scope")
        (self.root / "autoresearch.sh").write_text("#!/bin/sh\necho METRIC score=1\n")
        self.config = {"type": "config", "name": "test", "metricName": "score", "metricUnit": "",
                       "bestDirection": "higher", "noiseFloor": 0, "maxRuns": 3,
                       "maxSeconds": None, "targetMetric": None, "startedAt": time.time()}
        self.write_state()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], check=True,
                              capture_output=True, text=True)

    def write_state(self):
        (self.root / "autoresearch.jsonl").write_text(json.dumps(self.config) + "\n")

    def run_loop(self, *args, mode="ok"):
        return subprocess.run([sys.executable, str(RUNNER), "--workspace", str(self.root), *args],
                              capture_output=True, text=True, timeout=12,
                              env=dict(self.env, AR_TEST_MODE=mode))

    def test_budget_and_safe_cli_flags(self):
        result = self.run_loop()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in (self.root / "invocations").read_text().splitlines()]
        self.assertEqual(len(calls), 3)
        self.assertIn("workspace-write", calls[0])
        self.assertIn('approval_policy="never"', calls[0])
        grants = [calls[0][i + 1] for i, arg in enumerate(calls[0]) if arg == "--add-dir"]
        self.assertEqual(grants, [str(self.root / ".git")])
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", calls[0])
        self.assertEqual(calls[0][-1], "-")
        self.assertTrue((self.root / ".autoresearch-off").exists())
        self.assertEqual((self.root / "code.txt").read_text(), "original")

    def test_linked_worktree_grants_only_its_git_metadata_and_common_directory(self):
        original = self.root
        linked_temp = tempfile.TemporaryDirectory(prefix="ar linked worktree ")
        self.addCleanup(linked_temp.cleanup)
        linked = Path(linked_temp.name) / "experiment"
        self.git("worktree", "add", "-q", "-b", "linked-experiment", str(linked))
        for name in ("autoresearch.md", "autoresearch.sh", "autoresearch.jsonl"):
            (linked / name).write_bytes((original / name).read_bytes())
        self.root = linked
        result = self.run_loop("--max-turns", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        call = json.loads((linked / "invocations").read_text())
        grants = [call[i + 1] for i, arg in enumerate(call) if arg == "--add-dir"]
        metadata = self.git("rev-parse", "--absolute-git-dir").stdout.strip()
        self.assertEqual(grants, [metadata, str(original / ".git")])
        self.assertNotIn(str(original), grants)
        self.assertNotIn(str(linked), grants)
        self.assertIn("workspace-write", call)
        self.assertIn('approval_policy="never"', call)

    def test_git_metadata_grants_reject_missing_directories_and_workspace_ancestors(self):
        spec = importlib.util.spec_from_file_location("codex_loop_git_test", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        for path in (self.root, self.root.parent, Path("/"), self.root / "absent"):
            with self.subTest(path=path), mock.patch.object(runner, "git", return_value=str(path)):
                with self.assertRaises(runner.LoopError):
                    runner.git_metadata_roots(self.root)

    def test_max_turns(self):
        result = self.run_loop("--max-turns", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len((self.root / "invocations").read_text().splitlines()), 1)

    def test_status_and_dry_run_have_no_side_effects(self):
        before = set(self.root.rglob("*"))
        for flag in ("--status", "--dry-run"):
            result = self.run_loop(flag)
            self.assertEqual(result.returncode, 0, result.stderr)
            json.loads(result.stdout)
        self.assertEqual(before, set(self.root.rglob("*")))
        (self.root / ".autoresearch-off").touch()
        self.assertTrue(json.loads(self.run_loop("--status").stdout)["paused"])

    def test_failures_pause_without_retry_or_cleanup(self):
        for mode in ("none", "double", "config", "harness", "dirty", "error"):
            with self.subTest(mode=mode):
                self.write_state()
                (self.root / "autoresearch.sh").write_text("scorer")
                (self.root / ".autoresearch-off").unlink(missing_ok=True)
                (self.root / "invocations").unlink(missing_ok=True)
                result = self.run_loop(mode=mode)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertTrue((self.root / ".autoresearch-off").exists())
                self.assertEqual(len((self.root / "invocations").read_text().splitlines()), 1)
                if mode == "dirty":
                    self.assertEqual((self.root / "code.txt").read_text(), "unfinished experiment")
                    (self.root / "code.txt").write_text("original")

    def test_timeout_and_external_pause_stop_active_child(self):
        for mode in ("sleep", "pause"):
            with self.subTest(mode=mode):
                (self.root / ".autoresearch-off").unlink(missing_ok=True)
                started = time.monotonic()
                result = self.run_loop("--turn-timeout", "0.3", mode=mode)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertLess(time.monotonic() - started, 6)
                self.assertTrue((self.root / ".autoresearch-off").exists())

    def test_dirty_workspace_paused_budget_and_invalid_state_do_not_launch(self):
        (self.root / "code.txt").write_text("user work")
        self.assertEqual(self.run_loop().returncode, 2)
        (self.root / "code.txt").write_text("original")
        (self.root / ".autoresearch-off").touch()
        self.assertEqual(self.run_loop().returncode, 2)
        (self.root / ".autoresearch-off").unlink()
        self.config["maxRuns"] = 0
        self.write_state()
        self.assertEqual(self.run_loop().returncode, 2)
        (self.root / "autoresearch.jsonl").write_text("broken JSON")
        self.assertEqual(self.run_loop().returncode, 2)
        self.assertFalse((self.root / "invocations").exists())

    def test_lock_contention_does_not_pause_owner(self):
        with (self.root / ".autoresearch-codex.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_loop()
        self.assertEqual(result.returncode, 2)
        self.assertIn("another Codex supervisor", result.stderr)
        self.assertFalse((self.root / ".autoresearch-off").exists())
        self.assertFalse((self.root / "invocations").exists())

    def test_protected_symlink_is_rejected(self):
        (self.root / "autoresearch.sh").unlink()
        (self.root / "autoresearch.sh").symlink_to(self.root / "code.txt")
        self.assertEqual(self.run_loop().returncode, 2)
        self.assertFalse((self.root / "invocations").exists())

    def test_missing_or_fifo_protected_path_is_rejected_without_hanging(self):
        self.assertEqual(self.run_loop("--protect", "missing.py").returncode, 2)
        os.mkfifo(self.root / "checks.sh")
        self.assertEqual(self.run_loop().returncode, 2)
        self.assertFalse((self.root / "invocations").exists())

    def test_dangling_pause_symlink_stops_without_launch_or_writes(self):
        pause = self.root / ".autoresearch-off"
        pause.symlink_to(self.root / "absent-target")
        result = self.run_loop()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("session is paused", result.stderr)
        self.assertTrue(pause.is_symlink())
        self.assertFalse((self.root / "absent-target").exists())
        self.assertFalse((self.root / "invocations").exists())

    def test_state_symlink_and_fifo_are_rejected_without_launch(self):
        state = self.root / "autoresearch.jsonl"
        contents = state.read_text()
        state.unlink()
        target = self.root / ".autoresearch-actual.jsonl"
        target.write_text(contents)
        state.symlink_to(target)
        result = self.run_loop()
        self.assertEqual(result.returncode, 2, result.stderr)
        state.unlink()
        os.mkfifo(state)
        result = self.run_loop()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("regular file", result.stderr)
        self.assertFalse((self.root / "invocations").exists())

    def test_rewriting_prior_result_and_appending_one_is_rejected(self):
        result = self.run_loop("--max-turns", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.root / ".autoresearch-off").unlink()
        result = self.run_loop(mode="history")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("rewrote existing state history", result.stderr)
        self.assertTrue((self.root / ".autoresearch-off").exists())

    @staticmethod
    def process_running(pid):
        result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                                capture_output=True, text=True)
        status = result.stdout.strip()
        return bool(status) and not status.startswith("Z")

    def test_sigterm_pauses_and_stops_active_codex(self):
        supervisor = subprocess.Popen(
            [sys.executable, str(RUNNER), "--workspace", str(self.root)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=dict(self.env, AR_TEST_MODE="sleep"),
        )
        pid_path = self.root / ".autoresearch-test-parent.pid"
        try:
            deadline = time.monotonic() + 5
            while not pid_path.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(pid_path.exists(), "fake Codex did not launch")
            child_pid = int(pid_path.read_text())
            supervisor.send_signal(signal.SIGTERM)
            _, stderr = supervisor.communicate(timeout=8)
            self.assertEqual(supervisor.returncode, 143, stderr)
            self.assertTrue((self.root / ".autoresearch-off").exists())
            self.assertFalse(self.process_running(child_pid))
        finally:
            if supervisor.poll() is None:
                supervisor.kill()
                supervisor.communicate()
            if pid_path.exists() and self.process_running(int(pid_path.read_text())):
                os.killpg(int(pid_path.read_text()), signal.SIGKILL)

    def test_successful_parent_cannot_leave_sigterm_ignoring_descendant_running(self):
        child_pid = None
        try:
            result = self.run_loop("--max-turns", "1", mode="descendant_exit")
            self.assertEqual(result.returncode, 0, result.stderr)
            child_pid = int((self.root / ".autoresearch-test-child.pid").read_text())
            deadline = time.monotonic() + 2
            while self.process_running(child_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(self.process_running(child_pid))
        finally:
            if child_pid is not None and self.process_running(child_pid):
                os.kill(child_pid, signal.SIGKILL)

    def test_turn_timeout_includes_sending_prompt_to_unresponsive_stdin(self):
        spec = importlib.util.spec_from_file_location("codex_loop_test", RUNNER)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        log_dir = self.root / "experiments/direct"
        log_dir.mkdir(parents=True)
        started = time.monotonic()
        with self.assertRaisesRegex(runner.LoopError, "budget expired"):
            runner.run_turn([sys.executable, "-c", "import time; time.sleep(30)"],
                            "large prompt " * 100000, self.root, log_dir, 0.2)
        self.assertLess(time.monotonic() - started, 6)


if __name__ == "__main__":
    unittest.main()
