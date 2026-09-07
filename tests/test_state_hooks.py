"""Runtime regression tests; isolated temp directories, no real agent or benchmark."""
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "skills/autoresearch/scripts/ar_state.py"
LOGGER = HELPER.with_name("ar-log.sh")
SPEC = importlib.util.spec_from_file_location("ar_state_under_test", HELPER)
state = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(state)


def config(**updates):
    return {
        "type": "config", "name": "test", "metricName": "loss",
        "metricUnit": "", "bestDirection": "lower", "noiseFloor": 0,
        "maxRuns": 20, "maxSeconds": None, "targetMetric": None,
        "startedAt": 100, **updates,
    }


def result(run=1, segment=0, **updates):
    return {
        "run": run, "segment": segment, "commit": "abc1234", "metric": 10,
        "metrics": {}, "status": "keep", "description": "baseline",
        "timestamp": 101, **updates,
    }


class RuntimeFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="autoresearch state ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "autoresearch.jsonl"

    def write(self, *rows):
        self.path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def cli(self, *args, env=None):
        return subprocess.run(
            [sys.executable, str(HELPER), *map(str, args)], cwd=self.directory,
            text=True, capture_output=True, env=env, timeout=10,
        )

    def append(self, **kwargs):
        values = {"run": "auto", "commit": "abc1234", "metric": 9,
                  "status": "keep", "segment": "auto", "description": "candidate"}
        values.update(kwargs)
        return state.append_result(self.path, **values)


class StateTests(RuntimeFixture):
    def test_budgets_are_segment_local_and_indices_are_global(self):
        self.write(config(maxRuns=1, targetMetric=10), result(), config(maxRuns=2), result(2, 1, metric=99))
        current = state.read_state(self.path, now=101)
        self.assertEqual((current["total_runs"], current["segment_runs"], current["next_run"]), (2, 1, 3))
        self.assertFalse(current["budget_reached"])
        self.assertEqual(current["best"]["run"], 2)
        self.append(metric=50)
        self.assertTrue(state.read_state(self.path)["budget_reached"])

    def test_target_uses_only_kept_current_segment_in_both_directions(self):
        for direction, best, target, hit in (("lower", 3, 4, True), ("higher", 3, 4, False)):
            with self.subTest(direction=direction):
                self.write(config(bestDirection=direction, targetMetric=target), result(metric=best), result(2, metric=1000, status="discard"))
                self.assertEqual(state.read_state(self.path)["budget_reached"], hit)
        self.write(config(bestDirection="higher", targetMetric=3), result(metric=3))
        self.assertTrue(state.read_state(self.path)["budget_reached"])

    def test_elapsed_time_boundary(self):
        self.write(config(maxSeconds=5))
        self.assertFalse(state.read_state(self.path, now=104.9)["budget_reached"])
        self.assertTrue(state.read_state(self.path, now=105)["budget_reached"])

    def test_legacy_defaults_remain_bounded(self):
        self.write({"type": "config", "name": "old", "metricName": "x", "metricUnit": "", "bestDirection": "higher"})
        self.assertEqual(state.read_state(self.path)["config"]["maxRuns"], 200)

    def test_invalid_configs_are_not_silently_unbounded(self):
        invalid = [
            {"maxRuns": True}, {"maxRuns": 1.5}, {"maxRuns": "4"},
            {"maxRuns": -1}, {"maxRuns": 0}, {"maxRuns": float("nan")},
            {"noiseFloor": -1}, {"maxSeconds": -1}, {"startedAt": None},
            {"targetMetric": float("inf")}, {"bestDirection": "up"},
        ]
        for change in invalid:
            with self.subTest(change=change):
                self.write(config(**change))
                with self.assertRaises(state.StateError):
                    state.read_state(self.path)
        cfg = config(maxSeconds=10)
        del cfg["startedAt"]
        self.write(cfg)
        with self.assertRaises(state.StateError):
            state.read_state(self.path)

    def test_invalid_records_are_rejected_without_skipping(self):
        cases = [[], None, "text", {"run": 1}, result(run=2), result(run=True),
                 result(segment=1), result(metric=float("nan")), result(metric=True),
                 result(metrics={"x": "text"}), result(metrics={"x": float("inf")}),
                 result(status="win"), result(parent=1), result(commit="not-a-hash")]
        for row in cases:
            with self.subTest(row=row):
                self.write(config(), row)
                with self.assertRaises(state.StateError):
                    state.read_state(self.path)
        for raw in ('{"run":', '{"type":"config","type":"config"}', '{"x":1e999}'):
            self.path.write_text(json.dumps(config()) + "\n" + raw + "\n")
            with self.assertRaises(state.StateError):
                state.read_state(self.path)

    def test_missing_empty_or_result_first_is_invalid(self):
        with self.assertRaises(state.StateError):
            state.read_state(self.path)
        for text in ("", "\n ", json.dumps(result())):
            self.path.write_text(text)
            with self.assertRaises(state.StateError):
                state.read_state(self.path)

    def test_non_regular_state_is_rejected_without_opening_fifo(self):
        os.mkfifo(self.path)
        with self.assertRaises(state.StateError):
            state.read_state(self.path)

    def test_parse_snapshot_does_not_reread_state(self):
        self.write(config(), result())
        snapshot = self.path.read_text()
        self.path.write_text("corrupted later")
        self.assertEqual(state.parse_state(snapshot)["total_runs"], 1)

    def test_init_never_overwrites_history_and_reinit_preserves_numbering(self):
        state.init_state(self.path, config())
        self.append()
        before = self.path.read_bytes()
        with self.assertRaises(state.StateError):
            state.init_state(self.path, config())
        self.assertEqual(self.path.read_bytes(), before)
        current = state.init_state(self.path, config(name="second"), new_segment=True)
        self.assertEqual((current["segment"], current["segment_runs"], current["next_run"]), (1, 0, 2))
        self.assertEqual(self.append()["run"], 2)

    def test_invalid_append_preserves_original_bytes(self):
        self.write(config(), result())
        original = self.path.read_bytes()
        for change in ({"run": 1}, {"segment": 8}, {"metric": float("inf")}, {"parent": 4}):
            with self.subTest(change=change):
                with self.assertRaises((state.StateError, ValueError)):
                    self.append(**change)
                self.assertEqual(self.path.read_bytes(), original)

    def test_parent_must_be_in_current_segment(self):
        self.write(config(), result(), config(), result(2, 1))
        with self.assertRaises(state.StateError):
            self.append(parent=1)
        self.assertEqual(self.append(parent=2, op="improve")["parent"], 2)

    def test_cli_logging_escapes_strings_and_accepts_negative_scientific_numbers(self):
        self.write(config())
        description = 'don\'t use "echo"\nsecond line = here'
        logged = subprocess.run(
            [str(LOGGER), "--op", "draft", "--parent", "null", "auto", "abc1234", "-1e-4", "keep", "auto", description, "r2=-2e-5"],
            cwd=self.directory, text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(logged.returncode, 0, logged.stderr)
        row = json.loads(self.path.read_text().splitlines()[1])
        self.assertEqual(row["description"], description)
        self.assertEqual(row["metric"], -0.0001)
        self.assertEqual(row["metrics"], {"r2": -0.00002})
        self.assertEqual(row["op"], "draft")

    def test_cli_secondary_metrics_must_be_valid_unique_numeric_pairs(self):
        self.write(config())
        original = self.path.read_bytes()
        for metrics in (("broken",), ("=1",), ("x=1", "x=2"), ("x=NaN",), ("x=text",)):
            with self.subTest(metrics=metrics):
                completed = self.cli("append", "auto", "abc1234", "2", "keep", "auto", "bad", *metrics)
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(json.loads(completed.stdout)["valid"])
                self.assertEqual(self.path.read_bytes(), original)

    def test_status_error_is_machine_readable(self):
        completed = self.cli("status")
        self.assertEqual(completed.returncode, 2)
        self.assertFalse(json.loads(completed.stdout)["valid"])

    def test_cli_init_and_path_override(self):
        alternate = self.directory / "other state.jsonl"
        created = self.cli("init", "--path", alternate, "--name", "test", "--metric-name", "x", "--direction", "higher", "--max-runs", "none")
        self.assertEqual(created.returncode, 0, created.stderr)
        self.assertIsNone(state.read_state(alternate)["config"]["maxRuns"])
        completed = self.cli("append", "auto", "abc1234", "1", "keep", "auto", "baseline", env={**os.environ, "AR_JSONL": str(alternate)})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(self.path.exists())

    def test_atomic_replace_failure_preserves_original(self):
        self.write(config(), result())
        original = self.path.read_bytes()
        with mock.patch.object(state.os, "replace", side_effect=OSError("simulated failure")):
            with self.assertRaises(OSError):
                self.append()
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(sorted(p.name for p in self.directory.iterdir()), [".autoresearch.jsonl.lock", "autoresearch.jsonl"])

    def test_busy_writer_lock_has_a_bounded_wait(self):
        self.write(config())
        original = self.path.read_bytes()
        with mock.patch.object(state.fcntl, "flock", side_effect=BlockingIOError), \
                mock.patch.object(state.time, "monotonic", side_effect=(0, 6)):
            with self.assertRaisesRegex(state.StateError, "timed out"):
                self.append()
        self.assertEqual(self.path.read_bytes(), original)

    def test_writes_reject_state_and_lock_symlinks(self):
        actual = self.directory / "actual.jsonl"
        actual.write_text(json.dumps(config()) + "\n")
        original = actual.read_bytes()
        self.path.symlink_to(actual)
        with self.assertRaises(state.StateError):
            self.append()
        self.assertEqual(actual.read_bytes(), original)
        self.path.unlink()
        self.path.write_bytes(original)
        lock = self.directory / ".autoresearch.jsonl.lock"
        lock.unlink()
        lock.symlink_to(actual)
        with self.assertRaises(OSError):
            self.append()
        self.assertEqual(actual.read_bytes(), original)

    def test_concurrent_auto_writers_allocate_unique_runs(self):
        self.write(config())
        def log(index):
            return self.cli("append", "auto", "abc1234", str(index), "keep", "auto", f"parallel {index}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            completed = list(executor.map(log, range(12)))
        for item in completed:
            self.assertEqual(item.returncode, 0, item.stderr)
        rows = [json.loads(line) for line in self.path.read_text().splitlines()[1:]]
        self.assertEqual([row["run"] for row in rows], list(range(1, 13)))
        self.assertEqual({row["description"] for row in rows}, {f"parallel {index}" for index in range(12)})

    def test_concurrent_explicit_writers_reject_duplicate_run(self):
        self.write(config())
        def log(_):
            return self.cli("append", "1", "abc1234", "1", "keep", "0", "same run")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            completed = list(executor.map(log, range(2)))
        self.assertEqual(sorted(item.returncode for item in completed), [0, 2])
        self.assertEqual(state.read_state(self.path)["total_runs"], 1)


class HookTests(RuntimeFixture):
    def hook(self, event, payload=None, process_cwd=None, script=None):
        if payload is None:
            payload = {"cwd": str(self.directory)}
        return subprocess.run(
            [str(script or ROOT / "hooks" / f"autoresearch-{event}.sh")],
            input=payload if isinstance(payload, str) else json.dumps(payload),
            cwd=process_cwd or self.directory, text=True, capture_output=True, timeout=10,
        )

    def activate(self, *rows):
        (self.directory / "autoresearch.md").write_text("# Objective\nDo safe experiments.\n")
        self.write(*(rows or (config(),)))

    def test_stop_continues_with_clean_json_and_pauses_at_budget(self):
        self.activate(config(maxRuns=1))
        running = self.hook("stop")
        self.assertEqual(running.returncode, 0)
        self.assertEqual(json.loads(running.stdout)["decision"], "block")
        self.append()
        paused = self.hook("stop")
        self.assertEqual(paused.stdout, "")
        self.assertTrue((self.directory / ".autoresearch-off").exists())

    def test_bad_state_allows_stop_without_destroying_evidence(self):
        self.activate()
        self.path.write_text("broken\n")
        completed = self.hook("stop")
        self.assertEqual((completed.returncode, completed.stdout), (0, ""))
        self.assertIn("automatic continuation disabled", completed.stderr)
        self.assertEqual(self.path.read_text(), "broken\n")
        self.assertFalse((self.directory / ".autoresearch-off").exists())

    def test_every_hook_uses_payload_cwd_not_process_cwd(self):
        self.activate(config(), result())
        with tempfile.TemporaryDirectory() as elsewhere:
            for event in ("context", "sessionstart", "stop", "precompact"):
                with self.subTest(event=event):
                    completed = self.hook(event, process_cwd=elsewhere)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    if event != "precompact":
                        self.assertTrue(completed.stdout)
            self.assertFalse((Path(elsewhere) / "experiments").exists())
        self.assertTrue((self.directory / "experiments/autoresearch.jsonl.precompact.bak").exists())

    def test_invalid_payload_never_falls_back_to_active_process_directory(self):
        self.activate(config(maxRuns=1), result())
        for payload in ("broken", [], {}, {"cwd": None}, {"cwd": "relative"}, {"cwd": str(self.directory / "missing")}):
            for event in ("context", "sessionstart", "stop", "precompact"):
                with self.subTest(payload=payload, event=event):
                    completed = self.hook(event, payload)
                    self.assertEqual((completed.returncode, completed.stdout), (0, ""))
        self.assertFalse((self.directory / ".autoresearch-off").exists())
        self.assertFalse((self.directory / "experiments").exists())

    def test_paused_session_is_quiet_for_all_hooks(self):
        self.activate()
        (self.directory / ".autoresearch-off").touch()
        for event in ("context", "sessionstart", "stop", "precompact"):
            self.assertEqual(self.hook(event).stdout, "")
        self.assertFalse((self.directory / "experiments").exists())

    def test_dangling_pause_sentinel_still_pauses_all_hooks(self):
        self.activate()
        (self.directory / ".autoresearch-off").symlink_to(self.directory / "missing")
        for event in ("context", "sessionstart", "stop", "precompact"):
            completed = self.hook(event)
            self.assertEqual((completed.returncode, completed.stdout), (0, ""))
        self.assertFalse((self.directory / "experiments").exists())

    def test_inspection_unlinks_only_sentinel_even_for_dangling_symlink(self):
        self.activate()
        target = self.directory / "other-file"
        target.write_text("preserve")
        sentinel = self.directory / ".autoresearch-inspect"
        for destination in (target, self.directory / "missing"):
            sentinel.symlink_to(destination)
            self.assertEqual(self.hook("stop").stdout, "")
            self.assertFalse(os.path.lexists(sentinel))
            self.assertEqual(target.read_text(), "preserve")

    def test_inspection_sentinel_is_consumed_once(self):
        self.activate()
        sentinel = self.directory / ".autoresearch-inspect"
        sentinel.touch()
        self.assertEqual(self.hook("stop").stdout, "")
        self.assertFalse(sentinel.exists())
        self.assertEqual(json.loads(self.hook("stop").stdout)["decision"], "block")

    def test_checkpoint_and_rehydrate_parse_json_with_spaces(self):
        self.activate(config(), result(description='contains "run": and "status":"keep"'), result(2, status="discard"))
        original = self.path.read_bytes()
        checkpoint = self.hook("precompact")
        self.assertEqual((checkpoint.returncode, checkpoint.stdout), (0, ""))
        self.assertEqual((self.directory / "experiments/autoresearch.jsonl.precompact.bak").read_bytes(), original)
        self.assertIn("Runs so far: 2 | kept: 1", (self.directory / "experiments/worklog.md").read_text())
        restored = self.hook("sessionstart")
        self.assertIn("2 runs logged, 1 kept", restored.stdout)
        self.assertIn("Current best loss: 10 (run 1)", restored.stdout)

    def test_checkpoint_cannot_write_through_symlinked_directory(self):
        self.activate(config(), result())
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside)
            (self.directory / "experiments").symlink_to(target)
            completed = self.hook("precompact")
            self.assertEqual((completed.returncode, completed.stdout), (0, ""))
            self.assertIn("symlink/non-directory", completed.stderr)
            self.assertEqual(list(target.iterdir()), [])

    def test_bad_state_never_replaces_last_good_checkpoint(self):
        self.activate(config(), result())
        self.hook("precompact")
        snapshot = self.directory / "experiments/autoresearch.jsonl.precompact.bak"
        original = snapshot.read_bytes()
        self.path.write_text("broken\n")
        completed = self.hook("precompact")
        self.assertEqual((completed.returncode, completed.stdout), (0, ""))
        self.assertEqual(snapshot.read_bytes(), original)

    def test_hook_symlinks_work_from_paths_with_spaces(self):
        self.activate()
        linked = self.directory / "hook symlink.sh"
        linked.symlink_to(ROOT / "hooks/autoresearch-stop.sh")
        self.assertEqual(json.loads(self.hook("stop", script=linked).stdout)["decision"], "block")


if __name__ == "__main__":
    unittest.main()
