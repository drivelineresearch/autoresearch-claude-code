"""Installer regressions use fixture homes and never touch real user settings."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("autoresearch_installer", REPO / "scripts/install.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="autoresearch-install-")
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.home = base / "home ' $ spaces"
        self.home.mkdir()
        self.repo = base / "repo ' $ spaces"
        self.repo.mkdir()
        for name in ("install.sh", "uninstall.sh", "scripts/install.py"):
            destination = self.repo / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, destination)
        (self.repo / "skills/autoresearch").mkdir(parents=True)
        (self.repo / "skills/autoresearch/SKILL.md").write_text("fixture skill")
        (self.repo / "commands").mkdir()
        (self.repo / "commands/autoresearch.md").write_text("fixture command")
        (self.repo / "hooks").mkdir()
        for _, script in installer.HOOKS.values():
            path = self.repo / "hooks" / script
            path.write_text("#!/bin/sh\nprintf '%s' fixture-hook\n")
            path.chmod(0o755)
        self.settings = self.home / ".claude/settings.json"

    def cli(self, action="install", *arguments, success=True):
        result = subprocess.run(
            ["bash", str(self.repo / f"{action}.sh"), *arguments],
            env={**os.environ, "HOME": str(self.home)}, cwd=self.repo,
            text=True, capture_output=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def write_settings(self, value, mode=0o640):
        self.settings.parent.mkdir(parents=True, exist_ok=True)
        self.settings.write_text(value if isinstance(value, str) else json.dumps(value))
        self.settings.chmod(mode)

    def snapshot(self):
        return {str(path.relative_to(self.home)): (
            ("link", os.readlink(path)) if path.is_symlink() else
            ("dir",) if path.is_dir() else
            ("file", path.read_bytes(), stat.S_IMODE(path.stat().st_mode)))
            for path in self.home.rglob("*")}

    def backups(self):
        return list(self.settings.parent.glob("settings.json.autoresearch-backup-*"))

    def run_direct(self, action="install", claude=True, codex=True):
        with contextlib.redirect_stdout(io.StringIO()):
            installer.run(action, self.repo, self.home, claude=claude, codex=codex)

    def test_codex_only_installs_shared_skill_without_claude_configuration(self):
        self.cli("install", "--codex")
        destination = self.home / ".agents/skills/autoresearch"
        self.assertTrue(destination.is_symlink())
        self.assertEqual(destination.resolve(), (self.repo / "skills/autoresearch").resolve())
        self.assertFalse((self.home / ".claude").exists())
        snapshot = self.snapshot()
        self.cli("install", "--codex")
        self.assertEqual(snapshot, self.snapshot())
        self.cli("uninstall", "--codex")
        self.assertFalse(os.path.lexists(destination))
        self.cli("uninstall", "--codex")
        self.assertFalse((self.home / ".claude").exists())

    def test_default_claude_install_handles_spaces_and_shell_metacharacters(self):
        self.cli()
        self.assertFalse((self.home / ".agents").exists())
        config = json.loads(self.settings.read_text())
        self.assertEqual(set(config["hooks"]), set(installer.HOOKS))
        for event, (_, script) in installer.HOOKS.items():
            command = config["hooks"][event][0]["hooks"][0]["command"]
            self.assertEqual(shlex.split(command), [str(self.home / ".claude/hooks" / script)])
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "fixture-hook")
        self.assertEqual(stat.S_IMODE(self.settings.stat().st_mode), 0o600)
        snapshot = self.snapshot()
        inode = self.settings.stat().st_ino
        self.cli()
        self.assertEqual(snapshot, self.snapshot())
        self.assertEqual(inode, self.settings.stat().st_ino)
        self.assertFalse(self.backups())

    def test_all_and_combined_selectors_install_both_tools(self):
        self.cli("install", "--all")
        self.assertTrue((self.home / ".agents/skills/autoresearch").is_symlink())
        self.assertTrue((self.home / ".claude/skills/autoresearch").is_symlink())
        before = self.snapshot()
        self.cli("install", "--claude", "--codex")
        self.assertEqual(before, self.snapshot())
        self.cli("uninstall", "--all")
        self.assertFalse((self.home / ".agents/skills/autoresearch").is_symlink())
        self.assertFalse((self.home / ".claude/skills/autoresearch").is_symlink())
        before = self.snapshot()
        self.cli("uninstall", "--all")
        self.assertEqual(before, self.snapshot())

    def test_settings_replacement_is_atomic_preserves_mode_and_original_backup(self):
        original = b'{"permissions": {"allow": ["Read"]}, "custom": "untouched"}\n'
        self.write_settings(original.decode())
        inode = self.settings.stat().st_ino
        self.cli()
        self.assertNotEqual(inode, self.settings.stat().st_ino)
        self.assertEqual(stat.S_IMODE(self.settings.stat().st_mode), 0o640)
        backups = self.backups()
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        self.assertEqual(stat.S_IMODE(backups[0].stat().st_mode), 0o640)
        self.assertEqual(json.loads(self.settings.read_text())["custom"], "untouched")
        self.assertFalse(list(self.settings.parent.glob(".settings.json.*")))

    def test_invalid_settings_rejected_before_any_selected_install_changes(self):
        values = ["{broken", "[]", '{"hooks": [], "hooks": {}}', '{"custom": NaN}',
                  '{"hooks": []}', '{"hooks": {"Stop": {}}}',
                  '{"hooks": {"Stop": [null]}}', '{"hooks": {"Stop": [{"hooks": null}]}}',
                  '{"hooks": {"Stop": [{"hooks": ["bad"]}]}}',
                  '{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": null}]}]}}']
        for value in values:
            with self.subTest(value=value):
                self.write_settings(value)
                before = self.snapshot()
                self.cli("install", "--all", success=False)
                self.assertEqual(before, self.snapshot())

    def test_invalid_settings_rejected_before_any_selected_uninstall_changes(self):
        self.cli("install", "--all")
        self.write_settings("not JSON")
        before = self.snapshot()
        self.cli("uninstall", "--all", success=False)
        self.assertEqual(before, self.snapshot())

    def test_codex_does_not_inspect_invalid_claude_settings(self):
        self.write_settings("broken JSON")
        self.cli("install", "--codex")
        self.cli("uninstall", "--codex")
        self.assertEqual(self.settings.read_text(), "broken JSON")

    def test_existing_directory_conflict_prevents_entire_install(self):
        directory = self.home / ".claude/skills/autoresearch"
        directory.mkdir(parents=True)
        (directory / "valuable.md").write_text("keep")
        before = self.snapshot()
        self.cli("install", "--all", success=False)
        self.assertEqual(before, self.snapshot())

    def test_dangling_foreign_symlink_prevents_install_and_is_preserved_on_uninstall(self):
        destination = self.home / ".agents/skills/autoresearch"
        destination.parent.mkdir(parents=True)
        destination.symlink_to(self.home / "missing-foreign-target")
        before = self.snapshot()
        self.cli("install", "--all", success=False)
        self.assertEqual(before, self.snapshot())
        self.cli("uninstall", "--all")
        self.assertEqual(before, self.snapshot())

    def test_uninstall_preserves_nonowned_files_directories_and_hook_registration(self):
        skill = self.home / ".claude/skills/autoresearch"
        skill.mkdir(parents=True)
        (skill / "valuable.md").write_text("keep")
        command = self.home / ".claude/commands/autoresearch.md"
        command.parent.mkdir(parents=True)
        command.write_text("keep command")
        hook = self.home / ".claude/hooks/autoresearch-stop.sh"
        hook.parent.mkdir(parents=True)
        hook.write_text("keep hook")
        self.write_settings({"hooks": {"Stop": [{"hooks": [
            {"type": "command", "command": "~/.claude/hooks/autoresearch-stop.sh"}
        ]}]}})
        before = self.snapshot()
        result = self.cli("uninstall", "--all")
        self.assertIn("Preserved nonowned path", result.stdout)
        self.assertEqual(before, self.snapshot())

    def test_uninstall_matches_exact_command_and_preserves_mixed_hook_groups(self):
        exact = "~/.claude/hooks/autoresearch-stop.sh"
        surviving = [
            {"type": "command", "command": f"echo {exact}"},
            {"type": "command", "command": exact + ".custom"},
            {"type": "command", "command": "/custom/autoresearch-other.sh"},
            {"type": "prompt", "prompt": "autoresearch-stop.sh"},
        ]
        self.write_settings({"theme": "dark", "hooks": {"Stop": [
            {"matcher": "", "extra": "keep", "hooks": [
                {"type": "command", "command": exact}, *surviving]},
            {"hooks": []},
        ], "Other": [{"hooks": [{"type": "command", "command": exact}]}]}})
        self.cli("uninstall")
        config = json.loads(self.settings.read_text())
        self.assertEqual(config["theme"], "dark")
        self.assertEqual(config["hooks"]["Stop"], [
            {"matcher": "", "extra": "keep", "hooks": surviving}, {"hooks": []}])
        self.assertEqual(config["hooks"]["Other"][0]["hooks"][0]["command"], exact)

    def test_migrates_original_tilde_registration_without_duplicates(self):
        self.write_settings({"hooks": {"Stop": [{"hooks": [
            {"type": "command", "command": "~/.claude/hooks/autoresearch-stop.sh", "timeout": 12}
        ]}]}})
        self.cli()
        config = json.loads(self.settings.read_text())
        self.assertEqual(len(config["hooks"]["Stop"]), 1)
        hook = config["hooks"]["Stop"][0]["hooks"][0]
        self.assertEqual(hook["timeout"], 12)
        self.assertEqual(shlex.split(hook["command"]), [str(self.home / ".claude/hooks/autoresearch-stop.sh")])

    def test_settings_symlink_rejected_without_changes(self):
        target = self.home / "actual-settings.json"
        target.write_text("{}")
        self.settings.parent.mkdir()
        self.settings.symlink_to(target)
        before = self.snapshot()
        self.cli("install", "--all", success=False)
        self.assertEqual(before, self.snapshot())

    def test_missing_source_rejected_before_changes(self):
        (self.repo / "hooks/autoresearch-stop.sh").unlink()
        before = self.snapshot()
        self.cli("install", "--all", success=False)
        self.assertEqual(before, self.snapshot())

    def test_missing_skill_definition_rejected_before_changes(self):
        (self.repo / "skills/autoresearch/SKILL.md").unlink()
        before = self.snapshot()
        self.cli("install", "--all", success=False)
        self.assertEqual(before, self.snapshot())

    def test_nonexecutable_hook_source_rejected_before_changes(self):
        (self.repo / "hooks/autoresearch-stop.sh").chmod(0o644)
        before = self.snapshot()
        self.cli("install", "--all", success=False)
        self.assertEqual(before, self.snapshot())

    def test_parent_file_conflict_rejected_before_changes(self):
        (self.home / ".claude").write_text("keep")
        before = self.snapshot()
        self.cli("install", "--all", success=False)
        self.assertEqual(before, self.snapshot())

    def test_link_failure_rolls_back_earlier_links_and_created_directories(self):
        before = self.snapshot()
        original = installer.Transaction.link

        def fail_second(transaction, destination, source):
            if ".claude" in destination.parts:
                raise OSError("injected link failure")
            original(transaction, destination, source)

        with mock.patch.object(installer.Transaction, "link", fail_second):
            with self.assertRaisesRegex(installer.InstallError, "Changes rolled back"):
                self.run_direct()
        self.assertEqual(before, self.snapshot())

    def test_settings_write_failure_rolls_back_install_and_backup(self):
        self.write_settings('{"custom": true}\n')
        before = self.snapshot()
        with mock.patch.object(installer.os, "replace", side_effect=OSError("injected replace failure")):
            with self.assertRaisesRegex(installer.InstallError, "Changes rolled back"):
                self.run_direct()
        self.assertEqual(before, self.snapshot())

    def test_settings_write_failure_rolls_back_uninstall(self):
        self.run_direct()
        before = self.snapshot()
        with mock.patch.object(installer.os, "replace", side_effect=OSError("injected replace failure")):
            with self.assertRaisesRegex(installer.InstallError, "Changes rolled back"):
                self.run_direct("uninstall")
        self.assertEqual(before, self.snapshot())

    def test_concurrent_settings_edit_is_preserved_and_links_rolled_back(self):
        self.write_settings("{}")
        original_link = installer.Transaction.link

        def edit_settings(transaction, destination, source):
            original_link(transaction, destination, source)
            self.settings.write_text('{"concurrent": true}')

        with mock.patch.object(installer.Transaction, "link", edit_settings):
            with self.assertRaisesRegex(installer.InstallError, "Settings changed"):
                self.run_direct()
        self.assertEqual(self.settings.read_text(), '{"concurrent": true}')
        self.assertFalse((self.home / ".agents").exists())
        self.assertFalse((self.home / ".claude/skills").exists())
        self.assertFalse(self.backups())

    def test_invalid_selector_has_no_side_effects(self):
        before = self.snapshot()
        self.cli("install", "--other", success=False)
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
