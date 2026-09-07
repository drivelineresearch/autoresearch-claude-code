#!/usr/bin/env python3
"""Transactional user installation; only the Python standard library is needed."""

import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import stat
import sys
import tempfile
import uuid


HOOKS = {
    "Stop": ("", "autoresearch-stop.sh"),
    "PreCompact": ("", "autoresearch-precompact.sh"),
    "SessionStart": ("startup|resume|compact", "autoresearch-sessionstart.sh"),
    "UserPromptSubmit": ("", "autoresearch-context.sh"),
}


class InstallError(Exception):
    """A failed preflight or recoverable filesystem operation."""


def owned_link(destination, source):
    return destination.is_symlink() and destination.resolve() == source.resolve()


def exists(path):
    """Include dangling symlinks."""
    return os.path.lexists(path)


def check_parent(path):
    parent = path.parent
    while not exists(parent):
        parent = parent.parent
    if not parent.is_dir():
        raise InstallError(f"Not a directory: {parent}")


def validate_settings(config):
    if not isinstance(config, dict):
        raise InstallError("settings.json must contain a JSON object")
    hooks = config.get("hooks", {})
    if not isinstance(hooks, dict):
        raise InstallError("settings.json hooks must be an object")
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            raise InstallError(f"settings.json hooks.{event} must be a list")
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                raise InstallError(f"settings.json hooks.{event} groups need a hooks list")
            if "matcher" in group and not isinstance(group["matcher"], str):
                raise InstallError(f"settings.json hooks.{event} matcher must be a string")
            for hook in group["hooks"]:
                if not isinstance(hook, dict):
                    raise InstallError(f"settings.json hooks.{event} hook must be an object")
                if hook.get("type") == "command" and not isinstance(hook.get("command"), str):
                    raise InstallError(f"settings.json hooks.{event} command must be a string")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"non-JSON numeric constant: {value}")


def read_settings(path):
    check_parent(path)
    if not exists(path):
        return {}, None, 0o600
    # Replacing a settings symlink would change what file Claude reads.
    if path.is_symlink() or not path.is_file():
        raise InstallError(f"Refusing to replace nonregular settings file: {path}")
    data = path.read_bytes()
    try:
        config = json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, UnicodeError) as exc:
        raise InstallError(f"Invalid JSON in {path}: {exc}") from exc
    validate_settings(config)
    return config, data, stat.S_IMODE(path.stat().st_mode)


def update_settings(config, home, installing, removable_scripts):
    updated = copy.deepcopy(config)
    hooks = updated.setdefault("hooks", {}) if installing else updated.get("hooks", {})
    for event, (matcher, script) in HOOKS.items():
        command = shlex.quote(str(home / ".claude" / "hooks" / script))
        # Support the original installer without matching arbitrary substrings.
        recognized = {command, f"~/.claude/hooks/{script}"}
        if not installing and script not in removable_scripts:
            continue
        groups = hooks.setdefault(event, []) if installing else hooks.get(event, [])
        if installing:
            for group in groups:
                for hook in group["hooks"]:
                    if hook.get("type") == "command" and hook.get("command") in recognized:
                        hook["command"] = command
            if not any(
                group.get("matcher", "") == matcher
                and any(hook.get("type") == "command" and hook.get("command") == command
                        for hook in group["hooks"])
                for group in groups
            ):
                group = {"hooks": [{"type": "command", "command": command}]}
                if matcher:
                    group["matcher"] = matcher
                groups.append(group)
        else:
            survivors = []
            for group in groups:
                remaining = [hook for hook in group["hooks"]
                             if not (hook.get("type") == "command"
                                     and hook.get("command") in recognized)]
                # Leave unrelated empty groups as they were.
                if remaining or remaining == group["hooks"]:
                    group["hooks"] = remaining
                    survivors.append(group)
            if survivors:
                hooks[event] = survivors
            elif event in hooks and groups:
                del hooks[event]
    if not installing and "hooks" in updated and not hooks and config.get("hooks"):
        del updated["hooks"]
    return updated


def atomic_write(path, data, mode):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), mode)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


class Transaction:
    def __init__(self):
        self.undo = []

    def mkdir(self, path):
        missing = []
        while not exists(path):
            missing.append(path)
            path = path.parent
        for directory in reversed(missing):
            directory.mkdir()
            self.undo.append(directory.rmdir)

    def link(self, destination, source):
        self.mkdir(destination.parent)
        destination.symlink_to(source, target_is_directory=source.is_dir())
        self.undo.append(destination.unlink)

    def unlink(self, destination, source):
        if not owned_link(destination, source):
            raise InstallError(f"Installation changed during uninstall: {destination}")
        target = os.readlink(destination)
        destination.unlink()
        self.undo.append(lambda: destination.symlink_to(target))

    def settings(self, path, original, replacement, mode):
        # Detect ordinary concurrent edits after preflight rather than overwrite them.
        if (exists(path) != (original is not None)
                or (exists(path) and (path.is_symlink() or path.read_bytes() != original))):
            raise InstallError(f"Settings changed during operation: {path}; retry")
        self.mkdir(path.parent)
        backup = None
        if original is not None:
            backup = path.with_name(f"{path.name}.autoresearch-backup-{uuid.uuid4().hex}")
            descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                self.undo.append(backup.unlink)
                os.fchmod(handle.fileno(), mode)
                handle.write(original)
                handle.flush()
                os.fsync(handle.fileno())
        atomic_write(path, replacement, mode)
        return backup

    def rollback(self):
        errors = []
        for undo in reversed(self.undo):
            try:
                undo()
            except OSError as exc:
                errors.append(str(exc))
        return errors


def run(action, repo, home, claude, codex):
    installing = action == "install"
    links = []
    if codex:
        links.append((home / ".agents/skills/autoresearch", repo / "skills/autoresearch"))
    if claude:
        links.extend([
            (home / ".claude/skills/autoresearch", repo / "skills/autoresearch"),
            (home / ".claude/commands/autoresearch.md", repo / "commands/autoresearch.md"),
        ])
        links.extend((home / ".claude/hooks" / script, repo / "hooks" / script)
                     for _, script in HOOKS.values())
    pending = []
    skipped = []
    removable_scripts = set()
    for destination, source in links:
        check_parent(destination)
        owned = owned_link(destination, source)
        if installing:
            if not source.exists():
                raise InstallError(f"Missing installation source: {source}")
            if source.name == "autoresearch":
                if not source.is_dir() or not (source / "SKILL.md").is_file():
                    raise InstallError(f"Missing skill source: {source / 'SKILL.md'}")
            elif not source.is_file():
                raise InstallError(f"Installation source must be a file: {source}")
            if source.suffix == ".sh" and not os.access(source, os.X_OK):
                raise InstallError(f"Hook source must be executable: {source}")
            if exists(destination) and not owned:
                raise InstallError(f"Refusing to overwrite an existing file, directory, or foreign link: {destination}")
            if not owned:
                pending.append((destination, source))
        elif owned:
            pending.append((destination, source))
        elif exists(destination):
            skipped.append(destination)
        if not exists(destination) or owned:
            removable_scripts.add(destination.name)

    settings_path = home / ".claude/settings.json"
    original = replacement = None
    mode = 0o600
    if claude:
        config, original, mode = read_settings(settings_path)
        updated = update_settings(config, home, installing, removable_scripts)
        if updated != config:
            replacement = (json.dumps(updated, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    transaction = Transaction()
    backup = None
    try:
        for destination, source in pending:
            if installing:
                transaction.link(destination, source)
            else:
                transaction.unlink(destination, source)
        # Commit settings last, after all reversible filesystem changes succeed.
        if replacement is not None:
            backup = transaction.settings(settings_path, original, replacement, mode)
    except (OSError, InstallError) as exc:
        errors = transaction.rollback()
        detail = f" Rollback needs attention: {'; '.join(errors)}" if errors else " Changes rolled back."
        raise InstallError(f"{exc}.{detail}") from exc

    for destination, _ in pending:
        print(f"  {'Linked' if installing else 'Removed'}: {destination}")
    for destination in skipped:
        print(f"  Preserved nonowned path: {destination}")
    if backup:
        print(f"  Settings backup: {backup}")
    if installing:
        if claude:
            print("Claude Code: run /autoresearch to start.")
        if codex:
            print("Codex: start a new session and invoke $autoresearch.")
    print("Done." if pending or replacement is not None else "Already up to date.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["install", "uninstall"])
    parser.add_argument("--claude", action="store_true", help="select Claude Code (the default)")
    parser.add_argument("--codex", action="store_true", help="select Codex user skills in ~/.agents/skills")
    parser.add_argument("--all", action="store_true", help="select both Claude Code and Codex")
    args = parser.parse_args(argv)
    claude = args.claude or args.all or not args.codex
    codex = args.codex or args.all
    try:
        home = Path.home()
        if not home.is_absolute():
            raise InstallError("The user home directory must be an absolute path")
        run(args.action, Path(__file__).resolve().parents[1], home, claude, codex)
    except (OSError, InstallError, RuntimeError) as exc:
        print(f"autoresearch: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
