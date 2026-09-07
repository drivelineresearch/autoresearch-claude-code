# Codex sandbox repair and integration verification

Host: `dc-boddydev`, Ubuntu 24.04.4, kernel `6.8.0-111-generic`.
Date: 2026-09-07. Codex CLI `0.153.4`; Bubblewrap `0.9.0`.

## Cause

The original error was reproducible without a model:

```bash
bwrap --unshare-user --unshare-net --ro-bind / / -- /bin/true
# bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

Kernel audit records showed `/usr/bin/bwrap` transitioning from `unconfined` into
AppArmor's `unprivileged_userns` profile. That profile denies capabilities needed
by Bubblewrap while constructing its namespaces. The host had
`kernel.apparmor_restrict_unprivileged_userns=1`, but no dedicated Bubblewrap
AppArmor profile. This was an AppArmor integration prerequisite, not a failed
experiment, missing API authentication, or a need for unrestricted Codex execution.

## Host repair

Following [OpenAI's Ubuntu 24.04 sandbox guidance](https://learn.chatgpt.com/docs/sandboxing),
loaded the distribution-provided `bwrap-userns-restrict` profile. To keep the
change limited to Bubblewrap, downloaded and extracted `apparmor-profiles` without
installing/loading its other optional profiles:

1. Downloaded Ubuntu's `apparmor-profiles` package version
   `4.0.1really4.0.1-0ubuntu0.24.04.7` using `apt-get download` and extracted it with
   `dpkg-deb -x` in a temporary directory.
2. Validated `usr/share/apparmor/extra-profiles/bwrap-userns-restrict` using
   `sudo apparmor_parser -Q <extracted-profile>`.
3. Verified `/etc/apparmor.d/bwrap-userns-restrict` was absent, then installed the
   exact vendor bytes there with owner `root:root` and mode `0644`.
4. Loaded only that profile with
   `sudo apparmor_parser -r /etc/apparmor.d/bwrap-userns-restrict`.

Installed profile SHA-256:
`11d39094f044f0cda0febb3ad517b830301da6b2ce929664af09ee9e4dd264f9`.
It defines `bwrap` and `unpriv_bwrap`: Bubblewrap may construct its namespace;
its child profile denies capabilities. Both loaded in enforce mode. The existing
`unprivileged_userns` profile remained enforced and the global restriction stayed
`1`. AppArmor is enabled and active, so the profile is persisted for normal service
startup. No reboot or broad AppArmor reload was needed. No Codex user configuration
or sandbox bypass was introduced.

There was no previous file to back up. If this specific repair must be rolled back,
verify the checksum above, unload it with
`sudo apparmor_parser -R /etc/apparmor.d/bwrap-userns-restrict`, and remove only
that installed profile. This restores the previous missing-profile condition and
may make Bubblewrap fail again; do not use rollback during active sandbox work.

## Boundary verification

After repair, the raw Bubblewrap command above and this no-model Codex probe passed:

```bash
codex sandbox -P :workspace -C /path/to/experiment /usr/bin/true
```

CLI 0.153.4 uses `codex sandbox` directly; `linux` is not a subcommand. `:workspace`
is a built-in permission-profile name. These probes verified:

| Operation | Result |
|---|---|
| Write/read/remove a file inside the experiment workspace | Allowed |
| Write into a separate directory under the user's cache | Denied, read-only filesystem (`EROFS`) |
| Reach a listener in the host network namespace | Blocked |
| Create an Internet-family socket in the isolated policy probe | Denied (`EPERM`) |
| Write `.codex` under the workspace | Denied (`EROFS`) |
| Write `.git` under the default workspace policy | Denied (`EROFS`) |

## Git metadata needed by the experiment loop

The last check exposed a separate integration gap: a kept experiment must commit,
but the default sandbox protects Git metadata. The runner now grants only the
selected repository's resolved Git metadata directories through `--add-dir`,
while retaining `--sandbox workspace-write` and approval policy `never`.

For a regular checkout this is its `.git` directory. For a linked worktree it
includes the worktree's Git directory plus the common Git directory containing
objects and refs. That common metadata is shared with other worktrees: use a
separate clone when the task needs stronger repository isolation. No additional
source worktree or entire home directory is granted. Scoped git operations remain
required; the grant permits Git metadata writes and is not a per-ref permission.

A no-model probe through Codex's legacy `workspaceWrite` policy verified that adding
only the resolved `.git` write root allows Git metadata writes while `.codex`,
outside workspace paths, and network access remain blocked. Another sandbox probe
successfully staged and committed a single fixture candidate with the same scoped
metadata permission. The authenticated loop verification below checks the actual
runner's `--add-dir` integration.

## Authenticated rerun: passed

A fresh isolated fixture preserved the failed run's original evidence and used the
same deterministic experiment: change only `candidate.json` from `{"value":1}` to
`{"value":2}` under a fixed scorer and correctness check. No model override or
sandbox bypass was used. The actual command was:

```bash
python3 skills/autoresearch/scripts/codex_loop.py \
  --workspace /tmp/autoresearch-codex-verified-jclx_8qf \
  --max-turns 1 --max-seconds 300 --turn-timeout 240
```

The runner invoked Codex with `--sandbox workspace-write`, approval policy `never`,
and `--add-dir /tmp/autoresearch-codex-verified-jclx_8qf/.git`.
It exited **0**, reporting `run budget reached: 2 >= 2` and writing the pause sentinel.

Independent verification after the model finished confirmed:

- Exactly one new commit: `23da7748ade2080cdd895b73a3b5c9f53c997c5e`.
- That commit changes only `candidate.json`; its value is now 2.
- Exactly one appended result: run 2, segment 0, `keep`, metric 2, matching HEAD.
- Re-executed benchmark produced `METRIC score=2`; correctness check exited 0.
- Original config/baseline bytes remain the ledger prefix.
- `autoresearch.sh`, `checks.sh`, and `.gitignore` hashes are unchanged.
- Dashboard and worklog were updated, worktree is clean, and the loop is paused at
  the configured 2/2 run cap. No automatic retry or second candidate occurred.

The fixture and logs are temporary local evidence, retained at the workspace above.
The final model message is in `experiments/codex/run-2-irnku2ii/final.md` there;
independent check results are in `/tmp/autoresearch-codex-verified-result.json`.
The experiment is deliberately synthetic: it verifies integration and control
flow, not an optimization or scientific performance claim.

The separate linked-worktree native sandbox test also committed successfully
while preserving the original checkout's HEAD and candidate. The original
checkout's source files and the linked worktree's `.codex` remained protected,
and network access remained blocked. Common Git metadata is writable by
necessity; source-file isolation is not per-branch metadata isolation.

Final automated suite: **94 unique tests passed, no skips, in 27.872 seconds**,
including standalone and linked-worktree metadata grants and rejection of broad
or missing directory grants. Host repair is a system prerequisite; it is not
silently installed by this repository's user-level skill installer.
