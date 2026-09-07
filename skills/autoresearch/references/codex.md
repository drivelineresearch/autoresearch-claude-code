# Codex operation

The shared skill supports interactive Codex use. For continuation across model
turns, the included supervisor starts a fresh `codex exec` for each experiment and
rehydrates from files. It does not rely on Claude's hooks being understood by Codex.

## Install and initialize

From the plugin checkout, run `./install.sh --codex`. This links the complete skill
into `~/.agents/skills/autoresearch`; it does not change Codex configuration or
install Claude hooks. See [OpenAI's skill documentation](https://learn.chatgpt.com/docs/build-skills)
for current discovery locations and invocation. In CLI/IDE:

```text
$autoresearch set up a runtime optimization session with at most 10 runs; initialize and log the baseline, then pause
```

In the app, select the skill or ask to use it by name. Supply the actual benchmark,
metric, scoped files, and constraints. Review its initial baseline/harness. Use an
experiment branch and a clean worktree with session artifacts ignored. The
supervisor intentionally requires an already initialized session.

## Inspect, validate, run

From your experiment workspace:

```bash
python3 ~/.agents/skills/autoresearch/scripts/codex_loop.py --status
# After explicitly resuming and removing .autoresearch-off:
python3 ~/.agents/skills/autoresearch/scripts/codex_loop.py --dry-run
python3 ~/.agents/skills/autoresearch/scripts/codex_loop.py \
  --max-turns 10 --max-seconds 3600 --turn-timeout 300 \
  --protect path/to/scorer.py
```

`--workspace /absolute/path/to/worktree` overrides the current directory.
Repeat `--protect` for scorer, data, and split-definition files; these must be
files inside the worktree. `autoresearch.sh` and `checks.sh` (including its initial
absence) are always checked. Hashes are captured when the supervisor starts;
they do not prove a harness was unchanged before that invocation.

`--status` only reads state and reports whether the pause sentinel exists;
`--dry-run` also checks preconditions and prints the
command as a JSON argument array. Neither launches Codex nor changes files.

The runner uses `codex exec --sandbox workspace-write`, approval policy `never`,
`--ephemeral`, `--json`, and `--output-last-message`. It preserves configured model
selection unless `--model` is supplied. Configure authentication through Codex
normally; no separate API integration or key storage is added. Check
`codex exec --help` for your installed version and
[OpenAI's CLI documentation](https://learn.chatgpt.com/docs/developer-commands#codex-exec).
The implementation was checked against local CLI 0.153.4; deprecated `--full-auto`
is not used. Account, model, network, and host policy still determine availability.

The runner adds the selected repository's exact Git metadata directories with
`--add-dir` so a keep can commit despite the default `.git` protection. A linked
worktree also needs its common Git directory for objects and refs; this metadata
is shared across that repository's worktrees. Use a separate clone for stronger
repository isolation. The grant does not include other source worktrees or the
home directory, and the runner rejects broad workspace/ancestor grants.

## Boundaries and recovery

- Defaults: at most 20 invocations, 3600 supervisor seconds, 900 seconds per turn.
  The segment's run/time/target limits can stop it earlier. It never extends them.
- Each invocation must append exactly one result. No result, extra results, altered
  config, changed harness/branch, dirty final workspace, or a CLI failure pauses
  with an error instead of retrying. It preserves partial work for reconciliation.
- The supervisor owns `.autoresearch-codex.lock`. A second supervisor fails without
  pausing the first. Other agents/Claude sessions must not share that worktree.
- `touch .autoresearch-off` pauses; during a turn the supervisor terminates its
  process group. Ctrl-C or SIGTERM does the same. A timeout can interrupt between benchmark,
  commit, and result logging: inspect git/state/logs before resuming.
- Logs and the final message are in `experiments/codex/run-N-*/`. No assistant prose
  is parsed as a score. The JSONL ledger is the progress contract.
- A max-turn/time stop writes `.autoresearch-off`. Explicitly resume only after
  reviewing partial work and the remaining segment budget. Clearing the sentinel
  does not reset an expired segment budget.

The supervisor enforces scheduling and detects specified file changes after a
turn. It is not an adversarial security boundary: a workspace-capable agent can
still change its own outputs, and the supervisor does not independently rerun or
attest the score, git commit, correctness gate, or every off-limits file. Keep the
scorer outside editable scope, inspect kept diffs, and use independent final
evaluation before trusting an optimization claim.

Linux/macOS and Python 3.10+ are supported; Windows requires a POSIX environment
such as WSL. Tests use a fake Codex CLI to exercise progress, errors, cancellation,
locking, and budget boundaries. Real authenticated model execution is a separate
integration check and must be reported separately from those tests.

The 2026-09-07 development-host check initially failed with
`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`. Loading Ubuntu's
dedicated Bubblewrap AppArmor profile resolved it while retaining the global
user-namespace restriction. With scoped Git metadata writes, the authenticated
rerun completed one keep/commit/log cycle and stopped at its run cap. Independent
checks confirmed the score, commit, unchanged harness/history, and clean worktree;
outside-workspace, `.codex`, and network restrictions remained enforced in sandbox
probes. See the [repair and verification report](../../../docs/codex-sandbox.md).
