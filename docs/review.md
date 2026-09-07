# Review: Codex support and general hardening

Reviewed 2026-09-07 against original commit `66ca4f5` on
`codex/support-and-hardening`. The original local checkout was clean and matched
`origin/master`; existing ignored `brain/` and other artifacts were preserved.

**Decision:** keep one shared experiment skill/state format, retain Claude's hook
adapter, and add a bounded Codex CLI supervisor. The initial implementation was
mostly instructions around permissive shell hooks. Its largest risks were unsafe
git/filesystem operations and unreliable evaluation, not missing model APIs.

## 1. Findings and changes

| Priority | Original finding and consequence | Implemented change |
|---|---|---|
| High | Skill instructed `git add -A`, whole-tree checkout, and `git clean -fd`; unrelated changes/untracked work could be committed or deleted. | Explicit file scope, clean experiment worktree, reviewed index, scoped restore, and preserve partial work on uncertainty. |
| High | Uninstall recursively removed any same-named skill directory and broadly matched hook-command substrings. | Only owned links and exact event/command registrations are removed; foreign files and mixed hooks remain. |
| High | Installation mutated links before parsing settings, wrote settings in place, and could leave partial configuration. | Full preflight, atomic settings replacement, original-byte backup, mode preservation, and rollback for operation errors. |
| High | Hooks silently ignored corrupt JSON rows; invalid budgets/types could error or produce endless continuation. | One strict parser for all clients; malformed state disables automatic continuation with a diagnostic. Finite numbers, statuses, segments, numbering, and parent references are checked. |
| High | Failed/missing payload `cwd` fell back to an unrelated process directory; context hook ignored payload altogether. | All hooks validate an absolute payload workspace and never substitute process cwd. |
| High | Example selected supervised features globally and supplied outer held-out labels to early stopping. | Feature ranking is training-fold-local; outer labels never enter fitting. Models use fixed budgets; early stopping needs an inner split. |
| High | Historical headline compared different aggregation/CV protocols and treated removal of leakage as a failed candidate. | Headline claim withdrawn as validated evidence; original narrative preserved with a clear methodology caveat. |
| Medium | `session` was treated as athlete identity, unsafe for repeated sessions. | Join metadata `user` by unique `session_pitch`; validate mappings and group/aggregate by athlete. |
| Medium | jq/Python logger paths had inconsistent coercion, accepted nonfinite/invalid values, and raced on appends/run numbers. | Python standard-library writer, bounded advisory lock, validation before atomic publication, and `auto` run/segment allocation. |
| Medium | Hook counts depended on JSON whitespace via grep. | Parsed current/global counts and best kept result, independent of formatting. |
| Medium | Example mixed tunable configuration with metric-emitting evaluation; locking the whole evaluator prevented normal candidate edits. | Candidate model/feature proposals are separated into `examples/candidate.py`; evaluation remains in `train.py`. |
| Medium | Model wrappers had cloning/API/shape issues; GPU detection did not establish every backend's capability. | Seed propagation, wrapper fixes, explicit device choices, current core CPU API checks, and honest optional-backend limits. |
| Medium | Shared temporary log, stdout/stderr mixing, unbounded output reads, and GNU-only timing snippets made experiment instructions unreliable. | Unique bounded log guidance, stdout-only metrics, explicit exit/finite checks, portable timing guidance. |
| Medium | README copied files outside their uv project and overstated locked-harness, budget, and accuracy guarantees. | Run-in-place setup, declared dependencies/platforms, mechanism-versus-instruction table, corrected result provenance. |
| Medium | No automated regression suite or CI. | Temporary-workspace tests for state, hooks, installer, runner, and evaluation; Linux/macOS CI and scientific-dependency job. |

The review covered every tracked text/code surface: shared skill, command,
install/uninstall, hook scripts/registration, plugin metadata, example code/config,
README/contributor docs, historical worklog, ignore rules, and diagram prompts.
Images/plots were assessed as illustrative historical assets, not re-created or
used as validation evidence. License text remains intact.

## 2. Codex integration

The shared skill is discovered through `~/.agents/skills/autoresearch`. The
installer links the complete directory, including helpers/references. Claude's
hook file is not treated as a Codex hook manifest. Current sources:
[OpenAI skill discovery](https://learn.chatgpt.com/docs/build-skills) and
[Codex noninteractive CLI](https://learn.chatgpt.com/docs/developer-commands#codex-exec).
Installed CLI help was checked at version **0.153.4**; no model version is pinned.

The supervisor requires an initialized, clean experiment branch. Each fresh Codex
invocation reads durable state, performs one experiment, and ends. It checks:

- Independent finite invocation and wall-time budgets, plus segment budgets/target.
- Exactly one result appended while preserving earlier history and config.
- Protected harness hashes, unchanged branch, and clean final workspace.
- Single-supervisor lock, process timeout/cancellation, failure status and logs.

It uses the workspace-write sandbox with exact repository Git metadata write
roots for commits, and never enables a blanket sandbox bypass.
`--status` and `--dry-run` provide inspection before model execution. See the
[Codex operation guide](../skills/autoresearch/references/codex.md).

## 3. Compatibility and migration

Existing valid numeric JSONL sessions preserve their logical history. Legacy
headers without budget/noise fields receive documented defaults (200 runs and zero
floor). State previously accepted by the permissive logger may now be rejected:
string-valued secondary metrics, NaN/Infinity, malformed/truncated lines, invalid
hashes, wrong segments, nonsequential runs, or invalid parent references.

Run `ar_state.py status --path FILE` before resuming an older session. Preserve the
original and reconcile invalid rows from benchmark logs/commits; do not drop rows
or coerce unknown observations merely to satisfy validation. New helper writes
never silently reinitialize an existing ledger. An authorized additional budget
can use an explicit new segment and verified baseline with the same fixed scorer;
it must be recorded as an extension, not an independent new discovery.

Manual copies in user config directories remain intact during uninstall. To
replace a foreign/manual copy, inspect and move it yourself first. Symlinked
`settings.json` is refused because replacing it changes which file the client
reads. Choose one Claude installation mechanism to avoid duplicate hooks. Do not
run multiple clients against one experiment worktree.

Core runtime now consistently requires Python 3.10+ rather than claiming a jq-only
path. POSIX advisory locks/process groups require Linux/macOS or WSL. CPU example
checks do not guarantee optional GPU/model package behavior.

## 4. Validation

Final combined validation: **94 unique tests passed, no skips, in 27.872 seconds**:
33 state/hook tests, 22 installer tests, 17 Codex-supervisor tests, and 22 example
tests. The scientific test environment used CPU XGBoost 3.4.1 and scikit-learn
1.8.0, with no GPU/model-weight or dataset download. A synthetic full example
entrypoint also emitted two finite metrics and generated four expected plots.
That synthetic test is an execution check, not an OpenBiomechanics accuracy result.

ShellCheck, Python AST parsing, JSON manifests, the skill frontmatter validator,
local documentation links, and `git diff --check` passed. Regression tests use
isolated temporary homes/workspaces and fake clients; they never install into real
personal configuration or perform a production experiment. Tests check observable
safety/correctness outcomes, including concurrent append races, rollback under
injected failures, held-out label isolation, and child cancellation. Reproduce
core checks with `python3 -m unittest discover -s tests -v`; use the documented
example uv environment to exercise scientific tests rather than skip them.

GitHub Actions is configured with immutable action revisions and read-only
repository permissions. CI execution itself requires publishing this branch.

The first authenticated Codex CLI 0.153.4 check failed before commands with
`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`; the runner correctly
paused without modifying the candidate or retrying. A subsequent repair loaded
Ubuntu's missing Bubblewrap-specific AppArmor profile while retaining the global
user-namespace restriction. Native sandbox probes then exposed and verified the
need for scoped Git metadata writes, now handled by the runner's `--add-dir` flags.

The authenticated rerun **passed the complete keep/commit/log cycle**: exactly one
candidate commit and one new kept result, benchmark/check success, preserved
scorer and ledger history, updated dashboard/worklog, clean worktree, and automatic
pause at 2/2 logged runs. Independent benchmark/check reruns confirmed the result.
Standalone and linked-worktree sandbox probes also verified that metadata grants
leave other workspace, `.codex`, and network boundaries in place. This is a
synthetic integration fixture, not an ML performance result. Exact host repair,
evidence, scope, and rollback are in [codex-sandbox.md](codex-sandbox.md).

## 5. Remaining work, in order

1. **Independent score verification.** State validation proves well-formed records,
   not actual benchmark execution, correctness checks, noise-clearing decisions,
   or commit provenance. A future evaluator should independently run/attest the
   scorer and finalize keep/discard. Current hashes detect specified changes
   between turns; they are not an adversarial boundary.
2. **External-resource cancellation.** Process-group cleanup covers ordinary child
   trees. Deliberately detached sessions, remote jobs, and external side effects
   need workload-specific cancellation/reconciliation. No remote job is retried
   automatically by this implementation.
3. **Claude no-progress watchdog.** Claude hooks check at turn boundaries. They
   cannot impose per-process timeouts or independently prove each blocked turn
   progressed; the skill requires benchmark timeout and pausing on blockers.
4. **Reproducible scientific baseline.** Rerun the corrected example with pinned
   dataset/environment, fixed evaluation, raw seed measurements, and a final
   untouched cohort. Do not revive the historical 0.783 claim without new evidence.
5. **Optional backend matrix.** Add separately provisioned tests for Torch,
   CatBoost, LightGBM, TabNet, TabPFN/model access, and GPU builds as those backends
   are actually used. Dependency probing alone does not validate an installation.
6. **Release and installation concurrency.** Review schema-tightening as a release
   compatibility change. Installer rollback handles operation failures, but the
   multi-file operation is not power-loss atomic and has no cross-process installer
   lock. Publish/version only after these documented tradeoffs are accepted.

The branch makes current behavior reviewable without presenting these remaining
items as completed work or claiming a new research result.
