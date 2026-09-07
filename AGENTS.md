# Contributor guidance

This repository supports a shared autoresearch skill for Claude Code and Codex.
Reviewing or modifying this plugin does not authorize starting an experiment loop.

- `skills/autoresearch/SKILL.md` defines workflow and scope; `references/state.md`
  documents the schema. Update both when behavior changes.
- `skills/autoresearch/scripts/ar_state.py` owns state validation, budgets,
  transactional logging, and Claude hook behavior. Shell hooks resolve their
  actual path (including manual-install symlinks) before dispatching here.
- `skills/autoresearch/scripts/codex_loop.py` supervises already initialized
  experiments through `codex exec`. Keep continuation bounded; never enable
  sandbox/approval bypasses as an installation default.
- `scripts/install.py` owns installation. Preserve foreign files, links, settings,
  and exact hook ownership. Test only with fixture homes, not real user config.
- Experiment cleanup/staging must use explicit file scope. Do not introduce
  blanket git cleanup or overwrite experiment history on resume.
- Hook stdout must match the Claude event's output contract. Diagnostics go to
  stderr. Invalid state must not trap the agent in forced continuation.
- Use `uv` for example package management. Keep data/scoring/splits separate from
  candidate changes; no outer-fold labels in training, scaling, or selection.
- Run `python3 -m unittest discover -s tests -v` and ShellCheck after runtime
  changes. Scientific example tests may skip without deps; report that separately.
- Preserve ignored `brain/`, runtime state, and unrelated work. Avoid running
  experiments, making network/model calls, or modifying personal installs as a
  side effect of tests. Tests use temporary repositories and fake clients.

Layout: `commands/` and `hooks/hooks.json` are Claude-specific adapters;
`skills/autoresearch/` is a self-contained shared skill; `examples/` is an optional
ML workload; `docs/review.md` records review evidence and remaining work.
