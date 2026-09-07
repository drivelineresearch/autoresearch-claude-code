# Autoresearch: fastball velocity prediction

## Objective and evidence boundary

Predict `pitch_speed_mph` from biomechanical Point-of-Interest (POI) metrics in the
[Driveline OpenBiomechanics dataset](https://github.com/drivelineresearch/openbiomechanics).
The published pitching release contains 411 fastball trials from 100 athletes.
These are contemporaneous movement measurements: this example measures predictive
association, not whether changing a mechanical feature causes a velocity gain.

Optimize pooled out-of-fold R² under a **fixed athlete-disjoint evaluation**.
The athlete key is `user` from `baseball_pitching/data/metadata.csv`, joined to POI
rows using the unique `session_pitch` identifier. `session` alone is not a durable
athlete identifier. See the upstream [datasheet](https://github.com/drivelineresearch/openbiomechanics/blob/main/DATASHEET.md).

## Setup and execution

Run the example in place; keep `pyproject.toml`, the runner, `train.py`,
`models.py`, and `candidate.py` in the same directory. From the repository root:

```bash
cd examples
mkdir -p third_party
gh repo clone drivelineresearch/openbiomechanics third_party/openbiomechanics -- --depth 1
uv sync
# Record the actual dataset revision and environment for the experiment log.
git -C third_party/openbiomechanics rev-parse HEAD
uv pip freeze
./autoresearch.sh 42
```

The clone is a one-time step; do not run it over an existing checkout. The example
uses only the small in-repository CSVs and needs no raw capture/media downloads.
Install [uv](https://docs.astral.sh/uv/) first if necessary. Optional backends:

```bash
uv sync --extra torch       # PyTorch wrappers
uv sync --extra boost       # CatBoost and LightGBM
uv sync --extra tabpfn      # pretrained TabPFN backend
uv sync --extra tabnet      # TabNet
uv sync --extra all         # every optional backend; can be a large installation
```

`uv sync` selects the requested extras for that environment; include the extras
you intend to keep each time you sync. Optional framework installation does not
prove GPU compatibility or availability of model weights. In particular, TabPFN
may require a separate model download, network access, and applicable model access
terms when first fitted. Those backends have not all been runtime-tested here.

From the repository root, `./examples/autoresearch.sh 42` also works. The runner
changes to its own directory and uses one interpreter for checks and training.
For direct execution use `uv run --project examples python examples/train.py`.
By default data and plots are located relative to `examples/train.py`, independent
of the caller's working directory. For an existing dataset checkout, set absolute
`AR_DATA_PATH` and `AR_METADATA_PATH`; `AR_PLOT_DIR` overrides the output directory.

For the repository's research loop, create a root `autoresearch.sh` that executes
`./examples/autoresearch.sh "$@"`, or run the research loop from `examples/` with
this document copied to `autoresearch.md`. Keep the same working directory for
all runs in a session and ensure the session's declared runner path matches it.

## Fixed evaluation contract

- Primary output: `METRIC r2=number`, pooled across all out-of-fold predictions;
  higher is better. Secondary: `METRIC rmse=number`, in mph; lower is better.
- Leave-one-athlete-out CV is the default. Multiple sessions from the same athlete
  remain in the same fold. Never compute a mean of singleton-fold R² values.
- `AGGREGATE_TO_PLAYER=True` predicts each athlete's mean velocity from their mean
  mechanics. This is a different prediction task from individual-pitch velocity.
  Freeze that choice for a research session; do not compare both scores as if the
  task were unchanged.
- Supervised feature selection is fitted separately on each outer training fold.
  Standardization is fitted inside each model pipeline on that training fold.
- Outer held-out targets are never passed to fitting or early stopping. Default
  models use fixed iteration/epoch budgets. Adding early stopping requires a
  separate inner athlete split and preprocessing fitted on its training subset.
- Stacking uses athlete-aware inner CV. It needs at least three athlete groups;
  small training sets may also require reducing KNN's `n_neighbors`.
- Missing athlete mappings, ambiguous pitch IDs, non-fastballs, unknown handedness,
  missing/nonfinite features, and invalid targets fail with explicit errors.
  If a different dataset needs imputation, fit it within training folds.
- `AR_SEED` (or the runner's positional seed) reaches selection and every exposed
  estimator seed, including pipeline/stacking children. Repeat seeds measure
  stochastic training variation, not dataset uncertainty or protection against
  adaptive overfitting. GPU operations can still have backend-specific variance.
- Keep data revision, grouping, aggregation, outer splits, metrics, and target fixed
  while comparing candidates. Repeatedly optimizing this CV score makes it a
  **development score**. A final untouched cohort or nested evaluation of the whole
  search procedure is needed before claiming generalization performance.

## Files and allowed changes

For an experiment session, restrict candidate edits to model/feature proposals:

- `candidate.py`: `MODEL_TYPE`, `MODEL_PARAMS`, `TOP_N_FEATURES`, and row-local feature
  formulas. `TOP_N_FEATURES=None` disables supervised selection. Selection otherwise
  uses a fixed-budget XGBoost ranker trained only on the outer training subset.
- `models.py`: registered estimators and their hyperparameters.

Keep `train.py` (evaluation, target/identifier exclusion, and metadata joins),
raw `third_party/` data, runner, and dependency environment fixed
within that session. Protect `train.py` and the runner with the research tool's
`--protect` options; candidate edits belong in `candidate.py` and `models.py`. A
deliberate evaluation change starts a new baseline/session.
Do not change `skills/`, `commands/`, `hooks/`, or `.venv/` as an experiment candidate.
If dependencies must change, declare and resolve that change before comparing runs.

## Model registry and practical choices

There are 19 registered models. Dependency discovery is available without fitting
models or fetching weights:

```bash
uv run python -c 'from models import print_model_table; print_model_table()'
```

| Category | Model names | Useful parameters / limits |
| --- | --- | --- |
| Boosting | `xgboost`, `catboost`, `lightgbm`, `histgb` | Depth, iteration count, learning rate, regularization. |
| Neural | `pytorch_mlp`, `mc_dropout`, `ft_transformer`, `mlp` | Architecture, dropout, learning rate, epoch budget. PyTorch MLPs use layer normalization, which supports singleton batches. |
| Other neural | `tabpfn`, `tabnet` | TabPFN needs pretrained weights. TabNet supports `max_epochs`, `batch_size`, `virtual_batch_size`, and `optimizer_params`. |
| Linear | `ridge`, `elasticnet`, `lasso`, `huber` | Regularization; inexpensive baselines. |
| Bayesian | `bayesian_ridge`, `gp` | BayesianRidge uses `max_iter`. GP scales cubically in sample count. |
| Other | `svr`, `knn` | Kernel/regularization or neighbors; KNN needs enough training rows. |
| Ensemble | `stacking` | `passthrough`, `n_jobs`; group-aware inner folds. |

Begin with a linear baseline and a modest boosting budget, then use the same
locked evaluation for more expensive models. CPU execution is sufficient for this
small dataset. `AR_DEVICE=cpu` disables CUDA auto-detection for supporting builders.
LightGBM stays on CPU unless its device is explicitly configured after verifying
that installation's GPU support. Override other backend device settings in
`MODEL_PARAMS` when needed; PyTorch's CUDA availability does not validate them.

Four plots are written to `plots/` per run. Native model importance and held-out
permutation MSE increases have different scales. With singleton validation folds,
permutation importance cannot be estimated; the plot explains its unavailability.
MC Dropout, GP, and BayesianRidge can produce an additional uncertainty-versus-error
plot. This diagnostic is descriptive and does not establish calibrated intervals.

## Historical results are not the corrected baseline

The historical **R²=0.783 / RMSE=2.20 mph** in
[`experiments/worklog.md`](../experiments/worklog.md) used global supervised feature
selection and the outer held-out target for early stopping. It also compared
changes to aggregation and fold strategy. Those scores are not an unbiased estimate
of performance, and differences do not isolate model improvements. The current
example corrects those evaluation leaks; a new baseline must be measured before
making performance claims. The old record remains a historical experiment log.

Fast, synthetic regression checks for the evaluator, data joins, wrappers, seed
propagation, plotting, and runner can be run from the repository root:

```bash
uv run --project examples python -m unittest discover -s tests -p test_examples.py -v
```

These checks do not download the dataset, train optional GPU models, or certify
all 19 model backends. Preserve the dataset revision, dependency snapshot, candidate
commit, metric output, and run duration with each real experiment.
