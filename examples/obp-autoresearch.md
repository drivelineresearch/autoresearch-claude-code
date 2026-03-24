# Autoresearch: Fastball Velocity Prediction

## Objective
Predict fastball velocity (`pitch_speed_mph`) from biomechanical Point-of-Interest (POI) metrics using the [Driveline OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics) dataset. The dataset has 411 fastball pitches from 100 players, each with 78 biomechanical features covering joint angles, velocities, moments, ground reaction forces, and energy flow metrics at key phases (foot plant, max external rotation, ball release).

We optimize cross-validated R² using LeaveOneGroupOut (grouped by player/session) to prevent data leakage -- the model must generalize to unseen players.

## Prerequisites

Clone the OpenBiomechanics dataset into `third_party/`:

```bash
mkdir -p third_party
git clone https://github.com/drivelineresearch/openbiomechanics.git third_party/openbiomechanics
```

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                    # core deps (xgboost, sklearn, rich, etc.)
uv sync --extra torch      # + PyTorch/CUDA models
uv sync --extra boost      # + CatBoost, LightGBM
uv sync --extra all        # everything (torch, catboost, lightgbm, tabpfn, tabnet)
```

## Metrics
- **Primary**: r2 (unitless, higher is better) -- cross-validated R² score
- **Secondary**: rmse (mph) -- cross-validated root mean squared error

## How to Run
`./autoresearch.sh` -- outputs `METRIC name=number` lines.

Or directly: `uv run python train.py`

## Files in Scope
- `train.py` -- main training script; config constants, data loading, CV evaluation, visualization
  - `MODEL_TYPE` config: swap between any registered model (see models.py for full list)
  - `MODEL_PARAMS` dict: override default hyperparameters for the selected model
- `models.py` -- model registry; browse this to discover available models and tunable parameters
  - Categories: boosting, linear, neural, ensemble, bayesian, other
  - Each model has metadata: GPU support, scaling needs, importance type
- `pyproject.toml` -- uv project config with dependency groups

## Off Limits
- `third_party/` -- raw data, do not modify
- `skills/`, `commands/`, `hooks/` -- autoresearch infrastructure
- `.venv/` -- managed by uv

## Constraints
- Must use LeaveOneGroupOut on `session` column (player-level splits) -- no data leakage
- Must produce reproducible results (fixed random seeds)
- Script must output `METRIC r2=X.XXXX` and `METRIC rmse=X.XXXX` lines to stdout
- Must generate visualization plots to `plots/` directory on each run
- Core dependencies (always available): xgboost, scikit-learn, pandas, numpy, matplotlib, rich
- Optional dependencies (install for specific models): torch, catboost, lightgbm, tabpfn, pytorch-tabnet
- GPU acceleration available when CUDA is present (auto-detected by models.py)

## Data Summary
- 411 rows (all fastballs), 100 unique players (~4 pitches each)
- Target: `pitch_speed_mph` (range 69.5-94.4 mph, mean 84.7)
- Features: 78 biomechanical columns (columns 6-81 in the CSV)
- Categorical: `p_throws` (R/L) -- needs encoding
- ID columns (drop): `session_pitch`, `session`, `pitch_type`

## Current Best
- **R²=0.783, RMSE=2.20 mph** after 22 experiments
- Architecture: Player-level aggregation + LeaveOneGroupOut CV + two-pass feature selection (top 15) + XGBoost with early stopping
- See `experiments/worklog.md` for full experiment history

## What's Been Tried
See `experiments/worklog.md` for a detailed narrative of all 22 experiments, including what worked, what failed, and why. Key findings:

1. Feature selection (top 15 from importance-based two-pass) was the single biggest win
2. Player-level aggregation (mean metrics per player) removes within-player noise
3. LeaveOneGroupOut CV (100-fold) dramatically outperforms 5-fold GroupKFold
4. Energy transfer features dominate: elbow, thorax distal, and shoulder transfer (foot plant to ball release)
5. Hyperparameter tuning, ensemble approaches, and alternative boosters gave diminishing or negative returns

### Model zoo (19 models available)

#### Boosting (best for tabular data)
- `MODEL_TYPE="xgboost"` -- Current champion. Tune max_depth, learning_rate, reg_alpha, reg_lambda.
- `MODEL_TYPE="catboost"` -- CatBoost. Try depth, l2_leaf_reg. Supports GPU.
- `MODEL_TYPE="lightgbm"` -- LightGBM. Fastest. Try num_leaves, min_child_samples.
- `MODEL_TYPE="histgb"` -- sklearn HistGradientBoosting. No extra deps. Try max_depth, l2_regularization.

#### Neural networks
- `MODEL_TYPE="pytorch_mlp"` -- PyTorch MLP with dropout, batch norm, AdamW, early stopping. Tune hidden_dims, dropout, lr, weight_decay. CUDA-accelerated.
- `MODEL_TYPE="ft_transformer"` -- Feature Tokenizer + Transformer. Attention-based. Tune d_model, n_heads, n_layers.
- `MODEL_TYPE="tabpfn"` -- Pretrained transformer foundation model (Nature 2025). Zero-shot, no training. Try n_estimators.
- `MODEL_TYPE="tabnet"` -- Attention-based feature selection NN. Tune n_d, n_a, n_steps.
- `MODEL_TYPE="mc_dropout"` -- MC Dropout uncertainty quantification. Same arch as pytorch_mlp + reports prediction uncertainty.
- `MODEL_TYPE="mlp"` -- sklearn MLPRegressor (simpler, no CUDA).

#### Linear / regularized
- `MODEL_TYPE="ridge"` -- L2 regularization. If close to XGBoost, relationship is mostly linear.
- `MODEL_TYPE="elasticnet"` -- L1+L2. L1 zeros out features -- compare vs XGBoost importance.
- `MODEL_TYPE="lasso"` -- L1 only (sparse).
- `MODEL_TYPE="huber"` -- Robust to outliers.

#### Bayesian / probabilistic
- `MODEL_TYPE="bayesian_ridge"` -- Automatic regularization + uncertainty estimates.
- `MODEL_TYPE="gp"` -- Gaussian Process. Full Bayesian uncertainty. Ideal for N=100.

#### Other
- `MODEL_TYPE="svr"` -- Support vector regression (RBF kernel). Tune C, epsilon.
- `MODEL_TYPE="knn"` -- k-nearest neighbors. Tune n_neighbors, weights.

#### Ensemble
- `MODEL_TYPE="stacking"` -- Stacking with diverse base models + Ridge meta-learner. Group-aware inner CV.

### Suggested research strategy
1. **Boosting variants** (catboost, lightgbm, histgb) -- likely competitive with xgboost baseline
2. **TabPFN** -- zero-shot foundation model, could surprise on small data
3. **Stacking ensemble** -- combines diverse model strengths
4. **Neural networks** (pytorch_mlp, ft_transformer) with hyperparameter search
5. **GP and bayesian_ridge** for uncertainty quantification insights
6. **Linear models** as baselines to measure nonlinearity contribution
