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

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install xgboost scikit-learn pandas numpy matplotlib
```

## Metrics
- **Primary**: r2 (unitless, higher is better) -- cross-validated R² score
- **Secondary**: rmse (mph) -- cross-validated root mean squared error

## How to Run
`./autoresearch.sh` -- outputs `METRIC name=number` lines.

Or directly: `.venv/bin/python train.py`

## Files in Scope
- `train.py` -- main training script; feature engineering, model config, CV evaluation, visualization

## Off Limits
- `third_party/` -- raw data, do not modify
- `skills/`, `commands/`, `hooks/` -- autoresearch infrastructure
- `.venv/` -- Python environment

## Constraints
- Must use LeaveOneGroupOut on `session` column (player-level splits) -- no data leakage
- Must produce reproducible results (fixed random seeds)
- Script must output `METRIC r2=X.XXXX` and `METRIC rmse=X.XXXX` lines
- Must generate visualization plots to `plots/` directory on each run
- No new pip dependencies beyond what's installed (xgboost, scikit-learn, pandas, numpy, matplotlib)

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
