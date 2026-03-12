# Autoresearch: Fastball Velocity Prediction

## Objective
Predict fastball velocity (`pitch_speed_mph`) from biomechanical Point-of-Interest (POI) metrics using the Driveline OpenBiomechanics dataset. The dataset has 411 fastball pitches from 100 players, each with 78 biomechanical features covering joint angles, velocities, moments, ground reaction forces, and energy flow metrics at key phases (foot plant, max external rotation, ball release).

We optimize cross-validated R² using GroupKFold (grouped by player/session) to prevent data leakage — the model must generalize to unseen players.

## Metrics
- **Primary**: r2 (unitless, higher is better) — cross-validated R² score
- **Secondary**: rmse (mph) — cross-validated root mean squared error

## How to Run
`.venv/bin/python train.py` — outputs `METRIC name=number` lines.

## Files in Scope
- `train.py` — main training script; feature engineering, model config, CV evaluation, visualization

## Off Limits
- `third_party/` — raw data, do not modify
- `skills/`, `commands/`, `hooks/` — autoresearch infrastructure
- `.venv/` — Python environment

## Constraints
- Must use GroupKFold on `session` column (player-level splits) — no data leakage
- Must produce reproducible results (fixed random seeds)
- Script must output `METRIC r2=X.XXXX` and `METRIC rmse=X.XXXX` lines
- Must generate visualization plots to `plots/` directory on each run
- No new pip dependencies beyond what's installed (xgboost, scikit-learn, pandas, numpy, matplotlib)

## Data Summary
- 411 rows (all fastballs), 100 unique players (~4 pitches each)
- Target: `pitch_speed_mph` (range 69.5–94.4 mph, mean 84.7)
- Features: 78 biomechanical columns (columns 6–81 in the CSV)
- Categorical: `p_throws` (R/L) — needs encoding
- ID columns (drop): `session_pitch`, `session`, `pitch_type`

## What's Been Tried
(none yet — starting from baseline)
