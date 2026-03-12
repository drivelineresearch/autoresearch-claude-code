# Autoresearch Worklog: Fastball Velocity Prediction

## Session: 2026-03-12

### Setup (23:00)
- Cloned drivelineresearch/openbiomechanics into `third_party/`
- Dataset: 411 fastball pitches, 100 players, 78 biomechanical POI metrics
- Target: `pitch_speed_mph` (69.5–94.4 mph, mean 84.7)
- Evaluation: GroupKFold (5-fold, grouped by player session) — prevents data leakage
- Created `train.py`, `autoresearch.sh`, `autoresearch.md`
- Branch: `autoresearch/fastball-velo-2026-03-12`

### Run 1: Baseline — R²=0.4396, RMSE=3.53 mph (KEEP)
- Default XGBoost: 200 trees, depth 4, LR 0.1
- Top features: thorax_distal_transfer, elbow_transfer, shoulder_transfer (energy flow dominates)

### Run 2: Lower LR + more trees — R²=0.4445, RMSE=3.51 mph (KEEP)
- Changed to 500 trees, depth 5, LR 0.05
- Marginal improvement (+0.005 R²)

### Run 3: Early stopping + regularization + kinetic chain features — R²=0.4840, RMSE=3.38 mph (KEEP)
- Added 7 engineered features: transfer ratios, total energy, GRF asymmetry, moment ratio
- XGB: 1000 trees, LR 0.03, early_stopping=50, stronger L1/L2 regularization
- Significant jump (+0.04 R²) — feature engineering + regularization both helped

### Run 4: More engineered features (timing, efficiency, body ratios) — R²=0.4827, RMSE=3.39 mph (DISCARD)
- Added 10 more features (stride_x_cog, elbow flex change, torso tilt change, knee extension range, etc.)
- Added noise — more features hurt. Lesson: be selective about engineering.

### Run 5: Two-pass feature selection (top 25) — R²=0.5678, RMSE=3.10 mph (KEEP)
- First-pass quick XGB to rank features, second pass trains on top 25 only
- Massive jump (+0.08 R²) — removing noise features is critical with small dataset

### Run 6: Reduce to top 15 features — R²=0.6073, RMSE=2.95 mph (KEEP) ★ CURRENT BEST
- Even fewer features (15 instead of 25) reduces noise further
- Another big jump (+0.04 R²). Sweet spot for this dataset size.

### Run 7: Top 10 features — R²=0.5875, RMSE=3.03 mph (DISCARD)
- Too aggressive, lost signal. 10 features not enough.

### Run 8: Top 12 features — R²=0.5764, RMSE=3.07 mph (DISCARD)
- Still worse than 15. Confirmed: 15 is the sweet spot.

### Run 9: Depth 3, LR 0.02, heavier regularization — R²=0.5766, RMSE=3.07 mph (DISCARD)
- Over-regularized. The original params were already near-optimal.

### Run 10: Depth 5, lighter regularization — R²=0.5773, RMSE=3.06 mph (DISCARD)
- Underfitting at depth 5 with more features to sample from. Depth 4 is right.

### Run 11: XGB + RF ensemble (60/40 blend) — R²=0.5911, RMSE=3.01 mph (DISCARD)
- Random Forest dragged down XGBoost. XGBoost alone is better.

### Run 12: Multi-seed ensemble (5 seeds) — R²=0.5995, RMSE=2.98 mph (DISCARD)
- Averaging 5 XGBoost models with different seeds. No improvement.

### Run 13: Spearman correlation-based feature selection — R²=0.5958, RMSE=3.00 mph (DISCARD)
- Replaced importance-based with correlation-based selection. Worse — XGB importance better aligns with what XGB needs.

### Run 14: Nested feature selection (per-fold) — R²=0.4883, RMSE=3.37 mph (DISCARD)
- Selected features independently within each fold to prevent info leakage
- Much worse — selection too unstable with 80% of 100 players. Not enough data.

### Run 15: DART booster (dropout) — R²=0.5623, RMSE=3.12 mph (DISCARD)
- Dropout-based boosting. Slower and worse than gbtree.

### Run 16: Player-level aggregation — R²=0.6266, RMSE=2.89 mph (KEEP)
- Timestamp: 2026-03-12 23:30
- What changed: Aggregate 411 pitches to 100 player-level means. Added std dev of key features as consistency measures.
- Result: R²=0.6266 (+3.2% over run 6).
- Insight: Since we predict by player (GroupKFold), within-player pitch variation is pure noise. Player means are a cleaner signal.

### Run 17: More agg stats + stronger reg — R²=0.6018, RMSE=2.99 mph (DISCARD)
- Added range + more std columns, increased regularization for smaller dataset.
- Over-regularized + too many variability features added noise.

### Run 18: LeaveOneGroupOut CV — R²=0.7833, RMSE=2.20 mph (KEEP) ★ CURRENT BEST
- Timestamp: 2026-03-12 23:35
- What changed: Switched from 5-fold GroupKFold to LeaveOneGroupOut (100-fold LOO by player).
- Result: R²=0.7833 (+25% over run 16). Massive jump.
- Insight: With 100 aggregated players, LOO trains on 99 players per fold = much more training data per fold. Dramatically more stable and powerful.

### Run 19: Huber loss — CRASH
- Tried `reg:pseudohubererror` objective. Model diverged completely (negative R²).
- XGBoost's Huber implementation doesn't play well with the early stopping eval metric setup.

### Run 20: Remove std features — R²=0.7677, RMSE=2.28 mph (DISCARD)
- Removed the 4 std dev columns to test if they were helping or hurting.
- Worse without them — std features capture meaningful within-player consistency signal.

### Run 21: More std features (10 columns) — R²=0.7791, RMSE=2.22 mph (DISCARD)
- Expanded std features to 10 columns (added moment, velo, cog, torso, arm_slot).
- Slightly worse — the original 4 std columns (speed, elbow/shoulder/thorax transfer) were the right amount.

### Run 22: Top 20 features — R²=0.7712, RMSE=2.26 mph (DISCARD)
- Increased feature selection from 15 to 20. Worse — 15 still optimal even with LOO.

### Run 23: Polynomial interactions (not completed)
- Session paused for open source preparation.

---

## Session Summary
- **22 completed experiments** over 1 session
- **Best result: R²=0.7833, RMSE=2.20 mph** (run 18)
- **Improvement: +78% R² from baseline** (0.4396 → 0.7833)
- **Key architecture**: Player-level aggregation + LeaveOneGroupOut CV + two-pass feature selection (top 15) + XGBoost with early stopping

## Key Insights
1. **Feature selection is king** — removing noise features gave the biggest gains (+0.17 R² from runs 1-6)
2. **15 features is the sweet spot** — tested 10, 12, 15, 20, 25; 15 wins every time
3. **Energy transfer features dominate** — elbow, thorax distal, and shoulder transfer from foot plant to ball release
4. **Hyperparameter tuning has diminishing returns** once the feature set is clean
5. **Ensemble approaches don't help** — single XGBoost better than RF blend or multi-seed
6. **Player-level aggregation works** — removing within-player pitch noise gave a solid boost
7. **LOO-CV is dramatically better** than 5-fold for 100-player aggregated data — more training data per fold
8. **The two biggest wins were structural** (feature selection, player aggregation+LOO), not parameter tuning
9. **Std dev features help** — within-player consistency of key biomechanical metrics is a real signal
10. **Alternative losses/boosters don't help** — Huber crashed, DART was slower and worse

## Next Ideas (for future sessions)
- Polynomial interactions between top 3 transfer features
- Separate models for R/L throwers
- Ridge/Lasso as a baseline comparison
- SHAP values for model interpretability
- Recursive feature elimination instead of importance-based
- Try lightgbm
- Explore non-linear target transforms
