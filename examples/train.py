#!/usr/bin/env python3
"""Fastball velocity prediction from biomechanical POI metrics."""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

SEED = 42
DATA_PATH = "third_party/openbiomechanics/baseball_pitching/data/poi/poi_metrics.csv"
PLOT_DIR = "plots"
N_FOLDS = 5

# --- Config ---
XGB_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": SEED,
    "early_stopping_rounds": 50,
}

DROP_COLS = ["session_pitch", "session", "pitch_type", "pitch_speed_mph"]
TARGET = "pitch_speed_mph"
GROUP_COL = "session"


AGGREGATE_TO_PLAYER = True


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Encode p_throws (R/L) as numeric
    le = LabelEncoder()
    df["p_throws"] = le.fit_transform(df["p_throws"])

    if AGGREGATE_TO_PLAYER:
        # Aggregate to player-level means — reduces noise from pitch-level variation
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != GROUP_COL]
        agg_df = df.groupby(GROUP_COL)[numeric_cols].mean().reset_index()
        # Also add std dev of key features as consistency measures
        std_cols = ["pitch_speed_mph", "elbow_transfer_fp_br", "shoulder_transfer_fp_br",
                    "thorax_distal_transfer_fp_br"]
        for col in std_cols:
            if col in df.columns:
                std_series = df.groupby(GROUP_COL)[col].std().fillna(0)
                agg_df[f"{col}_std"] = agg_df[GROUP_COL].map(std_series)
        df = agg_df

    groups = df[GROUP_COL]
    y = df[TARGET].values
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop)

    # Feature engineering: kinetic chain ratios
    eps = 1e-6
    X["thorax_to_elbow_transfer_ratio"] = X["thorax_distal_transfer_fp_br"] / (X["elbow_transfer_fp_br"] + eps)
    X["shoulder_to_elbow_transfer_ratio"] = X["shoulder_transfer_fp_br"] / (X["elbow_transfer_fp_br"] + eps)
    X["pelvis_to_thorax_transfer_ratio"] = X["pelvis_lumbar_transfer_fp_br"] / (X["thorax_distal_transfer_fp_br"] + eps)
    # Rotational velocity ratios
    X["torso_to_pelvis_rot_ratio"] = X["max_torso_rotational_velo"] / (X["max_pelvis_rotational_velo"] + eps)
    # Total energy transfer (kinetic chain sum)
    X["total_energy_transfer"] = (X["shoulder_transfer_fp_br"] + X["elbow_transfer_fp_br"] +
                                   X["thorax_distal_transfer_fp_br"] + X["pelvis_lumbar_transfer_fp_br"])
    # GRF asymmetry
    X["grf_lead_rear_ratio"] = X["lead_grf_mag_max"] / (X["rear_grf_mag_max"] + eps)
    # Shoulder-elbow moment ratio
    X["moment_ratio"] = X["shoulder_internal_rotation_moment"] / (X["elbow_varus_moment"] + eps)

    return X, y, groups


TOP_N_FEATURES = 15  # Keep top N features from first-pass importance


def select_features(X, y, groups):
    """First pass: quick model to rank features by importance."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_importances = []
    quick_params = {**XGB_PARAMS, "n_estimators": 200, "early_stopping_rounds": 20}

    for train_idx, val_idx in gkf.split(X, y, groups):
        model = xgb.XGBRegressor(**quick_params)
        model.fit(X.iloc[train_idx], y[train_idx],
                  eval_set=[(X.iloc[val_idx], y[val_idx])], verbose=False)
        fold_importances.append(
            pd.Series(model.feature_importances_, index=X.columns)
        )

    avg_imp = pd.concat(fold_importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return avg_imp.head(TOP_N_FEATURES).index.tolist(), avg_imp


USE_LOGO = True  # LeaveOneGroupOut for player-level data


def cross_validate(X, y, groups):
    # Feature selection first pass
    top_features, full_importance = select_features(X, y, groups)
    X_selected = X[top_features]

    if USE_LOGO:
        cv = LeaveOneGroupOut()
    else:
        cv = GroupKFold(n_splits=N_FOLDS)

    oof_preds = np.zeros(len(y))
    fold_importances = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y, groups)):
        X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        oof_preds[val_idx] = model.predict(X_val)
        fold_importances.append(
            pd.Series(model.feature_importances_, index=X_selected.columns)
        )

    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    r2 = r2_score(y, oof_preds)
    avg_importance = pd.concat(fold_importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return oof_preds, rmse, r2, avg_importance


def plot_results(y, oof_preds, importance, rmse, r2):
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1. Actual vs Predicted scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y, oof_preds, alpha=0.5, s=30, edgecolors="k", linewidth=0.5)
    lo, hi = min(y.min(), oof_preds.min()) - 1, max(y.max(), oof_preds.max()) + 1
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("Actual Velocity (mph)")
    ax.set_ylabel("Predicted Velocity (mph)")
    ax.set_title(f"Fastball Velocity: Actual vs Predicted\nR²={r2:.4f}  RMSE={rmse:.2f} mph")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    # 2. Residuals plot
    residuals = y - oof_preds
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(oof_preds, residuals, alpha=0.5, s=30, edgecolors="k", linewidth=0.5)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted Velocity (mph)")
    ax.set_ylabel("Residual (mph)")
    ax.set_title("Residuals vs Predicted")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/residuals.png", dpi=150)
    plt.close(fig)

    # 3. Feature importance (top 20)
    top = importance.head(20)
    fig, ax = plt.subplots(figsize=(8, 7))
    top.sort_values().plot.barh(ax=ax)
    ax.set_xlabel("Mean Feature Importance (gain)")
    ax.set_title("Top 20 Feature Importances")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/feature_importance.png", dpi=150)
    plt.close(fig)

    # 4. Residual histogram
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=30, edgecolor="k", alpha=0.7)
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Residual (mph)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution (std={np.std(residuals):.2f} mph)")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/residual_histogram.png", dpi=150)
    plt.close(fig)


def main():
    X, y, groups = load_data()
    oof_preds, rmse, r2, importance = cross_validate(X, y, groups)
    plot_results(y, oof_preds, importance, rmse, r2)

    # Output metrics in autoresearch format
    print(f"METRIC r2={r2:.6f}")
    print(f"METRIC rmse={rmse:.6f}")

    # Print top 10 features for context
    print("\nTop 10 features:")
    for feat, imp in importance.head(10).items():
        print(f"  {feat}: {imp:.4f}")


if __name__ == "__main__":
    main()
