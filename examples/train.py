#!/usr/bin/env python3
"""Fastball velocity prediction from biomechanical POI metrics.

Uses the autoresearch model zoo (models.py) — 19 models across 6 categories.
Change MODEL_TYPE and MODEL_PARAMS below to switch models.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, PredefinedSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import xgboost as xgb

from models import build_model, USE_GPU

# ---------------------------------------------------------------------------
# Rich TUI (graceful fallback to plain text)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich import box
    console = Console(stderr=True)
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def info(msg):
    """Print info to stderr (rich or plain)."""
    if HAS_RICH:
        console.print(msg)
    else:
        print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Config — the autoresearch agent modifies these
# ---------------------------------------------------------------------------

SEED = 42
DATA_PATH = "third_party/openbiomechanics/baseball_pitching/data/poi/poi_metrics.csv"
PLOT_DIR = "plots"
N_FOLDS = 5

# --- Model Selection ---
# See models.py for full list: xgboost, catboost, lightgbm, histgb,
# pytorch_mlp, mc_dropout, ft_transformer, tabpfn, tabnet, mlp,
# ridge, elasticnet, lasso, huber, bayesian_ridge, gp, svr, knn, stacking
MODEL_TYPE = "xgboost"
MODEL_PARAMS = {}  # override default hyperparameters; empty = use model defaults

DROP_COLS = ["session_pitch", "session", "pitch_type", "pitch_speed_mph"]
TARGET = "pitch_speed_mph"
GROUP_COL = "session"

AGGREGATE_TO_PLAYER = True
TOP_N_FEATURES = 15
USE_LOGO = True


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)
    le = LabelEncoder()
    df["p_throws"] = le.fit_transform(df["p_throws"])

    if AGGREGATE_TO_PLAYER:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != GROUP_COL]
        agg_df = df.groupby(GROUP_COL)[numeric_cols].mean().reset_index()
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
    X["torso_to_pelvis_rot_ratio"] = X["max_torso_rotational_velo"] / (X["max_pelvis_rotational_velo"] + eps)
    X["total_energy_transfer"] = (X["shoulder_transfer_fp_br"] + X["elbow_transfer_fp_br"] +
                                   X["thorax_distal_transfer_fp_br"] + X["pelvis_lumbar_transfer_fp_br"])
    X["grf_lead_rear_ratio"] = X["lead_grf_mag_max"] / (X["rear_grf_mag_max"] + eps)
    X["moment_ratio"] = X["shoulder_internal_rotation_moment"] / (X["elbow_varus_moment"] + eps)

    return X, y, groups


# ---------------------------------------------------------------------------
# Feature selection (always uses XGBoost for importance ranking)
# ---------------------------------------------------------------------------

def select_features(X, y, groups):
    """First pass: quick XGBoost to rank features by importance."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_importances = []
    quick_params = {
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.03,
        "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 5,
        "reg_alpha": 0.5, "reg_lambda": 2.0, "random_state": SEED,
        "early_stopping_rounds": 20,
    }

    for train_idx, val_idx in gkf.split(X, y, groups):
        model = xgb.XGBRegressor(**quick_params)
        model.fit(X.iloc[train_idx], y[train_idx],
                  eval_set=[(X.iloc[val_idx], y[val_idx])], verbose=False)
        fold_importances.append(
            pd.Series(model.feature_importances_, index=X.columns)
        )

    avg_imp = pd.concat(fold_importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return avg_imp.head(TOP_N_FEATURES).index.tolist(), avg_imp


# ---------------------------------------------------------------------------
# Cross-validation (model-agnostic)
# ---------------------------------------------------------------------------

def cross_validate(X, y, groups):
    top_features, full_importance = select_features(X, y, groups)
    X_selected = X[top_features]

    cv = LeaveOneGroupOut() if USE_LOGO else GroupKFold(n_splits=N_FOLDS)
    splits = list(cv.split(X_selected, y, groups))
    n_folds = len(splits)

    oof_preds = np.zeros(len(y))
    oof_uncertainties = np.zeros(len(y))
    fold_importances = []

    # Progress bar (rich) or plain counter
    if HAS_RICH:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        )
        task = progress.add_task(f"CV ({MODEL_TYPE})", total=n_folds)
        progress.start()
    else:
        progress = None

    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model, meta = build_model(MODEL_TYPE, MODEL_PARAMS)

        # --- Fit with model-appropriate arguments ---
        fit_kwargs = {}

        if meta.get("supports_eval_set"):
            if MODEL_TYPE == "xgboost":
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["verbose"] = False
            elif MODEL_TYPE == "catboost":
                fit_kwargs["eval_set"] = (X_val, y_val)
            elif MODEL_TYPE == "lightgbm":
                import lightgbm as lgb
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["callbacks"] = [lgb.early_stopping(50), lgb.log_evaluation(0)]
            elif MODEL_TYPE in ("pytorch_mlp", "mc_dropout", "ft_transformer"):
                # PyTorch wrappers: eval_set passed to inner pipeline step
                # Step names: mlp, mc, ftt (must match models.py Pipeline keys)
                step_name = {"pytorch_mlp": "mlp", "mc_dropout": "mc",
                             "ft_transformer": "ftt"}[MODEL_TYPE]
                fit_kwargs[f"{step_name}__eval_set"] = [(X_val, y_val)]
            elif MODEL_TYPE == "tabnet":
                fit_kwargs["tabnet__eval_set"] = [(X_val.values, y_val)]
                fit_kwargs["tabnet__eval_metric"] = ["rmse"]

        # Stacking: inject group-aware inner CV
        if meta.get("is_stacking"):
            inner_gkf = GroupKFold(n_splits=min(5, len(np.unique(groups.iloc[train_idx]))))
            test_fold = np.full(len(train_idx), -1)
            for i, (_, inner_val) in enumerate(inner_gkf.split(
                    X_train, y_train, groups.iloc[train_idx])):
                test_fold[inner_val] = i
            model.cv = PredefinedSplit(test_fold)

        model.fit(X_train, y_train, **fit_kwargs)
        oof_preds[val_idx] = model.predict(X_val)

        # --- Uncertainty ---
        if MODEL_TYPE == "mc_dropout":
            # Extract uncertainty from the inner MC model
            inner = model.named_steps.get("mc")
            if inner and hasattr(inner, "uncertainty_"):
                oof_uncertainties[val_idx] = inner.uncertainty_
        elif MODEL_TYPE == "gp":
            gp_step = model.named_steps.get("gp")
            scaler_step = model.named_steps.get("scaler")
            if gp_step and scaler_step:
                X_val_scaled = scaler_step.transform(X_val)
                _, std = gp_step.predict(X_val_scaled, return_std=True)
                oof_uncertainties[val_idx] = std

        # --- Feature importance ---
        if meta.get("has_native_importance"):
            imp_model = model
            if hasattr(model, "named_steps"):
                for step_name, step in model.named_steps.items():
                    if hasattr(step, "feature_importances_"):
                        imp_model = step
                        break
            fold_importances.append(
                pd.Series(imp_model.feature_importances_, index=X_selected.columns)
            )
        else:
            perm = permutation_importance(model, X_val, y_val,
                                          n_repeats=10, random_state=SEED)
            fold_importances.append(
                pd.Series(perm.importances_mean, index=X_selected.columns)
            )

        if progress:
            progress.update(task, advance=1)

    if progress:
        progress.stop()

    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    r2 = r2_score(y, oof_preds)
    avg_importance = pd.concat(fold_importances, axis=1).mean(axis=1).sort_values(ascending=False)

    has_uncertainty = oof_uncertainties.sum() > 0
    return oof_preds, rmse, r2, avg_importance, (oof_uncertainties if has_uncertainty else None)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(y, oof_preds, importance, rmse, r2, uncertainties=None):
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

    # 5. Uncertainty calibration (if available)
    if uncertainties is not None:
        abs_errors = np.abs(residuals)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(uncertainties, abs_errors, alpha=0.5, s=30, edgecolors="k", linewidth=0.5)
        ax.set_xlabel("Predicted Uncertainty (mph)")
        ax.set_ylabel("Absolute Error (mph)")
        corr = np.corrcoef(uncertainties, abs_errors)[0, 1]
        ax.set_title(f"Uncertainty Calibration (corr={corr:.3f})")
        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}/uncertainty_calibration.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Startup banner ---
    if HAS_RICH:
        gpu_str = "[green]CUDA[/green]" if USE_GPU else "[dim]CPU only[/dim]"
        console.print(Panel(
            f"[bold]{MODEL_TYPE}[/bold]  |  GPU: {gpu_str}  |  Features: top {TOP_N_FEATURES}  |  "
            f"CV: {'LOGO' if USE_LOGO else f'{N_FOLDS}-fold GroupKFold'}",
            title="[bold]Autoresearch[/bold]",
            border_style="blue",
        ))
    else:
        gpu_str = "CUDA" if USE_GPU else "CPU only"
        info(f"Model: {MODEL_TYPE} | GPU: {gpu_str} | Features: top {TOP_N_FEATURES}")

    X, y, groups = load_data()
    info(f"Data: {len(X)} samples, {X.shape[1]} features, {len(groups.unique())} groups")

    oof_preds, rmse, r2, importance, uncertainties = cross_validate(X, y, groups)
    plot_results(y, oof_preds, importance, rmse, r2, uncertainties)

    # --- Results summary (rich or plain) ---
    if HAS_RICH:
        table = Table(title="Results", box=box.ROUNDED, show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("R²", f"{r2:.6f}")
        table.add_row("RMSE", f"{rmse:.4f} mph")
        if uncertainties is not None:
            table.add_row("Mean Uncertainty", f"{np.mean(uncertainties):.4f} mph")
        console.print(table)

        feat_table = Table(title="Top 10 Features", box=box.SIMPLE, show_header=True)
        feat_table.add_column("Feature", style="bold")
        feat_table.add_column("Importance", justify="right")
        for feat, imp in importance.head(10).items():
            feat_table.add_row(feat, f"{imp:.4f}")
        console.print(feat_table)
    else:
        info(f"R²={r2:.6f}  RMSE={rmse:.4f} mph")
        if uncertainties is not None:
            info(f"Mean Uncertainty={np.mean(uncertainties):.4f} mph")
        info("Top 10 features:")
        for feat, imp in importance.head(10).items():
            info(f"  {feat}: {imp:.4f}")

    # --- METRIC output (always stdout, always plain — autoresearch parses these) ---
    print(f"METRIC r2={r2:.6f}")
    print(f"METRIC rmse={rmse:.6f}")
    if uncertainties is not None:
        print(f"METRIC mean_uncertainty={np.mean(uncertainties):.6f}")


if __name__ == "__main__":
    main()
