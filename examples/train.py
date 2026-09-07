#!/usr/bin/env python3
"""Fastball velocity prediction from biomechanical POI metrics.

Uses the autoresearch model zoo (models.py) — 19 models across 6 categories.
Edit candidate.py to propose model and feature changes; keep this evaluator fixed.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, PredefinedSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

try:  # Support both `python examples/train.py` and module imports.
    from .models import build_model, USE_GPU
    from . import candidate
except ImportError:
    from models import build_model, USE_GPU
    import candidate

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
# Evaluation contract — fixed within a research session
# ---------------------------------------------------------------------------

SEED = int(os.environ.get("AR_SEED", "42"))  # autoresearch.sh passes SEED for noise-floor / confirm re-runs
EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("AR_DATA_PATH", EXAMPLE_DIR / "third_party/openbiomechanics/baseball_pitching/data/poi/poi_metrics.csv"))
METADATA_PATH = Path(os.environ.get("AR_METADATA_PATH", DATA_PATH.parent.parent / "metadata.csv"))
PLOT_DIR = Path(os.environ.get("AR_PLOT_DIR", EXAMPLE_DIR / "plots"))
N_FOLDS = 5

# Candidate settings are read on each process launch. Keep changes in candidate.py.
MODEL_TYPE = candidate.MODEL_TYPE
MODEL_PARAMS = candidate.MODEL_PARAMS
TOP_N_FEATURES = candidate.TOP_N_FEATURES

DROP_COLS = ["session_pitch", "session", "user", "pitch_type", "pitch_speed_mph"]
TARGET = "pitch_speed_mph"
GROUP_COL = "user"

AGGREGATE_TO_PLAYER = True
USE_LOGO = True


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------

def load_data():
    """Load POI metrics and validate the pitch-to-athlete mapping before splitting."""
    df = pd.read_csv(DATA_PATH)
    required = {"session_pitch", "session", "p_throws", "pitch_type", TARGET}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required POI columns: {sorted(missing)}")
    if df["session_pitch"].isna().any() or df["session_pitch"].duplicated().any():
        raise ValueError("POI session_pitch identifiers must be present and unique")
    metadata = pd.read_csv(METADATA_PATH, usecols=["session_pitch", GROUP_COL])
    if metadata.isna().any().any() or metadata["session_pitch"].duplicated().any():
        raise ValueError("Metadata must map each session_pitch to exactly one athlete")
    if GROUP_COL in df:
        raise ValueError(f"POI must not duplicate the metadata athlete column {GROUP_COL!r}")
    df = df.merge(metadata, on="session_pitch", how="left", validate="one_to_one")
    if df[GROUP_COL].isna().any():
        raise ValueError("Metadata does not identify the athlete for every POI pitch")
    if df["session"].isna().any() or (df.groupby("session")[GROUP_COL].nunique() > 1).any():
        raise ValueError("Each session must identify exactly one athlete")
    if not df["pitch_type"].eq("FF").all():
        raise ValueError("This benchmark expects fastballs (pitch_type=FF) only")
    handedness = df["p_throws"].map({"L": 0, "R": 1})
    if handedness.isna().any():
        raise ValueError("p_throws must contain only L or R")
    df["p_throws"] = handedness
    if not np.isfinite(pd.to_numeric(df[TARGET], errors="coerce")).all():
        raise ValueError("Target values must all be finite numbers")
    df[TARGET] = pd.to_numeric(df[TARGET])

    if AGGREGATE_TO_PLAYER:
        # Drop all identifiers before averaging: numeric IDs are not features.
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in set(DROP_COLS) | {GROUP_COL} or c == TARGET]
        agg_df = df.groupby(GROUP_COL)[numeric_cols].mean().reset_index()
        for col in ["elbow_transfer_fp_br", "shoulder_transfer_fp_br",
                    "thorax_distal_transfer_fp_br"]:
            if col in df:
                std_series = df.groupby(GROUP_COL)[col].std().fillna(0)
                agg_df[f"{col}_std"] = agg_df[GROUP_COL].map(std_series)
        df = agg_df

    groups = df[GROUP_COL].reset_index(drop=True)
    y = df[TARGET].to_numpy()
    X = df.drop(columns=list(set(DROP_COLS) | {TARGET, GROUP_COL}), errors="ignore").copy()

    # Candidate transforms receive features only, after target/ID exclusion.
    original_index = X.index.copy()
    X = candidate.engineer_features(X)
    if not isinstance(X, pd.DataFrame) or not X.index.equals(original_index):
        raise ValueError("Candidate features must preserve the DataFrame row order and index")
    if X.columns.duplicated().any() or set(X.columns) & (set(DROP_COLS) | {TARGET, GROUP_COL}):
        raise ValueError("Candidate features must have unique columns and exclude targets/identifiers")
    if X.empty or not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("Features must be a nonempty numeric table; explicitly encode or drop other columns")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Features contain missing/infinite values; add training-fold-only imputation before fitting")
    return X, y, groups


# ---------------------------------------------------------------------------
# Feature selection: fit exclusively on each outer training fold
# ---------------------------------------------------------------------------

def select_features(X, y, groups):
    """Rank only the supplied training data; never inspect outer held-out labels."""
    if TOP_N_FEATURES is None or TOP_N_FEATURES >= X.shape[1]:
        return X.columns.tolist()
    if TOP_N_FEATURES < 1:
        raise ValueError("TOP_N_FEATURES must be positive or None")
    ranker, _ = build_model("xgboost", {
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.03,
        "n_jobs": 1,
    }, random_state=SEED)
    ranker.fit(X, y)
    importance = pd.Series(ranker.feature_importances_, index=X.columns)
    return importance.sort_values(ascending=False, kind="stable").head(TOP_N_FEATURES).index.tolist()


# ---------------------------------------------------------------------------
# Cross-validation (model-agnostic)
# ---------------------------------------------------------------------------

def cross_validate(X, y, groups):
    """Compute pooled out-of-fold metrics with athlete-disjoint outer folds.

    The evaluation fold is never an early-stopping set. Models use fixed fit
    budgets; optional early stopping must be implemented with a separate inner
    athlete split. Repeated optimization on this CV score still requires a final
    untouched cohort before making a generalization claim.
    """
    y = np.asarray(y)
    groups = pd.Series(np.asarray(groups))
    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("X, y, and groups must have the same length")
    if len(y) < 2 or groups.isna().any() or groups.nunique() < 2:
        raise ValueError("Cross-validation needs at least two nonmissing athlete groups")
    if not np.isfinite(y).all() or np.ptp(y) == 0:
        raise ValueError("R2 requires a finite, nonconstant target")
    if not USE_LOGO and not 2 <= N_FOLDS <= groups.nunique():
        raise ValueError("N_FOLDS must be between 2 and the number of athlete groups")
    cv = LeaveOneGroupOut() if USE_LOGO else GroupKFold(n_splits=N_FOLDS)
    splits = list(cv.split(X, y, groups))
    oof_preds = np.full(len(y), np.nan)
    oof_uncertainties = np.full(len(y), np.nan)
    fold_importances = []
    progress = None
    if HAS_RICH:
        progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                            TextColumn("{task.completed}/{task.total}"),
                            TimeElapsedColumn(), console=console)
        task = progress.add_task(f"CV ({MODEL_TYPE})", total=len(splits))
        progress.start()
    try:
        for train_idx, val_idx in splits:
            features = select_features(X.iloc[train_idx], y[train_idx], groups.iloc[train_idx])
            X_train, X_val = X.iloc[train_idx][features], X.iloc[val_idx][features]
            y_train, y_val = y[train_idx], y[val_idx]
            model, meta = build_model(MODEL_TYPE, MODEL_PARAMS, random_state=SEED)
            if meta.get("is_stacking"):
                n_inner = min(5, groups.iloc[train_idx].nunique())
                if n_inner < 2:
                    raise ValueError("Stacking requires at least three outer athlete groups")
                inner = GroupKFold(n_splits=n_inner)
                test_fold = np.full(len(train_idx), -1)
                for i, (_, inner_val) in enumerate(inner.split(X_train, y_train, groups.iloc[train_idx])):
                    test_fold[inner_val] = i
                model.cv = PredefinedSplit(test_fold)
            # No outer held-out targets enter fit, including early stopping.
            model.fit(X_train, y_train)
            predictions = np.asarray(model.predict(X_val)).reshape(-1)
            if predictions.shape != y_val.shape or not np.isfinite(predictions).all():
                raise ValueError("Model predictions must be finite and match the validation rows")
            oof_preds[val_idx] = predictions
            if MODEL_TYPE == "mc_dropout":
                oof_uncertainties[val_idx] = model.named_steps["mc"].uncertainty_
            elif MODEL_TYPE in ("gp", "bayesian_ridge"):
                name = "gp" if MODEL_TYPE == "gp" else "bayes"
                scaled = model.named_steps["scaler"].transform(X_val)
                _, std = model.named_steps[name].predict(scaled, return_std=True)
                oof_uncertainties[val_idx] = std
            if meta.get("has_native_importance"):
                imp_model = model
                if hasattr(model, "named_steps"):
                    imp_model = next(step for step in model.named_steps.values()
                                     if hasattr(step, "feature_importances_"))
                fold_importances.append(pd.Series(imp_model.feature_importances_, index=features))
            elif len(val_idx) > 1:
                # R2 is undefined for singleton LOGO folds. MSE is defined, but
                # singleton feature permutations cannot measure importance.
                perm = permutation_importance(model, X_val, y_val, scoring="neg_mean_squared_error",
                                              n_repeats=5, random_state=SEED)
                fold_importances.append(pd.Series(perm.importances_mean, index=features))
            if progress:
                progress.update(task, advance=1)
    finally:
        if progress:
            progress.stop()
    if not np.isfinite(oof_preds).all():
        raise ValueError("Every row must receive one finite out-of-fold prediction")
    rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
    r2 = float(r2_score(y, oof_preds))
    importance = (pd.concat(fold_importances, axis=1).fillna(0).mean(axis=1)
                  .sort_values(ascending=False) if fold_importances else pd.Series(dtype=float))
    uncertainty = oof_uncertainties if np.isfinite(oof_uncertainties).all() else None
    return oof_preds, rmse, r2, importance, uncertainty


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(y, oof_preds, importance, rmse, r2, uncertainties=None):
    os.makedirs(PLOT_DIR, exist_ok=True)
    if uncertainties is None:
        # Do not leave a prior model's uncertainty artifact in the current run.
        (Path(PLOT_DIR) / "uncertainty_calibration.png").unlink(missing_ok=True)

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
    if not top.empty:
        top.sort_values().plot.barh(ax=ax)
    else:
        ax.text(0.5, 0.5, "Importance unavailable for singleton held-out folds",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Mean native importance or held-out permutation MSE increase")
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
        corr = (np.corrcoef(uncertainties, abs_errors)[0, 1]
                if np.std(uncertainties) > 0 and np.std(abs_errors) > 0 else float("nan"))
        ax.set_title(f"Uncertainty vs Error (descriptive corr={corr:.3f})")
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
