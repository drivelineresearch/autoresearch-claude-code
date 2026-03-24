"""Model registry for autoresearch experiments.

Each model is registered with a build function returning (model_or_pipeline, metadata).
The autoresearch agent reads this file to discover available models and their tunable
hyperparameters. Modify MODEL_TYPE and MODEL_PARAMS in train.py to switch models.

Models are grouped by category:
  - boosting: xgboost, catboost, lightgbm, histgb
  - neural:   pytorch_mlp, mc_dropout, ft_transformer, tabpfn, tabnet, mlp
  - linear:   ridge, elasticnet, lasso, huber
  - bayesian: bayesian_ridge, gp
  - other:    svr, knn
  - ensemble: stacking
"""

import sys
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# GPU auto-detection
# ---------------------------------------------------------------------------

def detect_gpu():
    """Return True if a CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


_USE_GPU = None


def _get_use_gpu():
    """Lazy GPU detection — avoids importing torch at module load time."""
    global _USE_GPU
    if _USE_GPU is None:
        _USE_GPU = detect_gpu()
    return _USE_GPU


class _LazyGPU:
    """Descriptor that defers GPU detection until first access."""
    __hash__ = None

    def __repr__(self):
        return repr(_get_use_gpu())
    def __bool__(self):
        return _get_use_gpu()
    def __eq__(self, other):
        return _get_use_gpu() == other


USE_GPU = _LazyGPU()


def _to_numpy(X):
    """Convert DataFrame/array to numpy, used by PyTorch wrappers."""
    return X.values if hasattr(X, "values") else X


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY = {}


def register(name, category):
    """Decorator to register a model builder function."""
    def decorator(fn):
        _REGISTRY[name] = {"builder": fn, "category": category}
        return fn
    return decorator


def build_model(model_type, params=None):
    """Build a model by name.

    Returns (model_or_pipeline, metadata_dict).

    metadata keys:
      needs_scaling    - bool, True if features should be scaled
      has_native_importance - bool, True → use .feature_importances_
      supports_gpu     - bool
      supports_eval_set - bool, True → pass eval_set to .fit()
      category         - str
    """
    if model_type not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}. Available: {available}")
    entry = _REGISTRY[model_type]
    model, meta = entry["builder"](params or {})
    meta["category"] = entry["category"]
    return model, meta


def list_models():
    """Return list of (name, category) tuples."""
    return [(k, v["category"]) for k, v in _REGISTRY.items()]


def check_available():
    """Check which models can be instantiated (deps installed).

    Returns dict of {name: (available, category, error_or_None)}.
    """
    results = {}
    for name, entry in _REGISTRY.items():
        try:
            entry["builder"]({})
            results[name] = (True, entry["category"], None)
        except ImportError as e:
            results[name] = (False, entry["category"], str(e))
        except Exception as e:
            results[name] = (False, entry["category"], str(e))
    return results


def print_model_table():
    """Print a rich table of available models (falls back to plain text)."""
    avail = check_available()
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console(stderr=True)
        table = Table(title="Model Zoo", box=box.ROUNDED, show_lines=False)
        table.add_column("Model", style="bold")
        table.add_column("Category")
        table.add_column("Status")
        for name, (ok, cat, err) in sorted(avail.items(), key=lambda x: (x[1][1], x[0])):
            status = "[green]ready[/green]" if ok else f"[red]missing:[/red] {err}"
            table.add_row(name, cat, status)
        console.print(table)
    except ImportError:
        print("Model Zoo:", file=sys.stderr)
        for name, (ok, cat, err) in sorted(avail.items(), key=lambda x: (x[1][1], x[0])):
            status = "ready" if ok else f"missing: {err}"
            print(f"  {name:20s} {cat:10s} {status}", file=sys.stderr)


# ===========================================================================
# PyTorch sklearn-compatible wrappers
# ===========================================================================

class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """PyTorch MLP with dropout, batch norm, AdamW, and early stopping.

    Sklearn-compatible: implements fit(X, y) and predict(X).
    Tunable: hidden_dims, dropout, lr, weight_decay, epochs, batch_size, patience.
    """

    def __init__(self, hidden_dims=(128, 64, 32), dropout=0.3, lr=1e-3,
                 weight_decay=1e-2, epochs=500, batch_size=32,
                 patience=30, random_state=42):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y, eval_set=None):
        import torch
        import torch.nn as nn

        if eval_set is None and self.patience < self.epochs:
            import warnings
            warnings.warn(
                f"No eval_set provided to {self.__class__.__name__}; "
                f"training for full {self.epochs} epochs without early stopping."
            )

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.device_ = torch.device("cuda" if USE_GPU else "cpu")

        # Build network
        layers = []
        in_dim = X.shape[1]
        for h in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net_ = nn.Sequential(*layers).to(self.device_)

        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        X_t = torch.tensor(_to_numpy(X), dtype=torch.float32).to(self.device_)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device_)

        # Validation data for early stopping
        X_val_t, y_val_t = None, None
        if eval_set is not None:
            X_v, y_v = eval_set[0]
            X_val_t = torch.tensor(_to_numpy(X_v), dtype=torch.float32).to(self.device_)
            y_val_t = torch.tensor(y_v, dtype=torch.float32).reshape(-1, 1).to(self.device_)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )

        self.net_.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.net_(X_batch)
                loss = nn.functional.mse_loss(pred, y_batch)
                loss.backward()
                optimizer.step()

            if X_val_t is not None:
                self.net_.eval()
                with torch.no_grad():
                    val_pred = self.net_(X_val_t)
                    val_loss = nn.functional.mse_loss(val_pred, y_val_t).item()
                self.net_.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self.net_.load_state_dict(best_state)

        return self

    def predict(self, X):
        import torch
        self.net_.eval()
        X_t = torch.tensor(_to_numpy(X), dtype=torch.float32).to(self.device_)
        with torch.no_grad():
            return self.net_(X_t).cpu().numpy().flatten()


class MCDropoutRegressor(TorchMLPRegressor):
    """MC Dropout: PyTorch MLP with dropout ON at inference for uncertainty.

    predict() returns mean of mc_samples forward passes.
    After predict(), self.uncertainty_ holds per-sample std.
    """

    def __init__(self, mc_samples=50, **kwargs):
        super().__init__(**kwargs)
        self.mc_samples = mc_samples

    def predict(self, X):
        import torch
        self.net_.train()  # keep dropout ON
        X_t = torch.tensor(_to_numpy(X), dtype=torch.float32).to(self.device_)
        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                preds.append(self.net_(X_t).cpu().numpy().flatten())
        preds = np.array(preds)
        self.uncertainty_ = preds.std(axis=0)
        return preds.mean(axis=0)


class FTTransformerRegressor(BaseEstimator, RegressorMixin):
    """Feature Tokenizer + Transformer for tabular regression.

    Each feature gets a learned Linear(1, d_model) embedding. A learnable CLS token
    is prepended, then passed through a TransformerEncoder. The CLS output is projected
    to a scalar prediction.

    Tunable: d_model, n_heads, n_layers, dropout, lr, weight_decay, epochs, patience.
    """

    def __init__(self, d_model=64, n_heads=4, n_layers=2, dropout=0.2,
                 lr=1e-3, weight_decay=1e-2, epochs=300, batch_size=32,
                 patience=30, random_state=42):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y, eval_set=None):
        import torch
        import torch.nn as nn

        if eval_set is None and self.patience < self.epochs:
            import warnings
            warnings.warn(
                f"No eval_set provided to {self.__class__.__name__}; "
                f"training for full {self.epochs} epochs without early stopping."
            )

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.device_ = torch.device("cuda" if USE_GPU else "cpu")
        n_features = X.shape[1]

        # Per-feature embeddings
        self.feat_embeddings_ = nn.ModuleList([
            nn.Linear(1, self.d_model) for _ in range(n_features)
        ]).to(self.device_)

        # Learnable CLS token
        self.cls_token_ = nn.Parameter(
            torch.randn(1, 1, self.d_model, device=self.device_)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads,
            dim_feedforward=self.d_model * 2, dropout=self.dropout,
            batch_first=True,
        )
        self.transformer_ = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        ).to(self.device_)

        # Output head
        self.head_ = nn.Linear(self.d_model, 1).to(self.device_)

        all_params = (
            list(self.feat_embeddings_.parameters())
            + [self.cls_token_]
            + list(self.transformer_.parameters())
            + list(self.head_.parameters())
        )
        optimizer = torch.optim.AdamW(all_params, lr=self.lr, weight_decay=self.weight_decay)

        X_t = torch.tensor(_to_numpy(X), dtype=torch.float32).to(self.device_)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device_)

        X_val_t, y_val_t = None, None
        if eval_set is not None:
            X_v, y_v = eval_set[0]
            X_val_t = torch.tensor(_to_numpy(X_v), dtype=torch.float32).to(self.device_)
            y_val_t = torch.tensor(y_v, dtype=torch.float32).reshape(-1, 1).to(self.device_)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )

        for epoch in range(self.epochs):
            self._set_train(True)
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self._forward(X_batch)
                loss = torch.nn.functional.mse_loss(pred, y_batch)
                loss.backward()
                optimizer.step()

            if X_val_t is not None:
                self._set_train(False)
                with torch.no_grad():
                    val_pred = self._forward(X_val_t)
                    val_loss = torch.nn.functional.mse_loss(val_pred, y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self._get_state()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self._load_state(best_state)

        return self

    def predict(self, X):
        import torch
        self._set_train(False)
        X_t = torch.tensor(_to_numpy(X), dtype=torch.float32).to(self.device_)
        with torch.no_grad():
            return self._forward(X_t).cpu().numpy().flatten()

    def _forward(self, X_t):
        """Tokenize features, prepend CLS, run transformer, project."""
        import torch
        # X_t: (batch, n_features) → tokens: (batch, n_features, d_model)
        tokens = torch.stack([
            emb(X_t[:, i:i+1]) for i, emb in enumerate(self.feat_embeddings_)
        ], dim=1)
        cls = self.cls_token_.expand(X_t.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, 1+n_features, d_model)
        out = self.transformer_(tokens)
        return self.head_(out[:, 0, :])  # CLS output → scalar

    def _set_train(self, mode):
        self.feat_embeddings_.train(mode)
        self.transformer_.train(mode)
        self.head_.train(mode)

    def _get_state(self):
        import torch
        state = {}
        for name, module in [("feat", self.feat_embeddings_),
                              ("trans", self.transformer_),
                              ("head", self.head_)]:
            state[name] = {k: v.clone() for k, v in module.state_dict().items()}
        state["cls"] = self.cls_token_.clone()
        return state

    def _load_state(self, state):
        self.feat_embeddings_.load_state_dict(state["feat"])
        self.transformer_.load_state_dict(state["trans"])
        self.head_.load_state_dict(state["head"])
        self.cls_token_.data.copy_(state["cls"])


# ===========================================================================
# Registered models — Boosting
# ===========================================================================

@register("xgboost", "boosting")
def _build_xgboost(params):
    """XGBoost gradient boosted trees.

    Tunable: n_estimators, max_depth, learning_rate, subsample,
    colsample_bytree, min_child_weight, reg_alpha, reg_lambda.
    """
    import xgboost as xgb
    defaults = {
        "n_estimators": 1000, "max_depth": 4, "learning_rate": 0.03,
        "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 5,
        "reg_alpha": 0.5, "reg_lambda": 2.0, "random_state": 42,
        "early_stopping_rounds": 50,
    }
    if USE_GPU:
        defaults["device"] = "cuda"
    defaults.update(params)
    return xgb.XGBRegressor(**defaults), {
        "needs_scaling": False,
        "has_native_importance": True,
        "supports_gpu": True,
        "supports_eval_set": True,
    }


@register("catboost", "boosting")
def _build_catboost(params):
    """CatBoost gradient boosting. GPU-capable.

    Tunable: iterations, depth, learning_rate, l2_leaf_reg.
    Note: eval_set is (X, y) tuple, not [(X, y)] list.
    """
    from catboost import CatBoostRegressor
    defaults = {
        "iterations": 1000, "depth": 4, "learning_rate": 0.03,
        "l2_leaf_reg": 3, "random_seed": 42, "verbose": 0,
        "early_stopping_rounds": 50,
    }
    if USE_GPU:
        defaults["task_type"] = "GPU"
    defaults.update(params)
    return CatBoostRegressor(**defaults), {
        "needs_scaling": False,
        "has_native_importance": True,
        "supports_gpu": True,
        "supports_eval_set": True,
    }


@register("lightgbm", "boosting")
def _build_lightgbm(params):
    """LightGBM gradient boosting. Fastest boosting library.

    Tunable: n_estimators, max_depth, learning_rate, num_leaves,
    subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_samples.
    Note: uses callbacks API for early stopping.
    """
    import lightgbm as lgb
    defaults = {
        "n_estimators": 1000, "max_depth": 4, "learning_rate": 0.03,
        "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5,
        "reg_lambda": 2.0, "random_state": 42, "verbose": -1,
    }
    if USE_GPU:
        defaults["device"] = "gpu"
    defaults.update(params)
    return lgb.LGBMRegressor(**defaults), {
        "needs_scaling": False,
        "has_native_importance": True,
        "supports_gpu": True,
        "supports_eval_set": True,
    }


@register("histgb", "boosting")
def _build_histgb(params):
    """sklearn HistGradientBoosting. No extra deps needed.

    Tunable: max_iter, max_depth, learning_rate, min_samples_leaf,
    l2_regularization, max_features.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    defaults = {
        "max_iter": 500, "max_depth": 4, "learning_rate": 0.05,
        "min_samples_leaf": 10, "l2_regularization": 1.0,
        "random_state": 42, "early_stopping": True,
        "validation_fraction": 0.15, "n_iter_no_change": 20,
    }
    defaults.update(params)
    return HistGradientBoostingRegressor(**defaults), {
        "needs_scaling": False,
        "has_native_importance": False,
        "supports_gpu": False,
        "supports_eval_set": False,
    }


# ===========================================================================
# Registered models — Neural
# ===========================================================================

@register("pytorch_mlp", "neural")
def _build_pytorch_mlp(params):
    """PyTorch MLP with dropout, batch norm, AdamW, early stopping. CUDA-capable.

    Tunable: hidden_dims, dropout, lr, weight_decay, epochs, batch_size, patience.
    """
    defaults = {
        "hidden_dims": (128, 64, 32), "dropout": 0.3, "lr": 1e-3,
        "weight_decay": 1e-2, "epochs": 500, "batch_size": 32,
        "patience": 30, "random_state": 42,
    }
    defaults.update(params)
    model = TorchMLPRegressor(**defaults)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", model),
    ]), {
        "needs_scaling": True,
        "has_native_importance": False,
        "supports_gpu": True,
        "supports_eval_set": True,
    }


@register("mc_dropout", "neural")
def _build_mc_dropout(params):
    """MC Dropout: PyTorch MLP with dropout ON at inference for uncertainty.

    Same architecture as pytorch_mlp. After predict(), model.uncertainty_ has
    per-sample std from mc_samples stochastic forward passes.

    Tunable: mc_samples, hidden_dims, dropout, lr, weight_decay, epochs, patience.
    """
    defaults = {
        "mc_samples": 50, "hidden_dims": (128, 64, 32), "dropout": 0.3,
        "lr": 1e-3, "weight_decay": 1e-2, "epochs": 500, "batch_size": 32,
        "patience": 30, "random_state": 42,
    }
    defaults.update(params)
    model = MCDropoutRegressor(**defaults)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mc", model),
    ]), {
        "needs_scaling": True,
        "has_native_importance": False,
        "supports_gpu": True,
        "supports_eval_set": True,
    }


@register("ft_transformer", "neural")
def _build_ft_transformer(params):
    """Feature Tokenizer + Transformer. Attention-based tabular model. CUDA-capable.

    Each feature gets its own embedding; a CLS token aggregates via self-attention.
    Tunable: d_model, n_heads, n_layers, dropout, lr, weight_decay, epochs, patience.
    """
    defaults = {
        "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.2,
        "lr": 1e-3, "weight_decay": 1e-2, "epochs": 300, "batch_size": 32,
        "patience": 30, "random_state": 42,
    }
    defaults.update(params)
    model = FTTransformerRegressor(**defaults)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ftt", model),
    ]), {
        "needs_scaling": True,
        "has_native_importance": False,
        "supports_gpu": True,
        "supports_eval_set": True,
    }


@register("tabpfn", "neural")
def _build_tabpfn(params):
    """TabPFN: Pretrained transformer foundation model for tabular data (Nature 2025).

    Zero-shot — no training, just inference. Works on datasets up to 10K samples.
    Tunable: n_estimators (internal ensemble size).
    """
    from tabpfn import TabPFNRegressor
    defaults = {"n_estimators": 8, "random_state": 42}
    defaults.update(params)
    model = TabPFNRegressor(**defaults)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("tabpfn", model),
    ]), {
        "needs_scaling": True,
        "has_native_importance": False,
        "supports_gpu": True,
        "supports_eval_set": False,
    }


@register("tabnet", "neural")
def _build_tabnet(params):
    """TabNet: Attention-based feature selection NN.

    Tunable: n_d, n_a, n_steps, gamma, lambda_sparse, lr.
    """
    from pytorch_tabnet.tab_model import TabNetRegressor as _TabNet
    defaults = {
        "n_d": 16, "n_a": 16, "n_steps": 3, "gamma": 1.3,
        "lambda_sparse": 1e-3, "seed": 42, "verbose": 0,
    }
    defaults.update(params)
    model = _TabNet(**defaults)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("tabnet", model),
    ]), {
        "needs_scaling": True,
        "has_native_importance": False,
        "supports_gpu": True,
        "supports_eval_set": True,
    }


@register("mlp", "neural")
def _build_mlp(params):
    """sklearn MLPRegressor. Simple neural net, no extra deps.

    Tunable: hidden_layer_sizes, activation, alpha, learning_rate_init, max_iter.
    """
    from sklearn.neural_network import MLPRegressor
    defaults = {
        "hidden_layer_sizes": (64, 32), "activation": "relu",
        "solver": "adam", "alpha": 0.001, "learning_rate": "adaptive",
        "learning_rate_init": 0.001, "max_iter": 2000,
        "early_stopping": True, "validation_fraction": 0.15,
        "random_state": 42,
    }
    defaults.update(params)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(**defaults)),
    ]), {
        "needs_scaling": True,
        "has_native_importance": False,
        "supports_gpu": False,
        "supports_eval_set": False,
    }


# ===========================================================================
# Registered models — Linear
# ===========================================================================

@register("ridge", "linear")
def _build_ridge(params):
    """Ridge regression (L2 regularization). Linear baseline.

    Tunable: alpha.
    """
    from sklearn.linear_model import Ridge
    defaults = {"alpha": 1.0}
    defaults.update(params)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(**defaults)),
    ]), {
        "needs_scaling": True,
        "has_native_importance": False,
        "supports_gpu": False,
        "supports_eval_set": False,
    }


@register("elasticnet", "linear")
def _build_elasticnet(params):
    """ElasticNet (L1+L2). Sparse linear model — L1 zeros out features.

    Tunable: alpha, l1_ratio, max_iter.
    """
    from sklearn.linear_model import ElasticNet
    defaults = {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 5000, "random_state": 42}
    defaults.update(params)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(**defaults)),
    ]), {
        "needs_scaling": True,
        "has_native_importance": False,
        "supports_gpu": False,
        "supports_eval_set": False,
    }


@register("lasso", "linear")
def _build_lasso(params):
    """Lasso regression (L1 only). Fully sparse feature selection.

    Tunable: alpha, max_iter.
    """
    from sklearn.linear_model import Lasso
    defaults = {"alpha": 0.1, "max_iter": 5000, "random_state": 42}
    defaults.update(params)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(**defaults)),
    ]), {
        "needs_scaling": True,
        "has_native_importance": False,
        "supports_gpu": False,
        "supports_eval_set": False,
    }


@register("huber", "linear")
def _build_huber(params):
    """HuberRegressor. Robust to outliers.

    Tunable: epsilon (outlier threshold), alpha, max_iter.
    """
    from sklearn.linear_model import HuberRegressor
    defaults = {"epsilon": 1.35, "max_iter": 1000, "alpha": 0.01}
    defaults.update(params)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("huber", HuberRegressor(**defaults)),
    ]), {
        "needs_scaling": True,
        "has_native_importance": False,
        "supports_gpu": False,
        "supports_eval_set": False,
    }


# ===========================================================================
# Registered models — Bayesian
# ===========================================================================

@register("bayesian_ridge", "bayesian")
def _build_bayesian_ridge(params):
    """Bayesian Ridge regression. Automatic regularization + uncertainty.

    Returns prediction uncertainty via model.predict(X, return_std=True).
    Tunable: n_iter, tol.
    """
    from sklearn.linear_model import BayesianRidge
    defaults = {"n_iter": 300, "tol": 1e-6, "compute_score": True}
    defaults.update(params)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("bayes", BayesianRidge(**defaults)),
    ]), {
        "needs_scaling": True,
        "has_native_importance": False,
        "supports_gpu": False,
        "supports_eval_set": False,
    }


@register("gp", "bayesian")
def _build_gp(params):
    """Gaussian Process regression. Bayesian with full uncertainty.

    O(n^3) complexity — fine for n<=200, impractical for n>5000.
    Tunable: kernel, alpha, n_restarts_optimizer.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    defaults = {
        "kernel": 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
        "alpha": 1e-2, "n_restarts_optimizer": 5, "random_state": 42,
    }
    defaults.update(params)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gp", GaussianProcessRegressor(**defaults)),
    ]), {
        "needs_scaling": True,
        "has_native_importance": False,
        "supports_gpu": False,
        "supports_eval_set": False,
    }


# ===========================================================================
# Registered models — Other
# ===========================================================================

@register("svr", "other")
def _build_svr(params):
    """Support Vector Regression (RBF kernel).

    Tunable: C, epsilon, gamma, kernel.
    """
    from sklearn.svm import SVR
    defaults = {"kernel": "rbf", "C": 10.0, "epsilon": 0.1, "gamma": "scale"}
    defaults.update(params)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(**defaults)),
    ]), {
        "needs_scaling": True,
        "has_native_importance": False,
        "supports_gpu": False,
        "supports_eval_set": False,
    }


@register("knn", "other")
def _build_knn(params):
    """k-Nearest Neighbors regression. Instance-based, no training.

    Tunable: n_neighbors, weights, metric, p.
    """
    from sklearn.neighbors import KNeighborsRegressor
    defaults = {"n_neighbors": 7, "weights": "distance", "metric": "minkowski"}
    defaults.update(params)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(**defaults)),
    ]), {
        "needs_scaling": True,
        "has_native_importance": False,
        "supports_gpu": False,
        "supports_eval_set": False,
    }


# ===========================================================================
# Registered models — Ensemble
# ===========================================================================

@register("stacking", "ensemble")
def _build_stacking(params):
    """Stacking ensemble: diverse base models + Ridge meta-learner.

    Base models: XGBoost, HistGradientBoosting, Ridge, ElasticNet, KNN.
    Inner CV is group-aware (handled by cross_validate in train.py).
    Tunable: passthrough (include original features in meta-learner).
    """
    from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    import xgboost as xgb

    base_estimators = [
        ("xgb", xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
        ("histgb", HistGradientBoostingRegressor(
            max_iter=200, max_depth=4, random_state=42)),
        ("ridge", Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=1.0))])),
        ("enet", Pipeline([("s", StandardScaler()),
                           ("e", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))])),
        ("knn", Pipeline([("s", StandardScaler()),
                          ("k", KNeighborsRegressor(n_neighbors=7, weights="distance"))])),
    ]

    defaults = {"passthrough": False}
    defaults.update(params)

    model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,  # overridden by cross_validate for group-awareness
        passthrough=defaults.get("passthrough", False),
        n_jobs=-1,
    )

    return model, {
        "needs_scaling": False,
        "has_native_importance": False,
        "supports_gpu": False,
        "supports_eval_set": False,
        "is_stacking": True,
    }
