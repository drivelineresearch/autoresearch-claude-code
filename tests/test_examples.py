"""Fast methodology/API regressions; no dataset downloads or GPU required."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

for dependency in ("numpy", "pandas", "sklearn", "matplotlib"):
    if importlib.util.find_spec(dependency) is None:
        raise unittest.SkipTest(f"Example tests require optional {dependency}")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone, is_regressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import models, train


class ExampleTests(unittest.TestCase):
    def setUp(self):
        self.settings = mock.patch.multiple(train, MODEL_TYPE="ridge", MODEL_PARAMS={},
                                            TOP_N_FEATURES=None, USE_LOGO=True, HAS_RICH=False)
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.X = pd.DataFrame({"signal": np.arange(8, dtype=float),
                               "noise": [2., 0., 3., 1., 0., 4., 1., 2.]})
        self.y = 70 + 2 * self.X.signal.to_numpy()
        self.groups = pd.Series(np.arange(8))

    def test_real_ridge_logo_is_finite_without_singleton_importance_warnings(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            preds, rmse, r2, importance, uncertainty = train.cross_validate(self.X, self.y, self.groups)
        self.assertEqual(preds.shape, self.y.shape)
        self.assertTrue(np.isfinite([rmse, r2]).all())
        self.assertGreater(r2, .9)
        self.assertTrue(importance.empty)
        self.assertIsNone(uncertainty)

    def test_held_out_target_never_reaches_selector_or_fit(self):
        selected_rows, fitted_rows = [], []

        def select(X, y, groups):
            selected_rows.append(set(X.index))
            self.assertEqual(set(X.index), set(groups.to_numpy()))
            np.testing.assert_array_equal(y, self.y[X.index])
            return ["signal"]

        class SpyModel:
            def fit(model, X, y, **kwargs):
                self.assertFalse(kwargs, "Outer validation data must not be passed to fit")
                fitted_rows.append(set(X.index))
                model.mean = np.mean(y)

            def predict(model, X):
                self.assertFalse(set(X.index) & fitted_rows[-1])
                self.assertFalse(set(X.index) & selected_rows[-1])
                return np.full(len(X), model.mean)

        with mock.patch.object(train, "select_features", side_effect=select), \
             mock.patch.object(train, "build_model", return_value=(SpyModel(), {})) as build:
            train.cross_validate(self.X, self.y, self.groups)
        self.assertEqual(len(fitted_rows), 8)
        self.assertEqual(fitted_rows, selected_rows)
        for call in build.call_args_list:
            self.assertEqual(call.kwargs["random_state"], train.SEED)

    def test_selector_fits_only_supplied_training_rows_without_eval_set(self):
        class Ranker:
            feature_importances_ = np.array([.1, .9])
            def fit(model, X, y, **kwargs):
                self.assertEqual(list(X.index), [0, 1, 2])
                self.assertFalse(kwargs)
        with mock.patch.object(train, "TOP_N_FEATURES", 1), \
             mock.patch.object(train, "build_model", return_value=(Ranker(), {})):
            chosen = train.select_features(self.X.iloc[:3], self.y[:3], self.groups[:3])
        self.assertEqual(chosen, ["noise"])

    def test_invalid_cv_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "athlete groups"):
            train.cross_validate(self.X, self.y, [1] * len(self.y))
        with self.assertRaisesRegex(ValueError, "nonconstant"):
            train.cross_validate(self.X, np.ones(8), self.groups)
        with mock.patch.multiple(train, USE_LOGO=False, N_FOLDS=9):
            with self.assertRaisesRegex(ValueError, "N_FOLDS"):
                train.cross_validate(self.X, self.y, self.groups)

    def test_prediction_length_or_nonfinite_values_are_rejected(self):
        class BadModel:
            def fit(self, X, y):
                return self
            def predict(self, X):
                return np.full(len(X), np.nan)
        with mock.patch.object(train, "build_model", return_value=(BadModel(), {})):
            with self.assertRaisesRegex(ValueError, "predictions must be finite"):
                train.cross_validate(self.X, self.y, self.groups)

    def test_non_singleton_permutation_importance_is_finite(self):
        groups = pd.Series(np.repeat(np.arange(4), 2))
        _, _, _, importance, _ = train.cross_validate(self.X, self.y, groups)
        self.assertEqual(set(importance.index), set(self.X.columns))
        self.assertTrue(np.isfinite(importance).all())

    def test_bayesian_model_uses_current_api_and_returns_uncertainty(self):
        with mock.patch.object(train, "MODEL_TYPE", "bayesian_ridge"):
            _, _, _, _, uncertainty = train.cross_validate(self.X, self.y, self.groups)
        self.assertTrue(np.isfinite(uncertainty).all())
        self.assertTrue((uncertainty >= 0).all())

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "optional XGBoost not installed")
    def test_xgboost_current_fit_api_and_reproducibility(self):
        with mock.patch.multiple(train, MODEL_TYPE="xgboost", TOP_N_FEATURES=1,
                                 MODEL_PARAMS={"n_estimators": 5, "min_child_weight": 1}):
            first = train.cross_validate(self.X, self.y, self.groups)
            second = train.cross_validate(self.X, self.y, self.groups)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertTrue(np.isfinite([first[1], first[2]]).all())

    def test_stacking_inner_folds_remain_athlete_disjoint(self):
        groups = pd.Series(np.repeat(np.arange(4), 2))
        class SpyStacking:
            def fit(model, X, y):
                model.mean = y.mean()
                outer_groups = groups.iloc[X.index].to_numpy()
                for inner_train, inner_val in model.cv.split():
                    self.assertFalse(set(outer_groups[inner_train]) & set(outer_groups[inner_val]))
            def predict(model, X):
                return np.full(len(X), model.mean)
        with mock.patch.object(train, "build_model", return_value=(SpyStacking(), {"is_stacking": True})), \
             mock.patch.object(train, "permutation_importance", return_value=mock.Mock(importances_mean=np.zeros(2))):
            train.cross_validate(self.X, self.y, groups)

    def _write_data(self, directory, metadata_transform=None, data_transform=None):
        poi = pd.DataFrame({"session_pitch": ["p1", "p2", "p3"],
                            "session": [10, 11, 12], "p_throws": ["R", "R", "L"],
                            "pitch_type": ["FF"] * 3, "pitch_speed_mph": [80., 84., 90.],
                            "elbow_transfer_fp_br": [0., 0., 1.],
                            "thorax_distal_transfer_fp_br": [1., 3., 4.],
                            "feature": [1., 3., 5.]})
        meta = pd.DataFrame({"session_pitch": ["p1", "p2", "p3"], "user": [101, 101, 102]})
        if metadata_transform:
            meta = metadata_transform(meta)
        if data_transform:
            poi = data_transform(poi)
        poi_path, meta_path = Path(directory) / "poi.csv", Path(directory) / "metadata.csv"
        poi.to_csv(poi_path, index=False)
        meta.to_csv(meta_path, index=False)
        return mock.patch.multiple(train, DATA_PATH=poi_path, METADATA_PATH=meta_path)

    def test_same_athlete_in_two_sessions_aggregates_once_and_ids_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp, self._write_data(tmp):
            X, y, groups = train.load_data()
        self.assertEqual(list(groups), [101, 102])
        np.testing.assert_array_equal(y, [82., 90.])
        self.assertFalse(set(train.DROP_COLS) & set(X.columns))
        self.assertTrue(np.isfinite(X.to_numpy()).all())
        self.assertEqual(X.loc[0, "thorax_to_elbow_transfer_ratio"], 0.)

    def test_pitch_level_mode_still_groups_by_athlete(self):
        with tempfile.TemporaryDirectory() as tmp, self._write_data(tmp), \
             mock.patch.object(train, "AGGREGATE_TO_PLAYER", False):
            X, y, groups = train.load_data()
        self.assertEqual(list(groups), [101, 101, 102])
        self.assertEqual(len(X), 3)
        self.assertFalse(set(train.DROP_COLS) & set(X.columns))

    def test_candidate_changes_are_loaded_without_editing_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp, self._write_data(tmp):
            for name in ("train.py", "models.py"):
                shutil.copy2(ROOT / "examples" / name, Path(tmp) / name)
            (Path(tmp) / "candidate.py").write_text(
                'MODEL_TYPE = "ridge"\nMODEL_PARAMS = {"alpha": 2}\nTOP_N_FEATURES = None\n'
                'def engineer_features(X):\n    return X.assign(candidate_signal=X.feature * 2)\n')
            env = dict(os.environ, AR_DATA_PATH=str(train.DATA_PATH),
                       AR_METADATA_PATH=str(train.METADATA_PATH), AR_PLOT_DIR=str(Path(tmp) / "plots"),
                       AR_DEVICE="cpu")
            code = ('import train; assert train.MODEL_TYPE == "ridge"; '
                    'assert train.MODEL_PARAMS == {"alpha": 2}; '
                    'assert "candidate_signal" in train.load_data()[0]; train.main()')
            result = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=env,
                                    text=True, capture_output=True, check=True, timeout=30)
            self.assertEqual((Path(tmp) / "train.py").read_bytes(),
                             (ROOT / "examples/train.py").read_bytes())
            self.assertIn("METRIC r2=", result.stdout)
            self.assertEqual(len(list((Path(tmp) / "plots").glob("*.png"))), 4)

    def test_candidate_cannot_reorder_rows_or_reintroduce_identifier_columns(self):
        for transform in (lambda X: X.iloc[::-1], lambda X: X.assign(session=1)):
            with self.subTest(transform=transform), tempfile.TemporaryDirectory() as tmp, \
                 self._write_data(tmp), mock.patch.object(train.candidate, "engineer_features", side_effect=transform):
                with self.assertRaisesRegex(ValueError, "Candidate features"):
                    train.load_data()

    def test_missing_and_duplicate_metadata_fail_closed(self):
        cases = [lambda meta: meta.iloc[:-1], lambda meta: pd.concat([meta, meta.iloc[:1]])]
        for transform in cases:
            with self.subTest(transform=transform), tempfile.TemporaryDirectory() as tmp, \
                 self._write_data(tmp, metadata_transform=transform):
                with self.assertRaisesRegex(ValueError, "Metadata"):
                    train.load_data()

    def test_invalid_pitch_type_and_target_are_rejected(self):
        for col, value in [("pitch_type", "SL"), ("p_throws", "?"), ("pitch_speed_mph", np.inf)]:
            with self.subTest(column=col), tempfile.TemporaryDirectory() as tmp, \
                 self._write_data(tmp, data_transform=lambda df: df.assign(**{col: value})):
                with self.assertRaises(ValueError):
                    train.load_data()

    def test_ar_seed_reaches_nested_model(self):
        model, _ = models.build_model("elasticnet", {"random_state": 42}, random_state=123)
        self.assertEqual(model.named_steps["enet"].random_state, 123)

    def test_mc_dropout_clone_preserves_hyperparameters_without_torch(self):
        model = models.MCDropoutRegressor(mc_samples=3, hidden_dims=(7, 5), dropout=.2,
                                          epochs=2, random_state=123)
        copy = clone(model)
        self.assertEqual(copy.get_params(), model.get_params())
        self.assertTrue(is_regressor(copy))
        self.assertTrue(is_regressor(models.FTTransformerRegressor()))

    def test_dependency_report_does_not_construct_or_import_models(self):
        with mock.patch.object(models.importlib.util, "find_spec", return_value=None), \
             mock.patch.object(models, "_get_use_gpu", side_effect=AssertionError("GPU probe")):
            available = models.check_available()
        for name in ("pytorch_mlp", "mc_dropout", "ft_transformer", "xgboost", "tabnet"):
            self.assertFalse(available[name][0], name)
        self.assertTrue(available["ridge"][0])

    def test_tabnet_adapter_reshapes_targets_and_predictions(self):
        class FakeTabNet(BaseEstimator):
            def fit(model, X, y, **kwargs):
                self.assertEqual(y.shape, (8, 1))
                self.assertEqual(kwargs["patience"], 0)
                self.assertEqual(X.dtype, np.float32)
                model.mean = y.mean()
            def predict(model, X):
                return np.full((len(X), 1), model.mean)
        model = models.TabNetRegressorAdapter(FakeTabNet(), max_epochs=1)
        self.assertEqual(clone(model).max_epochs, 1)
        self.assertEqual(model.fit(self.X, self.y).predict(self.X).shape, (8,))

    def test_empty_importance_plot_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(train, "PLOT_DIR", tmp):
            (Path(tmp) / "uncertainty_calibration.png").write_bytes(b"old-run")
            train.plot_results(self.y, self.y + .1, pd.Series(dtype=float), .1, .9)
            self.assertEqual(len(list(Path(tmp).glob("*.png"))), 4)
            self.assertFalse((Path(tmp) / "uncertainty_calibration.png").exists())

    def test_wrapper_resolves_runtime_and_files_from_another_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            uv = Path(tmp) / "uv"
            uv.write_text('#!/bin/sh\nprintf "seed=%s cwd=%s args=%s\\n" "$AR_SEED" "$PWD" "$*"\n')
            uv.chmod(0o755)
            env = dict(os.environ, PATH=f"{tmp}:{os.environ['PATH']}")
            result = subprocess.run(["bash", str(ROOT / "examples/autoresearch.sh"), "123"],
                                    cwd=tmp, env=env, text=True, capture_output=True, check=True)
        self.assertIn(f"seed=123 cwd={ROOT / 'examples'}", result.stdout)
        self.assertIn("-m py_compile train.py models.py candidate.py", result.stdout)
        self.assertIn("python train.py", result.stdout)

    def test_wrapper_rejects_bad_seed_before_running_uv(self):
        result = subprocess.run(["bash", str(ROOT / "examples/autoresearch.sh"), "bad"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
