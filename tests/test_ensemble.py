"""Unit tests for WeightedLogEnsemble and StackingLogEnsemble."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models.target_transform import TransformedTargetTreeModel
from src.models.tree import LogTransformedTreeModel
from src.models.ensemble import WeightedLogEnsemble, StackingLogEnsemble


class TestEnsembleModels(unittest.TestCase):
    """Test suite for Ensemble architectures."""

    def setUp(self):
        np.random.seed(42)
        n_samples = 250
        self.df = pd.DataFrame({
            "store_type": np.random.choice(["mall", "street", "drive_thru"], n_samples),
            "city": np.random.choice(["Hanoi", "HCM"], n_samples),
            "store_id": np.random.choice(["S01", "S02", "S03"], n_samples),
            "area_m2": np.random.uniform(50, 200, n_samples),
            "seats": np.random.uniform(20, 80, n_samples),
            "store_age_days": np.random.uniform(100, 1000, n_samples),
            "temp_c": np.random.uniform(20, 35, n_samples),
            "rain_mm": np.random.exponential(5, n_samples),
            "day_of_year": np.random.randint(1, 365, n_samples),
            "is_promo": np.random.choice([0, 1], n_samples),
            "is_weekend": np.random.choice([0, 1], n_samples),
            "is_holiday": np.random.choice([0, 1], n_samples),
            "is_rainy": np.random.choice([0, 1], n_samples),
            "revenue": np.random.lognormal(mean=2.8, sigma=0.4, size=n_samples),
        })
        self.features = [c for c in self.df.columns if c != "revenue"]
        self.val_df = self.df.iloc[:50].copy()

    def test_weighted_ensemble_fit_predict(self):
        m1 = TransformedTargetTreeModel(
            transform_type="log1p", objective="huber",
            n_estimators=15, num_leaves=10, random_state=42
        )
        m2 = LogTransformedTreeModel(
            model_type="random_forest", use_log_target=True,
            n_estimators=15, max_depth=8, random_state=42
        )

        ensemble = WeightedLogEnsemble(
            base_models={"m1": m1, "m2": m2},
            weights={"m1": 0.7, "m2": 0.3},
        )
        ensemble.fit(self.df[self.features], self.df["revenue"])
        self.assertTrue(ensemble.is_fitted)

        preds = ensemble.predict(self.val_df[self.features])
        self.assertEqual(len(preds), len(self.val_df))
        self.assertTrue(np.all(preds >= 0.0))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_weighted_ensemble_weight_optimization(self):
        m1 = TransformedTargetTreeModel(
            transform_type="log1p", objective="huber",
            n_estimators=15, num_leaves=10, random_state=42
        )
        m2 = LogTransformedTreeModel(
            model_type="random_forest", use_log_target=True,
            n_estimators=15, max_depth=8, random_state=42
        )

        ensemble = WeightedLogEnsemble(base_models={"m1": m1, "m2": m2})
        ensemble.fit(self.df[self.features], self.df["revenue"])

        opt_weights = ensemble.optimize_weights(
            self.val_df[self.features],
            self.val_df["revenue"]
        )

        self.assertAlmostEqual(sum(opt_weights.values()), 1.0, places=5)
        for w in opt_weights.values():
            self.assertGreaterEqual(w, -1e-6)

    def test_weighted_ensemble_serialization(self):
        m1 = TransformedTargetTreeModel(
            transform_type="log1p", objective="huber",
            n_estimators=10, num_leaves=8, random_state=42
        )
        ensemble = WeightedLogEnsemble(base_models={"m1": m1})
        ensemble.fit(self.df[self.features], self.df["revenue"])
        orig_preds = ensemble.predict(self.val_df[self.features])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_ensemble.joblib"
            ensemble.save(save_path)
            loaded = WeightedLogEnsemble.load(save_path)

            loaded_preds = loaded.predict(self.val_df[self.features])
            np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5)

    def test_stacking_ensemble(self):
        def factory():
            return {
                "m1": TransformedTargetTreeModel(
                    transform_type="log1p", objective="huber",
                    n_estimators=10, num_leaves=8, random_state=42
                ),
                "m2": LogTransformedTreeModel(
                    model_type="random_forest", use_log_target=True,
                    n_estimators=10, max_depth=6, random_state=42
                ),
            }

        stacking = StackingLogEnsemble(base_models_factory=factory, n_splits=3)
        stacking.fit(self.df[self.features], self.df["revenue"])
        self.assertTrue(stacking.is_fitted)

        preds = stacking.predict(self.val_df[self.features])
        self.assertEqual(len(preds), len(self.val_df))
        self.assertTrue(np.all(preds >= 0.0))
        self.assertAlmostEqual(sum(stacking.meta_weights_.values()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
