"""Tests for the XGBoost imputation baseline."""

import unittest

import numpy as np

from src.models.xgboost import XGBoost, XGBoostConfig


class XGBoostTests(unittest.TestCase):
    def test_impute_fills_missing_and_preserves_observations(self) -> None:
        feature = np.linspace(-1.0, 1.0, 40)
        train = np.column_stack((feature, 2 * feature + 1)).reshape(4, 10, 2)
        incomplete = train[:1].copy()
        incomplete[0, 3:6, 1] = np.nan

        model = XGBoost(
            XGBoostConfig(
                seq_len=10,
                n_features=2,
                n_estimators=20,
                max_depth=2,
                n_jobs=1,
            )
        )
        result = model.fit(train).impute(incomplete)

        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(
            result[~np.isnan(incomplete)],
            incomplete[~np.isnan(incomplete)],
        )

    def test_config_rejects_invalid_sampling_fraction(self) -> None:
        with self.assertRaises(ValueError):
            XGBoostConfig(subsample=0.0)


if __name__ == "__main__":
    unittest.main()
