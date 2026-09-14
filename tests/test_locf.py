"""Tests for the LOCF baseline."""

import unittest

import numpy as np

from src.models.locf import LOCF
from src.models.model import ModelConfig


class LOCFTests(unittest.TestCase):
    def test_impute(self) -> None:
        values = np.array(
            [
                [
                    [np.nan, np.nan],
                    [1.0, np.nan],
                    [np.nan, np.nan],
                    [3.0, np.nan],
                    [np.nan, np.nan],
                ]
            ]
        )
        expected = np.array(
            [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [3.0, 0.0], [3.0, 0.0]]]
        )

        model = LOCF(ModelConfig(seq_len=5, n_features=2))
        result = model.impute(values)

        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
