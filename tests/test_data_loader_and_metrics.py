"""Tests for processed loading and model-level metric delegation."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.data.metrics import compute_imputation_metrics
from src.data.loader import DataLoader
from src.models.model import AbstractModel, ModelConfig


class _Backend:
    def __init__(self) -> None:
        self.train_set = None
        self.val_set = None

    def fit(self, train_set, val_set=None) -> None:
        self.train_set = train_set
        self.val_set = val_set

    def predict(self, dataset):
        imputation = np.asarray(dataset["X"], dtype=float).copy()
        imputation[np.isnan(imputation)] = 0.0
        return {"imputation": imputation}


class _Model(AbstractModel):
    name = "test-model"

    def _build_backend(self):
        return _Backend()


class DataLoaderAndMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        parameters = {"segment_length": 3, "log_columns": ["A", "B"]}
        (self.root / "preprocessing_parameters.json").write_text(
            json.dumps(parameters), encoding="utf-8"
        )

        self.segments = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        self.mask = np.zeros_like(self.segments, dtype=bool)
        self.mask[:, 1, 0] = True
        for split in ("train", "val", "test"):
            split_path = self.root / split
            split_path.mkdir()
            np.save(split_path / "segments.npy", self.segments)
            pd.DataFrame({"WELL": ["A", "B"]}).to_csv(
                split_path / "metadata.csv", index=False
            )
        np.save(self.root / "val" / "mask_block_20.npy", self.mask)
        np.save(self.root / "test" / "mask_block_20.npy", self.mask)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_training_loader_returns_model_dataset(self) -> None:
        loader = DataLoader(self.root, split="train")

        dataset = loader.load()

        np.testing.assert_array_equal(dataset["X"], self.segments)
        self.assertEqual(loader.load_metadata()["WELL"].tolist(), ["A", "B"])

    def test_evaluation_loader_applies_fixed_mask(self) -> None:
        loader = DataLoader(self.root, split="val", scenario="Block-20")

        dataset = loader.load()

        np.testing.assert_array_equal(dataset["X_intact"], self.segments)
        np.testing.assert_array_equal(dataset["indicating_mask"], self.mask)
        self.assertTrue(np.isnan(dataset["X"][self.mask]).all())
        np.testing.assert_array_equal(dataset["X"][~self.mask], self.segments[~self.mask])

    def test_model_accepts_loaders_and_uses_metrics_module(self) -> None:
        train_loader = DataLoader(self.root, split="train")
        val_loader = DataLoader(self.root, split="val", scenario="block_20")
        model = _Model(ModelConfig(seq_len=3, n_features=2))

        model.fit(train_loader, val_loader)
        metrics = model.evaluate(val_loader)

        self.assertEqual(model.backend.train_set["X"].shape, (2, 3, 2))
        self.assertEqual(model.backend.val_set["X"].shape, (2, 3, 2))
        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["mae"], 5.0)

    def test_metric_bundle_scores_only_masked_positions(self) -> None:
        estimate = self.segments.copy()
        estimate[self.mask] = 0

        metrics = compute_imputation_metrics(self.segments, estimate, self.mask)

        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["mse"], 34.0)
        self.assertEqual(metrics["mape"], 100.0)


if __name__ == "__main__":
    unittest.main()
