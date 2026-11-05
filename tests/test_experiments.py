"""Experiment runner, split validation and masked evaluation regressions."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.experiments.runner import run_experiment
from src.experiments.data import load_and_check_data
from src.experiments.evaluation import evaluate_details, summarize
from src.preprocessing.pipeline import MISSING_SCENARIOS, BLOCK_LENGTHS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    parameters = {
        "segment_length": 100, "log_columns": ["A", "B"],
        "missing_scenarios": list(MISSING_SCENARIOS),
        "normalization": {"A": {"mean": 10., "std": 2.},
                          "B": {"mean": 20., "std": 3.}},
    }
    root.mkdir()
    (root / "preprocessing_parameters.json").write_text(json.dumps(parameters))
    rng = np.random.default_rng(912)
    for split in ("train", "val", "test"):
        folder = root / split
        folder.mkdir()
        values = rng.normal(size=(4, 100, 2)).astype(np.float32)
        np.save(folder / "segments.npy", values)
        pd.DataFrame({
            "WELL": [split + "_1"] * 2 + [split + "_2"] * 2,
            "START_DEPTH": [1000., 1010., 1200., 1210.],
            "END_DEPTH": [1009.9, 1019.9, 1209.9, 1219.9],
        }).to_csv(folder / "metadata.csv", index=False)
        if split != "train":
            for scenario in MISSING_SCENARIOS:
                mask = np.zeros_like(values, dtype=bool)
                length = BLOCK_LENGTHS.get(scenario, 1 if scenario == "Single" else 100)
                for segment in range(4):
                    mask[segment, :length, segment % 2] = True
                np.save(folder / f"mask_{scenario.lower().replace('-', '_')}.npy", mask)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        fixture(self.data)


    def test_disjoint_wells_and_mask_topology(self):
        load_and_check_data(self.data, True)
        pd.DataFrame({"WELL": ["train_1"] * 4}).to_csv(self.data / "val/metadata.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Well leakage"):
            load_and_check_data(self.data, True)

    def test_invalid_masks_fail_before_training(self):
        path = self.data / "val/mask_block_20.npy"
        mask = np.load(path)
        mask[0, 0, 0], mask[0, 30, 0] = False, True
        np.save(path, mask)
        with self.assertRaisesRegex(ValueError, "gap topology"):
            load_and_check_data(self.data, True)

    def test_test_split_not_opened_when_disabled(self):
        np.save(self.data / "test/segments.npy", np.array([np.nan]))
        _, data, _, hashes = load_and_check_data(self.data, False)
        self.assertNotIn("test", data)
        self.assertFalse(any(Path(name).parts[0] == "test" for name in hashes))

    def test_original_unit_metrics_and_no_target_exposure(self):
        preprocessing, data, metadata, _ = load_and_check_data(self.data, True)

        class Predictor:
            def impute(self, payload):
                assert set(payload) == {"X", "depth"}
                assert payload["depth"].shape == payload["X"].shape[:2]
                intact = data["val"]["Single"]["X_intact"]
                return np.where(np.isnan(payload["X"]), intact + 1, payload["X"])

        rows = evaluate_details(Predictor(), {"Single": data["val"]["Single"]},
                                preprocessing, "locf", 129, "val")
        original = [row for row in rows if row["units"] == "original"]
        self.assertEqual(len(original), 2)
        for row in original:
            self.assertAlmostEqual(row["mae"], preprocessing["normalization"][row["group"]]["std"], places=5)
            self.assertIn("mape", row)
        self.assertTrue(all("mape" not in row for row in rows if row["units"] == "normalized"))
        summary = summarize(rows)
        self.assertTrue(summary["mae_std"].isna().all())

    def test_full_runner_two_seeds_checkpoint_reload_and_stable_paths(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        args = argparse.Namespace(
            models=("locf", "xgboost", "saits", "brits", "conv_saits"), seeds=(129, 219),
            data_root=self.data, output_dir=self.root / "results", evaluate_test=True,
            xgb_estimators=2,
            saits_epochs=1, saits_patience=1,
            brits_epochs=1, brits_patience=1,
            conv_saits_epochs=1, conv_saits_patience=1,
        )
        small = {"d_model": 8, "d_inner": 8, "n_heads": 2, "n_layers": 1, "batch_size": 2}
        settings = {"xgboost": {"n_jobs": 1, "max_depth": 2},
                    "saits": small,
                    "brits": {"hidden_size": 4, "batch_size": 2},
                    "conv_saits": {**small, "encoder_channels": 4, "kernel_size": 3}}
        original_evaluate = evaluate_details
        def assert_phase(*positional, **kwargs):
            if positional[-1] == "test":
                manifest_path = self.root / "results" / "experiment_results.json"
                self.assertEqual(len(json.loads(manifest_path.read_text())["runs"]), 9)
            return original_evaluate(*positional, **kwargs)

        with patch("src.experiments.runner.evaluate_details", side_effect=assert_phase):
            output = run_experiment(args, PROJECT_ROOT, settings)
        self.assertEqual(output, args.output_dir.resolve())
        result_text = (output / "experiment_results.json").read_text()
        self.assertNotIn(": NaN", result_text)
        result = json.loads(result_text)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["runs"]), 9)
        self.assertFalse(Path(str(output) + ".zip").exists())  # Notebooks create the final ZIP.
        self.assertIn("src/experiments/runner.py", result["provenance"]["source_sha256"])
        for entry in result["runs"]:
            self.assertTrue((output / entry["artifact"]).is_file())
            self.assertEqual(len(entry["sha256"]), 64)
        summary = pd.read_csv(output / "summary.csv")
        learned = summary["model"] != "locf"
        self.assertTrue((summary.loc[learned, "n_runs"] == 2).all())
        self.assertTrue(np.isfinite(summary.loc[learned, "rmse_std"]).all())
        self.assertTrue(summary.loc[~learned, "rmse_std"].isna().all())
        self.assertEqual(set(summary["split"]), {"val", "test"})

        artifact_paths = [entry["artifact"] for entry in result["runs"] if entry["model"] == "locf"]
        (output / "training.log").write_text("OLD RUN SENTINEL\n")
        args.models, args.evaluate_test = ("locf",), False
        second = run_experiment(args, PROJECT_ROOT, settings)
        self.assertEqual(output, second)
        rerun = json.loads((second / "experiment_results.json").read_text())
        self.assertEqual(rerun["status"], "complete")
        self.assertEqual([entry["artifact"] for entry in rerun["runs"]], artifact_paths)
        self.assertNotIn("OLD RUN SENTINEL", (second / "training.log").read_text())
        self.assertEqual(rerun["run_id"], "results")
        self.assertEqual(set(pd.read_csv(second / "summary.csv")["split"]), {"val"})
        self.assertTrue(pd.read_csv(second / "training_history.csv").empty)


    def test_sample_std_and_duplicate_seed_rejection(self):
        row = {"model": "saits", "label": "SAITS", "split": "val", "scenario": "Single",
               "level": "overall", "group": "all", "units": "normalized", "count": 4}
        rows = [{**row, "seed": 129, "rmse": 1.}, {**row, "seed": 219, "rmse": 3.}]
        summary = summarize(rows).iloc[0]
        self.assertEqual(summary["rmse_mean"], 2.)
        self.assertAlmostEqual(summary["rmse_std"], np.sqrt(2.))
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            summarize([rows[0], rows[0]])

    def test_failed_scoring_preserves_checkpoint_and_status(self):
        args = argparse.Namespace(models=("locf",), seeds=(129, 219),
                                  data_root=self.data, output_dir=self.root / "failed",
                                  evaluate_test=False)
        with patch("src.experiments.runner.evaluate_details", side_effect=ValueError("Bad prediction")):
            with self.assertLogs("src.experiments.runner", level="ERROR"):
                with self.assertRaisesRegex(ValueError, "Bad prediction"):
                    run_experiment(args, PROJECT_ROOT, {})
        output = args.output_dir
        result = json.loads((output / "experiment_results.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["runs"]), 1)
        self.assertTrue((output / result["runs"][0]["artifact"]).is_file())


if __name__ == "__main__":
    unittest.main()
