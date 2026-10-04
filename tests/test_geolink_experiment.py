"""Regression tests for the notebook's training/evaluation protocol."""

import argparse
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.data.metrics import compute_imputation_metrics
from src.experiments.geolink import (
    evaluate_details, load_and_check_data, make_config, run_experiment, summarize,
)
from src.models.losses import masked_imputation_mae
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

    def test_metrics_reject_invalid_predictions_shapes_and_masks(self):
        truth, mask = np.array([1., 2.]), np.array([1, 1])
        for invalid in (np.nan, np.inf, -np.inf):
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                compute_imputation_metrics(truth, [1., invalid], mask)
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            compute_imputation_metrics(truth, [1.], mask)
        with self.assertRaisesRegex(ValueError, "binary"):
            compute_imputation_metrics(truth, truth, [1, 2])
        scored = compute_imputation_metrics(truth, [1., np.nan], [1, 0], include_mape=False)
        self.assertEqual(scored["count"], 1)
        self.assertNotIn("mape", scored)

    def test_mit_balances_segments_and_handles_empty_masks(self):
        prediction = torch.tensor([[[2.], [9.]], [[4.], [4.]]], requires_grad=True)
        mask = torch.tensor([[[1.], [0.]], [[1.], [1.]]])
        truth = torch.zeros_like(prediction)
        loss = masked_imputation_mae(prediction, truth, mask)
        self.assertAlmostEqual(loss.item(), 3.)
        self.assertAlmostEqual(masked_imputation_mae(prediction, truth, mask, "point").item(), 10/3, places=6)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[[.5], [0.]], [[.25], [.25]]]))
        zero = masked_imputation_mae(prediction, truth, torch.zeros_like(mask))
        self.assertEqual(zero.item(), 0.)
        self.assertTrue(zero.requires_grad)

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
                                metadata["val"], preprocessing, "locf", 129, "val")
        original = [row for row in rows if row["units"] == "original"]
        self.assertEqual(len(original), 2)
        for row in original:
            self.assertAlmostEqual(row["mae"], preprocessing["normalization"][row["group"]]["std"], places=5)
            self.assertIn("mape", row)
        self.assertTrue(all("mape" not in row for row in rows if row["units"] == "normalized"))
        summary = summarize(rows)
        self.assertTrue(summary["mae_std"].isna().all())

    def test_full_runner_two_seeds_checkpoint_reload_and_isolation(self):
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
                manifest_path = next((self.root / "results").glob("*/experiment_results.json"))
                self.assertEqual(len(json.loads(manifest_path.read_text())["runs"]), 9)
            return original_evaluate(*positional, **kwargs)

        with patch("src.experiments.geolink.evaluate_details", side_effect=assert_phase):
            output = run_experiment(args, PROJECT_ROOT, settings)
        result_text = (output / "experiment_results.json").read_text()
        self.assertNotIn(": NaN", result_text)
        result = json.loads(result_text)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["runs"]), 9)
        self.assertTrue(Path(str(output) + ".zip").is_file())
        self.assertTrue((output / "source_snapshot/src/experiments/geolink.py").is_file())
        for entry in result["runs"]:
            self.assertTrue((output / entry["artifact"]).is_file())
            self.assertEqual(len(entry["sha256"]), 64)
        summary = pd.read_csv(output / "summary.csv")
        learned = summary["model"] != "locf"
        self.assertTrue((summary.loc[learned, "n_runs"] == 2).all())
        self.assertTrue(np.isfinite(summary.loc[learned, "rmse_std"]).all())
        self.assertTrue(summary.loc[~learned, "rmse_std"].isna().all())
        self.assertEqual(set(summary["split"]), {"val", "test"})

        args.models, args.evaluate_test = ("locf",), False
        second = run_experiment(args, PROJECT_ROOT, settings)
        self.assertNotEqual(output, second)
        self.assertEqual(json.loads((output / "experiment_results.json").read_text())["status"], "complete")
        self.assertEqual(set(pd.read_csv(second / "summary.csv")["split"]), {"val"})
        self.assertTrue(pd.read_csv(second / "training_history.csv").empty)

    def test_notebook_cells_compile_and_include_both_seeds(self):
        notebook = json.loads((PROJECT_ROOT / "notebook/geolink.ipynb").read_text(encoding="utf-8"))
        sources = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                sources.append(source)
                if not source.lstrip().startswith("%"):
                    compile(source, f"cell_{index}", "exec")
        self.assertIn("SEEDS = (912, 219)", "\n".join(sources))

    def test_notebook_selected_configurations_are_valid(self):
        notebook = json.loads((PROJECT_ROOT / "notebook/geolink.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                           if cell["cell_type"] == "code" and
                           not "".join(cell["source"]).lstrip().startswith("%"))
        constants = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign):
                if isinstance(node.targets[0], ast.Name) and node.targets[0].id in {
                    "SEEDS", "NEURAL_EPOCHS", "NEURAL_PATIENCE", "MODEL_SETTINGS"
                }:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
        run_cell = ast.parse(source)
        assignment = next(node for node in run_cell.body if isinstance(node, ast.Assign) and
                          isinstance(node.targets[0], ast.Name) and node.targets[0].id == "args")
        scope = {"Namespace": argparse.Namespace, "DATA_ROOT": self.data, **constants}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "args", "exec"), scope)
        args = scope["args"]
        self.assertEqual(args.seeds, (912, 219))
        self.assertEqual(args.xgb_estimators, 300)
        for name in args.models:
            config = make_config(name, {"seq_len": 256, "n_features": 4},
                                 constants["MODEL_SETTINGS"], 219, args, self.root)
            self.assertEqual(config.seed, 219)
            if name in {"saits", "brits", "conv_saits"}:
                self.assertEqual(config.epochs, 500)
                self.assertEqual(config.patience, 30)
                self.assertEqual(config.mit_reduction, "segment")

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
        with patch("src.experiments.geolink.evaluate_details", side_effect=ValueError("Bad prediction")):
            with self.assertLogs("src.experiments.geolink", level="ERROR"):
                with self.assertRaisesRegex(ValueError, "Bad prediction"):
                    run_experiment(args, PROJECT_ROOT, {})
        output = next(args.output_dir.iterdir())
        result = json.loads((output / "experiment_results.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["runs"]), 1)
        self.assertTrue((output / result["runs"][0]["artifact"]).is_file())


if __name__ == "__main__":
    unittest.main()
