"""Point-level exports, frozen checkpoint loading and well ranking regressions."""
from dataclasses import asdict
from pathlib import Path, PurePosixPath, PureWindowsPath
import json
import tempfile
import unittest
import os

import numpy as np
import pandas as pd
import torch

from src.experiments.geolink import save_json, sha256
from src.experiments.inference import (
    discover_runs, run_test_inference, load_bundle, point_table, rank_wells,
    representative_wells, restore_run, plot_segment, plot_comparison,
)
from src.models.model import ModelConfig
from src.models.conv_saits import ConvSAITS, ConvSAITSConfig


class ForeignPath:
    def __reduce__(self):
        import pathlib
        cls = pathlib.PosixPath if os.name == "nt" else pathlib.WindowsPath
        return cls, ("/kaggle/working/training",)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.parameters = dict(segment_length=100, log_columns=["A", "B"],
            missing_scenarios=["Single", "Block-20", "Block-100", "Entire-Log"],
            normalization={"A": dict(mean=10., std=2.), "B": dict(mean=20., std=3.)})
        save_json(self.data / "preprocessing_parameters.json", self.parameters)
        rng = np.random.default_rng(42)
        for split in ("train", "val", "test"):
            folder = self.data / split
            folder.mkdir()
            values = rng.normal(size=(4, 100, 2)).astype(np.float32)
            np.save(folder / "segments.npy", values)
            pd.DataFrame(dict(WELL=[split+"_1"]*2 + [split+"_2"]*2,
                START_DEPTH=[1000, 1010, 1200, 1210], END_DEPTH=[1009.9, 1019.9, 1209.9, 1219.9])).to_csv(folder / "metadata.csv", index=False)
            if split != "train":
                for scenario, length in zip(self.parameters["missing_scenarios"], [1, 20, 100, 100]):
                    mask = np.zeros_like(values, dtype=bool)
                    for index in range(4):
                        mask[index, :length, index % 2] = True
                    np.save(folder / f"mask_{scenario.lower().replace('-', '_')}.npy", mask)
        self.results = self.root / "training"
        self.results.mkdir()
        artifact = self.results / "config.json"
        save_json(artifact, asdict(ModelConfig(seq_len=100, n_features=2, device="cpu", seed=129)))
        save_json(self.results / "preprocessing_parameters.json", self.parameters)
        save_json(self.results / "experiment_results.json", dict(runs=[dict(
            model="locf", seed=129, artifact="config.json", sha256=sha256(artifact))]))

    def test_inference_roundtrip_export_and_plot(self):
        runs, _ = discover_runs(self.results)
        output = run_test_inference(runs, self.results, self.data, self.root / "output", device="cpu")
        manifest = json.loads((output / "inference_manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(manifest["bundles"]), 4)
        bundle = load_bundle(output, "locf", 129, "Block-20")
        np.testing.assert_array_equal(bundle["actual"], np.load(self.data / "test/segments.npy"))
        np.testing.assert_array_equal(bundle["predict"][~bundle["mask"]], bundle["actual"][~bundle["mask"]])
        points = pd.read_csv(output / "run_00/seed_129/block_20_points.csv.gz")
        self.assertEqual(len(points), 80)
        self.assertTrue(points.is_missing.all())
        a = points[points.log == "A"]
        np.testing.assert_allclose(a.actual, a.actual_normalized * 2 + 10)
        np.testing.assert_allclose(a.error, a.predict - a.actual)
        metrics = pd.read_csv(output / "metrics_by_seed.csv")
        overall = metrics[(metrics.level == "overall") & (metrics.scenario == "Block-20")].iloc[0]
        self.assertAlmostEqual(overall.rmse, np.sqrt(np.mean((points.predict_normalized-points.actual_normalized)**2)))
        self.assertEqual(len(pd.read_csv(output / "segment_metrics.csv")), 16)
        rank = rank_wells(metrics)
        self.assertEqual(len(rank), 2)
        self.assertTrue(rank.score_std.isna().all())
        representatives = representative_wells(rank, "locf")
        self.assertEqual(representatives.well.nunique(), 2)
        metadata = pd.read_csv(output / "test_metadata.csv")
        full = point_table(bundle["actual"], bundle["predict"], bundle["mask"], metadata,
                           self.parameters, bundle["depth"], masked_only=False)
        self.assertEqual(len(full), 800)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, scored = plot_segment(bundle, self.parameters, 0)
        self.assertEqual(scored["count"], 20)
        fig.savefig(output / "example.png")
        plt.close(fig)

    def test_ranking_equal_scenario_weight_and_seed_mean(self):
        rows = []
        for well, errors in [("A", [0., 0., 0., 8.]), ("B", [3., 3., 3., 3.])]:
            for seed in (129, 219):
                for scenario, error, count in zip(self.parameters["missing_scenarios"], errors, [1,20,100,100]):
                    rows.append(dict(trial="trial", model="conv_saits", seed=seed, level="well", units="normalized",
                        group=well, scenario=scenario, rmse=error, mae=error, count=count))
        rank = rank_wells(pd.DataFrame(rows))
        self.assertEqual(rank.iloc[0].well, "A")
        self.assertEqual(rank.iloc[0].score_mean, 2.)
        self.assertEqual(rank.iloc[0].n_seeds, 2)
        self.assertEqual(rank_wells(pd.DataFrame(rows), "Entire-Log").iloc[0].well, "B")
        with self.assertRaisesRegex(ValueError, "every selected scenario"):
            rank_wells(pd.DataFrame(rows[:-1]))

    def test_checkpoint_corruption_and_preprocessing_change_rejected(self):
        runs, _ = discover_runs(self.results)
        original = (self.results / "config.json").read_bytes()
        (self.results / "config.json").write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            restore_run(runs.iloc[0].to_dict(), "cpu")
        (self.results / "config.json").write_bytes(original)
        modified = {**self.parameters, "log_columns": ["B", "A"]}
        save_json(self.results / "preprocessing_parameters.json", modified)
        with self.assertRaisesRegex(ValueError, "preprocessing differs"):
            run_test_inference(runs, self.results, self.data, self.root / "output", device="cpu")

    def test_neural_checkpoint_weights_and_cpu_restore(self):
        config = ConvSAITSConfig(seq_len=100, n_features=2, d_model=16, d_inner=16,
            n_layers=1, n_heads=2, encoder_channels=4, kernel_size=3, seed=129, device="cpu")
        model = ConvSAITS(config)
        model._is_fitted = True
        checkpoint = self.root / "best.pt"
        saved_config = {**asdict(config), "output_dir": ForeignPath()}
        torch.save(dict(config=saved_config, state_dict=model.backend.network.state_dict()), checkpoint)
        entry = dict(model="conv_saits", seed=129, artifact=str(checkpoint), sha256=sha256(checkpoint))
        restored = restore_run(entry, "cpu", batch_size=1)
        values = np.load(self.data / "test/segments.npy")[:1].copy()
        values[:, :20, 0] = np.nan
        np.testing.assert_allclose(model.impute({"X": values}), restored.impute({"X": values}))

    def test_foreign_paths_in_joblib_checkpoint(self):
        import joblib
        from src.experiments.inference import _load_joblib_checkpoint
        checkpoint = self.root / "model.joblib"
        joblib.dump(dict(config=dict(output_dir=ForeignPath()), models=[np.arange(10)]), checkpoint)
        saved = _load_joblib_checkpoint(checkpoint)
        self.assertIsInstance(saved["config"]["output_dir"], (PurePosixPath, PureWindowsPath))
        np.testing.assert_array_equal(saved["models"][0], np.arange(10))

    def test_parallel_runs_keep_both_predictions_and_compare_same_mask(self):
        runs, _ = discover_runs(self.results)
        second = {**runs.iloc[0].to_dict(), "trial": "locf_reference"}
        output = run_test_inference([runs.iloc[0].to_dict(), second], self.results,
            self.data, self.root / "parallel", device="cpu", workers=2)
        manifest = json.loads((output / "inference_manifest.json").read_text())
        self.assertEqual(len(manifest["selected_runs"]), 2)
        self.assertEqual(len(manifest["bundles"]), 8)
        self.assertEqual(len(pd.read_csv(output / "summary.csv")), 8)
        first = load_bundle(output, "locf", 129, "Block-20")
        second = load_bundle(output, "locf_reference", 129, "Block-20")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, comparison = plot_comparison({"first": first, "second": second}, self.parameters, 0)
        self.assertEqual(len(comparison), 2)
        self.assertEqual(comparison.iloc[0].rmse, comparison.iloc[1].rmse)
        plt.close(fig)
        second["mask"][0, 25, 0] = True
        with self.assertRaisesRegex(ValueError, "identical targets"):
            plot_comparison({"first": first, "second": second}, self.parameters, 0)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
