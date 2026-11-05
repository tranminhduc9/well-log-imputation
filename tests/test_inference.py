"""Checkpoint loading: weights, observed values, fingerprints and portable paths."""

from dataclasses import asdict
from pathlib import Path, PurePosixPath, PureWindowsPath
import os
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import torch

from src.experiments.inference import restore_run, _load_joblib_checkpoint
from src.experiments.utils import save_json, sha256
from src.models.base import ModelConfig
from src.models.conv_saits import ConvSAITS, ConvSAITSConfig


class ForeignPath:
    def __reduce__(self):
        import pathlib
        cls = pathlib.PosixPath if os.name == "nt" else pathlib.WindowsPath
        return cls, ("/kaggle/working/training",)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.values = np.random.default_rng(42).normal(size=(2, 100, 2)).astype(np.float32)
        self.artifact = self.root / 'config.json'
        config = ModelConfig(seq_len=100, n_features=2, device='cpu', seed=129)
        save_json(self.artifact, asdict(config))
        self.entry = dict(model='locf', seed=129, artifact=str(self.artifact), sha256=sha256(self.artifact))

    def test_locf_restore_preserves_observed_points(self):
        model = restore_run(self.entry, device='cpu')
        inputs = self.values.copy()
        mask = np.zeros_like(inputs, dtype=bool)
        mask[:, :20, 0] = True
        inputs[mask] = np.nan
        result = model.impute(inputs)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[~mask], inputs[~mask])

    def test_changed_checkpoint_and_mismatched_seed_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'seed differs'):
            restore_run({**self.entry, 'seed': 219}, device='cpu')
        self.artifact.write_bytes(self.artifact.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            restore_run(self.entry, device='cpu')

    def test_requested_cuda_is_not_silently_changed_to_cpu(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'CUDA requested but unavailable'):
                restore_run(self.entry, device='cuda')

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
        values = self.values[:1].copy()
        values[:, :20, 0] = np.nan
        np.testing.assert_allclose(model.impute({"X": values}), restored.impute({"X": values}))

    def test_foreign_paths_in_joblib_checkpoint(self):
        checkpoint = self.root / "model.joblib"
        joblib.dump(dict(config=dict(output_dir=ForeignPath()), models=[np.arange(10)]), checkpoint)
        saved = _load_joblib_checkpoint(checkpoint)
        self.assertIsInstance(saved["config"]["output_dir"], (PurePosixPath, PureWindowsPath))
        np.testing.assert_array_equal(saved["models"][0], np.arange(10))


if __name__ == '__main__':
    unittest.main()
