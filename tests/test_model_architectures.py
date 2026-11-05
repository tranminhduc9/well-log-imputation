"""Architectural contracts from the SAITS/Conv-SAITS figures and BRITS flow."""

from dataclasses import asdict
import io
import unittest

import numpy as np
import torch

from src.models._attention import AttentionLayer
from src.models.brits import BRITS, BRITSConfig, BRITSNetwork, FeatureRegression
from src.models.conv_saits import ConvSAITS, ConvSAITSConfig, ConvSAITSNetwork
from src.models.saits import SAITS, SAITSConfig, SAITSNetwork


def small_config(config_class):
    common = dict(seq_len=8, n_features=3, batch_size=2, epochs=1, patience=1, device="cpu")
    if config_class is BRITSConfig:
        return config_class(**common, hidden_size=5)
    attention = dict(d_model=12, d_inner=16, n_heads=2, n_layers=1,
                     dropout=0, attn_dropout=0)
    if config_class is ConvSAITSConfig:
        attention.update(encoder_channels=2)
    return config_class(**common, **attention)


class ArchitectureTests(unittest.TestCase):
    def test_diagonal_attention_cannot_copy_same_depth_sample(self):
        config = small_config(SAITSConfig)
        layer = AttentionLayer(config).eval()
        _, attention = layer(torch.randn(2, 8, 12))
        torch.testing.assert_close(attention.diagonal(dim1=-2, dim2=-1),
                                   torch.zeros(2, 2, 8), rtol=0, atol=0)
        torch.testing.assert_close(attention.sum(-1), torch.ones(2, 2, 8))

    def test_saits_and_conv_saits_stage_two_inputs_match_figures(self):
        values = torch.randn(2, 8, 3)
        observed = (torch.rand_like(values) > .4).float()
        values = values * observed
        for config_class, network_class in [(SAITSConfig, SAITSNetwork),
                                              (ConvSAITSConfig, ConvSAITSNetwork)]:
            with self.subTest(model=network_class.__name__):
                network = network_class(small_config(config_class)).eval()
                captured = {}
                embedding = (network.second_embedding if config_class is ConvSAITSConfig
                             else network.input_second)
                handle = embedding.register_forward_pre_hook(
                    lambda module, args: captured.update(second=args[0].detach()))
                encoder_handle = None
                if config_class is ConvSAITSConfig:
                    encoder_handle = network.encoder.register_forward_hook(
                        lambda module, args, result: captured.update(encoded=result.detach()))
                try:
                    output = network(values, observed)
                finally:
                    handle.remove()
                    if encoder_handle is not None:
                        encoder_handle.remove()
                filled = observed * values + (1 - observed) * output[1]
                extra = captured['encoded'] if config_class is ConvSAITSConfig else observed
                torch.testing.assert_close(captured['second'], torch.cat((filled, extra), -1))
                torch.testing.assert_close(output[0][observed.bool()], values[observed.bool()])

    def test_conv_saits_default_is_multiscale_and_handles_all_missing(self):
        network = ConvSAITSNetwork(small_config(ConvSAITSConfig)).eval()
        encoder = network.encoder
        self.assertEqual([branch.kernel_size[0] for branch in encoder.temporal_branches],
                         [3, 7, 15])
        for branch in encoder.temporal_branches:
            self.assertEqual(branch.groups, branch.in_channels)
        output = network(torch.zeros(2, 8, 3), torch.zeros(2, 8, 3))
        self.assertEqual(output[0].shape, (2, 8, 3))
        self.assertTrue(all(torch.isfinite(part).all() for part in output))

    def test_brits_feature_regression_excludes_target_log(self):
        regression = FeatureRegression(3)
        values = torch.randn(2, 3)
        original = regression(values)
        for target in range(3):
            changed = values.clone()
            changed[:, target] += 100
            torch.testing.assert_close(regression(changed)[:, target], original[:, target])

    def test_brits_consistency_and_both_recurrent_directions(self):
        network = BRITSNetwork(3, 5, .1)
        mask = torch.ones(2, 8, 3)
        mask[:, 2:6, 0] = 0
        mask[:, :, 1] = 0
        values = torch.randn_like(mask) * mask
        imputed, loss, reconstruction, consistency = network(values, mask, return_components=True)
        torch.testing.assert_close(loss, reconstruction + .1 * consistency)
        torch.testing.assert_close(imputed[mask.bool()], values[mask.bool()])
        loss.backward()
        for direction in (network.forward_rits, network.backward_rits):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in direction.parameters()))

    def test_neural_models_train_and_reload_through_public_imports(self):
        truth = np.random.default_rng(42).normal(size=(2, 8, 3)).astype(np.float32)
        mask = np.zeros_like(truth, dtype=bool)
        mask[:, 2:6, 0] = True
        inputs = truth.copy()
        inputs[mask] = np.nan
        dataset = dict(X=inputs, X_intact=truth, indicating_mask=mask)
        for model_class, config_class in [(BRITS, BRITSConfig), (SAITS, SAITSConfig),
                                           (ConvSAITS, ConvSAITSConfig)]:
            with self.subTest(model=model_class.__name__):
                model = model_class(small_config(config_class)).fit(dataset, dataset)
                prediction = model.impute(inputs)
                self.assertTrue(np.isfinite(prediction).all())
                np.testing.assert_array_equal(prediction[~mask], truth[~mask])
                self.assertEqual(model.backend.best_epoch, 1)
                self.assertEqual(len(model.backend.training_history), 1)
                self.assertGreaterEqual(model.backend.training_history[0]['epoch_seconds'], 0)
                buffer = io.BytesIO()
                saved = dict(config=asdict(model.config), state_dict=model.backend.network.state_dict())
                # Store config as plain data, as done by the experiment runner.
                saved['config']['output_dir'] = str(saved['config']['output_dir'])
                torch.save(saved, buffer)
                buffer.seek(0)
                saved = torch.load(buffer, weights_only=True)
                restored = model_class(config_class(**saved['config']))
                restored.backend.network.load_state_dict(saved['state_dict'], strict=True)
                restored._is_fitted = True
                np.testing.assert_allclose(restored.impute(inputs), prediction, rtol=1e-6, atol=1e-6)
                self.assertEqual(restored.impute(inputs[:0]).shape, (0, 8, 3))

    def test_single_scale_checkpoint_still_loads_with_explicit_configuration(self):
        from dataclasses import replace
        config = replace(small_config(ConvSAITSConfig), multi_scale_conv=False, kernel_size=3)
        original = ConvSAITSNetwork(config).eval()
        restored = ConvSAITSNetwork(ConvSAITSConfig(**asdict(config))).eval()
        restored.load_state_dict(original.state_dict(), strict=True)
        values, mask = torch.randn(2, 8, 3), torch.ones(2, 8, 3)
        torch.testing.assert_close(original(values, mask)[0], restored(values, mask)[0])


if __name__ == '__main__':
    unittest.main()
