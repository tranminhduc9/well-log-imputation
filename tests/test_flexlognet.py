"""FlexLogNet graph masking, target isolation and two-head training contracts."""

from dataclasses import asdict, replace
import unittest

import numpy as np
from scipy.interpolate import interp1d
import torch

from src.models.flexlognet import FlexLogNet, FlexLogNetConfig, FlexLogNetNetwork
from src.models.flexlognet.losses import flexlognet_loss, masked_ssim_loss
from src.models.flexlognet.network import HeterogeneousAggregation, MultiscaleInput


def small_config(**overrides):
    parameters = dict(seq_len=32, n_features=4, encoder_channels=(2, 4), stats_hidden=8,
                      batch_size=2, epochs=2, patience=2, device="cpu", dropout=0)
    return FlexLogNetConfig(**(parameters | overrides))


class FlexLogNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_multiscale_matches_median_and_quadratic_interpolation(self):
        values = torch.arange(32, dtype=torch.float32).reshape(1, 1, 32).square()
        output = MultiscaleInput(32, (4, 8))(values)
        for index, factor in enumerate((4, 8), 1):
            coarse = np.median(values.numpy().reshape(-1, factor), axis=-1)
            expected = interp1d(np.linspace(0, 1, len(coarse)), coarse, kind="quadratic")(
                np.linspace(0, 1, 32))
            np.testing.assert_allclose(output[0, 0, index].numpy(), expected, rtol=1e-5, atol=1e-4)

    def test_graph_excludes_self_edges_and_unobserved_sources(self):
        graph = HeterogeneousAggregation(3, 4, 0).eval()
        with torch.no_grad():
            graph.query.weight.zero_()
            graph.query.bias.zero_()
            graph.key.weight.zero_()
            graph.key.bias.zero_()
            graph.depth_maps.copy_(torch.eye(4).expand(6, -1, -1))
        values = torch.arange(12, dtype=torch.float32).reshape(1, 3, 1, 4)
        output = graph(values, torch.tensor([[True, False, True]]))
        torch.testing.assert_close(output[:, 0], values[:, 2])
        torch.testing.assert_close(output[:, 1], (values[:, 0] + values[:, 2]) / 2)
        torch.testing.assert_close(output[:, 2], values[:, 0])
        changed = values.clone()
        changed[:, 1] += 1000
        torch.testing.assert_close(graph(changed, torch.tensor([[True, False, True]])), output)
        torch.testing.assert_close(graph(values, torch.zeros(1, 3, dtype=torch.bool)), torch.zeros_like(values))

    def test_observed_values_and_targets_do_not_leak_into_gaps(self):
        network = FlexLogNetNetwork(small_config()).eval()
        values = torch.randn(5, 32, 4)
        observed = torch.ones_like(values, dtype=torch.bool)
        observed[0, 4, 0] = False
        observed[1, 3:23, 1] = False
        observed[2, :, 2] = False
        observed[3, :, 0] = False
        observed[4] = False
        changed = values.clone()
        changed[~observed] = float("nan")
        with torch.no_grad():
            output = network(values, observed)
            other = network(changed, observed)
            single = network(changed[2:3], observed[2:3])
        self.assertTrue(torch.isfinite(output["imputation"]).all())
        torch.testing.assert_close(output["imputation"], other["imputation"])
        torch.testing.assert_close(output["imputation"][observed], values[observed], rtol=0, atol=0)
        torch.testing.assert_close(single["imputation"], output["imputation"][2:3], rtol=1e-5, atol=1e-6)
        self.assertTrue((output["std"] > 0).all())

    def test_both_heads_receive_finite_gradients(self):
        config = small_config()
        network = FlexLogNetNetwork(config)
        truth = torch.randn(2, 32, 4)
        hidden = torch.zeros_like(truth)
        hidden[0, :, 0] = 1
        hidden[1, 4:24, 3] = 1
        observed = 1 - hidden
        output = network(truth, observed)
        loss = flexlognet_loss(output, truth, observed, hidden, config)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for prefix in ("encoders", "decoders", "aggregations", "statistics"):
            gradients = [parameter.grad for name, parameter in network.named_parameters() if name.startswith(prefix)]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
            self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients))

    def test_ssim_ignores_unscored_values_and_is_zero_for_identical_curves(self):
        truth = torch.randn(2, 32, 4)
        mask = torch.zeros_like(truth)
        mask[0, 8:24, 0] = 1
        mask[1, 4, 3] = 1
        prediction = truth.clone().requires_grad_()
        loss = masked_ssim_loss(prediction, truth, mask)
        self.assertAlmostEqual(loss.item(), 0, places=6)
        other = truth.clone()
        other[~mask.bool()] = float("nan")
        torch.testing.assert_close(masked_ssim_loss(prediction, other, mask), loss)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_fit_reload_partial_batches_and_constant_inputs(self):
        truth = np.random.default_rng(9).normal(size=(3, 32, 4)).astype(np.float32)
        truth[:, :, 1] = 0
        hidden = np.zeros_like(truth, dtype=bool)
        hidden[0, :, 0] = True
        hidden[1, 4:24, 1] = True
        hidden[2, 10, 2] = True
        values = np.where(hidden, np.nan, truth)
        dataset = dict(X=values, X_intact=truth, indicating_mask=hidden)
        model = FlexLogNet(small_config()).fit(dataset, dataset)
        prediction = model.impute(dataset)
        np.testing.assert_array_equal(prediction[~hidden], truth[~hidden])
        self.assertTrue(np.isfinite(prediction).all())
        self.assertIsNotNone(model.backend.best_epoch)
        self.assertEqual(len(model.backend.training_history), 2)
        misleading = dict(dataset, X_intact=truth + 10000)
        np.testing.assert_array_equal(model.impute(misleading), prediction)
        restored = FlexLogNet(FlexLogNetConfig(**(asdict(model.config) | {"batch_size": 1})))
        restored.backend.network.load_state_dict(model.backend.network.state_dict(), strict=True)
        restored._is_fitted = True
        np.testing.assert_allclose(restored.impute(values), prediction, rtol=1e-5, atol=1e-6)
        self.assertEqual(restored.impute(values[:0]).shape, values[:0].shape)
        with self.assertRaisesRegex(ValueError, "no observed"):
            FlexLogNet(small_config()).fit(np.full_like(truth, np.nan))

    def test_dynamic_training_masks_and_odd_length(self):
        config = small_config(seq_len=101, epochs=1)
        values = np.random.default_rng(91).normal(size=(3, 101, 4)).astype(np.float32)
        model = FlexLogNet(config).fit(values)
        for log in range(4):
            hidden = np.zeros_like(values, dtype=bool)
            hidden[:, :, log] = True
            result = model.impute(values, hidden)
            self.assertEqual(result.shape, values.shape)
            self.assertTrue(np.isfinite(result).all())

    def test_invalid_configs(self):
        for settings in ({"n_features": 1}, {"encoder_channels": ()}, {"kernel_size": 2},
                         {"encoder_channels": (0,)}, {"resample_factors": (4, 4)},
                         {"statistics_weight": float("nan")}, {"dropout": 1},
                         {"gradient_clip": 0}, {"seq_len": 3}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                replace(small_config(), **settings)


if __name__ == "__main__":
    unittest.main()
