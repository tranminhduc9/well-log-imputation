"""BRITS regression checks for predictions, losses and gradients."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.models._training import validation_metrics
from src.models.brits import BRITS, BRITSConfig, BRITSNetwork, _deltas, _masked_mae


def reference_deltas(masks):
    deltas = torch.zeros_like(masks)
    for step in range(1, masks.shape[1]):
        deltas[:, step] = 1 + (1 - masks[:, step - 1]) * deltas[:, step - 1]
    return deltas


def reference_rits(network, values, masks):
    """Original step-by-step forward, including its loss normalization."""
    deltas = reference_deltas(masks)
    hidden = values.new_zeros(len(values), network.hidden_size)
    cell = torch.zeros_like(hidden)
    loss = values.new_tensor(0.)
    imputations = []
    for step in range(values.shape[1]):
        x, mask, delta = values[:, step], masks[:, step], deltas[:, step]
        hidden = hidden * network.decay_h(delta)
        history = network.history(hidden)
        loss = loss + _masked_mae(history, x, mask)
        completed = mask * x + (1 - mask) * history
        feature = network.feature(completed)
        loss = loss + _masked_mae(feature, x, mask)
        weight = torch.sigmoid(network.combine(torch.cat((network.decay_x(delta), mask), 1)))
        estimate = weight * feature + (1 - weight) * history
        loss = loss + _masked_mae(estimate, x, mask)
        completed = mask * x + (1 - mask) * estimate
        hidden, cell = network.rnn(torch.cat((completed, mask), 1), (hidden, cell))
        imputations.append(completed)
    return torch.stack(imputations, 1), loss / (values.shape[1] * 3)


def reference_network(network, values, masks):
    forward, forward_loss = reference_rits(network.forward_rits, values, masks)
    backward, backward_loss = reference_rits(network.backward_rits,
                                            values.flip(1), masks.flip(1))
    backward = backward.flip(1)
    consistency = torch.mean(torch.abs(forward - backward))
    reconstruction = forward_loss + backward_loss
    return ((forward + backward) / 2,
            reconstruction + network.consistency_weight * consistency,
            reconstruction, consistency)


class BRITSNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_deltas_match_recurrence(self):
        torch.manual_seed(23)
        for length in (1, 7, 256):
            for observed_fraction in (0., .2, .8, 1.):
                masks = (torch.rand(3, length, 4) < observed_fraction).float()
                torch.testing.assert_close(_deltas(masks), reference_deltas(masks), rtol=0, atol=0)
                torch.testing.assert_close(_deltas(masks.flip(1)), reference_deltas(masks.flip(1)),
                                           rtol=0, atol=0)

    def test_predictions_losses_and_gradients_match_reference(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype, length in ((torch.float32, 11), (torch.float64, 11),
                                  (torch.float32, 256), (torch.float64, 256)):
                with self.subTest(device=device, dtype=dtype, length=length):
                    torch.manual_seed(23)
                    network_under_test = BRITSNetwork(4, 8, .1).to(device=device, dtype=dtype)
                    reference = deepcopy(network_under_test)
                    masks = (torch.rand(3, length, 4, device=device) > .3).to(dtype)
                    masks[:, 3] = 0  # A whole timestep with an empty denominator.
                    masks[:, :, 0] = 0  # An entirely missing log.
                    values = (torch.randn_like(masks) * masks).requires_grad_()
                    reference_values = values.detach().clone().requires_grad_()
                    actual = network_under_test(values, masks, return_components=True)
                    expected = reference_network(reference, reference_values, masks)
                    tolerance = 2e-6 if dtype == torch.float32 else 1e-10
                    for new, old in zip(actual, expected):
                        torch.testing.assert_close(new, old, rtol=tolerance, atol=tolerance)
                    # Include imputation supervision so both recurrent paths
                    # are checked beyond the reconstruction objective alone.
                    (actual[1] + actual[0].square().mean()).backward()
                    (expected[1] + expected[0].square().mean()).backward()
                    torch.testing.assert_close(values.grad, reference_values.grad,
                                               rtol=tolerance, atol=tolerance)
                    for (name, new), (_, old) in zip(network_under_test.named_parameters(), reference.named_parameters()):
                        with self.subTest(parameter=name):
                            torch.testing.assert_close(new.grad, old.grad, rtol=tolerance, atol=tolerance)

    def test_prediction_path_skips_loss_and_preserves_imputation(self):
        torch.manual_seed(23)
        network = BRITSNetwork(4, 8, .1)
        for observed_fraction in (0., .5, 1.):
            masks = (torch.rand(3, 7, 4) < observed_fraction).float()
            values = torch.randn_like(masks) * masks
            with torch.no_grad():
                expected, _ = network(values, masks)
                # Neither reconstruction nor consistency should calculate abs.
                with patch("src.models.brits.network.torch.abs", side_effect=AssertionError("Unused loss calculated")):
                    actual, loss = network(values, masks, compute_loss=False)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(loss.item(), 0.)

    def test_larger_validation_batches_preserve_all_scenario_scores(self):
        config = BRITSConfig(seq_len=128, n_features=4, hidden_size=8,
                             batch_size=2, validation_batch_size=5, device="cpu")
        backend = BRITS(config).backend
        truth = np.random.default_rng(42).normal(size=(7, 128, 4)).astype(np.float32)
        scenarios = {}
        for name, length in (("Single", 1), ("Block-20", 20), ("Block-100", 100), ("Entire-Log", 128)):
            mask = np.zeros_like(truth, dtype=bool)
            mask[:, :length, 0] = True
            scenarios[name] = dict(X=np.where(mask, np.nan, truth), X_intact=truth, indicating_mask=mask)
        dataset = dict(validation_scenarios=scenarios)
        expected = validation_metrics(dataset, backend.predict, config.batch_size, "BRITS")
        actual = backend._validation_metrics(dataset)
        for name in ("mse", "rmse"):
            self.assertAlmostEqual(actual[name], expected[name], places=6)
            for scenario in scenarios:
                self.assertAlmostEqual(actual["by_scenario"][scenario][name],
                                       expected["by_scenario"][scenario][name], places=6)
        for invalid in (0, -1, 1.5):
            with self.subTest(validation_batch_size=invalid), self.assertRaises(ValueError):
                BRITSConfig(validation_batch_size=invalid)


if __name__ == "__main__":
    unittest.main()
