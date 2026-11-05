"""BRITS = forward RITS + backward RITS + consistency loss.

Tensors use (batch, depth, logs). A mask of 1 means observed, 0 means missing.
Depth steps are equally spaced; deltas count samples, not physical metres.
"""

import math
import torch
from torch import nn


class TemporalDecay(nn.Module):
    """Learn gamma(delta) = exp(-ReLU(W delta + b)) in (0, 1]."""

    def __init__(self, input_size, output_size, diagonal=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.register_buffer("mask", torch.eye(input_size) if diagonal else None)
        self._reset_parameters()

    def _reset_parameters(self):
        """Match the parameter initialization used by PyPOTS."""

        bound = 1.0 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, delta):
        weight = self.weight if self.mask is None else self.weight * self.mask
        return torch.exp(-torch.relu(nn.functional.linear(delta, weight, self.bias)))


class FeatureRegression(nn.Module):
    """Predict each log from the other logs; diagonal weights stay zero."""

    def __init__(self, n_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, n_features))
        self.bias = nn.Parameter(torch.empty(n_features))
        self.register_buffer("mask", 1 - torch.eye(n_features))
        self._reset_parameters()

    def _reset_parameters(self):
        """Match the parameter initialization used by PyPOTS."""

        bound = 1.0 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, values):
        return nn.functional.linear(values, self.weight * self.mask, self.bias)


class RITS(nn.Module):
    """Fill a sequence in one direction, one depth sample at a time."""

    def __init__(self, n_features, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(n_features * 2, hidden_size)
        self.history = nn.Linear(hidden_size, n_features)
        self.feature = FeatureRegression(n_features)
        self.decay_h = TemporalDecay(n_features, hidden_size)
        self.decay_x = TemporalDecay(n_features, n_features, diagonal=True)
        self.combine = nn.Linear(n_features * 2, n_features)

    def forward(self, values, masks, deltas, compute_loss=True):
        batch_size, sequence_length, _ = values.shape
        hidden = values.new_zeros(batch_size, self.hidden_size)
        cell = values.new_zeros(batch_size, self.hidden_size)
        imputations = []
        loss = values.new_zeros(())

        for step in range(sequence_length):
            x, mask, delta = values[:, step], masks[:, step], deltas[:, step]

            # 1. Older observations contribute less after a long gap.
            hidden = hidden * self.decay_h(delta)
            history = self.history(hidden)

            # 2. Fill from history, then predict from the other logs.
            completed = mask * x + (1 - mask) * history
            feature = self.feature(completed)

            # 3. Learn how much to trust the two estimates.
            weight = torch.sigmoid(self.combine(torch.cat((self.decay_x(delta), mask), -1)))
            estimate = weight * feature + (1 - weight) * history
            if compute_loss:
                loss = loss + sum(_masked_mae(part, x, mask)
                                  for part in (history, feature, estimate))

            # 4. Keep observed values and feed the filled sample to the LSTM.
            completed = mask * x + (1 - mask) * estimate
            hidden, cell = self.rnn(torch.cat((completed, mask), 1), (hidden, cell))
            imputations.append(completed)

        # Equal weight for the three estimates and for each depth step.
        return torch.stack(imputations, 1), loss / (sequence_length * 3)


class BRITSNetwork(nn.Module):
    """Average two RITS estimates and penalize disagreement between directions."""

    def __init__(self, n_features, hidden_size, consistency_weight):
        super().__init__()
        self.forward_rits = RITS(n_features, hidden_size)
        self.backward_rits = RITS(n_features, hidden_size)
        self.consistency_weight = consistency_weight

    def forward(self, values, masks, return_components=False, compute_loss=True):
        # Recompute deltas after reversing: backward gaps differ from forward gaps.
        forward, forward_loss = self.forward_rits(values, masks, _deltas(masks), compute_loss)

        reverse_values = torch.flip(values, (1,))
        reverse_masks = torch.flip(masks, (1,))
        backward, backward_loss = self.backward_rits(
            reverse_values,
            reverse_masks,
            _deltas(reverse_masks),
            compute_loss,
        )
        backward = torch.flip(backward, (1,))

        consistency = (torch.mean(torch.abs(forward - backward)) if compute_loss
                       else values.new_tensor(0.0))
        reconstruction = forward_loss + backward_loss
        loss = reconstruction + self.consistency_weight * consistency
        imputation = (forward + backward) / 2
        if return_components:
            return imputation, loss, reconstruction, consistency
        return imputation, loss


def _masked_mae(prediction, target, mask):
    return torch.sum(torch.abs(prediction - target) * mask) / (torch.sum(mask) + 1e-5)


def _deltas(masks):
    """Unit depth steps since the latest observation for every log."""

    deltas = torch.zeros_like(masks)
    for step in range(1, masks.shape[1]):
        deltas[:, step] = 1 + (1 - masks[:, step - 1]) * deltas[:, step - 1]
    return deltas
