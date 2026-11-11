"""Train FlexLogNet with the shared masks, validation selection and checkpoints."""

import logging
import math

import numpy as np
import torch

from src.models._training import (
    EarlyStopping, training_loader, validation_metrics, predict_batches, synchronized_time,
)
from src.preprocessing.pipeline import create_missing_mask
from .losses import flexlognet_loss
from .network import FlexLogNetNetwork

LOGGER = logging.getLogger("src.models.flexlognet")


class _FlexLogNetBackend:
    def __init__(self, config):
        self.config = config
        wants_gpu = config.device.lower() in {"gpu", "cuda"}
        self.device = torch.device("cuda" if wants_gpu and torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.seed)
        self.network = FlexLogNetNetwork(config).to(self.device)
        self.training_history = []
        self.best_epoch = None

    def fit(self, train_set, val_set=None):
        truth = np.asarray(train_set.get("X_intact", train_set["X"]), dtype=np.float32)
        original_observed = np.isfinite(np.asarray(train_set["X"])) & np.isfinite(truth)
        predefined = train_set.get("indicating_mask")
        if not original_observed.any():
            raise ValueError("FlexLogNet training data has no observed values.")
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        stopping = EarlyStopping(self.config.patience, self.config.min_delta)
        self.training_history, self.best_epoch = [], None
        for epoch in range(self.config.epochs):
            started = synchronized_time(self.device)
            if predefined is None:
                hidden = create_missing_mask(truth, random_state=self.config.seed + epoch)
                hidden &= original_observed
            else:
                hidden = np.asarray(predefined, dtype=bool) & np.isfinite(truth)
            observed = original_observed & ~hidden
            if not hidden.any():
                raise ValueError("FlexLogNet requires known targets to mask for training.")
            loader = training_loader(truth, observed, hidden, self.config.batch_size,
                                     seed=self.config.seed + epoch)
            self.network.train()
            total = 0.0
            for inputs, targets, observed_batch, hidden_batch in loader:
                inputs, targets, observed_batch, hidden_batch = (
                    tensor.to(self.device) for tensor in (inputs, targets, observed_batch, hidden_batch))
                optimizer.zero_grad(set_to_none=True)
                output = self.network(inputs, observed_batch)
                loss = flexlognet_loss(output, targets, observed_batch, hidden_batch, self.config)
                if not torch.isfinite(loss):
                    raise RuntimeError("FlexLogNet training loss is non-finite.")
                loss.backward()
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip,
                                                   error_if_nonfinite=True)
                optimizer.step()
                total += loss.item()
            train_finished = synchronized_time(self.device)
            train_loss = total / len(loader)
            score = (validation_metrics(val_set, self.predict, self.config.batch_size, "FlexLogNet")["rmse"]
                     if val_set is not None else train_loss)
            if not math.isfinite(score):
                raise RuntimeError("FlexLogNet selection score is non-finite.")
            finished = synchronized_time(self.device)
            self.training_history.append({
                "epoch": epoch + 1, "loss": train_loss,
                "validation_rmse": score if val_set is not None else float("nan"),
                "train_seconds": train_finished - started,
                "validation_seconds": finished - train_finished, "epoch_seconds": finished - started,
            })
            LOGGER.info("FlexLogNet epoch %d/%d | loss=%.6f | selection_score=%.6f",
                        epoch + 1, self.config.epochs, train_loss, score)
            stopped = stopping.update(score, self.network, epoch + 1)
            self.best_epoch = stopping.best_epoch
            if stopped:
                break
        stopping.restore(self.network)

    def _predict_batch(self, values, depth=None):
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        return self.network(values, torch.isfinite(values))["imputation"].cpu().numpy()

    def predict(self, dataset):
        return predict_batches(self.network, self._predict_batch, dataset, self.config.batch_size)
