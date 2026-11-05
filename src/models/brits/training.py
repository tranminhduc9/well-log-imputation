"""Train BRITS with reconstruction, consistency and optional MIT losses."""

import logging
import time
import numpy as np
import torch
from torch import nn
from src.models.losses import masked_imputation_mae
from src.models._training import (
    EarlyStopping, training_loader, validation_metrics, predict_batches, synchronized_time,
)
from src.preprocessing.pipeline import BLOCK_LENGTHS, MISSING_SCENARIOS
from .config import BRITSConfig
from .network import BRITSNetwork

LOGGER = logging.getLogger("src.models.brits")


class _BRITSBackend:
    def __init__(self, config):
        self.config = config
        wants_gpu = config.device.lower() in {"gpu", "cuda"}
        self.device = torch.device("cuda" if wants_gpu and torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.seed)
        self.network = BRITSNetwork(
            config.n_features,
            config.hidden_size,
            config.consistency_weight,
        ).to(self.device)
        LOGGER.info("BRITS actual device: %s", self.device)
        if wants_gpu and self.device.type != "cuda":
            LOGGER.warning("BRITS requested GPU but CUDA is unavailable; using CPU.")
        self.training_history = []
        self.best_epoch = None

    def _mixed_missing_mask(self, values, random_state):
        """Sample one evaluation-style missingness scenario per segment."""

        generator = np.random.default_rng(random_state)
        segment_count, sequence_length, feature_count = values.shape
        available_scenarios = [
            scenario
            for scenario in MISSING_SCENARIOS
            if BLOCK_LENGTHS.get(scenario, 1) <= sequence_length
        ]
        scenarios = generator.choice(available_scenarios, size=segment_count)
        hidden = np.zeros(values.shape, dtype=bool)
        for segment, scenario in enumerate(scenarios):
            feature = generator.integers(feature_count)
            if scenario == "Single":
                step = generator.integers(sequence_length)
                hidden[segment, step, feature] = True
            elif scenario in BLOCK_LENGTHS:
                length = BLOCK_LENGTHS[scenario]
                start = generator.integers(sequence_length - length + 1)
                hidden[segment, start : start + length, feature] = True
            else:
                hidden[segment, :, feature] = True
        return hidden

    def _training_masks(self, train_set, truth, epoch):
        finite_truth = np.isfinite(truth)
        input_observed = np.isfinite(np.asarray(train_set["X"])) & finite_truth
        predefined = train_set.get("indicating_mask")
        if predefined is not None:
            hidden = np.asarray(predefined, dtype=bool) & finite_truth
        elif self.config.masking_strategy == "mixed":
            hidden = self._mixed_missing_mask(
                truth, random_state=self.config.seed + epoch
            )
            hidden &= input_observed
        elif self.config.masking_strategy == "random":
            generator = np.random.default_rng(self.config.seed + epoch)
            hidden = (
                generator.random(truth.shape) < self.config.masking_rate
            ) & input_observed
        else:
            hidden = np.zeros(truth.shape, dtype=bool)

        if (
            self.config.masking_strategy != "none"
            and predefined is None
            and not np.any(hidden)
            and np.any(input_observed)
        ):
            generator = np.random.default_rng(self.config.seed + epoch)
            hidden.flat[generator.choice(np.flatnonzero(input_observed))] = True
        observed = input_observed & ~hidden
        return observed, hidden

    def fit(self, train_set, val_set=None):
        truth = np.asarray(
            train_set.get("X_intact", train_set["X"]), dtype=np.float32
        )
        if not np.any(np.isfinite(truth) & np.isfinite(train_set["X"])):
            raise ValueError("BRITS training data has no observed values.")
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.training_history = []
        self.best_epoch = None
        stopping = EarlyStopping(self.config.patience, self.config.min_delta)
        training_started = time.perf_counter()
        for epoch in range(self.config.epochs):
            epoch_started = synchronized_time(self.device)
            observed, hidden = self._training_masks(train_set, truth, epoch)
            loader = training_loader(truth, observed, hidden, self.config.batch_size,
                                     pin_memory=self.device.type == "cuda")
            self.network.train()
            totals = torch.zeros(4, device=self.device)
            for batch, batch_truth, masks, batch_hidden in loader:
                batch, batch_truth, masks, batch_hidden = (
                    tensor.to(self.device, non_blocking=self.device.type == "cuda")
                    for tensor in (batch, batch_truth, masks, batch_hidden)
                )
                imputation, brits_loss, reconstruction, consistency = self.network(
                    batch, masks, return_components=True
                )
                mit = masked_imputation_mae(
                    imputation, batch_truth, batch_hidden, self.config.mit_reduction
                )
                loss = brits_loss + self.config.mit_weight * mit
                if not torch.isfinite(loss):
                    raise RuntimeError("BRITS training loss is non-finite.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.config.gradient_clip
                    )
                optimizer.step()
                totals += torch.stack((loss, reconstruction, consistency, mit)).detach()

            train_finished = synchronized_time(self.device)
            mean_loss, reconstruction_mean, consistency_mean, mit_mean = (totals / len(loader)).tolist()
            validation = self._validation_metrics(val_set) if val_set is not None else None
            validation_mse = None if validation is None else validation["mse"]
            validation_rmse = None if validation is None else validation["rmse"]
            # GeoLink reports RMSE, so checkpoint selection uses the same
            # metric by default. MSE remains available for PyPOTS parity.
            score = (
                validation[self.config.validation_metric]
                if validation is not None
                else mean_loss
            )
            if not np.isfinite(score):
                raise RuntimeError(f"BRITS produced a non-finite selection score at epoch {epoch + 1}.")
            validation_finished = synchronized_time(self.device)
            history_point = {
                "train_seconds": train_finished - epoch_started,
                "validation_seconds": validation_finished - train_finished,
                "epoch_seconds": validation_finished - epoch_started,
                "epoch": epoch + 1,
                "loss": mean_loss,
                "reconstruction_loss": reconstruction_mean,
                "consistency_loss": consistency_mean,
                "mit_loss": mit_mean,
                "validation_mse": validation_mse,
                "validation_rmse": validation_rmse,
            }
            if validation is not None:
                for scenario, metrics in validation["by_scenario"].items():
                    key = scenario.lower().replace("-", "_").replace(" ", "_")
                    history_point[f"validation_{key}_mse"] = metrics["mse"]
                    history_point[f"validation_{key}_rmse"] = metrics["rmse"]
            self.training_history.append(history_point)
            elapsed = time.perf_counter() - training_started
            LOGGER.info(
                "BRITS epoch %d/%d | loss=%.6f | validation_mse=%s | "
                "validation_rmse=%s | elapsed=%.1fs",
                epoch + 1,
                self.config.epochs,
                mean_loss,
                f"{validation_mse:.6f}" if validation_mse is not None else "n/a",
                f"{validation_rmse:.6f}" if validation_rmse is not None else "n/a",
                elapsed,
            )

            stopped = stopping.update(score, self.network, epoch + 1)
            self.best_epoch = stopping.best_epoch
            if stopped:
                LOGGER.info("BRITS early stopping at epoch %d | best epoch=%d | score=%.6f",
                            epoch + 1, self.best_epoch, stopping.best_score)
                break
        stopping.restore(self.network)

    def _validation_metrics(self, dataset):
        return validation_metrics(dataset, self.predict, self.config.batch_size, "BRITS")

    def _validation_rmse(self, dataset):
        """Retain the previous helper API for callers that only need RMSE."""

        return self._validation_metrics(dataset)["rmse"]

    def _predict_batch(self, values, depth=None):
        batch = torch.from_numpy(values).to(self.device)
        observed = torch.isfinite(batch).float()
        imputation, _ = self.network(torch.nan_to_num(batch), observed, compute_loss=False)
        return imputation.cpu().numpy()

    def predict(self, dataset):
        return predict_batches(self.network, self._predict_batch, dataset, self.config.batch_size)
