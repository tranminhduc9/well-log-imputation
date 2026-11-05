"""Conv-SAITS training and inference, including optional research objectives."""

import logging
import math
from typing import Any
import numpy as np
import torch
from src.models.losses import (
    masked_imputation_mae, masked_imputation_mse,
    masked_reconstruction_mae as _masked_mae,
)
from src.models._training import (
    EarlyStopping, training_loader, validation_metrics, predict_batches, synchronized_time,
)
from src.preprocessing.pipeline import create_missing_mask
from .config import ConvSAITSConfig
from .network import ConvSAITSNetwork
from .ablation import validation_objectives

LOGGER = logging.getLogger("src.models.conv_saits.model")


class _ConvSAITSBackend:
    def __init__(self, config: ConvSAITSConfig) -> None:
        self.config = config
        wants_gpu = config.device.lower() in {"gpu", "cuda"}
        self.device = torch.device("cuda" if wants_gpu and torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.seed)
        self.network = ConvSAITSNetwork(config).to(self.device)
        self.training_history: list[dict[str, float | int]] = []
        self.best_epoch: int | None = None

    def _training_masks(self, train_set, truth, epoch):
        finite_truth = np.isfinite(truth)
        input_observed = np.isfinite(np.asarray(train_set["X"])) & finite_truth
        predefined = train_set.get("indicating_mask")
        if predefined is not None:
            hidden = np.asarray(predefined, dtype=bool) & finite_truth
        elif self.config.masking_strategy == "mixed":
            # Sample Single, Block-20, Block-100 or Entire-Log per segment.
            hidden = create_missing_mask(truth, random_state=self.config.seed + epoch)
            hidden &= input_observed
        else:
            generator = np.random.default_rng(self.config.seed + epoch)
            hidden = (generator.random(truth.shape) < self.config.masking_rate) & input_observed
            if not np.any(hidden) and np.any(input_observed):
                candidates = np.flatnonzero(input_observed)
                hidden.flat[generator.choice(candidates)] = True
        observed = input_observed & ~hidden
        return observed, hidden

    def fit(self, train_set, val_set=None):
        truth = np.asarray(train_set.get("X_intact", train_set["X"]), dtype=np.float32)
        if not np.any(np.isfinite(truth) & np.isfinite(train_set["X"])):
            raise ValueError("Conv-SAITS training data has no observed values.")

        optimizer = self._make_optimizer()
        stopping = EarlyStopping(self.config.patience, self.config.min_delta)
        self.training_history = []
        self.best_epoch = None

        for epoch in range(self.config.epochs):
            epoch_started = synchronized_time(self.device)
            observed, hidden = self._training_masks(train_set, truth, epoch)
            if not np.any(hidden):
                raise ValueError("Conv-SAITS requires at least one known value for MIT training.")
            depth = train_set.get("depth")
            if self.config.depth_encoding:
                if depth is None:
                    raise ValueError("Training data needs depth when depth_encoding=True.")
                depth = np.asarray(depth, dtype=np.float32)
                if depth.shape != truth.shape[:2] or not np.isfinite(depth).all():
                    raise ValueError("Training depth must be finite and aligned with X.")
            else:
                depth = np.zeros(truth.shape[:2], dtype=np.float32)
            loader = training_loader(
                truth, observed, hidden, self.config.batch_size, depth=depth,
                seed=self.config.seed + epoch if self.config.independent_shuffle else None,
            )

            self.network.train()
            totals = dict.fromkeys(
                ("loss", "ort_loss", "mit_loss", "mit_mse_loss", "mit_relative_loss"), 0.0
            )
            for batch_inputs, batch_truth, batch_observed, batch_hidden, batch_depth in loader:
                batch_inputs, batch_truth, batch_observed, batch_hidden, batch_depth = (
                    tensor.to(self.device) for tensor in
                    (batch_inputs, batch_truth, batch_observed, batch_hidden, batch_depth)
                )
                _, first, second, combined, _ = self.network(batch_inputs, batch_observed, batch_depth)
                ort = sum(
                    _masked_mae(part, batch_truth, batch_observed)
                    for part in (first, second, combined)
                ) / 3
                mit = masked_imputation_mae(
                    combined, batch_truth, batch_hidden, self.config.mit_reduction
                )
                loss = self.config.ort_weight * ort + self.config.mit_weight * mit
                mit_mse = relative = combined.new_zeros(())
                if self.config.mit_mse_weight:
                    mit_mse = masked_imputation_mse(
                        combined, batch_truth, batch_hidden, self.config.mit_reduction
                    )
                    loss = loss + self.config.mit_mse_weight * mit_mse
                if self.config.mit_relative_weight:
                    scales = batch_truth.new_tensor(self.config.physical_stds)
                    means = batch_truth.new_tensor(self.config.physical_means)
                    floors = batch_truth.new_tensor(self.config.relative_floors)
                    physical_truth = batch_truth * scales + means
                    denominator = physical_truth.abs().clamp_min(floors)
                    relative_mask = batch_hidden
                    relative_prediction = (combined - batch_truth) * scales / denominator
                    relative = masked_imputation_mae(relative_prediction, torch.zeros_like(combined),
                                                      relative_mask, self.config.mit_reduction)
                    loss = loss + self.config.mit_relative_weight * relative
                if not torch.isfinite(loss):
                    raise RuntimeError("Conv-SAITS training loss is non-finite.")
                optimizer.zero_grad()
                loss.backward()
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.config.gradient_clip,
                        error_if_nonfinite=True,
                    )
                optimizer.step()
                for key, value in zip(totals, (loss, ort, mit, mit_mse, relative)):
                    totals[key] += float(value.item())

            train_finished = synchronized_time(self.device)
            losses = {key: value / len(loader) for key, value in totals.items()}
            train_loss = losses["loss"]
            validation_rmse, validation_mape = float("nan"), float("nan")
            if val_set is None:
                score = train_loss
            elif self.config.validation_objective == "rmse_mape":
                validation_rmse, validation_mape = self._validation_objectives(val_set)
                score = max(validation_rmse / self.config.reference_rmse,
                            validation_mape / self.config.reference_mape)
            else:
                score = validation_rmse = self._validation_rmse(val_set)
            if not math.isfinite(score):
                raise RuntimeError(
                    "Conv-SAITS produced a non-finite selection score at "
                    f"epoch {epoch + 1}."
                )
            validation_finished = synchronized_time(self.device)
            self.training_history.append(
                {
                    "train_seconds": train_finished - epoch_started,
                    "validation_seconds": validation_finished - train_finished,
                    "epoch_seconds": validation_finished - epoch_started,
                    "epoch": epoch + 1,
                    **losses,
                    "validation_rmse": validation_rmse,
                    **({"validation_mape": validation_mape}
                       if self.config.validation_objective == "rmse_mape" else {}),
                    "validation_score": score,
                }
            )
            LOGGER.info(
                "Conv-SAITS epoch %d/%d | loss=%.6f | selection_score=%.6f",
                epoch + 1, self.config.epochs, train_loss, score,
            )
            stopped = stopping.update(score, self.network, epoch + 1)
            self.best_epoch = stopping.best_epoch
            if stopped:
                LOGGER.info("Conv-SAITS early stopping at epoch %d; best epoch=%d", epoch + 1, self.best_epoch)
                break
        stopping.restore(self.network)

    def _make_optimizer(self) -> torch.optim.Optimizer:
        configured = self.config.optimizer
        if configured is None:
            return torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        if isinstance(configured, torch.optim.Optimizer):
            return configured
        if callable(configured):
            return configured(self.network.parameters(), lr=self.config.learning_rate)
        raise TypeError("optimizer must be an optimizer instance, callable, or None.")

    def _validation_rmse(self, dataset):
        return validation_metrics(dataset, self.predict, self.config.batch_size, "Conv-SAITS")["rmse"]

    def _validation_objectives(self, dataset):
        return validation_objectives(self.config, self.predict, dataset)

    def _predict_batch(self, values: np.ndarray, depth: np.ndarray | None = None) -> np.ndarray:
        batch = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        observed = torch.isfinite(batch).float()
        clean = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
        batch_depth = None
        if depth is not None:
            batch_depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        imputation, _, _, _, _ = self.network(clean, observed, batch_depth)
        return imputation.cpu().numpy()

    def predict(self, dataset: dict[str, Any]) -> dict[str, np.ndarray]:
        values = np.asarray(dataset["X"], dtype=np.float32)
        depth = dataset.get("depth")
        if self.config.depth_encoding:
            if depth is None:
                raise ValueError("Prediction data needs depth when depth_encoding=True.")
            depth = np.asarray(depth, dtype=np.float32)
            if depth.shape != values.shape[:2] or not np.isfinite(depth).all():
                raise ValueError("Prediction depth must be finite and aligned with X.")
        return predict_batches(self.network, self._predict_batch, dataset, self.config.batch_size)
