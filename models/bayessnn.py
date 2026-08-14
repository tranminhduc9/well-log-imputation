"""Approximate Bayesian neural-network imputer with predictive intervals."""

from copy import deepcopy
import logging
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _BayesianRegressor(nn.Module):
    """Heteroscedastic MLP whose dropout remains active during MC inference."""

    def __init__(self, n_inputs: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_inputs, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, x):
        output = self.output(self.backbone(x))
        return output[:, 0], output[:, 1].clamp(-10.0, 6.0)


class BayesianNNImputer:
    """Bayesian approximation using MC Dropout and heteroscedastic Gaussian NLL."""

    def __init__(
        self,
        num_models: int,
        batch_size: int = 32,
        epochs: int = 300,
        patience: int = 15,
        device: Optional[Union[str, torch.device]] = None,
        learning_rate: float = 1e-3,
        hidden_size: int = 64,
        dropout: float = 0.15,
        mc_samples: int = 50,
        prediction_batch_size: int = 8192,
        min_delta: float = 1e-4,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
    ) -> None:
        self.num_models = num_models
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.mc_samples = mc_samples
        self.prediction_batch_size = prediction_batch_size
        self.min_delta = min_delta
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        requested = str(device or "cpu")
        self.device = torch.device(
            requested if requested != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.models = {}
        self.scalers = {}

    @staticmethod
    def _arrays(data: dict[str, np.ndarray], target: int):
        X = np.asarray(data["X"], dtype=np.float32)
        X_intact = np.asarray(data["X_intact"], dtype=np.float32)
        y = X_intact[:, :, target].reshape(-1)
        x = np.delete(X, target, axis=2).reshape(-1, X.shape[2] - 1)
        valid = np.isfinite(y)
        return np.nan_to_num(x[valid], nan=0.0, posinf=0.0, neginf=0.0), y[valid]

    @staticmethod
    def _gaussian_nll(mean, log_variance, target):
        squared_error = (target - mean).square()
        return 0.5 * (log_variance + squared_error * torch.exp(-log_variance)).mean()

    def fit(self, train_set: dict[str, np.ndarray], val_set=None) -> None:
        for target in range(self.num_models):
            train_x, train_y = self._arrays(train_set, target)
            val_x, val_y = self._arrays(val_set or train_set, target)
            if len(train_y) == 0 or len(val_y) == 0:
                raise ValueError(f"Feature {target} has no finite train/validation targets.")

            x_mean, x_std = train_x.mean(axis=0), train_x.std(axis=0)
            x_std[x_std < 1e-6] = 1.0
            y_mean, y_std = float(train_y.mean()), float(train_y.std())
            y_std = y_std if y_std >= 1e-6 else 1.0
            self.scalers[target] = (x_mean, x_std, y_mean, y_std)
            train_x, val_x = (train_x - x_mean) / x_std, (val_x - x_mean) / x_std
            train_y, val_y = (train_y - y_mean) / y_std, (val_y - y_mean) / y_std

            pin_memory = self.device.type == "cuda"
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))),
                batch_size=self.batch_size, shuffle=True,
                pin_memory=pin_memory,
            )
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y.astype(np.float32))),
                batch_size=max(self.batch_size, 1024), shuffle=False,
                pin_memory=pin_memory,
            )
            model = _BayesianRegressor(
                self.num_models - 1, self.hidden_size, self.dropout
            ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            best_loss, best_state, remaining_patience = float("inf"), None, self.patience
            print(
                f"BayesNN feature {target + 1}/{self.num_models}: "
                f"{len(train_y):,} train points, batch size {self.batch_size}",
                flush=True,
            )

            for epoch in range(self.epochs):
                model.train()
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device, non_blocking=pin_memory)
                    batch_y = batch_y.to(self.device, non_blocking=pin_memory)
                    optimizer.zero_grad()
                    mean, log_variance = model(batch_x)
                    loss = self._gaussian_nll(mean, log_variance, batch_y)
                    loss.backward()
                    optimizer.step()

                model.eval()
                total_loss, total_items = 0.0, 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device, non_blocking=pin_memory)
                        batch_y = batch_y.to(self.device, non_blocking=pin_memory)
                        mean, log_variance = model(batch_x)
                        loss = self._gaussian_nll(mean, log_variance, batch_y)
                        total_loss += float(loss) * len(batch_y)
                        total_items += len(batch_y)
                val_loss = total_loss / total_items
                improved = np.isfinite(val_loss) and val_loss < best_loss - self.min_delta
                if improved:
                    best_loss, best_state = val_loss, deepcopy(model.state_dict())
                    remaining_patience = self.patience
                else:
                    remaining_patience -= 1
                if epoch == 0 or (epoch + 1) % 10 == 0 or remaining_patience == 0:
                    print(
                        f"  epoch {epoch + 1}/{self.epochs} - val NLL: {val_loss:.6f} "
                        f"- patience left: {remaining_patience}",
                        flush=True,
                    )
                if remaining_patience == 0:
                    break

            if best_state is None:
                raise RuntimeError(f"BayesNN feature {target} produced no finite validation loss.")
            model.load_state_dict(best_state)
            model.eval()
            self.models[target] = model
            logging.info(
                "BayesNN feature %d stopped at epoch %d with validation NLL %.6f",
                target, epoch + 1, best_loss,
            )
            print(
                f"BayesNN feature {target + 1}/{self.num_models} finished at "
                f"epoch {epoch + 1}; best val NLL: {best_loss:.6f}",
                flush=True,
            )

    def _posterior_quantile_batch(self, target: int, x: np.ndarray):
        model = self.models[target]
        x_mean, x_std, y_mean, y_std = self.scalers[target]
        x = (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - x_mean) / x_std
        x_tensor = torch.from_numpy(x.astype(np.float32)).to(self.device)
        draws = []
        model.train()  # Activate dropout for posterior sampling.
        with torch.no_grad():
            for _ in range(self.mc_samples):
                mean, log_variance = model(x_tensor)
                draw = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
                draws.append(draw.cpu().numpy() * y_std + y_mean)
        model.eval()
        draws = np.stack(draws, axis=0)
        return (
            np.quantile(draws, 0.5, axis=0),
            np.quantile(draws, self.lower_quantile, axis=0),
            np.quantile(draws, self.upper_quantile, axis=0),
        )

    def _posterior_quantiles(self, target: int, x: np.ndarray):
        medians, lowers, uppers = [], [], []
        for start in range(0, len(x), self.prediction_batch_size):
            batch = x[start : start + self.prediction_batch_size]
            median, lower, upper = self._posterior_quantile_batch(target, batch)
            medians.append(median)
            lowers.append(lower)
            uppers.append(upper)
        return tuple(np.concatenate(values) for values in (medians, lowers, uppers))

    def predict(self, val_set: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        X = np.asarray(val_set["X"], dtype=np.float32)
        mask = np.asarray(val_set["indicating_mask"], dtype=bool)
        observed = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        imputation, lower, upper = observed.copy(), observed.copy(), observed.copy()
        for target, model in self.models.items():
            missing = mask[:, :, target]
            if not np.any(missing):
                continue
            x = np.delete(X, target, axis=2)[missing]
            median, q_low, q_high = self._posterior_quantiles(target, x)
            imputation[:, :, target][missing] = median
            lower[:, :, target][missing] = q_low
            upper[:, :, target][missing] = q_high
        return {"imputation": imputation, "lower": lower, "upper": upper}
