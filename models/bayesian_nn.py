"""Variational Bayesian neural-network imputer following Feng et al. (2021)."""

from copy import deepcopy
import logging
import math
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base import AbstractModel


class _ScaleMixtureGaussian:
    """Zero-mean two-component Gaussian-mixture prior (paper equation 9)."""

    def __init__(self, pi: float, sigma1: float, sigma2: float) -> None:
        if not 0.0 < pi < 1.0:
            raise ValueError("pi must be strictly between zero and one.")
        if sigma1 <= 0.0 or sigma2 <= 0.0:
            raise ValueError("Prior standard deviations must be positive.")
        self.log_pi = math.log(pi)
        self.log_one_minus_pi = math.log1p(-pi)
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    @staticmethod
    def _normal_log_prob(value: torch.Tensor, sigma: float) -> torch.Tensor:
        return -0.5 * math.log(2.0 * math.pi) - math.log(sigma) - value.square() / (
            2.0 * sigma**2
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        component1 = self.log_pi + self._normal_log_prob(value, self.sigma1)
        component2 = self.log_one_minus_pi + self._normal_log_prob(value, self.sigma2)
        return torch.logsumexp(torch.stack((component1, component2)), dim=0).sum()


class _BayesianLinear(nn.Module):
    """Linear layer with a learned factorized Gaussian posterior over parameters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior: _ScaleMixtureGaussian,
        posterior_mu_std: float,
    ) -> None:
        super().__init__()
        self.prior = prior
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))
        nn.init.normal_(self.weight_mu, mean=0.0, std=posterior_mu_std)
        nn.init.normal_(self.bias_mu, mean=0.0, std=posterior_mu_std)

    @staticmethod
    def _sample(mu: torch.Tensor, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Equations 6-8: epsilon ~ N(0, I), sigma=softplus(rho), w=mu+sigma*epsilon.
        sigma = F.softplus(rho)
        return mu + sigma * torch.randn_like(mu), sigma

    @staticmethod
    def _posterior_log_prob(
        value: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        return (
            -0.5 * math.log(2.0 * math.pi)
            - torch.log(sigma)
            - (value - mu).square() / (2.0 * sigma.square())
        ).sum()

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if not sample:
            return F.linear(x, self.weight_mu, self.bias_mu), x.new_zeros(())

        weight, weight_sigma = self._sample(self.weight_mu, self.weight_rho)
        bias, bias_sigma = self._sample(self.bias_mu, self.bias_rho)
        log_posterior = self._posterior_log_prob(
            weight, self.weight_mu, weight_sigma
        ) + self._posterior_log_prob(bias, self.bias_mu, bias_sigma)
        log_prior = self.prior.log_prob(weight) + self.prior.log_prob(bias)
        return F.linear(x, weight, bias), log_posterior - log_prior


class _BayesianRegressor(nn.Module):
    """Paper architecture: two 10-neuron Bayesian hidden layers and ReLU."""

    def __init__(
        self,
        n_inputs: int,
        hidden_size: int,
        prior_pi: float,
        prior_sigma1: float,
        prior_sigma2: float,
    ) -> None:
        super().__init__()
        prior = _ScaleMixtureGaussian(prior_pi, prior_sigma1, prior_sigma2)
        # The paper initializes mu with variance pi*sigma1^2+(1-pi)*sigma2^2.
        mu_std = math.sqrt(
            prior_pi * prior_sigma1**2 + (1.0 - prior_pi) * prior_sigma2**2
        )
        self.layers = nn.ModuleList(
            (
                _BayesianLinear(n_inputs, hidden_size, prior, mu_std),
                _BayesianLinear(hidden_size, hidden_size, prior, mu_std),
                _BayesianLinear(hidden_size, 1, prior, mu_std),
            )
        )

    def forward(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        total_kl = x.new_zeros(())
        for layer in self.layers[:-1]:
            x, layer_kl = layer(x, sample=sample)
            total_kl = total_kl + layer_kl
            x = F.relu(x)
        x, layer_kl = self.layers[-1](x, sample=sample)
        return x.squeeze(-1), total_kl + layer_kl


class BayesianNNImputer:
    """Bayes-by-Backprop imputer using the variational posterior from the paper."""

    def __init__(
        self,
        num_models: int,
        batch_size: int = 32,
        epochs: int = 300,
        patience: int = 50,
        device: Optional[Union[str, torch.device]] = None,
        learning_rate: float = 1e-3,
        hidden_size: int = 10,
        mc_samples: int = 1000,
        prediction_batch_size: int = 8192,
        min_delta: float = 1e-4,
        uncertainty_std: float = 1.0,
        prior_pi: float = 0.5,
        prior_sigma1: float = 1.5,
        prior_sigma2: float = 0.1,
        train_mc_samples: int = 1,
    ) -> None:
        if mc_samples <= 0 or train_mc_samples <= 0:
            raise ValueError("Monte Carlo sample counts must be positive.")
        if uncertainty_std <= 0.0:
            raise ValueError("uncertainty_std must be positive.")
        self.num_models = num_models
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.mc_samples = mc_samples
        self.prediction_batch_size = prediction_batch_size
        self.min_delta = min_delta
        self.uncertainty_std = uncertainty_std
        self.prior_pi = prior_pi
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2
        self.train_mc_samples = train_mc_samples
        requested = str(device or "cpu")
        self.device = torch.device(
            requested if requested != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.models: dict[int, _BayesianRegressor] = {}
        self.scalers: dict[int, tuple[np.ndarray, np.ndarray, float, float]] = {}

    @staticmethod
    def _arrays(data: dict[str, np.ndarray], target: int) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(data["X"], dtype=np.float32)
        X_intact = np.asarray(data["X_intact"], dtype=np.float32)
        y = X_intact[:, :, target].reshape(-1)
        x = np.delete(X, target, axis=2).reshape(-1, X.shape[2] - 1)
        valid = np.isfinite(y)
        return np.nan_to_num(x[valid], nan=0.0, posinf=0.0, neginf=0.0), y[valid]

    def _variational_free_energy(
        self,
        model: _BayesianRegressor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        training_size: int,
    ) -> torch.Tensor:
        # Equation 5 divided by |D|. Averaging several samples reduces estimator noise.
        losses = []
        for _ in range(self.train_mc_samples):
            prediction, kl = model(inputs, sample=True)
            negative_log_likelihood = 0.5 * (targets - prediction).square().mean()
            losses.append(kl / training_size + negative_log_likelihood)
        return torch.stack(losses).mean()

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
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=pin_memory,
            )
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y.astype(np.float32))),
                batch_size=max(self.batch_size, 1024),
                shuffle=False,
                pin_memory=pin_memory,
            )
            model = _BayesianRegressor(
                self.num_models - 1,
                self.hidden_size,
                self.prior_pi,
                self.prior_sigma1,
                self.prior_sigma2,
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
                    loss = self._variational_free_energy(
                        model, batch_x, batch_y, len(train_y)
                    )
                    loss.backward()
                    optimizer.step()

                # Posterior-mean MSE is stable enough for early stopping and is the
                # performance curve reported in the paper.
                model.eval()
                total_squared_error, total_items = 0.0, 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device, non_blocking=pin_memory)
                        batch_y = batch_y.to(self.device, non_blocking=pin_memory)
                        prediction, _ = model(batch_x, sample=False)
                        total_squared_error += float((prediction - batch_y).square().sum())
                        total_items += len(batch_y)
                val_loss = total_squared_error / total_items
                improved = np.isfinite(val_loss) and val_loss < best_loss - self.min_delta
                if improved:
                    best_loss, best_state = val_loss, deepcopy(model.state_dict())
                    remaining_patience = self.patience
                else:
                    remaining_patience -= 1
                if epoch == 0 or (epoch + 1) % 10 == 0 or remaining_patience == 0:
                    print(
                        f"  epoch {epoch + 1}/{self.epochs} - val MSE: {val_loss:.6f} "
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
                "BayesNN feature %d stopped at epoch %d with validation MSE %.6f",
                target,
                epoch + 1,
                best_loss,
            )

    def _posterior_summary_batch(
        self, target: int, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        model = self.models[target]
        x_mean, x_std, y_mean, y_std = self.scalers[target]
        x = (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - x_mean) / x_std
        x_tensor = torch.from_numpy(x.astype(np.float32)).to(self.device)
        draws = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                prediction, _ = model(x_tensor, sample=True)
                draws.append(prediction.cpu().numpy() * y_std + y_mean)
        draws = np.stack(draws, axis=0)
        # The paper reports the ensemble mean and its empirical standard
        # deviation. Uncertainty comes only from posterior parameter draws.
        return draws.mean(axis=0), draws.std(axis=0)

    def _posterior_summary(
        self, target: int, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        means, standard_deviations = [], []
        for start in range(0, len(x), self.prediction_batch_size):
            batch = x[start : start + self.prediction_batch_size]
            mean, standard_deviation = self._posterior_summary_batch(target, batch)
            means.append(mean)
            standard_deviations.append(standard_deviation)
        return np.concatenate(means), np.concatenate(standard_deviations)

    def predict(self, val_set: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        X = np.asarray(val_set["X"], dtype=np.float32)
        mask = np.asarray(val_set["indicating_mask"], dtype=bool)
        observed = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        imputation, lower, upper = observed.copy(), observed.copy(), observed.copy()
        standard_deviation = np.zeros_like(observed)
        for target in self.models:
            missing = mask[:, :, target]
            if not np.any(missing):
                continue
            x = np.delete(X, target, axis=2)[missing]
            mean, std = self._posterior_summary(target, x)
            imputation[:, :, target][missing] = mean
            standard_deviation[:, :, target][missing] = std
            lower[:, :, target][missing] = mean - self.uncertainty_std * std
            upper[:, :, target][missing] = mean + self.uncertainty_std * std
        return {
            "imputation": imputation,
            "std": standard_deviation,
            "lower": lower,
            "upper": upper,
        }


class BayesianNNModel(AbstractModel):
    name = "bayesnn"

    def _build_backend(self):
        cfg = self.config
        return BayesianNNImputer(
            num_models=cfg.n_features,
            batch_size=max(cfg.batch_size, 4096),
            epochs=cfg.epochs,
            patience=cfg.patience,
            device=cfg.device,
            learning_rate=cfg.learning_rate,
            hidden_size=10,
            mc_samples=1000,
            min_delta=1e-4,
            prior_pi=0.5,
            prior_sigma1=1.5,
            prior_sigma2=0.1,
        )
