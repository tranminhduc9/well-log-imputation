"""Depth-aware Attentive Neural Process for well-log imputation."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, Dataset


DatasetDict = dict[str, np.ndarray | torch.Tensor]
TensorDict = dict[str, torch.Tensor]
PredictionDict = dict[str, np.ndarray]


def _mlp(input_size: int, hidden_size: int, output_size: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, output_size),
    )


class _DictionaryDataset(Dataset):
    """Dataset that preserves the dictionary interface used by the model."""

    def __init__(self, tensors: TensorDict) -> None:
        self.tensors = tensors
        self.length = tensors["X"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> TensorDict:
        return {name: value[index] for name, value in self.tensors.items()}


class DepthAwareCrossAttention(nn.Module):
    """Multi-head cross-attention with a learned distance-decay bias.

    The content score is augmented with ``-|depth_q-depth_k| / scale``. Each
    head learns its own positive scale, so some heads can focus locally while
    others retain a wider receptive field along the well.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        initial_depth_scale: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if initial_depth_scale <= 0:
            raise ValueError("initial_depth_scale must be positive")

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Inverse-softplus initialization keeps the effective initial scale
        # close to the user-facing value.
        raw_scale = math.log(math.expm1(initial_depth_scale))
        self.raw_depth_scale = nn.Parameter(torch.full((n_heads,), raw_scale))

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = tensor.shape
        return tensor.reshape(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_depth: torch.Tensor,
        key_depth: torch.Tensor,
        key_valid: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._heads(self.query(query))
        k = self._heads(self.key(key))
        v = self._heads(self.value(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        distance = torch.abs(query_depth.unsqueeze(-2) - key_depth.unsqueeze(-3))
        distance = distance.squeeze(-1).unsqueeze(1)
        depth_scale = F.softplus(self.raw_depth_scale).view(1, self.n_heads, 1, 1) + 1e-6
        scores = scores - distance / depth_scale

        if key_valid is not None:
            invalid = ~key_valid.bool().unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(invalid, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.reshape(query.shape[0], query.shape[1], self.hidden_dim)
        return self.output(attended), weights


class AttentionNeuralProcess(nn.Module):
    """Depth-aware ANP with the same ``fit``/``predict`` API as other models.

    Input dictionaries must contain ``X`` and ``indicating_mask`` plus either
    ``X_intact`` or ``X_ori``. Arrays have shape ``[N, L, F]``. An optional
    ``depth`` array may have shape ``[L]``, ``[N, L]`` or ``[N, L, 1]``. It is
    normalized per sequence; otherwise sample position in ``[-1, 1]`` is used.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        latent_dim: int = 32,
        *,
        attention: Optional[nn.Module] = None,
        n_steps: int = 256,
        n_features: int = 4,
        hidden_dim: int = 128,
        n_heads: int = 4,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        initial_depth_scale: float = 0.2,
        kl_weight: float = 1e-3,
        observed_loss_weight: float = 0.1,
        min_scale: float = 1e-3,
        device: str | torch.device = "cpu",
        saving_path: str | Path = ".",
    ) -> None:
        super().__init__()
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        if n_features < 1:
            raise ValueError("n_features must be positive")

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")

        self.n_steps = n_steps
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight
        self.observed_loss_weight = observed_loss_weight
        self.min_scale = min_scale
        self.device = requested_device
        self.saving_path = Path(saving_path)

        # Values are multiplied by the observation mask before encoding. The
        # mask itself is also supplied, so a real normalized value of zero is
        # distinguishable from a missing value.
        encoder_input_size = 1 + 2 * n_features
        self.encoder = encoder or _mlp(encoder_input_size, hidden_dim, hidden_dim, dropout)
        self.depth_encoder = _mlp(1, hidden_dim, hidden_dim, dropout)
        self.attention = attention or DepthAwareCrossAttention(
            hidden_dim, n_heads, dropout, initial_depth_scale
        )
        self.latent_encoder = _mlp(encoder_input_size, hidden_dim, hidden_dim, dropout)
        self.latent_parameters = nn.Linear(hidden_dim, 2 * latent_dim)
        decoder_input_size = 1 + hidden_dim + latent_dim
        self.decoder = decoder or _mlp(decoder_input_size, hidden_dim, 2 * n_features, dropout)

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.best_model_dict: Optional[dict[str, torch.Tensor]] = None
        self.best_loss = float("inf")
        self.is_fitted = False
        self.to(self.device)

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------
    def convert_input(self, data: DatasetDict) -> TensorDict:
        """Validate and convert a repository dataset dictionary to tensors."""
        if "X" not in data or "indicating_mask" not in data:
            raise KeyError("data must contain 'X' and 'indicating_mask'")
        target_key = "X_intact" if "X_intact" in data else "X_ori"
        if target_key not in data:
            raise KeyError("data must contain 'X_intact' or 'X_ori'")

        x_raw = torch.as_tensor(data["X"], dtype=torch.float32)
        target_raw = torch.as_tensor(data[target_key], dtype=torch.float32)
        indicating = torch.as_tensor(data["indicating_mask"], dtype=torch.float32)
        if x_raw.ndim != 3 or x_raw.shape != target_raw.shape or x_raw.shape != indicating.shape:
            raise ValueError("X, target and indicating_mask must share shape [N, L, F]")
        if x_raw.shape[1:] != (self.n_steps, self.n_features):
            raise ValueError(
                f"expected [N, {self.n_steps}, {self.n_features}], got {tuple(x_raw.shape)}"
            )

        finite_x = torch.isfinite(x_raw)
        finite_target = torch.isfinite(target_raw)
        indicating = (indicating > 0.5).float() * finite_target.float()
        observed_mask = finite_x.float() * (1.0 - indicating)
        x = torch.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target_raw, nan=0.0, posinf=0.0, neginf=0.0)
        if "depth" in data:
            depth = torch.as_tensor(data["depth"], dtype=torch.float32)
            if depth.ndim == 1:
                depth = depth.view(1, -1, 1).expand(x.shape[0], -1, -1)
            elif depth.ndim == 2:
                depth = depth.unsqueeze(-1)
            if depth.shape != (x.shape[0], self.n_steps, 1):
                raise ValueError("depth must have shape [L], [N, L] or [N, L, 1]")
            if not torch.isfinite(depth).all():
                raise ValueError("depth must contain only finite values")
            depth_min = depth.amin(dim=1, keepdim=True)
            depth_range = (depth.amax(dim=1, keepdim=True) - depth_min).clamp_min(1e-6)
            depth = 2.0 * (depth - depth_min) / depth_range - 1.0
        else:
            depth = torch.linspace(-1.0, 1.0, self.n_steps).view(1, self.n_steps, 1)
            depth = depth.expand(x.shape[0], -1, -1)
        return {
            "X": x,
            "target": target,
            "observed_mask": observed_mask,
            "indicating_mask": indicating,
            "target_valid": finite_target.float(),
            "depth": depth,
        }

    def _make_dataloader(self, data: DatasetDict, *, shuffle: bool) -> DataLoader:
        tensors = {name: value.cpu() for name, value in self.convert_input(data).items()}
        return DataLoader(
            _DictionaryDataset(tensors),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

    def _move_batch(self, batch: TensorDict) -> TensorDict:
        non_blocking = self.device.type == "cuda"
        return {
            name: value.to(self.device, non_blocking=non_blocking)
            for name, value in batch.items()
        }

    def _split_context_target(self, batch: TensorDict) -> TensorDict:
        """Expose the full depth grid while hiding indicated feature values."""
        return {
            "context_depth": batch["depth"],
            "context_values": batch["X"],
            "context_mask": batch["observed_mask"],
            "target_depth": batch["depth"],
            "target_values": batch["target"],
            "target_mask": batch["indicating_mask"],
            "target_valid": batch["target_valid"],
        }

    # ------------------------------------------------------------------
    # Attentive Neural Process computation
    # ------------------------------------------------------------------
    @staticmethod
    def _encoder_input(depth: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.cat((depth, values * mask, mask), dim=-1)

    def _latent_distribution(
        self, depth: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
    ) -> Normal:
        encoded = self.latent_encoder(self._encoder_input(depth, values, mask))
        point_valid = (mask.sum(dim=-1, keepdim=True) > 0).float()
        pooled = (encoded * point_valid).sum(dim=1) / point_valid.sum(dim=1).clamp_min(1.0)
        mu, raw_scale = self.latent_parameters(pooled).chunk(2, dim=-1)
        scale = self.min_scale + F.softplus(raw_scale)
        return Normal(mu, scale)

    def _encode_context(self, context: TensorDict) -> TensorDict:
        point_representation = self.encoder(
            self._encoder_input(
                context["context_depth"],
                context["context_values"],
                context["context_mask"],
            )
        )
        prior = self._latent_distribution(
            context["context_depth"], context["context_values"], context["context_mask"]
        )
        return {"points": point_representation, "prior": prior}

    def _apply_attention(
        self,
        context: TensorDict,
        target: TensorDict,
        representation: TensorDict,
    ) -> TensorDict:
        context_depth_embedding = self.depth_encoder(context["context_depth"])
        target_depth_embedding = self.depth_encoder(target["target_depth"])
        key_valid = context["context_mask"].sum(dim=-1) > 0
        attended, weights = self.attention(
            target_depth_embedding,
            context_depth_embedding,
            representation["points"],
            target["target_depth"],
            context["context_depth"],
            key_valid,
        )
        return {**representation, "deterministic": attended, "attention_weights": weights}

    def _decode_target(self, target: TensorDict, representation: TensorDict) -> TensorDict:
        latent = representation["latent"].unsqueeze(1)
        latent = latent.expand(-1, target["target_depth"].shape[1], -1)
        decoded = self.decoder(
            torch.cat(
                (target["target_depth"], representation["deterministic"], latent), dim=-1
            )
        )
        mean, raw_scale = decoded.chunk(2, dim=-1)
        scale = self.min_scale + F.softplus(raw_scale)
        return {"mean": mean, "scale": scale}

    def forward(self, batch: TensorDict) -> TensorDict:
        split = self._split_context_target(batch)
        context = {
            "context_depth": split["context_depth"],
            "context_values": split["context_values"],
            "context_mask": split["context_mask"],
        }
        target = {
            "target_depth": split["target_depth"],
            "target_values": split["target_values"],
            "target_mask": split["target_mask"],
            "target_valid": split["target_valid"],
        }
        representation = self._encode_context(context)

        posterior = None
        if self.training:
            posterior = self._latent_distribution(
                target["target_depth"],
                target["target_values"],
                target["target_valid"],
            )
            latent = posterior.rsample()
        else:
            latent = representation["prior"].mean

        representation = self._apply_attention(context, target, representation)
        representation["latent"] = latent
        outputs = self._decode_target(target, representation)
        outputs.update(
            {
                "prior": representation["prior"],
                "posterior": posterior,
                "attention_weights": representation["attention_weights"],
            }
        )
        return outputs

    def _compute_loss(self, outputs: TensorDict, target: TensorDict) -> torch.Tensor:
        distribution = Normal(outputs["mean"], outputs["scale"])
        nll = -distribution.log_prob(target["target"])
        missing_mask = target["indicating_mask"] * target["target_valid"]
        observed_mask = target["observed_mask"] * target["target_valid"]

        if missing_mask.sum() > 0:
            missing_loss = (nll * missing_mask).sum() / missing_mask.sum()
        else:
            missing_loss = nll.new_zeros(())
        observed_loss = (nll * observed_mask).sum() / observed_mask.sum().clamp_min(1.0)
        reconstruction = missing_loss + self.observed_loss_weight * observed_loss

        posterior = outputs.get("posterior")
        if posterior is None:
            return reconstruction
        latent_kl = kl_divergence(posterior, outputs["prior"]).sum(dim=-1).mean()
        return reconstruction + self.kl_weight * latent_kl

    # ------------------------------------------------------------------
    # Training and inference
    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, *, training: bool) -> float:
        self.train(training)
        total_loss = 0.0
        total_batches = 0
        for cpu_batch in loader:
            batch = self._move_batch(cpu_batch)
            if training:
                assert self.optimizer is not None
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                outputs = self.forward(batch)
                loss = self._compute_loss(outputs, batch)
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                    self.optimizer.step()

            if not torch.isfinite(loss):
                raise ValueError("ANP produced a non-finite loss")
            total_loss += float(loss.detach())
            total_batches += 1
        if total_batches == 0:
            raise ValueError("ANP received an empty dataset")
        return total_loss / total_batches

    def _train_epoch(self, loader: DataLoader) -> float:
        return self._run_epoch(loader, training=True)

    def _validate_epoch(self, loader: DataLoader) -> float:
        return self._run_epoch(loader, training=False)

    def fit(self, train_set: DatasetDict, val_set: Optional[DatasetDict] = None) -> None:
        train_loader = self._make_dataloader(train_set, shuffle=True)
        val_loader = self._make_dataloader(val_set, shuffle=False) if val_set is not None else None
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.best_loss = float("inf")
        self.best_model_dict = None
        patience_left = self.patience

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_loader)
            validation_loss = self._validate_epoch(val_loader) if val_loader is not None else train_loss
            if validation_loss < self.best_loss:
                self.best_loss = validation_loss
                self.best_model_dict = deepcopy(self.state_dict())
                patience_left = self.patience
            elif patience_left is not None:
                patience_left -= 1

            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(
                    f"ANP epoch {epoch + 1}/{self.epochs} - "
                    f"train: {train_loss:.6f} - val: {validation_loss:.6f}",
                    flush=True,
                )
            if patience_left == 0:
                break

        if self.best_model_dict is None:
            raise RuntimeError("ANP training did not produce a valid checkpoint")
        self.load_state_dict(self.best_model_dict)
        self.eval()
        self.is_fitted = True

    def _impute_batch(self, batch: TensorDict) -> TensorDict:
        outputs = self.forward(batch)
        mask = batch["indicating_mask"].bool()
        imputation = torch.where(mask, outputs["mean"], batch["X"])
        lower_prediction = outputs["mean"] - 1.6448536269514722 * outputs["scale"]
        upper_prediction = outputs["mean"] + 1.6448536269514722 * outputs["scale"]
        lower = torch.where(mask, lower_prediction, batch["X"])
        upper = torch.where(mask, upper_prediction, batch["X"])
        return {"imputation": imputation, "lower": lower, "upper": upper}

    def predict(self, test_set: DatasetDict) -> PredictionDict:
        if not self.is_fitted:
            raise RuntimeError("Call fit() or load() before predict()")
        loader = self._make_dataloader(test_set, shuffle=False)
        collectors: dict[str, list[np.ndarray]] = {
            "imputation": [], "lower": [], "upper": []
        }
        self.eval()
        with torch.no_grad():
            for cpu_batch in loader:
                batch = self._move_batch(cpu_batch)
                results = self._impute_batch(batch)
                for name, value in results.items():
                    collectors[name].append(value.cpu().numpy())
        return {name: np.concatenate(values, axis=0) for name, values in collectors.items()}

    def impute(self, test_set: DatasetDict) -> np.ndarray:
        return self.predict(test_set)["imputation"]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": None if self.optimizer is None else self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.best_loss = float(checkpoint.get("best_loss", float("inf")))
        if self.optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.eval()
        self.is_fitted = True
