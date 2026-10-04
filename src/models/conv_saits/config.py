"""Hyperparameters and ablation settings for Conv-SAITS."""

from dataclasses import dataclass
import math

import numpy as np

from src.models.model import ModelConfig


@dataclass(frozen=True)
class ConvSAITSConfig(ModelConfig):
    """Configuration for Conv-SAITS.

    ``kernel_size=51`` follows the example in Equation (3) of the paper.  The
    kernel is required to be odd so that temporal convolution preserves the
    sequence length exactly.
    """

    n_layers: int = 2
    d_model: int = 256
    d_inner: int = 128
    n_heads: int = 4
    encoder_channels: int = 16
    kernel_size: int = 51
    multi_scale_conv: bool = False
    multi_scale_kernels: tuple[int, ...] = (3, 7, 15)
    conv_expansion: int = 2
    dropout: float = 0.1
    attn_dropout: float = 0.1
    masking_strategy: str = "mixed"
    masking_rate: float = 0.2
    diagonal_attention_mask: bool = True
    ort_weight: float = 1.0
    mit_weight: float = 1.0
    # Optional squared-error supervision on artificial gaps; zero preserves
    # the existing MAE objective and checkpoint architecture.
    mit_mse_weight: float = 0.0
    gradient_clip: float | None = None
    mit_reduction: str = "segment"
    min_delta: float = 1e-4
    temporal_residual: bool = False
    encoder_norm: str = "batch"
    gap_aware_gate: bool = False
    decoder: str = "shared"
    decoder_width: int = 32
    independent_shuffle: bool = False
    depth_encoding: bool = False
    depth_mean: float = 0.0
    depth_std: float = 1.0
    cross_log_attention: bool = False
    cross_log_width: int = 32
    mit_relative_weight: float = 0.0
    physical_means: tuple[float, ...] = ()
    physical_stds: tuple[float, ...] = ()
    relative_floors: tuple[float, ...] = ()
    validation_objective: str = "rmse"
    reference_rmse: float = 1.0
    reference_mape: float = 1.0
    validation_mape_floor: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.encoder_norm not in {"batch", "layer"}:
            raise ValueError("encoder_norm must be 'batch' or 'layer'.")
        if self.decoder not in {"shared", "per_log", "conditional"} or self.decoder_width <= 0:
            raise ValueError("Invalid decoder or decoder_width.")
        if self.cross_log_width <= 0 or (self.cross_log_attention and
                (self.n_heads <= 0 or self.cross_log_width % self.n_heads)):
            raise ValueError("cross_log_width must be positive and divisible by n_heads")
        if self.validation_objective not in {"rmse", "rmse_mape"}:
            raise ValueError("Invalid validation_objective")
        if not math.isfinite(self.mit_relative_weight) or self.mit_relative_weight < 0:
            raise ValueError("mit_relative_weight must be finite and non-negative")
        for name in ("physical_means", "physical_stds", "relative_floors"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.mit_relative_weight or self.validation_objective == "rmse_mape":
            if any(len(getattr(self, name)) != self.n_features for name in
                   ("physical_means", "physical_stds", "relative_floors")):
                raise ValueError("Physical normalization/floors must cover every feature")
            if not np.isfinite(self.physical_means).all() or any(
                not np.isfinite(values).all() or np.any(np.asarray(values) <= 0)
                for values in (self.physical_stds, self.relative_floors)
            ):
                raise ValueError("Invalid physical normalization/floors")
        if any(not math.isfinite(v) or v <= 0 for v in (self.reference_rmse, self.reference_mape)):
            raise ValueError("Reference metrics must be finite and positive")
        if self.mit_reduction not in {"segment", "point"}:
            raise ValueError("mit_reduction must be 'segment' or 'point'.")
        sizes = ("n_layers", "d_model", "d_inner", "n_heads", "encoder_channels", "kernel_size", "conv_expansion")
        invalid = [name for name in sizes if getattr(self, name) <= 0]
        if invalid:
            raise ValueError("Conv-SAITS architecture settings must be positive: " + ", ".join(invalid))
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.d_model % 2:
            raise ValueError("d_model must be even for sinusoidal positional encoding.")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")
        kernels = tuple(self.multi_scale_kernels)
        object.__setattr__(self, "multi_scale_kernels", kernels)
        if not kernels or len(set(kernels)) != len(kernels):
            raise ValueError("multi_scale_kernels must contain unique values.")
        if any(type(value) is not int or value <= 0 or value % 2 == 0 for value in kernels):
            raise ValueError("multi_scale_kernels must be positive odd integers.")
        if not math.isfinite(self.depth_mean) or not math.isfinite(self.depth_std) or self.depth_std <= 0:
            raise ValueError("Depth normalization must be finite with depth_std > 0.")
        if self.diagonal_attention_mask and self.seq_len < 2:
            raise ValueError("Diagonal attention requires at least two time steps.")
        if not 0 <= self.dropout < 1 or not 0 <= self.attn_dropout < 1:
            raise ValueError("Conv-SAITS dropout rates must be in [0, 1).")
        if not 0 < self.masking_rate < 1:
            raise ValueError("masking_rate must be in (0, 1).")
        if self.masking_strategy not in {"mixed", "random"}:
            raise ValueError("masking_strategy must be 'mixed' or 'random'.")
        if self.ort_weight < 0 or self.mit_weight < 0 or self.min_delta < 0:
            raise ValueError("Conv-SAITS loss weights and min_delta must be non-negative.")
        if not math.isfinite(self.mit_mse_weight) or self.mit_mse_weight < 0:
            raise ValueError("mit_mse_weight must be finite and non-negative.")
        if self.gradient_clip is not None and (
            not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0
        ):
            raise ValueError("gradient_clip must be None or finite and positive.")
        if self.ort_weight == self.mit_weight == self.mit_mse_weight == self.mit_relative_weight == 0:
            raise ValueError("At least one Conv-SAITS loss weight must be positive.")
