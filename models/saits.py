"""SAITS model adapter and hyperparameter configuration."""

from .base import AbstractModel


class SAITSModel(AbstractModel):
    name = "saits"

    def _build_backend(self):
        from pypots.imputation import SAITS

        cfg = self.config
        return SAITS(
            n_steps=cfg.seq_len,
            n_features=cfg.n_features,
            n_layers=2,
            d_model=256,
            d_inner=128,
            n_heads=4,
            d_k=64,
            d_v=64,
            dropout=0.1,
            attn_dropout=0.1,
            diagonal_attention_mask=True,
            ORT_weight=1,
            MIT_weight=1,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            patience=cfg.patience,
            optimizer=cfg.optimizer,
            num_workers=0,
            device=cfg.device,
            saving_path=str(cfg.output_dir / self.name),
            model_saving_strategy="best",
        )
