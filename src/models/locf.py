"""Last observation carried forward (LOCF) baseline."""

import numpy as np

from src.models.model import AbstractModel


class _LOCFBackend:

    def fit(self, train_set, val_set=None) -> None:
        """LOCF has no parameters to learn."""

    def predict(self, dataset):
        values = np.asarray(dataset["X"], dtype=float).copy()

        # Carry the previous value forward along the depth axis.
        for depth in range(1, values.shape[1]):
            missing = np.isnan(values[:, depth, :])
            values[:, depth, :] = np.where(
                missing,
                values[:, depth - 1, :],
                values[:, depth, :],
            )

        # Fill gaps before the first observation from the next known value.
        for depth in range(values.shape[1] - 2, -1, -1):
            missing = np.isnan(values[:, depth, :])
            values[:, depth, :] = np.where(
                missing,
                values[:, depth + 1, :],
                values[:, depth, :],
            )

        # A fully missing normalized log has no observation to carry.
        values = np.nan_to_num(values, nan=0.0)
        return {"imputation": values}


class LOCF(AbstractModel):
    """Training-free LOCF model for segmented well logs."""

    name = "locf"
    requires_training = False

    def _build_backend(self):
        return _LOCFBackend()
