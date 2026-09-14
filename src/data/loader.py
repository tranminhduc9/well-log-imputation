"""Load train, validation, and test data from ``data/processed``."""

from pathlib import Path

import numpy as np
import pandas as pd


class DataLoader:
    """Load one processed data split."""

    split_directories = {
        "train": "train",
        "val": "val",
        "test": "test",
    }

    def __init__(
        self,
        root: str | Path = "data/processed",
        split: str = "train",
        scenario: str | None = None,
    ) -> None:
        self.path = Path(root) / self.split_directories[split]
        self.scenario = scenario

    def load(self) -> dict[str, np.ndarray]:
        """Return data in the format expected by ``AbstractModel``."""

        segments = np.load(self.path / "segments.npy")

        if self.scenario is None:
            return {"X": segments}

        scenario = self.scenario.lower().replace("-", "_")
        mask = np.load(self.path / f"mask_{scenario}.npy")
        masked_segments = segments.copy()
        masked_segments[mask] = np.nan

        return {
            "X": masked_segments,
            "X_intact": segments,
            "indicating_mask": mask,
        }

    def load_metadata(self) -> pd.DataFrame:
        """Load well names and segment depths."""

        return pd.read_csv(self.path / "metadata.csv")
