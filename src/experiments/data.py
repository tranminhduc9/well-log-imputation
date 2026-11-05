"""Load split data, verify well separation and apply the saved missing masks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.pipeline import MISSING_SCENARIOS, BLOCK_LENGTHS
from .utils import sha256


def load_and_check_data(data_root, evaluate_test):
    """Fail before training on split overlap, invalid masks or bad scaling."""
    root = Path(data_root)
    parameter_path = root / "preprocessing_parameters.json"
    preprocessing = json.loads(parameter_path.read_text(encoding="utf-8"))
    steps = preprocessing["segment_length"]
    logs = preprocessing["log_columns"]
    scenarios = preprocessing["missing_scenarios"]
    if set(scenarios) != set(MISSING_SCENARIOS) or len(scenarios) != 4:
        raise ValueError("GeoLink requires the four supported missingness scenarios.")
    if steps < max(BLOCK_LENGTHS.values()) or len(logs) != len(set(logs)) or not logs:
        raise ValueError("Invalid GeoLink sequence length or log names.")
    for log in logs:
        scaling = preprocessing["normalization"][log]
        if not np.isfinite([scaling["mean"], scaling["std"]]).all() or scaling["std"] <= 0:
            raise ValueError(f"Invalid normalization for {log}.")
    splits = ("train", "val", "test") if evaluate_test else ("train", "val")
    metadata, datasets, files = {}, {}, [parameter_path]
    for split in splits:
        value_path, meta_path = root / split / "segments.npy", root / split / "metadata.csv"
        values = np.load(value_path, allow_pickle=False)
        meta = pd.read_csv(meta_path, dtype={"WELL": str})
        if values.ndim != 3 or values.shape[1:] != (steps, len(logs)) or len(values) == 0:
            raise ValueError(f"Invalid segment shape for {split}: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"GeoLink intact {split} targets contain NaN/Inf.")
        if len(meta) != len(values) or "WELL" not in meta or meta["WELL"].isna().any():
            raise ValueError(f"Missing or misaligned WELL metadata in {split}.")
        if meta["WELL"].str.strip().eq("").any():
            raise ValueError(f"Empty well names in {split}.")
        for other, previous in metadata.items():
            overlap = set(meta["WELL"]) & set(previous["WELL"])
            if overlap:
                raise ValueError(f"Well leakage between {other} and {split}: {sorted(overlap)}")
        metadata[split] = meta
        depth = None
        if {"START_DEPTH", "END_DEPTH"} <= set(meta.columns):
            endpoints = meta[["START_DEPTH", "END_DEPTH"]].to_numpy(dtype=np.float64)
            if not np.isfinite(endpoints).all() or np.any(endpoints[:, 1] <= endpoints[:, 0]):
                raise ValueError(f"Invalid depth metadata in {split}.")
            fractions = np.linspace(0.0, 1.0, steps, dtype=np.float64)
            depth = (endpoints[:, :1] + (endpoints[:, 1:] - endpoints[:, :1]) * fractions).astype(np.float32)
        files.extend([value_path, meta_path])
        if split == "train":
            datasets[split] = {"X": values}
            if depth is not None:
                datasets[split]["depth"] = depth
            continue
        datasets[split] = {}
        for scenario in scenarios:
            mask_path = root / split / f"mask_{scenario.lower().replace('-', '_')}.npy"
            mask = np.load(mask_path, allow_pickle=False)
            if mask.shape != values.shape or not np.isin(mask, [0, 1]).all():
                raise ValueError(f"Invalid mask for {split}/{scenario}.")
            mask = mask.astype(bool)
            length = BLOCK_LENGTHS.get(scenario, 1 if scenario == "Single" else steps)
            for segment in mask:
                columns = np.flatnonzero(segment.any(axis=0))
                if len(columns) != 1:
                    raise ValueError(f"Mask must hide exactly one log: {split}/{scenario}.")
                indices = np.flatnonzero(segment[:, columns[0]])
                if len(indices) != length or (len(indices) > 1 and not (np.diff(indices) == 1).all()):
                    raise ValueError(f"Incorrect gap topology: {split}/{scenario}.")
            masked = values.copy()
            masked[mask] = np.nan
            datasets[split][scenario] = {
                "X": masked, "X_intact": values, "indicating_mask": mask,
            }
            if depth is not None:
                datasets[split][scenario]["depth"] = depth
            files.append(mask_path)
    fingerprints = {str(path.relative_to(root)): sha256(path) for path in files}
    return preprocessing, datasets, metadata, fingerprints
