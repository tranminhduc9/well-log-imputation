"""End-to-end preprocessing pipeline for well-log imputation."""

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .clean_data import SELECTED_LOGS, create_segments, filter_well_logs
from .read_data import read_data_directory
from .split_data import normalize_data, remove_outliers, split_data


MISSING_SCENARIOS = ("Single", "Block-20", "Block-100", "Entire-Log")
BLOCK_LENGTHS = {"Block-20": 20, "Block-100": 100}


def create_missing_mask(
    segments: np.ndarray,
    scenario: str | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Create a missing mask for clean segments.

    If ``scenario`` is omitted, every segment randomly receives one of the
    supported scenarios. In all scenarios, one log is selected at random per
    segment. Omit ``random_state`` when creating dynamic training masks.
    """
    valid_shape = (
        segments.ndim == 3
        and segments.shape[1] > 0
        and segments.shape[2] > 0
    )
    valid_scenario = scenario is None or scenario in MISSING_SCENARIOS
    required_length = (
        max(BLOCK_LENGTHS.values())
        if scenario is None
        else BLOCK_LENGTHS.get(scenario, 1)
    )
    if not (valid_shape and valid_scenario and required_length <= segments.shape[1]):
        raise ValueError("Invalid inputs for create_missing_mask")

    segment_count, sample_count, log_count = segments.shape

    generator = np.random.default_rng(random_state)
    mask = np.zeros(segments.shape, dtype=bool)
    scenarios = (
        generator.choice(MISSING_SCENARIOS, size=segment_count)
        if scenario is None
        else np.full(segment_count, scenario)
    )

    for segment_index, selected_scenario in enumerate(scenarios):
        log_index = generator.integers(log_count)
        if selected_scenario == "Single":
            sample_index = generator.integers(sample_count)
            mask[segment_index, sample_index, log_index] = True
        elif selected_scenario in BLOCK_LENGTHS:
            block_length = BLOCK_LENGTHS[selected_scenario]
            block_start = generator.integers(sample_count - block_length + 1)
            mask[
                segment_index,
                block_start : block_start + block_length,
                log_index,
            ] = True
        else:
            mask[segment_index, :, log_index] = True

    return mask


def create_fixed_scenario_masks(
    segments: np.ndarray,
    random_state: int = 912,
) -> dict[str, np.ndarray]:
    """Create one reproducible mask set for every evaluation scenario."""
    return {
        scenario: create_missing_mask(
            segments,
            scenario=scenario,
            random_state=random_state + scenario_index,
        )
        for scenario_index, scenario in enumerate(MISSING_SCENARIOS)
    }


def run_pipeline(
    data: pd.DataFrame,
    log_columns: Sequence[str] = SELECTED_LOGS,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    segment_length: int = 256,
    well_column: str = "WELL",
    depth_column: str | None = "DEPT",
    random_state: int = 912,
) -> dict[str, Any]:
    """Run the leakage-safe workflow defined in ``data_pipeline.md``."""
    filtered_data = filter_well_logs(data, log_columns, well_column)
    selected_columns = [well_column]
    if "SOURCE_FILE" in filtered_data:
        selected_columns.append("SOURCE_FILE")
    if depth_column is not None:
        if depth_column not in filtered_data:
            raise ValueError(f"Missing depth column: {depth_column}")
        selected_columns.append(depth_column)
    selected_columns.extend(log_columns)
    filtered_data = filtered_data[selected_columns]

    train, validation, test = split_data(
        filtered_data,
        train_ratio,
        validation_ratio,
        test_ratio,
        well_column,
        random_state,
    )
    train, validation, test, outlier_bounds = remove_outliers(
        train,
        validation,
        test,
        log_columns,
        lower_quantile,
        upper_quantile,
    )
    train, validation, test, normalization = normalize_data(
        train, validation, test, log_columns
    )

    train_segments, train_metadata = create_segments(
        train, log_columns, segment_length, well_column, depth_column
    )
    validation_segments, validation_metadata = create_segments(
        validation, log_columns, segment_length, well_column, depth_column
    )
    test_segments, test_metadata = create_segments(
        test, log_columns, segment_length, well_column, depth_column
    )

    parameters = {
        "log_columns": list(log_columns),
        "outlier_bounds": outlier_bounds,
        "normalization": normalization,
        "segment_length": segment_length,
        "random_seed": random_state,
        "missing_scenarios": list(MISSING_SCENARIOS),
        "block_lengths": BLOCK_LENGTHS,
        "validation_mask_seed": random_state + 1,
        "test_mask_seed": random_state + 2,
    }
    return {
        "train": {"segments": train_segments, "metadata": train_metadata},
        "validation": {
            "segments": validation_segments,
            "metadata": validation_metadata,
            "masks": create_fixed_scenario_masks(
                validation_segments, random_state + 1
            ),
        },
        "test": {
            "segments": test_segments,
            "metadata": test_metadata,
            "masks": create_fixed_scenario_masks(
                test_segments, random_state + 2
            ),
        },
        "parameters": parameters,
    }


def run_pipeline_from_directory(
    directory_path: str | Path,
    output_directory: str | Path | None = None,
    **pipeline_options: Any,
) -> dict[str, Any]:
    """Read all LAS files, run preprocessing and optionally save its outputs."""
    result = run_pipeline(read_data_directory(directory_path), **pipeline_options)
    if output_directory is not None:
        save_processed_data(result, output_directory)
    return result


def save_preprocessing_parameters(
    parameters: dict[str, Any], output_path: str | Path
) -> None:
    """Save fitted train-only parameters for later inference."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(parameters, file, indent=2)


def save_processed_data(
    result: dict[str, Any], output_directory: str | Path
) -> None:
    """Save processed segments, metadata, fixed masks and fitted parameters."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    split_directories = {"train": "train", "validation": "val", "test": "test"}
    for split_name, directory_name in split_directories.items():
        split = result[split_name]
        split_path = output_path / directory_name
        split_path.mkdir(parents=True, exist_ok=True)
        np.save(split_path / "segments.npy", split["segments"])
        split["metadata"].to_csv(split_path / "metadata.csv", index=False)
        if "masks" in split:
            for scenario, mask in split["masks"].items():
                scenario_name = scenario.lower().replace("-", "_")
                np.save(split_path / f"mask_{scenario_name}.npy", mask)

    save_preprocessing_parameters(
        result["parameters"], output_path / "preprocessing_parameters.json"
    )
