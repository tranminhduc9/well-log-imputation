"""Clean and segment well-log data before modeling."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


SELECTED_LOGS = ("GR", "RHOB", "NPHI", "DTC")


def filter_well_logs(
    data: pd.DataFrame,
    log_columns: Sequence[str] = SELECTED_LOGS,
    well_column: str = "WELL",
) -> pd.DataFrame:
    """Keep wells that contain valid data for every selected log."""
    required_columns = [well_column, *log_columns]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    availability = (
        data.groupby(well_column, sort=False)[list(log_columns)].count().gt(0).all(axis=1)
    )
    valid_wells = availability.index[availability]
    return data[data[well_column].isin(valid_wells)].copy()


def fit_outlier_bounds(
    train_data: pd.DataFrame,
    log_columns: Sequence[str] = SELECTED_LOGS,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> dict[str, tuple[float, float]]:
    """Fit per-log percentile bounds using training data only."""
    missing_columns = [column for column in log_columns if column not in train_data]
    empty_columns = [
        column
        for column in log_columns
        if column in train_data and train_data[column].dropna().empty
    ]
    if (
        not 0 <= lower_quantile < upper_quantile <= 1
        or missing_columns
        or empty_columns
    ):
        raise ValueError("Invalid inputs for fit_outlier_bounds")

    bounds = {}
    for column in log_columns:
        values = train_data[column].dropna()
        bounds[column] = (
            float(values.quantile(lower_quantile)),
            float(values.quantile(upper_quantile)),
        )
    return bounds


def remove_outliers(
    data: pd.DataFrame,
    bounds: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    """Replace values outside pre-fitted bounds with ``NaN``.

    Rows are retained so gaps remain visible when clean intervals are found.
    """
    cleaned = data.copy()
    if any(column not in cleaned for column in bounds):
        raise ValueError("Invalid inputs for remove_outliers")

    for column, (lower_bound, upper_bound) in bounds.items():
        is_outlier = ~cleaned[column].between(lower_bound, upper_bound)
        cleaned.loc[is_outlier & cleaned[column].notna(), column] = np.nan
    return cleaned


def create_segments(
    data: pd.DataFrame,
    log_columns: Sequence[str] = SELECTED_LOGS,
    segment_length: int = 256,
    well_column: str = "WELL",
    depth_column: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create non-overlapping segments from consecutive clean samples per well."""
    required_columns = [well_column, *log_columns]
    if depth_column is not None:
        required_columns.append(depth_column)
    if segment_length <= 0 or any(column not in data for column in required_columns):
        raise ValueError("Invalid inputs for create_segments")

    segments = []
    segment_metadata = []
    for well_name, well_data in data.groupby(well_column, sort=False):
        if depth_column is not None:
            well_data = well_data.sort_values(depth_column)

        values = well_data[list(log_columns)].to_numpy(dtype=float)
        clean_positions = np.flatnonzero(np.isfinite(values).all(axis=1))
        if clean_positions.size == 0:
            continue

        interval_breaks = np.flatnonzero(np.diff(clean_positions) > 1) + 1
        for interval in np.split(clean_positions, interval_breaks):
            usable_length = len(interval) - (len(interval) % segment_length)
            for offset in range(0, usable_length, segment_length):
                positions = interval[offset : offset + segment_length]
                segment_rows = well_data.iloc[positions]
                segments.append(
                    segment_rows[list(log_columns)].to_numpy(dtype=np.float32)
                )

                metadata = {
                    "WELL": well_name,
                    "SOURCE_FILE": (
                        segment_rows["SOURCE_FILE"].iloc[0]
                        if "SOURCE_FILE" in segment_rows
                        else None
                    ),
                }
                if depth_column is not None:
                    metadata["START_DEPTH"] = segment_rows[depth_column].iloc[0]
                    metadata["END_DEPTH"] = segment_rows[depth_column].iloc[-1]
                segment_metadata.append(metadata)

    if segments:
        segment_array = np.stack(segments)
    else:
        segment_array = np.empty(
            (0, segment_length, len(log_columns)), dtype=np.float32
        )
    return segment_array, pd.DataFrame(segment_metadata)
