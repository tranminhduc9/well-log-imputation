"""Split and transform well-log datasets without well leakage."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .clean_data import (
    SELECTED_LOGS,
    fit_outlier_bounds,
    remove_outliers as apply_outlier_bounds,
)


def split_data(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    well_column: str = "WELL",
    random_state: int = 912,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split complete wells into train, validation and test datasets."""
    ratios = (train_ratio, validation_ratio, test_ratio)
    invalid_wells = well_column not in data or (
        well_column in data
        and (data[well_column].isna().any() or data[well_column].nunique() < 4)
    )
    if (
        invalid_wells
        or any(ratio <= 0 for ratio in ratios)
        or not np.isclose(sum(ratios), 1.0)
    ):
        raise ValueError("Invalid inputs for split_data")

    train_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=train_ratio,
        random_state=random_state,
    )
    train_index, remaining_index = next(
        train_splitter.split(data, groups=data[well_column])
    )
    train = data.iloc[train_index].copy()
    remaining = data.iloc[remaining_index].copy()

    validation_share = validation_ratio / (validation_ratio + test_ratio)
    validation_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=validation_share,
        random_state=random_state + 1,
    )
    validation_index, test_index = next(
        validation_splitter.split(remaining, groups=remaining[well_column])
    )

    validation = remaining.iloc[validation_index].copy()
    test = remaining.iloc[test_index].copy()
    return train, validation, test


def remove_outliers(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    log_columns: Sequence[str] = SELECTED_LOGS,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[float, float]],
]:
    """Fit percentile bounds on train and apply them to every split."""
    bounds = fit_outlier_bounds(
        train_data, log_columns, lower_quantile, upper_quantile
    )
    return (
        apply_outlier_bounds(train_data, bounds),
        apply_outlier_bounds(validation_data, bounds),
        apply_outlier_bounds(test_data, bounds),
        bounds,
    )


def normalize_data(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    log_columns: Sequence[str] = SELECTED_LOGS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, float]],
]:
    """Fit train z-score statistics and apply them to every split."""
    datasets = (train_data, validation_data, test_data)
    invalid_columns = any(
        column not in dataset for dataset in datasets for column in log_columns
    ) or any(
        train_data[column].dropna().empty
        for column in log_columns
        if column in train_data
    )
    if invalid_columns:
        raise ValueError("Invalid inputs for normalize_data")

    statistics = {}
    for column in log_columns:
        mean = float(train_data[column].mean())
        standard_deviation = float(train_data[column].std(ddof=0))
        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            standard_deviation = 1.0
        statistics[column] = {"mean": mean, "std": standard_deviation}

    normalized_datasets = []
    for dataset in datasets:
        normalized = dataset.copy()
        for column, values in statistics.items():
            normalized[column] = (
                normalized[column] - values["mean"]
            ) / values["std"]
        normalized_datasets.append(normalized)

    return (*normalized_datasets, statistics)
