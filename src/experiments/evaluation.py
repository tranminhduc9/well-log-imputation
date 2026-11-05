"""Masked evaluation in normalized/original units, then aggregation across seeds."""

import numpy as np
import pandas as pd

from src.data.metrics import compute_imputation_metrics
from .registry import LABELS

METRICS = ['mae', 'mse', 'rmse', 'r2', 'mape']
GROUPS = ['model', 'label', 'split', 'scenario', 'level', 'group', 'units']


def evaluate_details(model, datasets, preprocessing, model_name, seed, split):
    """Predict once per scenario and score the hidden points overall/per log."""
    rows = []
    for scenario, dataset in datasets.items():
        truth, mask = dataset['X_intact'], dataset['indicating_mask']
        inputs = {key: dataset[key] for key in ('X', 'depth') if key in dataset}
        prediction = model.impute(inputs)
        if not np.isfinite(prediction).all():
            raise ValueError(f'{model_name}/{seed}/{split}/{scenario}: non-finite prediction.')
        if not np.allclose(prediction[~mask], dataset['X'][~mask], rtol=1e-6, atol=1e-7):
            raise ValueError(f'{model_name} changed observed values.')

        def record(target, estimate, scored, level, group, units):
            if not scored.any():
                return
            metrics = compute_imputation_metrics(target, estimate, scored, include_mape=units == 'original')
            if metrics['count'] != int(scored.sum()):
                raise ValueError('Evaluation count does not match the fixed mask.')
            rows.append({
                'model': model_name, 'label': LABELS[model_name], 'seed': seed,
                'split': split, 'scenario': scenario, 'level': level,
                'group': str(group), 'units': units, **metrics,
            })

        record(truth, prediction, mask, 'overall', 'all', 'normalized')
        for index, log in enumerate(preprocessing['log_columns']):
            target, estimate, scored = truth[..., index], prediction[..., index], mask[..., index]
            record(target, estimate, scored, 'log', log, 'normalized')
            scale = preprocessing['normalization'][log]
            record(target * scale['std'] + scale['mean'], estimate * scale['std'] + scale['mean'],
                   scored, 'log', log, 'original')
    return rows


def summarize(rows):
    """Average each metric across seeds; std is the sample std (ddof=1)."""
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(GROUPS, sort=False, dropna=False)
    if frame.duplicated([*GROUPS, 'seed']).any() or (grouped['count'].nunique() != 1).any():
        raise ValueError('Duplicate seed results or inconsistent evaluation counts.')
    metrics = frame.reindex(columns=METRICS).dropna(axis='columns', how='all').columns
    statistics = grouped[list(metrics)].agg(['mean', 'std'])
    statistics.columns = [f'{metric}_{stat}' for metric, stat in statistics.columns]
    counts = grouped.agg(n_runs=('seed', 'size'), count=('count', 'first'))
    return counts.join(statistics).reset_index()
