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


def summarize_test_models(rows, runs):
    """Return one test row per model and scenario, with MAPE aggregated per seed."""
    frame = pd.DataFrame(rows)
    overall = frame.query("split == 'test' and level == 'overall' and units == 'normalized'")
    logs = frame.query("split == 'test' and level == 'log' and units == 'original' "
                       "and group in ['GR', 'RHOB', 'DTC']")
    keys = ['model', 'scenario', 'seed']
    mape = logs.groupby(keys)['mape'].mean() if 'mape' in logs else None
    overall = overall.drop(columns='mape', errors='ignore')
    if mape is not None:
        overall = overall.join(mape, on=keys)
    summary = summarize(overall.to_dict('records'))
    parameters = pd.DataFrame(runs).groupby('model')['parameters'].first().astype('Int64')
    summary['parameters'] = summary['model'].map(parameters)
    columns = ['label', 'scenario', 'n_runs', 'parameters']
    columns += [f'{metric}_{stat}' for metric in ('rmse', 'mae', 'mape', 'mse', 'r2')
                for stat in ('mean', 'std')]
    return summary.reindex(columns=columns).rename(columns={'label': 'model', 'n_runs': 'n_seeds'})


def format_test_comparison(summary, scenarios):
    """Format one CSV cell per metric and group rows by scenario."""
    ordered = summary.sort_values(
        'scenario', key=lambda values: pd.Categorical(values, categories=list(scenarios)),
        kind='stable',
    )
    table = ordered[['model', 'scenario']].rename(columns={'model': 'Mô hình', 'scenario': 'Kịch bản'})
    for metric, label in [('rmse', 'RMSE'), ('mae', 'MAE'), ('mape', 'MAPE (%)'),
                          ('mse', 'MSE'), ('r2', 'R²')]:
        mean = ordered[f'{metric}_mean'].map('{:.4f}'.format, na_action='ignore').astype('string')
        std = ordered[f'{metric}_std'].map('{:.4f}'.format, na_action='ignore').astype('string')
        table[label] = mean.where(std.isna(), mean + ' ± ' + std)
    table['Số tham số'] = ordered['parameters']
    return table
