"""Paired comparisons to SAITS; no test-based model selection."""
import numpy as np
import pandas as pd


def compare_to_saits(rows, *, bootstrap_samples=2000, random_state=912,
                     practical_threshold=10.0):
    """Match seeds/masks and bootstrap entire wells, never individual depths.

    CIs are conditional on the available training seeds and fixed masks. They
    describe well sampling uncertainty, not a fresh holdout or training-seed CI.
    Each model must have exactly the same seeds as SAITS to receive a verdict.
    """
    frame = pd.DataFrame(rows)
    if frame.empty or 'saits' not in set(frame.model):
        return pd.DataFrame(), pd.DataFrame()
    keys = ['seed', 'split', 'scenario', 'level', 'group', 'units']
    baseline = frame[frame.model == 'saits']
    if baseline.duplicated(keys).any():
        raise ValueError('Duplicate SAITS comparison rows.')
    results, paired = [], []
    rng = np.random.default_rng(random_state)
    for name in frame.model.unique():
        if name == 'saits':
            continue
        candidate = frame[frame.model == name]
        if candidate.duplicated(keys).any():
            raise ValueError('Duplicate candidate comparison rows.')
        merged = candidate.merge(baseline, on=keys, suffixes=('', '_saits'), validate='one_to_one')
        if merged.empty:
            continue
        if (merged['count'] != merged.count_saits).any():
            raise ValueError('Mismatched scoring counts in SAITS comparison.')
        for metric in ('mae', 'rmse', 'mse'):
            merged[f'{metric}_gain_pct'] = 100 * (1 - merged[metric] / merged[f'{metric}_saits'])
        merged['r2_delta'] = merged.r2 - merged.r2_saits
        paired.extend(merged.to_dict('records'))
        overall = merged[(merged.level == 'overall') & (merged.units == 'normalized')]
        for (split, scenario), group in overall.groupby(['split', 'scenario'], sort=False):
            reference_seeds = set(baseline[(baseline.split == split) & (baseline.scenario == scenario)].seed)
            candidate_seeds = set(candidate[(candidate.split == split) & (candidate.scenario == scenario)].seed)
            complete = reference_seeds == candidate_seeds
            row = dict(model=name, label=group.label.iloc[0], split=split, scenario=scenario,
                       n_paired_seeds=len(group), complete_seed_pairing=complete,
                       practical_threshold_pct=practical_threshold)
            for metric in ('mae', 'rmse', 'mse'):
                row[f'{metric}_saits'] = group[f'{metric}_saits'].mean()
                row[metric] = group[metric].mean()
                row[f'{metric}_gain_pct'] = 100 * (1 - row[metric] / row[f'{metric}_saits'])
                row[f'{metric}_gain_seed_std'] = group[f'{metric}_gain_pct'].std(ddof=1)
            row['r2_delta'] = group.r2_delta.mean()
            row['all_seeds_improve_mae_rmse'] = bool(((group.mae_gain_pct > 0) & (group.rmse_gain_pct > 0)).all())
            wells = merged[(merged.split == split) & (merged.scenario == scenario) &
                           (merged.level == 'well') & (merged.units == 'normalized')]
            row['n_wells'] = wells.group.nunique()
            for metric in ('mae', 'rmse'):
                row[f'{metric}_gain_ci_low'] = np.nan
                row[f'{metric}_gain_ci_high'] = np.nan
            # Preserve paired seeds within each sampled well, then average
            # seed-level metrics (RMSE after pooling squared errors).
            if row['n_wells'] >= 2 and not wells.empty:
                well_names = sorted(wells.group.unique())
                seeds = sorted(group.seed.unique())
                index = pd.MultiIndex.from_product([seeds, well_names], names=['seed', 'group'])
                aligned = wells.set_index(['seed', 'group']).reindex(index)
                if aligned['count'].notna().all():
                    count = aligned['count'].to_numpy().reshape(len(seeds), -1)
                    draws = rng.integers(0, len(well_names), (bootstrap_samples, len(well_names)))
                    for metric, source in [('mae', 'mae'), ('rmse', 'mse')]:
                        estimates = []
                        for suffix in ('', '_saits'):
                            totals = (aligned[source + suffix].to_numpy().reshape(len(seeds), -1) * count)
                            value = totals[:, draws].sum(axis=2) / count[:, draws].sum(axis=2)
                            if metric == 'rmse':
                                value = np.sqrt(value)
                            estimates.append(value.mean(axis=0))
                        gains = 100 * (1 - estimates[0] / estimates[1])
                        low, high = np.quantile(gains, [.025, .975])
                        row[f'{metric}_gain_ci_low'], row[f'{metric}_gain_ci_high'] = low, high
            practical = row['mae_gain_pct'] >= practical_threshold and row['rmse_gain_pct'] >= practical_threshold
            supported = row['mae_gain_ci_low'] > 0 and row['rmse_gain_ci_low'] > 0
            if not complete or len(group) < 2:
                verdict = 'insufficient_paired_seeds'
            elif practical and supported and row['all_seeds_improve_mae_rmse']:
                verdict = 'clear_gain'
            elif row['mae_gain_pct'] <= 0 or row['rmse_gain_pct'] <= 0:
                verdict = 'mixed_or_no_gain'
            else:
                verdict = 'small_or_uncertain_gain'
            row['verdict'] = verdict
            results.append(row)
    return pd.DataFrame(results), pd.DataFrame(paired)


def write_comparison(rows, output_dir):
    summary, paired = compare_to_saits(rows)
    if summary.empty:
        return
    summary.to_csv(output_dir / 'comparison_vs_saits.csv', index=False)
    paired.to_csv(output_dir / 'paired_vs_saits.csv', index=False)
    # Each scenario has equal weight; Entire-Log must not dominate by gap size.
    overall = pd.DataFrame(rows)
    overall = overall[(overall.level == 'overall') & (overall.units == 'normalized')]
    macro = overall.groupby(['model', 'label', 'split', 'seed'], as_index=False).agg(
        macro_mae=('mae', 'mean'), macro_rmse=('rmse', 'mean'), n_scenarios=('scenario', 'nunique'))
    macro.to_csv(output_dir / 'macro_metrics_by_seed.csv', index=False)
    text = ['# Comparison to SAITS', '',
            'Positive gain means smaller error. Clear gain requires >=10% reductions in BOTH MAE and RMSE,',
            'a positive lower 95% paired well-bootstrap bound for both, and improvement at every matched seed (at least two).',
            'CIs are conditional on the fixed masks and trained seeds; validation estimates are fitted/tuned estimates.',
            'The existing test split is a development benchmark, not an untouched holdout.', '',
            '| Split | Model | Scenario | MAE reduction | RMSE reduction | Verdict |',
            '|---|---|---|---:|---:|---|']
    for row in summary.itertuples():
        text.append(f'| {row.split} | {row.model} | {row.scenario} | {row.mae_gain_pct:.2f}% | {row.rmse_gain_pct:.2f}% | {row.verdict} |')
    (output_dir / 'comparison_vs_saits.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
