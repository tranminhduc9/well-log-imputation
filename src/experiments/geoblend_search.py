"""Seeded GeoBlend-only random search on validation; never evaluates test."""
from argparse import Namespace
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import pandas as pd

from src.experiments.geolink import run_experiment, save_json


def candidates(baseline, count, seed):
    if count < 1:
        raise ValueError('n_trials must be positive')
    rng = random.Random(seed)
    result = [deepcopy(baseline)]
    while len(result) < count:
        config = deepcopy(baseline)
        config.update(epochs=200, patience=30)
        config['tree_settings'].update(
            n_estimators=2000, early_stopping_rounds=80,
            max_depth=rng.choice([3, 4, 6, 8]),
            learning_rate=rng.choice([0.015, 0.035, 0.07]),
            ridge_alpha=rng.choice([0.5, 2., 8.]))
        config['temporal_settings'].update(
            channels=rng.choice([16, 32, 64]),
            dilations=rng.choice([(1, 2, 4, 8, 16), (1, 2, 4, 8, 16, 32),
                                  (1, 2, 4, 8, 16, 32, 64)]),
            kernel_size=rng.choice([3, 5, 7]),
            batch_size=rng.choice([32, 64]),
            learning_rate=rng.choice([0.0003, 0.001, 0.003]),
            weight_decay=rng.choice([0., 0.0001, 0.001]))
        if config not in result:
            result.append(config)
    return result


def validation_score(folder):
    frame = pd.read_csv(Path(folder) / 'summary.csv')
    frame = frame[frame.split.eq('val') & frame.model.eq('geo_blend')]
    if len(frame) != 4 or not np.isfinite(frame.rmse_mean).all():
        raise ValueError('Expected four finite validation scenario scores')
    return float(frame.rmse_mean.mean())


def search(args, project_root, settings, n_trials=16, top_k=3, search_seed=912):
    """Screen on first seed and confirm shortlist on ALL seeds; retain the winner.

    The baseline always reaches confirmation. No test results enter selection.
    Every trial is a self-contained experiment with checkpoints and provenance.
    """
    if not 1 <= top_k <= n_trials:
        raise ValueError('Require 1 <= top_k <= n_trials')
    base = Path(args.output_dir or Path(project_root) / 'results/geoblend_search')
    root = base / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid4().hex[:8])
    root.mkdir(parents=True)
    options = candidates(settings['geo_blend'], n_trials, search_seed)
    save_json(root / 'search_plan.json', dict(candidates=options, seeds=args.seeds,
              top_k=top_k, search_seed=search_seed, objective='mean validation RMSE across four scenarios and seeds',
              protocol='first-seed screening; shortlist plus baseline across all seeds; validation only; no final retraining or test evaluation'))
    def run(folder, models, seeds, config, cache=None):
        current = deepcopy(settings)
        current['geo_blend'] = config
        config_args = Namespace(**{**vars(args), 'output_dir': root / folder,
                                   'models': models, 'seeds': seeds, 'evaluate_test': False,
                                   'report_models': ('geo_blend',),
                                   'saits_cache': cache})
        return run_experiment(config_args, project_root, current)

    # Train SAITS once per seed; each subsequent experiment copies its checkpoint.
    cache = run('saits_cache', ('saits',), args.seeds, options[0])
    records = []
    def trial(index, stage, seeds):
        folder = run(f'{stage}/trial_{index:03d}', ('saits', 'geo_blend'), seeds, options[index], cache)
        row = dict(trial=index, stage=stage, validation_rmse=validation_score(folder), output_dir=str(folder))
        records.append(row)
        pd.DataFrame(records).to_csv(root / 'trials.csv', index=False)
        print(f'{stage} trial {index}: validation RMSE={row["validation_rmse"]:.6f}', flush=True)
        return row
    screened = [trial(i, 'screen', args.seeds[:1]) for i in range(n_trials)]
    shortlist = sorted({0, *[r['trial'] for r in sorted(screened, key=lambda r: (r['validation_rmse'], r['trial']))[:top_k]]})
    confirmed = [trial(i, 'confirm', args.seeds) for i in shortlist]
    best = min(confirmed, key=lambda r: (r['validation_rmse'], r['trial']))
    chosen = {'geo_blend': options[best['trial']]}
    save_json(root / 'best_model_settings.json', chosen)
    save_json(root / 'selection.json', best)
    winner = Path(best['output_dir'])
    save_json(root / 'completed.json', dict(best_output=str(winner), best_trial=best['trial'], evaluate_test=False))
    return root, winner
