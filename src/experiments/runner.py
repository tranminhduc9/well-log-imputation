"""Train all requested models/seeds, select on validation, then evaluate test."""

from dataclasses import asdict
import gc
import logging
from pathlib import Path

import joblib
import pandas as pd
import torch

from src.models._training import synchronized_time
from .data import load_and_check_data
from .evaluation import evaluate_details, summarize
from .inference import restore_run
from .registry import MODEL_CLASSES, LABELS, make_config
from .utils import save_json, set_seed, sha256, source_info

LOGGER = logging.getLogger(__name__)


def _save_checkpoint(model, folder):
    """LOCF stores config, XGBoost stores trees, neural models store weights."""
    config = asdict(model.config)
    if model.name == 'locf':
        artifact = folder / 'config.json'
        save_json(artifact, config)
    elif model.name == 'xgboost':
        artifact = folder / 'model.joblib'
        joblib.dump({'config': config, 'models': model.backend.models}, artifact)
    else:
        artifact = folder / 'model.pt'
        torch.save({
            'config': config, 'state_dict': model.backend.network.state_dict(),
            'best_epoch': model.backend.best_epoch,
            'training_history': model.backend.training_history,
        }, artifact)
    return artifact


def _history_rows(model):
    """One row per neural epoch or per boosting round/target log."""
    identity = {'model': model.name, 'seed': model.config.seed}
    if model.name == 'locf':
        return []
    if model.name != 'xgboost':
        return [{**identity, **point} for point in model.backend.training_history]
    rows = []
    for curve in model.backend.training_history:
        validation = curve['validation_rmse']
        for iteration, train_rmse in enumerate(curve['train_rmse'], 1):
            rows.append({
                **identity, 'log_index': curve['log_index'], 'iteration': iteration,
                'train_rmse': train_rmse,
                'validation_rmse': validation[iteration - 1] if validation else None,
            })
    return rows


def _save_results(output_dir, manifest, metrics, history):
    """Persist progress after each checkpoint and evaluation."""
    if output_dir is None:
        return
    save_json(output_dir / 'experiment_results.json', manifest)
    pd.DataFrame(metrics).to_csv(output_dir / 'metrics_by_seed.csv', index=False)
    history_columns = None if history else ['model', 'seed', 'epoch', 'loss']
    pd.DataFrame(history, columns=history_columns).to_csv(output_dir / 'training_history.csv', index=False)
    summary = summarize(metrics)
    summary.to_csv(output_dir / 'summary.csv', index=False)


def run_experiment(args, project_root, settings, *, save_outputs=True):
    """Run fixed settings; test starts only after all checkpoints are saved.

    Paths and device come directly from the notebook. With evaluate_test=False,
    test data is never opened. LOCF runs once because it has no learned weights.
    Results use args.output_dir directly; reruns overwrite files at stable paths.
    With save_outputs=False, validation metrics are returned as a DataFrame and
    no files are written. This mode is for tuning and forbids test evaluation.
    """
    if not save_outputs and args.evaluate_test:
        raise ValueError('In-memory tuning must not evaluate test data.')
    names, seeds = tuple(args.models), tuple(args.seeds)
    if not names or len(set(names)) != len(names) or set(names) - MODEL_CLASSES.keys():
        raise ValueError('Select unique supported model names.')
    if not seeds or len(set(seeds)) != len(seeds) or any(
        type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds
    ):
        raise ValueError('Seeds must be unique integers in [0, 2**32).')

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir).resolve()
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
    run_id = output_dir.name
    manifest = {
        'run_id': run_id, 'status': 'validating_data', 'models': names, 'seeds': seeds,
        'data_root': str(data_root.resolve()), 'evaluate_test': args.evaluate_test, 'runs': [],
        'selection': 'Mean validation RMSE across four scenarios; no seed selection.',
        'std_definition': 'Sample standard deviation across seeds (ddof=1).',
        'test_role': 'Development test split; not an untouched holdout.',
    }
    metrics, history = [], []

    result_dir = output_dir if save_outputs else None
    logger = logging.getLogger()
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        # 1. Read data and validate every configuration before training.
        preprocessing, data, metadata, hashes = load_and_check_data(data_root, args.evaluate_test)
        manifest.update(data_sha256=hashes)
        if save_outputs:
            manifest['provenance'] = source_info(project_root)
        manifest['split_counts'] = {
            split: {'segments': len(frame), 'wells': frame['WELL'].nunique()}
            for split, frame in metadata.items()
        }
        if save_outputs:
            save_json(output_dir / 'preprocessing_parameters.json', preprocessing)
        common = {'seq_len': preprocessing['segment_length'], 'n_features': len(preprocessing['log_columns'])}
        validation = {**next(iter(data['val'].values())), 'validation_scenarios': data['val']}
        for name in names:
            make_config(name, common, settings, seeds[0], args, output_dir)
        manifest['status'] = 'training'
        _save_results(result_dir, manifest, metrics, history)

        # 2. Train, save the selected checkpoint, then report validation metrics.
        for name in names:
            for seed in (seeds[:1] if name == 'locf' else seeds):
                set_seed(seed)
                folder = output_dir / name / f'seed_{seed}'
                if save_outputs:
                    folder.mkdir(parents=True, exist_ok=True)
                config = make_config(name, common, settings, seed, args, folder)
                model = MODEL_CLASSES[name](config)
                LOGGER.info('%s | seed=%d | %s', LABELS[name], seed, asdict(config))
                fit_seconds = None
                if name != 'locf':
                    monitor = data['val']['Block-20'] if name == 'xgboost' else validation
                    started = synchronized_time(config.device)
                    model.fit(data['train'], monitor)
                    fit_seconds = synchronized_time(config.device) - started
                artifact = _save_checkpoint(model, folder) if save_outputs else None
                manifest['runs'].append({
                    'model': name, 'label': LABELS[name], 'seed': seed, 'config': asdict(config),
                    'fit_seconds': fit_seconds, 'best_epoch': getattr(model.backend, 'best_epoch', None),
                    'parameters': (sum(p.numel() for p in model.backend.network.parameters())
                                   if name not in ('locf', 'xgboost') else (0 if name == 'locf' else None)),
                    'artifact': str(artifact.relative_to(output_dir)) if artifact else None,
                    'sha256': sha256(artifact) if artifact else None,
                })
                history.extend(_history_rows(model))
                _save_results(result_dir, manifest, metrics, history)
                metrics.extend(evaluate_details(model, data['val'], preprocessing, name, seed, 'val'))
                _save_results(result_dir, manifest, metrics, history)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 3. Evaluate frozen checkpoints on test; test never selects a model.
        if args.evaluate_test:
            manifest['status'] = 'test_evaluation'
            _save_results(result_dir, manifest, metrics, history)
            for run in manifest['runs']:
                model = restore_run({**run, 'artifact': str(output_dir / run['artifact'])},
                                    device=run['config']['device'] if run['model'] != 'locf' else 'cpu',
                                    batch_size=run['config']['batch_size'])
                metrics.extend(evaluate_details(model, data['test'], preprocessing,
                                                run['model'], run['seed'], 'test'))
                _save_results(result_dir, manifest, metrics, history)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        manifest['status'] = 'complete'
        _save_results(result_dir, manifest, metrics, history)
    except Exception as error:
        # Keep completed checkpoints and the reason a partial run stopped.
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        if save_outputs:
            save_json(output_dir / 'experiment_results.json', manifest)
        LOGGER.exception('Experiment failed')
        raise
    finally:
        logger.setLevel(previous_level)

    return output_dir if save_outputs else pd.DataFrame(metrics)
