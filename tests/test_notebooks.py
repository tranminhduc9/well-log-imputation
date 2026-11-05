"""Notebook syntax, experiment settings, tuning scores and plotted convergence."""

import argparse
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.inputtransformer2 import TransformerManager

from src.experiments.registry import make_config

ROOT = Path(__file__).resolve().parents[1]


def code_cells(name):
    notebook = json.loads((ROOT / 'notebook' / f'{name}.ipynb').read_text(encoding='utf-8'))
    return [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']


def code_cell(name, marker):
    return next(source for source in code_cells(name) if marker in source)


def notebook_ast(name):
    transformer = TransformerManager()
    return ast.parse('\n'.join(transformer.transform_cell(source) for source in code_cells(name)))


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(plt.close, 'all')
        self.root = Path(self.temporary.name)

    def test_all_notebook_cells_compile(self):
        transformer = TransformerManager()
        for path in sorted((ROOT / 'notebook').glob('*.ipynb')):
            for index, source in enumerate(code_cells(path.stem)):
                with self.subTest(notebook=path.name, cell=index):
                    compile(transformer.transform_cell(source), f'{path.name}:cell{index}', 'exec')

    def test_training_configurations_match_notebook_settings(self):
        tree = notebook_ast('train_models')
        names = {'SEEDS', 'NEURAL_EPOCHS', 'NEURAL_PATIENCE', 'MODEL_SETTINGS', 'MODELS', 'DEVICE'}
        constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in names}
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'args')
        scope = {'Namespace': argparse.Namespace, 'DATA_ROOT': self.root / 'data',
                 'OUTPUT_ROOT': self.root / 'results', **constants}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'training_args', 'exec'), scope)
        args = scope['args']
        self.assertTrue(args.seeds)
        self.assertEqual(len(args.seeds), len(set(args.seeds)))
        self.assertEqual(args.xgb_estimators, constants['MODEL_SETTINGS']['xgboost']['n_estimators'])
        for name in args.models:
            with self.subTest(model=name), patch('torch.cuda.is_available', return_value=True):
                config = make_config(name, {'seq_len': 256, 'n_features': 4},
                                     constants['MODEL_SETTINGS'], args.seeds[0], args, self.root)
                self.assertEqual(config.seed, args.seeds[0])
                if name in ('brits', 'saits', 'conv_saits'):
                    self.assertEqual(config.epochs, constants['NEURAL_EPOCHS'])
                    self.assertEqual(config.patience, constants['NEURAL_PATIENCE'])

    def test_epoch_plot_excludes_xgboost_from_unfiltered_runs_and_history(self):
        tree = notebook_ast('train_models')
        filtering = next(node for node in tree.body if isinstance(node, ast.Assign)
                         and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'neural_history')
        plot_code = code_cell('train_models', 'neural_runs =')
        neural_runs = [dict(model=name, seed=912, label=name, best_epoch=1)
                       for name in ('saits', 'conv_saits')]
        neural_rows = [dict(model=run['model'], seed=912, epoch=1, loss=.5, validation_rmse=.4)
                       for run in neural_runs]
        for include_xgboost in (False, True):
            with self.subTest(include_xgboost=include_xgboost):
                output = self.root / str(include_xgboost)
                output.mkdir()
                runs, rows = list(neural_runs), list(neural_rows)
                if include_xgboost:
                    runs.append(dict(model='xgboost', seed=912, label='XGBoost', best_epoch=None))
                    rows.append(dict(model='xgboost', seed=912, log_index=0, iteration=1,
                                     train_rmse=.7, validation_rmse=.6))
                figures = []
                scope = dict(SEEDS=(912,), NEURAL_MODELS=('brits', 'saits', 'conv_saits'),
                             runs=pd.DataFrame(runs), history=pd.DataFrame(rows), output_dir=output, plt=plt)
                exec(compile(ast.Module(body=[filtering], type_ignores=[]), 'neural_filter', 'exec'), scope)
                with patch.object(plt, 'show', side_effect=lambda: figures.append(plt.gcf())):
                    exec(compile(plot_code, 'convergence_plot', 'exec'), scope)
                self.assertTrue((output / 'convergence_seed_912.png').is_file())
                for ax in figures[0].axes:
                    self.assertEqual({line.get_label() for line in ax.lines}, {'saits', 'conv_saits'})
                    for line in ax.lines:
                        self.assertTrue(np.isfinite(line.get_xdata()).all())
                        self.assertTrue(np.isfinite(line.get_ydata()).all())

    def test_tuning_weights_scenarios_and_seeds_equally(self):
        node = next(node for node in notebook_ast('finetune_hyper').body
                    if isinstance(node, ast.FunctionDef) and node.name == 'validation_score')
        scenarios = ['Single', 'Block-20', 'Block-100', 'Entire-Log']
        seeds = (129, 219, 309)
        scope = dict(pd=pd, np=np, SEEDS=seeds, SCENARIOS=scenarios)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'validation_score', 'exec'), scope)
        rows = [dict(model='brits', split='val', level='overall', units='normalized', seed=seed,
                     scenario=scenario, rmse=value + i, count=count)
                for i, seed in enumerate(seeds)
                for scenario, value, count in zip(scenarios, [1., 1., 1., 9.], [1, 20, 100, 256])]
        pd.DataFrame(rows).to_csv(self.root / 'metrics_by_seed.csv', index=False)
        # Per-seed means are 3, 4, 5 despite very different gap lengths.
        np.testing.assert_allclose(scope['validation_score'](self.root, 'brits'), [4., 1.])
        for invalid, error in [(rows[:-1], AssertionError), (rows + [rows[0]], ValueError)]:
            with self.subTest(error=error.__name__):
                pd.DataFrame(invalid).to_csv(self.root / 'metrics_by_seed.csv', index=False)
                with self.assertRaises(error):
                    scope['validation_score'](self.root, 'brits')

    def test_visualization_uses_window_five_and_recorded_checkpoint_time(self):
        models = ['brits', 'saits', 'conv_saits']
        values = [7., 5., 3., 1., 2., 4., 6.]
        history = pd.DataFrame([
            dict(model=model, seed=seed, epoch=epoch, epoch_seconds=60.,
                 validation_rmse=value + (100 if seed == 219 else 0))
            for model in models for seed in (912, 219) for epoch, value in enumerate(values, 1)
        ])
        figures = []
        scope = dict(pd=pd, plt=plt, history=history, SEED=912, SMOOTH_WINDOW=5,
                     NEURAL_MODELS=models, LABELS=dict(zip(models, models)),
                     COLORS=dict(zip(models, ['#bc8220', '#4477aa', '#228866'])),
                     runs=pd.DataFrame([dict(model=model, seed=912, best_epoch=4) for model in models]),
                     save_figure=lambda fig, name: figures.append(fig))
        exec(code_cell('visualize_results', 'curves = history'), scope)
        exec(code_cell('visualize_results', "plot_convergence('elapsed_minutes'"), scope)
        for model, curve in scope['curves'].groupby('model'):
            np.testing.assert_allclose(curve['rmse_smooth'], [5., 4., 3.6, 3., 3.2, 3.25, 4.])
            np.testing.assert_allclose(curve['elapsed_minutes'], np.arange(1, 8))
            self.assertEqual(scope['checkpoints'].loc[model, 'validation_rmse'], 1.)
            self.assertEqual(scope['checkpoints'].loc[model, 'elapsed_minutes'], 4.)
        self.assertEqual(len(figures), 2)
        for fig in figures:
            self.assertFalse(fig.texts)  # No footer descriptions below convergence figures.
            for marker in fig.axes[0].collections:
                np.testing.assert_allclose(marker.get_offsets(), [[4., 1.]])


if __name__ == '__main__':
    unittest.main()
