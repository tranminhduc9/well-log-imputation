"""Notebook syntax, experiment settings, tuning scores and plotted convergence."""

import argparse
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.inputtransformer2 import TransformerManager

from src.experiments.registry import MODEL_CLASSES, make_config

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
        self.assertCountEqual(args.seeds, (912, 219, 129))
        self.assertCountEqual(args.models, MODEL_CLASSES)
        self.assertEqual(args.xgb_estimators, constants['MODEL_SETTINGS']['xgboost']['n_estimators'])
        for name in args.models:
            with self.subTest(model=name), patch('torch.cuda.is_available', return_value=True):
                config = make_config(name, {'seq_len': 256, 'n_features': 4},
                                     constants['MODEL_SETTINGS'], args.seeds[0], args, self.root)
                self.assertEqual(config.seed, args.seeds[0])
                if name in ('brits', 'saits', 'flexlognet', 'conv_saits'):
                    self.assertEqual(config.epochs, constants['NEURAL_EPOCHS'])
                    self.assertEqual(config.patience, constants['NEURAL_PATIENCE'])

    def test_training_notebook_publishes_only_the_comparison(self):
        source = code_cell('train_models', 'csv_path =')
        path = self.root / 'test_model_comparison.csv'
        pd.DataFrame({'Mô hình': ['SAITS'], 'RMSE': ['0.1236 ± 0.0093']}).to_csv(path, index=False)
        with patch('src.experiments.runner.run_experiment', return_value=path) as run:
            displays = []
            scope = dict(run_experiment=run, PROJECT_ROOT=ROOT, MODEL_SETTINGS={'xgboost': {'n_estimators': 2}},
                         Namespace=argparse.Namespace, DATA_ROOT=self.root, OUTPUT_ROOT=self.root,
                         DEVICE='cpu', MODELS=('locf',), SEEDS=(129,), NEURAL_EPOCHS=1,
                         NEURAL_PATIENCE=1, pd=pd, display=displays.append, FileLink=Path)
            exec(compile(source, 'training_comparison', 'exec'), scope)
        run.assert_called_once_with(scope['args'], ROOT, scope['MODEL_SETTINGS'], summary_only=True)
        self.assertEqual(list(self.root.iterdir()), [path])
        self.assertEqual(displays[-1], path)
        self.assertEqual(displays[0].at[0, 'RMSE'], '0.1236 ± 0.0093')

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
        # Per-seed means are 3, 4, 5 despite very different gap lengths.
        np.testing.assert_allclose(scope['validation_score'](pd.DataFrame(rows), 'brits'), [4., 1.])
        for invalid, error in [(rows[:-1], AssertionError), (rows + [rows[0]], ValueError)]:
            with self.subTest(error=error.__name__):
                with self.assertRaises(error):
                    scope['validation_score'](pd.DataFrame(invalid), 'brits')

    def test_tuning_archive_includes_only_requested_files(self):
        expected = {'search_trials.csv', 'top3_validation_rmse.csv',
                    'best_model_settings.json', 'search_protocol.json'}
        output = self.root / 'well-log-tuning'
        output.mkdir()
        for name in expected | {'obsolete_details.csv'}:
            (output / name).write_bytes(b'test')
        scope = dict(OUTPUT_DIR=output, ZipFile=ZipFile, ZIP_DEFLATED=ZIP_DEFLATED,
                     display=lambda *args: None, FileLink=lambda path: path)
        exec(code_cell('finetune_hyper', 'result_files ='), scope)
        self.assertEqual(scope['archive_path'], self.root / 'well-log-tuning.zip')
        with ZipFile(scope['archive_path']) as archive:
            self.assertEqual(set(archive.namelist()), expected)

    def test_tuning_exports_compact_trials_and_only_top_curves(self):
        grid = [{'learning_rate': rate} for rate in (0.01, 0.02, 0.03, 0.04)]
        scores = {0.01: 0.4, 0.02: 0.2000002, 0.03: 0.2000001, 0.04: 0.2000002}

        def tune(args, project_root, settings, *, save_outputs, return_history):
            self.assertFalse(save_outputs)
            self.assertTrue(return_history)
            model, = args.models
            score = scores[settings[model]['learning_rate']]
            points = dict(seed=[129, 129], validation_rmse=[score + 0.1, score])
            if model == 'xgboost':
                points.update(iteration=[1, 2], log=['GR', 'GR'])
            else:
                points['epoch'] = [1, 2]
            return score, pd.DataFrame(points)

        scope = dict(Namespace=argparse.Namespace, pd=pd, np=np, json=json,
                     GRIDS={'xgboost': grid, 'saits': grid},
                     FIXED_SETTINGS={'locf': {}, 'xgboost': {}, 'saits': {'batch_size': 32}},
                     TRAINING_ARGS={}, SEEDS=(129,), run_experiment=tune,
                     validation_score=lambda score, name: (score, np.nan),
                     save_json=lambda path, data: path.write_text(json.dumps(data)),
                     OUTPUT_DIR=self.root, PROJECT_ROOT=ROOT, display=lambda value: None)
        exec(code_cell('finetune_hyper', 'def export_top_curves'), scope)
        trials = pd.read_csv(self.root / 'search_trials.csv')
        self.assertEqual(list(trials), ['model', 'trial', 'validation_rmse_mean'])
        self.assertEqual(len(trials), 8)
        curves = pd.read_csv(self.root / 'top3_validation_rmse.csv')
        self.assertEqual(len(curves), 12)
        for name, frame in curves.groupby('model'):
            descriptions = frame.dropna(subset=['settings'])
            self.assertEqual(descriptions['rank'].tolist(), [1, 2, 3])
            self.assertEqual(descriptions['trial'].tolist(), [3, 2, 4])
            self.assertEqual(json.loads(descriptions.iloc[0]['settings'])['learning_rate'], 0.03)
            self.assertEqual(frame['step'].tolist(), [1, 2] * 3)
            self.assertTrue(frame['step_type'].eq('iteration' if name == 'xgboost' else 'epoch').all())
        self.assertEqual(curves.query("model == 'xgboost'")['log'].unique().tolist(), ['GR'])

    def test_visualization_reads_only_the_comparison_csv(self):
        models_dir = self.root / 'results' / 'models'
        models_dir.mkdir(parents=True)
        csv_path = models_dir / 'test_model_comparison.csv'
        metrics = ['RMSE', 'MAE', 'MAPE (%)', 'MSE', 'R²']
        pd.DataFrame({
            'Mô hình': ['LOCF', 'FlexLogNet', 'LOCF', 'FlexLogNet'],
            'Kịch bản': ['Single', 'Single', 'Block-20', 'Block-20'],
            **{metric: ['0.1000', '0.2000 ± 0.0300', '0.3000', '0.4000 ± 0.0500']
               for metric in metrics},
            'Số tham số': [0, 100, 0, 100],
        }).to_csv(csv_path, index=False, encoding='utf-8-sig')
        displays, figures, scope = [], [], {}
        with patch.object(Path, 'cwd', return_value=self.root / 'notebook'):
            exec(code_cell('visualize_results', 'CSV_PATH ='), scope)
        scope['display'] = displays.append
        exec(code_cell('visualize_results', 'display(summary)'), scope)
        with patch.object(plt, 'show', side_effect=lambda: figures.append(plt.gcf())):
            exec(code_cell('visualize_results', 'def plot_metric'), scope)
        self.assertEqual(scope['CSV_PATH'], csv_path)
        self.assertEqual(displays[0]['Số tham số'].tolist(), ['0', '100', '0', '100'])
        self.assertEqual(displays[0].at[1, 'RMSE'], '0.2000 ± 0.0300')
        self.assertEqual(set(path.name for path in scope['FIG_DIR'].iterdir()),
                         {'test_rmse.png', 'test_mae.png', 'test_mape.png', 'test_mse.png', 'test_r2.png'})
        self.assertEqual(len(figures), 5)
        for figure in figures:
            axis, = figure.axes
            locf, flex, errors = axis.containers
            self.assertEqual(locf.get_label(), 'LOCF')
            self.assertEqual(flex.get_label(), 'FlexLogNet')
            np.testing.assert_allclose(flex.datavalues, [0.2, 0.4])
            segments = errors.lines[2][0].get_segments()
            np.testing.assert_allclose([segment[:, 1] for segment in segments],
                                       [[0.17, 0.23], [0.35, 0.45]])


if __name__ == '__main__':
    unittest.main()
