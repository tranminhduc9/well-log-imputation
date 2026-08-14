'''
Main script: It runs the proposed experiments for the benchmark
'''
from typing import Callable, Iterator, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging

# PyPOTS 0.3.2 imports the optional Raindrop graph model at startup and logs
# missing PyG packages even though this benchmark does not use Raindrop.
logging.disable(logging.CRITICAL)
from pypots.utils.metrics import calc_mae, calc_mse, calc_rmse
from pypots.utils.random import set_random_seed
logging.disable(logging.NOTSET)

from utils.metrics import cal_r2, cal_cc

import models
from data import mask_X

def select_optimizer(optimizer: str, lr: float):    
    '''
    Returns and enveloped optimizer from the pypots with the desired learning rate.
    '''
    # Import PyPOTS only for neural models. Classical models do not need its
    # optional graph stack (torch_geometric/torch_scatter).
    from pypots.optim import Adam, AdamW

    if optimizer == 'adam':
        return Adam(lr = lr)
    if optimizer == 'adamw':
        return AdamW(lr = lr)
    
    return Adam(lr = lr)


def loading_data(dataset_name: str, data_dir: str, fold: int) -> tuple[np.array, np.array]:
    '''
    Load the training/validation data of a fold partition.
    
    @param dataset_name: Name of the dataset. Ex: geolink, teapot, or taranaki
    @param data_dir: Root folder where the data is located
    @param fold: the desired fold partition to be loaded
    
    @returns data_train, data_val: a Numpy array with the data sequences for training and validation of the models with the formats [n_sequences, sequence_length, n_features] 
    '''
    
    data_train, data_val = None, None
    ## Loading training data
    with open( os.path.join(data_dir, f'{dataset_name.lower()}_fold_{fold}_well_log_sliced_train.npy'), 'rb' ) as f:
        data_train = np.load(file = f)
    ## Loading validation data
    with open( os.path.join(data_dir, f'{dataset_name.lower()}_fold_{fold}_well_log_sliced_val.npy'), 'rb' ) as f:
        data_val = np.load(file = f)
        
    return data_train, data_val

def get_dataset_dict(data: np.ndarray, mode: str, 
                     n_points: int = 1, 
                     b_size: list[int] = [20, 100], 
                     profile: int|None = None, 
                     include_mask: bool = False, 
                     fill: bool = False, do_copy:bool = False) -> dict[str, np.ndarray]:
    '''
    Creates a dictionary with the complete data, the data with artificial missing values, and the mask indicating the position of missing values.
    
    @param data: an ndarray with the data sequences [n_sequences, sequence_length, n_features]
    @param mode: a string indicating the type of missing patter to be applied to the sequences can be one of four: single, block, profile, random
    @param n_points: the number of to be masked on each sequence. Only used if mode == single or random
    @param b_size: the length of a block to be masked on each sequence. Only used if mode == block or random
    @param profile: the position of the profile to be masked, all data for this profile is masked. If is None, one random profile to be masked will be selected for each sequence
    @param include_mask: bool to include the missing mask array. The missing mask marks with zero the positions of missing data
    @param fill: bool to fill the returned data with nan values, or return it intact with the indicating masks.
    @param do_copy: bool to copy the data.
    
    @returns data_dict: a dictionary with the arrays 
    '''
    data_dict = {}
    
    X_intact, X, missing_mask, indicating_mask = mask_X(data,
                                                        mode=mode,
                                                        n_points=n_points,
                                                        b_size=b_size,
                                                        feat=profile)
    if fill:
        X = np.where(missing_mask, X, np.nan)
    
    if do_copy:
        data_dict = {'X': X.copy(),
                     'X_intact': X_intact,
                     'X_ori': X_intact,
                     'indicating_mask': indicating_mask.copy()}
    else:
        data_dict = {'X': X,
                     'X_intact': X_intact,
                     'X_ori': X_intact,
                     'indicating_mask': indicating_mask}
    
    if include_mask:
        data_dict['missing_mask'] = missing_mask.copy() if do_copy else missing_mask
    
    return data_dict
    
def run_experiments(model_fn: models.Factory, experiments: dict[str, list[int]], n_folds: int, 
                    dataset_name: str, data_dir: str, model_name: str = 'model') -> dict[str, dict[str, list[float]] ]:
    '''
    Runs all experiments for a single model
    
    @param model_fn: a factory object that is used to instantiate a new instance of the model to be experimented
    @param experiments: a dictionary with the experiments to be performed. It includes the missing data patterns (single, block, profile) 
                         and the configuration for these patterns (n_points, block_length, etc)
    @param n_folds: the number folds to be tesed and that the data is partioned (usually five fold)
    @param dataset_name: the name of the dataset
    @param data_dir: the path location of the data
    @param model_name: the name of the model. Used in prints/logs
    
    @returns metrics: a dictionary with the validation metrics of the model in all folds
    '''
    ## Create dict for metrics storage
    metrics = {}
    dataset_for_validating = {}

    print(f'\n\ntraining model: {model_name}...')
    logging.info(f'\n\ntraining model: {model_name}...')
    metrics[model_name] = {}
    metrics[model_name][f'training-time'] = []

    for experiment in experiments:
        mode = experiment['mode']
        metrics[model_name][f'inference-time-{mode}'] = []
        metrics[model_name][f'validation-mae-{mode}'] = []
        metrics[model_name][f'validation-mse-{mode}'] = []
        metrics[model_name][f'validation-rmse-{mode}'] = []
        metrics[model_name][f'validation-r2-{mode}'] = []
        metrics[model_name][f'validation-cc-{mode}'] = []

    ## To track training time
    init_time = time.time()

    # 
    for fold in range(n_folds):
        ## instantiate a new model
        model = model_fn.instantiate()
        
        ## To track training time
        init_time_fold = time.time()

        print(f'\ntraining in fold {fold}...')
        logging.info(f'\ntraining in fold {fold}...')
        ## Loading data
        data_train, data_val = loading_data(dataset_name, data_dir, fold)

        # Create a rand masking for training        
        dataset_for_training = get_dataset_dict(data_train, 'rand')
        
        # Create a rand validation set for measuring in training only        
        dataset_for_validating['rand'] = get_dataset_dict(data_val, 'rand', fill=True, do_copy=True)

        for experiment in experiments:
            mode = experiment['mode']
            _mode = mode.split('.')[0]
            
            dataset_for_validating[mode] = get_dataset_dict(data_val, mode=_mode, 
                                                            n_points=experiment['n_points'], 
                                                            b_size=experiment['b_size'],
                                                            profile=experiment['f_pos'],
                                                            fill=True, do_copy=True
                                                            )


        ## Train the model on the training set, and validate it on the validating set to select the best model for testing in the next step

        if model_name.lower() not in {'locf', 'mean'}: ## LOCF and MEAN don't need to be trained
            print('training model in training set and tracking performance in validation set..')
            logging.info('training model in training set and tracking performance in validation set..')
            model.fit(train_set=dataset_for_training, val_set=dataset_for_validating['rand'])

        ## Calculate training time
        training_time_secs = int(time.time() - init_time_fold)
        print(f'final training time in fold {fold}: {training_time_secs // 60}m {training_time_secs % 60}s')
        logging.info(f'final training time in fold {fold}: {training_time_secs // 60}m {training_time_secs % 60}s')

        ## Save metrics for the model:
        metrics[model_name]['training-time'].append(training_time_secs)

        ## Testing stage, impute the originally-missing values and artificially-missing values in the test set
        print('calculate validation set imputation...')
        logging.info('calculate validation set imputation...')
        for experiment in experiments:
            mode = experiment['mode']  
            print(f'mode {mode}...')
            logging.info(f'mode {mode}...')

            inference_start = time.perf_counter()
            model_imputation = model.predict(dataset_for_validating[mode])
            inference_time_secs = time.perf_counter() - inference_start

            print(f'masking {mode} inference time:{inference_time_secs:.6f}s')
            logging.info(f'masking {mode} inference time:{inference_time_secs:.6f}s')

            ## Calculate metrics on the ground truth (artificially-missing values):
            val_mae = calc_mae(model_imputation['imputation'], dataset_for_validating[mode]['X_intact'], dataset_for_validating[mode]['indicating_mask'])
            val_mse = calc_mse(model_imputation['imputation'], dataset_for_validating[mode]['X_intact'], dataset_for_validating[mode]['indicating_mask'])
            val_rmse = calc_rmse(model_imputation['imputation'], dataset_for_validating[mode]['X_intact'], dataset_for_validating[mode]['indicating_mask'])
            val_r2 = cal_r2(model_imputation['imputation'], dataset_for_validating[mode]['X_intact'], dataset_for_validating[mode]['indicating_mask'])
            val_cc = cal_cc(model_imputation['imputation'], dataset_for_validating[mode]['X_intact'], dataset_for_validating[mode]['indicating_mask'])

            # Uncertainty-aware models additionally return a lower and upper
            # predictive bound. Evaluate and persist only artificially masked
            # positions so the artifacts stay compact.
            if {'lower', 'upper'} <= model_imputation.keys():
                uncertainty_mask = dataset_for_validating[mode]['indicating_mask'].astype(bool)
                truth = dataset_for_validating[mode]['X_intact'][uncertainty_mask]
                point = model_imputation['imputation'][uncertainty_mask]
                lower = model_imputation['lower'][uncertainty_mask]
                upper = model_imputation['upper'][uncertainty_mask]
                finite = (
                    np.isfinite(truth) & np.isfinite(point)
                    & np.isfinite(lower) & np.isfinite(upper)
                )
                if not np.any(finite):
                    raise ValueError(
                        f'No finite uncertainty predictions for fold {fold}, mode {mode}.'
                    )
                truth, point, lower, upper = (
                    values[finite] for values in (truth, point, lower, upper)
                )
                val_picp = float(np.mean((truth >= lower) & (truth <= upper)))
                val_mpiw = float(np.mean(upper - lower))
                picp_key = f'validation-picp-{mode}'
                mpiw_key = f'validation-mpiw-{mode}'
                metrics[model_name].setdefault(picp_key, []).append(val_picp)
                metrics[model_name].setdefault(mpiw_key, []).append(val_mpiw)

                uncertainty_dir = os.path.join(model_fn.output_dir, 'uncertainty')
                os.makedirs(uncertainty_dir, exist_ok=True)
                safe_mode = mode.replace('.', '_')
                np.savez_compressed(
                    os.path.join(
                        uncertainty_dir,
                        f'{model_name}_{dataset_name}_fold_{fold}_{safe_mode}.npz',
                    ),
                    indices=np.argwhere(uncertainty_mask)[finite],
                    target=truth,
                    prediction=point,
                    lower=lower,
                    upper=upper,
                )
            
            print(f'masking {mode} testing mean absolute error:{val_mae:.4f}')
            logging.info(f'masking {mode} testing mean absolute error:{val_mae:.4f}')
            print(f'masking {mode} testing mean squared error:{val_mse:.4f}')
            logging.info(f'masking {mode} testing mean squared error:{val_mse:.4f}')
            print(f'masking {mode} testing mean root-mean squared error:{val_rmse:.4f}')
            logging.info(f'masking {mode} testing root-mean squared error:{val_rmse:.4f}')
            print(f'masking {mode} testing mean r2:{val_r2:.4f}')
            logging.info(f'masking {mode} testing r2:{val_r2:.4f}')
            print(f'masking {mode} testing correlation:{val_cc:.4f}')
            logging.info(f'masking {mode} testing correlation:{val_cc:.4f}')
            if {'lower', 'upper'} <= model_imputation.keys():
                print(f'masking {mode} prediction interval coverage (PICP):{val_picp:.4f}')
                logging.info(f'masking {mode} prediction interval coverage (PICP):{val_picp:.4f}')
                print(f'masking {mode} mean prediction interval width (MPIW):{val_mpiw:.4f}')
                logging.info(f'masking {mode} mean prediction interval width (MPIW):{val_mpiw:.4f}')

            ## Save metrics for the model:
            metrics[model_name][f'inference-time-{mode}'].append(inference_time_secs)
            metrics[model_name][f'validation-mae-{mode}'].append(val_mae)
            metrics[model_name][f'validation-mse-{mode}'].append(val_mse)
            metrics[model_name][f'validation-rmse-{mode}'].append(val_rmse)
            metrics[model_name][f'validation-r2-{mode}'].append(val_r2)
            metrics[model_name][f'validation-cc-{mode}'].append(val_cc)

    ## Calculate training time
    training_time_secs = int(time.time() - init_time)
    print(f'final training time: {training_time_secs // 60}m {training_time_secs % 60}s')
    logging.info(f'final training time: {training_time_secs // 60}m {training_time_secs % 60}s')
    print('----------------------------------------------')
    logging.info('----------------------------------------------')
    print('\n')
    logging.info('\n')
    
    return metrics

def print_table(model_name: str, metrics: dict[str, dict[str, list[float]]]) -> None:
    '''
    Print all the metrics for a single model. (Useful to extract this info later for analisys) 
    '''
    metrics = metrics[model_name]

    for name, metric in metrics.items():
        precision = 6 if name.startswith('inference-time-') else 4
        m_str = [f'{x:.{precision}f}' for x in metric]
        m_str = '\t'.join(m_str)
        print(f'{name}:\t {m_str}')
        logging.info(f'{name}: {m_str}')
    
def main(cfg):
    # Setting seed
    set_random_seed(cfg.seed)
    
    # Setting logging file
    log_path = os.path.join(cfg.output_dir, f'{cfg.model}_{cfg.dataset_name}.log')
    logging.basicConfig(filename=log_path, format='%(message)s', level=logging.INFO)
    
    # Model setup
    ## define optimizer
    neural_models = {'saits', 'transformer', 'brits', 'mrnn', 'unet', 'ae'}
    optim = select_optimizer(cfg.optimizer, cfg.lr) if cfg.model in neural_models else None
    
    ## define instantiate model function
    model = models.Factory(cfg.model,
                           seq_len = cfg.slice_len, 
                           n_features = cfg.n_features, 
                           batch_size = cfg.batch_size, 
                           epochs = cfg.epochs, 
                           patience = cfg.patience, 
                           optimizer = optim, 
                           learning_rate = cfg.lr,
                           device = cfg.device, 
                           output_dir = cfg.output_dir)
    

    # Run experiments
    metrics = run_experiments(model, cfg.experiments, cfg.n_folds, cfg.dataset_name, cfg.dataset_dir, model_name=cfg.model)
    
    # Log all validation metrics of the model
    print_table(cfg.model, metrics)

if __name__ == '__main__':
    from cfg import Configs
    cfg = Configs().parse_args()
    main(cfg)
    
