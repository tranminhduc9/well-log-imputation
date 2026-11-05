"""Model names and configuration construction used by the experiment runner."""

import torch

from src.models.base import ModelConfig
from src.models.locf import LOCF
from src.models.xgboost import XGBoost, XGBoostConfig
from src.models.brits import BRITS, BRITSConfig
from src.models.saits import SAITS, SAITSConfig
from src.models.conv_saits import ConvSAITS, ConvSAITSConfig

MODEL_CLASSES = {'locf': LOCF, 'xgboost': XGBoost, 'brits': BRITS,
                 'saits': SAITS, 'conv_saits': ConvSAITS}
CONFIG_CLASSES = {'locf': ModelConfig, 'xgboost': XGBoostConfig, 'brits': BRITSConfig,
                  'saits': SAITSConfig, 'conv_saits': ConvSAITSConfig}
LABELS = {'locf': 'LOCF', 'xgboost': 'XGBoost', 'brits': 'BRITS + MIT',
          'saits': 'SAITS (segment MIT)', 'conv_saits': 'Conv-SAITS (segment MIT)'}


def check_device(device):
    """Use the requested device explicitly; fail if CUDA cannot run."""
    if device not in {'cpu', 'cuda'}:
        raise ValueError('device must be cpu or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return device


def make_config(name, common, settings, seed, args, artifact_dir):
    parameters = {**common, **settings.get(name, {}), 'seed': seed, 'output_dir': artifact_dir}
    device = check_device(getattr(args, 'device', 'cpu'))
    if name == 'xgboost':
        parameters.update(n_estimators=args.xgb_estimators, device=device)
    elif name != 'locf':
        parameters.update(epochs=getattr(args, f'{name}_epochs'),
                          patience=getattr(args, f'{name}_patience'), device=device)
    return CONFIG_CLASSES[name](**parameters)
