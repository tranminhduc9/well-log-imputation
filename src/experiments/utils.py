"""JSON files, fingerprints and reproducible random seeds."""

import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, data):
    """Atomic strict JSON: undefined statistics are null, never NaN literals."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(data), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed):
    # Required before CUDA BLAS initialization for deterministic operations.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def source_info(project_root):
    """Record code fingerprints and versions without copying the source tree."""
    root = Path(project_root)
    sources = [*root.joinpath('src').rglob('*.py'), root / 'requirements.txt',
               *root.joinpath('notebook').glob('*.ipynb')]
    return {
        'source_sha256': {path.relative_to(root).as_posix(): sha256(path) for path in sources},
        'python': platform.python_version(), 'numpy': np.__version__,
        'pandas': pd.__version__, 'torch': str(torch.__version__),
        'deterministic_algorithms': True,
    }
