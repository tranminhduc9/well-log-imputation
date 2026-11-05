"""Load a saved model for the notebook's inference timing measurements."""

import json
import pickle
import types
from pathlib import Path, PurePosixPath, PureWindowsPath

import joblib
import torch

from .registry import CONFIG_CLASSES, MODEL_CLASSES, check_device
from .utils import set_seed, sha256


class _PortablePaths:
    def find_class(self, module, name):
        if module == "pathlib" and name in {"PosixPath", "WindowsPath"}:
            return PurePosixPath if name == "PosixPath" else PureWindowsPath
        return super().find_class(module, name)


class _CheckpointUnpickler(_PortablePaths, pickle.Unpickler):
    pass


def _load_torch_checkpoint(artifact):
    # Kaggle serializes config.output_dir as PosixPath; Windows cannot instantiate it.
    portable_pickle = types.ModuleType("portable_checkpoint_pickle")
    portable_pickle.__dict__.update({key: getattr(pickle, key) for key in dir(pickle)
                                    if not key.startswith("__")})
    portable_pickle.Unpickler = _CheckpointUnpickler
    return torch.load(artifact, map_location="cpu", weights_only=False, pickle_module=portable_pickle)


def _load_joblib_checkpoint(artifact):
    try:
        return joblib.load(artifact)
    except NotImplementedError:
        # Native benchmark artifacts use joblib.dump's uncompressed default.
        from joblib.numpy_pickle import NumpyUnpickler
        class PortableJoblibUnpickler(_PortablePaths, NumpyUnpickler):
            pass
        with Path(artifact).open("rb") as stream:
            return PortableJoblibUnpickler(str(artifact), stream, ensure_native_byte_order=True).load()


def restore_run(entry, device="cpu", batch_size=32):
    device = check_device(device)
    artifact = Path(entry["artifact"])
    if entry.get("sha256") and sha256(artifact) != entry["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {artifact}")
    name = entry["model"]
    # These are locally produced/trusted checkpoints, including dataclass configs.
    if name == "locf":
        saved = {"config": json.loads(artifact.read_text(encoding="utf-8"))}
    elif name == "xgboost":
        saved = _load_joblib_checkpoint(artifact)
    else:
        saved = _load_torch_checkpoint(artifact)
    config = {**saved["config"], "device": device, "batch_size": batch_size,
              "output_dir": str(artifact.parent)}
    if config["seed"] != int(entry["seed"]):
        raise ValueError("Manifest seed differs from checkpoint config")
    set_seed(int(entry["seed"]))
    model = MODEL_CLASSES[name](CONFIG_CLASSES[name](**config))
    if name == "xgboost":
        model.backend.models = saved["models"]
        for regressor in model.backend.models:
            regressor.set_params(device=device)
    elif name != "locf":
        model.backend.network.load_state_dict(saved["state_dict"], strict=True)
    model._is_fitted = True
    return model
