"""Reproducible GeoLink training and evaluation used by geo_link.ipynb.

Validation selects checkpoints. Test evaluation is a separate final phase and
never selects a model/seed. The existing test split has already informed model
development; reports explicitly record that it is not an untouched holdout.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import torch

from src.data.metrics import compute_imputation_metrics
from src.models.brits import BRITS, BRITSConfig
from src.models.locf import LOCF
from src.models.model import ModelConfig
from src.models.saits import SAITS, SAITSConfig
from src.models.m_saits import MSAITS, MSAITSConfig
from src.models.xgboost import XGBoost, XGBoostConfig
from src.preprocessing.pipeline import MISSING_SCENARIOS, BLOCK_LENGTHS


LOGGER = logging.getLogger(__name__)
MODEL_CLASSES = {"locf": LOCF, "xgboost": XGBoost, "brits": BRITS,
                 "saits": SAITS, "m_saits": MSAITS}
CONFIG_CLASSES = {"locf": ModelConfig, "xgboost": XGBoostConfig,
                  "brits": BRITSConfig, "saits": SAITSConfig, "m_saits": MSAITSConfig}
LABELS = {"locf": "LOCF", "xgboost": "XGBoost", "brits": "BRITS + MIT",
          "saits": "SAITS (segment MIT)", "m_saits": "M-SAITS (segment MIT)"}
METRICS = ("mae", "mse", "rmse", "r2", "mape")


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


def load_and_check_data(data_root, evaluate_test):
    """Fail before training on split overlap, invalid masks or bad scaling."""
    root = Path(data_root)
    parameter_path = root / "preprocessing_parameters.json"
    preprocessing = json.loads(parameter_path.read_text(encoding="utf-8"))
    steps = preprocessing["segment_length"]
    logs = preprocessing["log_columns"]
    scenarios = preprocessing["missing_scenarios"]
    if set(scenarios) != set(MISSING_SCENARIOS) or len(scenarios) != 4:
        raise ValueError("GeoLink requires the four supported missingness scenarios.")
    if steps < max(BLOCK_LENGTHS.values()) or len(logs) != len(set(logs)) or not logs:
        raise ValueError("Invalid GeoLink sequence length or log names.")
    for log in logs:
        scaling = preprocessing["normalization"][log]
        if not np.isfinite([scaling["mean"], scaling["std"]]).all() or scaling["std"] <= 0:
            raise ValueError(f"Invalid normalization for {log}.")
    splits = ("train", "val", "test") if evaluate_test else ("train", "val")
    metadata, datasets, files = {}, {}, [parameter_path]
    for split in splits:
        value_path, meta_path = root / split / "segments.npy", root / split / "metadata.csv"
        values = np.load(value_path, allow_pickle=False)
        meta = pd.read_csv(meta_path, dtype={"WELL": str})
        if values.ndim != 3 or values.shape[1:] != (steps, len(logs)) or len(values) == 0:
            raise ValueError(f"Invalid segment shape for {split}: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"GeoLink intact {split} targets contain NaN/Inf.")
        if len(meta) != len(values) or "WELL" not in meta or meta["WELL"].isna().any():
            raise ValueError(f"Missing or misaligned WELL metadata in {split}.")
        if meta["WELL"].str.strip().eq("").any():
            raise ValueError(f"Empty well names in {split}.")
        for other, previous in metadata.items():
            overlap = set(meta["WELL"]) & set(previous["WELL"])
            if overlap:
                raise ValueError(f"Well leakage between {other} and {split}: {sorted(overlap)}")
        metadata[split] = meta
        files.extend([value_path, meta_path])
        if split == "train":
            datasets[split] = {"X": values}
            continue
        datasets[split] = {}
        for scenario in scenarios:
            mask_path = root / split / f"mask_{scenario.lower().replace('-', '_')}.npy"
            mask = np.load(mask_path, allow_pickle=False)
            if mask.shape != values.shape or not np.isin(mask, [0, 1]).all():
                raise ValueError(f"Invalid mask for {split}/{scenario}.")
            mask = mask.astype(bool)
            length = BLOCK_LENGTHS.get(scenario, 1 if scenario == "Single" else steps)
            for segment in mask:
                columns = np.flatnonzero(segment.any(axis=0))
                if len(columns) != 1:
                    raise ValueError(f"Mask must hide exactly one log: {split}/{scenario}.")
                indices = np.flatnonzero(segment[:, columns[0]])
                if len(indices) != length or (len(indices) > 1 and not (np.diff(indices) == 1).all()):
                    raise ValueError(f"Incorrect gap topology: {split}/{scenario}.")
            masked = values.copy()
            masked[mask] = np.nan
            datasets[split][scenario] = {
                "X": masked, "X_intact": values, "indicating_mask": mask,
            }
            files.append(mask_path)
    fingerprints = {str(path.relative_to(root)): sha256(path) for path in files}
    return preprocessing, datasets, metadata, fingerprints


def evaluate_details(model, datasets, metadata, preprocessing, model_name, seed, split):
    """Evaluate one prediction pass per scenario, including per-log/per-well rows.

    Pooled/within-well errors are normalized. Per-log errors are additionally
    reported in original units; MAPE exists only on those physical-unit rows.
    """
    rows = []
    for scenario, dataset in datasets.items():
        truth, mask = dataset["X_intact"], dataset["indicating_mask"]
        # Do not expose targets to the prediction backend.
        prediction = model.impute({"X": dataset["X"]})
        if not np.isfinite(prediction).all():
            raise ValueError(f"{model_name}/{seed}/{split}/{scenario}: non-finite prediction.")
        if not np.allclose(prediction[~mask], dataset["X"][~mask], rtol=1e-6, atol=1e-7):
            raise ValueError(f"{model_name} changed observed values.")

        def record(target, estimate, scored, level, group, units, mape=False):
            if not scored.any():
                return
            metrics = compute_imputation_metrics(target, estimate, scored, include_mape=mape)
            if metrics["count"] != int(scored.sum()):
                raise ValueError("Evaluation count does not match the fixed mask.")
            rows.append({"model": model_name, "label": LABELS[model_name], "seed": seed,
                         "split": split, "scenario": scenario, "level": level,
                         "group": str(group), "units": units, **metrics})

        record(truth, prediction, mask, "overall", "all", "normalized")
        for index, log in enumerate(preprocessing["log_columns"]):
            record(truth[..., index], prediction[..., index], mask[..., index],
                   "log", log, "normalized")
            scale = preprocessing["normalization"][log]
            record(truth[..., index] * scale["std"] + scale["mean"],
                   prediction[..., index] * scale["std"] + scale["mean"],
                   mask[..., index], "log", log, "original", mape=True)
        for well in metadata["WELL"].unique():
            selected = (metadata["WELL"] == well).to_numpy()
            record(truth[selected], prediction[selected], mask[selected], "well", well, "normalized")
    return rows


def summarize(rows):
    """Aggregate seeds, not observations; one run has undefined sample std."""
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    keys = ["model", "label", "split", "scenario", "level", "group", "units"]
    output = []
    for key, group in frame.groupby(keys, dropna=False, sort=False):
        if group["seed"].duplicated().any() or group["count"].nunique() != 1:
            raise ValueError("Duplicate seed results or inconsistent evaluation counts.")
        row = dict(zip(keys, key))
        row.update(n_runs=len(group), count=int(group["count"].iloc[0]))
        for metric in METRICS:
            if metric not in group or group[metric].isna().all():
                continue
            values = group[metric]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
        output.append(row)
    return pd.DataFrame(output)


def _git(project_root, *args):
    try:
        return subprocess.check_output(["git", "-C", str(project_root), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture_provenance(project_root, output_dir):
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "xgboost", "joblib", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    files = list((project_root / "src").rglob("*.py"))
    files += [project_root / "notebook" / "geo_link.ipynb", project_root / "requirements.txt"]
    hashes = {}
    for source in files:
        if source.is_file():
            relative = source.relative_to(project_root)
            target = output_dir / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(relative)] = sha256(source)
    return {"git_commit": _git(project_root, "rev-parse", "HEAD"),
            "git_status": _git(project_root, "status", "--short"),
            "source_sha256": hashes, "packages": packages, "python": platform.python_version(),
            "platform": platform.platform(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
            "determinism_scope": "Same code, data, library versions and hardware; cross-platform equality is not guaranteed."}


def make_config(name, common, settings, seed, args, artifact_dir):
    parameters = {**common, **settings.get(name, {}), "seed": seed, "output_dir": artifact_dir}
    if name == "xgboost":
        parameters.update(n_estimators=args.xgb_estimators,
                          device="cuda" if torch.cuda.is_available() else "cpu")
    elif name != "locf":
        parameters.update(epochs=getattr(args, f"{name}_epochs"),
                          patience=getattr(args, f"{name}_patience"),
                          device="gpu" if torch.cuda.is_available() else "cpu")
    return CONFIG_CLASSES[name](**parameters)


def _restore(name, artifact, config):
    model = MODEL_CLASSES[name](CONFIG_CLASSES[name](**config))
    if name == "xgboost":
        model.backend.models = joblib.load(artifact)["models"]
    elif name != "locf":
        # Only locally produced artifacts from this run are loaded here.
        saved = torch.load(artifact, map_location=model.backend.device, weights_only=False)
        model.backend.network.load_state_dict(saved["state_dict"])
    model._is_fitted = True
    return model


def run_experiment(args, project_root, settings):
    """Train every requested seed, freeze artifacts, then evaluate test once."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    project_root = Path(project_root).resolve()
    names, seeds = tuple(args.models), tuple(args.seeds)
    if not names or len(set(names)) != len(names) or set(names) - MODEL_CLASSES.keys():
        raise ValueError("Select unique supported model names.")
    if not seeds or len(set(seeds)) != len(seeds) or any(
        type(s) is not int or not 0 <= s < 2**32 for s in seeds
    ):
        raise ValueError("Seeds must be unique integers in [0, 2**32).")
    kaggle = Path("/kaggle/working").exists()
    data_root = Path(args.data_root) if args.data_root else (
        Path("/kaggle/input/datasets/minhduc0912/geolink-dataset") if kaggle
        else project_root / "data" / "processed")
    base = Path(args.output_dir) if args.output_dir else (
        Path("/kaggle/working/well-log-results-final") if kaggle else project_root / "results" / "final_training")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    output_dir = base.resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output_dir / "training.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    manifest = {"run_id": run_id, "status": "validating_data", "seeds": seeds,
                "data_root": str(data_root.resolve()), "models": names, "runs": [],
                "evaluate_test": args.evaluate_test,
                "test_role": "development: this split has already informed model changes",
                "selection": "Fixed configs; neural checkpoints selected by mean validation RMSE; no seed selection.",
                "std_definition": "sample standard deviation across training seeds (ddof=1); null for one run",
                "mit_reduction": "per-segment MAE, then batch mean; uniform scenario sampling",
                "metric_units": "normalized except per-log original-unit rows; MAPE only in original units",
                "mape_zero_policy": "Exclude exactly zero physical targets from MAPE only",
                "holdout_note": "A new untouched well holdout is required for a final unbiased claim."}
    rows, history = [], []

    def persist():
        save_json(output_dir / "experiment_results.json", {**manifest, "metrics": rows})
        pd.DataFrame(rows).to_csv(output_dir / "metrics_by_seed.csv", index=False)
        pd.DataFrame(history, columns=None if history else ["model", "seed", "epoch", "loss"]
                     ).to_csv(output_dir / "training_history.csv", index=False)
        summary = summarize(rows)
        summary.to_csv(output_dir / "summary_detailed.csv", index=False)
        if not summary.empty:
            summary.loc[summary["level"] == "overall"].to_csv(output_dir / "summary.csv", index=False)

    try:
        preprocessing, data, metadata, hashes = load_and_check_data(data_root, args.evaluate_test)
        manifest["data_sha256"] = hashes
        manifest["provenance"] = capture_provenance(project_root, output_dir)
        manifest["split_counts"] = {s: {"segments": len(m), "wells": m["WELL"].nunique()}
                                    for s, m in metadata.items()}
        save_json(output_dir / "preprocessing_parameters.json", preprocessing)
        common = {"seq_len": preprocessing["segment_length"], "n_features": len(preprocessing["log_columns"])}
        monitor = {**next(iter(data["val"].values())), "validation_scenarios": data["val"]}
        # Validate every configuration before starting a potentially long run.
        for name in names:
            make_config(name, common, settings, seeds[0], args, output_dir)
        manifest["status"] = "training"
        persist()
        for name in names:
            for seed in (seeds[:1] if name == "locf" else seeds):
                set_seed(seed)
                artifact_dir = output_dir / name / f"seed_{seed}"
                artifact_dir.mkdir(parents=True)
                config = make_config(name, common, settings, seed, args, artifact_dir)
                LOGGER.info("%s seed=%d | %s", LABELS[name], seed, asdict(config))
                model = MODEL_CLASSES[name](config)
                if name != "locf":
                    validation = data["val"]["Block-20"] if name == "xgboost" else monitor
                    model.fit(data["train"], validation)
                entry = {"model": name, "label": LABELS[name], "seed": seed,
                         "config": asdict(config), "best_epoch": getattr(model.backend, "best_epoch", None)}
                if name == "xgboost":
                    artifact = artifact_dir / "model.joblib"
                    joblib.dump({"config": asdict(config), "models": model.backend.models}, artifact)
                    for point in model.backend.training_history:
                        for iteration, loss in enumerate(point["train_rmse"], 1):
                            curve = point["validation_rmse"]
                            history.append({"model": name, "seed": seed, "log_index": point["log_index"],
                                            "iteration": iteration, "train_rmse": loss,
                                            "validation_rmse": curve[iteration-1] if len(curve) >= iteration else None})
                elif name != "locf":
                    artifact = artifact_dir / "model.pt"
                    torch.save({"config": asdict(config), "state_dict": model.backend.network.state_dict(),
                                "best_epoch": model.backend.best_epoch,
                                "training_history": model.backend.training_history}, artifact)
                    history.extend({"model": name, "seed": seed, **point} for point in model.backend.training_history)
                else:
                    artifact = artifact_dir / "config.json"
                    save_json(artifact, asdict(config))
                entry.update(artifact=str(artifact.relative_to(output_dir)), sha256=sha256(artifact))
                manifest["runs"].append(entry)
                persist()  # Artifact/history survive even if scoring fails.
                rows.extend(evaluate_details(model, data["val"], metadata["val"], preprocessing, name, seed, "val"))
                persist()
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if args.evaluate_test:
            manifest["status"] = "test_evaluation"
            persist()
            for entry in manifest["runs"]:
                model = _restore(entry["model"], output_dir / entry["artifact"], entry["config"])
                rows.extend(evaluate_details(model, data["test"], metadata["test"], preprocessing,
                                             entry["model"], entry["seed"], "test"))
                persist()
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        manifest["status"] = "complete"
        persist()
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output_dir / "experiment_results.json", {**manifest, "metrics": rows})
        LOGGER.exception("Run failed; partial artifacts preserved at %s", output_dir)
        raise
    finally:
        root_logger.removeHandler(handler)
        handler.close()
        root_logger.setLevel(previous_level)
    shutil.make_archive(str(output_dir), "zip", output_dir)
    return output_dir
