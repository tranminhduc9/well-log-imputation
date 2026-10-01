"""Checkpoint-only test inference and portable point-level visualization results."""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4
import gc
import json
import pickle
import types
import shutil
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd
import torch

from src.data.metrics import compute_imputation_metrics
from src.experiments.geolink import (
    CONFIG_CLASSES, MODEL_CLASSES, load_and_check_data, save_json, sha256,
    evaluate_details, set_seed,
)


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


def discover_runs(result_dir):
    """Read a native MSAITSStudy or full benchmark manifest; never rank by test."""
    root = Path(result_dir).resolve()
    rows = []
    if (root / "study.json").is_file():
        manifest = json.loads((root / "study.json").read_text(encoding="utf-8"))
        winners = {stage.get("winner") for stage in manifest["stages"].values()}
        for trial, entry in manifest["trials"].items():
            for seed, run in entry["runs"].items():
                if run.get("status") != "complete":
                    continue
                artifact = root / trial / f"seed_{seed}" / "best.pt"
                rows.append(dict(trial=trial, model="m_saits", seed=int(seed),
                    artifact=str(artifact), sha256=run["hashes"]["best.pt"],
                    validation_winner=trial in winners))
    elif (root / "experiment_results.json").is_file():
        manifest = json.loads((root / "experiment_results.json").read_text(encoding="utf-8"))
        for run in manifest.get("runs", []):
            rows.append(dict(trial=run["model"], model=run["model"], seed=run["seed"],
                artifact=str(root / run["artifact"]), sha256=run["sha256"],
                validation_winner=False))
    else:
        raise FileNotFoundError(f"No study.json or experiment_results.json in {root}")
    if not rows:
        raise ValueError("No completed checkpoint runs in this manifest.")
    frame = pd.DataFrame(rows)
    frame["available"] = frame.artifact.map(lambda p: Path(p).is_file())
    return frame, manifest


def restore_run(entry, device="auto", batch_size=32):
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if device == "auto":
        device = "cpu"
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


def point_table(actual, prediction, mask, metadata, preprocessing, depth=None, *, masked_only=True):
    """Long table in original units, with stable segment/point/log coordinates."""
    selected = mask if masked_only else np.ones_like(mask, dtype=bool)
    segment, point, feature = np.nonzero(selected)
    logs = preprocessing["log_columns"]
    means = np.array([preprocessing["normalization"][log]["mean"] for log in logs])
    stds = np.array([preprocessing["normalization"][log]["std"] for log in logs])
    truth = actual[segment, point, feature].astype(np.float64)
    estimate = prediction[segment, point, feature].astype(np.float64)
    frame = pd.DataFrame(dict(segment_id=segment, point_index=point,
        well=metadata["WELL"].to_numpy()[segment], log=np.array(logs)[feature],
        is_missing=mask[segment, point, feature], actual_normalized=truth,
        predict_normalized=estimate, actual=truth * stds[feature] + means[feature],
        predict=estimate * stds[feature] + means[feature]))
    if depth is not None:
        frame["depth"] = depth[segment, point]
    frame["error"] = frame["predict"] - frame["actual"]
    frame["abs_error"] = frame["error"].abs()
    return frame


def _segment_metrics(actual, prediction, mask, metadata, preprocessing):
    rows = []
    for segment in range(len(actual)):
        feature = np.flatnonzero(mask[segment].any(axis=0))[0]
        log = preprocessing["log_columns"][feature]
        scale = preprocessing["normalization"][log]
        truth, estimate, scored = actual[segment, :, feature], prediction[segment, :, feature], mask[segment, :, feature]
        row = dict(segment_id=segment, well=str(metadata.iloc[segment]["WELL"]), log=log)
        row.update({f"{k}_normalized": v for k, v in compute_imputation_metrics(
            truth, estimate, scored, include_mape=False).items() if k != "count"})
        row.update(compute_imputation_metrics(truth.astype(np.float64) * scale["std"] + scale["mean"],
            estimate.astype(np.float64) * scale["std"] + scale["mean"], scored))
        rows.append(row)
    return rows


def _write_reports(output, metrics):
    frame = pd.DataFrame(metrics)
    keys = ["trial", "model", "split", "scenario", "level", "group", "units"]
    stats = frame.groupby(keys, sort=False)[["mae", "mse", "rmse", "r2", "mape"]].agg(["mean", "std"])
    stats.columns = ["_".join(column) for column in stats.columns]
    stats = stats.join(frame.groupby(keys).agg(n_runs=("seed", "nunique"), count=("count", "first")))
    stats.reset_index().to_csv(output / "summary_detailed.csv", index=False)
    stats.reset_index().query("level == 'overall'").to_csv(output / "summary.csv", index=False)
    rank_wells(frame).to_csv(output / "well_ranking.csv", index=False)


def run_test_inference(runs, result_dir, data_root, output_base, *, device="auto", batch_size=32, workers=1):
    """One forward pass per run/scenario; retain full arrays and masked-point CSVs."""
    runs = pd.DataFrame(runs).copy()
    if runs.empty or runs.duplicated(["trial", "model", "seed"]).any():
        raise ValueError("Select nonempty unique trial/model/seed runs")
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers > 1 and len(runs) > 1:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
        output = Path(output_base).resolve() / stamp
        output.mkdir(parents=True, exist_ok=False)
        parent_manifest = dict(status="running", selected_runs=runs.to_dict("records"), workers=workers)
        save_json(output / "inference_manifest.json", parent_manifest)
        try:
            # Each worker owns its model, output files and metrics. Inference is eval-only.
            def execute(item):
                index, entry = item
                return run_test_inference([entry], result_dir, data_root, output / "runs" / f"run_{index:02d}",
                                          device=device, batch_size=batch_size, workers=1)
            with ThreadPoolExecutor(max_workers=min(workers, len(runs))) as pool:
                children = list(pool.map(execute, enumerate(runs.to_dict("records"))))
            metrics, segments, bundles = [], [], []
            first_manifest = json.loads((children[0] / "inference_manifest.json").read_text(encoding="utf-8"))
            for child in children:
                child_manifest = json.loads((child / "inference_manifest.json").read_text(encoding="utf-8"))
                if child_manifest["data_sha256"] != first_manifest["data_sha256"]:
                    raise ValueError("Parallel runs used different data")
                metrics.append(pd.read_csv(child / "metrics_by_seed.csv", dtype={"group": str, "well": str}))
                segments.append(pd.read_csv(child / "segment_metrics.csv", dtype={"well": str}))
                bundles.extend({**b, "path": str((child / b["path"]).relative_to(output))}
                               for b in child_manifest["bundles"])
            for filename in ("preprocessing_parameters.json", "test_metadata.csv"):
                shutil.copy2(children[0] / filename, output / filename)
            combined = pd.concat(metrics, ignore_index=True)
            combined.to_csv(output / "metrics_by_seed.csv", index=False)
            pd.concat(segments, ignore_index=True).to_csv(output / "segment_metrics.csv", index=False)
            _write_reports(output, combined)
            parent_manifest = {**first_manifest, **parent_manifest, "status": "complete", "bundles": bundles}
            save_json(output / "inference_manifest.json", parent_manifest)
        except Exception as error:
            parent_manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
            save_json(output / "inference_manifest.json", parent_manifest)
            raise
        return output
    preprocessing, data, metadata, fingerprints = load_and_check_data(data_root, True)
    root = Path(result_dir).resolve()
    saved_parameters = json.loads((root / "preprocessing_parameters.json").read_text(encoding="utf-8"))
    for key in ("segment_length", "log_columns", "normalization", "missing_scenarios"):
        if saved_parameters[key] != preprocessing[key]:
            raise ValueError(f"Training/test preprocessing differs: {key}")
    manifest_file = root / ("study.json" if (root / "study.json").exists() else "experiment_results.json")
    source_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected = source_manifest.get("protocol", {}).get("data_sha256", source_manifest.get("data_sha256", {}))
    expected = {**expected, **source_manifest.get("test_data_sha256", {})}
    # Fingerprint keys may originate on Linux or Windows.
    current = {k.replace("\\", "/"): v for k, v in fingerprints.items()}
    for key, value in expected.items():
        if current.get(key.replace("\\", "/")) != value:
            raise ValueError(f"Data changed since training: {key}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    output = Path(output_base).resolve() / stamp
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "preprocessing_parameters.json", preprocessing)
    metadata["test"].to_csv(output / "test_metadata.csv", index=False)
    manifest = dict(status="running", source_results=str(root), data_root=str(Path(data_root).resolve()),
        data_sha256=fingerprints, selected_runs=runs.to_dict("records"), bundles=[],
        metric_policy="Artificially hidden points only. Overall/well normalized; per-log also original units.",
        ranking="Mean of four normalized per-well scenario RMSEs within seed, then mean across seeds.",
        depth_policy="Depth linearly interpolated from segment endpoints; segments are never stitched.",
        test_role="Development test split. Representative selection is descriptive, not model selection.")
    save_json(output / "inference_manifest.json", manifest)
    metrics, segments = [], []
    try:
        for run_index, entry in enumerate(runs.to_dict("records")):
            model = restore_run(entry, device, batch_size)
            if (model.config.seq_len, model.config.n_features) != (
                preprocessing["segment_length"], len(preprocessing["log_columns"])):
                raise ValueError("Checkpoint dimensions differ from test data")
            folder = output / f"run_{run_index:02d}" / f"seed_{entry['seed']}"
            folder.mkdir(parents=True)
            for scenario, dataset in data["test"].items():
                print(f"TEST {entry['trial']} / seed {entry['seed']} / {scenario}", flush=True)
                model_input = {"X": dataset["X"]}
                if "depth" in dataset:
                    model_input["depth"] = dataset["depth"]
                prediction = model.impute(model_input)
                actual, mask = dataset["X_intact"], dataset["indicating_mask"]
                if prediction.shape != actual.shape or not np.isfinite(prediction).all():
                    raise ValueError("Invalid prediction shape or non-finite values")
                if not np.allclose(prediction[~mask], actual[~mask], rtol=1e-6, atol=1e-7):
                    raise ValueError("Model changed observed values")
                identity = dict(trial=entry["trial"], model=entry["model"], seed=int(entry["seed"]), scenario=scenario)
                slug = scenario.lower().replace("-", "_")
                bundle_path = folder / f"{slug}.npz"
                arrays = dict(actual=actual, predict=prediction, mask=mask,
                    segment_id=np.arange(len(actual)), well=metadata["test"]["WELL"].to_numpy(dtype=str),
                    log_columns=np.array(preprocessing["log_columns"]))
                if "depth" in dataset:
                    arrays["depth"] = dataset["depth"]
                np.savez_compressed(bundle_path, **arrays)
                table = point_table(actual, prediction, mask, metadata["test"], preprocessing, dataset.get("depth"))
                for key, value in identity.items():
                    table[key] = value
                table.to_csv(folder / f"{slug}_points.csv.gz", index=False)
                # Score cached predictions, without a second forward pass.
                class CachedPrediction:
                    def impute(self, _):
                        return prediction
                details = evaluate_details(CachedPrediction(), {scenario: dataset}, metadata["test"],
                    preprocessing, entry["model"], entry["seed"], "test", include_well_log=True)
                metrics.extend({**row, "trial": entry["trial"]} for row in details)
                segments.extend({**row, **identity} for row in _segment_metrics(
                    actual, prediction, mask, metadata["test"], preprocessing))
                manifest["bundles"].append({**identity, "path": str(bundle_path.relative_to(output)),
                    "sha256": sha256(bundle_path)})
                pd.DataFrame(metrics).to_csv(output / "metrics_by_seed.csv", index=False)
                pd.DataFrame(segments).to_csv(output / "segment_metrics.csv", index=False)
                save_json(output / "inference_manifest.json", manifest)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _write_reports(output, metrics)
        manifest["status"] = "complete"
        save_json(output / "inference_manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "inference_manifest.json", manifest)
        raise
    return output


def rank_wells(metrics, scenario=None):
    frame = metrics[(metrics.level == "well") & (metrics.units == "normalized")].copy()
    if scenario is not None:
        frame = frame[frame.scenario == scenario]
    keys = ["trial", "model", "seed", "group"]
    per_seed = frame.groupby(keys).agg(score=("rmse", "mean"), mae=("mae", "mean"),
        n_scenarios=("scenario", "nunique"), count=("count", "sum")).reset_index()
    expected_count = 4 if scenario is None else 1
    if per_seed.empty or not per_seed.n_scenarios.eq(expected_count).all():
        raise ValueError("Well ranking requires every selected scenario for every well/seed")
    result = per_seed.groupby(["trial", "model", "group"]).agg(
        score_mean=("score", "mean"), score_std=("score", "std"), mae_mean=("mae", "mean"),
        n_seeds=("seed", "nunique"), count_per_seed=("count", "first")).reset_index().rename(columns={"group": "well"})
    result = result.sort_values(["trial", "score_mean", "well"])
    result["rank"] = result.groupby(["trial", "model"]).cumcount() + 1
    return result


def representative_wells(ranking, trial, n_extremes=2):
    """Distinct extremes plus wells near the 25th, 50th and 75th rank percentiles."""
    frame = ranking[ranking.trial == trial].sort_values(["score_mean", "well"]).reset_index(drop=True)
    if frame.empty or n_extremes < 1:
        raise ValueError("Choose an existing trial and positive n_extremes")
    candidates = [(i, "best") for i in range(min(n_extremes, len(frame)))]
    candidates += [(i, "worst") for i in range(len(frame)-1, max(-1, len(frame)-n_extremes-1), -1)]
    candidates += [(int(round(q * (len(frame)-1))), label) for q, label in
                   [(0.25, "q25"), (0.5, "median"), (0.75, "q75")]]
    output, seen = [], set()
    for index, category in candidates:
        if index not in seen:
            output.append({**frame.iloc[index].to_dict(), "category": category})
            seen.add(index)
    return pd.DataFrame(output)


def load_bundle(output_dir, trial, seed, scenario):
    root = Path(output_dir)
    manifest = json.loads((root / "inference_manifest.json").read_text(encoding="utf-8"))
    entry = next((b for b in manifest["bundles"] if
        (b["trial"], b["seed"], b["scenario"]) == (trial, int(seed), scenario)), None)
    if entry is None:
        raise KeyError(f"No predictions for {trial}/{seed}/{scenario}")
    with np.load(root / entry["path"], allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def plot_segment(bundle, preprocessing, segment_id, *, title=""):
    """Original-unit curves and scatter for hidden points of one segment."""
    import matplotlib.pyplot as plt
    segment = int(segment_id)
    mask = bundle["mask"][segment]
    feature = np.flatnonzero(mask.any(axis=0))[0]
    log = str(bundle["log_columns"][feature])
    scale = preprocessing["normalization"][log]
    actual = bundle["actual"][segment, :, feature].astype(np.float64) * scale["std"] + scale["mean"]
    predicted = bundle["predict"][segment, :, feature].astype(np.float64) * scale["std"] + scale["mean"]
    hidden = mask[:, feature]
    x = bundle["depth"][segment] if "depth" in bundle else np.arange(len(actual))
    metrics = compute_imputation_metrics(actual, predicted, hidden)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].plot(x, actual, color="black", lw=1.5, label="Actual (full context)")
    axes[0].plot(x, np.where(hidden, predicted, np.nan), color="tab:red", lw=1.5,
                 marker=".", markersize=3, label="Predict (hidden points)")
    axes[0].scatter(x[~hidden], actual[~hidden], s=6, color="tab:blue", alpha=.4, label="Observed input")
    indices = np.flatnonzero(hidden)
    spacing = (x[1] - x[0]) / 2 if len(x) > 1 else .5
    axes[0].axvspan(x[indices[0]] - spacing, x[indices[-1]] + spacing, color="orange", alpha=.15, label="Artificial gap")
    axes[0].set(xlabel="Depth (interpolated from endpoints)" if "depth" in bundle else "Point index",
                ylabel=f"{log} (original units)", title=f"{bundle['well'][segment]} / segment {segment} / {log}")
    axes[0].legend(fontsize=8)
    axes[1].scatter(actual[hidden], predicted[hidden], s=12, alpha=.7)
    limits = [min(actual[hidden].min(), predicted[hidden].min()), max(actual[hidden].max(), predicted[hidden].max())]
    if limits[0] == limits[1]:
        limits = [limits[0] - .5, limits[1] + .5]
    axes[1].plot(limits, limits, "k--", lw=1)
    axes[1].set(xlabel="Actual", ylabel="Predict", title=f"Hidden only (n={metrics['count']})\nRMSE={metrics['rmse']:.4g}; MAE={metrics['mae']:.4g}; R²={metrics['r2']:.3g}")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, metrics


def plot_comparison(bundles, preprocessing, segment_id, *, title=""):
    """Overlay model predictions on exactly the same segment and artificial gap."""
    import matplotlib.pyplot as plt
    first = next(iter(bundles.values()))
    segment = int(segment_id)
    feature = np.flatnonzero(first["mask"][segment].any(axis=0))[0]
    log = str(first["log_columns"][feature])
    scale = preprocessing["normalization"][log]
    actual = first["actual"][segment, :, feature].astype(np.float64) * scale["std"] + scale["mean"]
    hidden = first["mask"][segment, :, feature]
    x = first["depth"][segment] if "depth" in first else np.arange(len(actual))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].plot(x, actual, color="black", lw=1.5, label="Actual (full context)")
    axes[0].scatter(x[~hidden], actual[~hidden], s=7, color="gray", alpha=.5, label="Observed input")
    indices = np.flatnonzero(hidden)
    spacing = (x[1] - x[0])/2 if len(x) > 1 else .5
    axes[0].axvspan(x[indices[0]]-spacing, x[indices[-1]]+spacing, color="orange", alpha=.15, label="Artificial gap")
    results = []
    limits = [actual[hidden].min(), actual[hidden].max()]
    for label, bundle in bundles.items():
        if (not np.array_equal(first["well"], bundle["well"]) or
            not np.array_equal(first["log_columns"], bundle["log_columns"]) or
            not np.array_equal(first["actual"][segment], bundle["actual"][segment]) or
            not np.array_equal(first["mask"][segment], bundle["mask"][segment])):
            raise ValueError("Comparison requires identical targets, coordinates and masks")
        predicted = bundle["predict"][segment, :, feature].astype(np.float64)*scale["std"]+scale["mean"]
        result = compute_imputation_metrics(actual, predicted, hidden)
        results.append({"label": label, "log": log, **result})
        line, = axes[0].plot(x, np.where(hidden, predicted, np.nan), marker=".", markersize=3,
                            label=f"{label} (hidden)", alpha=.85)
        axes[1].scatter(actual[hidden], predicted[hidden], s=14, alpha=.6, color=line.get_color(),
                        label=f"{label}: RMSE={result['rmse']:.4g}, MAE={result['mae']:.4g}")
        limits = [min(limits[0], predicted[hidden].min()), max(limits[1], predicted[hidden].max())]
    if limits[0] == limits[1]:
        limits = [limits[0]-.5, limits[1]+.5]
    axes[1].plot(limits, limits, "k--", lw=1)
    axes[0].set(xlabel="Depth (interpolated from endpoints)" if "depth" in first else "Point index",
                ylabel=f"{log} (original units)", title=f"{first['well'][segment]} / segment {segment} / {log}")
    axes[1].set(xlabel="Actual", ylabel="Predict", title=f"Hidden only (n={int(hidden.sum())})")
    for axis in axes:
        axis.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, pd.DataFrame(results)
