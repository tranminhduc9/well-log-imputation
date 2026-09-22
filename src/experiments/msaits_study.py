"""Validation-only staged tuning and paired component ablations for M-SAITS.

Do not tune on test. Resumption requires the same data, source and environment.
Only load checkpoints from a trusted study directory (torch checkpoints pickle).
"""
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import gc
import json
import logging
import math
import re
import shutil
import time
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from src.models.m_saits import MSAITS, MSAITSConfig
from src.experiments.geolink import (
    capture_provenance, evaluate_details, load_and_check_data, save_json,
    set_seed, sha256, summarize,
)


BASE_SETTINGS = dict(
    d_model=256, d_inner=128, n_layers=2, n_heads=4, batch_size=32,
    learning_rate=0.0005, encoder_channels=16, kernel_size=51, conv_expansion=2,
    masking_strategy="mixed", masking_rate=0.2, mit_reduction="segment",
    ort_weight=1.0, mit_weight=1.0, min_delta=1e-4,
    # Same shuffle for the same seed/epoch, independent of architecture width.
    independent_shuffle=True,
)
COMPONENTS = {
    "residual": {"temporal_residual": True},
    "layernorm": {"encoder_norm": "layer"},
    "gap_gate": {"gap_aware_gate": True},
    "per_log_head": {"decoder": "per_log", "decoder_width": 32},
    "multi_scale_conv": {
        "multi_scale_conv": True,
        "multi_scale_kernels": (3, 7, 15),
    },
    "depth_encoding": {"depth_encoding": True},
}


def grid_candidates(reference, grid):
    """Full Cartesian product within ONE stage, not across all stages."""
    if any(not values for values in grid.values()):
        raise ValueError("Grid axes cannot be empty.")
    return [(f"grid_{index:02d}", {**reference, **dict(zip(grid, values))})
            for index, values in enumerate(product(*grid.values()))]


def component_candidates(reference, components):
    """Addition to reference and leave-one-out from full; never retune per arm."""
    if len(set(components)) != len(components) or set(components) - COMPONENTS.keys():
        raise ValueError("Select unique supported components.")
    candidates = [("reference", dict(reference))]
    candidates += [(f"add_{name}", {**reference, **COMPONENTS[name]}) for name in components]
    if len(components) >= 2:
        full = dict(reference)
        for name in components:
            full.update(COMPONENTS[name])
        candidates.append(("full", full))
        for omitted in components:
            values = dict(reference)
            for name in components:
                if name != omitted:
                    values.update(COMPONENTS[name])
            candidates.append((f"without_{omitted}", values))
    return candidates


def compact_component_candidates(reference, components):
    """Baseline, one-component additions, and one combined model.

    Unlike the full factorial-style ablation, this avoids duplicate leave-one-out
    arms. With two components it produces four configurations, which keeps the
    two-seed Kaggle run comfortably below the session limit.
    """
    if not components or len(set(components)) != len(components) or set(components) - COMPONENTS.keys():
        raise ValueError("Select unique supported components.")
    candidates = [("reference", dict(reference))]
    candidates += [(f"add_{name}", {**reference, **COMPONENTS[name]}) for name in components]
    if len(components) > 1:
        combined = dict(reference)
        for name in components:
            combined.update(COMPONENTS[name])
        candidates.append(("full", combined))
    return candidates


def score_seeds(rows):
    frame = pd.DataFrame(rows)
    frame = frame[(frame["level"] == "overall") & (frame["split"] == "val")]
    if frame.duplicated(["seed", "scenario"]).any():
        raise ValueError("Duplicate seed/scenario scores.")
    if not (frame.groupby("seed").size() == 4).all():
        raise ValueError("Each seed must score all four scenarios.")
    return frame.groupby("seed")["rmse"].mean()


class MSAITSStudy:
    def __init__(self, project_root, data_root, output_base, *, seeds=(129, 219),
                 epochs=500, patience=30, device="auto", resume_dir=None):
        self.project_root = Path(project_root).resolve()
        self.data_root = Path(data_root).resolve()
        if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
            raise ValueError("Seeds must be unique integers in [0, 2**32).")
        if epochs <= 0 or patience <= 0 or device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Invalid training budget or device.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable.")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        set_seed(seeds[0])
        self.preprocessing, self.data, self.metadata, hashes = load_and_check_data(self.data_root, False)
        train_depth = self.data["train"].get("depth")
        if train_depth is None:
            self.depth_mean, self.depth_std = 0.0, 1.0
        else:
            self.depth_mean = float(np.mean(train_depth, dtype=np.float64))
            self.depth_std = float(np.std(train_depth, dtype=np.float64))
            if not math.isfinite(self.depth_std) or self.depth_std <= 0:
                raise ValueError("Training depths need positive finite variation.")
        self.seeds, self.epochs, self.patience = tuple(seeds), epochs, patience
        protocol = dict(seeds=list(seeds), epochs=epochs, patience=patience, device=self.device,
                        data_sha256=hashes, source_sha256={
                            str(p.relative_to(self.project_root)): sha256(p)
                            for p in (self.project_root / "src").rglob("*.py")},
                        torch_version=str(torch.__version__), numpy_version=np.__version__,
                        cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(0) if self.device == "cuda" else None)
        if resume_dir is not None:
            self.output_dir = Path(resume_dir).resolve()
            self.manifest = json.loads((self.output_dir / "study.json").read_text(encoding="utf-8"))
            if self.manifest["protocol"] != protocol:
                raise ValueError("Resume rejected: data, source, seeds, budget or environment changed. Start a new study.")
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
            self.output_dir = Path(output_base).resolve() / stamp
            self.output_dir.mkdir(parents=True, exist_ok=False)
            self.manifest = dict(protocol=protocol, stages={}, trials={},
                selection="Mean of four validation scenario RMSEs per seed; average all requested seeds.",
                test_role="Development split, not an untouched holdout. Never used for study ranking.",
                resume="Latest completed epoch including optimizer, best weights and Torch RNG; same environment required.",
                gap_features="Mask-derived distances use sample indices.",
                depth_encoding=("START_DEPTH/END_DEPTH are linearly expanded per segment "
                                "and normalized using train only."),
                provenance=capture_provenance(self.project_root, self.output_dir))
            save_json(self.output_dir / "preprocessing_parameters.json", self.preprocessing)
            self._save()
        self.monitor = {**next(iter(self.data["val"].values())), "validation_scenarios": self.data["val"]}

    def _save(self):
        save_json(self.output_dir / "study.json", self.manifest)

    def _config(self, settings, seed, folder):
        forbidden = {"seq_len", "n_features", "epochs", "patience", "device", "seed",
                     "output_dir", "optimizer", "depth_mean", "depth_std"}
        if forbidden & settings.keys():
            raise ValueError(f"Use the study protocol for {sorted(forbidden & settings.keys())}")
        if settings.get("depth_encoding") and "depth" not in self.data["train"]:
            raise ValueError("depth_encoding requires START_DEPTH and END_DEPTH metadata.")
        return MSAITSConfig(**settings, seq_len=self.preprocessing["segment_length"],
            n_features=len(self.preprocessing["log_columns"]), epochs=self.epochs,
            patience=self.patience, device=self.device, seed=seed, output_dir=folder,
            depth_mean=self.depth_mean, depth_std=self.depth_std)

    def run_stage(self, stage, candidates):
        if not re.fullmatch(r"[a-z0-9_]+", stage) or not candidates:
            raise ValueError("Stage needs a safe name and at least one candidate.")
        names = [name for name, _ in candidates]
        if len(set(names)) != len(names) or any(not re.fullmatch(r"[a-z0-9_]+", n) for n in names):
            raise ValueError("Candidate names must be safe and unique.")
        specification = [{"name": n, "settings": s} for n, s in candidates]
        if stage in self.manifest["stages"] and self.manifest["stages"][stage]["candidates"] != specification:
            raise ValueError("Stage grid changed. Use a new study directory.")
        for name, settings in candidates:
            self._config(settings, self.seeds[0], self.output_dir / stage / name)
        self.manifest["stages"].setdefault(stage, {"candidates": specification})
        self._save()
        for index, (name, settings) in enumerate(candidates, 1):
            key = f"{stage}/{name}"
            entry = self.manifest["trials"].setdefault(key, dict(stage=stage, variant=name,
                                                              settings=settings, runs={}))
            for seed in self.seeds:
                folder = self.output_dir / stage / name / f"seed_{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                old = entry["runs"].get(str(seed), {})
                if old.get("status") == "complete":
                    for filename, expected in old["hashes"].items():
                        if sha256(folder / filename) != expected:
                            raise ValueError(f"Artifact changed: {folder / filename}")
                    print(f"SKIP completed {key} seed={seed}", flush=True)
                    continue
                print(f"[{stage} {index}/{len(candidates)}] {name}, seed={seed}", flush=True)
                handler = logging.FileHandler(folder / "training.log", encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
                logger = logging.getLogger()
                previous_level = logger.level
                logger.setLevel(logging.INFO)
                logger.addHandler(handler)
                entry["runs"][str(seed)] = dict(status="starting")
                self._save()
                try:
                    set_seed(seed)
                    config = self._config(settings, seed, folder)
                    model = MSAITS(config)
                    checkpoint = folder / "last.pt"
                    resumed = checkpoint.exists()
                    model.backend.training_checkpoint = checkpoint
                    def progress(point):
                        save_json(folder / "progress.json", {
                            "trial": key, "seed": seed, **point,
                            "best_epoch": model.backend.best_epoch,
                        })
                        pd.DataFrame(model.backend.training_history).to_csv(folder / "history.csv", index=False)
                    model.backend.epoch_callback = progress
                    entry["runs"][str(seed)] = dict(status="training", resumed=resumed)
                    self._save()
                    started = time.perf_counter()
                    model.fit(self.data["train"], self.monitor)
                    elapsed = time.perf_counter() - started
                    temporary = folder / "best.tmp"
                    torch.save({"config": asdict(config), "state_dict": model.backend.network.state_dict(),
                                "best_epoch": model.backend.best_epoch}, temporary)
                    temporary.replace(folder / "best.pt")
                    rows = evaluate_details(model, self.data["val"], self.metadata["val"],
                        self.preprocessing, "m_saits", seed, "val", include_well_log=True)
                    save_json(folder / "metrics.json", rows)
                    pd.DataFrame(model.backend.training_history).to_csv(folder / "history.csv", index=False)
                    entry["runs"][str(seed)] = dict(status="complete", resumed=resumed,
                        config=asdict(config), best_epoch=model.backend.best_epoch,
                        epochs_trained=len(model.backend.training_history),
                        fit_seconds_this_attempt=elapsed,
                        trainable_parameters=sum(p.numel() for p in model.backend.network.parameters() if p.requires_grad),
                        hashes={n: sha256(folder / n) for n in ("best.pt", "metrics.json", "history.csv")})
                    del model
                except BaseException as error:
                    entry["runs"][str(seed)].update(status="interrupted", error=f"{type(error).__name__}: {error}")
                    self._save()
                    raise
                finally:
                    logger.removeHandler(handler)
                    logger.setLevel(previous_level)
                    handler.close()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                self._save()
                self.export()
        board = self.export()
        ranked = board[(board.stage == stage) & board.complete].sort_values(["score_mean", "trial"])
        if ranked.empty:
            raise RuntimeError("No complete candidate in this stage.")
        winner = ranked.iloc[0].trial
        self.manifest["stages"][stage]["winner"] = winner
        self._save()
        print(f"Selected on VALIDATION: {winner}, score={ranked.iloc[0].score_mean:.6f}", flush=True)
        return dict(self.manifest["trials"][winner]["settings"])

    def export(self):
        rows, runs, curves, board = [], [], [], []
        for key, trial in self.manifest["trials"].items():
            trial_rows = []
            for seed, run in trial["runs"].items():
                if run["status"] != "complete":
                    continue
                folder = self.output_dir / key / f"seed_{seed}"
                metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                trial_rows.extend(metrics)
                identity = dict(trial=key, stage=trial["stage"], variant=trial["variant"])
                rows.extend({**row, **identity} for row in metrics)
                runs.append({**identity, "seed": int(seed), **{k: v for k, v in run.items() if k not in {"hashes", "config"}},
                             **{f"config_{k}": v for k, v in run["config"].items()}})
                curves.extend({**point, **identity, "seed": int(seed)} for point in
                              pd.read_csv(folder / "history.csv").to_dict("records"))
            if not trial_rows:
                continue
            scores = score_seeds(trial_rows)
            board.append(dict(trial=key, stage=trial["stage"], variant=trial["variant"],
                n_runs=len(scores), complete=set(scores.index) == set(self.seeds),
                score_mean=scores.mean(), score_std=scores.std(ddof=1),
                **{f"score_seed_{seed}": value for seed, value in scores.items()}))
        leaderboard = pd.DataFrame(board)
        leaderboard.to_csv(self.output_dir / "leaderboard.csv", index=False)
        pd.DataFrame(rows).to_csv(self.output_dir / "metrics_by_seed.csv", index=False)
        pd.DataFrame(runs).to_csv(self.output_dir / "runs.csv", index=False)
        pd.DataFrame(curves).to_csv(self.output_dir / "training_history.csv", index=False)
        detailed = []
        if rows:
            frame = pd.DataFrame(rows)
            for key, group in frame.groupby("trial"):
                summary = summarize(group.to_dict("records"))
                summary["trial"] = key
                detailed.extend(summary.to_dict("records"))
        pd.DataFrame(detailed).to_csv(self.output_dir / "summary_detailed.csv", index=False)
        self._ablation_contrasts(rows)
        return leaderboard

    def _ablation_contrasts(self, rows):
        if not rows:
            return
        frame = pd.DataFrame(rows)
        contrasts = []
        keys = ["seed", "scenario", "level", "group", "units"]
        for stage, group in frame.groupby("stage"):
            variants = set(group.variant)
            pairs = [(v, "reference", "addition") for v in variants if v.startswith("add_")]
            pairs += [("full", v, "removal") for v in variants if v.startswith("without_")]
            if "full" in variants:
                pairs.append(("full", "reference", "combined"))
                # Compact ablations omit duplicate leave-one-out arms. These
                # comparisons recover each component's incremental effect when
                # the other selected component is already enabled.
                if not any(v.startswith("without_") for v in variants):
                    pairs += [("full", v, "incremental")
                              for v in variants if v.startswith("add_")]
            for candidate, control, kind in pairs:
                if control not in variants:
                    continue
                merged = group[group.variant == candidate].merge(group[group.variant == control], on=keys,
                    suffixes=("_candidate", "_control"), validate="one_to_one")
                for row in merged.to_dict("records"):
                    if row["count_candidate"] != row["count_control"]:
                        raise ValueError("Ablation masks/counts differ.")
                    contrasts.append({**{k: row[k] for k in keys}, "stage": stage,
                        "candidate": candidate, "control": control, "contrast": kind,
                        "delta_rmse": row["rmse_candidate"] - row["rmse_control"],
                        "delta_mae": row["mae_candidate"] - row["mae_control"]})
        result = pd.DataFrame(contrasts)
        result.to_csv(self.output_dir / "ablation_paired_by_seed.csv", index=False)
        if not result.empty:
            keys = ["stage", "candidate", "control", "contrast", "scenario", "level", "group", "units"]
            result.groupby(keys).agg(n_pairs=("seed", "size"), delta_rmse_mean=("delta_rmse", "mean"),
                delta_rmse_std=("delta_rmse", "std"), delta_mae_mean=("delta_mae", "mean")).reset_index().to_csv(
                    self.output_dir / "ablation_summary.csv", index=False)
            # Main research endpoint: mean four-scenario RMSE difference,
            # paired within seed BEFORE computing mean/std across seeds.
            overall = result[(result.level == "overall") & (result.units == "normalized")]
            pair_keys = ["stage", "candidate", "control", "contrast", "seed"]
            if not overall.empty:
                if not (overall.groupby(pair_keys).size() == 4).all():
                    raise ValueError("Paired ablation needs all four scenarios.")
                scores = overall.groupby(pair_keys)[["delta_rmse", "delta_mae"]].mean().reset_index()
                scores.to_csv(self.output_dir / "ablation_score_by_seed.csv", index=False)
                scores.groupby(pair_keys[:-1]).agg(
                    n_pairs=("seed", "size"), delta_score_mean=("delta_rmse", "mean"),
                    delta_score_std=("delta_rmse", "std")
                ).reset_index().to_csv(self.output_dir / "ablation_score_summary.csv", index=False)

    def evaluate_test(self, trials):
        """Explicit post-selection development evaluation; never changes ranking."""
        trials = list(trials)
        if not trials or len(trials) != len(set(trials)):
            raise ValueError("Select unique completed trials explicitly.")
        for key in trials:
            entry = self.manifest["trials"][key]
            if any(entry["runs"].get(str(s), {}).get("status") != "complete" for s in self.seeds):
                raise ValueError(f"Incomplete trial: {key}")
        old = self.manifest.get("test_selection")
        if old is not None and old != trials:
            raise ValueError("Test selection is already frozen for this study.")
        self.manifest["test_selection"] = trials
        self._save()
        preprocessing, data, metadata, hashes = load_and_check_data(self.data_root, True)
        if any(hashes[k] != h for k, h in self.manifest["protocol"]["data_sha256"].items()):
            raise ValueError("Data changed since training.")
        self.manifest["test_data_sha256"] = hashes
        rows = []
        for key in trials:
            for seed in self.seeds:
                path = self.output_dir / key / f"seed_{seed}" / "best.pt"
                if sha256(path) != self.manifest["trials"][key]["runs"][str(seed)]["hashes"]["best.pt"]:
                    raise ValueError("Checkpoint changed since validation.")
                saved = torch.load(path, map_location=self.device, weights_only=False)
                model = MSAITS(MSAITSConfig(**saved["config"]))
                model.backend.network.load_state_dict(saved["state_dict"])
                model._is_fitted = True
                rows.extend({**row, "trial": key} for row in evaluate_details(model, data["test"],
                    metadata["test"], preprocessing, "m_saits", seed, "test", include_well_log=True))
                del model
        save_json(self.output_dir / "test_metrics.json", rows)
        pd.DataFrame(rows).to_csv(self.output_dir / "test_metrics_by_seed.csv", index=False)
        self._save()
        return pd.DataFrame(rows)

    def archive(self):
        leaderboard = self.export()
        metrics = self.output_dir / "metrics_by_seed.csv"
        try:
            records = pd.read_csv(metrics).to_dict("records")
        except pd.errors.EmptyDataError:
            records = []
        # Single portable research bundle as well as the analysis-friendly CSVs.
        save_json(self.output_dir / "experiment_results.json", {
            **self.manifest, "leaderboard": leaderboard.to_dict("records"), "metrics": records,
        })
        return Path(shutil.make_archive(str(self.output_dir), "zip", self.output_dir))
