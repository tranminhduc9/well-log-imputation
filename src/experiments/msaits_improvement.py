"""Cross-log M-SAITS search with joint validation RMSE/physical-MAPE selection.

Run locally: python -m src.experiments.msaits_improvement --device cpu
No test data is loaded or scored by this search.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from src.experiments.msaits_study import MSAITSStudy
from src.experiments.geolink import run_experiment, save_json, sha256


# Match the settings used by the existing 20260928 GeoLink result bundle.
SAITS_REFERENCE = dict(
    d_model=256, d_inner=128, n_layers=2, n_heads=4, learning_rate=0.0005,
    batch_size=32, ort_weight=1.0, mit_weight=1.0, mit_reduction="segment",
    min_delta=1e-4,
)
MSAITS_REFERENCE = dict(
    **{**SAITS_REFERENCE, "learning_rate": 0.0003},
    encoder_channels=8, kernel_size=15, conv_expansion=2,
    multi_scale_conv=True, multi_scale_kernels=(3, 7, 15),
    depth_encoding=False, masking_strategy="mixed", masking_rate=0.2,
    independent_shuffle=True,
)


def improvement_candidates():
    """Eight fixed arms isolate cross-log attention, direct conditioning and loss.

    Floors/scaling and per-seed SAITS references are injected by CrossLogStudy;
    no validation/test target statistics are used as input features.
    """
    cross = {**MSAITS_REFERENCE, "cross_log_attention": True, "cross_log_width": 32}
    conditional = {**MSAITS_REFERENCE, "decoder": "conditional", "decoder_width": 64}
    combined = {**cross, **{key: conditional[key] for key in ("decoder", "decoder_width")}}
    gated = {**combined, "gap_aware_gate": True, "gradient_clip": 1.0}
    return [
        ("reference", dict(MSAITS_REFERENCE)),
        ("add_cross_attention", cross),
        ("add_conditional_decoder", conditional),
        ("cross_log_full", combined),
        ("cross_log_gap", gated),
        ("hybrid_01", {**gated, "mit_mse_weight": .1}),
        ("hybrid_03", {**gated, "mit_mse_weight": .3}),
        ("relative_01", {**gated, "mit_mse_weight": .1, "mit_relative_weight": .1}),
    ]


def validation_objectives(rows, mape_column="mape", *, allow_undefined=False):
    """One row per seed: equal scenarios RMSE; equal logs/scenarios physical MAPE."""
    frame = pd.DataFrame(rows)
    frame = frame[frame.split == "val"]
    scenarios = {"Single", "Block-20", "Block-100", "Entire-Log"}
    overall = frame[(frame.level == "overall") & (frame.units == "normalized")]
    physical = frame[(frame.level == "log") & (frame.units == "original")]
    if overall.empty or physical.empty:
        raise ValueError("Joint selection requires validation overall RMSE and original-unit per-log MAPE")
    if overall.duplicated(["seed", "scenario"]).any() or physical.duplicated(["seed", "scenario", "group"]).any():
        raise ValueError("Duplicate validation objective rows")
    if set(physical.seed) != set(overall.seed):
        raise ValueError("RMSE/MAPE seed sets differ")
    output = []
    logs = set(physical.group)
    for seed, group in overall.groupby("seed"):
        detail = physical[physical.seed == seed]
        if set(group.scenario) != scenarios or set(detail.scenario) != scenarios or any(
            set(detail[detail.scenario == scenario].group) != logs for scenario in scenarios
        ):
            raise ValueError("Each seed requires all four scenarios and the same logs")
        if mape_column not in detail or not np.isfinite(group.rmse).all() or (
            not allow_undefined and not np.isfinite(detail[mape_column]).all()
        ):
            raise ValueError("Undefined RMSE/MAPE cannot participate in selection")
        output.append(dict(seed=int(seed), rmse=float(group.rmse.mean()), mape=float(np.mean(detail[mape_column].to_numpy())),
            entire_log_rmse=float(group.loc[group.scenario == "Entire-Log", "rmse"].iloc[0]),
            entire_log_mape=float(np.mean(detail.loc[detail.scenario == "Entire-Log", mape_column].to_numpy()))))
    return pd.DataFrame(output)


class CrossLogStudy(MSAITSStudy):
    """Resume-compatible study using the worst of the two SAITS-relative ratios."""
    def bind_reference(self, rows, mape_policy="raw"):
        if mape_policy not in {"raw", "floored"}:
            raise ValueError("mape_policy must be raw or floored")
        if self.manifest.get("mape_policy_name", mape_policy) != mape_policy:
            raise ValueError("Cannot change MAPE policy when resuming")
        self.mape_policy = mape_policy
        self.mape_column = "mape" if mape_policy == "raw" else "mape_floored"
        self.reference_rows = pd.DataFrame(rows)
        self.reference_scores = validation_objectives(rows, self.mape_column).set_index("seed")
        if set(self.reference_scores.index) != set(self.seeds):
            raise ValueError("SAITS reference must cover every requested seed")
        if (self.reference_scores[["rmse", "mape"]] <= 0).any().any():
            raise ValueError("Positive SAITS errors are required for relative improvement")
        self.manifest["selection"] = (
            "Minimize max(mean-seed macro RMSE / SAITS macro RMSE, "
            "mean-seed macro physical MAPE / SAITS macro physical MAPE). "
            "Checkpoint criterion uses paired per-seed references. No best-seed selection."
        )
        self.manifest["mape_policy"] = (
            "Equal logs and four scenarios in original units; exclude exactly zero targets only. "
            "Training relative-loss denominator floor = 0.01 * training normalization std; "
            "floor is NOT applied to reported/selection MAPE. Undefined MAPE fails selection."
        )
        self.manifest["mape_policy_name"] = mape_policy
        if mape_policy == "floored":
            self.manifest["mape_policy"] = (
                "Floored MAPE in original units, denominator=max(abs(target), 0.01*train std), "
                "including zero targets. Equal logs/scenarios/seeds. Raw MAPE retained separately."
            )
        self._save()

    def _config(self, settings, seed, folder):
        # Physical scales are injected before enabling the relative objective.
        config = super()._config({**settings, "mit_relative_weight": 0.0}, seed, folder)
        if not hasattr(self, "reference_scores"):
            raise ValueError("Bind the completed SAITS validation reference first")
        scaling = [self.preprocessing["normalization"][log] for log in self.preprocessing["log_columns"]]
        return replace(config, validation_objective="rmse_mape",
            mit_relative_weight=settings.get("mit_relative_weight", 0.0),
            validation_mape_floor=self.mape_policy == "floored",
            reference_rmse=float(self.reference_scores.loc[seed, "rmse"]),
            reference_mape=float(self.reference_scores.loc[seed, "mape"]),
            physical_means=tuple(s["mean"] for s in scaling),
            physical_stds=tuple(s["std"] for s in scaling),
            relative_floors=tuple(.01 * s["std"] for s in scaling))

    def export(self):
        board = super().export()
        if board.empty:
            return board
        metrics = pd.read_csv(self.output_dir / "metrics_by_seed.csv")
        reference = self.reference_scores
        for index, row in board.iterrows():
            trial_metrics = metrics[metrics.trial == row.trial]
            scores = validation_objectives(trial_metrics, self.mape_column).set_index("seed")
            paired = reference.loc[scores.index]
            rmse_ratio = scores.rmse.mean() / paired.rmse.mean()
            mape_ratio = scores.mape.mean() / paired.mape.mean()
            individual = np.maximum(scores.rmse / paired.rmse, scores.mape / paired.mape)
            board.loc[index, "score_mean"] = max(rmse_ratio, mape_ratio)
            board.loc[index, "score_std"] = individual.std(ddof=1)
            board.loc[index, "mape_policy"] = self.mape_policy
            for seed, value in individual.items():
                board.loc[index, f"score_seed_{seed}"] = value
            for metric in ("rmse", "mape", "entire_log_rmse", "entire_log_mape"):
                board.loc[index, f"{metric}_mean"] = scores[metric].mean()
                board.loc[index, f"{metric}_std"] = scores[metric].std(ddof=1)
                board.loc[index, f"{metric}_gain_pct"] = 100 * (1-scores[metric].mean()/paired[metric].mean())
            for column in ("mape", "mape_floored"):
                extra = validation_objectives(trial_metrics, column, allow_undefined=True).set_index("seed")
                base = validation_objectives(self.reference_rows, column, allow_undefined=True).set_index("seed").loc[scores.index]
                measured, denominator = np.mean(extra.mape.to_numpy()), np.mean(base.mape.to_numpy())
                board.loc[index, f"{column}_reported_mean"] = measured
                board.loc[index, f"{column}_reported_gain_pct"] = (
                    100*(1-measured/denominator) if denominator > 0 else np.nan)
            for target in (7, 10):
                board.loc[index, f"both_at_least_{target}pct"] = (
                    bool(row.complete) and min(100*(1-rmse_ratio), 100*(1-mape_ratio)) >= target)
        board = board.sort_values(["score_mean", "trial"])
        board.to_csv(self.output_dir / "leaderboard.csv", index=False)
        return board


def run_improvement_study(project_root, data_root, output_base, *, seeds=(129, 219, 912),
                          epochs=500, patience=30, device="auto", resume_dir=None, trials=None,
                          mape_policy="raw"):
    """Train a fixed SAITS reference and candidates; export all seeds of the winner."""
    if mape_policy not in {"raw", "floored"}:
        raise ValueError("mape_policy must be raw or floored")
    candidates = improvement_candidates()
    if trials is not None:
        unknown = set(trials) - {name for name, _ in candidates}
        if unknown or not trials or len(trials) != len(set(trials)):
            raise ValueError(f"Select unique supported trials; unknown={unknown}")
        candidates = [(name, settings) for name, settings in candidates if name in trials]
    study = CrossLogStudy(project_root, data_root, output_base, seeds=seeds, epochs=epochs,
                         patience=patience, device=device, resume_dir=resume_dir)
    if resume_dir is not None and study.manifest.get("experiment_kind") != "cross_log_joint":
        raise ValueError("Resume requires a cross-log joint-objective study, not an older RMSE-only study")
    study.manifest["experiment_kind"] = "cross_log_joint"
    study._save()
    print("Study directory (resume path):", study.output_dir, flush=True)
    if "saits_reference" not in study.manifest:
        args = argparse.Namespace(data_root=str(data_root), output_dir=str(study.output_dir / "saits_reference"),
            models=("saits",), seeds=tuple(seeds), evaluate_test=False,
            saits_epochs=epochs, saits_patience=patience, device=study.device,
            mape_floors=tuple(.01*study.preprocessing["normalization"][log]["std"]
                              for log in study.preprocessing["log_columns"]))
        reference_dir = run_experiment(args, project_root, {"saits": SAITS_REFERENCE})
        study.manifest["saits_reference"] = dict(directory=str(reference_dir.relative_to(study.output_dir)),
            manifest_sha256=sha256(reference_dir / "experiment_results.json"),
            metrics_sha256=sha256(reference_dir / "metrics_by_seed.csv"))
        study._save()
    reference = study.manifest["saits_reference"]
    reference_dir = study.output_dir / reference["directory"]
    for filename, key in [("experiment_results.json", "manifest_sha256"), ("metrics_by_seed.csv", "metrics_sha256")]:
        if sha256(reference_dir / filename) != reference[key]:
            raise ValueError("SAITS reference changed since this study")
    manifest = json.loads((reference_dir / "experiment_results.json").read_text(encoding="utf-8"))
    if (manifest["status"] != "complete" or manifest["models"] != ["saits"] or
        manifest["seeds"] != list(seeds) or
        manifest["data_sha256"] != study.manifest["protocol"]["data_sha256"]):
        raise ValueError("SAITS reference data/protocol mismatch")
    study.bind_reference(pd.read_csv(reference_dir / "metrics_by_seed.csv"), mape_policy)
    audit = []
    for scenario, dataset in study.data["val"].items():
        for index, log in enumerate(study.preprocessing["log_columns"]):
            scale = study.preprocessing["normalization"][log]
            truth = dataset["X_intact"][..., index] * scale["std"] + scale["mean"]
            values = truth[dataset["indicating_mask"][..., index]].astype(np.float64)
            floor = .01 * scale["std"]
            audit.append(dict(scenario=scenario, log=log, count=len(values), floor=floor,
                exact_zero_count=int((values == 0).sum()), below_floor_count=int((np.abs(values) < floor).sum()),
                minimum_abs_target=float(np.abs(values).min())))
    pd.DataFrame(audit).to_csv(study.output_dir / "mape_denominator_audit.csv", index=False)
    study.reference_scores.reset_index().to_csv(study.output_dir / "saits_validation_objectives.csv", index=False)
    study.run_stage("cross_log", candidates)
    winner = study.manifest["stages"]["cross_log"]["winner"]
    board = study.export()
    validation_rows = pd.read_csv(study.output_dir / "metrics_by_seed.csv")
    per_seed = []
    for trial, group in validation_rows.groupby("trial"):
        scored = validation_objectives(group, study.mape_column).set_index("seed")
        for seed, values in scored.iterrows():
            baseline = study.reference_scores.loc[seed]
            per_seed.append(dict(trial=trial, seed=int(seed), mape_policy=mape_policy, **values.to_dict(),
                rmse_saits=float(baseline.rmse), mape_saits=float(baseline.mape),
                rmse_gain_pct=100*(1-values.rmse/baseline.rmse),
                mape_gain_pct=100*(1-values.mape/baseline.mape)))
    pd.DataFrame(per_seed).to_csv(study.output_dir / "validation_objectives_by_seed.csv", index=False)
    selected = board[board.trial == winner].iloc[0]
    artifacts = []
    for seed in seeds:
        source = study.output_dir / winner / f"seed_{seed}" / "best.pt"
        destination = study.output_dir / "best_models" / f"seed_{seed}" / "best.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        artifacts.append(dict(model="m_saits", trial=winner, seed=seed,
            artifact=str(destination.relative_to(study.output_dir)), sha256=sha256(destination)))
    save_json(study.output_dir / "best_model.json", dict(trial=winner,
        settings=study.manifest["trials"][winner]["settings"], runs=artifacts,
        validation=selected.to_dict(), mape_policy=mape_policy, target_met_7pct=bool(selected.both_at_least_7pct),
        target_met_10pct=bool(selected.both_at_least_10pct),
        note="Best configuration across all requested seeds; target is measured, not guaranteed."))
    print(board.to_string(index=False), flush=True)
    print("Best validation configuration:", winner, flush=True)
    print("Both RMSE/MAPE >=7%:", bool(selected.both_at_least_7pct), flush=True)
    print("Model manifest:", study.output_dir / "best_model.json", flush=True)
    return study


def compare_with_saits(candidate_rows, saits_rows):
    """Pair by seed/split/scenario, then report observed relative reductions.

    Macro rows compare arithmetic means of the four errors, not the mean of
    four percentages. Positive gain means a lower error than SAITS. Achieving
    7% or more is accepted; 7% is not an upper cap to force results into.
    """
    def overall(rows):
        frame = pd.DataFrame(rows)
        return frame[(frame.level == "overall") & (frame.units == "normalized")].copy()

    candidate, baseline = overall(candidate_rows), overall(saits_rows)
    if candidate.empty or baseline.empty:
        raise ValueError("Both candidate and SAITS need overall metrics.")
    if set(baseline.model) != {"saits"}:
        raise ValueError("Reference rows must belong to SAITS only.")
    keys = ["seed", "split", "scenario"]
    if "trial" not in candidate:
        candidate["trial"] = candidate["model"]
    if candidate.duplicated(["trial", *keys]).any() or baseline.duplicated(keys).any():
        raise ValueError("Duplicate comparison rows.")
    joined = candidate.merge(
        baseline[keys + ["count", "mae", "rmse"]], on=keys, how="left",
        suffixes=("", "_saits"), validate="many_to_one", indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Every candidate needs the matching SAITS seed and scenario.")
    if not joined["count"].eq(joined["count_saits"]).all():
        raise ValueError("Candidate and SAITS scored different point counts.")
    errors = ["mae", "rmse", "mae_saits", "rmse_saits"]
    if not np.isfinite(joined[errors]).all().all() or (joined[errors] < 0).any().any():
        raise ValueError("Metrics must be finite non-negative errors.")
    raw_candidate, raw_baseline = pd.DataFrame(candidate_rows), pd.DataFrame(saits_rows)
    physical_candidate = raw_candidate[(raw_candidate.level == "log") & (raw_candidate.units == "original")].copy()
    physical_baseline = raw_baseline[(raw_baseline.level == "log") & (raw_baseline.units == "original")]
    if not physical_candidate.empty or not physical_baseline.empty:
        if "trial" not in physical_candidate:
            physical_candidate["trial"] = physical_candidate.model
        physical_keys = [*keys, "group"]
        if physical_candidate.duplicated(["trial", *physical_keys]).any() or physical_baseline.duplicated(physical_keys).any():
            raise ValueError("Duplicate physical MAPE rows")
        physical = physical_candidate.merge(physical_baseline[physical_keys + ["mape", "count"]],
            on=physical_keys, how="left", suffixes=("", "_saits"), validate="many_to_one", indicator=True)
        if not physical["_merge"].eq("both").all() or not physical["count"].eq(physical["count_saits"]).all():
            raise ValueError("Physical MAPE reference masks/logs differ")
        for (_, seed, split, scenario), group in physical.groupby(["trial", *keys]):
            baseline_group = physical_baseline[(physical_baseline.seed == seed) &
                (physical_baseline.split == split) & (physical_baseline.scenario == scenario)]
            if set(group.group) != set(baseline_group.group):
                raise ValueError("Physical MAPE requires every reference log")
        if not np.isfinite(physical[["mape", "mape_saits"]]).all().all():
            raise ValueError("Undefined physical MAPE")
        aggregate = physical.groupby(["trial", *keys])[["mape", "mape_saits"]].mean().reset_index()
        joined = joined.drop(columns=["mape"], errors="ignore").merge(
            aggregate, on=["trial", *keys], how="left", validate="one_to_one")
        if not np.isfinite(joined[["mape", "mape_saits"]]).all().all():
            raise ValueError("Missing physical MAPE scenario rows")
        errors += ["mape", "mape_saits"]
    scenarios = {"Single", "Block-20", "Block-100", "Entire-Log"}
    groups = joined.groupby(["trial", "seed", "split"])
    if any(set(group.scenario) != scenarios for _, group in groups):
        raise ValueError("Each comparison must contain all four scenarios.")
    macro = groups[errors].mean().reset_index()
    macro["scenario"] = "Mean (4 scenarios)"
    result = pd.concat([
        joined[["trial", *keys, *errors]], macro,
    ], ignore_index=True)
    for metric in (["mae", "rmse", "mape"] if "mape" in errors else ["mae", "rmse"]):
        denominator = result[f"{metric}_saits"].replace(0, np.nan)
        result[f"{metric}_gain_pct"] = 100 * (1 - result[metric] / denominator)
    result["both_at_least_5pct"] = (
        (result.mae_gain_pct >= 5) & (result.rmse_gain_pct >= 5)
    )
    if "mape" in errors:
        for target in (7, 10):
            result[f"both_at_least_{target}pct"] = (
                (result.rmse_gain_pct >= target) & (result.mape_gain_pct >= target))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-root", type=Path, default=root / "data/processed")
    parser.add_argument("--output-base", type=Path, default=root / "results/msaits_cross_log")
    parser.add_argument("--seeds", type=int, nargs="+", default=[129, 219, 912])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--trials", nargs="+", help="Optional subset of the fixed eight candidates")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mape-policy", choices=["raw", "floored"], default="raw",
                        help="Raw MAPE (default) or explicit denominator-floor MAPE; both are exported")
    args = parser.parse_args()
    import torch
    import logging
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    study = run_improvement_study(root, args.data_root, args.output_base,
        seeds=tuple(args.seeds), epochs=args.epochs, patience=args.patience,
        device=args.device, resume_dir=args.resume_dir, trials=args.trials, mape_policy=args.mape_policy)
    print("Study directory:", study.output_dir, flush=True)


if __name__ == "__main__":
    main()
