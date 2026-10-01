"""Predeclared M-SAITS candidates and paired comparisons to SAITS."""
import numpy as np
import pandas as pd

from src.experiments.msaits_study import compact_component_candidates


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
    """Four gate/head arms plus two predeclared hybrid-loss candidates."""
    candidates = compact_component_candidates(
        MSAITS_REFERENCE, ["gap_gate", "per_log_head"]
    )
    combined = dict(candidates[-1][1])
    for weight in (0.1, 0.3):
        candidates.append((f"hybrid_{int(weight * 10):02d}", {
            **combined, "mit_mse_weight": weight, "gradient_clip": 1.0,
        }))
    return candidates


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
    scenarios = {"Single", "Block-20", "Block-100", "Entire-Log"}
    groups = joined.groupby(["trial", "seed", "split"])
    if any(set(group.scenario) != scenarios for _, group in groups):
        raise ValueError("Each comparison must contain all four scenarios.")
    macro = groups[errors].mean().reset_index()
    macro["scenario"] = "Mean (4 scenarios)"
    result = pd.concat([
        joined[["trial", *keys, *errors]], macro,
    ], ignore_index=True)
    for metric in ("mae", "rmse"):
        denominator = result[f"{metric}_saits"].replace(0, np.nan)
        result[f"{metric}_gain_pct"] = 100 * (1 - result[metric] / denominator)
    result["both_at_least_5pct"] = (
        (result.mae_gain_pct >= 5) & (result.rmse_gain_pct >= 5)
    )
    return result
