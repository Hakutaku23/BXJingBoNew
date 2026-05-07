from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["time", "t90", "y_ok", "y_low", "y_high", "y_out_spec"]
PRIMARY_DOSE_PRIORITY = [
    "ca_per_rubber_flow_win_60_mean",
    "ca_per_rubber_flow_lag_165",
    "ca_win_60_mean",
    "ca_lag_165",
]
PREFERRED_CONTEXT_FEATURES = [
    "rubber_flow_2_win_60_mean",
    "bromine_feed_win_60_mean",
    "tank_rubber_conc_win_60_mean",
    "esbo_feed_win_60_mean",
    "neutral_alkali_feed_win_60_mean",
    "r513_temp_win_60_mean",
    "r514_temp_win_60_mean",
]
LEAKAGE_COLUMNS = {
    "time",
    "t90",
    "t90_C",
    "t90_D",
    "t90_E",
    "t90_label_count",
    "y_ok",
    "y_low",
    "y_high",
    "y_out_spec",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an offline calcium consumption prescription policy from historical dose-response."
    )
    parser.add_argument("--input", type=Path, default=Path("data/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--feature-report", type=Path, default=Path("data/t90_ca_feature_report.json"))
    parser.add_argument("--dose-response-report", type=Path, default=Path("data/t90_ca_dose_response_report.json"))
    parser.add_argument("--dose-response-bins", type=Path, default=Path("data/t90_ca_dose_response_bins.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/t90_ca_policy_recommendations.parquet"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/t90_ca_policy_summary.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/t90_ca_policy_report.json"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--min-bin-samples", type=int, default=50)
    parser.add_argument("--min-stratum-samples", type=int, default=80)
    parser.add_argument("--neighbor-max-k", type=int, default=50)
    parser.add_argument("--min-neighbors", type=int, default=20)
    parser.add_argument("--min-expected-gain", type=float, default=0.03)
    return parser.parse_args()


def as_jsonable(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): as_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [as_jsonable(item) for item in value]
    return value


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object], dict[str, object], pd.DataFrame]:
    if not args.input.exists():
        raise FileNotFoundError(f"Input parquet does not exist: {args.input}")
    if not args.dose_response_bins.exists():
        raise FileNotFoundError(f"Dose-response bins CSV does not exist: {args.dose_response_bins}")
    frame = pd.read_parquet(args.input)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    invalid_time = int(frame["time"].isna().sum())
    if invalid_time:
        raise ValueError(f"Input dataset contains {invalid_time} invalid time values.")
    frame = frame.sort_values("time").reset_index(drop=True)
    feature_report = load_json(args.feature_report)
    dose_report = load_json(args.dose_response_report)
    dose_bins = pd.read_csv(args.dose_response_bins, encoding="utf-8-sig")
    return frame, feature_report, dose_report, dose_bins


def is_leakage_column(column: str) -> bool:
    lowered = column.lower()
    return (
        column in LEAKAGE_COLUMNS
        or lowered.startswith("pred_")
        or lowered.startswith("p_")
        or lowered.endswith("_pred")
        or lowered.startswith("target_")
        or lowered.endswith("_target")
    )


def select_primary_dose_feature(frame: pd.DataFrame, dose_report: dict[str, object]) -> str:
    reported = dose_report.get("primary_dose_feature")
    if isinstance(reported, str) and reported in frame.columns:
        return reported
    for feature in PRIMARY_DOSE_PRIORITY:
        if feature in frame.columns:
            return feature
    raise ValueError("No usable calcium dose feature found from report or fallback priority list.")


def select_context_features(frame: pd.DataFrame, feature_report: dict[str, object]) -> list[str]:
    groups = feature_report.get("feature_groups", {})
    process_context = set()
    if isinstance(groups, dict) and isinstance(groups.get("process_context_features"), list):
        process_context = {str(feature) for feature in groups["process_context_features"]}
    features = []
    for feature in PREFERRED_CONTEXT_FEATURES:
        if feature in frame.columns and not is_leakage_column(feature):
            if not process_context or feature in process_context:
                features.append(feature)
    return features


def make_quantile_bins(values: pd.Series, requested_bins: int) -> tuple[pd.Series, int, list[str]]:
    warnings: list[str] = []
    dose = pd.to_numeric(values, errors="coerce")
    usable = dose.dropna()
    if usable.empty:
        return pd.Series(pd.NA, index=values.index, dtype="Int64"), 0, ["No usable dose values for binning."]
    max_bins = min(int(requested_bins), int(usable.nunique()), int(len(usable)))
    if max_bins < 2:
        return pd.Series(0, index=values.index, dtype="Int64").where(dose.notna(), pd.NA), 1, [
            "Only one dose bin is possible because dose values have too little variation."
        ]
    for bins in range(max_bins, 1, -1):
        try:
            cut = pd.qcut(dose, q=bins, labels=False, duplicates="drop")
            effective = int(pd.Series(cut).dropna().nunique())
            if effective >= 2:
                if effective < requested_bins:
                    warnings.append(f"Effective dose bin count {effective} is lower than requested {requested_bins}.")
                return pd.Series(cut, index=values.index, dtype="Int64"), effective, warnings
        except ValueError:
            continue
    ranked = dose.rank(method="first")
    fallback_bins = min(requested_bins, int(ranked.dropna().nunique()))
    cut = pd.qcut(ranked, q=fallback_bins, labels=False, duplicates="drop")
    effective = int(pd.Series(cut).dropna().nunique())
    warnings.append("Used rank-based fallback for dose quantile bins.")
    return pd.Series(cut, index=values.index, dtype="Int64"), effective, warnings


def summarize_bin_group(group: pd.DataFrame, dose_feature: str) -> dict[str, object]:
    dose = pd.to_numeric(group[dose_feature], errors="coerce")
    return {
        "sample_count": int(len(group)),
        "dose_min": float(dose.min()),
        "dose_max": float(dose.max()),
        "dose_mean": float(dose.mean()),
        "t90_mean": float(group["t90"].mean()),
        "ok_rate": float(group["y_ok"].mean()),
        "low_rate": float(group["y_low"].mean()),
        "high_rate": float(group["y_high"].mean()),
        "out_spec_rate": float(group["y_out_spec"].mean()),
    }


def build_global_bins(
    frame: pd.DataFrame,
    dose_feature: str,
    n_bins: int,
    min_bin_samples: int,
) -> tuple[pd.DataFrame, dict[str, object] | None, list[dict[str, object]], list[dict[str, object]], list[str]]:
    bin_ids, effective_bins, warnings = make_quantile_bins(frame[dose_feature], n_bins)
    work = frame.copy()
    work["dose_bin_id"] = bin_ids
    rows: list[dict[str, object]] = []
    for bin_id, group in work[work["dose_bin_id"].notna()].groupby("dose_bin_id", sort=True):
        bin_id_int = int(bin_id)
        row = {
            "bin_id": bin_id_int,
            "bin_label": "",
            **summarize_bin_group(group, dose_feature),
        }
        row["bin_label"] = f"{bin_id_int}: [{row['dose_min']:.6g}, {row['dose_max']:.6g}]"
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("bin_id").reset_index(drop=True)
    if len(table) != effective_bins:
        warnings.append("Effective bin count differs from summarized bin count after dropping missing dose rows.")

    eligible = table[table["sample_count"] >= min_bin_samples] if not table.empty else pd.DataFrame()
    global_best = None
    if not eligible.empty:
        best = eligible.sort_values(["ok_rate", "sample_count"], ascending=[False, False]).iloc[0].to_dict()
        global_best = {key: as_jsonable(value) for key, value in best.items()}

    overall_ok = float(frame["y_ok"].mean())
    overall_low = float(frame["y_low"].mean())
    overall_high = float(frame["y_high"].mean())
    safe_rows = table[
        (table["ok_rate"] >= overall_ok)
        & (table["high_rate"] <= overall_high)
        & (table["low_rate"] <= overall_low + 0.02)
    ]
    risk_rows = table[
        (table["high_rate"] >= overall_high + 0.05)
        | (table["low_rate"] >= overall_low + 0.03)
    ]
    return (
        table,
        global_best,
        safe_rows.to_dict(orient="records"),
        risk_rows.to_dict(orient="records"),
        warnings,
    )


def valid_spearman(x: pd.Series, y: pd.Series) -> float:
    values = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(values) < 30 or values["x"].nunique() <= 1 or values["y"].nunique() <= 1:
        return 0.0
    corr = values["x"].corr(values["y"], method="spearman")
    return float(corr) if corr is not None and np.isfinite(corr) else 0.0


def tertile_labels(values: pd.Series) -> tuple[pd.Series, list[str]]:
    warnings: list[str] = []
    numeric = pd.to_numeric(values, errors="coerce")
    try:
        labels = pd.qcut(numeric, q=3, labels=["low", "mid", "high"], duplicates="drop")
        result = pd.Series(labels, index=values.index, dtype="object")
        if result.dropna().nunique() < 3:
            warnings.append(f"{values.name}: fewer than 3 tertile groups due to duplicate values.")
        return result, warnings
    except ValueError:
        ranked = numeric.rank(method="first")
        labels = pd.qcut(ranked, q=min(3, ranked.dropna().nunique()), labels=False, duplicates="drop")
        label_map = {0: "low", 1: "mid", 2: "high"}
        result = pd.Series(labels, index=values.index).map(label_map)
        warnings.append(f"{values.name}: used rank-based fallback for tertile labels.")
        return result, warnings


def best_bin_for_subset(subset: pd.DataFrame, min_bin_samples: int) -> dict[str, object] | None:
    if subset.empty or "dose_bin_id" not in subset.columns:
        return None
    rows = []
    for bin_id, group in subset[subset["dose_bin_id"].notna()].groupby("dose_bin_id", sort=True):
        if len(group) < min_bin_samples:
            continue
        rows.append(
            {
                "bin_id": int(bin_id),
                "sample_count": int(len(group)),
                "ok_rate": float(group["y_ok"].mean()),
                "low_rate": float(group["y_low"].mean()),
                "high_rate": float(group["y_high"].mean()),
                "out_spec_rate": float(group["y_out_spec"].mean()),
            }
        )
    if not rows:
        return None
    return sorted(rows, key=lambda item: (-item["ok_rate"], -item["sample_count"]))[0]


def build_strata(
    frame: pd.DataFrame,
    context_features: list[str],
    dose_bin_ids: pd.Series,
    min_stratum_samples: int,
    min_bin_samples: int,
) -> tuple[pd.Series, list[dict[str, object]], list[str], list[str]]:
    warnings: list[str] = []
    work = frame.copy()
    work["dose_bin_id"] = dose_bin_ids
    single_summaries: list[dict[str, object]] = []
    tertiles: dict[str, pd.Series] = {}
    for feature in context_features:
        labels, feature_warnings = tertile_labels(work[feature].rename(feature))
        warnings.extend(feature_warnings)
        tertiles[feature] = labels
        for label in ["low", "mid", "high"]:
            mask = labels == label
            if int(mask.sum()) < min_stratum_samples:
                continue
            best = best_bin_for_subset(work[mask], min_bin_samples=max(10, min_bin_samples // 2))
            single_summaries.append(
                {
                    "stratum_type": "single",
                    "stratum_id": f"{feature}={label}",
                    "context_features": [feature],
                    "sample_count": int(mask.sum()),
                    "best_bin": best,
                }
            )

    ranked_context = sorted(
        context_features,
        key=lambda feature: abs(valid_spearman(work[feature], work["y_out_spec"])),
        reverse=True,
    )
    combined_features = ranked_context[:3]
    if combined_features:
        combined = pd.Series("", index=work.index, dtype="object")
        valid_mask = pd.Series(True, index=work.index)
        for feature in combined_features:
            labels = tertiles[feature]
            valid_mask &= labels.notna()
            combined = combined + feature + "=" + labels.fillna("missing").astype(str) + "|"
        combined = combined.str.rstrip("|")
        combined = combined.where(valid_mask, pd.NA)
    else:
        combined = pd.Series(pd.NA, index=work.index, dtype="object")

    stratum_summary = single_summaries.copy()
    valid_combined_ids = set()
    for stratum_id, group in work[combined.notna()].groupby(combined[combined.notna()], sort=True):
        if len(group) < min_stratum_samples:
            continue
        best = best_bin_for_subset(group, min_bin_samples=max(10, min_bin_samples // 2))
        valid_combined_ids.add(stratum_id)
        stratum_summary.append(
            {
                "stratum_type": "combined",
                "stratum_id": str(stratum_id),
                "context_features": combined_features,
                "sample_count": int(len(group)),
                "best_bin": best,
            }
        )

    sample_stratum = combined.where(combined.isin(valid_combined_ids), "global")
    return sample_stratum, stratum_summary, combined_features, warnings


def robust_scale_context(frame: pd.DataFrame, context_features: list[str]) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    scaled = []
    stats: dict[str, dict[str, float]] = {}
    for feature in context_features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        median = float(values.median())
        q75 = float(values.quantile(0.75))
        q25 = float(values.quantile(0.25))
        iqr = q75 - q25
        scale = iqr if np.isfinite(iqr) and iqr > 0 else 1.0
        stats[feature] = {"median": median, "iqr": scale}
        scaled.append(((values - median) / scale).to_numpy(dtype=float))
    if not scaled:
        return np.empty((len(frame), 0)), stats
    return np.vstack(scaled).T, stats


def bin_stats_from_indices(
    frame: pd.DataFrame,
    indices: np.ndarray,
    bin_id: int,
) -> dict[str, float | int] | None:
    subset = frame.iloc[indices]
    subset = subset[subset["dose_bin_id"] == bin_id]
    if subset.empty:
        return None
    return {
        "neighbor_count": int(len(subset)),
        "ok_rate": float(subset["y_ok"].mean()),
        "low_rate": float(subset["y_low"].mean()),
        "high_rate": float(subset["y_high"].mean()),
        "out_spec_rate": float(subset["y_out_spec"].mean()),
    }


def global_bin_lookup(global_table: pd.DataFrame) -> dict[int, dict[str, object]]:
    return {int(row["bin_id"]): row.to_dict() for _, row in global_table.iterrows()}


def choose_recommendation(
    frame: pd.DataFrame,
    row_index: int,
    neighbor_indices: np.ndarray,
    current_bin_id: int,
    global_bins: dict[int, dict[str, object]],
    high_risk_bin_ids: set[int],
    min_expected_gain: float,
) -> tuple[int, dict[str, object], str]:
    current_stats = bin_stats_from_indices(frame, neighbor_indices, current_bin_id)
    if current_stats is None or int(current_stats["neighbor_count"]) < 10:
        return current_bin_id, {"reason": "insufficient_current_bin_support"}, "insufficient_current_bin_support"

    current_ok = float(current_stats["ok_rate"])
    current_low = float(current_stats["low_rate"])
    current_high = float(current_stats["high_rate"])
    candidates: list[dict[str, object]] = []
    for bin_id in sorted(global_bins):
        local_stats = bin_stats_from_indices(frame, neighbor_indices, bin_id)
        if local_stats is None or int(local_stats["neighbor_count"]) < 10:
            continue
        ok_gain = float(local_stats["ok_rate"]) - current_ok
        low_worse = float(local_stats["low_rate"]) - current_low
        high_worse = float(local_stats["high_rate"]) - current_high
        safe = high_worse <= 0.03 and low_worse <= 0.02
        if not safe:
            continue
        if ok_gain < min_expected_gain and bin_id != current_bin_id:
            continue
        candidates.append(
            {
                "bin_id": int(bin_id),
                "neighbor_count": int(local_stats["neighbor_count"]),
                "ok_rate": float(local_stats["ok_rate"]),
                "low_rate": float(local_stats["low_rate"]),
                "high_rate": float(local_stats["high_rate"]),
                "out_spec_rate": float(local_stats["out_spec_rate"]),
                "expected_ok_rate_gain": ok_gain,
            }
        )

    if not candidates:
        return current_bin_id, {**current_stats, "expected_ok_rate_gain": 0.0}, "no_safe_better_bin"

    if current_bin_id in high_risk_bin_ids:
        lower_safe = [item for item in candidates if int(item["bin_id"]) < current_bin_id]
        if lower_safe:
            best = sorted(lower_safe, key=lambda item: (-float(item["ok_rate"]), -int(item["neighbor_count"])))[0]
            return int(best["bin_id"]), best, "high_risk_high_dose_prefer_lower_safe_bin"

    best = sorted(candidates, key=lambda item: (-float(item["ok_rate"]), -int(item["neighbor_count"])))[0]
    if int(best["bin_id"]) == current_bin_id:
        return current_bin_id, best, "current_bin_best_supported"
    return int(best["bin_id"]), best, "safe_better_similar_bin"


def build_recommendations(
    frame: pd.DataFrame,
    dose_feature: str,
    global_table: pd.DataFrame,
    safe_bins: list[dict[str, object]],
    high_risk_bins: list[dict[str, object]],
    context_features: list[str],
    stratum_ids: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object], list[str]]:
    warnings: list[str] = []
    work = frame.copy()
    global_bins = global_bin_lookup(global_table)
    high_risk_bin_ids = {int(item["bin_id"]) for item in high_risk_bins}
    safe_bin_ids = {int(item["bin_id"]) for item in safe_bins}
    scaled_context, scale_stats = robust_scale_context(work, context_features)
    outputs: list[dict[str, object]] = []

    for i, row in work.iterrows():
        dose = row[dose_feature]
        current_bin = row.get("dose_bin_id")
        base = {
            "time": row["time"],
            "t90": row["t90"],
            "y_ok": row["y_ok"],
            "y_low": row["y_low"],
            "y_high": row["y_high"],
            "y_out_spec": row["y_out_spec"],
            "dose_feature": dose_feature,
            "dose_current": dose,
            "current_bin_id": pd.NA,
            "current_bin_label": None,
            "recommended_bin_id": pd.NA,
            "recommended_bin_label": None,
            "recommended_dose_min": math.nan,
            "recommended_dose_max": math.nan,
            "current_bin_ok_rate": math.nan,
            "recommended_bin_ok_rate": math.nan,
            "expected_ok_rate_gain": math.nan,
            "current_bin_low_rate": math.nan,
            "recommended_bin_low_rate": math.nan,
            "current_bin_high_rate": math.nan,
            "recommended_bin_high_rate": math.nan,
            "neighbor_count": 0,
            "stratum_id": stratum_ids.iloc[i] if i < len(stratum_ids) else "global",
            "action": "hold",
            "reason": "",
        }
        if pd.isna(dose):
            base["reason"] = "missing_dose"
            outputs.append(base)
            continue
        if pd.isna(current_bin):
            base["reason"] = "missing_current_bin"
            outputs.append(base)
            continue
        current_bin_id = int(current_bin)
        current_global = global_bins.get(current_bin_id)
        if current_global:
            base["current_bin_id"] = current_bin_id
            base["current_bin_label"] = current_global["bin_label"]
            base["current_bin_ok_rate"] = current_global["ok_rate"]
            base["current_bin_low_rate"] = current_global["low_rate"]
            base["current_bin_high_rate"] = current_global["high_rate"]

        if scaled_context.shape[1] == 0:
            base["reason"] = "no_context_features"
            outputs.append(base)
            continue
        row_context = scaled_context[i]
        valid_dims = np.isfinite(row_context) & np.isfinite(scaled_context).any(axis=0)
        if not valid_dims.any():
            base["reason"] = "missing_context"
            outputs.append(base)
            continue
        diff = np.abs(scaled_context[:, valid_dims] - row_context[valid_dims])
        pair_valid_counts = np.isfinite(diff).sum(axis=1)
        distances = np.full(len(work), np.inf, dtype=float)
        comparable = pair_valid_counts > 0
        if comparable.any():
            distances[comparable] = np.nanmedian(diff[comparable], axis=1)
        distances[i] = np.inf
        finite = np.isfinite(distances)
        if int(finite.sum()) < args.min_neighbors:
            base["reason"] = "insufficient_support"
            outputs.append(base)
            continue
        order = np.argsort(distances[finite])
        finite_indices = np.where(finite)[0]
        neighbor_indices = finite_indices[order[: args.neighbor_max_k]]
        if len(neighbor_indices) < args.min_neighbors:
            base["reason"] = "insufficient_support"
            outputs.append(base)
            continue

        recommended_bin_id, rec_stats, reason = choose_recommendation(
            work,
            row_index=i,
            neighbor_indices=neighbor_indices,
            current_bin_id=current_bin_id,
            global_bins=global_bins,
            high_risk_bin_ids=high_risk_bin_ids,
            min_expected_gain=args.min_expected_gain,
        )
        recommended_global = global_bins.get(recommended_bin_id)
        base["neighbor_count"] = int(len(neighbor_indices))
        base["recommended_bin_id"] = recommended_bin_id
        if recommended_global:
            base["recommended_bin_label"] = recommended_global["bin_label"]
            base["recommended_dose_min"] = recommended_global["dose_min"]
            base["recommended_dose_max"] = recommended_global["dose_max"]
        base["recommended_bin_ok_rate"] = rec_stats.get("ok_rate", math.nan)
        base["recommended_bin_low_rate"] = rec_stats.get("low_rate", math.nan)
        base["recommended_bin_high_rate"] = rec_stats.get("high_rate", math.nan)
        base["expected_ok_rate_gain"] = rec_stats.get("expected_ok_rate_gain", 0.0)

        expected_gain = float(base["expected_ok_rate_gain"]) if pd.notna(base["expected_ok_rate_gain"]) else 0.0
        low_worse = float(base["recommended_bin_low_rate"]) - float(base["current_bin_low_rate"]) if pd.notna(base["recommended_bin_low_rate"]) and pd.notna(base["current_bin_low_rate"]) else 0.0
        high_worse = float(base["recommended_bin_high_rate"]) - float(base["current_bin_high_rate"]) if pd.notna(base["recommended_bin_high_rate"]) and pd.notna(base["current_bin_high_rate"]) else 0.0
        if reason in {"insufficient_current_bin_support", "no_safe_better_bin", "current_bin_best_supported"}:
            base["action"] = "hold"
            base["reason"] = reason
        elif expected_gain < args.min_expected_gain:
            base["action"] = "hold"
            base["reason"] = "expected_gain_below_threshold"
        elif low_worse > 0.02 or high_worse > 0.03:
            base["action"] = "hold"
            base["reason"] = "risk_worsening_guardrail"
        elif recommended_bin_id > current_bin_id:
            base["action"] = "increase_ca_small_step"
            base["reason"] = reason
        elif recommended_bin_id < current_bin_id:
            base["action"] = "decrease_ca_small_step"
            base["reason"] = reason
        else:
            base["action"] = "hold"
            base["reason"] = "current_bin_equals_recommended_bin"
        if current_bin_id in high_risk_bin_ids and recommended_bin_id < current_bin_id and recommended_bin_id in safe_bin_ids:
            base["action"] = "decrease_ca_small_step"
            base["reason"] = "high_risk_high_dose_decrease_to_safe_bin"
        outputs.append(base)

    config = {
        "neighbor_max_k": int(args.neighbor_max_k),
        "min_neighbors": int(args.min_neighbors),
        "min_expected_gain": float(args.min_expected_gain),
        "risk_guardrails": {"max_low_rate_worsening": 0.02, "max_high_rate_worsening": 0.03},
        "context_scale_stats": scale_stats,
    }
    return pd.DataFrame(outputs), config, warnings


def action_summary(recommendations: pd.DataFrame) -> dict[str, object]:
    total = int(len(recommendations))
    action_counts = recommendations["action"].value_counts().to_dict()
    actionable = recommendations[recommendations["action"].isin(["increase_ca_small_step", "decrease_ca_small_step"])]
    summary = {
        "total_samples": total,
        "actionable_samples": int(len(actionable)),
        "hold_count": int(action_counts.get("hold", 0)),
        "increase_count": int(action_counts.get("increase_ca_small_step", 0)),
        "decrease_count": int(action_counts.get("decrease_ca_small_step", 0)),
        "missing_dose_count": int((recommendations["reason"] == "missing_dose").sum()),
        "insufficient_support_count": int(recommendations["reason"].astype(str).str.contains("insufficient|no_context|missing_context").sum()),
        "mean_expected_ok_rate_gain": float(actionable["expected_ok_rate_gain"].mean()) if len(actionable) else 0.0,
        "median_expected_ok_rate_gain": float(actionable["expected_ok_rate_gain"].median()) if len(actionable) else 0.0,
    }
    for action_name, suffix in [
        ("hold", "hold"),
        ("increase_ca_small_step", "increase"),
        ("decrease_ca_small_step", "decrease"),
    ]:
        subset = recommendations[recommendations["action"] == action_name]
        summary[f"actual_ok_rate_{suffix}"] = float(subset["y_ok"].mean()) if len(subset) else math.nan
        summary[f"actual_high_rate_{suffix}"] = float(subset["y_high"].mean()) if len(subset) else math.nan
        summary[f"actual_low_rate_{suffix}"] = float(subset["y_low"].mean()) if len(subset) else math.nan
    return summary


def decide_next_step(summary: dict[str, object], global_best: dict[str, object] | None, overall_ok: float) -> str:
    actionable = int(summary["actionable_samples"])
    mean_gain = float(summary["mean_expected_ok_rate_gain"])
    hold_high = summary.get("actual_high_rate_hold")
    inc_high = summary.get("actual_high_rate_increase")
    dec_high = summary.get("actual_high_rate_decrease")
    hold_low = summary.get("actual_low_rate_hold")
    inc_low = summary.get("actual_low_rate_increase")
    dec_low = summary.get("actual_low_rate_decrease")
    risk_worse = False
    for value in [inc_high, dec_high]:
        if value is not None and np.isfinite(value) and hold_high is not None and np.isfinite(hold_high):
            risk_worse = risk_worse or float(value) > float(hold_high) + 0.05
    for value in [inc_low, dec_low]:
        if value is not None and np.isfinite(value) and hold_low is not None and np.isfinite(hold_low):
            risk_worse = risk_worse or float(value) > float(hold_low) + 0.03
    if risk_worse:
        return "do_not_use_policy"
    if actionable >= 100 and mean_gain >= 0.03:
        return "inspect_policy_before_shadow_trial"
    global_signal = bool(global_best and float(global_best.get("ok_rate", 0.0)) >= overall_ok + 0.03)
    if global_signal:
        return "insufficient_support_refine_strata_or_collect_more_data"
    return "do_not_use_policy"


def next_doc_section_number(doc_path: Path) -> int:
    if not doc_path.exists():
        return 10
    text = doc_path.read_text(encoding="utf-8")
    numbers = [int(match.group(1)) for match in re.finditer(r"^##\s+(\d+)\.", text, flags=re.MULTILINE)]
    return max(numbers, default=9) + 1


def append_documentation(
    doc_path: Path,
    args: argparse.Namespace,
    section_number: int,
    primary_dose_feature: str,
    global_best: dict[str, object] | None,
    safe_bins: list[dict[str, object]],
    high_risk_bins: list[dict[str, object]],
    context_features: list[str],
    similar_config: dict[str, object],
    policy_summary: dict[str, object],
    recommended_next_step: str,
    warnings: list[str],
) -> bool:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    best_text = (
        f"bin {global_best['bin_id']}，范围 [{global_best['dose_min']:.6g}, {global_best['dose_max']:.6g}]，ok_rate={global_best['ok_rate']:.4f}"
        if global_best
        else "无满足样本数要求的全局最佳分箱"
    )
    safe_text = ", ".join(
        f"bin {item['bin_id']}([{item['dose_min']:.6g}, {item['dose_max']:.6g}])" for item in safe_bins
    ) or "无"
    risk_text = ", ".join(
        f"bin {item['bin_id']}([{item['dose_min']:.6g}, {item['dose_max']:.6g}])" for item in high_risk_bins
    ) or "无"
    lines = [
        "",
        f"## {section_number}. 硬脂酸钙单耗处方优化实验",
        "",
        "- 本阶段跳过通用 T90 预测模型训练。此前预警主线和控制模型探索说明，直接 T90 预测不足以作为控制依据，因此本阶段改为基于历史剂量响应、工况分层和相似样本的离线钙单耗处方优化。",
        f"- 输入文件：`{args.input}`、`{args.feature_report}`、`{args.dose_response_report}`、`{args.dose_response_bins}`。",
        f"- 输出文件：`{args.output}`、`{args.summary_output}`、`{args.report}`。",
        f"- 主剂量特征：`{primary_dose_feature}`。",
        f"- 全局最佳钙单耗范围：{best_text}。",
        f"- 全局安全分箱：{safe_text}。",
        f"- 全局高风险分箱：{risk_text}。",
        "- 工况上下文特征：" + ("，".join(f"`{feature}`" for feature in context_features) if context_features else "无可用上下文特征") + "。",
        (
            "- 相似样本配置："
            f"最多近邻 {similar_config['neighbor_max_k']}，"
            f"最少近邻 {similar_config['min_neighbors']}，"
            f"最小期望合格率增益 {similar_config['min_expected_gain']}。"
        ),
        (
            "- 策略摘要："
            f"总样本 {policy_summary['total_samples']}，"
            f"可行动样本 {policy_summary['actionable_samples']}，"
            f"hold {policy_summary['hold_count']}，"
            f"increase {policy_summary['increase_count']}，"
            f"decrease {policy_summary['decrease_count']}。"
        ),
        (
            "- 期望收益："
            f"平均 expected_ok_rate_gain={policy_summary['mean_expected_ok_rate_gain']:.4f}，"
            f"中位数={policy_summary['median_expected_ok_rate_gain']:.4f}。"
        ),
        "- 风险与限制：该策略只来自离线历史相似样本，不是自动闭环控制；高剂量区域存在高 T90 风险，低/高 T90 风险需分别检查；样本支持不足或风险边界变差时保持 hold。",
        f"- recommended_next_step：`{recommended_next_step}`。",
        "- 警告：" + ("；".join(warnings) if warnings else "无。"),
    ]
    with doc_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    return True


def build_report(
    args: argparse.Namespace,
    row_count: int,
    primary_dose_feature: str,
    global_table: pd.DataFrame,
    global_best: dict[str, object] | None,
    safe_bins: list[dict[str, object]],
    high_risk_bins: list[dict[str, object]],
    context_features: list[str],
    stratum_summary: list[dict[str, object]],
    similar_config: dict[str, object],
    policy_summary: dict[str, object],
    warnings: list[str],
    recommended_next_step: str,
) -> dict[str, object]:
    return {
        "input_path": str(args.input),
        "feature_report_path": str(args.feature_report),
        "dose_response_report_path": str(args.dose_response_report),
        "dose_response_bins_path": str(args.dose_response_bins),
        "output_path": str(args.output),
        "summary_output_path": str(args.summary_output),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": int(row_count),
        "primary_dose_feature": primary_dose_feature,
        "global_bin_summary": global_table.to_dict(orient="records"),
        "global_best_bin": global_best,
        "global_safe_bins": safe_bins,
        "global_high_risk_bins": high_risk_bins,
        "context_features_used": context_features,
        "stratum_summary": stratum_summary,
        "similar_sample_config": similar_config,
        "policy_summary": policy_summary,
        "warnings": warnings,
        "assumptions": [
            "No generic T90 prediction model is trained in this script.",
            "Policy recommendations are based on historical dose-response bins and similar context samples.",
            "Calcium dose values are not imputed for policy decisions.",
            "Context features exclude leakage and target columns.",
            "Recommendations are offline prescriptions only and are not automatic closed-loop control.",
            "Actions are conservative and hold when support or risk guardrails are insufficient.",
        ],
        "recommended_next_step": recommended_next_step,
    }


def print_summary(
    primary_dose_feature: str,
    global_best: dict[str, object] | None,
    safe_bins: list[dict[str, object]],
    high_risk_bins: list[dict[str, object]],
    context_features: list[str],
    policy_summary: dict[str, object],
    recommended_next_step: str,
    doc_appended: bool,
) -> None:
    print("T90 calcium consumption policy optimization complete.")
    print(f"  primary dose feature: {primary_dose_feature}")
    print(f"  global best bin: {global_best}")
    print(f"  global safe bins: {[item['bin_id'] for item in safe_bins]}")
    print(f"  global high-risk bins: {[item['bin_id'] for item in high_risk_bins]}")
    print(f"  context features used: {context_features}")
    print(
        "  action counts: "
        f"hold={policy_summary['hold_count']}, "
        f"increase={policy_summary['increase_count']}, "
        f"decrease={policy_summary['decrease_count']}"
    )
    print(f"  mean expected gain: {policy_summary['mean_expected_ok_rate_gain']}")
    print(f"  recommended next step: {recommended_next_step}")
    print(f"  docs appended: {doc_appended}")


def main() -> None:
    args = parse_args()
    if args.n_bins < 2:
        raise ValueError("--n-bins must be at least 2")
    warnings: list[str] = []
    frame, feature_report, dose_report, _dose_bins = load_inputs(args)
    primary_dose_feature = select_primary_dose_feature(frame, dose_report)
    context_features = select_context_features(frame, feature_report)
    if not context_features:
        warnings.append("No preferred process-context features are available; all recommendations will hold.")

    global_table, global_best, safe_bins, high_risk_bins, bin_warnings = build_global_bins(
        frame,
        primary_dose_feature,
        args.n_bins,
        args.min_bin_samples,
    )
    warnings.extend(bin_warnings)
    if global_best is None:
        warnings.append("No global best bin met min_bin_samples.")

    work = frame.copy()
    dose_bin_ids, _effective_bins, _ = make_quantile_bins(work[primary_dose_feature], args.n_bins)
    work["dose_bin_id"] = dose_bin_ids
    stratum_ids, stratum_summary, combined_context_features, stratum_warnings = build_strata(
        work,
        context_features,
        dose_bin_ids,
        args.min_stratum_samples,
        args.min_bin_samples,
    )
    warnings.extend(stratum_warnings)

    recommendations, similar_config, rec_warnings = build_recommendations(
        work,
        primary_dose_feature,
        global_table,
        safe_bins,
        high_risk_bins,
        context_features,
        stratum_ids,
        args,
    )
    similar_config["combined_strata_context_features"] = combined_context_features
    warnings.extend(rec_warnings)

    policy_summary = action_summary(recommendations)
    recommended_next_step = decide_next_step(
        policy_summary,
        global_best,
        overall_ok=float(frame["y_ok"].mean()),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_parquet(args.output, index=False)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([policy_summary]).to_csv(args.summary_output, index=False, encoding="utf-8-sig")

    report = build_report(
        args=args,
        row_count=len(frame),
        primary_dose_feature=primary_dose_feature,
        global_table=global_table,
        global_best=global_best,
        safe_bins=safe_bins,
        high_risk_bins=high_risk_bins,
        context_features=context_features,
        stratum_summary=stratum_summary,
        similar_config=similar_config,
        policy_summary=policy_summary,
        warnings=warnings,
        recommended_next_step=recommended_next_step,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(as_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")

    section_number = next_doc_section_number(args.doc)
    doc_appended = append_documentation(
        args.doc,
        args,
        section_number,
        primary_dose_feature,
        global_best,
        safe_bins,
        high_risk_bins,
        context_features,
        similar_config,
        policy_summary,
        recommended_next_step,
        warnings,
    )
    print_summary(
        primary_dose_feature,
        global_best,
        safe_bins,
        high_risk_bins,
        context_features,
        policy_summary,
        recommended_next_step,
        doc_appended,
    )


if __name__ == "__main__":
    main()
