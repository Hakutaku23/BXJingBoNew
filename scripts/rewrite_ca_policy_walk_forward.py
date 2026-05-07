from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


T90_LOW = 8.20
T90_HIGH = 8.70
PRIMARY_DOSE_PRIORITY = [
    "ca_per_rubber_flow_win_60_mean",
    "ca_per_rubber_flow_lag_165",
    "ca_win_60_mean",
    "ca_lag_165",
]
PROCESS_CONTEXT_FEATURES = [
    "rubber_flow_2_win_60_mean",
    "bromine_feed_win_60_mean",
    "tank_rubber_conc_win_60_mean",
    "esbo_feed_win_60_mean",
    "neutral_alkali_feed_win_60_mean",
    "r513_temp_win_60_mean",
    "r514_temp_win_60_mean",
]
OPTIONAL_IR_FEATURE = "output_ir_corrected_win_15_slope"
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
ACTION_VALUES = ["hold", "increase_ca_small_step", "decrease_ca_small_step"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict walk-forward calcium consumption prescription evaluator.")
    parser.add_argument("--features", type=Path, default=Path("data/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--feature-report", type=Path, default=Path("data/t90_ca_feature_report.json"))
    parser.add_argument("--data-with-ir", type=Path, default=Path("data/data_clean_with_ir.parquet"))
    parser.add_argument("--ir-report", type=Path, default=Path("data/output_ir_proxy_evaluation.json"))
    parser.add_argument("--dose-response-report", type=Path, default=Path("data/t90_ca_dose_response_report.json"))
    parser.add_argument("--policy-validation-report", type=Path, default=Path("data/t90_ca_policy_validation_report.json"))
    parser.add_argument("--output", type=Path, default=Path("data/t90_ca_walk_forward_policy_recommendations.parquet"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/t90_ca_walk_forward_policy_summary.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/t90_ca_walk_forward_policy_report.json"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--label-release-delay-hours", type=float, default=24.0)
    parser.add_argument("--min-history-samples", type=int, default=300)
    parser.add_argument("--n-dose-bins", type=int, default=5)
    parser.add_argument("--min-bin-samples", type=int, default=30)
    parser.add_argument("--neighbor-max-k", type=int, default=50)
    parser.add_argument("--min-neighbors", type=int, default=20)
    parser.add_argument("--min-expected-gain", type=float, default=0.03)
    parser.add_argument("--max-high-risk-worsen", type=float, default=0.03)
    parser.add_argument("--max-low-risk-worsen", type=float, default=0.02)
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
        raise FileNotFoundError(f"Required JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_targets(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["t90"] = pd.to_numeric(frame["t90"], errors="coerce")
    if "y_ok" not in frame.columns:
        frame["y_ok"] = ((frame["t90"] >= T90_LOW) & (frame["t90"] <= T90_HIGH)).astype(int)
    if "y_low" not in frame.columns:
        frame["y_low"] = (frame["t90"] < T90_LOW).astype(int)
    if "y_high" not in frame.columns:
        frame["y_high"] = (frame["t90"] > T90_HIGH).astype(int)
    if "y_out_spec" not in frame.columns:
        frame["y_out_spec"] = ((frame["t90"] < T90_LOW) | (frame["t90"] > T90_HIGH)).astype(int)
    return frame


def is_leakage_column(column: str) -> bool:
    lowered = column.lower()
    return (
        column in LEAKAGE_COLUMNS
        or lowered.startswith("pred_")
        or lowered.startswith("p_")
        or lowered.endswith("_pred")
        or lowered.startswith("target_")
    )


def load_supervised(args: argparse.Namespace, dose_report: dict[str, object]) -> tuple[pd.DataFrame, str, list[str]]:
    if not args.features.exists():
        raise FileNotFoundError(f"Feature parquet does not exist: {args.features}")
    frame = pd.read_parquet(args.features)
    required = ["time", "t90"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {missing}")
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    if frame["time"].isna().any():
        raise ValueError("Feature dataset contains invalid time values.")
    frame = ensure_targets(frame)
    frame = frame[frame["t90"].notna()].sort_values("time").reset_index(drop=True)

    primary = dose_report.get("primary_dose_feature")
    if not isinstance(primary, str) or primary not in frame.columns:
        primary = next((feature for feature in PRIMARY_DOSE_PRIORITY if feature in frame.columns), None)
    if primary is None:
        raise ValueError("No primary calcium dose feature is available.")

    context = [feature for feature in PROCESS_CONTEXT_FEATURES if feature in frame.columns and not is_leakage_column(feature)]
    return frame, primary, context


def add_ir_context(frame: pd.DataFrame, data_with_ir_path: Path, warnings: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    if not data_with_ir_path.exists():
        warnings.append(f"data-with-ir is missing: {data_with_ir_path}; IR optional branch will run without IR context.")
        frame[OPTIONAL_IR_FEATURE] = np.nan
        return frame
    ir = pd.read_parquet(data_with_ir_path, columns=["time", OPTIONAL_IR_FEATURE])
    ir["time"] = pd.to_datetime(ir["time"], errors="coerce")
    ir = ir.dropna(subset=["time"]).sort_values("time")
    frame = frame.merge(ir, on="time", how="left")
    return frame


def quantile_edges(values: pd.Series, n_bins: int) -> tuple[np.ndarray | None, list[str]]:
    warnings: list[str] = []
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) < 2 or len(np.unique(clean)) < 2:
        return None, ["Dose values have insufficient variation for bins."]
    for bins in range(min(n_bins, len(np.unique(clean))), 1, -1):
        edges = np.unique(np.quantile(clean, np.linspace(0, 1, bins + 1)))
        if len(edges) >= 3:
            if len(edges) - 1 < n_bins:
                warnings.append(f"Effective dose bin count {len(edges) - 1} is lower than requested {n_bins}.")
            edges[0] = -np.inf
            edges[-1] = np.inf
            return edges, warnings
    return None, ["Could not construct valid dose bins."]


def assign_bins(values: pd.Series, edges: np.ndarray) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    ids = pd.cut(numeric, bins=edges, labels=False, include_lowest=True)
    return pd.Series(ids, index=values.index, dtype="float")


def bin_summary(history: pd.DataFrame, dose_feature: str, bin_ids: pd.Series, min_bin_samples: int) -> dict[str, object]:
    work = history.copy()
    work["dose_bin_id"] = bin_ids
    rows = []
    for bin_id, group in work[work["dose_bin_id"].notna()].groupby("dose_bin_id", sort=True):
        dose = pd.to_numeric(group[dose_feature], errors="coerce")
        rows.append(
            {
                "bin_id": int(bin_id),
                "sample_count": int(len(group)),
                "dose_min": float(dose.min()),
                "dose_max": float(dose.max()),
                "dose_mean": float(dose.mean()),
                "ok_rate": float(group["y_ok"].mean()),
                "low_rate": float(group["y_low"].mean()),
                "high_rate": float(group["y_high"].mean()),
                "out_spec_rate": float(group["y_out_spec"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return {"table": [], "best_bin": None, "safe_bins": [], "high_risk_bins": []}
    overall_ok = float(history["y_ok"].mean())
    overall_low = float(history["y_low"].mean())
    overall_high = float(history["y_high"].mean())
    eligible = table[table["sample_count"] >= min_bin_samples]
    best = None if eligible.empty else eligible.sort_values(["ok_rate", "sample_count"], ascending=[False, False]).iloc[0].to_dict()
    safe = table[
        (table["ok_rate"] >= overall_ok)
        & (table["high_rate"] <= overall_high)
        & (table["low_rate"] <= overall_low + 0.02)
    ]
    risk = table[
        (table["high_rate"] >= overall_high + 0.05)
        | (table["low_rate"] >= overall_low + 0.03)
    ]
    return {
        "table": table.to_dict(orient="records"),
        "best_bin": {k: as_jsonable(v) for k, v in best.items()} if best is not None else None,
        "safe_bins": safe.to_dict(orient="records"),
        "high_risk_bins": risk.to_dict(orient="records"),
    }


def robust_scaled(history: pd.DataFrame, current: pd.Series, features: list[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not features:
        return None, None
    scaled_cols = []
    current_vals = []
    for feature in features:
        values = pd.to_numeric(history[feature], errors="coerce")
        cur = pd.to_numeric(pd.Series([current.get(feature)]), errors="coerce").iloc[0]
        if not np.isfinite(cur):
            continue
        median = values.median()
        q25 = values.quantile(0.25)
        q75 = values.quantile(0.75)
        iqr = q75 - q25
        scale = float(iqr) if np.isfinite(iqr) and iqr > 0 else 1.0
        scaled_cols.append(((values - median) / scale).to_numpy(dtype=float))
        current_vals.append(float((cur - median) / scale))
    if not scaled_cols:
        return None, None
    return np.vstack(scaled_cols).T, np.asarray(current_vals, dtype=float)


def nearest_neighbors(history: pd.DataFrame, current: pd.Series, features: list[str], max_k: int) -> np.ndarray:
    matrix, row = robust_scaled(history, current, features)
    if matrix is None or row is None:
        return np.asarray([], dtype=int)
    diff = np.abs(matrix - row)
    valid_count = np.isfinite(diff).sum(axis=1)
    distances = np.full(len(history), np.inf, dtype=float)
    comparable = valid_count > 0
    if comparable.any():
        distances[comparable] = np.nanmedian(diff[comparable], axis=1)
    finite = np.isfinite(distances)
    if not finite.any():
        return np.asarray([], dtype=int)
    order = np.argsort(distances[finite])
    finite_positions = np.where(finite)[0]
    return finite_positions[order[:max_k]]


def local_bin_stats(history: pd.DataFrame, neighbor_idx: np.ndarray, bin_id: int) -> dict[str, object] | None:
    subset = history.iloc[neighbor_idx]
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


def choose_candidate(
    history: pd.DataFrame,
    neighbor_idx: np.ndarray,
    current_bin: int,
    global_info: dict[str, object],
    args: argparse.Namespace,
) -> tuple[int, dict[str, object], str]:
    current_stats = local_bin_stats(history, neighbor_idx, current_bin)
    if current_stats is None or current_stats["neighbor_count"] < 10:
        return current_bin, {"expected_ok_rate_gain": 0.0}, "insufficient_current_bin_support"
    high_risk_ids = {int(item["bin_id"]) for item in global_info["high_risk_bins"]}
    safe_ids = {int(item["bin_id"]) for item in global_info["safe_bins"]}
    best = None
    for item in global_info["table"]:
        bin_id = int(item["bin_id"])
        if bin_id in high_risk_ids:
            continue
        stats = local_bin_stats(history, neighbor_idx, bin_id)
        if stats is None or stats["neighbor_count"] < 10:
            continue
        gain = stats["ok_rate"] - current_stats["ok_rate"]
        high_worse = stats["high_rate"] - current_stats["high_rate"]
        low_worse = stats["low_rate"] - current_stats["low_rate"]
        if gain < args.min_expected_gain or high_worse > args.max_high_risk_worsen or low_worse > args.max_low_risk_worsen:
            continue
        candidate = {
            **stats,
            "expected_ok_rate_gain": float(gain),
            "bin_id": bin_id,
            "high_worse": float(high_worse),
            "low_worse": float(low_worse),
        }
        if best is None or (candidate["ok_rate"], candidate["neighbor_count"]) > (best["ok_rate"], best["neighbor_count"]):
            best = candidate
    if current_bin in high_risk_ids:
        lower_safe = [int(bin_id) for bin_id in safe_ids if int(bin_id) < current_bin]
        for bin_id in lower_safe:
            stats = local_bin_stats(history, neighbor_idx, bin_id)
            if stats and stats["neighbor_count"] >= 10:
                gain = stats["ok_rate"] - current_stats["ok_rate"]
                high_worse = stats["high_rate"] - current_stats["high_rate"]
                low_worse = stats["low_rate"] - current_stats["low_rate"]
                if gain >= args.min_expected_gain and high_worse <= args.max_high_risk_worsen and low_worse <= args.max_low_risk_worsen:
                    return bin_id, {**stats, "expected_ok_rate_gain": float(gain)}, "high_risk_current_bin_decrease_to_safe_bin"
    if best is None:
        return current_bin, {**current_stats, "expected_ok_rate_gain": 0.0}, "no_safe_candidate"
    return int(best["bin_id"]), best, "safe_better_historical_neighbor_bin"


def blank_row(row: pd.Series, branch: str, has_ir: bool, dose_feature: str, action: str, reason: str) -> dict[str, object]:
    return {
        "time": row["time"],
        "t90": row["t90"],
        "y_ok": row["y_ok"],
        "y_low": row["y_low"],
        "y_high": row["y_high"],
        "y_out_spec": row["y_out_spec"],
        "branch": branch,
        "has_ir": bool(has_ir),
        "dose_feature": dose_feature,
        "dose_current": row.get(dose_feature, math.nan),
        "current_bin_id": pd.NA,
        "recommended_bin_id": pd.NA,
        "recommended_dose_min": math.nan,
        "recommended_dose_max": math.nan,
        "current_neighbor_ok_rate": math.nan,
        "recommended_neighbor_ok_rate": math.nan,
        "expected_ok_rate_gain": 0.0,
        "current_neighbor_low_rate": math.nan,
        "recommended_neighbor_low_rate": math.nan,
        "current_neighbor_high_rate": math.nan,
        "recommended_neighbor_high_rate": math.nan,
        "neighbor_count": 0,
        "eligible_history_count": 0,
        "effective_bin_count": 0,
        "action": action,
        "reason": reason,
    }


def evaluate_sample(
    data: pd.DataFrame,
    index: int,
    branch: str,
    dose_feature: str,
    context_features: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    current = data.iloc[index]
    has_ir = pd.notna(current.get(OPTIONAL_IR_FEATURE))
    delay = pd.Timedelta(hours=args.label_release_delay_hours)
    history = data[data["time"] <= current["time"] - delay].copy()
    base = blank_row(current, branch, has_ir, dose_feature, "hold", "")
    base["eligible_history_count"] = int(len(history))
    if pd.isna(current.get(dose_feature)):
        base["reason"] = "missing_dose"
        return base
    if len(history) < args.min_history_samples:
        base["reason"] = "insufficient_history"
        return base
    edges, _warnings = quantile_edges(history[dose_feature], args.n_dose_bins)
    if edges is None:
        base["reason"] = "invalid_history_bins"
        return base
    history["dose_bin_id"] = assign_bins(history[dose_feature], edges)
    current_bin = assign_bins(pd.Series([current[dose_feature]]), edges).iloc[0]
    if pd.isna(current_bin):
        base["reason"] = "invalid_current_bin"
        return base
    current_bin = int(current_bin)
    global_info = bin_summary(history, dose_feature, history["dose_bin_id"], args.min_bin_samples)
    base["effective_bin_count"] = int(len(global_info["table"]))
    context = list(context_features)
    if branch == "ir_optional_policy":
        enough_ir_history = int(history[OPTIONAL_IR_FEATURE].notna().sum()) >= args.min_neighbors if OPTIONAL_IR_FEATURE in history.columns else False
        if has_ir and enough_ir_history:
            context = context + [OPTIONAL_IR_FEATURE]
        else:
            branch = "ir_optional_policy_without_ir_context"
            base["branch"] = branch
    neighbor_idx = nearest_neighbors(history, current, context, args.neighbor_max_k)
    base["neighbor_count"] = int(len(neighbor_idx))
    if len(neighbor_idx) < args.min_neighbors:
        base["reason"] = "insufficient_neighbors"
        return base
    current_stats = local_bin_stats(history, neighbor_idx, current_bin)
    if current_stats is None:
        base["current_bin_id"] = current_bin
        base["reason"] = "insufficient_current_bin_support"
        return base
    rec_bin, rec_stats, reason = choose_candidate(history, neighbor_idx, current_bin, global_info, args)
    base["current_bin_id"] = current_bin
    base["recommended_bin_id"] = rec_bin
    base["current_neighbor_ok_rate"] = current_stats["ok_rate"]
    base["current_neighbor_low_rate"] = current_stats["low_rate"]
    base["current_neighbor_high_rate"] = current_stats["high_rate"]
    base["recommended_neighbor_ok_rate"] = rec_stats.get("ok_rate", math.nan)
    base["recommended_neighbor_low_rate"] = rec_stats.get("low_rate", math.nan)
    base["recommended_neighbor_high_rate"] = rec_stats.get("high_rate", math.nan)
    base["expected_ok_rate_gain"] = rec_stats.get("expected_ok_rate_gain", 0.0)
    rec_global = next((item for item in global_info["table"] if int(item["bin_id"]) == rec_bin), None)
    if rec_global:
        base["recommended_dose_min"] = rec_global["dose_min"]
        base["recommended_dose_max"] = rec_global["dose_max"]
    if rec_bin == current_bin:
        base["action"] = "hold"
        base["reason"] = reason if reason else "current_bin_equals_recommended"
    elif rec_bin > current_bin:
        base["action"] = "increase_ca_small_step"
        base["reason"] = reason
    else:
        base["action"] = "decrease_ca_small_step"
        base["reason"] = reason
    return base


def summarize(recommendations: pd.DataFrame, split_index: int) -> pd.DataFrame:
    work = recommendations.copy()
    sorted_times = sorted(work["time"].drop_duplicates())
    split_time = sorted_times[split_index] if split_index < len(sorted_times) else sorted_times[-1]
    work["split"] = np.where(work["time"] < split_time, "train_like", "test_like")
    rows = []
    for branch, branch_df in work.groupby("branch", sort=True):
        for split, subset in branch_df.groupby("split", sort=True):
            hold = subset[subset["action"] == "hold"]
            inc = subset[subset["action"] == "increase_ca_small_step"]
            dec = subset[subset["action"] == "decrease_ca_small_step"]
            actionable = subset[subset["action"].isin(["increase_ca_small_step", "decrease_ca_small_step"])]
            row = {
                "branch": branch,
                "split": split,
                "total_samples": int(len(subset)),
                "evaluable_samples": int((subset["reason"] != "insufficient_history").sum()),
                "actionable_samples": int(len(actionable)),
                "action_rate": float(len(actionable) / max(1, len(subset))),
                "hold_count": int(len(hold)),
                "increase_count": int(len(inc)),
                "decrease_count": int(len(dec)),
                "actual_ok_rate_hold": float(hold["y_ok"].mean()) if len(hold) else math.nan,
                "actual_ok_rate_actionable": float(actionable["y_ok"].mean()) if len(actionable) else math.nan,
                "actual_ok_rate_increase": float(inc["y_ok"].mean()) if len(inc) else math.nan,
                "actual_ok_rate_decrease": float(dec["y_ok"].mean()) if len(dec) else math.nan,
                "actual_high_rate_hold": float(hold["y_high"].mean()) if len(hold) else math.nan,
                "actual_high_rate_actionable": float(actionable["y_high"].mean()) if len(actionable) else math.nan,
                "actual_high_rate_increase": float(inc["y_high"].mean()) if len(inc) else math.nan,
                "actual_high_rate_decrease": float(dec["y_high"].mean()) if len(dec) else math.nan,
                "actual_low_rate_hold": float(hold["y_low"].mean()) if len(hold) else math.nan,
                "actual_low_rate_actionable": float(actionable["y_low"].mean()) if len(actionable) else math.nan,
                "actual_low_rate_increase": float(inc["y_low"].mean()) if len(inc) else math.nan,
                "actual_low_rate_decrease": float(dec["y_low"].mean()) if len(dec) else math.nan,
                "mean_expected_ok_rate_gain": float(actionable["expected_ok_rate_gain"].mean()) if len(actionable) else 0.0,
                "median_expected_ok_rate_gain": float(actionable["expected_ok_rate_gain"].median()) if len(actionable) else 0.0,
                "mean_neighbor_count": float(subset["neighbor_count"].mean()),
                "median_neighbor_count": float(subset["neighbor_count"].median()),
                "mean_eligible_history_count": float(subset["eligible_history_count"].mean()),
                "ir_available_rate": float(subset["has_ir"].mean()),
            }
            rows.append(row)
    summary = pd.DataFrame(rows)
    decisions = []
    for _, row in summary.iterrows():
        if row["split"] != "test_like":
            decisions.append("")
            continue
        if row["actionable_samples"] < 50:
            decisions.append("insufficient_history_or_support")
        elif row["actual_ok_rate_actionable"] < row["actual_ok_rate_hold"] + 0.03:
            decisions.append("do_not_use_policy")
        elif row["actual_high_rate_actionable"] > row["actual_high_rate_hold"] + 0.03:
            decisions.append("do_not_use_policy")
        elif row["actual_low_rate_actionable"] > row["actual_low_rate_hold"] + 0.02:
            decisions.append("do_not_use_policy")
        elif row["action_rate"] > 0.20:
            decisions.append("tighten_rules_and_revalidate")
        else:
            decisions.append("valid_for_manual_review_only")
    summary["recommended_next_step_for_branch"] = decisions
    return summary


def overall_decision(summary: pd.DataFrame) -> tuple[str, dict[str, object]]:
    test = summary[summary["split"] == "test_like"].copy()
    branch_map = {row["branch"]: row.to_dict() for _, row in test.iterrows()}
    no_ir = branch_map.get("no_ir_policy")
    ir_rows = [row for key, row in branch_map.items() if key.startswith("ir_optional_policy")]
    ir_best = sorted(ir_rows, key=lambda row: row.get("actual_ok_rate_actionable", -1), reverse=True)[0] if ir_rows else None
    no_ir_valid = no_ir and no_ir.get("recommended_next_step_for_branch") == "valid_for_manual_review_only"
    ir_valid = ir_best and ir_best.get("recommended_next_step_for_branch") == "valid_for_manual_review_only"
    comparison = {"no_ir_policy": no_ir, "ir_optional_policy_best": ir_best}
    if no_ir_valid and not ir_valid:
        return "prepare_manual_review_table", comparison
    if ir_valid and no_ir:
        ir_better = (
            ir_best["actual_ok_rate_actionable"] >= no_ir["actual_ok_rate_actionable"] + 0.01
            or ir_best["actual_high_rate_actionable"] <= no_ir["actual_high_rate_actionable"] - 0.01
        )
        if ir_better:
            return "prepare_manual_review_table", comparison
    if no_ir_valid and ir_valid:
        return "prepare_manual_review_table", comparison
    if any(row.get("recommended_next_step_for_branch") == "insufficient_history_or_support" for row in branch_map.values()):
        if all(row.get("recommended_next_step_for_branch") == "insufficient_history_or_support" for row in branch_map.values()):
            return "stop_policy_work_until_more_data", comparison
    if no_ir_valid and ir_best and ir_best.get("recommended_next_step_for_branch") != "valid_for_manual_review_only":
        return "keep_ir_monitoring_only_and_stop_policy", comparison
    return "tighten_walk_forward_policy_rules", comparison


def next_doc_section_number(path: Path) -> int:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    nums = [int(m.group(1)) for m in re.finditer(r"^##\s+(\d+)\.", text, flags=re.MULTILINE)]
    return max(nums, default=13) + 1


def fmt(value: object) -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.4f}"
    except Exception:
        return "NA"


def append_doc(path: Path, section: int, args: argparse.Namespace, report: dict[str, object]) -> bool:
    summary = pd.DataFrame(report["branch_reports"])
    test_rows = summary[summary["split"] == "test_like"]
    lines = [
        "",
        f"## {section}. 严格 Walk-forward 钙单耗处方策略重写",
        "",
        "- 本次重写是因为上一版处方策略存在潜在时间泄漏，且动作组实际表现差于 hold，不能进入 shadow trial。",
        "- 严格 walk-forward 规则：每个样本只使用样本时刻之前且已过标签释放延迟的历史 LIMS 样本；分箱、上下文尺度和近邻池均在历史样本内即时计算。",
        f"- 标签释放延迟：{report['label_release_delay_hours']} 小时。",
        "- 剂量分箱规则：每个评价时刻仅用历史钙单耗构造分位数分箱，并识别历史最佳、安全和高风险分箱。",
        "- 近邻搜索规则：仅使用过程上下文；IR 分支仅在当前样本和足够历史样本均有 IR 时加入 `output_ir_corrected_win_15_slope`，缺失 IR 不删样本。",
        f"- 输入文件：`{args.features}`、`{args.data_with_ir}`、`{args.ir_report}`、`{args.dose_response_report}`、`{args.policy_validation_report}`。",
        f"- 输出文件：`{args.output}`、`{args.summary_output}`、`{args.report}`。",
    ]
    for _, row in test_rows.iterrows():
        lines.append(
            f"- {row['branch']} test_like：actionable={int(row['actionable_samples'])}，"
            f"action_rate={fmt(row['action_rate'])}，"
            f"hold_ok={fmt(row['actual_ok_rate_hold'])}，"
            f"action_ok={fmt(row['actual_ok_rate_actionable'])}，"
            f"hold_high={fmt(row['actual_high_rate_hold'])}，"
            f"action_high={fmt(row['actual_high_rate_actionable'])}，"
            f"branch_next=`{row['recommended_next_step_for_branch']}`。"
        )
    lines.extend(
        [
            f"- no-IR 与 IR 分支比较：{json.dumps(as_jsonable(report['comparison_no_ir_vs_ir']), ensure_ascii=False)}",
            f"- 最终 recommended_next_step：`{report['recommended_next_step']}`。",
            "- 限制：该结果仍为离线观测验证，不是因果证明，不是自动控制；只有通过验证的动作才可进入人工复核表，不能直接上线。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    return True


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    dose_report = load_json(args.dose_response_report)
    _feature_report = load_json(args.feature_report)
    ir_report = load_json(args.ir_report)
    _validation_report = load_json(args.policy_validation_report)
    data, dose_feature, context_features = load_supervised(args, dose_report)
    data = add_ir_context(data, args.data_with_ir, warnings)
    if OPTIONAL_IR_FEATURE not in data.columns:
        data[OPTIONAL_IR_FEATURE] = np.nan

    rows = []
    for i in range(len(data)):
        rows.append(evaluate_sample(data, i, "no_ir_policy", dose_feature, context_features, args))
        rows.append(evaluate_sample(data, i, "ir_optional_policy", dose_feature, context_features, args))
    rec = pd.DataFrame(rows)
    split_index = int(len(data) * 0.8)
    summary = summarize(rec, split_index)
    next_step, comparison = overall_decision(summary)
    diagnostic_flags = {
        "strict_walk_forward_applied": True,
        "label_release_delay_applied": True,
        "future_neighbors_forbidden": True,
        "dose_bins_fit_on_history_only": True,
        "context_scaling_fit_on_history_only": True,
        "ir_used_only_when_available": True,
        "no_shadow_trial_recommended": True,
    }
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features_path": str(args.features),
        "data_with_ir_path": str(args.data_with_ir),
        "ir_report_path": str(args.ir_report),
        "dose_response_report_path": str(args.dose_response_report),
        "policy_validation_report_path": str(args.policy_validation_report),
        "output_path": str(args.output),
        "summary_output_path": str(args.summary_output),
        "row_count": int(len(data)),
        "primary_dose_feature": dose_feature,
        "context_features_used": context_features,
        "optional_ir_feature": OPTIONAL_IR_FEATURE,
        "label_release_delay_hours": float(args.label_release_delay_hours),
        "min_history_samples": int(args.min_history_samples),
        "branch_reports": summary.to_dict(orient="records"),
        "comparison_no_ir_vs_ir": comparison,
        "diagnostic_flags": diagnostic_flags,
        "warnings": warnings,
        "assumptions": [
            "No generic T90 prediction model is trained.",
            "Every recommendation uses only eligible historical labels released before the sample time.",
            "Dose bins, context scaling, and neighbors are recomputed from history for each sample.",
            "IR is optional context only and is never imputed or forward-filled.",
            "This is offline evaluation, not automatic control.",
        ],
        "recommended_next_step": next_step,
        "ir_proxy_prior_result": {
            "best_ir_feature": ir_report.get("best_ir_feature"),
            "recommended_next_step": ir_report.get("recommended_next_step"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rec.to_parquet(args.output, index=False)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(as_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    section = next_doc_section_number(args.doc)
    doc_appended = append_doc(args.doc, section, args, report)
    print("Strict walk-forward calcium policy evaluation complete.")
    print(f"  primary dose feature: {dose_feature}")
    print(f"  optional IR feature: {OPTIONAL_IR_FEATURE}")
    print(f"  label release delay hours: {args.label_release_delay_hours}")
    test_summary = summary[summary["split"] == "test_like"]
    for _, row in test_summary.iterrows():
        print(
            f"  {row['branch']} test_like: actionable={row['actionable_samples']}, "
            f"hold_ok={fmt(row['actual_ok_rate_hold'])}, action_ok={fmt(row['actual_ok_rate_actionable'])}, "
            f"hold_high={fmt(row['actual_high_rate_hold'])}, action_high={fmt(row['actual_high_rate_actionable'])}, "
            f"next={row['recommended_next_step_for_branch']}"
        )
    print(f"  overall recommended_next_step: {next_step}")
    print(f"  docs appended: {doc_appended}")


if __name__ == "__main__":
    main()
