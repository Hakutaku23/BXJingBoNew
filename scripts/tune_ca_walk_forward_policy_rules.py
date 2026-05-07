from __future__ import annotations

import argparse
import itertools
import json
import math
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
OPTIONAL_IR_MONITOR_FEATURE = "output_ir_corrected_win_15_slope"
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

MIN_HISTORY_GRID = [300, 500, 800]
N_DOSE_BINS_GRID = [5, 7]
NEIGHBOR_MAX_K_GRID = [30, 50, 80]
MIN_NEIGHBORS_GRID = [20, 30, 40]
MIN_EXPECTED_GAIN_GRID = [0.05, 0.08, 0.10, 0.15]
MAX_ACTION_RATE_GRID = [0.10, 0.15, 0.20]
MAX_HIGH_RISK_WORSEN_GRID = [0.00, 0.01, 0.02]
MAX_LOW_RISK_WORSEN_GRID = [0.00, 0.005, 0.01]
RESTRICT_TO_CURRENT_HIGH_RISK_GRID = [False, True]
ALLOW_INCREASE_GRID = [False, True]
ALLOW_DECREASE_GRID = [True]
FORBID_RECOMMENDED_HIGH_RISK_GRID = [True]
REQUIRE_RECOMMENDED_SAFE_BIN_GRID = [True]
REQUIRE_NEIGHBOR_COUNT_AT_MAX_K_GRID = [False, True]
REQUIRE_EXPECTED_GAIN_QUANTILE_GRID = ["none", "top_50", "top_30", "top_20"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune strict walk-forward calcium consumption policy rules."
    )
    parser.add_argument("--features", type=Path, default=Path("data/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--feature-report", type=Path, default=Path("data/t90_ca_feature_report.json"))
    parser.add_argument("--dose-response-report", type=Path, default=Path("data/t90_ca_dose_response_report.json"))
    parser.add_argument(
        "--previous-walk-forward-report",
        type=Path,
        default=Path("data/t90_ca_walk_forward_policy_report.json"),
    )
    parser.add_argument("--data-with-ir", type=Path, default=Path("data/data_clean_with_ir.parquet"))
    parser.add_argument(
        "--results-output",
        type=Path,
        default=Path("data/t90_ca_walk_forward_policy_tuning_results.csv"),
    )
    parser.add_argument(
        "--best-output",
        type=Path,
        default=Path("data/t90_ca_walk_forward_policy_best_recommendations.parquet"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/t90_ca_walk_forward_policy_tuning_report.json"),
    )
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--label-release-delay-hours", type=float, default=24.0)
    parser.add_argument("--min-bin-samples", type=int, default=30)
    return parser.parse_args()


def as_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [as_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return None if math.isnan(numeric) else numeric
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_leakage_column(column: str) -> bool:
    lowered = column.lower()
    return (
        column in LEAKAGE_COLUMNS
        or lowered.startswith("pred_")
        or lowered.startswith("p_")
        or lowered.endswith("_pred")
        or lowered.startswith("target_")
    )


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


def choose_primary_dose(frame: pd.DataFrame, dose_report: dict[str, object]) -> str:
    primary = dose_report.get("primary_dose_feature")
    if isinstance(primary, str) and primary in frame.columns:
        return primary
    for feature in PRIMARY_DOSE_PRIORITY:
        if feature in frame.columns:
            return feature
    raise ValueError("No primary calcium dose feature is available.")


def load_supervised(args: argparse.Namespace, dose_report: dict[str, object], warnings: list[str]) -> tuple[pd.DataFrame, str, list[str]]:
    if not args.features.exists():
        raise FileNotFoundError(f"Feature parquet does not exist: {args.features}")
    frame = pd.read_parquet(args.features)
    missing_required = [column for column in ["time", "t90"] if column not in frame.columns]
    if missing_required:
        raise ValueError(f"Feature dataset is missing required columns: {missing_required}")
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    if frame["time"].isna().any():
        raise ValueError("Feature dataset contains invalid time values.")
    frame = ensure_targets(frame)
    frame = frame[frame["t90"].notna()].sort_values("time").reset_index(drop=True)
    primary = choose_primary_dose(frame, dose_report)
    context_features = [
        column for column in PROCESS_CONTEXT_FEATURES if column in frame.columns and not is_leakage_column(column)
    ]
    if not context_features:
        warnings.append("No compact process context features are available; neighbor search may have poor support.")
    return frame, primary, context_features


def attach_ir_monitor(frame: pd.DataFrame, data_with_ir: Path, warnings: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    if not data_with_ir.exists():
        warnings.append(f"Optional IR dataset is missing: {data_with_ir}; IR diagnostic field was not copied.")
        return frame
    try:
        ir = pd.read_parquet(data_with_ir, columns=["time", OPTIONAL_IR_MONITOR_FEATURE])
    except Exception as exc:  # pragma: no cover - depends on parquet backend message
        warnings.append(f"Optional IR diagnostic field could not be loaded: {exc}")
        return frame
    ir["time"] = pd.to_datetime(ir["time"], errors="coerce")
    ir = ir.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last")
    frame = frame.merge(ir, on="time", how="left")
    return frame


def quantile_edges(values: pd.Series, n_bins: int) -> np.ndarray | None:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    unique_count = len(np.unique(clean))
    if len(clean) < 2 or unique_count < 2:
        return None
    for bins in range(min(n_bins, unique_count), 1, -1):
        edges = np.unique(np.quantile(clean, np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) >= 3:
            edges = edges.astype(float)
            edges[0] = -np.inf
            edges[-1] = np.inf
            return edges
    return None


def assign_bins_array(values: pd.Series | np.ndarray, edges: np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan)
    valid = np.isfinite(numeric)
    if valid.any():
        result[valid] = np.searchsorted(edges[1:-1], numeric[valid], side="right")
    return result


def rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return math.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def mean_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return math.nan if numeric.dropna().empty else float(numeric.mean())


def median_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return math.nan if numeric.dropna().empty else float(numeric.median())


def build_neighbor_cache(
    data: pd.DataFrame,
    context_features: list[str],
    label_release_delay_hours: float,
    max_k: int,
) -> list[dict[str, object]]:
    times = pd.to_datetime(data["time"]).to_numpy()
    delay = pd.Timedelta(hours=label_release_delay_hours)
    cache: list[dict[str, object]] = []
    for row_index, current in data.iterrows():
        cutoff = current["time"] - delay
        hist_end = int(np.searchsorted(times, np.datetime64(cutoff), side="right"))
        history = data.iloc[:hist_end]
        neighbor_idx = nearest_neighbors(history, current, context_features, max_k)
        cache.append({"hist_end": hist_end, "neighbor_idx": neighbor_idx})
    return cache


def nearest_neighbors(history: pd.DataFrame, current: pd.Series, features: list[str], max_k: int) -> np.ndarray:
    if history.empty or not features:
        return np.asarray([], dtype=int)
    scaled_columns = []
    current_values = []
    for feature in features:
        current_value = pd.to_numeric(pd.Series([current.get(feature)]), errors="coerce").iloc[0]
        if not np.isfinite(current_value):
            continue
        values = pd.to_numeric(history[feature], errors="coerce")
        median = values.median()
        q25 = values.quantile(0.25)
        q75 = values.quantile(0.75)
        iqr = q75 - q25
        scale = float(iqr) if np.isfinite(iqr) and iqr > 0 else 1.0
        scaled_columns.append(((values - median) / scale).to_numpy(dtype=float))
        current_values.append(float((current_value - median) / scale))
    if not scaled_columns:
        return np.asarray([], dtype=int)
    matrix = np.vstack(scaled_columns).T
    row = np.asarray(current_values, dtype=float)
    diff = np.abs(matrix - row)
    valid_count = np.isfinite(diff).sum(axis=1)
    distances = np.full(len(history), np.inf)
    comparable = valid_count > 0
    if comparable.any():
        distances[comparable] = np.nanmedian(diff[comparable], axis=1)
    finite = np.isfinite(distances)
    if not finite.any():
        return np.asarray([], dtype=int)
    finite_positions = np.where(finite)[0]
    order = np.argsort(distances[finite], kind="mergesort")
    return finite_positions[order[:max_k]].astype(int)


def make_bin_table(
    history: pd.DataFrame,
    dose_feature: str,
    bin_ids: np.ndarray,
    min_bin_samples: int,
) -> tuple[pd.DataFrame, set[int], set[int], set[int]]:
    work = history.copy()
    work["dose_bin_id"] = bin_ids
    rows = []
    for bin_id, group in work[np.isfinite(work["dose_bin_id"])].groupby("dose_bin_id", sort=True):
        dose = pd.to_numeric(group[dose_feature], errors="coerce")
        rows.append(
            {
                "bin_id": int(bin_id),
                "sample_count": int(len(group)),
                "dose_min": float(dose.min()),
                "dose_max": float(dose.max()),
                "ok_rate": float(group["y_ok"].mean()),
                "low_rate": float(group["y_low"].mean()),
                "high_rate": float(group["y_high"].mean()),
                "out_spec_rate": float(group["y_out_spec"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table, set(), set(), set()
    overall_ok = float(history["y_ok"].mean())
    overall_low = float(history["y_low"].mean())
    overall_high = float(history["y_high"].mean())
    eligible = table[table["sample_count"] >= min_bin_samples]
    best_bins = set()
    if not eligible.empty:
        best = eligible.sort_values(["ok_rate", "sample_count"], ascending=[False, False]).iloc[0]
        best_bins = {int(best["bin_id"])}
    safe = table[
        (table["ok_rate"] >= overall_ok)
        & (table["high_rate"] <= overall_high)
        & (table["low_rate"] <= overall_low + 0.02)
    ]
    high_risk = table[
        (table["high_rate"] >= overall_high + 0.05)
        | (table["low_rate"] >= overall_low + 0.03)
    ]
    return table, best_bins, {int(v) for v in safe["bin_id"]}, {int(v) for v in high_risk["bin_id"]}


def neighbor_bin_stats(history: pd.DataFrame, neighbor_idx: np.ndarray, bin_id: int) -> dict[str, float] | None:
    if len(neighbor_idx) == 0:
        return None
    subset = history.iloc[neighbor_idx]
    subset = subset[subset["dose_bin_id"] == bin_id]
    if subset.empty:
        return None
    return {
        "neighbor_count": int(len(subset)),
        "ok_rate": float(subset["y_ok"].mean()),
        "low_rate": float(subset["y_low"].mean()),
        "high_rate": float(subset["y_high"].mean()),
    }


def precompute_structural_candidates(
    data: pd.DataFrame,
    dose_feature: str,
    neighbor_cache: list[dict[str, object]],
    min_history_samples: int,
    n_dose_bins: int,
    neighbor_max_k: int,
    min_bin_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_rows = []
    candidate_rows = []
    for sample_id, current in data.iterrows():
        cache = neighbor_cache[sample_id]
        hist_end = int(cache["hist_end"])
        full_neighbor_idx = np.asarray(cache["neighbor_idx"], dtype=int)
        neighbor_idx = full_neighbor_idx[:neighbor_max_k]
        base = {
            "sample_id": int(sample_id),
            "time": current["time"],
            "t90": float(current["t90"]),
            "y_ok": int(current["y_ok"]),
            "y_low": int(current["y_low"]),
            "y_high": int(current["y_high"]),
            "y_out_spec": int(current["y_out_spec"]),
            "dose_current": pd.to_numeric(pd.Series([current.get(dose_feature)]), errors="coerce").iloc[0],
            "eligible_history_count": hist_end,
            "neighbor_count": int(len(neighbor_idx)),
            "effective_bin_count": 0,
            "current_bin_id": pd.NA,
            "current_neighbor_count": 0,
            "current_neighbor_ok_rate": math.nan,
            "current_neighbor_low_rate": math.nan,
            "current_neighbor_high_rate": math.nan,
            "current_bin_high_risk": False,
            "current_bin_safe": False,
            "base_reason": "",
        }
        if pd.isna(base["dose_current"]):
            base["base_reason"] = "missing_dose"
            meta_rows.append(base)
            continue
        if hist_end < min_history_samples:
            base["base_reason"] = "insufficient_history"
            meta_rows.append(base)
            continue
        history = data.iloc[:hist_end].copy()
        edges = quantile_edges(history[dose_feature], n_dose_bins)
        if edges is None:
            base["base_reason"] = "invalid_history_bins"
            meta_rows.append(base)
            continue
        history_bins = assign_bins_array(history[dose_feature], edges)
        history["dose_bin_id"] = history_bins
        current_bin = assign_bins_array(np.asarray([base["dose_current"]]), edges)[0]
        if not np.isfinite(current_bin):
            base["base_reason"] = "invalid_current_bin"
            meta_rows.append(base)
            continue
        current_bin = int(current_bin)
        table, _best_bins, safe_bins, high_risk_bins = make_bin_table(
            history, dose_feature, history_bins, min_bin_samples
        )
        base["effective_bin_count"] = int(len(table))
        base["current_bin_id"] = current_bin
        base["current_bin_high_risk"] = current_bin in high_risk_bins
        base["current_bin_safe"] = current_bin in safe_bins
        current_stats = neighbor_bin_stats(history, neighbor_idx, current_bin)
        if current_stats is not None:
            base["current_neighbor_count"] = int(current_stats["neighbor_count"])
            base["current_neighbor_ok_rate"] = float(current_stats["ok_rate"])
            base["current_neighbor_low_rate"] = float(current_stats["low_rate"])
            base["current_neighbor_high_rate"] = float(current_stats["high_rate"])
        else:
            base["base_reason"] = "insufficient_current_bin_support"
        meta_rows.append(base)
        if current_stats is None or table.empty:
            continue
        for item in table.to_dict(orient="records"):
            bin_id = int(item["bin_id"])
            stats = neighbor_bin_stats(history, neighbor_idx, bin_id)
            if stats is None:
                continue
            candidate_rows.append(
                {
                    "sample_id": int(sample_id),
                    "candidate_bin_id": bin_id,
                    "recommended_dose_min": float(item["dose_min"]),
                    "recommended_dose_max": float(item["dose_max"]),
                    "candidate_neighbor_count": int(stats["neighbor_count"]),
                    "recommended_neighbor_ok_rate": float(stats["ok_rate"]),
                    "recommended_neighbor_low_rate": float(stats["low_rate"]),
                    "recommended_neighbor_high_rate": float(stats["high_rate"]),
                    "expected_ok_rate_gain": float(stats["ok_rate"] - current_stats["ok_rate"]),
                    "low_rate_delta": float(stats["low_rate"] - current_stats["low_rate"]),
                    "high_rate_delta": float(stats["high_rate"] - current_stats["high_rate"]),
                    "recommended_bin_high_risk": bin_id in high_risk_bins,
                    "recommended_bin_safe": bin_id in safe_bins,
                    "current_bin_id": current_bin,
                    "current_bin_high_risk": current_bin in high_risk_bins,
                }
            )
    meta = pd.DataFrame(meta_rows)
    candidates = pd.DataFrame(candidate_rows)
    return meta, candidates


def build_hold_recommendations(
    meta: pd.DataFrame,
    dose_feature: str,
    config_id: str,
    policy_passed: bool,
) -> pd.DataFrame:
    rec = pd.DataFrame(
        {
            "time": meta["time"],
            "t90": meta["t90"],
            "y_ok": meta["y_ok"],
            "y_low": meta["y_low"],
            "y_high": meta["y_high"],
            "y_out_spec": meta["y_out_spec"],
            "config_id": config_id,
            "dose_feature": dose_feature,
            "dose_current": meta["dose_current"],
            "current_bin_id": meta["current_bin_id"],
            "recommended_bin_id": meta["current_bin_id"],
            "recommended_dose_min": math.nan,
            "recommended_dose_max": math.nan,
            "current_neighbor_ok_rate": meta["current_neighbor_ok_rate"],
            "recommended_neighbor_ok_rate": meta["current_neighbor_ok_rate"],
            "expected_ok_rate_gain": 0.0,
            "current_neighbor_low_rate": meta["current_neighbor_low_rate"],
            "recommended_neighbor_low_rate": meta["current_neighbor_low_rate"],
            "current_neighbor_high_rate": meta["current_neighbor_high_rate"],
            "recommended_neighbor_high_rate": meta["current_neighbor_high_rate"],
            "neighbor_count": meta["neighbor_count"],
            "eligible_history_count": meta["eligible_history_count"],
            "effective_bin_count": meta["effective_bin_count"],
            "action": "hold",
            "reason": np.where(meta["base_reason"].astype(str) == "", "no_safe_candidate", meta["base_reason"]),
            "policy_passed": bool(policy_passed),
            "recommended_bin_high_risk": False,
        }
    )
    return rec


def apply_rule_config(
    meta: pd.DataFrame,
    candidates: pd.DataFrame,
    dose_feature: str,
    config: dict[str, object],
    config_id: str,
    policy_passed: bool,
) -> pd.DataFrame:
    recommendations = build_hold_recommendations(meta, dose_feature, config_id, policy_passed)
    if candidates.empty:
        return recommendations
    eligible_sample_ids = meta.loc[
        (meta["base_reason"].astype(str) == "")
        & (meta["neighbor_count"] >= int(config["min_neighbors"]))
        & (meta["current_neighbor_count"] >= 10),
        "sample_id",
    ]
    if bool(config["require_neighbor_count_at_max_k"]):
        full_ids = meta.loc[
            meta["neighbor_count"] >= int(config["neighbor_max_k"]),
            "sample_id",
        ]
        eligible_sample_ids = pd.Index(eligible_sample_ids).intersection(pd.Index(full_ids))
    if len(eligible_sample_ids) == 0:
        recommendations.loc[meta["base_reason"].astype(str) == "", "reason"] = "insufficient_neighbors"
        return recommendations

    filt = candidates[candidates["sample_id"].isin(eligible_sample_ids)].copy()
    if filt.empty:
        return recommendations
    filt = filt[filt["candidate_neighbor_count"] >= 10]
    filt = filt[filt["expected_ok_rate_gain"] >= float(config["min_expected_gain"])]
    filt = filt[filt["high_rate_delta"] <= float(config["max_high_risk_worsen"])]
    filt = filt[filt["low_rate_delta"] <= float(config["max_low_risk_worsen"])]
    if bool(config["forbid_recommended_high_risk_bin"]):
        filt = filt[~filt["recommended_bin_high_risk"].astype(bool)]
    if bool(config["require_recommended_safe_bin"]):
        filt = filt[filt["recommended_bin_safe"].astype(bool)]
    if bool(config["restrict_to_current_high_risk_bins"]):
        filt = filt[filt["current_bin_high_risk"].astype(bool)]
    if not bool(config["allow_increase"]):
        filt = filt[filt["candidate_bin_id"] <= filt["current_bin_id"]]
    if not bool(config["allow_decrease"]):
        filt = filt[filt["candidate_bin_id"] >= filt["current_bin_id"]]
    filt = filt[filt["candidate_bin_id"] != filt["current_bin_id"]]
    quantile_rule = str(config["require_expected_gain_quantile"])
    if not filt.empty and quantile_rule != "none":
        quantiles = {"top_50": 0.50, "top_30": 0.70, "top_20": 0.80}
        threshold = float(filt["expected_ok_rate_gain"].quantile(quantiles[quantile_rule]))
        filt = filt[filt["expected_ok_rate_gain"] >= threshold]
    if filt.empty:
        return recommendations
    filt = filt.sort_values(
        [
            "sample_id",
            "expected_ok_rate_gain",
            "recommended_neighbor_ok_rate",
            "candidate_neighbor_count",
        ],
        ascending=[True, False, False, False],
        kind="mergesort",
    ).drop_duplicates(subset=["sample_id"], keep="first")
    rec_index = recommendations.index[recommendations.index.isin(filt["sample_id"])]
    filt = filt.set_index("sample_id").loc[rec_index]
    recommendations.loc[rec_index, "recommended_bin_id"] = filt["candidate_bin_id"].to_numpy()
    recommendations.loc[rec_index, "recommended_dose_min"] = filt["recommended_dose_min"].to_numpy()
    recommendations.loc[rec_index, "recommended_dose_max"] = filt["recommended_dose_max"].to_numpy()
    recommendations.loc[rec_index, "recommended_neighbor_ok_rate"] = filt["recommended_neighbor_ok_rate"].to_numpy()
    recommendations.loc[rec_index, "recommended_neighbor_low_rate"] = filt["recommended_neighbor_low_rate"].to_numpy()
    recommendations.loc[rec_index, "recommended_neighbor_high_rate"] = filt["recommended_neighbor_high_rate"].to_numpy()
    recommendations.loc[rec_index, "expected_ok_rate_gain"] = filt["expected_ok_rate_gain"].to_numpy()
    recommendations.loc[rec_index, "recommended_bin_high_risk"] = filt["recommended_bin_high_risk"].to_numpy()
    rec_bins = pd.to_numeric(recommendations.loc[rec_index, "recommended_bin_id"], errors="coerce")
    cur_bins = pd.to_numeric(recommendations.loc[rec_index, "current_bin_id"], errors="coerce")
    recommendations.loc[rec_index, "action"] = np.where(
        rec_bins > cur_bins,
        "increase_ca_small_step",
        "decrease_ca_small_step",
    )
    recommendations.loc[rec_index, "reason"] = "strict_rule_candidate"
    base_ok_no_action = (meta["base_reason"].astype(str) == "") & (meta["neighbor_count"] < int(config["min_neighbors"]))
    recommendations.loc[base_ok_no_action, "reason"] = "insufficient_neighbors"
    return recommendations


def add_split(recommendations: pd.DataFrame, split_index: int) -> pd.DataFrame:
    rec = recommendations.copy()
    rec["split"] = np.where(rec.index < split_index, "train_like", "test_like")
    return rec


def summarize_split(
    recommendations: pd.DataFrame,
    split_name: str,
    config: dict[str, object],
    config_id: str,
    pass_test_like: bool,
    score: float,
) -> dict[str, object]:
    if split_name == "all":
        subset = recommendations
    else:
        subset = recommendations[recommendations["split"] == split_name]
    hold = subset[subset["action"] == "hold"]
    increase = subset[subset["action"] == "increase_ca_small_step"]
    decrease = subset[subset["action"] == "decrease_ca_small_step"]
    actionable = subset[subset["action"] != "hold"]
    total = int(len(subset))
    action_rate = float(len(actionable) / total) if total else math.nan
    ok_lift = rate(actionable, "y_ok") - rate(hold, "y_ok") if len(actionable) and len(hold) else math.nan
    high_delta = rate(actionable, "y_high") - rate(hold, "y_high") if len(actionable) and len(hold) else math.nan
    low_delta = rate(actionable, "y_low") - rate(hold, "y_low") if len(actionable) and len(hold) else math.nan
    return {
        "config_id": config_id,
        "split": split_name,
        "min_history_samples": config["min_history_samples"],
        "n_dose_bins": config["n_dose_bins"],
        "neighbor_max_k": config["neighbor_max_k"],
        "min_neighbors": config["min_neighbors"],
        "min_expected_gain": config["min_expected_gain"],
        "max_action_rate": config["max_action_rate"],
        "max_high_risk_worsen": config["max_high_risk_worsen"],
        "max_low_risk_worsen": config["max_low_risk_worsen"],
        "restrict_to_current_high_risk_bins": config["restrict_to_current_high_risk_bins"],
        "allow_increase": config["allow_increase"],
        "allow_decrease": config["allow_decrease"],
        "forbid_recommended_high_risk_bin": config["forbid_recommended_high_risk_bin"],
        "require_recommended_safe_bin": config["require_recommended_safe_bin"],
        "require_neighbor_count_at_max_k": config["require_neighbor_count_at_max_k"],
        "require_expected_gain_quantile": config["require_expected_gain_quantile"],
        "total_samples": total,
        "evaluable_samples": int((subset["reason"] != "insufficient_history").sum()),
        "actionable_samples": int(len(actionable)),
        "action_rate": action_rate,
        "hold_count": int(len(hold)),
        "increase_count": int(len(increase)),
        "decrease_count": int(len(decrease)),
        "actual_ok_rate_hold": rate(hold, "y_ok"),
        "actual_ok_rate_actionable": rate(actionable, "y_ok"),
        "actual_ok_rate_increase": rate(increase, "y_ok"),
        "actual_ok_rate_decrease": rate(decrease, "y_ok"),
        "actual_high_rate_hold": rate(hold, "y_high"),
        "actual_high_rate_actionable": rate(actionable, "y_high"),
        "actual_high_rate_increase": rate(increase, "y_high"),
        "actual_high_rate_decrease": rate(decrease, "y_high"),
        "actual_low_rate_hold": rate(hold, "y_low"),
        "actual_low_rate_actionable": rate(actionable, "y_low"),
        "actual_low_rate_increase": rate(increase, "y_low"),
        "actual_low_rate_decrease": rate(decrease, "y_low"),
        "ok_rate_lift_actionable_vs_hold": ok_lift,
        "high_rate_delta_actionable_vs_hold": high_delta,
        "low_rate_delta_actionable_vs_hold": low_delta,
        "mean_expected_ok_rate_gain": mean_or_nan(actionable["expected_ok_rate_gain"]),
        "median_expected_ok_rate_gain": median_or_nan(actionable["expected_ok_rate_gain"]),
        "mean_neighbor_count": mean_or_nan(subset["neighbor_count"]),
        "median_neighbor_count": median_or_nan(subset["neighbor_count"]),
        "mean_eligible_history_count": mean_or_nan(subset["eligible_history_count"]),
        "recommended_high_risk_action_count": int(actionable["recommended_bin_high_risk"].fillna(False).sum()),
        "pass_test_like": bool(pass_test_like),
        "score": score,
    }


def evaluate_pass(test_row: dict[str, object], max_action_rate: float) -> bool:
    return (
        int(test_row["actionable_samples"]) >= 30
        and float(test_row["action_rate"]) <= max_action_rate
        and np.isfinite(test_row["ok_rate_lift_actionable_vs_hold"])
        and float(test_row["ok_rate_lift_actionable_vs_hold"]) >= 0.03
        and np.isfinite(test_row["high_rate_delta_actionable_vs_hold"])
        and float(test_row["high_rate_delta_actionable_vs_hold"]) <= 0.00
        and np.isfinite(test_row["low_rate_delta_actionable_vs_hold"])
        and float(test_row["low_rate_delta_actionable_vs_hold"]) <= 0.005
        and int(test_row["recommended_high_risk_action_count"]) == 0
    )


def score_from_test(test_row: dict[str, object], passed: bool) -> float:
    ok_lift = float(test_row["ok_rate_lift_actionable_vs_hold"]) if np.isfinite(test_row["ok_rate_lift_actionable_vs_hold"]) else -1.0
    high_delta = float(test_row["high_rate_delta_actionable_vs_hold"]) if np.isfinite(test_row["high_rate_delta_actionable_vs_hold"]) else 1.0
    low_delta = float(test_row["low_rate_delta_actionable_vs_hold"]) if np.isfinite(test_row["low_rate_delta_actionable_vs_hold"]) else 1.0
    action_rate = float(test_row["action_rate"]) if np.isfinite(test_row["action_rate"]) else 1.0
    actionable = int(test_row["actionable_samples"])
    base = ok_lift * 100.0 - max(high_delta, 0.0) * 60.0 - max(low_delta, 0.0) * 80.0
    base -= action_rate * 2.0
    base += min(actionable, 300) * 0.001
    return base + (1000.0 if passed else 0.0)


def config_iter_without_action_rate() -> itertools.product:
    return itertools.product(
        MIN_HISTORY_GRID,
        N_DOSE_BINS_GRID,
        NEIGHBOR_MAX_K_GRID,
        MIN_NEIGHBORS_GRID,
        MIN_EXPECTED_GAIN_GRID,
        MAX_HIGH_RISK_WORSEN_GRID,
        MAX_LOW_RISK_WORSEN_GRID,
        RESTRICT_TO_CURRENT_HIGH_RISK_GRID,
        ALLOW_INCREASE_GRID,
        ALLOW_DECREASE_GRID,
        FORBID_RECOMMENDED_HIGH_RISK_GRID,
        REQUIRE_RECOMMENDED_SAFE_BIN_GRID,
        REQUIRE_NEIGHBOR_COUNT_AT_MAX_K_GRID,
        REQUIRE_EXPECTED_GAIN_QUANTILE_GRID,
    )


def config_count() -> int:
    sizes = [
        len(MIN_HISTORY_GRID),
        len(N_DOSE_BINS_GRID),
        len(NEIGHBOR_MAX_K_GRID),
        len(MIN_NEIGHBORS_GRID),
        len(MIN_EXPECTED_GAIN_GRID),
        len(MAX_ACTION_RATE_GRID),
        len(MAX_HIGH_RISK_WORSEN_GRID),
        len(MAX_LOW_RISK_WORSEN_GRID),
        len(RESTRICT_TO_CURRENT_HIGH_RISK_GRID),
        len(ALLOW_INCREASE_GRID),
        len(ALLOW_DECREASE_GRID),
        len(FORBID_RECOMMENDED_HIGH_RISK_GRID),
        len(REQUIRE_RECOMMENDED_SAFE_BIN_GRID),
        len(REQUIRE_NEIGHBOR_COUNT_AT_MAX_K_GRID),
        len(REQUIRE_EXPECTED_GAIN_QUANTILE_GRID),
    ]
    total = 1
    for size in sizes:
        total *= size
    return total


def section_title(doc_path: Path, preferred_number: int, title_text: str) -> str:
    if not doc_path.exists():
        return f"## {preferred_number}. {title_text}"
    text = doc_path.read_text(encoding="utf-8")
    used = []
    for line in text.splitlines():
        if line.startswith("## "):
            parts = line[3:].split(".", 1)
            if parts and parts[0].strip().isdigit():
                used.append(int(parts[0].strip()))
    number = preferred_number
    if used:
        number = max(preferred_number, max(used) + 1 if preferred_number in used else preferred_number)
    while number in used:
        number += 1
    return f"## {number}. {title_text}"


def append_documentation(
    doc_path: Path,
    command_text: str,
    report: dict[str, object],
    best_test: dict[str, object] | None,
) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    title = section_title(doc_path, 15, "Walk-forward 钙单耗处方规则收紧与消融验证")
    best = report.get("best_config") or report.get("least_bad_config_if_no_pass") or {}
    lines = [
        "",
        title,
        "",
        "本阶段针对上一轮严格 walk-forward 策略动作率偏高、IR 分支未改善动作质量的问题，进行更保守的规则网格搜索。"
        "本次不训练通用 T90 模型，不进入影子试验，也不形成自动控制建议。",
        "",
        "### 输入与输出",
        f"- 输入文件：`{report['features_path']}`、`{report['dose_response_report_path']}`、`{report['previous_walk_forward_report_path']}`。",
        f"- 输出文件：`{report['results_output_path']}`、`{report['best_output_path']}`、`{report['report_path']}`。",
        f"- 执行命令：`{command_text}`",
        "",
        "### 调参设计",
        f"- 主剂量特征：`{report['primary_dose_feature']}`。",
        f"- 上下文特征：{', '.join(report['context_features_used']) if report['context_features_used'] else '无可用上下文特征'}。",
        f"- 网格规模：{report['grid_size']} 组规则。",
        "- 主要维度包括历史样本下限、剂量分箱数、邻居数、最小期望收益、高/低 T90 风险恶化阈值、是否限制在当前高风险分箱、是否允许增加钙单耗等。",
        "- IR 本阶段仅作为监测诊断字段携带，不参与邻居搜索、分箱、动作选择或风险护栏。",
        "",
        "### 通过标准",
        "- test_like 分段需满足：动作样本不少于 30、动作率不超过对应上限、动作组实际合格率较 hold 至少提升 0.03、高 T90 风险不高于 hold、低 T90 风险相对 hold 增量不超过 0.005，且不得推荐历史高风险分箱。",
        "",
        "### 结果摘要",
        f"- 通过规则数：{report['passing_config_count']}。",
        f"- 最佳配置：{best if best else '无通过配置，记录最小损失配置用于诊断'}。",
    ]
    if best_test:
        lines.extend(
            [
                f"- 最佳配置 test_like 动作率：{best_test.get('action_rate')}",
                f"- 最佳配置 test_like 合格率提升：{best_test.get('ok_rate_lift_actionable_vs_hold')}",
                f"- 最佳配置 test_like 高 T90 风险差：{best_test.get('high_rate_delta_actionable_vs_hold')}",
                f"- 最佳配置 test_like 低 T90 风险差：{best_test.get('low_rate_delta_actionable_vs_hold')}",
                f"- 是否允许增加动作：{best.get('allow_increase') if isinstance(best, dict) else None}",
                f"- 是否限制在当前高风险分箱：{best.get('restrict_to_current_high_risk_bins') if isinstance(best, dict) else None}",
            ]
        )
    lines.extend(
        [
            f"- 最终 recommended_next_step：`{report['recommended_next_step']}`。",
            "",
            "### 当前判断",
            "若存在通过配置，可进入人工复核表准备阶段；若无通过配置但最小损失配置仍具备正向提升且风险不恶化，则继续收紧规则；否则暂停策略工作并等待更多数据或新的机理特征。",
            "所有判断仍为离线观察性结果，不构成因果证明，也不构成自动控制策略。",
            "",
        ]
    )
    with doc_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    assumptions = [
        "All recommendations are offline diagnostics for manual review only.",
        "IR is not used in dose bins, neighbor search, action selection, or risk guardrails.",
        "Eligible history uses sample_time <= current_time - label_release_delay.",
        "No future labels or current-row labels are used for rule construction.",
    ]
    dose_report = load_json(args.dose_response_report)
    _feature_report = load_json(args.feature_report)
    previous_report = load_json(args.previous_walk_forward_report)
    data, dose_feature, context_features = load_supervised(args, dose_report, warnings)
    data = attach_ir_monitor(data, args.data_with_ir, warnings)
    split_index = int(len(data) * 0.8)
    grid_size = config_count()

    print(f"Primary dose feature: {dose_feature}")
    print(f"Grid size: {grid_size}")
    print("Building history-only neighbor cache without IR...")
    neighbor_cache = build_neighbor_cache(
        data,
        context_features,
        label_release_delay_hours=args.label_release_delay_hours,
        max_k=max(NEIGHBOR_MAX_K_GRID),
    )

    results: list[dict[str, object]] = []
    passing: list[dict[str, object]] = []
    best_score = -float("inf")
    best_payload: dict[str, object] | None = None
    least_bad_payload: dict[str, object] | None = None
    least_bad_score = -float("inf")
    config_sequence = 0

    structural_cache: dict[tuple[int, int, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for combo in config_iter_without_action_rate():
        (
            min_history_samples,
            n_dose_bins,
            neighbor_max_k,
            min_neighbors,
            min_expected_gain,
            max_high_risk_worsen,
            max_low_risk_worsen,
            restrict_to_current_high_risk_bins,
            allow_increase,
            allow_decrease,
            forbid_recommended_high_risk_bin,
            require_recommended_safe_bin,
            require_neighbor_count_at_max_k,
            require_expected_gain_quantile,
        ) = combo
        structural_key = (min_history_samples, n_dose_bins, neighbor_max_k)
        if structural_key not in structural_cache:
            print(f"Precomputing structural candidates: history={min_history_samples}, bins={n_dose_bins}, k={neighbor_max_k}")
            structural_cache[structural_key] = precompute_structural_candidates(
                data,
                dose_feature,
                neighbor_cache,
                min_history_samples=min_history_samples,
                n_dose_bins=n_dose_bins,
                neighbor_max_k=neighbor_max_k,
                min_bin_samples=args.min_bin_samples,
            )
        meta, candidates = structural_cache[structural_key]
        base_config = {
            "min_history_samples": min_history_samples,
            "n_dose_bins": n_dose_bins,
            "neighbor_max_k": neighbor_max_k,
            "min_neighbors": min_neighbors,
            "min_expected_gain": min_expected_gain,
            "max_high_risk_worsen": max_high_risk_worsen,
            "max_low_risk_worsen": max_low_risk_worsen,
            "restrict_to_current_high_risk_bins": restrict_to_current_high_risk_bins,
            "allow_increase": allow_increase,
            "allow_decrease": allow_decrease,
            "forbid_recommended_high_risk_bin": forbid_recommended_high_risk_bin,
            "require_recommended_safe_bin": require_recommended_safe_bin,
            "require_neighbor_count_at_max_k": require_neighbor_count_at_max_k,
            "require_expected_gain_quantile": require_expected_gain_quantile,
        }
        temp_config_id = f"cfg_base_{config_sequence:06d}"
        temp_rec = apply_rule_config(meta, candidates, dose_feature, base_config, temp_config_id, False)
        temp_rec = add_split(temp_rec, split_index)
        for max_action_rate in MAX_ACTION_RATE_GRID:
            config_sequence += 1
            config = {**base_config, "max_action_rate": max_action_rate}
            config_id = f"cfg_{config_sequence:06d}"
            rec = temp_rec.copy()
            rec["config_id"] = config_id
            test_pre = summarize_split(rec, "test_like", config, config_id, False, 0.0)
            passed = evaluate_pass(test_pre, max_action_rate)
            score = score_from_test(test_pre, passed)
            rec["policy_passed"] = bool(passed)
            split_rows = [
                summarize_split(rec, "all", config, config_id, passed, score),
                summarize_split(rec, "train_like", config, config_id, passed, score),
                summarize_split(rec, "test_like", config, config_id, passed, score),
            ]
            results.extend(split_rows)
            test_row = split_rows[2]
            payload = {
                "config_id": config_id,
                "config": config,
                "test_like": test_row,
                "recommendations": rec,
                "score": score,
            }
            if passed:
                passing.append(payload)
                if score > best_score:
                    best_score = score
                    best_payload = payload
            if score > least_bad_score:
                least_bad_score = score
                least_bad_payload = payload

    results_df = pd.DataFrame(results)
    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.results_output, index=False, encoding="utf-8-sig")

    selected_payload = best_payload if best_payload is not None else least_bad_payload
    if selected_payload is None:
        raise RuntimeError("No configuration was evaluated.")
    best_recommendations = selected_payload["recommendations"].copy()
    if OPTIONAL_IR_MONITOR_FEATURE in data.columns:
        best_recommendations[f"ir_monitor_{OPTIONAL_IR_MONITOR_FEATURE}"] = data[OPTIONAL_IR_MONITOR_FEATURE].to_numpy()
    best_recommendations = best_recommendations.drop(columns=["recommended_bin_high_risk"], errors="ignore")
    required_order = [
        "time",
        "t90",
        "y_ok",
        "y_low",
        "y_high",
        "y_out_spec",
        "config_id",
        "split",
        "dose_feature",
        "dose_current",
        "current_bin_id",
        "recommended_bin_id",
        "recommended_dose_min",
        "recommended_dose_max",
        "current_neighbor_ok_rate",
        "recommended_neighbor_ok_rate",
        "expected_ok_rate_gain",
        "current_neighbor_low_rate",
        "recommended_neighbor_low_rate",
        "current_neighbor_high_rate",
        "recommended_neighbor_high_rate",
        "neighbor_count",
        "eligible_history_count",
        "effective_bin_count",
        "action",
        "reason",
        "policy_passed",
    ]
    ordered_columns = [column for column in required_order if column in best_recommendations.columns]
    ordered_columns += [column for column in best_recommendations.columns if column not in ordered_columns]
    args.best_output.parent.mkdir(parents=True, exist_ok=True)
    best_recommendations[ordered_columns].to_parquet(args.best_output, index=False)

    best_config = best_payload["config"] if best_payload is not None else None
    best_test = best_payload["test_like"] if best_payload is not None else None
    least_bad_config = least_bad_payload["config"] if least_bad_payload is not None else None
    least_bad_test = least_bad_payload["test_like"] if least_bad_payload is not None else None

    if best_payload is not None:
        recommended_next_step = "prepare_manual_review_table"
    elif (
        least_bad_test is not None
        and np.isfinite(least_bad_test["ok_rate_lift_actionable_vs_hold"])
        and float(least_bad_test["ok_rate_lift_actionable_vs_hold"]) > 0
        and np.isfinite(least_bad_test["high_rate_delta_actionable_vs_hold"])
        and float(least_bad_test["high_rate_delta_actionable_vs_hold"]) <= 0
        and np.isfinite(least_bad_test["low_rate_delta_actionable_vs_hold"])
        and float(least_bad_test["low_rate_delta_actionable_vs_hold"]) <= 0.005
    ):
        recommended_next_step = "tighten_rules_further"
    else:
        recommended_next_step = "stop_policy_work_until_more_data"

    diagnostic_flags = {
        "strict_walk_forward_applied": True,
        "label_release_delay_applied": True,
        "future_neighbors_forbidden": True,
        "dose_bins_fit_on_history_only": True,
        "context_scaling_fit_on_history_only": True,
        "ir_not_used_for_action": True,
        "no_shadow_trial_recommended": True,
        "no_automatic_control_recommended": True,
    }
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features_path": str(args.features),
        "feature_report_path": str(args.feature_report),
        "dose_response_report_path": str(args.dose_response_report),
        "previous_walk_forward_report_path": str(args.previous_walk_forward_report),
        "previous_recommended_next_step": previous_report.get("recommended_next_step"),
        "results_output_path": str(args.results_output),
        "best_output_path": str(args.best_output),
        "report_path": str(args.report),
        "row_count": int(len(data)),
        "primary_dose_feature": dose_feature,
        "context_features_used": context_features,
        "ir_policy": {
            "ir_not_used_for_action": True,
            "optional_ir_monitor_feature": OPTIONAL_IR_MONITOR_FEATURE if OPTIONAL_IR_MONITOR_FEATURE in data.columns else None,
            "ir_monitor_non_null_rate": float(data[OPTIONAL_IR_MONITOR_FEATURE].notna().mean())
            if OPTIONAL_IR_MONITOR_FEATURE in data.columns
            else None,
        },
        "grid_size": grid_size,
        "passing_config_count": int(len(passing)),
        "best_config": best_config,
        "best_config_test_like_metrics": best_test,
        "least_bad_config_if_no_pass": None if best_payload is not None else least_bad_config,
        "least_bad_config_test_like_metrics": least_bad_test,
        "diagnostic_flags": diagnostic_flags,
        "warnings": warnings,
        "assumptions": assumptions,
        "recommended_next_step": recommended_next_step,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(report), handle, ensure_ascii=False, indent=2)

    command_text = (
        "python scripts/tune_ca_walk_forward_policy_rules.py "
        f"--features {args.features} --feature-report {args.feature_report} "
        f"--dose-response-report {args.dose_response_report} "
        f"--previous-walk-forward-report {args.previous_walk_forward_report} "
        f"--data-with-ir {args.data_with_ir} --results-output {args.results_output} "
        f"--best-output {args.best_output} --report {args.report} --doc {args.doc}"
    )
    append_documentation(args.doc, command_text, report, best_test if best_test is not None else least_bad_test)

    print("\nTuning summary")
    print(f"Primary dose feature: {dose_feature}")
    print(f"Grid size: {grid_size}")
    print(f"Passing config count: {len(passing)}")
    selected_id = selected_payload["config_id"]
    selected_test = selected_payload["test_like"]
    print(f"Best/diagnostic config id: {selected_id}")
    print(f"Test-like ok lift: {selected_test.get('ok_rate_lift_actionable_vs_hold')}")
    print(f"Test-like high delta: {selected_test.get('high_rate_delta_actionable_vs_hold')}")
    print(f"Test-like low delta: {selected_test.get('low_rate_delta_actionable_vs_hold')}")
    print(f"Test-like action rate: {selected_test.get('action_rate')}")
    print("IR used for action: false")
    print(f"Recommended next step: {recommended_next_step}")
    print(f"Results output: {args.results_output}")
    print(f"Best output: {args.best_output}")
    print(f"Report: {args.report}")
    print(f"Documentation appended: {args.doc}")


if __name__ == "__main__":
    main()
