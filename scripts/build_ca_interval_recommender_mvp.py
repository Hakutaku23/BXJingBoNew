from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


T90_LOW = 8.20
T90_HIGH = 8.70
PRIMARY_DOSE_FEATURE = "ca_per_rubber_flow_win_60_mean"
RUBBER_FLOW_FEATURE = "rubber_flow_2_win_60_mean"
IR_LAG_FEATURE = "output_ir_corrected_offset_20_win_15_std"
ACCEPTED_STATUSES = {"accept_for_manual_case_review"}
ACCEPTED_GRADES = {"A", "B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate MVP calcium-consumption interval recommender.")
    parser.add_argument("--features", type=Path, default=Path("data/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--feature-report", type=Path, default=Path("data/t90_ca_feature_report.json"))
    parser.add_argument("--rules", type=Path, default=Path("data/ca_regime_calcium_band_rules_ir_lag.csv"))
    parser.add_argument("--rule-validation", type=Path, default=Path("data/ca_regime_calcium_band_rule_validation_ir_lag.csv"))
    parser.add_argument("--manual-review-candidates", type=Path, default=Path("data/ca_regime_calcium_band_manual_review_candidates.csv"))
    parser.add_argument("--rules-report", type=Path, default=Path("data/ca_regime_calcium_band_rules_ir_lag_report.json"))
    parser.add_argument("--data-with-ir", type=Path, default=Path("data/data_clean_with_ir.parquet"))
    parser.add_argument("--artifact-output", type=Path, default=Path("models/ca_interval_recommender/rule_artifact.json"))
    parser.add_argument("--replay-output", type=Path, default=Path("data/ca_interval_recommender_replay.parquet"))
    parser.add_argument("--oracle-output", type=Path, default=Path("data/ca_interval_recommender_validation_oracle.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/ca_interval_recommender_metrics.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/ca_interval_recommender_report.json"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--oracle-min-bin-samples", type=int, default=10)
    parser.add_argument("--n-oracle-bins", type=int, default=5)
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
        val = float(value)
        return None if math.isnan(val) else val
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_targets(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["t90"] = pd.to_numeric(data["t90"], errors="coerce")
    if "y_ok" not in data.columns:
        data["y_ok"] = ((data["t90"] >= T90_LOW) & (data["t90"] <= T90_HIGH)).astype(int)
    if "y_low" not in data.columns:
        data["y_low"] = (data["t90"] < T90_LOW).astype(int)
    if "y_high" not in data.columns:
        data["y_high"] = (data["t90"] > T90_HIGH).astype(int)
    if "y_out_spec" not in data.columns:
        data["y_out_spec"] = ((data["t90"] < T90_LOW) | (data["t90"] > T90_HIGH)).astype(int)
    return data


def rolling_std_feature(values: pd.Series, window: str = "15min") -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rolling(window, min_periods=2).std()


def derive_ir_lag_feature(samples: pd.DataFrame, data_with_ir_path: Path, warnings: list[str]) -> pd.DataFrame:
    if IR_LAG_FEATURE in samples.columns:
        return samples
    result = samples.copy()
    if not data_with_ir_path.exists():
        warnings.append(f"data-with-ir is missing: {data_with_ir_path}; continuing without IR-lag metadata.")
        result[IR_LAG_FEATURE] = np.nan
        return result
    columns = pd.read_parquet(data_with_ir_path, columns=None).columns.tolist()
    if IR_LAG_FEATURE in columns:
        ir = pd.read_parquet(data_with_ir_path, columns=["time", IR_LAG_FEATURE])
        ir["time"] = pd.to_datetime(ir["time"], errors="coerce")
        ir = ir.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last")
        return result.merge(ir, on="time", how="left")
    if "output_ir_corrected" not in columns:
        warnings.append("data-with-ir does not contain output_ir_corrected; continuing without IR-lag metadata.")
        result[IR_LAG_FEATURE] = np.nan
        return result
    ir = pd.read_parquet(data_with_ir_path, columns=["time", "output_ir_corrected"])
    ir["time"] = pd.to_datetime(ir["time"], errors="coerce")
    ir = ir.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last").sort_values("time")
    indexed = ir.set_index("time")
    indexed[IR_LAG_FEATURE] = rolling_std_feature(indexed["output_ir_corrected"], "15min")
    lookup = result[["time"]].copy()
    lookup["ir_lookup_time"] = lookup["time"] - pd.to_timedelta(20, unit="m")
    merged = lookup.merge(indexed[[IR_LAG_FEATURE]].reset_index(), left_on="ir_lookup_time", right_on="time", how="left", suffixes=("", "_ir"))
    result[IR_LAG_FEATURE] = pd.to_numeric(merged[IR_LAG_FEATURE], errors="coerce")
    return result


def load_samples(args: argparse.Namespace, warnings: list[str]) -> pd.DataFrame:
    if not args.features.exists():
        raise FileNotFoundError(f"Feature parquet does not exist: {args.features}")
    data = pd.read_parquet(args.features)
    for required in ["time", "t90", PRIMARY_DOSE_FEATURE]:
        if required not in data.columns:
            raise ValueError(f"Feature dataset is missing required column: {required}")
    data = data.copy()
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    if data["time"].isna().any():
        raise ValueError("Feature dataset contains invalid time values.")
    data = ensure_targets(data)
    data = data[data["t90"].notna()].sort_values("time").reset_index(drop=True)
    data = derive_ir_lag_feature(data, args.data_with_ir, warnings)
    return data


def load_rules(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Rules table does not exist: {path}")
    rules = pd.read_csv(path)
    required = [
        "rule_id",
        "regime_feature",
        "regime_bin",
        "recommended_dose_min",
        "recommended_dose_max",
        "rule_grade",
        "rule_status",
    ]
    missing = [column for column in required if column not in rules.columns]
    if missing:
        raise ValueError(f"Rules table is missing required columns: {missing}")
    return rules


def truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def filter_accepted_rules(rules: pd.DataFrame) -> pd.DataFrame:
    accepted = rules[
        rules["rule_status"].isin(ACCEPTED_STATUSES)
        & rules["rule_grade"].isin(ACCEPTED_GRADES)
    ].copy()
    if "time_stable" in accepted.columns:
        accepted = accepted[accepted["time_stable"].map(truthy)]
    accepted = accepted.reset_index(drop=True)
    return accepted


def tertile_boundaries(train: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    boundaries: dict[str, dict[str, float]] = {}
    for feature in features:
        values = pd.to_numeric(train[feature], errors="coerce").dropna()
        if len(values) < 3 or values.nunique() < 3:
            boundaries[feature] = {"q_low_mid": math.nan, "q_mid_high": math.nan}
            continue
        q1, q2 = np.quantile(values.to_numpy(dtype=float), [1.0 / 3.0, 2.0 / 3.0])
        boundaries[feature] = {"q_low_mid": float(q1), "q_mid_high": float(q2)}
    return boundaries


def assign_regime(value: object, boundary: dict[str, float]) -> str | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(numeric):
        return None
    q1 = boundary.get("q_low_mid")
    q2 = boundary.get("q_mid_high")
    if not np.isfinite(q1) or not np.isfinite(q2):
        return None
    if numeric <= q1:
        return "low"
    if numeric <= q2:
        return "mid"
    return "high"


def rule_priority_key(rule: pd.Series) -> tuple:
    grade_rank = {"A": 0, "B": 1, "C": 2}.get(str(rule.get("rule_grade")), 9)
    time_rank = 0 if truthy(rule.get("time_stable", True)) else 1
    sample = -float(rule.get("sample_count", 0.0) or 0.0)
    ok_lift = -float(rule.get("ok_lift_vs_overall", 0.0) or 0.0)
    high_delta = float(rule.get("high_delta_vs_overall", 0.0) or 0.0)
    return grade_rank, time_rank, sample, ok_lift, high_delta


def confidence_for_rule(rule: pd.Series) -> str:
    grade = str(rule.get("rule_grade"))
    sample_count = float(rule.get("sample_count", 0.0) or 0.0)
    ok_lift = float(rule.get("ok_lift_vs_overall", 0.0) or 0.0)
    high_delta = float(rule.get("high_delta_vs_overall", 0.0) or 0.0)
    low_delta = float(rule.get("low_delta_vs_overall", 0.0) or 0.0)
    if grade == "A" and sample_count >= 100 and ok_lift >= 0.05 and high_delta <= 0 and low_delta <= 0.01:
        return "high"
    if grade in {"A", "B"} and sample_count >= 80 and ok_lift >= 0.03:
        return "medium"
    return "low"


def match_rules(row: pd.Series, rules: pd.DataFrame, boundaries: dict[str, dict[str, float]]) -> pd.DataFrame:
    matched = []
    for _, rule in rules.iterrows():
        feature = str(rule["regime_feature"])
        if feature not in row.index or feature not in boundaries:
            continue
        regime = assign_regime(row.get(feature), boundaries[feature])
        if regime == str(rule["regime_bin"]):
            matched.append(rule)
    if not matched:
        return pd.DataFrame(columns=rules.columns)
    return pd.DataFrame(matched)


def selected_rules(matched: pd.DataFrame) -> pd.DataFrame:
    if matched.empty:
        return matched
    sorted_rules = matched.assign(_key=matched.apply(rule_priority_key, axis=1)).sort_values("_key", kind="mergesort")
    best_grade = sorted_rules.iloc[0]["rule_grade"]
    best_time = truthy(sorted_rules.iloc[0].get("time_stable", True))
    selected = sorted_rules[(sorted_rules["rule_grade"] == best_grade) & (sorted_rules["time_stable"].map(truthy) == best_time)].copy()
    return selected.drop(columns=["_key"], errors="ignore")


def action_hint(current: float, rec_min: float, rec_max: float, status: str) -> str:
    if status != "recommended":
        return "hold_or_manual_check"
    if not np.isfinite(current):
        return "no_recommendation_missing_current_dose"
    if current < rec_min:
        return "increase_to_band"
    if current > rec_max:
        return "decrease_to_band"
    return "hold_in_band"


def build_artifact(rules: pd.DataFrame, boundaries: dict[str, dict[str, float]], warnings: list[str], args: argparse.Namespace) -> dict[str, object]:
    context_features = sorted(rules["regime_feature"].dropna().unique().tolist())
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "primary_dose_feature": PRIMARY_DOSE_FEATURE,
        "context_features": context_features,
        "ir_lag_feature": IR_LAG_FEATURE,
        "calcium_feed_conversion_formula": "recommended_ca_feed = recommended_ca_consumption * rubber_flow_2_win_60_mean when rubber_flow_2_win_60_mean is available",
        "regime_boundary_method": {
            "method": "train_like_tertiles",
            "boundaries": boundaries,
            "note": "Boundaries are fitted on first 80% time-ordered train_like samples only.",
        },
        "accepted_rule_count": int(len(rules)),
        "rules": rules.to_dict(orient="records"),
        "rule_priority_logic": [
            "Prefer grade A over B.",
            "Prefer time_stable=true.",
            "Prefer higher sample_count.",
            "Prefer higher ok_lift_vs_overall.",
            "Prefer lower high_delta_vs_overall.",
            "Aggregate selected matched intervals by median.",
        ],
        "fallback_logic": {
            "no_match": "no_recommendation with action_hint hold_or_manual_check",
            "missing_current_dose": "no_recommendation_missing_current_dose",
            "monitor_only_rules": "loaded for traceability but not used for recommendations",
        },
        "warnings": warnings,
        "assumptions": [
            "The recommender outputs intervals, not fixed setpoints.",
            "The statement is probabilistic and must be reviewed by process engineers.",
            "No DCS writeback, automatic control, or shadow-trial recommendation is made.",
        ],
        "source_paths": {
            "features": str(args.features),
            "rules": str(args.rules),
            "rules_report": str(args.rules_report),
        },
    }


def replay_recommender(data: pd.DataFrame, rules: pd.DataFrame, boundaries: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    split_index = int(len(data) * 0.8)
    for idx, row in data.iterrows():
        matched = match_rules(row, rules, boundaries)
        current = pd.to_numeric(pd.Series([row.get(PRIMARY_DOSE_FEATURE)]), errors="coerce").iloc[0]
        base = {
            "time": row["time"],
            "t90": row["t90"],
            "y_ok": row["y_ok"],
            "y_low": row["y_low"],
            "y_high": row["y_high"],
            "y_out_spec": row["y_out_spec"],
            "split": "train_like" if idx < split_index else "test_like",
            "matched_rule_count": int(len(matched)),
            "matched_rule_ids": ";".join(matched["rule_id"].astype(str).tolist()) if not matched.empty else "",
            "current_ca_consumption": current,
            RUBBER_FLOW_FEATURE: row.get(RUBBER_FLOW_FEATURE, math.nan),
            "ir_lag_context_value": row.get(IR_LAG_FEATURE, math.nan),
            "engineering_review_required": True,
        }
        if matched.empty:
            rows.append(
                {
                    **base,
                    "recommendation_status": "no_recommendation",
                    "action_hint": "hold_or_manual_check",
                    "selected_rule_ids": "",
                    "confidence_level": "none",
                    "recommended_ca_consumption_min": math.nan,
                    "recommended_ca_consumption_max": math.nan,
                    "recommended_ca_consumption_target": math.nan,
                    "evidence_ok_rate": math.nan,
                    "evidence_high_rate": math.nan,
                    "evidence_low_rate": math.nan,
                    "evidence_sample_count": 0,
                }
            )
            continue
        selected = selected_rules(matched)
        rec_min = float(pd.to_numeric(selected["recommended_dose_min"], errors="coerce").median())
        rec_max = float(pd.to_numeric(selected["recommended_dose_max"], errors="coerce").median())
        rec_target = (rec_min + rec_max) / 2.0
        best_rule = selected.assign(_key=selected.apply(rule_priority_key, axis=1)).sort_values("_key", kind="mergesort").iloc[0]
        status = "recommended" if np.isfinite(current) else "no_recommendation_missing_current_dose"
        rows.append(
            {
                **base,
                "recommendation_status": status,
                "action_hint": action_hint(current, rec_min, rec_max, status),
                "selected_rule_ids": ";".join(selected["rule_id"].astype(str).tolist()),
                "confidence_level": confidence_for_rule(best_rule),
                "recommended_ca_consumption_min": rec_min,
                "recommended_ca_consumption_max": rec_max,
                "recommended_ca_consumption_target": rec_target,
                "evidence_ok_rate": float(pd.to_numeric(selected["best_ok_rate"], errors="coerce").median()),
                "evidence_high_rate": float(pd.to_numeric(selected["best_high_rate"], errors="coerce").median()),
                "evidence_low_rate": float(pd.to_numeric(selected["best_low_rate"], errors="coerce").median()),
                "evidence_sample_count": int(pd.to_numeric(selected["sample_count"], errors="coerce").median()),
            }
        )
    replay = pd.DataFrame(rows)
    flow = pd.to_numeric(replay.get(RUBBER_FLOW_FEATURE), errors="coerce")
    replay["recommended_ca_feed_min"] = replay["recommended_ca_consumption_min"] * flow
    replay["recommended_ca_feed_max"] = replay["recommended_ca_consumption_max"] * flow
    replay["recommended_ca_feed_target"] = replay["recommended_ca_consumption_target"] * flow
    return replay


def make_quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    clean = numeric.dropna()
    if len(clean) < 2 or clean.nunique() < 2:
        return result
    for bins in range(min(n_bins, clean.nunique()), 1, -1):
        try:
            labels = pd.qcut(clean, q=bins, labels=False, duplicates="drop")
            if labels.nunique(dropna=True) >= 2:
                result.loc[labels.index] = labels.astype("Int64")
                return result
        except ValueError:
            continue
    ranks = clean.rank(method="first")
    labels = pd.qcut(ranks, q=min(n_bins, len(clean)), labels=False, duplicates="drop")
    result.loc[labels.index] = labels.astype("Int64")
    return result


def oracle_for_rule(
    data: pd.DataFrame,
    replay_row: pd.Series,
    rules_by_id: dict[str, pd.Series],
    boundaries: dict[str, dict[str, float]],
    min_bin_samples: int,
    n_bins: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected_ids = [item for item in str(replay_row.get("selected_rule_ids", "")).split(";") if item]
    if not selected_ids or selected_ids[0] not in rules_by_id:
        return blank_oracle("no_selected_rule"), []
    rule = rules_by_id[selected_ids[0]]
    feature = str(rule["regime_feature"])
    regime_bin = str(rule["regime_bin"])
    test = data.iloc[int(len(data) * 0.8):].copy()
    labels = test[feature].apply(lambda value: assign_regime(value, boundaries[feature]))
    subset = test[labels == regime_bin].copy()
    subset = subset[pd.to_numeric(subset[PRIMARY_DOSE_FEATURE], errors="coerce").notna()]
    if len(subset) < min_bin_samples:
        return blank_oracle("insufficient_regime_support"), []
    subset["dose_bin"] = make_quantile_bins(subset[PRIMARY_DOSE_FEATURE], n_bins)
    bin_rows = []
    for bin_id, group in subset.dropna(subset=["dose_bin"]).groupby("dose_bin", sort=True):
        dose = pd.to_numeric(group[PRIMARY_DOSE_FEATURE], errors="coerce")
        bin_rows.append(
            {
                "rule_id": str(rule["rule_id"]),
                "regime_feature": feature,
                "regime_bin": regime_bin,
                "oracle_bin_id": int(bin_id),
                "sample_count": int(len(group)),
                "ok_rate": float(group["y_ok"].mean()),
                "high_rate": float(group["y_high"].mean()),
                "low_rate": float(group["y_low"].mean()),
                "out_spec_rate": float(group["y_out_spec"].mean()),
                "dose_min": float(dose.min()),
                "dose_max": float(dose.max()),
            }
        )
    bins = pd.DataFrame(bin_rows)
    eligible = bins[bins["sample_count"] >= min_bin_samples].copy()
    if eligible.empty:
        return blank_oracle("insufficient_bin_support"), bin_rows
    best = eligible.sort_values(["ok_rate", "high_rate", "low_rate", "sample_count"], ascending=[False, True, True, False]).iloc[0]
    acceptable = eligible[
        (eligible["ok_rate"] >= float(best["ok_rate"]) - 0.03)
        & (eligible["high_rate"] <= float(best["high_rate"]) + 0.02)
        & (eligible["low_rate"] <= float(best["low_rate"]) + 0.01)
    ]
    current = pd.to_numeric(pd.Series([replay_row.get("current_ca_consumption")]), errors="coerce").iloc[0]
    oracle_action = "hold"
    if np.isfinite(current):
        if current < float(best["dose_min"]):
            oracle_action = "increase_to_band"
        elif current > float(best["dose_max"]):
            oracle_action = "decrease_to_band"
        else:
            oracle_action = "hold_in_band"
    oracle = {
        "oracle_status": "ok",
        "oracle_rule_id": str(rule["rule_id"]),
        "oracle_regime_feature": feature,
        "oracle_regime_bin": regime_bin,
        "oracle_ca_band_min": float(best["dose_min"]),
        "oracle_ca_band_max": float(best["dose_max"]),
        "oracle_ca_target": float((best["dose_min"] + best["dose_max"]) / 2.0),
        "oracle_action": oracle_action,
        "oracle_best_ok_rate": float(best["ok_rate"]),
        "oracle_best_high_rate": float(best["high_rate"]),
        "oracle_best_low_rate": float(best["low_rate"]),
        "oracle_best_sample_count": int(best["sample_count"]),
        "oracle_acceptable_bands": [
            {
                "dose_min": float(item["dose_min"]),
                "dose_max": float(item["dose_max"]),
                "ok_rate": float(item["ok_rate"]),
                "high_rate": float(item["high_rate"]),
                "low_rate": float(item["low_rate"]),
                "sample_count": int(item["sample_count"]),
            }
            for _, item in acceptable.iterrows()
        ],
    }
    return oracle, bin_rows


def blank_oracle(status: str) -> dict[str, object]:
    return {
        "oracle_status": status,
        "oracle_rule_id": "",
        "oracle_regime_feature": "",
        "oracle_regime_bin": "",
        "oracle_ca_band_min": math.nan,
        "oracle_ca_band_max": math.nan,
        "oracle_ca_target": math.nan,
        "oracle_action": "",
        "oracle_best_ok_rate": math.nan,
        "oracle_best_high_rate": math.nan,
        "oracle_best_low_rate": math.nan,
        "oracle_best_sample_count": 0,
        "oracle_acceptable_bands": [],
    }


def overlap_ratio(rec_min: float, rec_max: float, band_min: float, band_max: float) -> float:
    if not all(np.isfinite(v) for v in [rec_min, rec_max, band_min, band_max]) or band_max <= band_min:
        return math.nan
    overlap = max(0.0, min(rec_max, band_max) - max(rec_min, band_min))
    return float(overlap / (band_max - band_min))


def positive_overlap(rec_min: float, rec_max: float, band_min: float, band_max: float) -> bool:
    if not all(np.isfinite(v) for v in [rec_min, rec_max, band_min, band_max]):
        return False
    return min(rec_max, band_max) > max(rec_min, band_min)


def attach_oracle_and_accuracy(
    data: pd.DataFrame,
    replay: pd.DataFrame,
    rules: pd.DataFrame,
    boundaries: dict[str, dict[str, float]],
    min_bin_samples: int,
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules_by_id = {str(row["rule_id"]): row for _, row in rules.iterrows()}
    replay = replay.copy()
    oracle_records = []
    oracle_bin_rows = []
    defaults = {
        "oracle_ca_band_min": math.nan,
        "oracle_ca_band_max": math.nan,
        "oracle_ca_target": math.nan,
        "oracle_action": "",
        "band_hit": pd.NA,
        "relaxed_band_hit": pd.NA,
        "direction_hit": pd.NA,
        "target_hit_3pct": pd.NA,
        "target_hit_5pct": pd.NA,
        "target_hit_10pct": pd.NA,
    }
    for column, value in defaults.items():
        replay[column] = value
    for idx, row in replay[replay["split"].eq("test_like") & replay["recommendation_status"].eq("recommended")].iterrows():
        oracle, bin_rows = oracle_for_rule(data, row, rules_by_id, boundaries, min_bin_samples, n_bins)
        for bin_row in bin_rows:
            bin_row["time"] = row["time"]
            oracle_bin_rows.append(bin_row)
        oracle_record = {"time": row["time"], **oracle}
        oracle_records.append(oracle_record)
        replay.loc[idx, "oracle_ca_band_min"] = oracle["oracle_ca_band_min"]
        replay.loc[idx, "oracle_ca_band_max"] = oracle["oracle_ca_band_max"]
        replay.loc[idx, "oracle_ca_target"] = oracle["oracle_ca_target"]
        replay.loc[idx, "oracle_action"] = oracle["oracle_action"]
        if oracle["oracle_status"] != "ok":
            continue
        rec_min = float(row["recommended_ca_consumption_min"])
        rec_max = float(row["recommended_ca_consumption_max"])
        rec_target = float(row["recommended_ca_consumption_target"])
        best_ratio = overlap_ratio(rec_min, rec_max, oracle["oracle_ca_band_min"], oracle["oracle_ca_band_max"])
        acceptable_hits = [
            overlap_ratio(rec_min, rec_max, band["dose_min"], band["dose_max"])
            for band in oracle["oracle_acceptable_bands"]
        ]
        max_ratio = max([best_ratio] + [ratio for ratio in acceptable_hits if np.isfinite(ratio)])
        any_positive = positive_overlap(rec_min, rec_max, oracle["oracle_ca_band_min"], oracle["oracle_ca_band_max"]) or any(
            positive_overlap(rec_min, rec_max, band["dose_min"], band["dose_max"])
            for band in oracle["oracle_acceptable_bands"]
        )
        relative_error = abs(rec_target - float(oracle["oracle_ca_target"])) / abs(float(oracle["oracle_ca_target"])) if np.isfinite(oracle["oracle_ca_target"]) and oracle["oracle_ca_target"] != 0 else math.nan
        replay.loc[idx, "band_hit"] = bool(max_ratio >= 0.50)
        replay.loc[idx, "relaxed_band_hit"] = bool(any_positive)
        replay.loc[idx, "direction_hit"] = bool(row["action_hint"] == oracle["oracle_action"])
        replay.loc[idx, "target_hit_3pct"] = bool(np.isfinite(relative_error) and relative_error <= 0.03)
        replay.loc[idx, "target_hit_5pct"] = bool(np.isfinite(relative_error) and relative_error <= 0.05)
        replay.loc[idx, "target_hit_10pct"] = bool(np.isfinite(relative_error) and relative_error <= 0.10)
    oracle_table = pd.DataFrame(oracle_records)
    if not oracle_table.empty:
        oracle_table["acceptable_band_count"] = oracle_table["oracle_acceptable_bands"].map(len)
        oracle_table["oracle_acceptable_bands"] = oracle_table["oracle_acceptable_bands"].map(lambda x: json.dumps(as_jsonable(x), ensure_ascii=False))
    return replay, oracle_table


def bool_rate(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return math.nan
    return float(valid.astype(bool).mean())


def metrics_for_subset(name: str, subset: pd.DataFrame, total_reference: pd.DataFrame) -> list[dict[str, object]]:
    recommended = subset[subset["recommendation_status"].eq("recommended")]
    total = len(subset)
    rows = []
    values = {
        "sample_count": total,
        "recommended_sample_count": int(len(recommended)),
        "recommendation_coverage": float(len(recommended) / total) if total else math.nan,
        "no_recommendation_rate": float((subset["recommendation_status"] != "recommended").mean()) if total else math.nan,
        "band_accuracy": bool_rate(recommended["band_hit"]) if len(recommended) else math.nan,
        "relaxed_band_accuracy": bool_rate(recommended["relaxed_band_hit"]) if len(recommended) else math.nan,
        "direction_accuracy": bool_rate(recommended["direction_hit"]) if len(recommended) else math.nan,
        "target_accuracy_3pct": bool_rate(recommended["target_hit_3pct"]) if len(recommended) else math.nan,
        "target_accuracy_5pct": bool_rate(recommended["target_hit_5pct"]) if len(recommended) else math.nan,
        "target_accuracy_10pct": bool_rate(recommended["target_hit_10pct"]) if len(recommended) else math.nan,
        "ok_rate": float(subset["y_ok"].mean()) if total else math.nan,
        "high_rate": float(subset["y_high"].mean()) if total else math.nan,
        "low_rate": float(subset["y_low"].mean()) if total else math.nan,
        "out_spec_rate": float(subset["y_out_spec"].mean()) if total else math.nan,
        "mean_t90": float(subset["t90"].mean()) if total else math.nan,
    }
    for metric, value in values.items():
        rows.append({"group": name, "metric": metric, "value": value})
    return rows


def build_metrics(replay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {"all": replay}
    for split, subset in replay.groupby("split", sort=True):
        groups[split] = subset
    for column in ["recommendation_status", "action_hint", "confidence_level"]:
        for value, subset in replay.groupby(column, dropna=False, sort=True):
            groups[f"{column}={value}"] = subset
    for name, subset in groups.items():
        rows.extend(metrics_for_subset(name, subset, replay))
    return pd.DataFrame(rows)


def metric_lookup(metrics: pd.DataFrame, group: str, metric: str) -> float:
    row = metrics[(metrics["group"] == group) & (metrics["metric"] == metric)]
    if row.empty:
        return math.nan
    return float(row.iloc[0]["value"])


def risk_guardrails(replay: pd.DataFrame) -> dict[str, object]:
    test = replay[replay["split"].eq("test_like")]
    rec = test[test["recommendation_status"].eq("recommended")]
    no_rec = test[test["recommendation_status"].ne("recommended")]
    if rec.empty or no_rec.empty:
        return {
            "recommended_high_rate": float(rec["y_high"].mean()) if len(rec) else None,
            "no_recommendation_high_rate": float(no_rec["y_high"].mean()) if len(no_rec) else None,
            "recommended_low_rate": float(rec["y_low"].mean()) if len(rec) else None,
            "no_recommendation_low_rate": float(no_rec["y_low"].mean()) if len(no_rec) else None,
            "high_guardrail_pass": False,
            "low_guardrail_pass": False,
            "note": "insufficient comparison support",
        }
    rec_high = float(rec["y_high"].mean())
    no_high = float(no_rec["y_high"].mean())
    rec_low = float(rec["y_low"].mean())
    no_low = float(no_rec["y_low"].mean())
    return {
        "recommended_high_rate": rec_high,
        "no_recommendation_high_rate": no_high,
        "recommended_low_rate": rec_low,
        "no_recommendation_low_rate": no_low,
        "high_guardrail_pass": bool(rec_high <= no_high),
        "low_guardrail_pass": bool(rec_low <= no_low + 0.005),
        "note": "T90 rates are guardrails, not recommendation accuracy.",
    }


def decide_mvp(metrics: pd.DataFrame, guardrails: dict[str, object]) -> tuple[str, str]:
    band = metric_lookup(metrics, "test_like", "band_accuracy")
    direction = metric_lookup(metrics, "test_like", "direction_accuracy")
    rec_n = metric_lookup(metrics, "test_like", "recommended_sample_count")
    coverage = metric_lookup(metrics, "test_like", "recommendation_coverage")
    high_ok = bool(guardrails.get("high_guardrail_pass"))
    low_ok = bool(guardrails.get("low_guardrail_pass"))
    if band >= 0.70 and direction >= 0.70 and rec_n >= 30 and coverage >= 0.10 and high_ok and low_ok:
        return "pass_for_deployment_chain", "prepare_monitor_deployment_chain"
    if (band >= 0.70 or direction >= 0.70) and (rec_n >= 10 or coverage >= 0.05):
        return "pass_for_monitor_only_chain", "manual_review_before_deployment_chain"
    if band >= 0.60 or direction >= 0.60:
        return "fail_recommendation_accuracy", "refine_interval_rules"
    return "fail_recommendation_accuracy", "stop_until_more_data"


def section_title(doc_path: Path, preferred: int, title: str) -> str:
    if not doc_path.exists():
        return f"## {preferred}. {title}"
    used = []
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            prefix = line[3:].split(".", 1)[0].strip()
            if prefix.isdigit():
                used.append(int(prefix))
    number = preferred
    while number in used:
        number += 1
    return f"## {number}. {title}"


def append_docs(doc_path: Path, report: dict[str, object]) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    title = section_title(doc_path, 20, "钙单耗区间推荐器 MVP 与验证集推荐准确率评估")
    lines = [
        "",
        title,
        "",
        "本阶段将已通过人工复核候选筛选的分工况钙单耗规则封装为 MVP 区间推荐器，并在验证集上评估推荐准确率。推荐输出为钙单耗区间而非固定值，因为历史关系呈非单调且存在高/低 T90 风险转移。",
        "",
        "### 方法",
        "- 70% 指推荐准确率，不是 T90 合格率。",
        "- 推荐准确率由推荐区间与验证集 oracle 合理钙单耗区间的重叠，以及推荐方向与 oracle 方向是否一致来衡量。",
        "- 工况匹配边界使用 train_like 样本重建 tertile，推理时不使用标签。",
        "- IR-lag 只作为辅助诊断元数据，不作为主规则驱动。",
        "",
        "### 结果",
        f"- artifact_rule_count：{report['artifact_rule_count']}。",
        f"- test_like recommendation coverage：{report['test_like_recommendation_coverage']}。",
        f"- test_like band accuracy：{report['test_like_band_accuracy']}。",
        f"- test_like direction accuracy：{report['test_like_direction_accuracy']}。",
        f"- target accuracy 3%/5%/10%：{report['test_like_target_accuracy_3pct']} / {report['test_like_target_accuracy_5pct']} / {report['test_like_target_accuracy_10pct']}。",
        f"- T90 风险护栏：{report['test_like_risk_guardrails']}。",
        f"- mvp_status：`{report['mvp_status']}`。",
        f"- recommended_next_step：`{report['recommended_next_step']}`。",
        "",
        "### 局限",
        "- 这是离线代理验证，不是因果证明。",
        "- 不执行自动控制，不写入 DCS，不推荐影子试验。",
        "- 后续需要在线监测链路验证和工程人工复核。",
        "",
    ]
    with doc_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    assumptions = [
        "Recommendations are probabilistic interval suggestions under similar process regimes.",
        "T90 ok/high/low rates are risk guardrails, not recommendation-accuracy definitions.",
        "No automatic control, DCS writeback, or shadow-trial recommendation is made.",
        "Monitor-only rules are loaded for traceability but do not drive recommendations.",
    ]
    _feature_report = load_json(args.feature_report)
    _rules_report = load_json(args.rules_report)
    data = load_samples(args, warnings)
    rules_all = load_rules(args.rules)
    accepted_rules = filter_accepted_rules(rules_all)
    if accepted_rules.empty:
        warnings.append("No accepted A/B time-stable rules are available; recommender will produce no recommendations.")
    split_index = int(len(data) * 0.8)
    train = data.iloc[:split_index]
    context_features = sorted(accepted_rules["regime_feature"].dropna().unique().tolist()) if not accepted_rules.empty else []
    boundaries = tertile_boundaries(train, context_features)
    artifact = build_artifact(accepted_rules, boundaries, warnings, args)
    args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    with args.artifact_output.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(artifact), handle, ensure_ascii=False, indent=2)

    replay = replay_recommender(data, accepted_rules, boundaries)
    replay, oracle = attach_oracle_and_accuracy(
        data,
        replay,
        accepted_rules,
        boundaries,
        min_bin_samples=args.oracle_min_bin_samples,
        n_bins=args.n_oracle_bins,
    )
    metrics = build_metrics(replay)
    guardrails = risk_guardrails(replay)
    mvp_status, next_step = decide_mvp(metrics, guardrails)

    args.replay_output.parent.mkdir(parents=True, exist_ok=True)
    replay.to_parquet(args.replay_output, index=False)
    args.oracle_output.parent.mkdir(parents=True, exist_ok=True)
    oracle.to_csv(args.oracle_output, index=False, encoding="utf-8-sig")
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_output, index=False, encoding="utf-8-sig")

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features_path": str(args.features),
        "rules_path": str(args.rules),
        "artifact_output_path": str(args.artifact_output),
        "replay_output_path": str(args.replay_output),
        "oracle_output_path": str(args.oracle_output),
        "metrics_output_path": str(args.metrics_output),
        "primary_dose_feature": PRIMARY_DOSE_FEATURE,
        "artifact_rule_count": int(len(accepted_rules)),
        "replay_row_count": int(len(replay)),
        "test_like_recommended_sample_count": int(metric_lookup(metrics, "test_like", "recommended_sample_count")),
        "test_like_recommendation_coverage": metric_lookup(metrics, "test_like", "recommendation_coverage"),
        "test_like_band_accuracy": metric_lookup(metrics, "test_like", "band_accuracy"),
        "test_like_direction_accuracy": metric_lookup(metrics, "test_like", "direction_accuracy"),
        "test_like_target_accuracy_3pct": metric_lookup(metrics, "test_like", "target_accuracy_3pct"),
        "test_like_target_accuracy_5pct": metric_lookup(metrics, "test_like", "target_accuracy_5pct"),
        "test_like_target_accuracy_10pct": metric_lookup(metrics, "test_like", "target_accuracy_10pct"),
        "test_like_risk_guardrails": guardrails,
        "mvp_status": mvp_status,
        "warnings": warnings,
        "assumptions": assumptions,
        "recommended_next_step": next_step,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(report), handle, ensure_ascii=False, indent=2)
    append_docs(args.doc, report)

    print("Calcium interval recommender MVP summary")
    print(f"Artifact rule count: {len(accepted_rules)}")
    print(f"Test-like recommendation coverage: {report['test_like_recommendation_coverage']}")
    print(f"Test-like band accuracy: {report['test_like_band_accuracy']}")
    print(f"Test-like direction accuracy: {report['test_like_direction_accuracy']}")
    print(f"Target accuracy 3/5/10 pct: {report['test_like_target_accuracy_3pct']} / {report['test_like_target_accuracy_5pct']} / {report['test_like_target_accuracy_10pct']}")
    print(f"Risk guardrails: {guardrails}")
    print(f"MVP status: {mvp_status}")
    print(f"Recommended next step: {next_step}")
    print(f"Documentation appended: {args.doc}")


if __name__ == "__main__":
    main()
