from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEY_CONTEXT_FEATURES = [
    "bromine_feed_win_60_mean",
    "rubber_flow_2_win_60_mean",
    "neutral_alkali_feed_win_60_mean",
    "tank_rubber_conc_win_60_mean",
    "esbo_feed_win_60_mean",
    "r513_temp_win_60_mean",
    "r514_temp_win_60_mean",
    "r510a_temp_win_60_mean",
    "r511a_temp_win_60_mean",
    "r512a_temp_win_60_mean",
]
ACCEPTED_STATUS = "accept_for_manual_case_review"
POSITION_ORDER = ["inside_band", "below_band", "above_band", "missing"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit calcium interval recommendation diversity.")
    parser.add_argument("--visualization-table", type=Path, default=Path("runs/ca_interval_recommendation_visualization_table.csv"))
    parser.add_argument("--replay", type=Path, default=Path("runs/ca_interval_recommender_replay.parquet"))
    parser.add_argument("--rules", type=Path, default=Path("runs/ca_regime_calcium_band_rules_ir_lag.csv"))
    parser.add_argument("--rule-audit", type=Path, default=Path("runs/ca_interval_recommender_rule_audit.csv"))
    parser.add_argument("--artifact", type=Path, default=Path("models/ca_interval_recommender/rule_artifact.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ca_interval_diversity_audit"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
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


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130


def load_json(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_by_name(name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None


def resolve_path(path: Path, *, required: bool, search_roots: list[Path], warnings: list[str]) -> Path | None:
    if path.exists():
        return path
    found = find_by_name(path.name, search_roots)
    if found is not None:
        warnings.append(f"Input {path} not found; using recursive match {found}.")
        return found
    if required:
        raise FileNotFoundError(f"Required input file not found: {path}. Searched recursively under {[str(p) for p in search_roots]}.")
    warnings.append(f"Optional input file not found: {path}.")
    return None


def read_table(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def numeric_series(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")
    return pd.to_numeric(data[column], errors="coerce")


def quantile_stats(series: pd.Series, prefix: str) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            f"{prefix}_min": None,
            f"{prefix}_q25": None,
            f"{prefix}_median": None,
            f"{prefix}_q75": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
        }
    return {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_q25": float(values.quantile(0.25)),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_q75": float(values.quantile(0.75)),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_mean": float(values.mean()),
    }


def derive_interval_position(data: pd.DataFrame) -> pd.Series:
    current = numeric_series(data, "current_ca_consumption")
    rec_min = numeric_series(data, "recommended_ca_consumption_min")
    rec_max = numeric_series(data, "recommended_ca_consumption_max")
    position = pd.Series("missing", index=data.index, dtype="object")
    valid = current.notna() & rec_min.notna() & rec_max.notna()
    position.loc[valid & (current < rec_min)] = "below_band"
    position.loc[valid & (current > rec_max)] = "above_band"
    position.loc[valid & (current >= rec_min) & (current <= rec_max)] = "inside_band"
    return position


def prepare_sample_table(vis: pd.DataFrame, replay: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    if vis.empty and replay.empty:
        raise ValueError("No sample-level table is available. Provide visualization table or replay parquet.")
    if not replay.empty and "split" in replay.columns:
        replay = replay.copy()
        replay["time"] = pd.to_datetime(replay["time"], errors="coerce") if "time" in replay.columns else pd.NaT
        replay_test = replay.loc[replay["split"].astype(str) == "test_like"].copy()
    else:
        replay_test = pd.DataFrame()

    if not vis.empty:
        data = vis.copy()
        if "time" in data.columns:
            data["time"] = pd.to_datetime(data["time"], errors="coerce")
        if "actual_ca_consumption" in data.columns and "current_ca_consumption" not in data.columns:
            data["current_ca_consumption"] = data["actual_ca_consumption"]
        if "split" not in data.columns and not replay_test.empty:
            supplement_cols = [col for col in replay_test.columns if col not in data.columns]
            if len(replay_test) == len(data):
                for col in supplement_cols:
                    data[col] = replay_test[col].to_numpy()
            elif "time" in data.columns:
                data = data.merge(
                    replay_test[["time"] + supplement_cols],
                    on="time",
                    how="left",
                    suffixes=("", "_replay"),
                )
            else:
                warnings.append("Visualization table has no split/time and replay length does not match; using visualization rows as-is.")
        elif "split" not in data.columns:
            warnings.append("Sample table has no split column; using all rows for the diversity audit.")
    else:
        data = replay_test.copy() if not replay_test.empty else replay.copy()
        if "actual_ca_consumption" not in data.columns and "current_ca_consumption" in data.columns:
            data["actual_ca_consumption"] = data["current_ca_consumption"]

    if "recommended_ca_consumption_target" not in data.columns:
        data["recommended_ca_consumption_target"] = (
            numeric_series(data, "recommended_ca_consumption_min") + numeric_series(data, "recommended_ca_consumption_max")
        ) / 2.0
    if "interval_position" not in data.columns:
        data["interval_position"] = derive_interval_position(data)
    if "split" in data.columns:
        test = data.loc[data["split"].astype(str) == "test_like"].copy()
        if not test.empty:
            data = test
    else:
        warnings.append("No split column available after preparation; audit uses all provided sample rows.")
    data = data.reset_index(drop=True)
    return data


def interval_key(data: pd.DataFrame) -> pd.Series:
    rec_min = numeric_series(data, "recommended_ca_consumption_min").round(9)
    rec_max = numeric_series(data, "recommended_ca_consumption_max").round(9)
    return rec_min.astype(str) + " - " + rec_max.astype(str)


def summarize_interval_distribution(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    sample = sample.copy()
    sample["interval_width"] = numeric_series(sample, "recommended_ca_consumption_max") - numeric_series(sample, "recommended_ca_consumption_min")
    sample["interval_key"] = interval_key(sample)
    target = numeric_series(sample, "recommended_ca_consumption_target")
    top_counts = sample["interval_key"].value_counts(dropna=False)
    sample_count = int(len(sample))
    top_5_coverage = float(top_counts.head(5).sum() / sample_count) if sample_count else None
    top_10_coverage = float(top_counts.head(10).sum() / sample_count) if sample_count else None
    target_iqr = float(target.quantile(0.75) - target.quantile(0.25)) if target.notna().any() else None
    target_range = float(target.max() - target.min()) if target.notna().any() else None
    target_std = float(target.std(ddof=0)) if target.notna().sum() > 1 else None
    summary = {
        "sample_count": sample_count,
        "unique_recommended_interval_count": int(top_counts.size),
        "unique_target_count": int(target.round(9).nunique(dropna=True)),
        **quantile_stats(numeric_series(sample, "recommended_ca_consumption_min"), "recommended_min"),
        **quantile_stats(numeric_series(sample, "recommended_ca_consumption_max"), "recommended_max"),
        **quantile_stats(target, "recommended_target"),
        **quantile_stats(sample["interval_width"], "interval_width"),
        **quantile_stats(numeric_series(sample, "current_ca_consumption"), "actual_ca_consumption"),
        "target_std": target_std,
        "target_iqr": target_iqr,
        "target_range": target_range,
        "interval_width_mean": float(sample["interval_width"].mean()) if sample["interval_width"].notna().any() else None,
        "top_5_interval_coverage": top_5_coverage,
        "top_10_interval_coverage": top_10_coverage,
        "top_intervals": top_counts.head(10).to_dict(),
        "narrow_interval_output": bool((target_iqr is not None and target_iqr <= 0.0002) or (top_5_coverage is not None and top_5_coverage >= 0.80)),
    }
    rows = [{"metric": key, "value": value} for key, value in summary.items() if key != "top_intervals"]
    for idx, (key, count) in enumerate(top_counts.head(10).items(), start=1):
        rows.append({"metric": f"top_{idx}_interval", "value": key})
        rows.append({"metric": f"top_{idx}_interval_count", "value": int(count)})
        rows.append({"metric": f"top_{idx}_interval_rate", "value": float(count / sample_count) if sample_count else None})
    return pd.DataFrame(rows), summary


def accepted_rules(rules: pd.DataFrame) -> pd.DataFrame:
    data = rules.copy()
    if "rule_status" in data.columns:
        data = data.loc[data["rule_status"].astype(str) == ACCEPTED_STATUS].copy()
    if "rule_grade" in data.columns:
        data = data.loc[data["rule_grade"].astype(str).isin(["A", "B"])].copy()
    return data


def summarize_rule_diversity(rules: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    data = rules.copy()
    data["recommended_target"] = (
        numeric_series(data, "recommended_dose_min") + numeric_series(data, "recommended_dose_max")
    ) / 2.0
    data["interval_width"] = numeric_series(data, "recommended_dose_max") - numeric_series(data, "recommended_dose_min")
    data["interval_key"] = numeric_series(data, "recommended_dose_min").round(9).astype(str) + " - " + numeric_series(data, "recommended_dose_max").round(9).astype(str)
    accepted = accepted_rules(data)
    target = numeric_series(accepted, "recommended_target")
    interval_counts = accepted["interval_key"].value_counts(dropna=False)
    top_5_cover = float(interval_counts.head(5).sum() / len(accepted)) if len(accepted) else None
    target_iqr = float(target.quantile(0.75) - target.quantile(0.25)) if target.notna().any() else None
    target_range = float(target.max() - target.min()) if target.notna().any() else None
    summary = {
        "accepted_rule_count": int(len(accepted)),
        "monitor_only_rule_count": int((data.get("rule_status", pd.Series("", index=data.index)).astype(str) == "monitor_only").sum()),
        "unique_rule_interval_count": int(interval_counts.size),
        **quantile_stats(numeric_series(accepted, "recommended_dose_min"), "rule_recommended_min"),
        **quantile_stats(numeric_series(accepted, "recommended_dose_max"), "rule_recommended_max"),
        **quantile_stats(target, "rule_recommended_target"),
        **quantile_stats(numeric_series(accepted, "interval_width"), "rule_interval_width"),
        "rule_target_iqr": target_iqr,
        "rule_target_range": target_range,
        "top_5_rule_interval_coverage": top_5_cover,
        "rule_interval_concentrated": bool(target_iqr is not None and target_iqr <= 0.00025),
        "most_rules_same_band": bool(top_5_cover is not None and top_5_cover >= 0.60),
    }
    rows = []
    for level, frame in [("all_rules", data), ("accepted_rules", accepted)]:
        target_level = numeric_series(frame, "recommended_target")
        rows.append({"summary_level": level, "metric": "rule_count", "value": int(len(frame))})
        rows.append({"summary_level": level, "metric": "unique_interval_count", "value": int(frame["interval_key"].nunique(dropna=True))})
        rows.append({"summary_level": level, "metric": "target_iqr", "value": float(target_level.quantile(0.75) - target_level.quantile(0.25)) if target_level.notna().any() else None})
        rows.append({"summary_level": level, "metric": "target_range", "value": float(target_level.max() - target_level.min()) if target_level.notna().any() else None})
    for col in ["regime_feature", "regime_bin"]:
        if col in accepted.columns:
            grouped = accepted.groupby(col, dropna=False)
            for group_key, group in grouped:
                group_target = numeric_series(group, "recommended_target")
                rows.append({
                    "summary_level": f"by_{col}",
                    "metric": str(group_key),
                    "value": float(group_target.median()) if group_target.notna().any() else None,
                    "rule_count": int(len(group)),
                    "target_std": float(group_target.std(ddof=0)) if group_target.notna().sum() > 1 else 0.0,
                    "target_min": float(group_target.min()) if group_target.notna().any() else None,
                    "target_max": float(group_target.max()) if group_target.notna().any() else None,
                })
    return pd.DataFrame(rows), summary


def parse_rule_ids(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    for sep in [";", ",", "|"]:
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def grade_rank(value: object) -> int:
    return {"A": 0, "B": 1, "C": 2}.get(str(value), 9)


def build_rule_map(rules: pd.DataFrame) -> dict[str, dict[str, object]]:
    data = rules.copy()
    data["recommended_target"] = (
        numeric_series(data, "recommended_dose_min") + numeric_series(data, "recommended_dose_max")
    ) / 2.0
    return {str(row["rule_id"]): row.to_dict() for _, row in data.iterrows() if "rule_id" in row and pd.notna(row["rule_id"])}


def choose_top_rule(rule_ids: list[str], rule_map: dict[str, dict[str, object]]) -> dict[str, object] | None:
    candidates = [rule_map[rule_id] for rule_id in rule_ids if rule_id in rule_map]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda r: (
            grade_rank(r.get("rule_grade")),
            -float(r.get("sample_count", 0) or 0),
            -float(r.get("ok_lift_vs_overall", 0) or 0),
            float(r.get("high_delta_vs_overall", 0) or 0),
        ),
    )[0]


def aggregation_compression(sample: pd.DataFrame, rules: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rule_map = build_rule_map(rules)
    rows = []
    for idx, row in sample.iterrows():
        matched_ids = parse_rule_ids(row.get("matched_rule_ids"))
        selected_ids = parse_rule_ids(row.get("selected_rule_ids"))
        if not matched_ids and selected_ids:
            matched_ids = selected_ids
        matched_targets = [float(rule_map[rid]["recommended_target"]) for rid in matched_ids if rid in rule_map and pd.notna(rule_map[rid].get("recommended_target"))]
        selected_targets = [float(rule_map[rid]["recommended_target"]) for rid in selected_ids if rid in rule_map and pd.notna(rule_map[rid].get("recommended_target"))]
        top_rule = choose_top_rule(matched_ids or selected_ids, rule_map)
        final_target = pd.to_numeric(pd.Series([row.get("recommended_ca_consumption_target")]), errors="coerce").iloc[0]
        top_target = float(top_rule["recommended_target"]) if top_rule is not None and pd.notna(top_rule.get("recommended_target")) else np.nan
        median_target = float(np.nanmedian(matched_targets)) if matched_targets else np.nan
        rows.append({
            "row_index": int(idx),
            "time": row.get("time"),
            "matched_rule_count": len(matched_ids),
            "selected_rule_count": len(selected_ids),
            "matched_rule_target_min": float(np.nanmin(matched_targets)) if matched_targets else np.nan,
            "matched_rule_target_max": float(np.nanmax(matched_targets)) if matched_targets else np.nan,
            "matched_rule_target_range": float(np.nanmax(matched_targets) - np.nanmin(matched_targets)) if matched_targets else np.nan,
            "matched_rule_target_std": float(np.nanstd(matched_targets)) if len(matched_targets) > 1 else 0.0 if matched_targets else np.nan,
            "final_recommended_target": float(final_target) if pd.notna(final_target) else np.nan,
            "top_rule_id": top_rule.get("rule_id") if top_rule is not None else None,
            "top_rule_recommended_target": top_target,
            "final_vs_rule_target_median_delta": float(final_target - median_target) if pd.notna(final_target) and pd.notna(median_target) else np.nan,
            "final_vs_top_rule_target_delta": float(final_target - top_target) if pd.notna(final_target) and pd.notna(top_target) else np.nan,
        })
    audit = pd.DataFrame(rows)
    final_std = numeric_series(audit, "final_recommended_target").std(ddof=0)
    top_std = numeric_series(audit, "top_rule_recommended_target").std(ddof=0)
    matched_range_median = numeric_series(audit, "matched_rule_target_range").median()
    ratio = float(final_std / top_std) if pd.notna(final_std) and pd.notna(top_std) and top_std > 0 else None
    summary = {
        "has_rule_id_information": bool((sample.get("matched_rule_ids", pd.Series(index=sample.index)).notna().any() if "matched_rule_ids" in sample.columns else False) or ("selected_rule_ids" in sample.columns and sample["selected_rule_ids"].notna().any())),
        "final_target_std": float(final_std) if pd.notna(final_std) else None,
        "top_rule_target_std": float(top_std) if pd.notna(top_std) else None,
        "matched_rule_target_range_median": float(matched_range_median) if pd.notna(matched_range_median) else None,
        "aggregation_compression_ratio": ratio,
        "aggregation_compresses_diversity": bool(ratio is not None and ratio < 0.75),
    }
    return audit, summary


def context_variability(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    concentrated_features = []
    available = []
    target = numeric_series(sample, "recommended_ca_consumption_target")
    for feature in KEY_CONTEXT_FEATURES:
        if feature not in sample.columns:
            continue
        values = numeric_series(sample, feature)
        if values.notna().sum() == 0:
            continue
        available.append(feature)
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        mean = values.mean()
        std = values.std(ddof=0)
        cv = float(std / abs(mean)) if pd.notna(mean) and mean != 0 else None
        try:
            tertiles = pd.qcut(values.rank(method="first"), q=3, labels=["low", "mid", "high"])
            counts = tertiles.value_counts().to_dict()
        except Exception:
            counts = {}
        max_share = max(counts.values()) / len(sample) if counts else None
        context_concentrated = bool(max_share is not None and max_share >= 0.60)
        if context_concentrated:
            concentrated_features.append(feature)
        corr = values.corr(target, method="spearman") if values.notna().sum() >= 3 and target.notna().sum() >= 3 else np.nan
        rows.append({
            "context_feature": feature,
            "sample_count": int(values.notna().sum()),
            "min": float(values.min()),
            "q25": float(q1),
            "median": float(values.median()),
            "q75": float(q3),
            "max": float(values.max()),
            "iqr": float(iqr),
            "coefficient_of_variation": cv,
            "tertile_low_count": int(counts.get("low", 0)),
            "tertile_mid_count": int(counts.get("mid", 0)),
            "tertile_high_count": int(counts.get("high", 0)),
            "max_tertile_share": float(max_share) if max_share is not None else None,
            "context_concentrated": context_concentrated,
            "spearman_corr_with_recommended_target": float(corr) if pd.notna(corr) else None,
        })
    summary = {
        "available_context_features": available,
        "available_context_feature_count": len(available),
        "concentrated_context_features": concentrated_features,
        "any_context_concentrated": bool(concentrated_features),
    }
    return pd.DataFrame(rows), summary


def regime_interval_mapping(rules: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    accepted = accepted_rules(rules).copy()
    accepted["recommended_target"] = (
        numeric_series(accepted, "recommended_dose_min") + numeric_series(accepted, "recommended_dose_max")
    ) / 2.0
    rows = []
    material_features = []
    if accepted.empty or "regime_feature" not in accepted.columns:
        return pd.DataFrame(rows), {"materially_different_feature_count": 0, "materially_different_features": []}
    for feature, group in accepted.groupby("regime_feature", dropna=False):
        targets = numeric_series(group, "recommended_target")
        mins = numeric_series(group, "recommended_dose_min")
        target_range = float(targets.max() - targets.min()) if targets.notna().any() else None
        min_range = float(mins.max() - mins.min()) if mins.notna().any() else None
        materially = bool((target_range is not None and target_range >= 0.0003) or (min_range is not None and min_range >= 0.0003))
        if materially:
            material_features.append(str(feature))
        rows.append({
            "regime_feature": feature,
            "rule_count": int(len(group)),
            "regime_bin_count": int(group["regime_bin"].nunique(dropna=True)) if "regime_bin" in group.columns else None,
            "recommended_target_min": float(targets.min()) if targets.notna().any() else None,
            "recommended_target_max": float(targets.max()) if targets.notna().any() else None,
            "recommended_target_range": target_range,
            "recommended_target_std": float(targets.std(ddof=0)) if targets.notna().sum() > 1 else 0.0,
            "recommended_dose_min_range": min_range,
            "materially_different_intervals": materially,
        })
    summary = {
        "materially_different_feature_count": len(material_features),
        "materially_different_features": material_features,
    }
    return pd.DataFrame(rows), summary


def classify_behavior(
    interval_summary: dict[str, object],
    rule_summary: dict[str, object],
    aggregation_summary: dict[str, object],
    context_summary: dict[str, object],
    mapping_summary: dict[str, object],
) -> tuple[str, str, str]:
    narrow = bool(interval_summary.get("narrow_interval_output"))
    rules_concentrated = bool(rule_summary.get("rule_interval_concentrated") or rule_summary.get("most_rules_same_band"))
    compressed = bool(aggregation_summary.get("aggregation_compresses_diversity"))
    material_count = int(mapping_summary.get("materially_different_feature_count") or 0)
    context_concentrated = bool(context_summary.get("any_context_concentrated"))

    if not interval_summary.get("sample_count") or not rule_summary.get("accepted_rule_count"):
        return "insufficient_information", "unknown", "insufficient_data_for_conclusion"
    if compressed and not rules_concentrated:
        return "aggregation_over_smoothed_recommender", "aggregation_compression", "test_top_rule_without_median_aggregation"
    if narrow and rules_concentrated and material_count == 0:
        if context_concentrated:
            return "stable_safe_band_recommender", "historical_operation_concentrated", "collect_more_diverse_regime_data"
        return "stable_safe_band_recommender", "rule_intervals_concentrated", "keep_stable_safe_band_mvp"
    if narrow and material_count > 0:
        return "weakly_regime_adaptive_recommender", "single_variable_regime_too_coarse", "build_multivariate_regime_rules"
    if not narrow and material_count >= 2:
        return "strongly_regime_adaptive_recommender", "mixed", "build_multivariate_regime_rules"
    if context_concentrated:
        return "weakly_regime_adaptive_recommender", "historical_operation_concentrated", "collect_more_diverse_regime_data"
    return "weakly_regime_adaptive_recommender", "mixed", "build_multivariate_regime_rules"


def save_interval_target_distribution(sample: pd.DataFrame, path: Path) -> None:
    values = numeric_series(sample, "recommended_ca_consumption_target").dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=min(30, max(5, values.nunique())), color="#3F6FB5", alpha=0.82, edgecolor="white")
    if not values.empty:
        ax.axvline(values.median(), color="#C62828", linestyle="--", linewidth=1.5, label=f"中位数 {values.median():.6f}")
    ax.set_title("测试集推荐钙单耗中心值分布")
    ax.set_xlabel("推荐钙单耗中心值")
    ax.set_ylabel("样本数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_width_distribution(sample: pd.DataFrame, path: Path) -> None:
    width = (numeric_series(sample, "recommended_ca_consumption_max") - numeric_series(sample, "recommended_ca_consumption_min")).dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(width, bins=min(30, max(5, width.nunique())), color="#00897B", alpha=0.82, edgecolor="white")
    ax.set_title("测试集推荐钙单耗区间宽度分布")
    ax.set_xlabel("推荐区间宽度")
    ax.set_ylabel("样本数")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_top_frequency(sample: pd.DataFrame, path: Path) -> None:
    keys = interval_key(sample).value_counts().head(10)
    labels = [f"{i + 1}" for i in range(len(keys))]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, keys.values, color="#6A5ACD", alpha=0.86)
    ax.set_title("测试集高频推荐钙单耗区间")
    ax.set_xlabel("高频区间排名")
    ax.set_ylabel("样本数")
    for bar, key in zip(bars, keys.index):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height())}\n{key}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_rule_interval_by_feature(rules: pd.DataFrame, path: Path) -> None:
    accepted = accepted_rules(rules).copy()
    accepted["recommended_target"] = (
        numeric_series(accepted, "recommended_dose_min") + numeric_series(accepted, "recommended_dose_max")
    ) / 2.0
    fig, ax = plt.subplots(figsize=(10, 5.6))
    if accepted.empty:
        ax.text(0.5, 0.5, "无可用规则", ha="center", va="center")
    else:
        features = list(dict.fromkeys(accepted["regime_feature"].astype(str)))
        x_map = {feature: idx for idx, feature in enumerate(features)}
        xs = accepted["regime_feature"].astype(str).map(x_map)
        colors = accepted.get("rule_grade", pd.Series("B", index=accepted.index)).astype(str).map({"A": "#2E7D32", "B": "#1565C0"}).fillna("#757575")
        ax.scatter(xs, numeric_series(accepted, "recommended_target"), c=colors, s=55, alpha=0.85)
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=35, ha="right", fontsize=8)
    ax.set_title("不同工况变量对应的规则推荐钙单耗中心值")
    ax.set_xlabel("工况变量")
    ax.set_ylabel("规则推荐钙单耗中心值")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_aggregation_compression(audit: pd.DataFrame, path: Path) -> bool:
    if audit.empty or "top_rule_recommended_target" not in audit.columns or numeric_series(audit, "top_rule_recommended_target").notna().sum() == 0:
        return False
    fig, ax = plt.subplots(figsize=(7, 6))
    x = numeric_series(audit, "top_rule_recommended_target")
    y = numeric_series(audit, "final_recommended_target")
    ax.scatter(x, y, s=14, alpha=0.55, color="#455A64")
    finite = x.notna() & y.notna()
    if finite.any():
        lo = min(float(x[finite].min()), float(y[finite].min()))
        hi = max(float(x[finite].max()), float(y[finite].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#C62828", linewidth=1)
    ax.set_title("多规则聚合前后推荐钙单耗差异")
    ax.set_xlabel("最高优先级规则中心值")
    ax.set_ylabel("最终聚合推荐中心值")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def save_context_vs_target(sample: pd.DataFrame, path: Path) -> bool:
    available = [feature for feature in KEY_CONTEXT_FEATURES if feature in sample.columns and numeric_series(sample, feature).notna().sum() > 0]
    if not available:
        return False
    selected = available[:4]
    fig, axes = plt.subplots(len(selected), 1, figsize=(8, 3.0 * len(selected)), squeeze=False)
    target = numeric_series(sample, "recommended_ca_consumption_target")
    for ax, feature in zip(axes[:, 0], selected):
        values = numeric_series(sample, feature)
        ax.scatter(values, target, s=14, alpha=0.55, color="#1976D2")
        ax.set_title(f"{feature} 与推荐中心值")
        ax.set_xlabel(feature)
        ax.set_ylabel("推荐钙单耗中心值")
    fig.suptitle("工况变量与推荐钙单耗中心值关系", y=1.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def append_doc(
    doc_path: Path,
    interval_summary: dict[str, object],
    rule_summary: dict[str, object],
    aggregation_summary: dict[str, object],
    context_summary: dict[str, object],
    classification: str,
    reason: str,
    next_step: str,
    generated_outputs: dict[str, list[str]],
) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    existing = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""
    section_no = 23
    while f"## {section_no}." in existing:
        section_no += 1
    section = f"""

## {section_no}. 钙单耗推荐区间差异性审计

### {section_no}.1 审计目的

本阶段用于解释测试集推荐钙单耗区间为何呈现近似稳定带。该分析只审计既有推荐器输出，不训练模型、不修改规则、不进行策略搜索，也不形成自动控制或 DCS 写回建议。

目录策略同步更新：`data/` 仅保留原始或必要基础数据；本阶段生成的审计 CSV/JSON 输出写入 `runs/ca_interval_diversity_audit/`；图像和人工可读表写入 `reports/`；实验说明仅追加到 `docs/Experimental_Procedure_cn.md`。

### {section_no}.2 主要结果

- 测试集样本数：{interval_summary.get('sample_count')}
- 唯一推荐区间数：{interval_summary.get('unique_recommended_interval_count')}
- 推荐中心值中位数：{interval_summary.get('recommended_target_median')}
- 推荐中心值 IQR：{interval_summary.get('target_iqr')}
- 推荐中心值范围：{interval_summary.get('target_range')}
- Top 5 推荐区间覆盖率：{interval_summary.get('top_5_interval_coverage')}
- 接受规则数：{rule_summary.get('accepted_rule_count')}
- 规则中心值 IQR：{rule_summary.get('rule_target_iqr')}
- 规则中心值范围：{rule_summary.get('rule_target_range')}
- 聚合压缩标记：{aggregation_summary.get('aggregation_compresses_diversity')}
- 可用上下文字段：{', '.join(context_summary.get('available_context_features', [])) if context_summary.get('available_context_features') else '无'}

### {section_no}.3 判断

当前推荐器行为分类为：`{classification}`。稳定区间的主要解释为：`{reason}`。如果区间稳定主要来自规则本身集中，则它更接近“稳定安全带 MVP”；如果来自多规则中位数聚合，则后续应测试最高优先级规则输出；如果来自单变量规则过粗，则应构建多变量工况规则。

### {section_no}.4 输出文件

- 机器可读审计输出：`runs/ca_interval_diversity_audit/`
- 图像输出：{', '.join(generated_outputs.get('figures', []))}
- 人工可读汇总表：{', '.join(generated_outputs.get('tables', []))}

### {section_no}.5 下一步

推荐下一步：`{next_step}`。

局限性：本阶段为离线审计；不提供因果证明；不生成控制动作；结论依赖既有规则、replay 和人工复核审计产物。
"""
    with doc_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(section)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    warnings: list[str] = []
    assumptions = [
        "The audit uses existing recommender replay/rule artifacts only and does not alter recommendation logic.",
        "Generated machine-readable audit outputs are written under runs/, not data/.",
        "Stable interval classification is descriptive and not causal proof.",
    ]

    runs_root = Path("runs")
    models_root = Path("models")
    vis_path = resolve_path(args.visualization_table, required=False, search_roots=[runs_root], warnings=warnings)
    replay_path = resolve_path(args.replay, required=False, search_roots=[runs_root], warnings=warnings)
    rules_path = resolve_path(args.rules, required=True, search_roots=[runs_root], warnings=warnings)
    rule_audit_path = resolve_path(args.rule_audit, required=False, search_roots=[runs_root], warnings=warnings)
    artifact_path = resolve_path(args.artifact, required=False, search_roots=[models_root, runs_root], warnings=warnings)

    vis = read_table(vis_path)
    replay = read_table(replay_path)
    rules = read_table(rules_path)
    _rule_audit = read_table(rule_audit_path)
    artifact = load_json(artifact_path)

    sample = prepare_sample_table(vis, replay, warnings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    interval_table, interval_summary = summarize_interval_distribution(sample)
    interval_table.to_csv(args.output_dir / "interval_distribution_summary.csv", index=False, encoding="utf-8-sig")

    rule_diversity, rule_summary = summarize_rule_diversity(rules)
    rule_diversity.to_csv(args.output_dir / "rule_interval_diversity.csv", index=False, encoding="utf-8-sig")

    aggregation_audit, aggregation_summary = aggregation_compression(sample, rules)
    aggregation_audit.to_csv(args.output_dir / "aggregation_compression_audit.csv", index=False, encoding="utf-8-sig")

    context_audit, context_summary = context_variability(sample)
    context_audit.to_csv(args.output_dir / "context_variability_audit.csv", index=False, encoding="utf-8-sig")

    regime_mapping, mapping_summary = regime_interval_mapping(rules)
    regime_mapping.to_csv(args.output_dir / "regime_interval_mapping_audit.csv", index=False, encoding="utf-8-sig")

    classification, reason, next_step = classify_behavior(
        interval_summary,
        rule_summary,
        aggregation_summary,
        context_summary,
        mapping_summary,
    )

    stable_band = bool(interval_summary.get("narrow_interval_output") and (rule_summary.get("rule_interval_concentrated") or rule_summary.get("most_rules_same_band")))
    interval_summary["stable_band_recommender"] = stable_band

    figures: list[str] = []
    fig1 = args.figure_dir / "ca_interval_target_distribution.png"
    save_interval_target_distribution(sample, fig1)
    figures.append(str(fig1))
    fig2 = args.figure_dir / "ca_interval_width_distribution.png"
    save_width_distribution(sample, fig2)
    figures.append(str(fig2))
    fig3 = args.figure_dir / "ca_interval_top_frequency.png"
    save_top_frequency(sample, fig3)
    figures.append(str(fig3))
    fig4 = args.figure_dir / "ca_rule_interval_by_regime_feature.png"
    save_rule_interval_by_feature(rules, fig4)
    figures.append(str(fig4))
    fig5 = args.figure_dir / "ca_aggregation_compression.png"
    if save_aggregation_compression(aggregation_audit, fig5):
        figures.append(str(fig5))
    else:
        warnings.append("Aggregation compression figure skipped because rule-id target information was unavailable.")
    fig6 = args.figure_dir / "ca_context_vs_recommended_target.png"
    if save_context_vs_target(sample, fig6):
        figures.append(str(fig6))
    else:
        warnings.append("Context-vs-target figure skipped because sample-level context features were unavailable.")

    summary_rows = [
        {"metric": "sample_count", "value": interval_summary.get("sample_count"), "interpretation_cn": "用于审计的测试集样本数。"},
        {"metric": "unique_recommended_interval_count", "value": interval_summary.get("unique_recommended_interval_count"), "interpretation_cn": "唯一推荐钙单耗区间数量。"},
        {"metric": "top_5_interval_coverage", "value": interval_summary.get("top_5_interval_coverage"), "interpretation_cn": "Top 5 高频区间覆盖率，越高表示输出越集中。"},
        {"metric": "recommended_target_iqr", "value": interval_summary.get("target_iqr"), "interpretation_cn": "推荐中心值四分位距。"},
        {"metric": "rule_target_iqr", "value": rule_summary.get("rule_target_iqr"), "interpretation_cn": "接受规则自身推荐中心值的四分位距。"},
        {"metric": "aggregation_compresses_diversity", "value": aggregation_summary.get("aggregation_compresses_diversity"), "interpretation_cn": "是否存在多规则聚合压缩差异。"},
        {"metric": "interpretation_classification", "value": classification, "interpretation_cn": "推荐器当前更接近稳定安全带还是工况自适应。"},
        {"metric": "likely_reason_for_stable_interval", "value": reason, "interpretation_cn": "推荐区间稳定的主要解释。"},
        {"metric": "recommended_next_step", "value": next_step, "interpretation_cn": "建议的下一步实验或工程动作。"},
    ]
    summary_table_path = args.table_dir / "ca_interval_diversity_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_table_path, index=False, encoding="utf-8-sig")

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_paths": {
            "visualization_table": str(vis_path) if vis_path else None,
            "replay": str(replay_path) if replay_path else None,
            "rules": str(rules_path),
            "rule_audit": str(rule_audit_path) if rule_audit_path else None,
            "artifact": str(artifact_path) if artifact_path else None,
        },
        "output_dir": str(args.output_dir),
        "figure_dir": str(args.figure_dir),
        "table_dir": str(args.table_dir),
        "sample_level_summary": {
            "source": "visualization_table_with_replay_supplement" if vis_path and replay_path else "visualization_table" if vis_path else "replay",
            "row_count": int(len(sample)),
            "has_split": "split" in sample.columns,
            "artifact_accepted_rule_count": artifact.get("accepted_rule_count") if artifact else None,
        },
        "interval_distribution_summary": interval_summary,
        "rule_interval_diversity_summary": rule_summary,
        "aggregation_compression_summary": aggregation_summary,
        "context_variability_summary": context_summary,
        "regime_interval_mapping_summary": mapping_summary,
        "interpretation_classification": classification,
        "likely_reason_for_stable_interval": reason,
        "key_findings": [
            f"Top 5 interval coverage = {interval_summary.get('top_5_interval_coverage')}.",
            f"Rule target IQR = {rule_summary.get('rule_target_iqr')}.",
            f"Aggregation compression flag = {aggregation_summary.get('aggregation_compresses_diversity')}.",
            f"Materially different regime features = {mapping_summary.get('materially_different_features')}.",
        ],
        "warnings": warnings,
        "assumptions": assumptions,
        "recommended_next_step": next_step,
        "generated_figures": figures,
        "generated_tables": [str(summary_table_path)],
        "generated_machine_outputs": [
            str(args.output_dir / "interval_distribution_summary.csv"),
            str(args.output_dir / "rule_interval_diversity.csv"),
            str(args.output_dir / "aggregation_compression_audit.csv"),
            str(args.output_dir / "context_variability_audit.csv"),
            str(args.output_dir / "regime_interval_mapping_audit.csv"),
            str(args.output_dir / "ca_interval_diversity_audit_report.json"),
        ],
    }
    report_path = args.output_dir / "ca_interval_diversity_audit_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(report), handle, ensure_ascii=False, indent=2)

    append_doc(
        args.doc,
        interval_summary,
        rule_summary,
        aggregation_summary,
        context_summary,
        classification,
        reason,
        next_step,
        {"figures": figures, "tables": [str(summary_table_path)]},
    )

    print("Calcium interval diversity audit summary")
    print(f"sample_count: {interval_summary.get('sample_count')}")
    print(f"unique interval count: {interval_summary.get('unique_recommended_interval_count')}")
    print(f"top 5 interval coverage: {interval_summary.get('top_5_interval_coverage')}")
    print(
        "recommended target median/IQR/range: "
        f"{interval_summary.get('recommended_target_median')} / {interval_summary.get('target_iqr')} / {interval_summary.get('target_range')}"
    )
    print(f"accepted rule count: {rule_summary.get('accepted_rule_count')}")
    print(f"rule target IQR/range: {rule_summary.get('rule_target_iqr')} / {rule_summary.get('rule_target_range')}")
    print(f"aggregation compression flag: {aggregation_summary.get('aggregation_compresses_diversity')}")
    print(f"interpretation classification: {classification}")
    print(f"likely reason for stable interval: {reason}")
    print(f"recommended_next_step: {next_step}")
    print("Generated figures:")
    for path in figures:
        print(f"  {path}")
    print("Generated tables/reports:")
    for path in [summary_table_path, report_path]:
        print(f"  {path}")
    print(f"Documentation appended: {args.doc}")


if __name__ == "__main__":
    main()
