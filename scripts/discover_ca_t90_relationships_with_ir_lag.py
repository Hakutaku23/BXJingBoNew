from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from discover_ca_t90_relationships import (
    CONTEXT_CANDIDATES,
    T90_HIGH,
    T90_LOW,
    as_jsonable,
    bin_rate_spread,
    ensure_targets,
    experiment_a_regime_dose_response,
    experiment_e_band_map,
    is_leakage_column,
    load_json,
    make_quantile_bins,
    safe_corr,
    section_title,
    summarize_dose_bins,
    train_screen_model,
    write_csv,
)


PRIMARY_DOSE_PRIORITY = [
    "ca_per_rubber_flow_win_60_mean",
    "ca_per_rubber_flow_lag_165",
    "ca_win_60_mean",
    "ca_lag_165",
]
IR_LAG_DEFAULT_FEATURE = "output_ir_corrected_offset_20_win_15_std"
TARGETS = ["y_high", "y_low", "y_out_spec", "y_ok"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run relationship discovery with best online-safe IR lag feature.")
    parser.add_argument("--features", type=Path, default=Path("data/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--feature-report", type=Path, default=Path("data/t90_ca_feature_report.json"))
    parser.add_argument("--dose-response-report", type=Path, default=Path("data/t90_ca_dose_response_report.json"))
    parser.add_argument("--data-with-ir", type=Path, default=Path("data/data_clean_with_ir.parquet"))
    parser.add_argument("--ir-lag-report", type=Path, default=Path("data/output_ir_lag_sensitivity_report.json"))
    parser.add_argument("--previous-relationship-report", type=Path, default=Path("data/ca_t90_relationship_discovery_report.json"))
    parser.add_argument("--previous-regime-output", type=Path, default=Path("data/ca_regime_dose_response.csv"))
    parser.add_argument("--previous-interaction-output", type=Path, default=Path("data/ca_context_interaction_screen.csv"))
    parser.add_argument("--regime-output", type=Path, default=Path("data/ca_regime_dose_response_ir_lag.csv"))
    parser.add_argument("--interaction-output", type=Path, default=Path("data/ca_context_interaction_screen_ir_lag.csv"))
    parser.add_argument("--ir-strat-output", type=Path, default=Path("data/ca_ir_lag_stratified_dose_response.csv"))
    parser.add_argument("--mediation-output", type=Path, default=Path("data/ir_lag_mediation_diagnostic.csv"))
    parser.add_argument("--band-map-output", type=Path, default=Path("data/ca_regime_optimal_band_map_ir_lag.csv"))
    parser.add_argument("--comparison-output", type=Path, default=Path("data/ca_relationship_discovery_ir_lag_comparison.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/ca_t90_relationship_discovery_ir_lag_report.json"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--n-bins", type=int, default=5)
    return parser.parse_args()


def choose_primary_dose(frame: pd.DataFrame, dose_report: dict[str, object]) -> str:
    primary = dose_report.get("primary_dose_feature")
    if isinstance(primary, str) and primary in frame.columns:
        return primary
    for feature in PRIMARY_DOSE_PRIORITY:
        if feature in frame.columns:
            return feature
    raise ValueError("No primary calcium dose feature is available.")


def rolling_std_feature(values: pd.Series, window: str = "15min") -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rolling(window, min_periods=2).std()


def derive_ir_lag_feature(
    supervised: pd.DataFrame,
    data_with_ir_path: Path,
    ir_lag_report: dict[str, object],
    warnings: list[str],
) -> tuple[pd.DataFrame, str, dict[str, object]]:
    best = ir_lag_report.get("best_online_safe_ir_alignment") or {}
    feature_name = best.get("feature_column") if isinstance(best, dict) else None
    if not isinstance(feature_name, str):
        feature_name = IR_LAG_DEFAULT_FEATURE
    offset_minutes = int(best.get("offset_minutes", 20)) if isinstance(best, dict) else 20
    variant = str(best.get("ir_feature_variant", "win_15_std")) if isinstance(best, dict) else "win_15_std"
    creation = {
        "feature_name": feature_name,
        "source": "derived_from_output_ir_corrected",
        "offset_minutes": offset_minutes,
        "window": "15min",
        "variant": variant,
        "online_safe": offset_minutes >= 0,
    }
    if feature_name in supervised.columns:
        creation["source"] = "already_present_in_feature_dataset"
        return supervised, feature_name, creation
    if not data_with_ir_path.exists():
        warnings.append(f"data-with-ir is missing: {data_with_ir_path}; IR-lag analyses will be limited.")
        supervised[feature_name] = np.nan
        creation["source"] = "missing_data_with_ir"
        return supervised, feature_name, creation
    columns = pd.read_parquet(data_with_ir_path, columns=None).columns.tolist()
    if feature_name in columns:
        ir = pd.read_parquet(data_with_ir_path, columns=["time", feature_name])
        ir["time"] = pd.to_datetime(ir["time"], errors="coerce")
        ir = ir.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last")
        merged = supervised.merge(ir, on="time", how="left")
        creation["source"] = "already_present_in_data_with_ir"
        return merged, feature_name, creation
    if "output_ir_corrected" not in columns:
        warnings.append("data-with-ir does not contain output_ir_corrected; IR-lag feature could not be derived.")
        supervised[feature_name] = np.nan
        creation["source"] = "missing_output_ir_corrected"
        return supervised, feature_name, creation
    ir = pd.read_parquet(data_with_ir_path, columns=["time", "output_ir_corrected"])
    ir["time"] = pd.to_datetime(ir["time"], errors="coerce")
    ir = ir.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last").sort_values("time")
    indexed = ir.set_index("time")
    if variant != "win_15_std":
        warnings.append(f"Best IR variant is {variant}; this script derives win_15_std as requested by current stage.")
    indexed[feature_name] = rolling_std_feature(indexed["output_ir_corrected"], "15min")
    lookup = supervised[["time"]].copy()
    lookup["ir_lookup_time"] = lookup["time"] - pd.to_timedelta(offset_minutes, unit="m")
    feature_timeline = indexed[[feature_name]].reset_index()
    merged_values = lookup.merge(feature_timeline, left_on="ir_lookup_time", right_on="time", how="left", suffixes=("", "_ir"))
    result = supervised.copy()
    result[feature_name] = pd.to_numeric(merged_values[feature_name], errors="coerce")
    creation["non_null_count"] = int(result[feature_name].notna().sum())
    creation["non_null_rate"] = float(result[feature_name].notna().mean())
    return result, feature_name, creation


def load_supervised(args: argparse.Namespace, dose_report: dict[str, object], ir_lag_report: dict[str, object], warnings: list[str]) -> tuple[pd.DataFrame, str, list[str], str, dict[str, object]]:
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
    dose_feature = choose_primary_dose(frame, dose_report)
    context_features = [feature for feature in CONTEXT_CANDIDATES if feature in frame.columns and not is_leakage_column(feature)]
    frame, ir_feature, creation = derive_ir_lag_feature(frame, args.data_with_ir, ir_lag_report, warnings)
    return frame, dose_feature, context_features, ir_feature, creation


def experiment_b_interaction_screen_ir_lag(data: pd.DataFrame, dose_feature: str, context_features: list[str]) -> pd.DataFrame:
    from discover_ca_t90_relationships import experiment_b_interaction_screen

    table = experiment_b_interaction_screen(data, dose_feature, context_features)
    if table.empty:
        table["suspicious_large_delta"] = []
        return table
    table["suspicious_large_delta"] = (
        (pd.to_numeric(table["delta_auc"], errors="coerce") > 0.25)
        | (pd.to_numeric(table["delta_ap"], errors="coerce") > 0.25)
    )
    return table


def experiment_c_ir_lag_stratified(data: pd.DataFrame, dose_feature: str, ir_feature: str, n_bins: int) -> pd.DataFrame:
    columns = [
        "ir_feature", "ir_regime", "dose_bin", "sample_count", "regime_sample_count", "dose_min",
        "dose_max", "dose_mean", "t90_mean", "t90_median", "ok_count", "ok_rate", "low_count",
        "low_rate", "high_count", "high_rate", "out_spec_count", "out_spec_rate", "support_level",
        "source", "best_dose_bin_in_regime",
    ]
    if ir_feature not in data.columns:
        return pd.DataFrame(columns=columns)
    work = data[[ir_feature, dose_feature, "t90", "y_ok", "y_low", "y_high", "y_out_spec"]].copy()
    regime_ids = make_quantile_bins(work[ir_feature], 3)
    work["ir_regime"] = regime_ids.map({0: "low", 1: "mid", 2: "high"}).astype("object")
    table = summarize_dose_bins(work, dose_feature, "ir_regime", "ir_regime", n_bins, source="ir_lag_tertile")
    if table.empty:
        return pd.DataFrame(columns=columns)
    table.insert(0, "ir_feature", ir_feature)
    return table[columns]


def experiment_d_ir_lag_mediation(
    data: pd.DataFrame,
    dose_feature: str,
    context_features: list[str],
    ir_feature: str,
    n_bins: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    summary = {
        "calcium_to_ir_lag_signal": False,
        "ir_lag_to_t90_risk_signal": False,
        "ir_lag_incremental_signal": False,
        "calcium_ir_lag_interaction_signal": False,
        "descriptive_mediation_possible": False,
        "not_causal_proof": True,
    }
    usable = data[[dose_feature, ir_feature, "t90", "y_ok", "y_low", "y_high", "y_out_spec"]].dropna(subset=[ir_feature])
    ca_ir_s = safe_corr(usable[dose_feature], usable[ir_feature], "spearman")
    ca_ir_p = safe_corr(usable[dose_feature], usable[ir_feature], "pearson")
    summary["calcium_to_ir_lag_signal"] = bool(np.isfinite(ca_ir_s) and abs(ca_ir_s) >= 0.08)
    rows.extend(
        [
            metric_row("calcium_to_ir_lag", "output_ir_lag", f"{dose_feature}->{ir_feature}", "spearman", ca_ir_s, len(usable.dropna(subset=[dose_feature])), "calcium-to-IR-lag signal" if summary["calcium_to_ir_lag_signal"] else "weak_or_unclear"),
            metric_row("calcium_to_ir_lag", "output_ir_lag", f"{dose_feature}->{ir_feature}", "pearson", ca_ir_p, len(usable.dropna(subset=[dose_feature])), "linear relation check"),
        ]
    )
    work = usable.dropna(subset=[dose_feature]).copy()
    work["dose_bin"] = make_quantile_bins(work[dose_feature], n_bins)
    for dose_bin, group in work.dropna(subset=["dose_bin"]).groupby("dose_bin", sort=True):
        rows.append(metric_row("calcium_bin_ir_lag", "output_ir_lag", dose_feature, f"ir_lag_mean_bin_{int(dose_bin)}", float(group[ir_feature].mean()), len(group), "IR-lag mean by calcium dose bin"))
    for target in ["t90", "y_ok", "y_low", "y_high", "y_out_spec"]:
        corr = safe_corr(usable[ir_feature], usable[target], "spearman")
        rows.append(metric_row("ir_lag_to_t90_risk", target, ir_feature, "spearman", corr, len(usable.dropna(subset=[target])), "IR-lag-to-risk correlation"))
    high_spread = bin_rate_spread(usable, ir_feature, "y_high", n_bins)
    out_spread = bin_rate_spread(usable, ir_feature, "y_out_spec", n_bins)
    y_high_corr = safe_corr(usable[ir_feature], usable["y_high"], "spearman")
    y_out_corr = safe_corr(usable[ir_feature], usable["y_out_spec"], "spearman")
    summary["ir_lag_to_t90_risk_signal"] = bool(
        (np.isfinite(high_spread) and high_spread >= 0.05)
        or (np.isfinite(out_spread) and out_spread >= 0.05)
        or (np.isfinite(y_high_corr) and abs(y_high_corr) >= 0.08)
        or (np.isfinite(y_out_corr) and abs(y_out_corr) >= 0.08)
    )
    rows.extend(
        [
            metric_row("ir_lag_bin_risk_spread", "y_high", ir_feature, "high_rate_spread", high_spread, len(usable), "IR-lag bin high-T90 spread"),
            metric_row("ir_lag_bin_risk_spread", "y_out_spec", ir_feature, "out_spec_rate_spread", out_spread, len(usable), "IR-lag bin out-spec spread"),
        ]
    )
    split = int(len(data) * 0.8)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()
    base_features = [dose_feature] + context_features[:5]
    for target in ["y_high", "y_out_spec"]:
        base = train_screen_model(train, test, base_features, target)
        plus_ir = train_screen_model(train, test, base_features + [ir_feature], target)
        delta_ap = plus_ir["ap"] - base["ap"] if np.isfinite(plus_ir["ap"]) and np.isfinite(base["ap"]) else math.nan
        delta_auc = plus_ir["auc"] - base["auc"] if np.isfinite(plus_ir["auc"]) and np.isfinite(base["auc"]) else math.nan
        delta_brier = plus_ir["brier"] - base["brier"] if np.isfinite(plus_ir["brier"]) and np.isfinite(base["brier"]) else math.nan
        if (np.isfinite(delta_ap) and delta_ap >= 0.03) or (np.isfinite(delta_auc) and delta_auc >= 0.03):
            summary["ir_lag_incremental_signal"] = True
        rows.extend(
            [
                metric_row("incremental_ir_lag", target, "calcium_context_plus_ir_lag", "delta_ap", delta_ap, len(test), "IR-lag incremental screening"),
                metric_row("incremental_ir_lag", target, "calcium_context_plus_ir_lag", "delta_auc", delta_auc, len(test), "IR-lag incremental screening"),
                metric_row("incremental_ir_lag", target, "calcium_context_plus_ir_lag", "delta_brier", delta_brier, len(test), "negative brier delta is better"),
            ]
        )
        tr = train[[dose_feature, ir_feature, target]].copy()
        te = test[[dose_feature, ir_feature, target]].copy()
        dose_med = pd.to_numeric(tr[dose_feature], errors="coerce").median()
        ir_med = pd.to_numeric(tr[ir_feature], errors="coerce").median()
        for frame in [tr, te]:
            frame["ca_ir_lag_interaction"] = (
                pd.to_numeric(frame[dose_feature], errors="coerce") - dose_med
            ) * (pd.to_numeric(frame[ir_feature], errors="coerce") - ir_med)
        ca_ir = train_screen_model(tr, te, [dose_feature, ir_feature], target)
        ca_ir_inter = train_screen_model(tr, te, [dose_feature, ir_feature, "ca_ir_lag_interaction"], target)
        d_ap = ca_ir_inter["ap"] - ca_ir["ap"] if np.isfinite(ca_ir_inter["ap"]) and np.isfinite(ca_ir["ap"]) else math.nan
        d_auc = ca_ir_inter["auc"] - ca_ir["auc"] if np.isfinite(ca_ir_inter["auc"]) and np.isfinite(ca_ir["auc"]) else math.nan
        if (np.isfinite(d_ap) and d_ap >= 0.03) or (np.isfinite(d_auc) and d_auc >= 0.03):
            summary["calcium_ir_lag_interaction_signal"] = True
        rows.extend(
            [
                metric_row("calcium_ir_lag_interaction", target, f"{dose_feature}*{ir_feature}", "delta_ap", d_ap, len(te), "calcium-IR-lag interaction screening"),
                metric_row("calcium_ir_lag_interaction", target, f"{dose_feature}*{ir_feature}", "delta_auc", d_auc, len(te), "calcium-IR-lag interaction screening"),
            ]
        )
    summary["descriptive_mediation_possible"] = bool(summary["calcium_to_ir_lag_signal"] and summary["ir_lag_to_t90_risk_signal"])
    return pd.DataFrame(rows), summary


def metric_row(diagnostic_type: str, target: str, feature_or_pair: str, metric: str, value: float, sample_count: int, interpretation: str) -> dict[str, object]:
    return {
        "diagnostic_type": diagnostic_type,
        "target": target,
        "feature_or_pair": feature_or_pair,
        "metric": metric,
        "value": value,
        "sample_count": int(sample_count),
        "interpretation": interpretation,
    }


def add_ir_columns_to_band_map(band_map: pd.DataFrame, data: pd.DataFrame, ir_feature: str) -> pd.DataFrame:
    if band_map.empty or ir_feature not in data.columns:
        return band_map
    result = band_map.copy()
    result["ir_lag_available_rate_in_regime"] = np.nan
    result["ir_lag_mean_in_regime"] = np.nan
    result["ir_lag_regime_if_available"] = ""
    tertiles = make_quantile_bins(data[ir_feature], 3).map({0: "low", 1: "mid", 2: "high"}).astype("object")
    for idx, row in result.iterrows():
        feature = row["regime_feature"]
        regime = row["regime_bin"]
        if feature not in data.columns:
            continue
        regimes = make_quantile_bins(data[feature], 3).map({0: "low", 1: "mid", 2: "high"}).astype("object")
        subset = data[regimes == regime]
        result.loc[idx, "ir_lag_available_rate_in_regime"] = float(subset[ir_feature].notna().mean()) if len(subset) else math.nan
        result.loc[idx, "ir_lag_mean_in_regime"] = float(subset[ir_feature].mean()) if subset[ir_feature].notna().any() else math.nan
        if subset[ir_feature].notna().any():
            result.loc[idx, "ir_lag_regime_if_available"] = str(tertiles.loc[subset.index].mode(dropna=True).iloc[0]) if not tertiles.loc[subset.index].dropna().empty else ""
    return result


def compare_with_previous(
    previous_report: dict[str, object],
    previous_regime_path: Path,
    previous_interaction_path: Path,
    current_regime: pd.DataFrame,
    current_interaction: pd.DataFrame,
    current_band: pd.DataFrame,
    current_mediation: dict[str, object],
    current_next: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    prev_stable = previous_report.get("optimal_band_map_summary", {}).get("stable_candidate_count")
    prev_passed = previous_report.get("interaction_screen_summary", {}).get("passed_interaction_count")
    prev_enough = previous_report.get("regime_dose_response_summary", {}).get("enough_support_group_count")
    prev_next = previous_report.get("recommended_next_step")
    prev_mediation = previous_report.get("mediation_diagnostic_summary", {})
    prev_interaction = pd.read_csv(previous_interaction_path) if previous_interaction_path.exists() else pd.DataFrame()
    previous_top_context = set()
    if not prev_interaction.empty and "passed_interaction_screen" in prev_interaction.columns:
        previous_top_context = set(prev_interaction[prev_interaction["passed_interaction_screen"].astype(bool)]["context_feature"].head(10))
    current_passed = current_interaction[current_interaction["passed_interaction_screen"].astype(bool)] if not current_interaction.empty else pd.DataFrame()
    current_top_context = set(current_passed["context_feature"].head(10)) if not current_passed.empty else set()
    important = {"bromine_feed_win_60_mean", "neutral_alkali_feed_win_60_mean", "rubber_flow_2_win_60_mean"}
    current_stable = int(current_band["risk_note"].str.contains("stable_candidate", na=False).sum()) if not current_band.empty else 0
    current_enough = int((current_regime["support_level"] == "enough_support").sum()) if not current_regime.empty else 0
    rows = [
        comparison_row("regime_enough_support_group_count", prev_enough, current_enough, "process-regime calcium dose-response"),
        comparison_row("stable_candidate_count", prev_stable, current_stable, "optimal calcium band map"),
        comparison_row("passed_interaction_count", prev_passed, int(len(current_passed)), "calcium-context interaction screening"),
        comparison_row("previous_ir_incremental_signal", prev_mediation.get("ir_incremental_signal"), current_mediation.get("ir_lag_incremental_signal"), "IR vs IR-lag incremental flag"),
        comparison_row("previous_calcium_ir_interaction_signal", prev_mediation.get("calcium_ir_interaction_signal"), current_mediation.get("calcium_ir_lag_interaction_signal"), "IR vs IR-lag interaction flag"),
        comparison_row("recommended_next_step", prev_next, current_next, "decision comparison"),
        comparison_row("top_context_overlap_count", len(previous_top_context), len(previous_top_context & current_top_context), "top interaction context overlap"),
        comparison_row("bromine_neutral_alkali_rubber_flow_remain_important", True, bool(important & current_top_context), "key process context persistence"),
    ]
    summary = {
        "previous_stable_candidate_count": prev_stable,
        "current_stable_candidate_count": current_stable,
        "previous_passed_interaction_count": prev_passed,
        "current_passed_interaction_count": int(len(current_passed)),
        "previous_ir_mediation_flags": prev_mediation,
        "current_ir_lag_mediation_flags": current_mediation,
        "top_context_overlap": sorted(previous_top_context & current_top_context),
        "key_contexts_remain_important": sorted(important & current_top_context),
        "previous_recommended_next_step": prev_next,
        "current_recommended_next_step": current_next,
    }
    return pd.DataFrame(rows), summary


def comparison_row(metric: str, previous: object, current: object, note: str) -> dict[str, object]:
    return {
        "comparison_metric": metric,
        "previous_value": previous,
        "current_value": current,
        "note": note,
    }


def decide_next(
    stable_count: int,
    passed_interaction_count: int,
    ir_strat_enough: int,
    mediation: dict[str, object],
    suspicious_count: int,
) -> str:
    if stable_count >= 2 and passed_interaction_count >= 1 and ir_strat_enough >= 3 and (
        mediation["ir_lag_to_t90_risk_signal"]
        or mediation["ir_lag_incremental_signal"]
        or mediation["calcium_ir_lag_interaction_signal"]
    ):
        return "define_regime_specific_calcium_band_rules_with_ir_lag_context"
    if stable_count >= 2 and passed_interaction_count >= 1:
        return "define_regime_specific_calcium_band_rules_without_ir"
    if stable_count > 0 or passed_interaction_count > 0 or suspicious_count > 0:
        return "audit_promising_regime_cases"
    if ir_strat_enough < 3:
        return "collect_more_data_or_new_features"
    return "stop_policy_work_for_now"


def append_docs(doc_path: Path, report: dict[str, object]) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    title = section_title(doc_path, 18, "基于最佳出口 IR 小时滞的关系发现复验")
    comparison = report["comparison_with_previous_relationship_discovery"]
    lines = [
        "",
        title,
        "",
        "本阶段在出口 IR 小时滞敏感性评估之后，使用最佳在线安全 IR 滞后特征重新运行钙单耗、工况与 T90 风险关系发现。该实验仍然只用于关系发现，不生成钙设定值建议，不进入影子试验，也不构成自动控制。",
        "",
        "### 特征与解释",
        f"- 主钙单耗特征：`{report['primary_dose_feature']}`。",
        f"- IR 滞后特征：`{report['ir_lag_feature_used']}`。",
        "- 该 IR 特征解释为 T-20min 对齐的出口质量波动代理，即尾随 15 分钟 IR 标准差，不是直接 T90 测量。",
        f"- 工况特征：{', '.join(report['context_features_used']) if report['context_features_used'] else '无'}。",
        "",
        "### 复验摘要",
        f"- 工况分层剂量响应：有效支持分组数 {report['regime_dose_response_summary']['enough_support_group_count']}。",
        f"- 钙×工况交互筛查：通过项数 {report['interaction_screen_summary']['passed_interaction_count']}。",
        f"- IR-lag 分层剂量响应：有效支持分组数 {report['ir_lag_stratified_summary']['enough_support_group_count']}。",
        f"- IR-lag 中介/驱动诊断：{report['ir_lag_mediation_diagnostic_summary']}。",
        f"- 稳定钙单耗候选数：{report['optimal_band_map_summary']['stable_candidate_count']}。",
        "",
        "### 与第 16 阶段比较",
        f"- 稳定候选数：previous={comparison['previous_stable_candidate_count']}，current={comparison['current_stable_candidate_count']}。",
        f"- 交互通过项：previous={comparison['previous_passed_interaction_count']}，current={comparison['current_passed_interaction_count']}。",
        f"- 关键工况是否保持重要：{comparison['key_contexts_remain_important']}。",
        f"- recommended_next_step：`{report['recommended_next_step']}`。",
        "",
        "### 局限",
        "- 仍为离线观察性分析，不构成因果证明。",
        "- LIMS 标签稀疏，T90 测量精度和人工误差限制仍存在。",
        "- IR 覆盖率有限，IR-lag 只能作为解释/上下文/交互候选，不作为动作触发变量。",
        "- 本阶段不推荐自动控制和影子试验。",
        "",
    ]
    with doc_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    assumptions = [
        "IR-lag is used only as explanatory/context/interaction candidate.",
        "No calcium setpoint recommendations, automatic control, or shadow-trial recommendation is generated.",
        "Relationship-screening models are not production T90 models.",
        "The derived IR-lag feature uses only historical outlet IR ending at T-20min.",
    ]
    feature_report = load_json(args.feature_report)
    dose_report = load_json(args.dose_response_report)
    ir_lag_report = load_json(args.ir_lag_report)
    previous_report = load_json(args.previous_relationship_report)
    data, dose_feature, context_features, ir_feature, ir_creation = load_supervised(args, dose_report, ir_lag_report, warnings)
    target_counts = {target: data[target].value_counts(dropna=False).to_dict() for target in ["y_ok", "y_low", "y_high", "y_out_spec"]}
    overall_rates = {
        "ok_rate": float(data["y_ok"].mean()),
        "low_rate": float(data["y_low"].mean()),
        "high_rate": float(data["y_high"].mean()),
        "out_spec_rate": float(data["y_out_spec"].mean()),
    }

    regime_table = experiment_a_regime_dose_response(data, dose_feature, context_features, args.n_bins)
    interaction_table = experiment_b_interaction_screen_ir_lag(data, dose_feature, context_features)
    ir_strat_table = experiment_c_ir_lag_stratified(data, dose_feature, ir_feature, args.n_bins)
    mediation_table, mediation_summary = experiment_d_ir_lag_mediation(data, dose_feature, context_features, ir_feature, args.n_bins)
    band_map = experiment_e_band_map(regime_table, overall_rates)
    band_map = add_ir_columns_to_band_map(band_map, data, ir_feature)

    enough_regime_count = int((regime_table["support_level"] == "enough_support").sum()) if not regime_table.empty else 0
    passed_interactions = interaction_table[interaction_table["passed_interaction_screen"].astype(bool)] if not interaction_table.empty else pd.DataFrame()
    suspicious_count = int(interaction_table["suspicious_large_delta"].sum()) if "suspicious_large_delta" in interaction_table.columns else 0
    enough_ir_strat = int((ir_strat_table["support_level"] == "enough_support").sum()) if not ir_strat_table.empty else 0
    stable_band = band_map[band_map["risk_note"].str.contains("stable_candidate", na=False)] if not band_map.empty else pd.DataFrame()

    recommended_next_step = decide_next(
        stable_count=len(stable_band),
        passed_interaction_count=len(passed_interactions),
        ir_strat_enough=enough_ir_strat,
        mediation=mediation_summary,
        suspicious_count=suspicious_count,
    )
    comparison_table, comparison_summary = compare_with_previous(
        previous_report,
        args.previous_regime_output,
        args.previous_interaction_output,
        regime_table,
        interaction_table,
        band_map,
        mediation_summary,
        recommended_next_step,
    )

    write_csv(args.regime_output, regime_table)
    write_csv(args.interaction_output, interaction_table)
    write_csv(args.ir_strat_output, ir_strat_table)
    write_csv(args.mediation_output, mediation_table)
    write_csv(args.band_map_output, band_map)
    write_csv(args.comparison_output, comparison_table)

    strongest = []
    if not passed_interactions.empty:
        strongest = passed_interactions.sort_values(["delta_auc", "delta_ap"], ascending=[False, False])[
            ["context_feature", "target", "delta_auc", "delta_ap", "suspicious_large_delta"]
        ].head(10).to_dict(orient="records")
    key_findings = [
        f"IR-lag 特征 {ir_feature} 有效样本数为 {int(data[ir_feature].notna().sum())}。",
        f"稳定钙单耗候选数为 {len(stable_band)}，Stage 16 为 {comparison_summary.get('previous_stable_candidate_count')}。",
        f"钙×工况交互通过 {len(passed_interactions)} 项，Stage 16 为 {comparison_summary.get('previous_passed_interaction_count')}。",
        f"IR-lag 风险分层信号为 {mediation_summary['ir_lag_to_t90_risk_signal']}，增量信号为 {mediation_summary['ir_lag_incremental_signal']}。",
        "IR-lag 适合作为规则定义时的解释/上下文候选，不作为动作触发变量。",
    ]

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features_path": str(args.features),
        "feature_report_path": str(args.feature_report),
        "dose_response_report_path": str(args.dose_response_report),
        "data_with_ir_path": str(args.data_with_ir),
        "ir_lag_report_path": str(args.ir_lag_report),
        "previous_relationship_report_path": str(args.previous_relationship_report),
        "row_count": int(len(data)),
        "t90_non_null_count": int(data["t90"].notna().sum()),
        "target_counts": target_counts,
        "primary_dose_feature": dose_feature,
        "context_features_used": context_features,
        "ir_lag_feature_used": ir_feature,
        "ir_lag_feature_creation": ir_creation,
        "experiment_outputs": {
            "regime_output": str(args.regime_output),
            "interaction_output": str(args.interaction_output),
            "ir_strat_output": str(args.ir_strat_output),
            "mediation_output": str(args.mediation_output),
            "band_map_output": str(args.band_map_output),
            "comparison_output": str(args.comparison_output),
        },
        "regime_dose_response_summary": {
            "row_count": int(len(regime_table)),
            "enough_support_group_count": enough_regime_count,
            "best_bin_rows": int(regime_table["best_dose_bin_in_regime"].sum()) if not regime_table.empty else 0,
        },
        "interaction_screen_summary": {
            "row_count": int(len(interaction_table)),
            "passed_interaction_count": int(len(passed_interactions)),
            "suspicious_large_delta_count": suspicious_count,
            "strongest_context_relationships": strongest,
        },
        "ir_lag_stratified_summary": {
            "row_count": int(len(ir_strat_table)),
            "enough_support_group_count": enough_ir_strat,
            "best_bin_rows": int(ir_strat_table["best_dose_bin_in_regime"].sum()) if not ir_strat_table.empty else 0,
            "ir_lag_non_null_rate": float(data[ir_feature].notna().mean()),
        },
        "ir_lag_mediation_diagnostic_summary": mediation_summary,
        "optimal_band_map_summary": {
            "row_count": int(len(band_map)),
            "stable_candidate_count": int(len(stable_band)),
            "high_t90_risk_candidate_count": int(band_map["risk_note"].str.contains("high_t90_risk", na=False).sum()) if not band_map.empty else 0,
            "top_candidates": band_map.sort_values(["best_ok_rate", "sample_count"], ascending=[False, False]).head(10).to_dict(orient="records") if not band_map.empty else [],
        },
        "comparison_with_previous_relationship_discovery": comparison_summary,
        "key_findings": key_findings,
        "warnings": warnings,
        "assumptions": assumptions,
        "recommended_next_step": recommended_next_step,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(report), handle, ensure_ascii=False, indent=2)
    append_docs(args.doc, report)

    print("IR-lag relationship discovery summary")
    print(f"Primary dose feature: {dose_feature}")
    print(f"IR lag feature used: {ir_feature}")
    print(f"Regime groups with enough support: {enough_regime_count}")
    if not passed_interactions.empty:
        top = passed_interactions.sort_values(["delta_auc", "delta_ap"], ascending=[False, False]).head(5)
        print("Strongest interactions:")
        print(top[["context_feature", "target", "delta_auc", "delta_ap", "suspicious_large_delta"]].to_string(index=False))
    else:
        print("Strongest interactions: none passed")
    print(f"IR-lag mediation flags: {mediation_summary}")
    print(f"Stable calcium band candidates: {len(stable_band)}")
    print(f"Comparison with Stage 16: {comparison_summary}")
    print(f"Recommended next step: {recommended_next_step}")
    print(f"Documentation appended: {args.doc}")


if __name__ == "__main__":
    main()
