from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


SAFETY_CONSTRAINTS = {
    "monitor_only": True,
    "automatic_control": False,
    "dcs_writeback": False,
    "no_operational_increase_hint": True,
    "human_review_required_before_deployment": True,
}


def configure_chinese_font() -> None:
    preferred = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pre-deployment evidence closure for the stable calcium safe-band MVP."
    )
    parser.add_argument("--safe-band-final-report", type=Path, default=Path("runs/ca_safe_band_mvp/ca_safe_band_mvp_finalization_report.json"))
    parser.add_argument("--safe-band-dry-run", type=Path, default=Path("runs/ca_safe_band_mvp/final_monitor_dry_run.parquet"))
    parser.add_argument("--safe-band-rule-summary", type=Path, default=Path("runs/ca_safe_band_mvp/final_rule_summary.csv"))
    parser.add_argument("--safe-band-risk-summary", type=Path, default=Path("runs/ca_safe_band_mvp/final_risk_summary.csv"))
    parser.add_argument("--evidence-synthesis-report", type=Path, default=Path("runs/ca_t90_relationship_evidence_synthesis/ca_t90_relationship_evidence_synthesis_report.json"))
    parser.add_argument("--regime-threshold-report", type=Path, default=Path("runs/regime_specific_ca_t90_thresholds/regime_specific_ca_t90_threshold_report.json"))
    parser.add_argument("--cluster-specific-report", type=Path, default=Path("runs/cluster_specific_ca_t90_relationship/cluster_specific_ca_t90_relationship_report.json"))
    parser.add_argument("--manual-explanation-report", type=Path, default=Path("runs/ca_t90_manual_review_explanation_layer/ca_t90_manual_review_layer_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/predeployment_evidence_closure"))
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--method-doc", type=Path, default=Path("docs/ca_safe_band_mvp_method_and_dataflow.md"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    return parser.parse_args()


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize(v) for v in value]
    if hasattr(value, "item"):
        try:
            return sanitize(value.item())
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def resolve_input(path: Path, warnings: list[str], required: bool = False) -> Path | None:
    if path.exists():
        return path
    matches: list[Path] = []
    if Path("runs").exists():
        matches.extend(sorted(Path("runs").rglob(path.name)))
    if not matches and path.name == "safe_band_artifact.json":
        matches.extend(sorted(Path("models").rglob(path.name)))
    if matches:
        warnings.append(f"输入路径不存在，已使用递归匹配文件: {path} -> {matches[0]}")
        return matches[0]
    msg = f"输入文件缺失: {path}"
    if required:
        raise FileNotFoundError(msg)
    warnings.append(msg)
    return None


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_parquet(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def first_row(df: pd.DataFrame, **conditions: str) -> dict[str, Any]:
    if df.empty:
        return {}
    mask = pd.Series(True, index=df.index)
    for col, val in conditions.items():
        if col not in df.columns:
            return {}
        mask &= df[col].astype(str).eq(val)
    if not mask.any():
        return {}
    return df.loc[mask].iloc[0].to_dict()


def safe_float(row: dict[str, Any], key: str) -> float | None:
    val = row.get(key)
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except Exception:
        return None


def delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def build_safe_band_review(
    final_report: dict[str, Any],
    dry_run: pd.DataFrame,
    risk: pd.DataFrame,
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if risk.empty and not dry_run.empty:
        risk = compute_risk_summary_from_dry_run(dry_run)
        warnings.append("final_risk_summary 缺失，已从 dry-run 表重新计算风险摘要。")

    split = "test_like" if "split" in risk.columns and risk["split"].astype(str).eq("test_like").any() else "all"
    inside = first_row(risk, split=split, interval_position="inside_band")
    outside = first_row(risk, split=split, interval_position="outside_band")
    above = first_row(risk, split=split, interval_position="above_band")
    below = first_row(risk, split=split, interval_position="below_band")

    if dry_run.empty:
        coverage = None
    else:
        valid_recs = dry_run["recommended_ca_consumption_min"].notna() if "recommended_ca_consumption_min" in dry_run else pd.Series(False, index=dry_run.index)
        coverage = float(valid_recs.mean()) if len(dry_run) else None

    inside_count = safe_float(inside, "sample_count") or 0
    outside_count = safe_float(outside, "sample_count") or 0
    inside_high = safe_float(inside, "high_rate")
    outside_high = safe_float(outside, "high_rate")
    inside_out = safe_float(inside, "out_spec_rate")
    outside_out = safe_float(outside, "out_spec_rate")

    baseline_defensible = bool(
        inside_count >= 30
        and outside_count >= 30
        and inside_high is not None
        and outside_high is not None
        and inside_high < outside_high
        and (inside_out is None or outside_out is None or inside_out < outside_out)
    )

    summary = {
        "evaluation_split": split,
        "recommendation_coverage": coverage,
        "inside_band_sample_count": inside_count,
        "outside_band_sample_count": outside_count,
        "inside_band_ok_rate": safe_float(inside, "ok_rate"),
        "inside_band_high_rate": inside_high,
        "inside_band_low_rate": safe_float(inside, "low_rate"),
        "inside_band_out_spec_rate": inside_out,
        "outside_band_ok_rate": safe_float(outside, "ok_rate"),
        "outside_band_high_rate": outside_high,
        "outside_band_low_rate": safe_float(outside, "low_rate"),
        "outside_band_out_spec_rate": outside_out,
        "above_band_high_rate": safe_float(above, "high_rate"),
        "below_band_low_rate": safe_float(below, "low_rate"),
        "inside_vs_outside_ok_rate_delta": delta(safe_float(inside, "ok_rate"), safe_float(outside, "ok_rate")),
        "inside_vs_outside_high_rate_delta": delta(inside_high, outside_high),
        "inside_vs_outside_low_rate_delta": delta(safe_float(inside, "low_rate"), safe_float(outside, "low_rate")),
        "inside_vs_outside_out_spec_rate_delta": delta(inside_out, outside_out),
        "final_strategy": final_report.get("final_strategy"),
        "product_positioning": final_report.get("product_positioning"),
        "action_visibility_policy": final_report.get("action_visibility_policy", {}),
        "safe_band_baseline_defensible": baseline_defensible,
    }

    rows = []
    for name, val in summary.items():
        if name == "action_visibility_policy":
            val = json.dumps(sanitize(val), ensure_ascii=False)
        rows.append({"metric": name, "value": val, "interpretation_cn": safe_band_metric_interpretation(name, val)})
    return pd.DataFrame(rows), summary


def compute_risk_summary_from_dry_run(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = df.copy()
    if "split" not in work.columns:
        work["split"] = "all"
    if "interval_position" not in work.columns:
        return pd.DataFrame()
    for split in list(work["split"].dropna().unique()) + ["all"]:
        part = work if split == "all" else work[work["split"].eq(split)]
        if part.empty:
            continue
        for pos in ["inside_band", "above_band", "below_band", "missing"]:
            sub = part[part["interval_position"].eq(pos)]
            if sub.empty:
                continue
            rows.append(risk_row(split, pos, sub))
        outside = part[part["interval_position"].isin(["above_band", "below_band"])]
        if not outside.empty:
            rows.append(risk_row(split, "outside_band", outside))
    return pd.DataFrame(rows)


def risk_row(split: str, pos: str, sub: pd.DataFrame) -> dict[str, Any]:
    out = {"split": split, "interval_position": pos, "sample_count": len(sub)}
    for col, metric in [("y_ok", "ok_rate"), ("y_high", "high_rate"), ("y_low", "low_rate"), ("y_out_spec", "out_spec_rate")]:
        out[metric] = float(sub[col].mean()) if col in sub and len(sub) else None
    out["mean_t90"] = float(sub["t90"].mean()) if "t90" in sub and len(sub) else None
    return out


def safe_band_metric_interpretation(name: str, value: Any) -> str:
    mapping = {
        "safe_band_baseline_defensible": "若为 true，表示区间内样本支持数和区间内外风险分离满足监测基线要求。",
        "inside_vs_outside_high_rate_delta": "负值表示区间内高 T90 风险低于区间外。",
        "inside_vs_outside_out_spec_rate_delta": "负值表示区间内总体出规格风险低于区间外。",
        "final_strategy": "冻结版本采用中位数聚合策略。",
        "product_positioning": "当前定位应保持为稳定安全带 MVP。",
    }
    return mapping.get(name, "")


def build_regime_basis_review(rule_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rule_df.empty or "regime_feature" not in rule_df.columns:
        return pd.DataFrame(), {
            "q33_q66_regime_basis_valid_for_v1": False,
            "reason_cn": "缺少最终规则摘要，无法复核 q33/q66 工况基础。",
        }
    df = rule_df.copy()
    for col in ["monitor_chain_candidate", "manual_review_only", "reject_or_refine"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    rows = []
    for feature, group in df.groupby("regime_feature", dropna=False):
        accepted = group[group.get("rule_status", "").astype(str).eq("accept_for_manual_case_review")] if "rule_status" in group else group
        monitor_count = int(group["monitor_chain_candidate"].sum()) if "monitor_chain_candidate" in group else 0
        rows.append(
            {
                "regime_feature": feature,
                "rule_count": len(group),
                "bins_covered": ",".join(sorted(group["regime_bin"].dropna().astype(str).unique())) if "regime_bin" in group else "",
                "accepted_rule_count": len(accepted),
                "monitor_candidate_count": monitor_count,
                "recommended_interval_min_median": float(group["recommended_dose_min"].median()) if "recommended_dose_min" in group else None,
                "recommended_interval_max_median": float(group["recommended_dose_max"].median()) if "recommended_dose_max" in group else None,
                "recommended_target_median": float(group["recommended_dose_target"].median()) if "recommended_dose_target" in group else None,
                "notes_cn": "覆盖最终规则中的单变量低/中/高工况；仍仅作为 V1 冻结基线，不由聚类直接替代。",
            }
        )
    review = pd.DataFrame(rows).sort_values(["rule_count", "regime_feature"], ascending=[False, True])
    key_features = set(
        [
            "bromine_feed_win_60_mean",
            "rubber_flow_2_win_60_mean",
            "tank_rubber_conc_win_60_mean",
            "neutral_alkali_feed_win_60_mean",
            "esbo_feed_win_60_mean",
            "r513_temp_win_60_mean",
            "r514_temp_win_60_mean",
        ]
    )
    features_used = set(review["regime_feature"].dropna().astype(str))
    valid = bool(len(features_used & key_features) >= 3 and len(df) >= 5)
    summary = {
        "final_rule_count": int(len(df)),
        "regime_feature_count": int(review["regime_feature"].nunique()),
        "regime_features_used": sorted(features_used),
        "regime_bins_used": sorted(df["regime_bin"].dropna().astype(str).unique()) if "regime_bin" in df else [],
        "monitor_candidate_rule_count": int(df["monitor_chain_candidate"].sum()) if "monitor_chain_candidate" in df else None,
        "rule_grade_counts": df["rule_grade"].value_counts(dropna=False).to_dict() if "rule_grade" in df else {},
        "rules_dominated_by_few_features": bool(review["rule_count"].max() / max(len(df), 1) > 0.40),
        "rule_intervals_stable": bool(df["recommended_dose_target"].quantile(0.75) - df["recommended_dose_target"].quantile(0.25) <= 0.00035) if "recommended_dose_target" in df else None,
        "q33_q66_regime_basis_valid_for_v1": valid,
        "reason_cn": "规则覆盖多个关键过程变量，且当前关系发现没有直接推翻 q33/q66 单变量工况基础；聚类仅用于解释和监测上下文。",
    }
    return review, summary


def build_relationship_review(
    evidence: dict[str, Any],
    regime: dict[str, Any],
    cluster: dict[str, Any],
    manual_expl: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    synthesis_decision = evidence.get("algorithm_modification_decision")
    matrix = evidence.get("evidence_matrix", [])
    matrix_lookup = {row.get("evidence_item"): row for row in matrix if isinstance(row, dict)}
    regime_synth = regime.get("global_synthesis", {})
    cluster_flags = cluster.get("decision_flags", {})
    global_conclusion_raw = evidence.get("global_conclusion", {})
    global_conclusion = global_conclusion_raw if isinstance(global_conclusion_raw, dict) else {}

    rows = [
        {
            "evidence_source": "global_threshold_relation",
            "relationship_conclusion": evidence_item_text(matrix_lookup.get("global_ca_t90_relation")),
            "support_level": evidence_item_level(matrix_lookup.get("global_ca_t90_relation")),
            "supporting_context_count": None,
            "contradictory_context_count": None,
            "algorithm_implication_cn": "可作为关系解释，不作为当前区间算法修改依据。",
            "deployment_implication_cn": "用于人工复核提示中的背景证据。",
        },
        {
            "evidence_source": "regime_specific_thresholds",
            "relationship_conclusion": f"relation_type={regime_synth.get('relation_type')}",
            "support_level": "moderate" if regime_synth.get("relation_type") == "broadly_consistent" else "weak",
            "supporting_context_count": regime_synth.get("positive_relation_regime_count"),
            "contradictory_context_count": regime_synth.get("contradictory_regime_count"),
            "algorithm_implication_cn": "支持解释层，但矛盾工况要求暂缓算法修改。",
            "deployment_implication_cn": "V1/V1.1 均需保留人工复核。",
        },
        {
            "evidence_source": "cluster_specific_relationships",
            "relationship_conclusion": f"cluster_specific_supported={cluster_flags.get('cluster_specific_ca_t90_supported')}",
            "support_level": "moderate" if cluster_flags.get("cluster_specific_ca_t90_supported") else "weak",
            "supporting_context_count": count_cluster_relation(cluster, "supporting"),
            "contradictory_context_count": count_cluster_relation(cluster, "contradictory"),
            "algorithm_implication_cn": "聚类可增强解释和监测上下文，但不是冻结 V1 的工况划分基础。",
            "deployment_implication_cn": "可进入人工看板说明，不直接改变推荐区间。",
        },
        {
            "evidence_source": "manual_explanation_layer_package",
            "relationship_conclusion": "已生成" if manual_expl else "未发现独立解释层报告",
            "support_level": "sufficient" if manual_expl else "pending",
            "supporting_context_count": None,
            "contradictory_context_count": None,
            "algorithm_implication_cn": "解释层不改变推荐区间。",
            "deployment_implication_cn": "若未生成或未复核，部署测试应先用 V1 或等待人工复核。",
        },
    ]
    review_df = pd.DataFrame(rows)

    high_risk_supported = any(
        row.get("evidence_item") == "global_high_calcium_high_t90_risk"
        and row.get("support_level") in {"moderate", "strong"}
        for row in matrix
        if isinstance(row, dict)
    )
    relationship_supports_explanation = bool(
        high_risk_supported
        or regime_synth.get("positive_relation_regime_count", 0) >= 15
        or synthesis_decision in {"explanation_layer_only", "prepare_context_specific_algorithm_later"}
    )
    contradictory_count = int(regime_synth.get("contradictory_regime_count") or 0) + count_cluster_relation(cluster, "contradictory")
    threshold_stable = bool(regime_synth.get("threshold_evidence_regime_count", 0) >= 20 and contradictory_count <= 1)
    safe_band_consistency = global_conclusion.get("safe_band_consistency") or {}
    safe_band_strong = safe_band_consistency.get("threshold_near_safe_band_upper") is True
    relationship_justifies_change = bool(
        relationship_supports_explanation
        and contradictory_count <= 1
        and threshold_stable
        and safe_band_strong
        and synthesis_decision == "modify_recommender_now"
    )

    summary = {
        "global_evidence_strength": global_conclusion.get("evidence_strength"),
        "positive_relation_supported": global_conclusion.get("positive_relation_supported"),
        "high_calcium_high_t90_risk_supported": high_risk_supported,
        "nonlinear_threshold_supported": evidence_item_level(matrix_lookup.get("nonlinear_threshold")),
        "flat_safe_region_supported": evidence_item_level(matrix_lookup.get("flat_safe_region")),
        "relation_type": regime_synth.get("relation_type"),
        "positive_regime_count": regime_synth.get("positive_relation_regime_count"),
        "high_risk_regime_count": regime_synth.get("high_calcium_high_t90_risk_regime_count"),
        "contradictory_regime_count": regime_synth.get("contradictory_regime_count"),
        "cluster_specific_supporting_count": count_cluster_relation(cluster, "supporting"),
        "cluster_specific_contradictory_count": count_cluster_relation(cluster, "contradictory"),
        "algorithm_modification_decision_from_synthesis": synthesis_decision,
        "manual_explanation_layer_report_available": bool(manual_expl),
        "relationship_supports_explanation_layer": relationship_supports_explanation,
        "relationship_justifies_algorithm_change_now": relationship_justifies_change,
    }
    return review_df, summary


def evidence_item_text(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    return str(row.get("interpretation_cn") or row.get("key_numbers") or "")


def evidence_item_level(row: dict[str, Any] | None) -> str:
    if not row:
        return "insufficient"
    return str(row.get("support_level") or "insufficient")


def count_cluster_relation(cluster_report: dict[str, Any], relation_type: str) -> int:
    rows = cluster_report.get("cluster_ca_t90_relations", [])
    return int(sum(1 for row in rows if str(row.get("cluster_relation_type")).lower() == relation_type))


def contexts_from_evidence(evidence: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def to_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
        cols = [
            "context_type",
            "context_name",
            "sample_count",
            "spearman_ca_t90",
            "spearman_ca_y_high",
            "high_rate_delta",
            "threshold",
            "ok_rate",
            "high_rate",
            "low_rate",
            "interpretation_cn",
            "caution_cn",
        ]
        out = pd.DataFrame(rows)
        for col in cols:
            if col not in out.columns:
                out[col] = None
        return out[cols]

    return (
        to_df(evidence.get("strongest_supporting_contexts", [])),
        to_df(evidence.get("contradictory_contexts", [])),
        to_df(evidence.get("mixed_contexts", [])),
    )


def build_algorithm_decision_matrix(
    safe_summary: dict[str, Any],
    regime_summary: dict[str, Any],
    relationship_summary: dict[str, Any],
) -> tuple[pd.DataFrame, str, str, str, str]:
    safe_ok = bool(safe_summary.get("safe_band_baseline_defensible"))
    regime_ok = bool(regime_summary.get("q33_q66_regime_basis_valid_for_v1"))
    supports_expl = bool(relationship_summary.get("relationship_supports_explanation_layer"))
    justifies_change = bool(relationship_summary.get("relationship_justifies_algorithm_change_now"))
    expl_ready = bool(relationship_summary.get("manual_explanation_layer_report_available"))

    if not safe_ok:
        algorithm_decision = "keep_current_baseline"
        deploy_decision = "do_not_deploy_fix_baseline"
        next_step = "fix_baseline_before_factory_test"
        reason = "安全带区间内外风险证据不足，需先修复或复核基线。"
    elif justifies_change:
        algorithm_decision = "prepare_context_specific_algorithm_later"
        deploy_decision = "design_V2_later_do_not_deploy_as_V2"
        next_step = "design_V2_context_specific_recommender_experiment"
        reason = "关系证据有算法研究价值，但仍需新一轮离线验证，不能直接作为当前部署版本。"
    elif supports_expl and expl_ready:
        algorithm_decision = "explanation_layer_only"
        deploy_decision = "V1_1_explanation_layer_candidate_after_human_review"
        next_step = "human_review_predeployment_evidence"
        reason = "基线可防守，解释层已形成且仅用于人工复核，不改变推荐区间。"
    elif supports_expl and not expl_ready:
        algorithm_decision = "explanation_layer_only"
        deploy_decision = "V1_monitor_only_candidate"
        next_step = "prepare_V1_monitor_only_factory_test"
        reason = "关系发现支持解释层，但独立解释层尚未形成或尚未人工复核；当前可先保持 V1 监测基线。"
    elif safe_ok and regime_ok:
        algorithm_decision = "keep_current_baseline"
        deploy_decision = "V1_monitor_only_candidate"
        next_step = "prepare_V1_monitor_only_factory_test"
        reason = "当前证据不足以加入解释层或修改算法，保留冻结安全带基线。"
    else:
        algorithm_decision = "insufficient_evidence"
        deploy_decision = "defer_deployment_until_evidence_review"
        next_step = "human_review_predeployment_evidence"
        reason = "证据链存在缺口，需要人工复核后再决定。"

    rows = [
        {
            "decision_item": "safe_band_baseline_defensible",
            "status": safe_ok,
            "deployment_implication_cn": "true 表示 V1 监测基线具备厂区测试前的历史证据支撑。",
        },
        {
            "decision_item": "q33_q66_regime_basis_valid_for_v1",
            "status": regime_ok,
            "deployment_implication_cn": "true 表示冻结规则基础仍可用于 V1，不被聚类结果替代。",
        },
        {
            "decision_item": "relationship_supports_explanation_layer",
            "status": supports_expl,
            "deployment_implication_cn": "true 表示关系发现可用于人工复核解释，不改变区间。",
        },
        {
            "decision_item": "relationship_justifies_algorithm_change_now",
            "status": justifies_change,
            "deployment_implication_cn": "true 才允许考虑立即改算法；当前预期为 false。",
        },
        {
            "decision_item": "manual_explanation_layer_report_available",
            "status": expl_ready,
            "deployment_implication_cn": "true 表示 V1.1 解释层候选已打包；false 时优先 V1 或先补解释层。",
        },
        {
            "decision_item": "algorithm_modification_decision",
            "status": algorithm_decision,
            "deployment_implication_cn": reason,
        },
        {
            "decision_item": "deploy_test_decision",
            "status": deploy_decision,
            "deployment_implication_cn": "厂区测试版本建议。",
        },
    ]
    return pd.DataFrame(rows), algorithm_decision, deploy_decision, reason, next_step


def build_logging_schema() -> pd.DataFrame:
    rows = [
        ("timestamp", "datetime", True, "推荐输出时间戳。"),
        ("raw_dcs_input_availability", "json/string", True, "记录原始 DCS 点位是否齐备。"),
        ("engineered_feature_values", "json", True, "记录进入运行包的窗口特征值。"),
        ("recommended_ca_consumption_min", "float", True, "推荐钙单耗区间下限。"),
        ("recommended_ca_consumption_max", "float", True, "推荐钙单耗区间上限。"),
        ("recommended_ca_consumption_target", "float", True, "区间中心，仅作展示，不是单点设定值。"),
        ("current_ca_consumption", "float", True, "当前窗口钙单耗。"),
        ("interval_position", "string", True, "inside_band / above_band / below_band / missing。"),
        ("action_visibility", "string", True, "monitor_only / manual_review_required / diagnostic_only / no_recommendation。"),
        ("manual_review_required", "bool", True, "是否需要人工复核。"),
        ("supporting_contexts", "json/string", False, "关系发现中支持高钙高 T90 风险的上下文。"),
        ("contradictory_contexts", "json/string", False, "关系发现中矛盾或反向上下文。"),
        ("relationship_explanation_category", "string", False, "supporting / contradictory / mixed / neutral。"),
        ("warning_flags", "json/string", True, "输入缺失、边界外、IR 缺失等告警。"),
        ("later_lims_t90", "float", False, "后续 LIMS 回填 T90，不能用同一时刻 T90 评价。"),
        ("t90_sample_time", "datetime", False, "LIMS 样品时间。"),
        ("estimated_residence_time_alignment_key", "string/int", False, "用于把推荐时刻与后续 T90 对齐的停留时间键。"),
        ("operator_manual_ca_change_after_recommendation", "bool/string", False, "记录推荐后人工是否调整钙单耗。"),
    ]
    return pd.DataFrame(rows, columns=["field_name", "data_type", "required_for_plant_test", "description_cn"])


def build_inventory(paths: dict[str, Path | None], reports: dict[str, dict[str, Any]], dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        available = bool(path and path.exists())
        rows.append(
            {
                "source_name": name,
                "source_path": str(path) if path else None,
                "available": available,
                "source_type": "json_report" if str(name).endswith("report") else "table_or_parquet",
                "row_count": int(len(dfs.get(name, pd.DataFrame()))) if name in dfs else None,
                "top_level_keys": ",".join(list(reports.get(name, {}).keys())[:20]) if name in reports and reports.get(name) else None,
                "notes_cn": "可用于证据闭环" if available else "缺失，已在报告中记录警告",
            }
        )
    return pd.DataFrame(rows)


def save_tables(
    output_dir: Path,
    table_dir: Path,
    tables: dict[str, pd.DataFrame],
    human_tables: list[str],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, df in tables.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written[name] = str(path)
        if name in human_tables:
            human_path = table_dir / f"{name}.csv"
            df.to_csv(human_path, index=False, encoding="utf-8-sig")
            written[f"reports/{name}"] = str(human_path)
    return written


def plot_decision_flow(path: Path, algorithm_decision: str, deploy_decision: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    boxes = [
        ("安全带基线证据", 0.08, 0.62),
        ("关系发现证据", 0.36, 0.62),
        ("解释层适用性", 0.64, 0.62),
        ("部署测试决策", 0.36, 0.20),
    ]
    for label, x, y in boxes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f0f6ff", edgecolor="#4c78a8"),
            transform=ax.transAxes,
        )
    arrows = [((0.18, 0.62), (0.27, 0.62)), ((0.46, 0.62), (0.55, 0.62)), ((0.64, 0.52), (0.46, 0.28))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.8, color="#444"))
    ax.text(0.36, 0.04, f"算法判断: {algorithm_decision}\n部署建议: {deploy_decision}", ha="center", va="bottom", fontsize=11, transform=ax.transAxes)
    ax.set_title("厂区部署测试前决策流程", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_safe_band_risk(path: Path, safe_summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["inside", "outside", "above", "below"]
    ok = [
        safe_summary.get("inside_band_ok_rate"),
        safe_summary.get("outside_band_ok_rate"),
        None,
        None,
    ]
    high = [
        safe_summary.get("inside_band_high_rate"),
        safe_summary.get("outside_band_high_rate"),
        safe_summary.get("above_band_high_rate"),
        None,
    ]
    low = [
        safe_summary.get("inside_band_low_rate"),
        safe_summary.get("outside_band_low_rate"),
        None,
        safe_summary.get("below_band_low_rate"),
    ]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (vals, name, color) in enumerate([(ok, "ok_rate", "#59a14f"), (high, "high_rate", "#e15759"), (low, "low_rate", "#4e79a7")]):
        y = [np.nan if v is None else float(v) for v in vals]
        ax.bar(x + (idx - 1) * width, y, width=width, label=name, color=color, alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("比例")
    ax.set_title("稳定安全带区间内外风险摘要")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_relationship_contexts(path: Path, supporting: pd.DataFrame, contradictory: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    support_count = len(supporting)
    contra_count = len(contradictory)
    mixed_count = 0
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["支持上下文", "矛盾上下文", "混合/待判定"], [support_count, contra_count, mixed_count], color=["#59a14f", "#e15759", "#f28e2b"], alpha=0.88)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{int(bar.get_height())}", ha="center", va="bottom")
    ax.set_ylabel("数量")
    ax.set_title("钙单耗-T90 支持与矛盾上下文摘要")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def update_method_doc(path: Path) -> None:
    section_title = "## 厂区部署测试前证据闭环"
    section = f"""

{section_title}

当前冻结基线为 V1 稳定钙单耗安全带 MVP。V1 继续使用 q33/q66 单变量工况划分、低/中/高工况规则匹配和 `median_aggregation_baseline` 聚合策略，仅输出监测状态、推荐区间和人工复核可见性，不执行自动控制。

关系发现阶段的结论只作为解释层证据：高钙与高 T90 风险在全局、多数三分位工况和部分稳健聚类中得到支持，但支持与矛盾上下文并存。因此，关系发现不直接改变推荐区间、不替代 V1 的 q33/q66 规则基础，也不作为 DCS 写回逻辑。

聚类结果用于事后上下文解释和监测看板：稳健 k=5 聚类可帮助识别支持、矛盾或混合工况，但不是当前运行包的工况分裂依据。若后续要进入 V2，需要重新设计上下文特异推荐实验，并完成独立离线验证和人工工艺复核。

V1 厂区测试应记录推荐时刻、原始 DCS 输入可用性、工程化特征、推荐钙单耗区间、当前钙单耗、区间位置、动作可见性、告警标志，以及后续按停留时间对齐的 LIMS T90。不能用同一时刻 T90 评价在线推荐结果。

V1.1 可以在 V1 基础上增加人工复核解释层：展示支持/矛盾/混合上下文和高钙高 T90 风险说明，但不改变推荐区间、不给出加钙操作建议。V2 则需要分工况或聚类上下文算法重新验证后再考虑。

安全边界保持不变：监测模式、无自动控制、无 DCS 写回、无操作性加钙提示，且任何区间内状态都不保证 T90 必然合格。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if section_title in text:
            text = text.split(section_title)[0].rstrip() + section
        else:
            text = text.rstrip() + section
    else:
        text = "# 稳定钙单耗安全带 MVP 方法与数据流\n" + section
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def next_section_number(doc_path: Path, preferred: int) -> int:
    if not doc_path.exists():
        return preferred
    text = doc_path.read_text(encoding="utf-8")
    used = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("## "):
            rest = line[3:].strip()
            num = rest.split(".", 1)[0]
            if num.isdigit():
                used.add(int(num))
    n = preferred
    while n in used:
        n += 1
    return n


def append_experiment_doc(
    path: Path,
    safe_summary: dict[str, Any],
    regime_summary: dict[str, Any],
    relationship_summary: dict[str, Any],
    algorithm_decision: str,
    deploy_decision: str,
    next_step: str,
) -> None:
    num = next_section_number(path, 38)
    section = f"""

## {num}. 厂区部署测试前证据闭环与解释层适用性论证

本阶段用于将稳定钙单耗安全带 MVP、钙单耗-T90 关系发现、分工况验证和聚类解释结果合并为厂区部署测试前的证据闭环。目标不是修改推荐算法，而是判断厂区测试应采用 V1 监测基线、V1.1 人工解释层版本，还是暂缓部署。

安全带基线复核：测试集区间内样本数为 {safe_summary.get('inside_band_sample_count')}，区间内高 T90 率为 {safe_summary.get('inside_band_high_rate')}，区间外高 T90 率为 {safe_summary.get('outside_band_high_rate')}，above_band 高 T90 率为 {safe_summary.get('above_band_high_rate')}。safe_band_baseline_defensible = {safe_summary.get('safe_band_baseline_defensible')}。

q33/q66 工况基础复核：最终规则数为 {regime_summary.get('final_rule_count')}，覆盖工况变量数为 {regime_summary.get('regime_feature_count')}，q33_q66_regime_basis_valid_for_v1 = {regime_summary.get('q33_q66_regime_basis_valid_for_v1')}。稳健聚类仅作为解释和监测上下文，不替代冻结 V1 的单变量三分位规则基础。

关系发现复核：relationship_supports_explanation_layer = {relationship_summary.get('relationship_supports_explanation_layer')}；relationship_justifies_algorithm_change_now = {relationship_summary.get('relationship_justifies_algorithm_change_now')}。当前证据支持将高钙高 T90 风险作为人工复核解释，但支持与矛盾上下文并存，尚不足以直接修改推荐区间算法。

算法修改判断：algorithm_modification_decision = {algorithm_decision}。部署测试建议：deploy_test_decision = {deploy_decision}。推荐下一步：{next_step}。

局限性：全部证据仍来自离线历史数据，不构成因果证明；T90 存在测量和对齐误差；过程上下文存在混杂；厂区测试仍必须为 monitor-only；不实施自动控制，不实施 DCS 写回。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(section)


def main() -> None:
    configure_chinese_font()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    path_map = {
        "safe_band_final_report": resolve_input(args.safe_band_final_report, warnings),
        "safe_band_dry_run": resolve_input(args.safe_band_dry_run, warnings),
        "safe_band_rule_summary": resolve_input(args.safe_band_rule_summary, warnings),
        "safe_band_risk_summary": resolve_input(args.safe_band_risk_summary, warnings),
        "evidence_synthesis_report": resolve_input(args.evidence_synthesis_report, warnings),
        "regime_threshold_report": resolve_input(args.regime_threshold_report, warnings),
        "cluster_specific_report": resolve_input(args.cluster_specific_report, warnings),
        "manual_explanation_report": resolve_input(args.manual_explanation_report, warnings),
    }

    safe_available = any(path_map[k] for k in ["safe_band_final_report", "safe_band_dry_run", "safe_band_risk_summary"])
    relationship_available = any(path_map[k] for k in ["evidence_synthesis_report", "regime_threshold_report", "cluster_specific_report"])
    if not safe_available and not relationship_available:
        raise FileNotFoundError("safe-band evidence and relationship evidence are both unavailable; cannot build closure report.")

    reports = {
        "safe_band_final_report": load_json(path_map["safe_band_final_report"]),
        "evidence_synthesis_report": load_json(path_map["evidence_synthesis_report"]),
        "regime_threshold_report": load_json(path_map["regime_threshold_report"]),
        "cluster_specific_report": load_json(path_map["cluster_specific_report"]),
        "manual_explanation_report": load_json(path_map["manual_explanation_report"]),
    }
    dfs = {
        "safe_band_dry_run": load_parquet(path_map["safe_band_dry_run"]),
        "safe_band_rule_summary": load_csv(path_map["safe_band_rule_summary"]),
        "safe_band_risk_summary": load_csv(path_map["safe_band_risk_summary"]),
    }

    inventory = build_inventory(path_map, reports, dfs)
    safe_band_table, safe_band_summary = build_safe_band_review(
        reports["safe_band_final_report"],
        dfs["safe_band_dry_run"],
        dfs["safe_band_risk_summary"],
        warnings,
    )
    regime_table, regime_summary = build_regime_basis_review(dfs["safe_band_rule_summary"])
    relationship_table, relationship_summary = build_relationship_review(
        reports["evidence_synthesis_report"],
        reports["regime_threshold_report"],
        reports["cluster_specific_report"],
        reports["manual_explanation_report"],
    )
    supporting, contradictory, mixed = contexts_from_evidence(reports["evidence_synthesis_report"])
    decision_matrix, algorithm_decision, deploy_decision, decision_reason, next_step = build_algorithm_decision_matrix(
        safe_band_summary,
        regime_summary,
        relationship_summary,
    )
    logging_schema = build_logging_schema()

    closure_summary = pd.DataFrame(
        [
            {"metric": "safe_band_baseline_defensible", "value": safe_band_summary.get("safe_band_baseline_defensible"), "interpretation_cn": "V1 监测基线是否仍可防守。"},
            {"metric": "q33_q66_regime_basis_valid_for_v1", "value": regime_summary.get("q33_q66_regime_basis_valid_for_v1"), "interpretation_cn": "冻结单变量三分位规则基础是否仍适合 V1。"},
            {"metric": "relationship_supports_explanation_layer", "value": relationship_summary.get("relationship_supports_explanation_layer"), "interpretation_cn": "关系发现是否支持人工复核解释层。"},
            {"metric": "relationship_justifies_algorithm_change_now", "value": relationship_summary.get("relationship_justifies_algorithm_change_now"), "interpretation_cn": "关系发现是否足以现在改推荐区间算法。"},
            {"metric": "algorithm_modification_decision", "value": algorithm_decision, "interpretation_cn": decision_reason},
            {"metric": "deploy_test_decision", "value": deploy_decision, "interpretation_cn": "建议的厂区测试版本。"},
            {"metric": "recommended_next_step", "value": next_step, "interpretation_cn": "下一步行动。"},
        ]
    )

    tables = {
        "predeployment_evidence_inventory": inventory,
        "predeployment_safe_band_baseline_review": safe_band_table,
        "predeployment_regime_basis_review": regime_table,
        "predeployment_relationship_evidence_review": relationship_table,
        "predeployment_algorithm_decision_matrix": decision_matrix,
        "predeployment_plant_test_logging_schema": logging_schema,
        "predeployment_evidence_closure_summary": closure_summary,
    }
    written_tables = save_tables(
        args.output_dir,
        args.table_dir,
        tables,
        [
            "predeployment_evidence_closure_summary",
            "predeployment_regime_basis_review",
            "predeployment_relationship_evidence_review",
            "predeployment_algorithm_decision_matrix",
            "predeployment_plant_test_logging_schema",
        ],
    )

    figures = {
        "decision_flow": args.figure_dir / "predeployment_decision_flow.png",
        "safe_band_risk": args.figure_dir / "predeployment_safe_band_risk_summary.png",
        "relationship_context": args.figure_dir / "predeployment_relationship_context_summary.png",
    }
    plot_decision_flow(figures["decision_flow"], algorithm_decision, deploy_decision)
    plot_safe_band_risk(figures["safe_band_risk"], safe_band_summary)
    plot_relationship_contexts(figures["relationship_context"], supporting, contradictory)

    update_method_doc(args.method_doc)
    append_experiment_doc(args.doc, safe_band_summary, regime_summary, relationship_summary, algorithm_decision, deploy_decision, next_step)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_paths": {k: str(v) if v else None for k, v in path_map.items()},
        "missing_inputs": [k for k, v in path_map.items() if v is None],
        "output_dir": str(args.output_dir),
        "table_dir": str(args.table_dir),
        "figure_dir": str(args.figure_dir),
        "safe_band_baseline_review": safe_band_summary,
        "q33_q66_regime_basis_review": regime_summary,
        "relationship_evidence_review": relationship_summary,
        "explanation_layer_review": {
            "manual_explanation_report_available": bool(reports["manual_explanation_report"]),
            "manual_explanation_report_path": str(path_map["manual_explanation_report"]) if path_map["manual_explanation_report"] else None,
            "explanation_layer_role": "manual_review_only",
            "changes_recommendation_interval": False,
        },
        "algorithm_modification_decision": algorithm_decision,
        "deploy_test_decision": deploy_decision,
        "decision_reason_cn": decision_reason,
        "plant_test_logging_plan": logging_schema.to_dict(orient="records"),
        "runtime_package_modified": False,
        "safety_constraints": SAFETY_CONSTRAINTS,
        "generated_tables": written_tables,
        "generated_figures": {k: str(v) for k, v in figures.items()},
        "warnings": warnings,
        "limitations": [
            "离线历史证据，不构成因果证明。",
            "T90 测量和停留时间对齐存在误差。",
            "过程上下文存在混杂，支持和矛盾上下文并存。",
            "厂区测试仍为 monitor-only，不自动控制，不写回 DCS。",
        ],
        "recommended_next_step": next_step,
    }
    write_json(args.output_dir / "predeployment_evidence_closure_report.json", report)

    print("safe_band_baseline_defensible:", safe_band_summary.get("safe_band_baseline_defensible"))
    print("q33_q66_regime_basis_valid_for_v1:", regime_summary.get("q33_q66_regime_basis_valid_for_v1"))
    print("relationship_supports_explanation_layer:", relationship_summary.get("relationship_supports_explanation_layer"))
    print("relationship_justifies_algorithm_change_now:", relationship_summary.get("relationship_justifies_algorithm_change_now"))
    print("algorithm_modification_decision:", algorithm_decision)
    print("deploy_test_decision:", deploy_decision)
    print("recommended_next_step:", next_step)
    print("runtime_package_modified:", False)
    print("report:", args.output_dir / "predeployment_evidence_closure_report.json")
    print("docs_appended:", args.doc.exists())


if __name__ == "__main__":
    main()
