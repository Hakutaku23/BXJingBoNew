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


DECISION_TO_NEXT_STEP = {
    "no_change_keep_current_baseline": "keep_current_baseline_and_collect_more_data",
    "explanation_layer_only": "build_manual_review_explanation_layer",
    "manual_review_warning_only": "build_manual_review_explanation_layer",
    "prepare_context_specific_algorithm_later": "design_context_specific_recommender_experiment",
    "modify_recommender_now": "design_context_specific_recommender_experiment",
    "insufficient_evidence": "insufficient_evidence_stop",
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
    parser = argparse.ArgumentParser(description="Synthesize calcium-T90 relationship evidence and assess recommendation algorithm readiness.")
    parser.add_argument("--global-threshold-report", type=Path, default=Path("runs/ca_t90_threshold_relation/ca_t90_threshold_relation_report.json"))
    parser.add_argument("--regime-threshold-report", type=Path, default=Path("runs/regime_specific_ca_t90_thresholds/regime_specific_ca_t90_threshold_report.json"))
    parser.add_argument("--clustering-robustness-report", type=Path, default=Path("runs/t90_clustering_robustness_audit/t90_clustering_robustness_audit_report.json"))
    parser.add_argument("--cluster-specific-report", type=Path, default=Path("runs/cluster_specific_ca_t90_relationship/cluster_specific_ca_t90_relationship_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ca_t90_relationship_evidence_synthesis"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--method-doc", type=Path, default=Path("docs/ca_safe_band_mvp_method_and_dataflow.md"))
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


def resolve_report(path: Path, warnings: list[str]) -> Path | None:
    if path.exists():
        return path
    if Path("runs").exists():
        matches = sorted(Path("runs").rglob(path.name))
        if matches:
            warnings.append(f"Report path {path} missing; using recursive match {matches[0]}.")
            return matches[0]
    warnings.append(f"Report missing: {path}")
    return None


def load_report(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def inventory_rows(paths: dict[str, Path | None], reports: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    meta = {
        "global_threshold": ("global", "global calcium-T90 threshold relationship"),
        "regime_threshold": ("regime_specific", "tertile process-regime calcium-T90 validation"),
        "clustering_robustness": ("clustering", "robust operating-regime clustering validation"),
        "cluster_specific": ("cluster_specific", "calcium-T90 validation inside robust clusters"),
    }
    for name, report in reports.items():
        if not report:
            continue
        evidence_type, desc = meta[name]
        key_metrics = {}
        conclusion = None
        usable_algorithm = "partial"
        usable_explanation = "yes"
        if name == "global_threshold":
            flags = report.get("decision_flags", {})
            best = report.get("best_threshold_candidate") or {}
            key_metrics = {
                "spearman_ca_t90": report.get("basic_correlations", {}).get("t90", {}).get("spearman", {}).get("correlation"),
                "threshold": best.get("threshold"),
                "high_rate_delta": best.get("high_rate_delta"),
                "evidence_strength": report.get("evidence_strength"),
            }
            conclusion = f"global evidence {report.get('evidence_strength')}, high-calcium risk={flags.get('high_calcium_high_t90_risk_supported')}"
            usable_algorithm = "partial"
        elif name == "regime_threshold":
            synth = report.get("global_synthesis", {})
            key_metrics = synth
            conclusion = f"relation_type={synth.get('relation_type')}, contradictions={synth.get('contradictory_regime_count')}"
            usable_algorithm = "partial"
        elif name == "clustering_robustness":
            selected = report.get("selected_result") or {}
            key_metrics = {
                "algorithm": selected.get("algorithm"),
                "k": selected.get("k"),
                "min_cluster_size": selected.get("min_cluster_size"),
                "high_rate_range": selected.get("high_rate_range"),
                "silhouette": selected.get("silhouette_score"),
            }
            conclusion = f"robust k={selected.get('k')} clustering selected"
            usable_algorithm = "partial"
        elif name == "cluster_specific":
            flags = report.get("decision_flags", {})
            key_metrics = flags
            conclusion = f"cluster-specific support={flags.get('cluster_specific_ca_t90_supported')}, contradictions={flags.get('cluster_specific_contradictions_identified')}"
            usable_algorithm = "partial"
        rows.append(
            {
                "source_name": name,
                "source_path": str(paths.get(name)) if paths.get(name) else None,
                "evidence_type": evidence_type,
                "description": desc,
                "sample_count": report.get("sample_count"),
                "usable_count": report.get("usable_count"),
                "key_metrics": json.dumps(sanitize(key_metrics), ensure_ascii=False),
                "relationship_conclusion": conclusion,
                "limitations": ";".join(report.get("limitations", [])),
                "usable_for_algorithm_change": usable_algorithm,
                "usable_for_explanation_layer": usable_explanation,
            }
        )
    return pd.DataFrame(rows)


def evidence_matrix(reports: dict[str, dict[str, Any]]) -> pd.DataFrame:
    global_r = reports.get("global_threshold", {})
    regime = reports.get("regime_threshold", {})
    cluster = reports.get("cluster_specific", {})
    robust = reports.get("clustering_robustness", {})
    g_flags = global_r.get("decision_flags", {})
    g_best = global_r.get("best_threshold_candidate") or {}
    r_synth = regime.get("global_synthesis", {})
    c_flags = cluster.get("decision_flags", {})
    selected = robust.get("selected_result") or {}
    c_rel = cluster.get("cluster_ca_t90_relations", [])
    safe_band = global_r.get("safe_band_consistency") or {}
    rows = [
        item("global_ca_t90_relation", support_from_bool(g_flags.get("positive_relation_supported"), "moderate"), 1 if g_flags.get("positive_relation_supported") else 0, 0, f"Spearman={global_r.get('basic_correlations', {}).get('t90', {}).get('spearman', {}).get('correlation')}", "全局钙单耗与 T90 为弱正相关。", "可作为解释证据，不能单独改算法。", "继续分工况验证和人工复核。"),
        item("global_high_calcium_high_t90_risk", support_from_bool(g_flags.get("high_calcium_high_t90_risk_supported"), "moderate"), 1 if g_flags.get("high_calcium_high_t90_risk_supported") else 0, 0, f"high_rate_delta={g_best.get('high_rate_delta')}", "全局阈值后高 T90 风险明显升高。", "可支持高钙风险提示。", "验证阈值在不同工况下是否稳定。"),
        item("nonlinear_threshold", "weak" if not g_flags.get("nonlinear_threshold_supported") else "moderate", 1 if g_flags.get("nonlinear_threshold_supported") else 0, 1 if not g_flags.get("nonlinear_threshold_supported") else 0, f"threshold={g_best.get('threshold')}", "分段阈值有风险差，但非线性斜率证据不足。", "不宜直接把阈值写入推荐算法。", "做更稳健的分工况阈值复验。"),
        item("flat_safe_region", "not_supported" if not g_flags.get("flat_safe_region_supported") else "moderate", 1 if g_flags.get("flat_safe_region_supported") else 0, 1 if not g_flags.get("flat_safe_region_supported") else 0, f"safe_band_upper={safe_band.get('recommended_ca_consumption_max_median')}", "全局未证明平坦安全平台，但现有安全带仍有历史验证基础。", "保持当前安全带为冻结基线。", "不要用全局平坦区假设替换验证过的区间。"),
        item("regime_specific_positive_relation", "strong" if r_synth.get("positive_relation_regime_count", 0) >= 20 else "moderate", r_synth.get("positive_relation_regime_count"), r_synth.get("contradictory_regime_count"), f"{r_synth.get('positive_relation_regime_count')}/{r_synth.get('regime_count')}", "多数工况三分位支持正向关系。", "适合解释层和后续上下文算法研究。", "验证矛盾工况是否可稳定识别。"),
        item("regime_specific_high_risk_relation", "moderate" if r_synth.get("high_calcium_high_t90_risk_regime_count", 0) >= 8 else "weak", r_synth.get("high_calcium_high_t90_risk_regime_count"), r_synth.get("contradictory_regime_count"), f"high-risk regimes={r_synth.get('high_calcium_high_t90_risk_regime_count')}", "部分工况中高钙高 T90 风险清晰。", "可作为人工复核风险解释。", "收集更多近期验证样本。"),
        item("contradictory_regimes", "contradictory" if r_synth.get("contradictory_regime_count", 0) else "not_supported", 0, r_synth.get("contradictory_regime_count"), f"contradictory={r_synth.get('contradictory_regime_count')}", "低溴、低流量、低碱等场景存在反向/矛盾证据。", "阻止立即修改推荐区间算法。", "单独审计矛盾工况的工艺机理。"),
        item("robust_clustering_validity", "moderate" if selected else "insufficient", 1 if selected else 0, 0, f"k={selected.get('k')}, min_size={selected.get('min_cluster_size')}, high_range={selected.get('high_rate_range')}", "稳健 k=5 聚类有效，前一版 k=2 不可信。", "可作为解释分层，不直接作为规则。", "做近期样本稳定性验证。"),
        item("cluster_specific_supporting_relation", "moderate" if c_flags.get("cluster_specific_ca_t90_supported") else "weak", sum(1 for r in c_rel if r.get("high_calcium_high_t90_risk_supported")), 0, cluster_key_numbers(c_rel), "多个聚类内支持高钙高 T90 风险。", "适合添加聚类上下文说明。", "避免在矛盾聚类内套用统一阈值。"),
        item("cluster_specific_contradictions", "contradictory" if c_flags.get("cluster_specific_contradictions_identified") else "not_supported", 0, sum(1 for r in c_rel if not r.get("positive_relation_supported")), cluster_key_numbers(c_rel), "cluster 2 等聚类存在反向关系。", "不应现在修改运行算法。", "先形成人工解释和监测面板。"),
        item("current_safe_band_consistency", "moderate" if c_flags.get("safe_band_explanation_enhanced") else "weak", 1 if c_flags.get("safe_band_explanation_enhanced") else 0, 0, f"safe_band_upper={safe_band.get('recommended_ca_consumption_max_median')}", "安全带在部分聚类内解释力增强，但不是所有聚类都有足够样本。", "当前安全带继续作为冻结基线。", "针对样本不足聚类补数据。"),
        item("ir_optional_context", "insufficient", 0, 0, "IR not central in latest synthesis", "IR 可保留为辅助上下文，但当前不是算法修改依据。", "不作为动作驱动。", "继续监测，不进入推荐规则。"),
    ]
    return pd.DataFrame(rows)


def item(name: str, support: str, support_count: Any, contradiction_count: Any, key_numbers: str, interpretation: str, implication: str, next_validation: str) -> dict[str, Any]:
    return {
        "evidence_item": name,
        "support_level": support,
        "support_count": support_count,
        "contradiction_count": contradiction_count,
        "key_numbers": key_numbers,
        "interpretation_cn": interpretation,
        "algorithm_implication": implication,
        "required_next_validation": next_validation,
    }


def support_from_bool(flag: Any, positive_level: str) -> str:
    return positive_level if bool(flag) else "not_supported"


def cluster_key_numbers(cluster_relations: list[dict[str, Any]]) -> str:
    bits = []
    for row in cluster_relations:
        bits.append(f"c{row.get('cluster')}:rho={row.get('spearman_ca_t90')},delta={row.get('high_rate_delta')}")
    return "; ".join(bits)


def contexts_from_reports(reports: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    support_rows = []
    contradiction_rows = []
    mixed_rows = []
    regime = reports.get("regime_threshold", {})
    for row in regime.get("strongest_supporting_regimes", [])[:20]:
        support_rows.append(context_row("regime", f"{row.get('regime_feature')}:{row.get('regime_bin')}", row, "该工况支持高钙高 T90 风险，可用于人工复核说明。"))
    for row in regime.get("contradictory_regimes", [])[:20]:
        contradiction_rows.append(context_row("regime", f"{row.get('regime_feature')}:{row.get('regime_bin')}", row, "该工况与全局方向相反或较弱，不能套用统一阈值。"))
    cluster = reports.get("cluster_specific", {})
    consistency = {row.get("cluster"): row for row in cluster.get("consistency_with_regime_thresholds", [])}
    t90_map = {row.get("cluster"): row for row in cluster.get("cluster_t90_profiles", [])}
    for row in cluster.get("cluster_ca_t90_relations", []):
        cid = row.get("cluster")
        merged = dict(row)
        merged.update(t90_map.get(cid, {}))
        rel_type = consistency.get(cid, {}).get("cluster_relation_type")
        name = f"cluster_{cid}"
        if rel_type == "supporting" or row.get("high_calcium_high_t90_risk_supported"):
            support_rows.append(context_row("cluster", name, merged, "该聚类内高钙高 T90 风险关系较强，适合解释层提示。"))
        elif rel_type == "contradictory" or not row.get("positive_relation_supported"):
            contradiction_rows.append(context_row("cluster", name, merged, "该聚类呈矛盾/反向关系，算法修改前必须单独验证。"))
        else:
            mixed_rows.append(context_row("cluster", name, merged, "该聚类为混合证据，只适合监测说明。"))
    return pd.DataFrame(support_rows), pd.DataFrame(contradiction_rows), pd.DataFrame(mixed_rows)


def context_row(kind: str, name: str, row: dict[str, Any], caution: str) -> dict[str, Any]:
    return {
        "context_type": kind,
        "context_name": name,
        "sample_count": row.get("sample_count"),
        "spearman_ca_t90": row.get("spearman_ca_t90"),
        "spearman_ca_y_high": row.get("spearman_ca_y_high"),
        "high_rate_delta": row.get("high_rate_delta"),
        "threshold": row.get("best_threshold") or row.get("threshold"),
        "ok_rate": row.get("ok_rate"),
        "high_rate": row.get("high_rate"),
        "low_rate": row.get("low_rate"),
        "interpretation_cn": "支持高钙高 T90 风险" if (row.get("high_rate_delta") or 0) > 0 else "矛盾或混合关系",
        "caution_cn": caution,
    }


def assess_algorithm(reports: dict[str, dict[str, Any]], matrix: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str]:
    regime_synth = reports.get("regime_threshold", {}).get("global_synthesis", {})
    cluster_flags = reports.get("cluster_specific", {}).get("decision_flags", {})
    global_flags = reports.get("global_threshold", {}).get("decision_flags", {})
    positive_broad = regime_synth.get("relation_type") == "broadly_consistent"
    contradictions = int(regime_synth.get("contradictory_regime_count") or 0)
    cluster_contra = bool(cluster_flags.get("cluster_specific_contradictions_identified"))
    safe_enhanced = bool(cluster_flags.get("safe_band_explanation_enhanced"))
    global_high = bool(global_flags.get("high_calcium_high_t90_risk_supported"))
    threshold_count = int(regime_synth.get("threshold_evidence_regime_count") or 0)

    if not reports.get("global_threshold") and not reports.get("regime_threshold") and not reports.get("cluster_specific"):
        decision = "insufficient_evidence"
        reason = "缺少可用关系发现报告，无法判断是否修改算法。"
    elif positive_broad and contradictions <= 1 and not cluster_contra and threshold_count >= 12 and safe_enhanced:
        decision = "modify_recommender_now"
        reason = "证据广泛一致且矛盾上下文很少，可以考虑修改算法。"
    elif positive_broad and (contradictions > 0 or cluster_contra):
        decision = "explanation_layer_only"
        reason = "关系有解释价值，但支持和矛盾上下文并存，当前证据不足以改变推荐区间算法。"
    elif global_high:
        decision = "manual_review_warning_only"
        reason = "高钙高 T90 风险存在，但阈值/上下文一致性不足，只适合人工复核提示。"
    elif cluster_flags.get("cluster_specific_ca_t90_supported"):
        decision = "prepare_context_specific_algorithm_later"
        reason = "聚类/分工况关系有意义，但需进一步验证后才能改算法。"
    else:
        decision = "no_change_keep_current_baseline"
        reason = "总体证据弱或矛盾，保持当前基线并继续收集数据。"

    rows = [
        {"criterion": "positive_relation_broadly_consistent", "value": positive_broad, "interpretation_cn": "分工况关系是否总体一致"},
        {"criterion": "contradictory_context_count", "value": contradictions, "interpretation_cn": "矛盾工况数量"},
        {"criterion": "cluster_contradictions_identified", "value": cluster_contra, "interpretation_cn": "聚类内是否有反向/矛盾关系"},
        {"criterion": "threshold_evidence_regime_count", "value": threshold_count, "interpretation_cn": "分工况阈值证据数量"},
        {"criterion": "safe_band_explanation_enhanced", "value": safe_enhanced, "interpretation_cn": "安全带是否被聚类解释增强"},
        {"criterion": "algorithm_modification_decision", "value": decision, "interpretation_cn": reason},
    ]
    next_step = DECISION_TO_NEXT_STEP[decision]
    return pd.DataFrame(rows), decision, reason, next_step


def roadmap(decision: str) -> list[dict[str, Any]]:
    steps = [
        ("freeze_current_safe_band_baseline", "冻结当前稳定安全带 MVP，不修改运行包与规则。"),
        ("add_manual_review_explanation_layer", "把高钙高 T90 风险、支持/矛盾上下文加入人工复核说明。"),
        ("build_context_specific_validation_dataset", "构建近期/回填验证集，按工况和聚类追踪关系稳定性。"),
        ("test_context_specific_safe_band_or_threshold_rules", "离线测试上下文安全带或阈值规则，不进入自动控制。"),
        ("only_then_consider_recommender_update", "仅在分工况证据稳定、矛盾场景可解释后，再考虑重设推荐算法。"),
    ]
    return [{"step_order": i + 1, "roadmap_step": key, "description_cn": text, "active_now": key in {"freeze_current_safe_band_baseline", "add_manual_review_explanation_layer"} if decision == "explanation_layer_only" else key == "freeze_current_safe_band_baseline"} for i, (key, text) in enumerate(steps)]


def make_figures(matrix: pd.DataFrame, support: pd.DataFrame, contradiction: pd.DataFrame, roadmap_rows: list[dict[str, Any]], figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    level_order = {"strong": 5, "moderate": 4, "weak": 3, "contradictory": 2, "not_supported": 1, "insufficient": 0}
    fig, ax = plt.subplots(figsize=(10, 6))
    values = [level_order.get(x, 0) for x in matrix["support_level"]]
    colors = ["#3b7f3b" if v >= 4 else "#d4a63a" if v == 3 else "#b64040" if v == 2 else "#999999" for v in values]
    ax.barh(matrix["evidence_item"], values, color=colors)
    ax.set_title("钙单耗-T90 关系证据矩阵")
    ax.set_xlabel("支持等级编码")
    out = figure_dir / "ca_t90_evidence_support_matrix.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    counts = pd.DataFrame(
        [
            {"context_group": "supporting", "count": len(support)},
            {"context_group": "contradictory", "count": len(contradiction)},
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts["context_group"], counts["count"], color=["#3b7f3b", "#b64040"])
    ax.set_title("支持与矛盾工况对比")
    ax.set_ylabel("上下文数量")
    out = figure_dir / "ca_t90_supporting_vs_contradictory_contexts.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    fig, ax = plt.subplots(figsize=(10, 4))
    y = np.arange(len(roadmap_rows))
    active = [r["active_now"] for r in roadmap_rows]
    ax.barh(y, [1] * len(y), color=["#3b7f3b" if a else "#cccccc" for a in active])
    ax.set_yticks(y)
    ax.set_yticklabels([r["roadmap_step"] for r in roadmap_rows])
    ax.set_xticks([])
    ax.set_title("推荐算法修改准备度路线图")
    out = figure_dir / "ca_t90_algorithm_readiness_roadmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))
    return generated


def append_method_doc(path: Path, report: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# 稳定钙单耗安全带 MVP 方法与数据流说明\n"
    marker = "## 钙单耗-T90 关系发现阶段性结论"
    section = f"""

{marker}

当前关系发现阶段显示：钙单耗与 T90/高 T90 风险存在历史相关性，但不是简单、无条件、全局单调规律。全局分析为 `{report.get('global_conclusion')}`；分工况分析显示 `{report.get('relationship_summary_cn')}`。

全局证据：钙单耗与 T90 Spearman 约为 {report.get('evidence_matrix_lookup', {}).get('global_ca_t90_relation')}；高钙阈值后高 T90 风险上升，但非线性阈值和平坦安全区证据不足。

分工况证据：多数三分位工况支持正向/高风险关系，但存在低溴、低流量、低中和碱、部分高温等矛盾上下文。聚类证据显示部分 cluster 内高钙高 T90 风险很强，同时也存在反向或混合 cluster。

支持上下文示例：{report.get('strongest_supporting_contexts')[:5]}。

矛盾上下文示例：{report.get('contradictory_contexts')[:5]}。

算法修改判断：`{report.get('algorithm_modification_decision')}`。原因：{report.get('decision_reason_cn')}

当前推荐区间系统暂作为冻结基线；后续是否修改推荐算法，取决于分工况/聚类上下文关系是否进一步稳定。

安全约束不变：无自动控制、无 DCS 写回、不输出操作性加钙指令；当前证据只能用于人工复核解释和后续离线实验设计。
"""
    if marker in existing:
        prefix = existing.split(marker)[0].rstrip()
        path.write_text(prefix + section, encoding="utf-8")
    else:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(section)


def append_experiment_doc(path: Path, report: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    section_no = 36
    while f"## {section_no}." in existing:
        section_no += 1
    section = f"""

## {section_no}. 钙单耗-T90 关系发现证据汇总与推荐算法修改判断

本阶段汇总全局阈值、分工况阈值、稳健聚类和聚类内钙-T90 关系证据，目标是判断是否应立即修改当前推荐算法。当前阶段暂停运行包迭代，不修改 `deploy/ca_safe_band_mvp/`。

可用证据源：{report.get('available_evidence_sources')}；缺失证据源：{report.get('missing_evidence_sources')}。

全局关系结论：{report.get('global_conclusion')}。

分工况结论：{report.get('relationship_summary_cn')}。

聚类内结论：支持和矛盾 cluster 并存，可增强人工复核解释，但不足以直接修改推荐区间算法。

支持上下文：{report.get('strongest_supporting_contexts')[:5]}。

矛盾上下文：{report.get('contradictory_contexts')[:5]}。

算法修改判断：`{report.get('algorithm_modification_decision')}`。推荐下一步：`{report.get('recommended_next_step')}`。

局限性：全部证据仍为离线历史证据，不是因果证明；T90 存在人为测量误差；工况混杂和矛盾上下文仍存在；不建议自动控制，不进行 DCS 写回。
"""
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(section)


def main() -> None:
    args = parse_args()
    configure_chinese_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    paths = {
        "global_threshold": resolve_report(args.global_threshold_report, warnings),
        "regime_threshold": resolve_report(args.regime_threshold_report, warnings),
        "clustering_robustness": resolve_report(args.clustering_robustness_report, warnings),
        "cluster_specific": resolve_report(args.cluster_specific_report, warnings),
    }
    reports = {name: load_report(path) for name, path in paths.items()}
    if not any(reports.values()):
        raise FileNotFoundError("No relationship evidence reports were available.")
    available = [name for name, report in reports.items() if report]
    missing = [name for name, report in reports.items() if not report]

    inventory = inventory_rows(paths, reports)
    matrix = evidence_matrix(reports)
    support, contradiction, mixed = contexts_from_reports(reports)
    readiness, decision, reason, next_step = assess_algorithm(reports, matrix)
    roadmap_rows = roadmap(decision)
    figures = make_figures(matrix, support, contradiction, roadmap_rows, args.figure_dir)

    inventory.to_csv(args.output_dir / "evidence_source_inventory.csv", index=False, encoding="utf-8-sig")
    matrix.to_csv(args.output_dir / "ca_t90_evidence_matrix.csv", index=False, encoding="utf-8-sig")
    support.to_csv(args.output_dir / "supporting_contexts.csv", index=False, encoding="utf-8-sig")
    contradiction.to_csv(args.output_dir / "contradictory_contexts.csv", index=False, encoding="utf-8-sig")
    mixed.to_csv(args.output_dir / "mixed_contexts.csv", index=False, encoding="utf-8-sig")
    readiness.to_csv(args.output_dir / "algorithm_modification_readiness.csv", index=False, encoding="utf-8-sig")
    matrix.to_csv(args.table_dir / "ca_t90_relationship_evidence_summary.csv", index=False, encoding="utf-8-sig")
    readiness.to_csv(args.table_dir / "ca_t90_algorithm_decision_summary.csv", index=False, encoding="utf-8-sig")

    regime_synth = reports.get("regime_threshold", {}).get("global_synthesis", {})
    global_conclusion = f"全局证据为 {reports.get('global_threshold', {}).get('evidence_strength', 'unknown')}；高钙高 T90 风险支持={reports.get('global_threshold', {}).get('decision_flags', {}).get('high_calcium_high_t90_risk_supported')}"
    relationship_summary = f"分工况关系类型={regime_synth.get('relation_type')}，正向工况={regime_synth.get('positive_relation_regime_count')}/{regime_synth.get('regime_count')}，矛盾工况={regime_synth.get('contradictory_regime_count')}"
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_paths": {k: str(v) if v else None for k, v in paths.items()},
        "available_evidence_sources": available,
        "missing_evidence_sources": missing,
        "evidence_source_inventory": inventory.to_dict(orient="records"),
        "evidence_matrix": matrix.to_dict(orient="records"),
        "strongest_supporting_contexts": support.to_dict(orient="records"),
        "contradictory_contexts": contradiction.to_dict(orient="records"),
        "mixed_contexts": mixed.to_dict(orient="records"),
        "global_conclusion": global_conclusion,
        "relationship_summary_cn": relationship_summary,
        "algorithm_modification_assessment": readiness.to_dict(orient="records"),
        "algorithm_modification_decision": decision,
        "decision_reason_cn": reason,
        "future_roadmap": roadmap_rows,
        "current_baseline_status": "frozen_current_safe_band_mvp_baseline",
        "runtime_package_modified": False,
        "generated_figures": figures,
        "warnings": warnings,
        "limitations": [
            "offline_historical_evidence_only",
            "not_causal_proof",
            "t90_measurement_error",
            "process_confounding",
            "contradiction_contexts_exist",
            "no_automatic_control",
            "no_dcs_writeback",
            "current_stable_safe_band_mvp_remains_baseline",
        ],
        "recommended_next_step": next_step,
    }
    report["evidence_matrix_lookup"] = {row["evidence_item"]: row["key_numbers"] for row in report["evidence_matrix"]}
    write_json(args.output_dir / "ca_t90_relationship_evidence_synthesis_report.json", report)
    append_method_doc(args.method_doc, report)
    append_experiment_doc(args.doc, report)

    print("Calcium-T90 relationship evidence synthesis summary")
    print(f"available_evidence_sources: {available}")
    print(f"missing_evidence_sources: {missing}")
    print(f"algorithm_modification_decision: {decision}")
    print(f"decision_reason_cn: {reason}")
    print(f"recommended_next_step: {next_step}")
    print(f"runtime_package_modified: False")
    print("No generated outputs were written under data/.")


if __name__ == "__main__":
    main()
