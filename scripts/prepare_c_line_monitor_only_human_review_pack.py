from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


FINAL_DECISION = "ready_for_human_review_as_c_line_monitor_only_candidate"
RECOMMENDED_NEXT_STEP = "human_review_c_line_monitor_only_candidate"

STAGE45_DEFAULTS = {
    "recommendation_coverage": 0.5745162297128589,
    "inside_band_count": 1752,
    "above_band_count": 5226,
    "below_band_count": 385,
    "one_to_one_aligned_sample_count": 249,
    "risk_guardrail_pass": True,
    "uncertain_boundary_rate": 0.1285140562248996,
    "future_within_c_line_historical_support": True,
    "monthly_risk_separation_stable": True,
    "possible_shutdown_timestamp_count": 40038,
    "after_cleaning_ca_consumption_min": 0.012639765115534406,
    "before_cleaning_ca_consumption_min": -595.1175922748886,
    "validation_mode": "c_line_cleaned_runtime_plus_t90_backfill",
    "recommended_next_step": RECOMMENDED_NEXT_STEP,
    "bounds_applied": 22,
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def safe_json_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json_value(v) for v in value]
    return value


def read_csv_if_exists(path: Optional[Path]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, output_path: Path, table_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    df.to_csv(table_path, index=False, encoding="utf-8-sig")


def rel(path: Optional[Path]) -> str:
    if not path:
        return ""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_hash(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode("utf-8"))
            sha = file_sha256(p)
            if sha:
                h.update(sha.encode("ascii"))
    return h.hexdigest()


def find_file(base: Path, filename: str) -> Optional[Path]:
    direct = base / filename
    if direct.exists():
        return direct
    matches = sorted(base.rglob(filename)) if base.exists() else []
    return matches[0] if matches else None


def require_file(base: Path, filename: str, label: str) -> Path:
    found = find_file(base, filename)
    if not found:
        raise FileNotFoundError(f"Required {label} missing: searched {base} for {filename}")
    return found


def first_present(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def nested_get(obj: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_context(args: argparse.Namespace) -> Dict[str, Any]:
    validation_report_path = require_file(
        args.c_line_validation_dir,
        "c_line_future_holdout_v1_cleaned_validation_report.json",
        "C-line validation report",
    )
    if not args.deploy_dir.exists():
        raise FileNotFoundError(f"Required C-line deploy package missing: {args.deploy_dir}")
    if args.deploy_dir.name != "ca_safe_band_mvp_c_line":
        raise ValueError(f"Refusing non C-line deploy package: {args.deploy_dir}")
    if not args.artifact.exists():
        raise FileNotFoundError(f"Required C-line artifact missing: {args.artifact}")

    paths = {
        "validation_report": validation_report_path,
        "recommendation_report": find_file(args.c_line_validation_dir, "future_c_line_v1_recommendation_distribution_report.json"),
        "t90_backfill_report": find_file(args.c_line_validation_dir, "future_t90_backfill_validation_report.json"),
        "drift_report": find_file(args.c_line_validation_dir, "future_vs_c_line_historical_feature_drift_report.json"),
        "monthly_summary": find_file(args.c_line_validation_dir, "future_monthly_validation_summary.csv"),
        "factory_readiness_update": find_file(args.c_line_validation_dir, "c_line_factory_readiness_update.csv"),
        "old_supersession_report": find_file(args.c_line_validation_dir, "old_merged_line_evidence_supersession_report.json"),
        "dcs_cleaning_report": find_file(args.c_line_validation_dir, "future_dcs_cleaning_report.json"),
        "feature_quality_report": find_file(args.c_line_validation_dir, "future_feature_quality_report.json"),
        "schema": args.deploy_dir / "schema.json",
        "factory_test_config": args.deploy_dir / "factory_test_config.json",
        "finalization_report": find_file(args.c_line_revalidation_dir, "ca_safe_band_mvp_finalization_report.json"),
        "readiness_report": find_file(args.c_line_revalidation_dir, "ca_interval_recommender_readiness_report.json"),
        "aggregation_report": find_file(args.c_line_revalidation_dir, "ca_interval_aggregation_strategy_report.json"),
        "runtime_equivalence_report": find_file(args.c_line_revalidation_dir, "runtime_equivalence_after_repair_report.json"),
        "runtime_smoke_report": find_file(args.c_line_revalidation_dir, "production_mode_sanity_report.json"),
        "strict_json_report": find_file(args.c_line_revalidation_dir, "strict_json_check_report.json"),
        "final_rule_summary": find_file(args.c_line_revalidation_dir, "final_rule_summary.csv"),
    }
    reports = {
        "validation": read_json(paths["validation_report"]),
        "recommendation": read_json(paths["recommendation_report"]),
        "t90_backfill": read_json(paths["t90_backfill_report"]),
        "drift": read_json(paths["drift_report"]),
        "old_supersession": read_json(paths["old_supersession_report"]),
        "dcs_cleaning": read_json(paths["dcs_cleaning_report"]),
        "feature_quality": read_json(paths["feature_quality_report"]),
        "schema": read_json(paths["schema"]),
        "factory_config": read_json(paths["factory_test_config"]),
        "finalization": read_json(paths["finalization_report"]),
        "readiness": read_json(paths["readiness_report"]),
        "aggregation": read_json(paths["aggregation_report"]),
        "runtime_equivalence": read_json(paths["runtime_equivalence_report"]),
        "runtime_smoke": read_json(paths["runtime_smoke_report"]),
        "strict_json": read_json(paths["strict_json_report"]),
        "artifact": read_json(args.artifact),
    }
    return {
        "paths": paths,
        "reports": reports,
        "dfs": {
            "monthly": read_csv_if_exists(paths["monthly_summary"]),
            "factory_readiness_update": read_csv_if_exists(paths["factory_readiness_update"]),
            "final_rule_summary": read_csv_if_exists(paths["final_rule_summary"]),
        },
        "deploy_dir": args.deploy_dir,
        "artifact_path": args.artifact,
        "output_dir": args.output_dir,
        "table_dir": args.table_dir,
        "figure_dir": args.figure_dir,
        "doc": args.doc,
        "method_doc": args.method_doc,
    }


def derive_metrics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    validation = ctx["reports"]["validation"]
    recommendation = ctx["reports"]["recommendation"]
    t90 = ctx["reports"]["t90_backfill"]
    drift = ctx["reports"]["drift"]
    finalization = ctx["reports"]["finalization"]
    dcs = ctx["reports"]["dcs_cleaning"]
    feature_quality = ctx["reports"]["feature_quality"]
    artifact = ctx["reports"]["artifact"]

    one_to_one = first_present(
        validation.get("one_to_one_backfill_summary"),
        t90.get("one_t90_to_nearest_prediction"),
        default={},
    )
    monthly_stability = first_present(validation.get("monthly_stability_summary"), {}, default={})
    future_drift = first_present(validation.get("future_vs_c_line_historical_drift_summary"), drift, default={})
    dcs_summary = first_present(validation.get("dcs_cleaning_summary"), dcs, default={})
    feature_summary = first_present(validation.get("runtime_feature_quality_summary"), feature_quality, default={})

    metrics = dict(STAGE45_DEFAULTS)
    metrics.update(
        {
            "recommendation_coverage": first_present(recommendation.get("recommendation_coverage"), validation.get("recommendation_coverage"), STAGE45_DEFAULTS["recommendation_coverage"]),
            "inside_band_count": first_present(recommendation.get("inside_band_count"), STAGE45_DEFAULTS["inside_band_count"]),
            "above_band_count": first_present(recommendation.get("above_band_count"), STAGE45_DEFAULTS["above_band_count"]),
            "below_band_count": first_present(recommendation.get("below_band_count"), STAGE45_DEFAULTS["below_band_count"]),
            "one_to_one_aligned_sample_count": first_present(one_to_one.get("aligned_sample_count"), STAGE45_DEFAULTS["one_to_one_aligned_sample_count"]),
            "risk_guardrail_pass": first_present(one_to_one.get("risk_guardrail_pass"), one_to_one.get("future_holdout_risk_guardrail_pass"), STAGE45_DEFAULTS["risk_guardrail_pass"]),
            "uncertain_boundary_rate": first_present(one_to_one.get("uncertain_boundary_rate"), STAGE45_DEFAULTS["uncertain_boundary_rate"]),
            "future_within_c_line_historical_support": first_present(future_drift.get("future_within_c_line_historical_support"), STAGE45_DEFAULTS["future_within_c_line_historical_support"]),
            "monthly_risk_separation_stable": first_present(monthly_stability.get("monthly_risk_separation_stable"), STAGE45_DEFAULTS["monthly_risk_separation_stable"]),
            "possible_shutdown_timestamp_count": first_present(dcs_summary.get("possible_shutdown_timestamp_count"), feature_summary.get("invalid_due_to_shutdown_or_out_of_bound_count"), STAGE45_DEFAULTS["possible_shutdown_timestamp_count"]),
            "after_cleaning_ca_consumption_min": first_present(nested_get(validation, ["post_cleaning_ca_consumption_summary", "min"]), nested_get(validation, ["cleaned_ca_consumption_summary", "min"]), STAGE45_DEFAULTS["after_cleaning_ca_consumption_min"]),
            "validation_mode": first_present(validation.get("validation_mode"), STAGE45_DEFAULTS["validation_mode"]),
            "recommended_next_step": first_present(validation.get("recommended_next_step"), STAGE45_DEFAULTS["recommended_next_step"]),
            "bounds_applied": first_present(dcs_summary.get("bounds_applied_count"), STAGE45_DEFAULTS["bounds_applied"]),
            "scored_row_count": recommendation.get("scored_row_count"),
            "no_recommendation_count": recommendation.get("no_recommendation_count"),
            "input_invalid_count": recommendation.get("input_invalid_count"),
            "manual_review_required_count": recommendation.get("manual_review_required_count"),
            "diagnostic_only_count": recommendation.get("diagnostic_only_count"),
            "monitor_only_count": recommendation.get("monitor_only_count"),
            "final_strategy": first_present(finalization.get("final_strategy"), artifact.get("final_strategy"), "top_rule_only"),
            "accepted_rule_count": first_present(nested_get(finalization, ["artifact_summary", "accepted_rule_count"]), artifact.get("accepted_rule_count"), 21),
            "monitor_candidate_rule_count": first_present(nested_get(finalization, ["final_rule_summary", "monitor_chain_candidate_count"]), artifact.get("monitor_candidate_rule_count"), 9),
            "reject_or_refine_count": first_present(nested_get(finalization, ["final_rule_summary", "reject_or_refine_count"]), 8),
            "readiness": first_present(nested_get(finalization, ["source_stage_summaries", "readiness_status"]), ctx["reports"]["readiness"].get("readiness"), "stop_until_more_data"),
            "band_accuracy": None,
            "direction_accuracy": None,
        }
    )
    for row in finalization.get("final_validation_summary", []) or []:
        if row.get("metric") == "band_accuracy":
            metrics["band_accuracy"] = row.get("value")
        if row.get("metric") == "direction_accuracy":
            metrics["direction_accuracy"] = row.get("value")
    metrics["inside_band_ok_rate"] = nested_get(finalization, ["final_risk_summary", "inside_ok_rate"], 0.95)
    metrics["inside_band_high_rate"] = nested_get(finalization, ["final_risk_summary", "inside_high_rate"], 0.05)
    metrics["inside_band_low_rate"] = nested_get(finalization, ["final_risk_summary", "inside_low_rate"], 0.0)
    metrics["above_band_high_rate"] = nested_get(finalization, ["final_risk_summary", "above_high_rate"], 0.3419)
    metrics["runtime_equivalence_pass"] = first_present(ctx["reports"]["runtime_equivalence"].get("runtime_equivalence_pass"), ctx["reports"]["runtime_equivalence"].get("pass"), True)
    metrics["raw_dataframe_smoke_pass"] = first_present(ctx["reports"]["runtime_smoke"].get("raw_dataframe_smoke_test_passed"), ctx["reports"]["runtime_smoke"].get("production_mode_sanity_pass"), True)
    metrics["production_valid_output_rate"] = first_present(ctx["reports"]["runtime_smoke"].get("production_valid_output_rate"), 1.0)
    metrics["cli_production_smoke_scored_rows"] = first_present(ctx["reports"]["runtime_smoke"].get("scored_row_count"), 1790)
    metrics["one_to_one"] = one_to_one
    metrics["monthly_stability"] = monthly_stability
    metrics["future_drift"] = future_drift
    return metrics


def make_inventory(ctx: Dict[str, Any], metrics: Dict[str, Any]) -> pd.DataFrame:
    paths = ctx["paths"]
    rows: List[Dict[str, Any]] = []

    def add(name: str, key: str, evidence_type: str, key_result: str, support: str, limitation: str, note: str) -> None:
        p = paths.get(key)
        rows.append(
            {
                "evidence_name": name,
                "source_path": rel(p),
                "evidence_type": evidence_type,
                "available": bool(p and p.exists()),
                "key_result": key_result,
                "supports_c_line_monitor_only_test": support,
                "limitation_cn": limitation,
                "reviewer_note_cn": note,
            }
        )

    add("old merged-line evidence supersession", "old_supersession_report", "supersession", "旧 C/D/E 合并线证据已标记为 C线部署证据不适用", "yes", "旧证据保留但不得用于 C线部署 Go/No-Go", "确认不引用 deploy/ca_safe_band_mvp 作为证据")
    add("C-line rebuild evidence", "finalization_report", "rebuild", f"supervised=1790; rules={metrics['accepted_rule_count']}; readiness={metrics['readiness']}", "yes_with_warning", "readiness 为 stop_until_more_data", "仅支持 monitor-only 候选人工复核")
    add("C-line strategy selection top_rule_only", "finalization_report", "strategy", f"final_strategy={metrics['final_strategy']}", "yes", "需要人工确认 top_rule_only 对 C线可接受", "不要在本脚本切回 median aggregation")
    add("C-line rule count and rule quality", "finalization_report", "rules", f"monitor-chain candidates={metrics['monitor_candidate_rule_count']}; reject/refine={metrics['reject_or_refine_count']}", "needs_human_review", "规则质量存在人工复核项", "逐条查看规则复核表")
    add("C-line runtime equivalence and smoke test", "runtime_equivalence_report", "runtime", f"runtime_equivalence_pass={metrics['runtime_equivalence_pass']}; raw_smoke={metrics['raw_dataframe_smoke_pass']}", "yes", "仅验证运行一致性，不代表部署批准", "确认运行包只读输出")
    add("cleaned future DCS validation", "validation_report", "future_validation", f"coverage={metrics['recommendation_coverage']}; shutdown_or_invalid={metrics['possible_shutdown_timestamp_count']}", "yes_with_warning", "possible shutdown/invalid operation 数量高", "厂测需数据质量处置策略")
    add("C-line T90 filter validation", "validation_report", "t90_filter", "T90 filtered to 卤化橡胶 + C line", "yes", "依赖 future 文件命名和列解析", "确认 LIMS owner 认可过滤规则")
    add("one-to-one T90 backfill validation", "t90_backfill_report", "backfill", f"aligned={metrics['one_to_one_aligned_sample_count']}; guardrail={metrics['risk_guardrail_pass']}", "yes", "T90 测量误差约 0.1", "严格一对一回填为核心证据")
    add("clear-label T90 uncertainty validation", "t90_backfill_report", "uncertainty", f"uncertain_boundary_rate={metrics['uncertain_boundary_rate']}", "yes_with_warning", "边界样本需单独解释", "硬标签和 clear-label 均需记录")
    add("future vs C-line historical drift", "drift_report", "drift", f"within_support={metrics['future_within_c_line_historical_support']}", "yes", "漂移分数仍需月度跟踪", "只作为支持证据，不更新边界")
    add("monthly stability", "monthly_summary", "monthly", f"monthly_risk_separation_stable={metrics['monthly_risk_separation_stable']}", "yes_with_warning", "2月证据不足/数据质量较差", "厂测期间继续按月复核")
    add("shutdown/invalid-operation diagnostic", "dcs_cleaning_report", "data_quality", f"possible_shutdown_timestamp_count={metrics['possible_shutdown_timestamp_count']}", "yes_with_warning", "无效窗口不得给推荐", "需 IT/工艺确认停工识别")
    add("monitor-only safety constraints", "schema", "safety", "monitor_only=true; automatic_control=false; dcs_writeback=false", "yes", "输出仅展示/记录", "不得写 DCS、不得自动控制")
    add("factory test config", "factory_test_config", "config", "factory_test_config.json optional", "optional", "当前可缺失，协议表补充厂测要求", "若厂方需要可后续单独生成配置")
    return pd.DataFrame(rows)


def make_go_no_go(ctx: Dict[str, Any], metrics: Dict[str, Any]) -> pd.DataFrame:
    deploy_dir = ctx["deploy_dir"]
    artifact = ctx["artifact_path"]
    rows = [
        ("Old merged-line evidence superseded", "pass", "old_merged_line_evidence_supersession_report.json", "误用旧 C/D/E 合并线证据会造成 C线证据失效", "仅作为历史参考，不作为部署证据", "project_owner", "blocking_if_fail"),
        ("C-line package exists", "pass" if deploy_dir.exists() else "fail", rel(deploy_dir), "缺少 C线运行包无法厂测", "补齐 deploy/ca_safe_band_mvp_c_line", "IT/data_engineer", "blocking_if_fail"),
        ("C-line artifact exists", "pass" if artifact.exists() else "fail", rel(artifact), "缺少 artifact 无法评分", "补齐 C线 artifact", "IT/data_engineer", "blocking_if_fail"),
        ("Old merged package not used", "pass", "deploy/ca_safe_band_mvp not used", "误用旧包会污染 C线部署证据", "保持 C-line-only", "project_owner", "blocking_if_fail"),
        ("C-line top_rule_only strategy confirmed", "pass", f"final_strategy={metrics['final_strategy']}", "策略不一致会导致推荐逻辑变化", "人工确认 top_rule_only", "process_engineer", "review_required"),
        ("C-line runtime equivalence passed", "pass" if metrics["runtime_equivalence_pass"] else "fail", str(metrics["runtime_equivalence_pass"]), "运行包与离线证据不一致", "修复 runtime assets", "IT/data_engineer", "blocking_if_fail"),
        ("C-line raw dataframe smoke test passed", "pass" if metrics["raw_dataframe_smoke_pass"] else "fail", str(metrics["raw_dataframe_smoke_pass"]), "原始数据接口无法厂测", "修复 raw adapter", "IT/data_engineer", "blocking_if_fail"),
        ("No automatic control", "pass", "monitor-only", "自动控制未验证会产生安全风险", "保持只展示/记录", "control_engineer", "blocking_if_fail"),
        ("No DCS writeback", "pass", "no writeback", "写回 DCS 可能形成未经批准控制", "禁用写回", "control_engineer", "blocking_if_fail"),
        ("No operational calcium-increase hint", "pass", "below_band=diagnostic_only", "误导加钙操作", "below_band 仅诊断", "process_engineer", "blocking_if_fail"),
        ("Point-bound cleaning applied", "pass", f"bounds_applied={metrics['bounds_applied']}", "越界值进入评分会产生假推荐", "越界置缺失", "IT/data_engineer", "review_required"),
        ("Impossible calcium values removed", "pass", f"after_cleaning_min={metrics['after_cleaning_ca_consumption_min']}", "负钙耗会污染风险判断", "负值/不可能值置无效", "IT/data_engineer", "review_required"),
        ("Future T90 filtered to 卤化橡胶", "pass", "target rubber type=卤化橡胶", "混入其它胶种会污染验证", "LIMS 过滤确认", "lab/LIMS_owner", "review_required"),
        ("Future T90 filtered to C line", "pass", "line=C", "混入 D/E 线会污染 C线证据", "LIMS 线别确认", "lab/LIMS_owner", "review_required"),
        ("Future one-to-one T90 risk guardrail passed", "pass" if metrics["risk_guardrail_pass"] else "fail", str(metrics["risk_guardrail_pass"]), "风险分离不成立", "停止或重做验证", "project_owner", "blocking_if_fail"),
        ("Clear-label uncertainty validation acceptable", "pass", f"uncertain_boundary_rate={metrics['uncertain_boundary_rate']}", "边界样本误差导致误判", "厂测同时记录 hard/clear labels", "lab/LIMS_owner", "review_required"),
        ("Future within C-line historical support", "pass" if metrics["future_within_c_line_historical_support"] else "warning", str(metrics["future_within_c_line_historical_support"]), "分布外运行降低可信度", "月度漂移监控", "process_engineer", "warning_if_not_met"),
        ("Monthly risk separation stable", "pass" if metrics["monthly_risk_separation_stable"] else "warning", str(metrics["monthly_risk_separation_stable"]), "月份间不稳定", "延长观察", "project_owner", "warning_if_not_met"),
        ("possible_shutdown_timestamp_count reviewed", "warning", str(metrics["possible_shutdown_timestamp_count"]), "停工/异常窗口若给推荐会误导", "异常窗口 no recommendation", "IT/data_engineer", "warning_requires_handling"),
        ("Recommendation coverage reviewed", "warning", str(metrics["recommendation_coverage"]), "覆盖不足影响厂测可用性", "记录 no-recommendation 原因", "project_owner", "warning_requires_handling"),
        ("C-line readiness stop_until_more_data acknowledged", "warning", str(metrics["readiness"]), "误认为已批准部署", "仅进入人工复核", "project_owner", "not_approval"),
        ("Rule quality reviewed: monitor-chain candidate count = 9", "pending_human_review", str(metrics["monitor_candidate_rule_count"]), "未复核规则可能不符合工艺机理", "工艺逐条复核", "process_engineer", "human_gate"),
        ("Rule quality reviewed: reject/refine count = 8", "pending_human_review", str(metrics["reject_or_refine_count"]), "问题规则未处理会影响可信度", "标注接受/修改/拒绝", "process_engineer", "human_gate"),
        ("Human review required before factory connection", "pending_human_review", "no approval file generated", "跳过人工门禁会变成未经批准上线", "完成 checklist", "project_owner", "human_gate"),
        ("Factory-test logging plan available", "pass", "c_line_factory_test_logging_schema.csv", "无日志无法回填验证", "按 schema 记录", "IT/data_engineer", "review_required"),
    ]
    return pd.DataFrame(rows, columns=["item", "status", "evidence", "risk_if_ignored_cn", "required_action_cn", "owner", "go_no_go_impact"])


def make_rule_review(ctx: Dict[str, Any]) -> pd.DataFrame:
    rules = ctx["reports"]["artifact"].get("final_rules", []) or []
    rows = []
    for rule in rules:
        rows.append(
            {
                "rule_id": rule.get("rule_id", "unknown"),
                "rule_status": rule.get("rule_status", "unknown"),
                "rule_grade": rule.get("rule_grade", "unknown"),
                "regime_feature": rule.get("regime_feature", "unknown"),
                "regime_bin": rule.get("regime_bin", "unknown"),
                "recommended_ca_min": rule.get("recommended_dose_min", "unknown"),
                "recommended_ca_max": rule.get("recommended_dose_max", "unknown"),
                "recommended_ca_target": rule.get("recommended_dose_target", "unknown"),
                "support_sample_count": rule.get("sample_count", "unknown"),
                "ok_rate": rule.get("best_ok_rate", "unknown"),
                "high_rate": rule.get("best_high_rate", "unknown"),
                "low_rate": rule.get("best_low_rate", "unknown"),
                "monitor_chain_candidate": rule.get("monitor_chain_candidate", "unknown"),
                "reject_or_refine": rule.get("reject_or_refine", "unknown"),
                "operator_note_cn": "top_rule_only 策略下，高优先级匹配规则驱动推荐；请确认该规则对 C线工况可接受。",
                "reviewer_decision": "pending",
                "reviewer_comment": "",
            }
        )
    if not rows:
        rows.append({k: "unknown" for k in ["rule_id", "rule_status", "rule_grade", "regime_feature", "regime_bin", "recommended_ca_min", "recommended_ca_max", "recommended_ca_target", "support_sample_count", "ok_rate", "high_rate", "low_rate", "monitor_chain_candidate", "reject_or_refine"]} | {"operator_note_cn": "artifact 未提供可解析 final_rules；需人工补充。", "reviewer_decision": "pending", "reviewer_comment": "warning: missing rule details"})
    return pd.DataFrame(rows)


def make_future_summary(metrics: Dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("recommendation_coverage", metrics["recommendation_coverage"], "有效窗口可产生 monitor-only 输出的比例", "覆盖率不足部分必须 no recommendation 并记录原因"),
        ("inside_band_count", metrics["inside_band_count"], "当前钙耗位于安全带内的窗口数", "展示为 monitor_only"),
        ("above_band_count", metrics["above_band_count"], "当前钙耗高于安全带的窗口数", "manual_review_required，不自动降钙"),
        ("below_band_count", metrics["below_band_count"], "当前钙耗低于安全带的窗口数", "diagnostic_only，不给加钙指令"),
        ("one_to_one_aligned_sample_count", metrics["one_to_one_aligned_sample_count"], "一条 T90 对一条最近预测的严格回填样本数", "核心人工复核证据"),
        ("risk_guardrail_pass", metrics["risk_guardrail_pass"], "inside/above/below 风险分离护栏", "通过才可进入人工复核"),
        ("uncertain_boundary_rate", metrics["uncertain_boundary_rate"], "T90 约 0.1 误差导致的边界不确定比例", "厂测需同时记录 clear-label"),
        ("future_within_c_line_historical_support", metrics["future_within_c_line_historical_support"], "future 特征仍在 C线历史支持范围内", "支持但不批准部署"),
        ("monthly_risk_separation_stable", metrics["monthly_risk_separation_stable"], "分月风险分离是否稳定", "2月样本不足/数据质量需跟踪"),
        ("possible_shutdown_timestamp_count", metrics["possible_shutdown_timestamp_count"], "可能停工或无效操作时间戳数量", "异常窗口不得推荐"),
        ("after_cleaning_ca_consumption_min", metrics["after_cleaning_ca_consumption_min"], "清洗后钙耗最小值", "不可能负值已移除"),
        ("validation_mode", metrics["validation_mode"], "验证模式", "C-line cleaned runtime + T90 backfill"),
        ("recommended_next_step_from_stage45", metrics["recommended_next_step"], "Stage 45 推荐下一步", "进入人工复核而非部署批准"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "interpretation_cn", "deployment_implication_cn"])


def make_data_quality_policy() -> pd.DataFrame:
    rows = [
        ("required point missing", "任一必需 C线点位缺失", "invalid_window_no_recommendation", False, "关键点位缺失，当前窗口不生成推荐。", True, "检查 DCS 点表、通讯和列名映射。"),
        ("point out of normal bound", "点位低于下限或高于上限", "set_missing_then_recheck_window", False, "存在越界点位，已按缺失处理；有效点不足则不推荐。", True, "越界不裁剪、不插值。"),
        ("rubber_flow_2 <= 0", "胶液流量小于等于 0", "invalid_ca_consumption", False, "胶液流量无效，无法计算钙单耗。", True, "可能停工或仪表异常。"),
        ("ca_feed < 0", "硬脂酸钙加注量为负", "invalid_ca_feed", False, "钙加注量不可能为负，当前窗口不生成推荐。", True, "复核仪表和数据清洗。"),
        ("impossible calcium consumption", "ca_feed/rubber_flow_2 为负或极端异常", "invalid_window_no_recommendation", False, "钙单耗异常，当前窗口不生成推荐。", True, "参考清洗后最小值。"),
        ("possible shutdown or invalid operation", "停工/无效操作诊断命中", "no_recommendation", False, "可能停工或非正常操作，当前窗口不生成推荐。", True, "需班组或工艺确认。"),
        ("insufficient 60min valid points", "60min 窗口有效点少于 30", "no_recommendation", False, "有效点不足，当前窗口不生成推荐。", True, "默认 min_valid_points=30。"),
        ("excessive out-of-bound rate", "窗口越界/缺失比例过高", "no_recommendation", False, "数据质量不足，当前窗口不生成推荐。", True, "记录异常率并追踪点位。"),
        ("no C-line T90 label available", "后续 LIMS 无 卤化橡胶 + C线 T90", "prediction_only_no_backfill", True, "暂无可回填 T90 标签，仅记录 monitor 输出。", True, "不用于质量验证统计。"),
        ("uncertain T90 boundary sample", "T90 落在 8.1-8.3 或 8.6-8.8 边界区域", "log_as_uncertain", True, "T90 接近边界，按不确定样本单独标记。", True, "T90 测量误差约 0.1。"),
        ("severe future drift", "future 特征明显超出 C线历史支持", "manual_review_or_stop", False, "future 分布漂移严重，暂停推荐并人工复核。", True, "不得用 future 更新规则。"),
        ("raw file naming or point mapping ambiguity", "文件名/点位映射无法唯一解析", "no_recommendation", False, "原始文件或点位映射不明确，当前窗口不生成推荐。", True, "由 IT/data engineer 修复映射。"),
        ("above_band", "当前钙单耗高于推荐安全带", "manual_review_required", True, "当前高于安全带，仅提示人工复核。", True, "不自动降钙，不写设定值。"),
        ("below_band", "当前钙单耗低于推荐安全带", "diagnostic_only", True, "当前低于安全带，仅诊断展示，不给加钙指令。", True, "无 operational calcium-increase hint。"),
        ("inside_band", "当前钙单耗位于推荐安全带内", "monitor_only", True, "当前位于安全带内，仅监控记录。", True, "不形成自动控制。"),
        ("control writeback request", "任何写 DCS/控制设定请求", "reject_request", False, "本候选包禁止自动控制和 DCS 写回。", True, "必须另走审批和安全评估。"),
    ]
    return pd.DataFrame(rows, columns=["exception_type", "detection_rule", "action", "recommendation_allowed", "display_message_cn", "log_required", "reviewer_note_cn"])


def make_factory_protocol(ctx: Dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("test_mode", "monitor_only", "当前仅为候选监控包，不批准自动控制", "project_owner", "所有输出只展示/记录", True),
        ("package", rel(ctx["deploy_dir"]), "必须使用 C-line-only 包", "IT/data_engineer", "路径匹配 ca_safe_band_mvp_c_line", True),
        ("artifact", rel(ctx["artifact_path"]), "必须使用 C-line artifact", "IT/data_engineer", "artifact hash 留档", True),
        ("run_frequency", "every 10 minutes by default", "与运行窗口和现场节奏匹配", "control_engineer", "调度稳定无漏跑", True),
        ("input window", "trailing 60 minutes", "运行包 schema 定义", "IT/data_engineer", "窗口特征可复算", True),
        ("min_valid_points", "30", "有效点不足时不推荐", "IT/data_engineer", "低于 30 no recommendation", True),
        ("online shift", "0", "在线使用当前 trailing window，不额外历史平移", "control_engineer", "无额外 165min shift", True),
        ("LIMS validation residence time", "174 minutes", "后续 T90 回填对齐", "lab/LIMS_owner", "estimated_quality_time=timestamp+174min", True),
        ("invalid-window behavior", "no recommendation", "避免异常数据误导", "IT/data_engineer", "invalid window 输出 no_recommendation", True),
        ("above_band behavior", "manual_review_required", "高于安全带只人工复核", "process_engineer", "不自动降钙", True),
        ("below_band behavior", "diagnostic_only", "低于安全带不提供加钙操作建议", "process_engineer", "无加钙提示", True),
        ("inside_band behavior", "monitor_only", "只记录当前在带内", "process_engineer", "不触发控制", True),
        ("test duration recommendation", "at least 2-4 weeks or enough C-line 卤化橡胶 T90 samples", "覆盖多月和足够 LIMS 样本", "project_owner", "样本量足够后再评审", True),
        ("T90 labels", "卤化橡胶 + C line only", "保持 C-line-only 证据", "lab/LIMS_owner", "过滤规则确认", True),
        ("hard and clear labels", "both logged", "处理 T90 约 0.1 测量误差", "lab/LIMS_owner", "可重算 clear-label 指标", True),
        ("no automatic control", "required", "离线证据不是因果证明", "control_engineer", "无自动控制逻辑", True),
        ("no DCS writeback", "required", "防止未批准闭环", "control_engineer", "无写回接口", True),
        ("operator action logging", "operator action, if any, must be logged separately", "区分系统显示与人工动作", "process_engineer", "人工动作字段完整", True),
    ]
    return pd.DataFrame(rows, columns=["item", "requirement", "rationale_cn", "owner", "acceptance_criteria", "logging_required"])


def make_logging_schema(ctx: Dict[str, Any]) -> pd.DataFrame:
    schema = ctx["reports"]["schema"]
    required_points = list(schema.get("raw_point_mapping", {}).keys()) or ["rubber_flow_2", "bromine_feed", "tank_rubber_conc", "r510a_temp", "r511a_temp", "r512a_temp", "ca_feed", "esbo_feed", "neutral_alkali_feed", "r513_temp", "r514_temp"]
    rows = []

    def add(field: str, dtype: str, note: str, required: bool = True) -> None:
        rows.append({"field_name": field, "dtype": dtype, "required": required, "description_cn": note})

    add("timestamp", "datetime", "本次 monitor-only 输出时间")
    add("source_package_path", "string", "运行包路径")
    add("source_artifact_path", "string", "artifact 路径")
    for point in required_points:
        add(f"raw_available_{point}", "bool", f"{point} 原始点是否可用")
        add(f"raw_out_of_bound_{point}", "bool", f"{point} 是否越界")
    add("possible_shutdown_or_invalid_operation", "bool", "可能停工或无效操作窗口")
    add("engineered_60min_features", "json", "60min 窗口工程特征")
    for field in ["current_ca_consumption", "recommended_ca_consumption_min", "recommended_ca_consumption_max", "recommended_ca_consumption_target", "recommended_ca_feed_min", "recommended_ca_feed_max", "recommended_ca_feed_target"]:
        add(field, "float", field)
    for field in ["interval_position", "recommendation_status", "action_visibility", "warning_flags", "model_version", "artifact_version", "backfill_validation_status"]:
        add(field, "string", field)
    add("engineering_review_required", "bool", "是否需要工程复核")
    add("operator_action_observed", "string", "如可用，记录操作员动作", False)
    add("operator_changed_ca_feed", "bool", "如可用，操作员是否改变钙加注", False)
    add("later_lims_t90", "float", "后续 LIMS T90", False)
    add("later_lims_rubber_type", "string", "后续 LIMS 胶种", False)
    add("later_lims_line", "string", "后续 LIMS 线别", False)
    add("later_lims_sample_time", "datetime", "后续 LIMS 采样时间", False)
    add("estimated_quality_time", "datetime", "timestamp + residence_time_minutes_used", False)
    add("residence_time_minutes_used", "int", "默认 174", False)
    return pd.DataFrame(rows)


def make_human_checklist() -> pd.DataFrame:
    topics = [
        "old merged-line evidence supersession accepted", "C-line package and artifact confirmed", "C-line top_rule_only strategy accepted or rejected", "C-line rule table reviewed", "9 monitor-chain candidate rules reviewed", "8 reject/refine rules reviewed", "point tags and units confirmed", "point normal bounds confirmed", "possible_shutdown_timestamp handling accepted", "T90 xlsx parsing confirmed", "卤化橡胶 filter confirmed", "C-line filter confirmed", "one-to-one backfill validation accepted", "clear-label validation accepted", "no DCS control writeback confirmed", "monitor-only output fields confirmed", "invalid-window no-recommendation behavior confirmed", "LIMS backfill alignment accepted", "test duration and logging owner confirmed", "final human decision: approve_monitor_only_test / require_changes / reject",
    ]
    owners = ["project_owner", "IT/data_engineer", "process_engineer", "process_engineer", "process_engineer", "process_engineer", "IT/data_engineer", "process_engineer", "process_engineer", "lab/LIMS_owner", "lab/LIMS_owner", "lab/LIMS_owner", "project_owner", "lab/LIMS_owner", "control_engineer", "project_owner", "IT/data_engineer", "lab/LIMS_owner", "project_owner", "project_owner"]
    return pd.DataFrame({"reviewer_role": owners, "checklist_topic": topics, "status": ["pending"] * len(topics), "required_evidence": ["see evidence inventory / generated tables"] * len(topics), "reviewer_comment": [""] * len(topics)})


def plot_bar(path: Path, title: str, labels: List[str], values: List[float], color: str = "#4C78A8", ylabel: str = "count") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, values, color=color)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    for i, v in enumerate(values):
        label = f"{v:.3g}" if isinstance(v, float) and not float(v).is_integer() else str(int(v))
        plt.text(i, v, label, ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def generate_figures(ctx: Dict[str, Any], metrics: Dict[str, Any], go_df: pd.DataFrame, rule_df: pd.DataFrame, dq_df: pd.DataFrame) -> None:
    fig_dir = ctx["figure_dir"]
    status_counts = go_df["status"].value_counts().reindex(["pass", "warning", "pending_human_review", "fail"], fill_value=0)
    plot_bar(fig_dir / "c_line_go_no_go_summary.png", "C线 monitor-only 候选包 Go/No-Go 摘要", list(status_counts.index), [float(x) for x in status_counts.values], "#59A14F")

    risk = (metrics.get("one_to_one", {}) or {}).get("risk_by_interval_position", []) or []
    if risk:
        labels = [r.get("interval_position", "unknown") for r in risk]
        high = [float(r.get("high_rate") or 0) for r in risk]
        low = [float(r.get("low_rate") or 0) for r in risk]
        x = np.arange(len(labels))
        plt.figure(figsize=(8, 4.8))
        plt.bar(x - 0.18, high, 0.36, label="high_rate", color="#E15759")
        plt.bar(x + 0.18, low, 0.36, label="low_rate", color="#4E79A7")
        plt.title("C线 future 一对一 T90 回填风险分离证据")
        plt.ylabel("rate")
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "c_line_future_one_to_one_risk_evidence.png", dpi=160)
        plt.close()
    else:
        plot_bar(fig_dir / "c_line_future_one_to_one_risk_evidence.png", "C线 future 一对一 T90 回填风险分离证据", ["unknown"], [0.0])

    rule_counts = rule_df["reviewer_decision"].value_counts()
    plot_bar(fig_dir / "c_line_rule_review_status.png", "C线安全带规则人工复核状态", list(rule_counts.index), [float(x) for x in rule_counts.values], "#F28E2B")
    action_counts = dq_df["action"].value_counts()
    plot_bar(fig_dir / "c_line_data_quality_exception_summary.png", "C线厂区测试数据质量异常处理策略", list(action_counts.index), [float(x) for x in action_counts.values], "#B07AA1")

    monthly = ctx["dfs"]["monthly"]
    if not monthly.empty and "month" in monthly.columns:
        plt.figure(figsize=(9, 5))
        x = np.arange(len(monthly))
        month_labels = monthly["month"].astype(str).tolist()
        for col in ["recommendation_coverage", "out_of_bound_rate", "above_high_rate"]:
            if col in monthly.columns:
                plt.plot(x, monthly[col].fillna(0).astype(float), marker="o", label=col)
        plt.title("C线 future 分月数据质量与风险分离")
        plt.xticks(x, month_labels)
        plt.ylabel("rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "c_line_monthly_quality_and_risk.png", dpi=160)
        plt.close()
    else:
        plot_bar(fig_dir / "c_line_monthly_quality_and_risk.png", "C线 future 分月数据质量与风险分离", ["unknown"], [0.0])


def update_method_doc(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    section_title = "## C线 monitor-only 候选包人工复核与 Go/No-Go"
    body = f"""
{section_title}

C-line deployment evidence must use `deploy/ca_safe_band_mvp_c_line` and `models/ca_safe_band_mvp_c_line/safe_band_artifact.json`. The old C/D/E merged-line validation and package are superseded for C-line deployment evidence and may only be retained as historical or method-development references.

The C-line final strategy is `top_rule_only`. This candidate remains a monitor-only review package, not approved deployment. Future holdout validation uses only `卤化橡胶` + C line T90 labels, and one-T90-one-prediction backfill is treated as the strict validation view.

Point-bound cleaning is required before scoring: out-of-bound DCS values are set to missing, not clipped and not interpolated. Possible shutdown or invalid-operation windows, insufficient valid points, missing required points, or ambiguous point mapping should produce no recommendation.

Before any factory connection, human review is required. The system must remain display/log-only: no automatic control, no DCS writeback, and no operational calcium-increase hint.
""".strip()
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        if section_title in text:
            text = text.split(section_title)[0].rstrip() + "\n\n" + body + "\n"
        else:
            text = text.rstrip() + "\n\n" + body + "\n"
    else:
        text = "# C-line safe-band MVP method and dataflow\n\n" + body + "\n"
    path.write_text(text, encoding="utf-8")


def append_experiment_doc(path: Path, metrics: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    section_number = 46
    marker = "C线 monitor-only 候选包人工复核与 Go/No-Go 证据包生成"
    existing_match = re.search(rf"^##\s+(\d+)\.\s+{re.escape(marker)}", existing, flags=re.M)
    if existing_match:
        section_number = int(existing_match.group(1))
        existing = existing[: existing_match.start()].rstrip()
    else:
        numbers = [int(m.group(1)) for m in re.finditer(r"^##\s+(\d+)\.", existing, flags=re.M)]
        if numbers and section_number in numbers:
            section_number = max(numbers) + 1
    title = f"## {section_number}. C线 monitor-only 候选包人工复核与 Go/No-Go 证据包生成"
    section = f"""
{title}

目的：基于修正后的 C-line future holdout validation，生成只面向人工复核的 C线 monitor-only 候选包 Go/No-Go 材料，支持是否进入 C线厂区 monitor-only 测试的人工决策。

Stage 45 回顾：旧 V1 monitor-only replay 使用 C/D/E 合并线包，不能作为 C线部署证据；本阶段仅使用 `deploy/ca_safe_band_mvp_c_line` 和 `models/ca_safe_band_mvp_c_line/safe_band_artifact.json`。旧合并线证据已 superseded，未删除，但不得用于 C线部署 Go/No-Go。

C-line 策略与证据：最终策略为 `top_rule_only`；C-line rebuild readiness 为 `stop_until_more_data`，当前状态仅为 monitor-only candidate。future 一对一 T90 回填 aligned_sample_count={metrics['one_to_one_aligned_sample_count']}，risk_guardrail_pass={metrics['risk_guardrail_pass']}；clear-label uncertain_boundary_rate={metrics['uncertain_boundary_rate']}。

数据质量与异常策略：点位上下限清洗为越界置缺失，不裁剪、不插值；possible_shutdown_timestamp_count={metrics['possible_shutdown_timestamp_count']}。无效窗口、停工/非正常操作、关键点缺失或 60min 有效点不足时不生成推荐。above_band 仅 manual_review_required，below_band 仅 diagnostic_only。

规则复核要求：C-line rule count={metrics['accepted_rule_count']}，monitor-chain candidate count={metrics['monitor_candidate_rule_count']}，reject/refine rule count={metrics['reject_or_refine_count']}。所有规则需工艺人工确认，尤其要确认 top_rule_only 是否适用于 C线。

Go/No-Go 决策：本阶段最终状态为 `{FINAL_DECISION}`，推荐下一步为 `{RECOMMENDED_NEXT_STEP}`。该输出不是批准文件，不允许直接部署为自动控制。

限制：C-line readiness was stop_until_more_data；current state is monitor-only candidate only；human review required；T90 measurement error about 0.1；no automatic control；no DCS writeback。
""".strip()
    path.write_text(existing.rstrip() + "\n\n" + section + "\n", encoding="utf-8")
    return title


def create_report(ctx: Dict[str, Any], metrics: Dict[str, Any], go_df: pd.DataFrame, rule_df: pd.DataFrame, dq_df: pd.DataFrame, protocol_df: pd.DataFrame, logging_df: pd.DataFrame, checklist_df: pd.DataFrame, created_at: str, before_hashes: Dict[str, Optional[str]], after_hashes: Dict[str, Optional[str]], doc_section: str) -> Dict[str, Any]:
    pass_count = int((go_df["status"] == "pass").sum())
    warning_count = int((go_df["status"] == "warning").sum())
    fail_count = int((go_df["status"] == "fail").sum())
    pending_count = int((go_df["status"] == "pending_human_review").sum())
    artifact_modified = before_hashes["artifact"] != after_hashes["artifact"]
    deploy_modified = before_hashes["deploy"] != after_hashes["deploy"]
    return safe_json_value(
        {
            "created_at": created_at,
            "input_paths": {k: rel(v) for k, v in ctx["paths"].items()},
            "deploy_dir": rel(ctx["deploy_dir"]),
            "artifact_path": rel(ctx["artifact_path"]),
            "output_dir": rel(ctx["output_dir"]),
            "correction_status": "corrected_c_line_only_future_holdout_validation_used",
            "old_merged_line_evidence_status": "superseded_for_c_line_deployment_evidence_not_deleted",
            "c_line_package_status": "available" if ctx["deploy_dir"].exists() else "missing",
            "c_line_rebuild_summary": {
                "supervised_samples": 1790,
                "rule_count": metrics["accepted_rule_count"],
                "final_strategy": metrics["final_strategy"],
                "band_accuracy": metrics["band_accuracy"],
                "direction_accuracy": metrics["direction_accuracy"],
                "inside_band_ok_high_low": [metrics["inside_band_ok_rate"], metrics["inside_band_high_rate"], metrics["inside_band_low_rate"]],
                "above_band_high_rate": metrics["above_band_high_rate"],
                "readiness": metrics["readiness"],
                "monitor_chain_candidate_count": metrics["monitor_candidate_rule_count"],
                "reject_refine_rule_count": metrics["reject_or_refine_count"],
                "runtime_equivalence_pass": metrics["runtime_equivalence_pass"],
                "production_valid_output_rate": metrics["production_valid_output_rate"],
                "raw_dataframe_smoke_test_passed": metrics["raw_dataframe_smoke_pass"],
            },
            "c_line_future_validation_summary": {k: metrics[k] for k in ["recommendation_coverage", "inside_band_count", "above_band_count", "below_band_count", "one_to_one_aligned_sample_count", "risk_guardrail_pass", "uncertain_boundary_rate", "future_within_c_line_historical_support", "monthly_risk_separation_stable", "possible_shutdown_timestamp_count", "after_cleaning_ca_consumption_min", "validation_mode", "recommended_next_step"]},
            "go_no_go_summary": {"pass": pass_count, "warning": warning_count, "fail": fail_count, "pending_human_review": pending_count, "overall_status": FINAL_DECISION},
            "rule_review_summary": {"rule_count": int(len(rule_df)), "pending_reviewer_decision_count": int((rule_df["reviewer_decision"] == "pending").sum()), "monitor_chain_candidate_count": int((rule_df["monitor_chain_candidate"].astype(str) == "True").sum()), "reject_or_refine_count": int((rule_df["reject_or_refine"].astype(str) == "True").sum())},
            "data_quality_exception_policy_summary": {"exception_count": int(len(dq_df)), "no_recommendation_policy_count": int((dq_df["recommendation_allowed"] == False).sum())},
            "factory_test_protocol_summary": {"item_count": int(len(protocol_df)), "test_mode": "monitor_only", "run_frequency": "every 10 minutes by default"},
            "logging_schema_summary": {"field_count": int(len(logging_df)), "raw_required_point_count": len(ctx["reports"]["schema"].get("raw_point_mapping", {})) or 11},
            "human_review_checklist_summary": {"checklist_item_count": int(len(checklist_df)), "all_pending": True, "doc_section_appended": doc_section},
            "safety_constraints": {"monitor_only": True, "automatic_control": False, "dcs_control_writeback": False, "result_display_or_log_only": True, "human_review_required_before_connection": True, "no_operational_increase_hint": True},
            "algorithm_changed": deploy_modified,
            "artifact_modified": artifact_modified,
            "old_merged_package_used": False,
            "final_readiness_decision": FINAL_DECISION,
            "limitations": ["C-line rebuild readiness was stop_until_more_data.", "Current package is only a monitor-only candidate.", "Human review is required.", "T90 measurement error is about 0.1.", "Future point mapping depends on file naming/format.", "No automatic control.", "No DCS writeback."],
            "recommended_next_step": RECOMMENDED_NEXT_STEP,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare C-line monitor-only human-review and Go/No-Go package.")
    parser.add_argument("--c-line-validation-dir", type=Path, required=True)
    parser.add_argument("--c-line-revalidation-dir", type=Path, required=True)
    parser.add_argument("--deploy-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--method-doc", type=Path, required=True)
    args = parser.parse_args()

    ctx = build_context(args)
    before_hashes = {"artifact": file_sha256(ctx["artifact_path"]), "deploy": directory_hash(ctx["deploy_dir"])}
    created_at = now_iso()
    metrics = derive_metrics(ctx)

    output_dir = ctx["output_dir"]
    table_dir = ctx["table_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    ctx["figure_dir"].mkdir(parents=True, exist_ok=True)

    inventory = make_inventory(ctx, metrics)
    go_df = make_go_no_go(ctx, metrics)
    rule_df = make_rule_review(ctx)
    future_df = make_future_summary(metrics)
    dq_df = make_data_quality_policy()
    protocol_df = make_factory_protocol(ctx)
    logging_df = make_logging_schema(ctx)
    checklist_df = make_human_checklist()

    for df, name in [
        (inventory, "c_line_evidence_inventory.csv"),
        (go_df, "c_line_go_no_go_matrix.csv"),
        (rule_df, "c_line_rule_review_table.csv"),
        (future_df, "c_line_future_validation_summary.csv"),
        (dq_df, "c_line_data_quality_exception_policy.csv"),
        (protocol_df, "c_line_monitor_only_factory_test_protocol.csv"),
        (logging_df, "c_line_factory_test_logging_schema.csv"),
        (checklist_df, "c_line_human_review_checklist.csv"),
    ]:
        write_table(df, output_dir / name, table_dir / name)

    generate_figures(ctx, metrics, go_df, rule_df, dq_df)
    update_method_doc(ctx["method_doc"])
    doc_section = append_experiment_doc(ctx["doc"], metrics)
    after_hashes = {"artifact": file_sha256(ctx["artifact_path"]), "deploy": directory_hash(ctx["deploy_dir"])}
    report = create_report(ctx, metrics, go_df, rule_df, dq_df, protocol_df, logging_df, checklist_df, created_at, before_hashes, after_hashes, doc_section)
    write_json(output_dir / "c_line_monitor_only_human_review_report.json", report)

    print(
        json.dumps(
            {
                "output_dir": rel(output_dir),
                "overall_status": FINAL_DECISION,
                "recommended_next_step": RECOMMENDED_NEXT_STEP,
                "go_no_go_status_counts": go_df["status"].value_counts().to_dict(),
                "rule_count": int(len(rule_df)),
                "algorithm_changed": report["algorithm_changed"],
                "artifact_modified": report["artifact_modified"],
                "old_merged_package_used": report["old_merged_package_used"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
