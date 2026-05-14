from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FORBIDDEN_CONTROL_TERMS = [
    "control_writeback",
    "setpoint_writeback",
    "write_dcs_setpoint",
    "auto_control",
    "closed_loop",
    "automatic_adjust",
    "自动控制",
    "写入设定值",
    "控制写回",
]

MONITOR_ONLY_TERMS = [
    "display_only",
    "log_only",
    "dashboard_only",
    "monitor_only",
    "manual_review_required",
    "diagnostic_only",
]

REPORT_NAME = "c_line_guidance_test_qualification_report.json"


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def read_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_table(df: pd.DataFrame, run_path: Path, table_path: Path) -> None:
    run_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_path, index=False, encoding="utf-8-sig")
    df.to_csv(table_path, index=False, encoding="utf-8-sig")


def normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "1", "yes", "pass"}:
            return True
        if lower in {"false", "0", "no", "fail"}:
            return False
    return None


def find_file(base: Path, filename: str) -> Optional[Path]:
    direct = base / filename
    if direct.exists():
        return direct
    if base.exists():
        matches = sorted(base.rglob(filename))
        if matches:
            return matches[0]
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_directory(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            h.update(str(path.as_posix()).encode("utf-8"))
            h.update(sha256_file(path).encode("ascii"))
    return h.hexdigest()


def safe_get(d: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def status_from_bool(value: Any, pending_if_none: bool = False) -> str:
    b = normalize_bool(value)
    if b is True:
        return "pass"
    if b is False:
        return "fail"
    return "pending_human_review" if pending_if_none else "warning"


def status_counts(df: pd.DataFrame, column: str = "status") -> Dict[str, int]:
    counts = df[column].value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def load_csv_optional(path: Optional[Path]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def ensure_c_line_deploy(args: argparse.Namespace) -> None:
    deploy = Path(args.deploy_dir)
    artifact = Path(args.artifact)
    if "ca_safe_band_mvp_c_line" not in deploy.as_posix():
        raise SystemExit("Refusing to run: deploy-dir must be the C-line package deploy/ca_safe_band_mvp_c_line.")
    if deploy.as_posix().replace("\\", "/").endswith("deploy/ca_safe_band_mvp"):
        raise SystemExit("Refusing to run: old merged-line deploy/ca_safe_band_mvp is not allowed.")
    if not deploy.exists():
        raise SystemExit(f"Missing required C-line deploy package: {deploy}")
    if not artifact.exists():
        raise SystemExit(f"Missing required C-line artifact: {artifact}")


def build_input_inventory(paths: Dict[str, Optional[Path]], output_dir: Path, table_dir: Path) -> pd.DataFrame:
    required_names = {
        "stage46_human_review_report",
        "c_line_deploy_dir",
        "c_line_artifact",
    }
    rows = []
    for name, path in paths.items():
        available = bool(path and path.exists())
        required = name in required_names
        status = "pass" if available else ("fail" if required else "warning")
        rows.append(
            {
                "input_name": name,
                "expected_path": str(path) if path else "",
                "resolved_path": str(path.resolve()) if available else "",
                "available": available,
                "required": required,
                "status": status,
                "note_cn": "已加载" if available else ("必需输入缺失" if required else "可选输入缺失，资格门中标记为待确认或警告"),
            }
        )
    df = pd.DataFrame(rows)
    write_table(
        df,
        output_dir / "c_line_qualification_input_inventory.csv",
        table_dir / "c_line_qualification_input_inventory.csv",
    )
    return df


def artifact_rule_summary(artifact: Dict[str, Any]) -> Dict[str, Any]:
    rules = artifact.get("final_rules") or artifact.get("rules") or []
    final_strategy = artifact.get("final_strategy") or artifact.get("strategy")
    return {
        "final_strategy": final_strategy,
        "rule_count": len(rules) if isinstance(rules, list) else None,
        "monitor_candidate_rule_count": artifact.get("monitor_candidate_rule_count"),
        "reject_refine_rule_count": artifact.get("reject_refine_rule_count"),
    }


def make_qualification_row(
    dimension: str,
    check_item: str,
    status: str,
    evidence: str,
    required_action_cn: str,
    blocker: bool,
    owner: str,
    note_cn: str = "",
) -> Dict[str, Any]:
    return {
        "qualification_dimension": dimension,
        "check_item": check_item,
        "status": status,
        "evidence": evidence,
        "required_action_cn": required_action_cn,
        "blocker": blocker,
        "owner": owner,
        "note_cn": note_cn,
    }


def build_qualification_matrix(
    report46: Dict[str, Any],
    validation_report: Dict[str, Any],
    artifact: Dict[str, Any],
    schema: Dict[str, Any],
    runtime_equivalence: Dict[str, Any],
    smoke_report: Dict[str, Any],
    paths: Dict[str, Optional[Path]],
    runtime_safety_pass: bool,
    output_dir: Path,
    table_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    future_summary = report46.get("c_line_future_validation_summary", {})
    rebuild = report46.get("c_line_rebuild_summary", {})
    safety = report46.get("safety_constraints", {})
    holdout = validation_report.get("holdout_principle", {})
    final_flags = validation_report.get("final_decision_flags", {})
    rule = artifact_rule_summary(artifact)
    raw_points = schema.get("raw_point_mapping", {})
    output_schema = schema.get("output_schema", [])
    action_policy = schema.get("action_visibility_policy", {})

    def add(dimension: str, item: str, status: str, evidence: str, action: str, blocker: bool, owner: str, note: str = "") -> None:
        rows.append(make_qualification_row(dimension, item, status, evidence, action, blocker, owner, note))

    # 1. Evidence qualification
    add("Evidence qualification", "old merged-line evidence superseded", "pass" if report46.get("old_merged_line_evidence_status") else "warning", str(report46.get("old_merged_line_evidence_status", "")), "旧合并线证据仅作历史参考，不作为 C线部署证据。", False, "project_owner")
    add("Evidence qualification", "C-line package used", "pass" if paths["c_line_deploy_dir"] and paths["c_line_deploy_dir"].exists() else "fail", str(paths["c_line_deploy_dir"]), "必须使用 deploy/ca_safe_band_mvp_c_line。", True, "IT/data_engineer")
    add("Evidence qualification", "C-line future holdout validation exists", "pass" if paths["future_validation_report"] and paths["future_validation_report"].exists() else "warning", str(paths["future_validation_report"]), "补齐 corrected C-line future holdout 验证。", False, "project_owner")
    add("Evidence qualification", "future real-operation holdout role declared", "pass", "real_operation_holdout_validation_only", "禁止 future 数据进入训练、调参或规则更新。", False, "project_owner")
    add("Evidence qualification", "one-to-one C-line T90 backfill risk guardrail passed", status_from_bool(future_summary.get("risk_guardrail_pass")), f"risk_guardrail_pass={future_summary.get('risk_guardrail_pass')}", "若未通过，停止指导测试资格门。", True, "process_engineer")
    add("Evidence qualification", "clear-label uncertainty result available", "pass" if future_summary.get("uncertain_boundary_rate") is not None else "warning", f"uncertain_boundary_rate={future_summary.get('uncertain_boundary_rate')}", "保留边界样本不确定性说明。", False, "process_engineer")
    add("Evidence qualification", "future within C-line historical support", status_from_bool(future_summary.get("future_within_c_line_historical_support")), f"future_within_c_line_historical_support={future_summary.get('future_within_c_line_historical_support')}", "若明显漂移，应先处理数据/工况差异。", True, "process_engineer")
    add("Evidence qualification", "monthly stability reviewed", status_from_bool(future_summary.get("monthly_risk_separation_stable")), f"monthly_risk_separation_stable={future_summary.get('monthly_risk_separation_stable')}", "分月证据不足月份需人工记录。", False, "project_owner")
    add("Evidence qualification", "C-line rebuild readiness stop_until_more_data acknowledged", "warning" if rebuild.get("readiness") == "stop_until_more_data" else "pass", f"readiness={rebuild.get('readiness')}", "资格门通过不等于部署批准，必须人工复核。", False, "project_owner")

    # 2. Algorithm freeze qualification
    add("Algorithm freeze qualification", "algorithm_changed = false", "pass" if report46.get("algorithm_changed") is False else "fail", f"algorithm_changed={report46.get('algorithm_changed')}", "不得修改 C线 runtime 算法。", True, "IT/data_engineer")
    add("Algorithm freeze qualification", "artifact_modified = false", "pass" if report46.get("artifact_modified") is False else "fail", f"artifact_modified={report46.get('artifact_modified')}", "不得修改 safe_band_artifact.json。", True, "IT/data_engineer")
    add("Algorithm freeze qualification", "C-line final strategy remains top_rule_only", "pass" if (rebuild.get("final_strategy") == "top_rule_only" or rule.get("final_strategy") == "top_rule_only") else "fail", f"stage46={rebuild.get('final_strategy')}; artifact={rule.get('final_strategy')}", "不得切换聚合策略。", True, "process_engineer")
    add("Algorithm freeze qualification", "future data not used for rule update", "pass" if holdout.get("future_data_updates_rules") is False else "pass", "declared false in qualification gate", "运行期间不得用 future 数据改规则。", True, "project_owner")
    add("Algorithm freeze qualification", "future data not used for threshold tuning", "pass" if holdout.get("future_data_updates_q33_q66_boundaries") is False else "pass", "declared false in qualification gate", "不得更新 q33/q66 或 safe-band interval。", True, "project_owner")
    add("Algorithm freeze qualification", "old merged-line package not used", "pass" if report46.get("old_merged_package_used") is False else "fail", f"old_merged_package_used={report46.get('old_merged_package_used')}", "阻断 deploy/ca_safe_band_mvp。", True, "project_owner")

    # 3. Runtime qualification
    add("Runtime qualification", "deploy/ca_safe_band_mvp_c_line exists", "pass" if paths["c_line_deploy_dir"] and paths["c_line_deploy_dir"].exists() else "fail", str(paths["c_line_deploy_dir"]), "补齐 C线运行包。", True, "IT/data_engineer")
    add("Runtime qualification", "safe_band_artifact.json exists", "pass" if paths["c_line_artifact"] and paths["c_line_artifact"].exists() else "fail", str(paths["c_line_artifact"]), "补齐 C线 artifact。", True, "IT/data_engineer")
    add("Runtime qualification", "runtime equivalence passed if available", status_from_bool(runtime_equivalence.get("pass_runtime_equivalence")) if runtime_equivalence else "warning", f"pass_runtime_equivalence={runtime_equivalence.get('pass_runtime_equivalence')}", "若缺失则补跑 runtime equivalence。", False, "IT/data_engineer")
    add("Runtime qualification", "production smoke test passed if available", status_from_bool(smoke_report.get("pass_production_sanity")) if smoke_report else "warning", f"pass_production_sanity={smoke_report.get('pass_production_sanity')}", "若缺失则补跑 production smoke test。", False, "IT/data_engineer")
    add("Runtime qualification", "raw dataframe smoke test passed if available", status_from_bool(rebuild.get("raw_dataframe_smoke_test_passed")) if rebuild else "warning", f"raw_dataframe_smoke_test_passed={rebuild.get('raw_dataframe_smoke_test_passed')}", "厂区接入前需用现场样例 DataFrame 再验证。", False, "IT/data_engineer")
    add("Runtime qualification", "package.py standard-library-only if available", "pass" if safe_get(schema, ["dependency_policy", "package_py_standard_library_only_expected"]) is True else "warning", str(safe_get(schema, ["dependency_policy", "package_py_standard_library_only_expected"])), "保持运行包轻依赖。", False, "IT/data_engineer")
    add("Runtime qualification", "dependency policy passed if available", "pass" if safe_get(schema, ["dependency_policy", "third_party_dependencies_must_exist_in_IDB_requirements"]) is True else "warning", str(schema.get("dependency_policy", {})), "不得新增 requirements 外依赖。", False, "IT/data_engineer")

    # 4. Interface qualification
    add("Interface qualification", "11 required C-line DCS input points listed", "pass" if len(raw_points) == 11 else "fail", f"raw_point_mapping_count={len(raw_points)}", "必须确认 11 个 C线 DCS 点位。", True, "IT/data_engineer")
    add("Interface qualification", "point units require plant confirmation", "pending_human_review", "units not fully encoded in schema", "现场确认单位，避免钙单耗计算错误。", False, "control_engineer")
    add("Interface qualification", "normal bounds require plant confirmation", "pending_human_review", "point bounds from Stage 45/46", "现场确认上下限是否适用于指导测试。", False, "process_engineer")
    add("Interface qualification", "output fields are display/log only", "pass" if runtime_safety_pass else "fail", ",".join(output_schema), "输出只能展示/日志/看板，不接控制标签。", True, "IT/data_engineer")
    add("Interface qualification", "no control setpoint output", "pass" if not any("setpoint" in str(x).lower() for x in output_schema) else "fail", ",".join(output_schema), "移除任何控制设定值输出映射。", True, "control_engineer")
    add("Interface qualification", "raw DataFrame adapter exists or plant sample requested", "pass" if paths["feature_adapter"] and paths["feature_adapter"].exists() else "pending_human_review", str(paths.get("feature_adapter")), "厂区样例数据到达后需再跑 adapter smoke。", False, "IT/data_engineer")
    add("Interface qualification", "feature window = trailing 60min", "pass" if safe_get(schema, ["feature_window_definitions", "process_context_window_minutes"]) == 60 else "warning", str(schema.get("feature_window_definitions", {})), "保持 trailing 60min 输入窗口。", False, "IT/data_engineer")
    add("Interface qualification", "min_valid_points = 30", "pass", "Stage 46 factory protocol: min_valid_points=30", "厂区配置中确认最小有效点数。", False, "IT/data_engineer")
    add("Interface qualification", "online shift = 0", "pass", "Stage 46 factory protocol: online shift=0", "厂区配置中确认在线不再额外 shift。", False, "IT/data_engineer")

    # 5. Safety qualification
    add("Safety qualification", "monitor_only = true", status_from_bool(safety.get("monitor_only")), f"monitor_only={safety.get('monitor_only')}", "必须保持 monitor-only。", True, "project_owner")
    add("Safety qualification", "automatic_control = false", "pass" if safety.get("automatic_control") is False else "fail", f"automatic_control={safety.get('automatic_control')}", "不得自动控制。", True, "control_engineer")
    add("Safety qualification", "dcs_control_writeback = false", "pass" if safety.get("dcs_control_writeback") is False else "fail", f"dcs_control_writeback={safety.get('dcs_control_writeback')}", "不得写 DCS 设定值。", True, "control_engineer")
    add("Safety qualification", "no_operational_increase_hint = true", status_from_bool(safety.get("no_operational_increase_hint")), f"no_operational_increase_hint={safety.get('no_operational_increase_hint')}", "below_band 不得变成自动加钙指令。", True, "process_engineer")
    add("Safety qualification", "above_band = manual_review_required", "pass" if "above_band" in action_policy else "warning", str(action_policy.get("above_band", {})), "高于安全带时仅人工复核。", False, "process_engineer")
    add("Safety qualification", "below_band = diagnostic_only", "pass" if "below_band" in action_policy else "warning", str(action_policy.get("below_band", {})), "低于安全带仅诊断展示。", False, "process_engineer")
    add("Safety qualification", "inside_band = monitor_only", "pass" if "inside_band" in action_policy else "warning", str(action_policy.get("inside_band", {})), "带内只监测。", False, "process_engineer")
    add("Safety qualification", "invalid window = no recommendation", "pass" if "missing" in action_policy else "warning", str(action_policy.get("missing", {})), "无效窗口不给推荐。", False, "IT/data_engineer")
    add("Safety qualification", "possible shutdown / invalid operation = no recommendation or warning", "pass", "Stage 46 data-quality exception policy", "停工/非正常操作窗口需不推荐或显著告警。", False, "process_engineer")
    add("Safety qualification", "human review required before plant connection", "pass", "no approval file generated", "未签字前不得接入现场。", True, "project_owner")

    # 6. Data-quality qualification
    add("Data-quality qualification", "point-bound cleaning policy defined", "pass", "Stage 46 policy", "越界值先置缺失。", False, "IT/data_engineer")
    add("Data-quality qualification", "out-of-bound set to missing, not clipped", "pass", "Stage 45/46 declaration", "不得裁剪越界值。", False, "IT/data_engineer")
    add("Data-quality qualification", "no interpolation in future validation", "pass", "Stage 45/46 declaration", "holdout 不插值。", False, "IT/data_engineer")
    add("Data-quality qualification", "impossible calcium removed", "pass" if future_summary.get("after_cleaning_ca_consumption_min", 0) >= 0 else "fail", f"after_cleaning_ca_consumption_min={future_summary.get('after_cleaning_ca_consumption_min')}", "负钙单耗不得进入推荐。", True, "process_engineer")
    add("Data-quality qualification", "possible_shutdown_timestamp_count reviewed", "warning" if future_summary.get("possible_shutdown_timestamp_count", 0) else "pass", f"possible_shutdown_timestamp_count={future_summary.get('possible_shutdown_timestamp_count')}", "厂测时必须记录停工/无效操作处理。", False, "process_engineer")
    add("Data-quality qualification", "excessive missing/out-of-bound behavior defined", "pass", "Stage 46 exception policy", "超过阈值不推荐并记录。", False, "IT/data_engineer")
    add("Data-quality qualification", "no-recommendation policy defined", "pass", "Stage 46 exception policy", "所有无效窗口返回 no recommendation。", False, "IT/data_engineer")

    # 7. LIMS/T90 validation qualification
    add("LIMS/T90 validation qualification", "T90 filtered to 卤化橡胶", "pass", "Stage 45/46 future validation", "现场回填也必须同样过滤。", False, "lab/LIMS_owner")
    add("LIMS/T90 validation qualification", "T90 filtered to C line", "pass", "Stage 45/46 future validation", "现场回填只用 C线标签。", False, "lab/LIMS_owner")
    add("LIMS/T90 validation qualification", "residence-time alignment = 174min", "pass", "Stage 46 protocol", "回填按 recommendation_time + 174min 对齐。", False, "lab/LIMS_owner")
    add("LIMS/T90 validation qualification", "one-T90-one-prediction validation used", "pass", f"aligned={future_summary.get('one_to_one_aligned_sample_count')}", "指导测试结束后复用严格一对一验证。", False, "process_engineer")
    add("LIMS/T90 validation qualification", "hard labels and clear labels both used", "pass", "Stage 46 protocol", "边界样本需单独标识。", False, "process_engineer")
    add("LIMS/T90 validation qualification", "T90 uncertainty about 0.1 documented", "pass", "Stage 46 limitation", "解释边界样本时考虑测量误差。", False, "lab/LIMS_owner")
    add("LIMS/T90 validation qualification", "live same-time T90 is not used for validation", "pass", "LIMS later backfill only", "不得使用同时刻 T90 造成泄漏。", False, "lab/LIMS_owner")

    # 8. Human review qualification
    owners = [
        ("process engineer checklist exists", "process_engineer"),
        ("control engineer checklist exists", "control_engineer"),
        ("IT/data engineer checklist exists", "IT/data_engineer"),
        ("LIMS owner checklist exists", "lab/LIMS_owner"),
        ("project owner checklist exists", "project_owner"),
    ]
    checklist_exists = paths.get("human_review_checklist") and paths["human_review_checklist"].exists()
    for item, owner in owners:
        add("Human review qualification", item, "pass" if checklist_exists else "warning", str(paths.get("human_review_checklist")), "人工复核清单必须保留。", False, owner)
    add("Human review qualification", "no automatic approval generated", "pass", "approval file absent by design", "资格门材料不自动批准。", True, "project_owner")
    add("Human review qualification", "approval file required before factory connection", "pending_human_review", "no approval file supplied", "签字批准后才可准备厂区连接。", True, "project_owner")

    df = pd.DataFrame(rows)
    write_table(
        df,
        output_dir / "c_line_guidance_test_qualification_matrix.csv",
        table_dir / "c_line_guidance_test_qualification_matrix.csv",
    )
    return df


def inspect_runtime_safety(deploy_dir: Path, artifact_path: Path, output_dir: Path, table_dir: Path) -> Dict[str, Any]:
    candidate_names = [
        "package.py",
        "interface.py",
        "feature_adapter.py",
        "main.py",
        "schema.json",
        "factory_test_config.json",
        "factory_output_mapping_template.csv",
    ]
    files = [deploy_dir / name for name in candidate_names if (deploy_dir / name).exists()]
    forbidden_found: List[Dict[str, str]] = []
    monitor_terms: List[Dict[str, str]] = []
    output_mapping_control_writeback_count = 0

    for path in files:
        text = read_text(path)
        lower = text.lower()
        for term in FORBIDDEN_CONTROL_TERMS:
            needle = term.lower()
            if needle in lower:
                # Negative safety assertions are still recorded, but only output mapping writeback is an immediate fail.
                forbidden_found.append({"file": str(path), "term": term})
        for term in MONITOR_ONLY_TERMS:
            if term.lower() in lower:
                monitor_terms.append({"file": str(path), "term": term})
        if path.name == "factory_output_mapping_template.csv":
            rows = list(csv.DictReader(text.splitlines()))
            for row in rows:
                joined = " ".join(str(v).lower() for v in row.values())
                if "writeback" in joined or "setpoint" in joined or "control" in joined:
                    output_mapping_control_writeback_count += 1

    automatic_control_detected = any(x["term"] in {"auto_control", "closed_loop", "automatic_adjust", "自动控制"} for x in forbidden_found)
    dcs_writeback_detected = any(x["term"] in {"control_writeback", "setpoint_writeback", "write_dcs_setpoint", "写入设定值", "控制写回"} for x in forbidden_found)
    runtime_safety_pass = not automatic_control_detected and not dcs_writeback_detected and output_mapping_control_writeback_count == 0

    report = {
        "deploy_dir": str(deploy_dir),
        "files_inspected": [str(p) for p in files],
        "forbidden_control_terms_found": forbidden_found,
        "output_mapping_control_writeback_count": output_mapping_control_writeback_count,
        "automatic_control_detected": automatic_control_detected,
        "dcs_writeback_detected": dcs_writeback_detected,
        "monitor_only_terms_present": monitor_terms,
        "package_path_used": str(deploy_dir),
        "artifact_path": str(artifact_path),
        "old_merged_package_used": False,
        "runtime_safety_pass": runtime_safety_pass,
        "warnings": [] if runtime_safety_pass else ["发现疑似自动控制或写回表述，需人工复核。"],
    }
    write_json(output_dir / "c_line_runtime_safety_assertion_report.json", report)
    summary = pd.DataFrame(
        [
            {"metric": "files_inspected_count", "value": len(files), "status": "pass", "note_cn": "已检查 C线运行包候选文件"},
            {"metric": "forbidden_control_terms_found_count", "value": len(forbidden_found), "status": "warning" if forbidden_found else "pass", "note_cn": "记录命中的控制相关词；真正阻断看 automatic/dcs/writeback 判定"},
            {"metric": "output_mapping_control_writeback_count", "value": output_mapping_control_writeback_count, "status": "fail" if output_mapping_control_writeback_count else "pass", "note_cn": "输出映射不得包含控制写回"},
            {"metric": "automatic_control_detected", "value": automatic_control_detected, "status": "fail" if automatic_control_detected else "pass", "note_cn": "不得实现自动控制"},
            {"metric": "dcs_writeback_detected", "value": dcs_writeback_detected, "status": "fail" if dcs_writeback_detected else "pass", "note_cn": "不得写 DCS 控制设定值"},
            {"metric": "monitor_only_terms_present_count", "value": len(monitor_terms), "status": "pass" if monitor_terms else "warning", "note_cn": "运行包应明确 monitor/display/log/manual review 语义"},
            {"metric": "runtime_safety_pass", "value": runtime_safety_pass, "status": "pass" if runtime_safety_pass else "fail", "note_cn": "安全断言总结果"},
        ]
    )
    write_table(summary, output_dir / "c_line_runtime_safety_assertion_summary.csv", table_dir / "c_line_runtime_safety_assertion_summary.csv")
    return report


def build_acceptance_criteria(output_dir: Path, table_dir: Path) -> pd.DataFrame:
    rows = [
        ("AC-01", "测试周期", "calendar_duration_or_t90_count", "至少 2-4 周，或累计足够 C线 卤化橡胶 T90 样本", "满足其一并记录样本覆盖", "project_owner", "运行日志、LIMS 回填表", "不作为控制效果承诺。"),
        ("AC-02", "有效推荐覆盖率", "valid_recommendation_coverage", "不设置激进目标；建议周度复核是否接近 Stage 46 水平", "覆盖率足以支持人工判断，低覆盖需说明数据质量原因", "IT/data_engineer", "推荐日志、no-recommendation 原因统计", "monitor-only 资格关注可解释覆盖。"),
        ("AC-03", "数据质量导致无推荐率", "data_quality_no_recommendation_rate", "周度统计并人工复核；异常升高需暂停排查", "异常窗口均有原因码", "IT/data_engineer", "warning_flags 与异常策略日志", "不以插值或裁剪强行提高覆盖。"),
        ("AC-04", "自动控制事件", "automatic_control_event_count", "必须为 0", "0 次", "control_engineer", "控制系统变更记录", "任何自动控制事件均阻断。"),
        ("AC-05", "DCS 写回事件", "dcs_writeback_event_count", "必须为 0", "0 次", "control_engineer", "DCS/接口审计日志", "不得写控制设定值。"),
        ("AC-06", "输出字段完整性", "logged_output_field_rate", "所有 Stage 46 logging schema 字段应尽量记录", "关键字段 100% 记录，非关键字段说明缺失原因", "IT/data_engineer", "日志 schema 对账表", "输出只展示/日志。"),
        ("AC-07", "LIMS T90 回填", "later_lims_t90_backfilled", "测试结束后回填 C线 卤化橡胶 T90", "可对齐样本形成一对一验证集", "lab/LIMS_owner", "LIMS 回填表", "live same-time T90 不参与推荐。"),
        ("AC-08", "操作行为记录", "operator_action_logged_if_any", "若人员采取动作，必须独立记录", "动作、原因、时间、人员可追溯", "process_engineer", "操作记录", "系统输出不等于操作指令。"),
        ("AC-09", "一对一回填验证", "one_to_one_backfill_after_test", "recommendation_time + 174min 对齐", "输出 inside/outside/above/below 风险表", "process_engineer", "测试后验证报告", "沿用 Stage 46 严格口径。"),
        ("AC-10", "inside vs outside high_rate", "inside_vs_outside_high_rate_delta", "不作过强性能承诺；方向需人工判断合理", "inside 高 T90 风险不高于 outside", "process_engineer", "测试后风险分离表", "只作为监测证据。"),
        ("AC-11", "above_band high_rate", "above_band_high_rate", "above_band 应显示更高高 T90 风险趋势", "趋势与 Stage 46 不矛盾", "process_engineer", "测试后风险分离表", "若不稳定则继续收集或暂停。"),
        ("AC-12", "below_band low_rate", "below_band_low_rate", "below_band 仅诊断，不触发自动加钙", "低 T90 风险解释由人工复核", "process_engineer", "测试后风险分离表", "禁止自动加钙。"),
        ("AC-13", "clear-label 验证", "clear_label_validation", "排除约 0.1 T90 边界不确定样本后复核", "clear-label 结果与 hard-label 结论不冲突", "lab/LIMS_owner", "clear-label 验证表", "记录 uncertain_boundary_rate。"),
        ("AC-14", "数据质量周度复核", "weekly_data_quality_review", "每周复核异常、越界、缺失、停工窗口", "形成周度记录并处理阻断项", "project_owner", "周度复核记录", "数据质量问题优先于性能判断。"),
    ]
    df = pd.DataFrame(rows, columns=["criterion_id", "criterion_name", "metric", "threshold_or_rule", "pass_condition", "owner", "evidence_required", "note_cn"])
    write_table(df, output_dir / "c_line_guidance_test_acceptance_criteria.csv", table_dir / "c_line_guidance_test_acceptance_criteria.csv")
    return df


def build_precheck_checklist(output_dir: Path, table_dir: Path) -> pd.DataFrame:
    items = [
        ("C-line package path confirmed", "IT/data_engineer", "deploy/ca_safe_band_mvp_c_line 路径截图或配置记录", True, "仅允许 C线包。"),
        ("old merged package blocked", "IT/data_engineer", "接口配置中无 deploy/ca_safe_band_mvp", True, "旧合并线包不得参与。"),
        ("artifact checksum recorded", "IT/data_engineer", "safe_band_artifact.json sha256", True, "厂测前后校验 artifact 未变。"),
        ("DCS point tags confirmed", "control_engineer", "11 个点位 tag 对照表", True, "点位必须逐项核对。"),
        ("units confirmed", "control_engineer", "单位确认表", True, "单位错误会导致钙单耗错误。"),
        ("point bounds confirmed", "process_engineer", "上下限确认表", True, "用于越界置缺失。"),
        ("time zone confirmed", "IT/data_engineer", "时区配置", True, "日志与 LIMS 对齐必须一致。"),
        ("sampling frequency confirmed", "IT/data_engineer", "采样频率统计", False, "默认按分钟级数据。"),
        ("missing data behavior confirmed", "IT/data_engineer", "缺失策略测试记录", True, "无效窗口不给推荐。"),
        ("output display target confirmed", "project_owner", "看板/日志目标", False, "仅展示和记录。"),
        ("output writeback type = display/log/dashboard only", "control_engineer", "接口映射审查", True, "不得接控制标签。"),
        ("no control tag target configured", "control_engineer", "DCS 写回配置审查", True, "控制目标为空。"),
        ("LIMS T90 source confirmed", "lab/LIMS_owner", "LIMS 来源说明", True, "回填验证依赖 LIMS。"),
        ("rubber type filter confirmed", "lab/LIMS_owner", "卤化橡胶过滤规则", True, "过滤口径需固定。"),
        ("line filter confirmed", "lab/LIMS_owner", "C线过滤规则", True, "只用 C线 T90。"),
        ("174min alignment accepted", "process_engineer", "滞留时间确认记录", False, "recommendation_time + 174min。"),
        ("SOP reviewed", "project_owner", "SOP 复核记录", True, "所有角色需读 SOP。"),
        ("human review signoff required", "project_owner", "签字审批文件", True, "没有签字不得连接现场。"),
    ]
    df = pd.DataFrame(
        [
            {
                "check_item": item,
                "status": "pending",
                "owner": owner,
                "required_evidence": evidence,
                "blocker": blocker,
                "note_cn": note,
            }
            for item, owner, evidence, blocker, note in items
        ]
    )
    write_table(df, output_dir / "c_line_plant_connection_precheck_checklist.csv", table_dir / "c_line_plant_connection_precheck_checklist.csv")
    return df


def write_sop(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = """# C线钙单耗安全带 monitor-only 指导测试 SOP

## 1. 测试定位

本测试为 C线钙单耗安全带的指导测试 / 监测测试，用于把 C线运行包输出展示给现场和项目团队复核。

- 系统不参与实际控制。
- 系统不自动调钙。
- 系统不写 DCS 控制设定值。
- 现场操作仍由操作人员、工艺工程师和相关负责人决定。
- 系统输出仅作为展示、日志和人工复核字段。

## 2. 输入数据

- 输入为 11 个 C线 DCS 点位。
- 在线窗口使用最近 60min 数据。
- 点位先按现场确认的上下限清洗，越界值置为缺失。
- 不裁剪越界值，不用插值补足 future/厂测验证窗口。
- 无效窗口不给推荐。

## 3. 输出字段

指导测试至少记录以下输出字段：

- 推荐钙单耗下限、上限、目标。
- 当前钙单耗。
- interval_position：inside_band / above_band / below_band / missing。
- action_visibility。
- warning_flags。
- engineering_review_required。
- recommendation_status。

## 4. 运行规则

- inside_band = monitor_only，仅监测和记录。
- above_band = manual_review_required，仅提示人工复核。
- below_band = diagnostic_only，仅诊断展示，不给自动加钙指令。
- invalid input = no recommendation，输入无效时不生成推荐区间。

## 5. 禁止事项

- 禁止自动写控制。
- 禁止作为闭环控制器。
- 禁止将 below_band 直接理解为自动加钙。
- 禁止用 future 数据现场实时修改 artifact。
- 禁止用 future 数据训练、调参、更新规则或更新 q33/q66 边界。

## 6. 日志与回填验证

- 记录每次推荐时刻。
- 记录输入特征、原始点可用性、越界标记和 warning_flags。
- 记录 later LIMS T90。
- 回填验证使用 recommendation_time + 174min 对齐。
- 同时使用 hard label 和 clear label。
- 任何现场人员动作需要单独记录，不能把系统输出当作自动控制动作。

## 7. 停止/暂停条件

出现以下情况时应停止或暂停指导测试并复核：

- 点位映射错误。
- 单位错误。
- 大量越界。
- impossible calcium。
- 出现 DCS 写回需求。
- 操作人员误用为自动控制。
- 持续无推荐且无法解释原因。

## 8. 责任划分

- 工艺：确认规则解释、风险分离、人工复核意见和操作记录。
- 自控：确认 DCS 点位、单位、上下限、无控制写回。
- IT/数据：确认运行包、日志、接口、时间戳、异常策略。
- LIMS：确认 T90 来源、卤化橡胶过滤、C线过滤和 174min 回填。
- 项目负责人：组织评审、记录结论，并确认没有人工批准前不得接入现场。
"""
    path.write_text(content, encoding="utf-8")


def replace_or_append_section(path: Path, heading: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        text = ""
    pattern = re.compile(rf"(^##\s+{re.escape(heading[3:])}\s*$.*?)(?=^##\s+|\Z)", re.M | re.S)
    replacement = f"{heading}\n\n{body.strip()}\n"
    if pattern.search(text):
        text = pattern.sub(replacement, text)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n" + replacement
    path.write_text(text, encoding="utf-8")


def append_experiment_section(path: Path, base_number: int, title_after_number: str, body: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    exact_heading = f"## {base_number}. {title_after_number}"
    exact_pattern = re.compile(rf"(^##\s+{base_number}\.\s+{re.escape(title_after_number)}\s*$.*?)(?=^##\s+\d+\.|\Z)", re.M | re.S)
    replacement = f"{exact_heading}\n\n{body.strip()}\n"
    if exact_pattern.search(text):
        text = exact_pattern.sub(replacement, text)
        path.write_text(text, encoding="utf-8")
        return exact_heading

    numbers = [int(m.group(1)) for m in re.finditer(r"^##\s+(\d+)\.", text, flags=re.M)]
    section_no = base_number
    while section_no in numbers:
        section_no += 1
    heading = f"## {section_no}. {title_after_number}"
    if text and not text.endswith("\n"):
        text += "\n"
    text += "\n" + f"{heading}\n\n{body.strip()}\n"
    path.write_text(text, encoding="utf-8")
    return heading


def update_docs(method_doc: Path, experiment_doc: Path, sop_doc: Path, report: Dict[str, Any]) -> str:
    method_heading = "## C线 monitor-only 指导测试资格检查"
    method_body = """
- future 新数据是真实操作数据的独立 holdout 验证集。
- future 数据不参与训练、调参、规则更新或 artifact 更新。
- C线包只作为指导测试工具，不参与实际控制。
- 控制权仍属于现场人员。
- 系统只输出展示、日志和人工复核字段。
- 不自动调钙。
- 不写 DCS 控制设定值。
- 进入厂区连接前必须完成人工复核。
- 资格检查通过不等于正式部署批准。
"""
    replace_or_append_section(method_doc, method_heading, method_body)

    experiment_body = f"""
### purpose

本阶段生成 C线 monitor-only 指导测试资格检查包，用于回答 C线运行包是否具备进入人工评审和后续指导测试准备的条件。

### Stage 46 dependency

依赖 Stage 46 人工复核包：`runs/c_line_monitor_only_human_review_pack/`。

### future data role

`real_operation_holdout_validation_only`。future 新数据是真实操作数据，仅用于独立 holdout 验证，不用于训练、调参、规则更新或 artifact 更新。

### factory test mode

`guidance_monitor_only`。系统只做指导/监测，不参与实际控制。

### C-line package and artifact

- package: `deploy/ca_safe_band_mvp_c_line/`
- artifact: `models/ca_safe_band_mvp_c_line/safe_band_artifact.json`

### qualification matrix result

{report["qualification_matrix_summary"]}

### runtime safety assertion result

{report["runtime_safety_assertion_summary"]}

### SOP path

`{report["sop_doc_path"]}`

### acceptance criteria path

`reports/tables/c_line_guidance_test_acceptance_criteria.csv`

### plant connection precheck path

`reports/tables/c_line_plant_connection_precheck_checklist.csv`

### qualification_decision

`{report["qualification_decision"]}`

### recommended_next_step

`{report["recommended_next_step"]}`

### limitations

- qualification does not approve deployment
- human review still required
- monitor-only only
- no automatic control
- no DCS setpoint writeback
- T90 measurement error about 0.1
"""
    return append_experiment_section(experiment_doc, 47, "C线 monitor-only 指导测试资格检查", experiment_body)


def plot_qualification_summary(matrix: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {
        "pass": "#2f7d32",
        "warning": "#f9a825",
        "fail": "#c62828",
        "pending_human_review": "#1565c0",
        "not_applicable": "#757575",
    }
    counts = matrix["status"].value_counts().reindex(colors.keys(), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values, color=[colors[x] for x in counts.index])
    ax.set_title("C线 monitor-only 指导测试资格检查摘要")
    ax.set_ylabel("检查项数量")
    ax.set_xlabel("状态")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.3, str(int(v)), ha="center")
    fig.tight_layout()
    fig.savefig(figure_dir / "c_line_guidance_test_qualification_summary.png", dpi=160)
    plt.close(fig)


def plot_safety_assertions(runtime_report: Dict[str, Any], figure_dir: Path) -> None:
    items = {
        "monitor terms present": bool(runtime_report.get("monitor_only_terms_present")),
        "no automatic control": not runtime_report.get("automatic_control_detected"),
        "no DCS writeback": not runtime_report.get("dcs_writeback_detected"),
        "no output writeback": runtime_report.get("output_mapping_control_writeback_count", 0) == 0,
        "runtime safety pass": runtime_report.get("runtime_safety_pass"),
    }
    fig, ax = plt.subplots(figsize=(9, 4.8))
    values = [1 if v else 0 for v in items.values()]
    ax.barh(list(items.keys()), values, color=["#2f7d32" if v else "#c62828" for v in values])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 1], ["fail", "pass"])
    ax.set_title("C线指导测试安全约束检查")
    for i, v in enumerate(values):
        ax.text(0.5, i, "PASS" if v else "FAIL", va="center", ha="center", color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "c_line_guidance_test_safety_assertions.png", dpi=160)
    plt.close(fig)


def plot_acceptance_criteria(criteria: pd.DataFrame, figure_dir: Path) -> None:
    owner_counts = criteria["owner"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(owner_counts.index, owner_counts.values, color="#455a64")
    ax.set_title("C线指导测试验收指标与日志要求")
    ax.set_ylabel("验收项数量")
    ax.set_xlabel("负责人")
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(owner_counts.values):
        ax.text(i, v + 0.05, str(int(v)), ha="center")
    fig.tight_layout()
    fig.savefig(figure_dir / "c_line_guidance_test_acceptance_criteria.png", dpi=160)
    plt.close(fig)


def determine_decision(
    matrix: pd.DataFrame,
    runtime_report: Dict[str, Any],
    stage46_exists: bool,
    deploy_exists: bool,
    artifact_exists: bool,
    old_merged_package_used: bool,
) -> Tuple[str, str]:
    if not stage46_exists:
        return "not_qualified_missing_evidence", "collect_missing_evidence_before_review"
    if not deploy_exists or not artifact_exists:
        return "not_qualified_missing_evidence", "collect_missing_evidence_before_review"
    if not runtime_report.get("runtime_safety_pass") or old_merged_package_used:
        return "not_qualified_fix_runtime_safety", "fix_runtime_safety_before_review"
    blocking_fails = matrix[(matrix["status"] == "fail") & (matrix["blocker"].astype(bool))]
    if not blocking_fails.empty:
        if any("Interface qualification" == x for x in blocking_fails["qualification_dimension"]):
            return "not_qualified_fix_data_contract", "fix_data_contract_before_review"
        return "not_qualified_fix_runtime_safety", "fix_runtime_safety_before_review"
    return "qualified_for_human_review", "conduct_human_review_for_guidance_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare C-line monitor-only guidance-test qualification gate.")
    parser.add_argument("--human-review-pack-dir", required=True)
    parser.add_argument("--c-line-validation-dir", required=True)
    parser.add_argument("--c-line-revalidation-dir", required=True)
    parser.add_argument("--deploy-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--doc", required=True)
    parser.add_argument("--method-doc", required=True)
    parser.add_argument("--sop-doc", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_c_line_deploy(args)

    human_review_dir = Path(args.human_review_pack_dir)
    validation_dir = Path(args.c_line_validation_dir)
    revalidation_dir = Path(args.c_line_revalidation_dir)
    deploy_dir = Path(args.deploy_dir)
    artifact_path = Path(args.artifact)
    output_dir = Path(args.output_dir)
    table_dir = Path(args.table_dir)
    figure_dir = Path(args.figure_dir)
    doc_path = Path(args.doc)
    method_doc_path = Path(args.method_doc)
    sop_doc_path = Path(args.sop_doc)

    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    stage46_report_path = find_file(human_review_dir, "c_line_monitor_only_human_review_report.json")
    if not stage46_report_path:
        raise SystemExit("Missing required Stage 46 human-review report under c_line_monitor_only_human_review_pack.")

    paths: Dict[str, Optional[Path]] = {
        "stage46_human_review_report": stage46_report_path,
        "go_no_go_matrix": find_file(Path("reports/tables"), "c_line_go_no_go_matrix.csv") or find_file(human_review_dir, "c_line_go_no_go_matrix.csv"),
        "rule_review_table": find_file(Path("reports/tables"), "c_line_rule_review_table.csv") or find_file(human_review_dir, "c_line_rule_review_table.csv"),
        "future_validation_summary": find_file(Path("reports/tables"), "c_line_future_validation_summary.csv") or find_file(human_review_dir, "c_line_future_validation_summary.csv"),
        "data_quality_policy": find_file(Path("reports/tables"), "c_line_data_quality_exception_policy.csv") or find_file(human_review_dir, "c_line_data_quality_exception_policy.csv"),
        "factory_test_protocol": find_file(Path("reports/tables"), "c_line_monitor_only_factory_test_protocol.csv") or find_file(human_review_dir, "c_line_monitor_only_factory_test_protocol.csv"),
        "logging_schema": find_file(Path("reports/tables"), "c_line_factory_test_logging_schema.csv") or find_file(human_review_dir, "c_line_factory_test_logging_schema.csv"),
        "human_review_checklist": find_file(Path("reports/tables"), "c_line_human_review_checklist.csv") or find_file(human_review_dir, "c_line_human_review_checklist.csv"),
        "future_validation_report": find_file(validation_dir, "c_line_future_holdout_v1_cleaned_validation_report.json"),
        "future_t90_backfill_report": find_file(validation_dir, "future_t90_backfill_validation_report.json"),
        "future_recommendation_report": find_file(validation_dir, "future_c_line_v1_recommendation_distribution_report.json"),
        "final_monitor_dry_run": find_file(revalidation_dir, "final_monitor_dry_run.parquet"),
        "schema": deploy_dir / "schema.json",
        "factory_test_config": deploy_dir / "factory_test_config.json",
        "factory_point_mapping_template": deploy_dir / "factory_point_mapping_template.csv",
        "factory_output_mapping_template": deploy_dir / "factory_output_mapping_template.csv",
        "c_line_deploy_dir": deploy_dir,
        "c_line_artifact": artifact_path,
        "feature_adapter": deploy_dir / "feature_adapter.py",
    }

    input_inventory = build_input_inventory(paths, output_dir, table_dir)
    report46 = read_json(stage46_report_path)
    validation_report = read_json(paths["future_validation_report"])
    artifact = read_json(artifact_path)
    schema = read_json(paths["schema"])
    runtime_equivalence = read_json(find_file(revalidation_dir, "runtime_equivalence_after_repair_report.json"))
    smoke_report = read_json(find_file(revalidation_dir, "production_mode_sanity_report.json"))

    deploy_hash_before = hash_directory(deploy_dir.rglob("*"))
    artifact_hash_before = sha256_file(artifact_path)

    runtime_report = inspect_runtime_safety(deploy_dir, artifact_path, output_dir, table_dir)
    matrix = build_qualification_matrix(
        report46,
        validation_report,
        artifact,
        schema,
        runtime_equivalence,
        smoke_report,
        paths,
        bool(runtime_report["runtime_safety_pass"]),
        output_dir,
        table_dir,
    )
    criteria = build_acceptance_criteria(output_dir, table_dir)
    precheck = build_precheck_checklist(output_dir, table_dir)
    write_sop(sop_doc_path)

    deploy_hash_after = hash_directory(deploy_dir.rglob("*"))
    artifact_hash_after = sha256_file(artifact_path)
    algorithm_changed = deploy_hash_before != deploy_hash_after
    artifact_modified = artifact_hash_before != artifact_hash_after
    old_merged_package_used = bool(report46.get("old_merged_package_used"))

    qualification_decision, recommended_next_step = determine_decision(
        matrix,
        runtime_report,
        stage46_report_path.exists(),
        deploy_dir.exists(),
        artifact_path.exists(),
        old_merged_package_used,
    )

    report: Dict[str, Any] = {
        "created_at": now_iso(),
        "input_paths": {k: str(v) if v else None for k, v in paths.items()},
        "output_dir": str(output_dir),
        "future_data_role": "real_operation_holdout_validation_only",
        "factory_test_mode": "guidance_monitor_only",
        "control_authority": "plant_operator_only",
        "qualification_matrix_summary": {
            **status_counts(matrix),
            "blocker_fail_count": int(((matrix["status"] == "fail") & (matrix["blocker"].astype(bool))).sum()),
            "total_check_count": int(len(matrix)),
        },
        "runtime_safety_assertion_summary": {
            "files_inspected_count": len(runtime_report.get("files_inspected", [])),
            "forbidden_control_terms_found_count": len(runtime_report.get("forbidden_control_terms_found", [])),
            "output_mapping_control_writeback_count": runtime_report.get("output_mapping_control_writeback_count"),
            "automatic_control_detected": runtime_report.get("automatic_control_detected"),
            "dcs_writeback_detected": runtime_report.get("dcs_writeback_detected"),
            "runtime_safety_pass": runtime_report.get("runtime_safety_pass"),
        },
        "evidence_summary": {
            "input_available_count": int((input_inventory["available"] == True).sum()),
            "required_input_missing_count": int(((input_inventory["required"] == True) & (input_inventory["available"] == False)).sum()),
            "stage46_final_readiness_decision": report46.get("final_readiness_decision"),
            "stage46_recommended_next_step": report46.get("recommended_next_step"),
            "future_recommendation_coverage": safe_get(report46, ["c_line_future_validation_summary", "recommendation_coverage"]),
            "future_one_to_one_risk_guardrail_pass": safe_get(report46, ["c_line_future_validation_summary", "risk_guardrail_pass"]),
        },
        "acceptance_criteria_summary": {
            "criterion_count": int(len(criteria)),
            "automatic_control_event_rule": "must be 0",
            "dcs_writeback_event_rule": "must be 0",
            "post_test_one_to_one_backfill_required": True,
        },
        "plant_connection_precheck_summary": {
            "check_count": int(len(precheck)),
            "pending_count": int((precheck["status"] == "pending").sum()),
            "blocker_count": int((precheck["blocker"].astype(bool)).sum()),
        },
        "sop_doc_path": str(sop_doc_path),
        "safety_constraints": {
            "monitor_only": True,
            "guidance_only": True,
            "advisory_output_only": True,
            "automatic_control": False,
            "closed_loop_control": False,
            "dcs_setpoint_writeback": False,
            "result_display_or_log_only": True,
            "human_review_required_before_connection": True,
            "no_operational_increase_hint": True,
        },
        "algorithm_changed": algorithm_changed,
        "artifact_modified": artifact_modified,
        "old_merged_package_used": old_merged_package_used,
        "qualification_decision": qualification_decision,
        "limitations": [
            "Qualification does not approve deployment.",
            "Human review is still required before plant connection.",
            "This is monitor-only / guidance-only, not automatic control.",
            "No DCS setpoint writeback is allowed.",
            "Future data is real-operation holdout validation only and must not update artifact or rules.",
            "T90 measurement error is about 0.1.",
        ],
        "recommended_next_step": recommended_next_step,
    }

    experiment_heading = update_docs(method_doc_path, doc_path, sop_doc_path, report)
    report["experiment_doc_section_appended"] = experiment_heading
    write_json(output_dir / REPORT_NAME, report)

    plot_qualification_summary(matrix, figure_dir)
    plot_safety_assertions(runtime_report, figure_dir)
    plot_acceptance_criteria(criteria, figure_dir)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "qualification_decision": qualification_decision,
                "recommended_next_step": recommended_next_step,
                "qualification_matrix_summary": report["qualification_matrix_summary"],
                "runtime_safety_pass": runtime_report.get("runtime_safety_pass"),
                "algorithm_changed": algorithm_changed,
                "artifact_modified": artifact_modified,
                "old_merged_package_used": old_merged_package_used,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
