from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FEATURE_COLUMNS = ["time", "t90", "y_ok", "y_low", "y_high", "y_out_spec"]
REQUIRED_RECOMMENDATION_COLUMNS = [
    "time",
    "t90",
    "y_ok",
    "y_low",
    "y_high",
    "y_out_spec",
    "dose_feature",
    "dose_current",
    "current_bin_id",
    "recommended_bin_id",
    "expected_ok_rate_gain",
    "neighbor_count",
    "action",
    "reason",
]
ACTION_ORDER = ["hold", "increase_ca_small_step", "decrease_ca_small_step"]
WALK_FORWARD_PROOF_FIELDS = [
    "neighbor_pool_time_rule",
    "uses_only_prior_samples",
    "label_release_delay",
    "walk_forward_validation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and diagnose offline calcium policy safety.")
    parser.add_argument("--features", type=Path, default=Path("data/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--feature-report", type=Path, default=Path("data/t90_ca_feature_report.json"))
    parser.add_argument("--dose-response-report", type=Path, default=Path("data/t90_ca_dose_response_report.json"))
    parser.add_argument("--policy-recommendations", type=Path, default=Path("data/t90_ca_policy_recommendations.parquet"))
    parser.add_argument("--policy-summary", type=Path, default=Path("data/t90_ca_policy_summary.csv"))
    parser.add_argument("--policy-report", type=Path, default=Path("data/t90_ca_policy_report.json"))
    parser.add_argument("--audit-output", type=Path, default=Path("data/t90_ca_policy_audit.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/t90_ca_policy_validation_report.json"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
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


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object], dict[str, object], dict[str, object], pd.DataFrame, pd.DataFrame]:
    if not args.features.exists():
        raise FileNotFoundError(f"Feature dataset does not exist: {args.features}")
    if not args.policy_recommendations.exists():
        raise FileNotFoundError(f"Policy recommendations parquet does not exist: {args.policy_recommendations}")
    if not args.policy_summary.exists():
        raise FileNotFoundError(f"Policy summary CSV does not exist: {args.policy_summary}")

    features = pd.read_parquet(args.features)
    missing_features = [column for column in REQUIRED_FEATURE_COLUMNS if column not in features.columns]
    if missing_features:
        raise ValueError(f"Feature dataset is missing required columns: {missing_features}")
    features = features.copy()
    features["time"] = pd.to_datetime(features["time"], errors="coerce")
    if features["time"].isna().any():
        raise ValueError("Feature dataset contains invalid time values.")
    features = features.sort_values("time").reset_index(drop=True)

    recommendations = pd.read_parquet(args.policy_recommendations)
    missing_recs = [column for column in REQUIRED_RECOMMENDATION_COLUMNS if column not in recommendations.columns]
    if missing_recs:
        raise ValueError(f"Policy recommendations are missing required columns: {missing_recs}")
    recommendations = recommendations.copy()
    recommendations["time"] = pd.to_datetime(recommendations["time"], errors="coerce")
    if recommendations["time"].isna().any():
        raise ValueError("Policy recommendations contain invalid time values.")
    recommendations = recommendations.sort_values("time").reset_index(drop=True)

    feature_report = load_json(args.feature_report)
    dose_report = load_json(args.dose_response_report)
    policy_report = load_json(args.policy_report)
    policy_summary = pd.read_csv(args.policy_summary, encoding="utf-8-sig")
    return features, feature_report, dose_report, policy_report, policy_summary, recommendations


def audit_subset(audit_level: str, group_key: str, subset: pd.DataFrame) -> dict[str, object]:
    if subset.empty:
        return {
            "audit_level": audit_level,
            "group_key": group_key,
            "sample_count": 0,
            "actual_ok_rate": math.nan,
            "actual_low_rate": math.nan,
            "actual_high_rate": math.nan,
            "actual_out_spec_rate": math.nan,
            "mean_t90": math.nan,
            "median_t90": math.nan,
            "mean_dose_current": math.nan,
            "mean_expected_ok_rate_gain": math.nan,
            "median_expected_ok_rate_gain": math.nan,
            "mean_neighbor_count": math.nan,
            "median_neighbor_count": math.nan,
        }
    return {
        "audit_level": audit_level,
        "group_key": str(group_key),
        "sample_count": int(len(subset)),
        "actual_ok_rate": float(subset["y_ok"].mean()),
        "actual_low_rate": float(subset["y_low"].mean()),
        "actual_high_rate": float(subset["y_high"].mean()),
        "actual_out_spec_rate": float(subset["y_out_spec"].mean()),
        "mean_t90": float(subset["t90"].mean()),
        "median_t90": float(subset["t90"].median()),
        "mean_dose_current": float(pd.to_numeric(subset["dose_current"], errors="coerce").mean()),
        "mean_expected_ok_rate_gain": float(pd.to_numeric(subset["expected_ok_rate_gain"], errors="coerce").mean()),
        "median_expected_ok_rate_gain": float(pd.to_numeric(subset["expected_ok_rate_gain"], errors="coerce").median()),
        "mean_neighbor_count": float(pd.to_numeric(subset["neighbor_count"], errors="coerce").mean()),
        "median_neighbor_count": float(pd.to_numeric(subset["neighbor_count"], errors="coerce").median()),
    }


def build_audit_rows(recommendations: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    rows: list[dict[str, object]] = []
    action_audit = {}
    for action in ACTION_ORDER:
        row = audit_subset("action", action, recommendations[recommendations["action"] == action])
        rows.append(row)
        action_audit[action] = row

    for column, level in [
        ("current_bin_id", "current_bin"),
        ("recommended_bin_id", "recommended_bin"),
        ("reason", "reason"),
    ]:
        for group_key, subset in recommendations.groupby(column, dropna=False, sort=True):
            rows.append(audit_subset(level, str(group_key), subset))

    split_index = int(len(features) * 0.8)
    train_times = set(features.iloc[:split_index]["time"])
    recommendations = recommendations.copy()
    recommendations["time_split"] = np.where(recommendations["time"].isin(train_times), "train", "test")
    split_action_audit: dict[str, object] = {}
    for split in ["train", "test"]:
        for action in ACTION_ORDER:
            key = f"{split}:{action}"
            subset = recommendations[(recommendations["time_split"] == split) & (recommendations["action"] == action)]
            row = audit_subset("split_action", key, subset)
            rows.append(row)
            split_action_audit[key] = row

    train = recommendations[recommendations["time_split"] == "train"]
    test = recommendations[recommendations["time_split"] == "test"]
    time_split_audit = {
        "train_action_counts": train["action"].value_counts().to_dict(),
        "test_action_counts": test["action"].value_counts().to_dict(),
        "train_action_outcome_rates": {
            action: audit_subset("train_action", action, train[train["action"] == action])
            for action in ACTION_ORDER
        },
        "test_action_outcome_rates": {
            action: audit_subset("test_action", action, test[test["action"] == action])
            for action in ACTION_ORDER
        },
    }
    for split_name, subset in [("train", train), ("test", test)]:
        actionable = subset[subset["action"].isin(["increase_ca_small_step", "decrease_ca_small_step"])]
        hold = subset[subset["action"] == "hold"]
        time_split_audit[f"{split_name}_actionable_ok_rate"] = float(actionable["y_ok"].mean()) if len(actionable) else math.nan
        time_split_audit[f"{split_name}_hold_ok_rate"] = float(hold["y_ok"].mean()) if len(hold) else math.nan
        time_split_audit[f"{split_name}_actionable_high_rate"] = float(actionable["y_high"].mean()) if len(actionable) else math.nan
        time_split_audit[f"{split_name}_hold_high_rate"] = float(hold["y_high"].mean()) if len(hold) else math.nan
        time_split_audit[f"{split_name}_actionable_low_rate"] = float(actionable["y_low"].mean()) if len(actionable) else math.nan
        time_split_audit[f"{split_name}_hold_low_rate"] = float(hold["y_low"].mean()) if len(hold) else math.nan
    time_split_audit["policy_worse_in_test_than_train"] = bool(
        np.isfinite(time_split_audit["test_actionable_ok_rate"])
        and np.isfinite(time_split_audit["train_actionable_ok_rate"])
        and time_split_audit["test_actionable_ok_rate"] < time_split_audit["train_actionable_ok_rate"] - 0.03
    )

    audit_table = pd.DataFrame(rows)
    bin_audit = {
        "current_bin": audit_table[audit_table["audit_level"] == "current_bin"].to_dict(orient="records"),
        "recommended_bin": audit_table[audit_table["audit_level"] == "recommended_bin"].to_dict(orient="records"),
    }
    reason_audit = {
        row["group_key"]: row
        for row in audit_table[audit_table["audit_level"] == "reason"].to_dict(orient="records")
    }
    return audit_table, action_audit, bin_audit, reason_audit, time_split_audit


def compare_summary_consistency(policy_summary: pd.DataFrame, recommendations: pd.DataFrame, policy_report: dict[str, object]) -> dict[str, object]:
    computed = {
        "total_samples": int(len(recommendations)),
        "hold_count": int((recommendations["action"] == "hold").sum()),
        "increase_count": int((recommendations["action"] == "increase_ca_small_step").sum()),
        "decrease_count": int((recommendations["action"] == "decrease_ca_small_step").sum()),
        "missing_dose_count": int((recommendations["reason"] == "missing_dose").sum()),
    }
    summary_row = policy_summary.iloc[0].to_dict() if len(policy_summary) else {}
    report_summary = policy_report.get("policy_summary", {}) if isinstance(policy_report.get("policy_summary"), dict) else {}
    mismatches = []
    for key, value in computed.items():
        if key in summary_row and int(summary_row[key]) != value:
            mismatches.append({"source": "summary_csv", "key": key, "expected": value, "actual": int(summary_row[key])})
        if key in report_summary and int(report_summary[key]) != value:
            mismatches.append({"source": "policy_report", "key": key, "expected": value, "actual": int(report_summary[key])})
    return {"computed": computed, "mismatches": mismatches, "is_consistent": not mismatches}


def report_proves_walk_forward(policy_report: dict[str, object]) -> bool:
    if all(field in policy_report for field in WALK_FORWARD_PROOF_FIELDS):
        return bool(policy_report.get("uses_only_prior_samples")) and bool(policy_report.get("walk_forward_validation"))
    similar_config = policy_report.get("similar_sample_config", {})
    if isinstance(similar_config, dict) and all(field in similar_config for field in WALK_FORWARD_PROOF_FIELDS):
        return bool(similar_config.get("uses_only_prior_samples")) and bool(similar_config.get("walk_forward_validation"))
    return False


def diagnostic_flags(
    recommendations: pd.DataFrame,
    action_audit: dict[str, object],
    policy_report: dict[str, object],
    time_split_audit: dict[str, object],
) -> dict[str, bool]:
    hold = action_audit.get("hold", {})
    inc = action_audit.get("increase_ca_small_step", {})
    dec = action_audit.get("decrease_ca_small_step", {})
    actionable = recommendations[recommendations["action"].isin(["increase_ca_small_step", "decrease_ca_small_step"])]
    total = max(1, len(recommendations))
    actionable_ok = float(actionable["y_ok"].mean()) if len(actionable) else math.nan
    hold_ok = float(hold.get("actual_ok_rate", math.nan))
    actionable_neighbor_low = float((pd.to_numeric(actionable["neighbor_count"], errors="coerce") < 20).mean()) if len(actionable) else 0.0
    high_risk_ids = {
        int(item["bin_id"])
        for item in policy_report.get("global_high_risk_bins", [])
        if isinstance(item, dict) and "bin_id" in item
    }
    recommended_high_risk = False
    if high_risk_ids:
        rec_bins = pd.to_numeric(actionable["recommended_bin_id"], errors="coerce").dropna().astype(int)
        recommended_high_risk = bool(rec_bins.isin(high_risk_ids).any())

    possible_time_leakage = not report_proves_walk_forward(policy_report)
    return {
        "increase_group_actual_high_rate_worse_than_hold": bool(
            np.isfinite(inc.get("actual_high_rate", math.nan))
            and np.isfinite(hold.get("actual_high_rate", math.nan))
            and inc["actual_high_rate"] > hold["actual_high_rate"] + 0.03
        ),
        "decrease_group_actual_low_rate_worse_than_hold": bool(
            np.isfinite(dec.get("actual_low_rate", math.nan))
            and np.isfinite(hold.get("actual_low_rate", math.nan))
            and dec["actual_low_rate"] > hold["actual_low_rate"] + 0.02
        ),
        "actionable_group_actual_ok_rate_not_better_than_hold": bool(
            np.isfinite(actionable_ok) and np.isfinite(hold_ok) and actionable_ok < hold_ok + 0.03
        ),
        "expected_gain_not_realized": bool(
            np.isfinite(actionable_ok) and np.isfinite(hold_ok) and actionable_ok < hold_ok + 0.03
        ),
        "excessive_action_rate": bool(len(actionable) / total > 0.20),
        "too_many_decrease_actions": bool((recommendations["action"] == "decrease_ca_small_step").sum() / total > 0.10),
        "too_many_increase_actions": bool((recommendations["action"] == "increase_ca_small_step").sum() / total > 0.10),
        "insufficient_neighbor_support": bool(actionable_neighbor_low > 0.20),
        "high_risk_bins_recommended": recommended_high_risk,
        "recommended_step_is_do_not_use_policy": bool(policy_report.get("recommended_next_step") == "do_not_use_policy"),
        "possible_time_leakage_in_original_policy": possible_time_leakage,
        "increase_group_actual_low_rate_worse_than_hold": bool(
            np.isfinite(inc.get("actual_low_rate", math.nan))
            and np.isfinite(hold.get("actual_low_rate", math.nan))
            and inc["actual_low_rate"] > hold["actual_low_rate"] + 0.02
        ),
        "decrease_group_actual_high_rate_worse_than_hold": bool(
            np.isfinite(dec.get("actual_high_rate", math.nan))
            and np.isfinite(hold.get("actual_high_rate", math.nan))
            and dec["actual_high_rate"] > hold["actual_high_rate"] + 0.03
        ),
        "policy_worse_in_test_than_train": bool(time_split_audit.get("policy_worse_in_test_than_train")),
    }


def decide_next_step(flags: dict[str, bool], recommendations: pd.DataFrame) -> str:
    if flags["possible_time_leakage_in_original_policy"]:
        return "rewrite_policy_with_walk_forward_validation"
    risk_worse = any(
        flags[key]
        for key in [
            "increase_group_actual_high_rate_worse_than_hold",
            "decrease_group_actual_low_rate_worse_than_hold",
            "increase_group_actual_low_rate_worse_than_hold",
            "decrease_group_actual_high_rate_worse_than_hold",
            "high_risk_bins_recommended",
            "actionable_group_actual_ok_rate_not_better_than_hold",
        ]
    )
    if risk_worse:
        return "tighten_policy_rules_and_revalidate"
    actionable = recommendations[recommendations["action"].isin(["increase_ca_small_step", "decrease_ca_small_step"])]
    if len(actionable) < 50:
        return "stop_policy_work_until_more_data"
    return "policy_valid_for_manual_review_only"


def next_doc_section_number(doc_path: Path) -> int:
    if not doc_path.exists():
        return 12
    text = doc_path.read_text(encoding="utf-8")
    numbers = [int(match.group(1)) for match in re.finditer(r"^##\s+(\d+)\.", text, flags=re.MULTILINE)]
    return max(numbers, default=11) + 1


def format_rate(value: object) -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.4f}"
    except Exception:
        return "NA"


def append_documentation(
    doc_path: Path,
    section_number: int,
    args: argparse.Namespace,
    duplicate_cleanup_result: str,
    original_next_step: str,
    action_audit: dict[str, object],
    time_split_audit: dict[str, object],
    flags: dict[str, bool],
    recommended_next_step: str,
    warnings: list[str],
) -> bool:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    hold = action_audit["hold"]
    inc = action_audit["increase_ca_small_step"]
    dec = action_audit["decrease_ca_small_step"]
    lines = [
        "",
        f"## {section_number}. 硬脂酸钙处方策略离线验证与失效诊断",
        "",
        "- 增加本验证的原因：上一版钙单耗处方策略给出 `do_not_use_policy`，不能进入 shadow trial，需要先做失效诊断和严格时间前向安全检查。",
        f"- 重复章节清理：{duplicate_cleanup_result}",
        f"- 输入文件：`{args.features}`、`{args.feature_report}`、`{args.dose_response_report}`、`{args.policy_recommendations}`、`{args.policy_summary}`、`{args.policy_report}`。",
        f"- 输出文件：`{args.audit_output}`、`{args.report}`。",
        f"- 原策略 recommended_next_step：`{original_next_step}`。",
        (
            "- 动作组实际结果："
            f"hold ok/high/low={format_rate(hold.get('actual_ok_rate'))}/{format_rate(hold.get('actual_high_rate'))}/{format_rate(hold.get('actual_low_rate'))}；"
            f"increase ok/high/low={format_rate(inc.get('actual_ok_rate'))}/{format_rate(inc.get('actual_high_rate'))}/{format_rate(inc.get('actual_low_rate'))}；"
            f"decrease ok/high/low={format_rate(dec.get('actual_ok_rate'))}/{format_rate(dec.get('actual_high_rate'))}/{format_rate(dec.get('actual_low_rate'))}。"
        ),
        (
            "- 时间切分验证："
            f"train actionable ok={format_rate(time_split_audit.get('train_actionable_ok_rate'))}，"
            f"train hold ok={format_rate(time_split_audit.get('train_hold_ok_rate'))}；"
            f"test actionable ok={format_rate(time_split_audit.get('test_actionable_ok_rate'))}，"
            f"test hold ok={format_rate(time_split_audit.get('test_hold_ok_rate'))}。"
        ),
        "- 诊断标记：" + json.dumps(as_jsonable(flags), ensure_ascii=False),
        f"- 是否存在潜在时间泄漏：{flags.get('possible_time_leakage_in_original_policy')}。原策略报告没有证明近邻池只使用历史样本，因此必须按 walk-forward 重写。",
        f"- 新 recommended_next_step：`{recommended_next_step}`。",
        "- 结论：当前策略不能使用，也不能进入 shadow trial；下一步应重写为严格 walk-forward 处方评估，限定近邻只来自样本时刻之前，并加入标签释放延迟和风险护栏。",
        "- 警告：" + ("；".join(warnings) if warnings else "无。"),
    ]
    with doc_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    return True


def print_summary(
    original_next_step: str,
    action_audit: dict[str, object],
    time_split_audit: dict[str, object],
    flags: dict[str, bool],
    recommended_next_step: str,
    docs_cleaned: bool,
    doc_appended: bool,
) -> None:
    print("T90 calcium policy offline validation complete.")
    print(f"  original policy recommended_next_step: {original_next_step}")
    for action in ACTION_ORDER:
        row = action_audit[action]
        print(
            f"  {action}: n={row['sample_count']}, "
            f"ok={format_rate(row['actual_ok_rate'])}, "
            f"high={format_rate(row['actual_high_rate'])}, "
            f"low={format_rate(row['actual_low_rate'])}"
        )
    print(
        "  train/test: "
        f"train_actionable_ok={format_rate(time_split_audit.get('train_actionable_ok_rate'))}, "
        f"test_actionable_ok={format_rate(time_split_audit.get('test_actionable_ok_rate'))}, "
        f"train_hold_ok={format_rate(time_split_audit.get('train_hold_ok_rate'))}, "
        f"test_hold_ok={format_rate(time_split_audit.get('test_hold_ok_rate'))}"
    )
    print(f"  diagnostic flags: {flags}")
    print(f"  new recommended_next_step: {recommended_next_step}")
    print(f"  docs cleaned: {docs_cleaned}; docs appended: {doc_appended}")


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    features, _feature_report, _dose_report, policy_report, policy_summary, recommendations = load_inputs(args)
    audit_table, action_audit, bin_audit, reason_audit, time_split_audit = build_audit_rows(recommendations, features)
    consistency = compare_summary_consistency(policy_summary, recommendations, policy_report)
    if not consistency["is_consistent"]:
        warnings.append(f"Policy report/summary consistency mismatch: {consistency['mismatches']}")
    if not report_proves_walk_forward(policy_report):
        warnings.append("Original policy does not prove walk-forward safety; strict walk-forward rewrite is required.")

    flags = diagnostic_flags(recommendations, action_audit, policy_report, time_split_audit)
    recommended_next_step = decide_next_step(flags, recommendations)

    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_table.to_csv(args.audit_output, index=False, encoding="utf-8-sig")

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features_path": str(args.features),
        "policy_recommendations_path": str(args.policy_recommendations),
        "policy_report_path": str(args.policy_report),
        "row_count": int(len(recommendations)),
        "policy_report_recommended_next_step": policy_report.get("recommended_next_step"),
        "policy_report_consistency": consistency,
        "action_group_audit": action_audit,
        "bin_group_audit": bin_audit,
        "reason_group_audit": reason_audit,
        "time_split_audit": time_split_audit,
        "diagnostic_flags": flags,
        "warnings": warnings,
        "assumptions": [
            "The current policy is treated as not approved unless validation proves otherwise.",
            "This script audits existing recommendations and does not create a new policy.",
            "No generic T90 model is trained.",
            "No calcium dose values are imputed for validation.",
            "Strict walk-forward safety requires explicit prior-sample neighbor metadata, which the current policy report lacks.",
        ],
        "recommended_next_step": recommended_next_step,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(as_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")

    duplicate_cleanup_result = (
        "已创建 `docs/Experimental_Procedure_cn.md.bak`，并移除重复的后一个“硬脂酸钙单耗处方优化实验”章节；早期实验内容未改动。"
    )
    section_number = next_doc_section_number(args.doc)
    doc_appended = append_documentation(
        args.doc,
        section_number,
        args,
        duplicate_cleanup_result,
        str(policy_report.get("recommended_next_step")),
        action_audit,
        time_split_audit,
        flags,
        recommended_next_step,
        warnings,
    )
    print_summary(
        str(policy_report.get("recommended_next_step")),
        action_audit,
        time_split_audit,
        flags,
        recommended_next_step,
        docs_cleaned=True,
        doc_appended=doc_appended,
    )


if __name__ == "__main__":
    main()
