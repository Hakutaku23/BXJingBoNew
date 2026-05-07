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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit calcium interval recommender readiness before deployment-chain testing.")
    parser.add_argument("--features", type=Path, default=Path("data/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--replay", type=Path, default=Path("data/ca_interval_recommender_replay.parquet"))
    parser.add_argument("--oracle", type=Path, default=Path("data/ca_interval_recommender_validation_oracle.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("data/ca_interval_recommender_metrics.csv"))
    parser.add_argument("--recommender-report", type=Path, default=Path("data/ca_interval_recommender_report.json"))
    parser.add_argument("--artifact", type=Path, default=Path("models/ca_interval_recommender/rule_artifact.json"))
    parser.add_argument("--rules", type=Path, default=Path("data/ca_regime_calcium_band_rules_ir_lag.csv"))
    parser.add_argument("--manual-review-candidates", type=Path, default=Path("data/ca_regime_calcium_band_manual_review_candidates.csv"))
    parser.add_argument("--risk-audit-output", type=Path, default=Path("data/ca_interval_recommender_risk_audit.csv"))
    parser.add_argument("--rule-audit-output", type=Path, default=Path("data/ca_interval_recommender_rule_audit.csv"))
    parser.add_argument("--manual-review-output", type=Path, default=Path("data/ca_interval_recommender_manual_review_sheet.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/ca_interval_recommender_readiness_report.json"))
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


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
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


def load_replay(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Replay parquet does not exist: {path}")
    replay = pd.read_parquet(path)
    if "time" in replay.columns:
        replay["time"] = pd.to_datetime(replay["time"], errors="coerce")
    replay = ensure_targets(replay)
    replay = replay[replay["t90"].notna()].sort_values("time").reset_index(drop=True)
    return replay


def bool_rate(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return math.nan
    return float(valid.astype(bool).mean())


def interval_position(row: pd.Series) -> str:
    if row.get("recommendation_status") != "recommended":
        return "not_recommended"
    current = pd.to_numeric(pd.Series([row.get("current_ca_consumption")]), errors="coerce").iloc[0]
    rec_min = pd.to_numeric(pd.Series([row.get("recommended_ca_consumption_min")]), errors="coerce").iloc[0]
    rec_max = pd.to_numeric(pd.Series([row.get("recommended_ca_consumption_max")]), errors="coerce").iloc[0]
    if not np.isfinite(current):
        return "missing_current_dose"
    if not np.isfinite(rec_min) or not np.isfinite(rec_max):
        return "missing_recommended_band"
    if current < rec_min:
        return "below_band"
    if current > rec_max:
        return "above_band"
    return "inside_band"


def subgroup_metrics(subset: pd.DataFrame) -> dict[str, float]:
    if subset.empty:
        return {
            "sample_count": 0,
            "ok_rate": math.nan,
            "high_rate": math.nan,
            "low_rate": math.nan,
            "out_spec_rate": math.nan,
            "mean_t90": math.nan,
            "band_accuracy": math.nan,
            "direction_accuracy": math.nan,
            "target_accuracy_5pct": math.nan,
        }
    recommended = subset[subset["recommendation_status"].eq("recommended")]
    return {
        "sample_count": int(len(subset)),
        "ok_rate": float(subset["y_ok"].mean()),
        "high_rate": float(subset["y_high"].mean()),
        "low_rate": float(subset["y_low"].mean()),
        "out_spec_rate": float(subset["y_out_spec"].mean()),
        "mean_t90": float(subset["t90"].mean()),
        "band_accuracy": bool_rate(recommended["band_hit"]) if len(recommended) else math.nan,
        "direction_accuracy": bool_rate(recommended["direction_hit"]) if len(recommended) else math.nan,
        "target_accuracy_5pct": bool_rate(recommended["target_hit_5pct"]) if len(recommended) else math.nan,
    }


def make_metric_rows(audit_level: str, group_key: str, metrics: dict[str, float], note: str = "") -> list[dict[str, object]]:
    rows = []
    for metric, value in metrics.items():
        rows.append(
            {
                "audit_level": audit_level,
                "group_key": group_key,
                "metric": metric,
                "value": value,
                "note": note,
            }
        )
    return rows


def build_risk_audit(replay: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], dict[str, object], dict[str, object]]:
    work = replay.copy()
    work["interval_position"] = work.apply(interval_position, axis=1)
    rows: list[dict[str, object]] = []
    for name, subset in {
        "all": work,
        "train_like": work[work["split"].eq("train_like")],
        "test_like": work[work["split"].eq("test_like")],
    }.items():
        rows.extend(make_metric_rows("split", name, subgroup_metrics(subset)))
    for group in ["confidence_level", "action_hint", "interval_position", "recommendation_status"]:
        for value, subset in work.groupby(group, dropna=False, sort=True):
            rows.extend(make_metric_rows(group, str(value), subgroup_metrics(subset)))
    for group in ["confidence_level", "action_hint", "interval_position"]:
        for split, split_df in work.groupby("split", sort=True):
            for value, subset in split_df.groupby(group, dropna=False, sort=True):
                rows.extend(make_metric_rows(f"split_{group}", f"{split}:{value}", subgroup_metrics(subset)))

    test = work[work["split"].eq("test_like")]
    recommended_test = test[test["recommendation_status"].eq("recommended")]
    inside = recommended_test[recommended_test["interval_position"].eq("inside_band")]
    outside = recommended_test[recommended_test["interval_position"].isin(["below_band", "above_band"])]
    below = recommended_test[recommended_test["interval_position"].eq("below_band")]
    above = recommended_test[recommended_test["interval_position"].eq("above_band")]
    inside_metrics = subgroup_metrics(inside)
    outside_metrics = subgroup_metrics(outside)
    below_metrics = subgroup_metrics(below)
    above_metrics = subgroup_metrics(above)
    comparisons = {
        "inside_vs_outside": compare_risk(inside_metrics, outside_metrics),
        "inside_vs_below": compare_risk(inside_metrics, below_metrics),
        "inside_vs_above": compare_risk(inside_metrics, above_metrics),
    }
    support_pass = inside_metrics["sample_count"] >= 30 and outside_metrics["sample_count"] >= 30
    guardrail_pass = (
        support_pass
        and inside_metrics["high_rate"] <= outside_metrics["high_rate"]
        and inside_metrics["low_rate"] <= outside_metrics["low_rate"] + 0.005
        and inside_metrics["out_spec_rate"] <= outside_metrics["out_spec_rate"]
    )
    inside_outside_summary = {
        "inside_band_test_like": inside_metrics,
        "outside_band_test_like": outside_metrics,
        "below_band_test_like": below_metrics,
        "above_band_test_like": above_metrics,
        "comparisons": comparisons,
        "support_pass": bool(support_pass),
        "risk_guardrail_pass": bool(guardrail_pass),
    }
    action_summary = {
        action: subgroup_metrics(subset)
        for action, subset in test.groupby("action_hint", dropna=False, sort=True)
    }
    hold = action_summary.get("hold_in_band", {})
    inc = action_summary.get("increase_to_band", {})
    dec = action_summary.get("decrease_to_band", {})
    action_flags = {
        "unsafe_increase_hint": bool(
            inc
            and hold
            and inc.get("sample_count", 0) >= 10
            and hold.get("sample_count", 0) >= 10
            and (
                inc.get("low_rate", 0) > hold.get("low_rate", 0)
                or inc.get("high_rate", 0) > hold.get("high_rate", 0)
            )
        ),
        "unsafe_decrease_hint": bool(
            dec
            and hold
            and dec.get("sample_count", 0) >= 10
            and hold.get("sample_count", 0) >= 10
            and dec.get("low_rate", 0) > hold.get("low_rate", 0)
        ),
        "safe_hold_band_candidate": bool(
            hold
            and hold.get("sample_count", 0) >= 30
            and hold.get("high_rate", 1) <= test["y_high"].mean()
            and hold.get("low_rate", 1) <= test["y_low"].mean() + 0.005
        ),
    }
    action_type_summary = {"metrics": action_summary, "flags": action_flags}
    return pd.DataFrame(rows), inside_outside_summary, action_type_summary, {"interval_position_counts": work["interval_position"].value_counts().to_dict()}


def compare_risk(left: dict[str, float], right: dict[str, float]) -> dict[str, object]:
    return {
        "left_sample_count": left.get("sample_count", 0),
        "right_sample_count": right.get("sample_count", 0),
        "ok_rate_delta": safe_delta(left.get("ok_rate"), right.get("ok_rate")),
        "high_rate_delta": safe_delta(left.get("high_rate"), right.get("high_rate")),
        "low_rate_delta": safe_delta(left.get("low_rate"), right.get("low_rate")),
        "out_spec_rate_delta": safe_delta(left.get("out_spec_rate"), right.get("out_spec_rate")),
    }


def safe_delta(a: object, b: object) -> float:
    try:
        if np.isfinite(a) and np.isfinite(b):
            return float(a - b)
    except TypeError:
        pass
    return math.nan


def explode_selected_rules(replay: pd.DataFrame) -> pd.DataFrame:
    work = replay[replay["recommendation_status"].eq("recommended")].copy()
    work["selected_rule_id"] = work["selected_rule_ids"].fillna("").astype(str).str.split(";")
    work = work.explode("selected_rule_id")
    work = work[work["selected_rule_id"].astype(str).str.len() > 0].copy()
    return work


def rule_accuracy_metrics(subset: pd.DataFrame) -> dict[str, float]:
    metrics = subgroup_metrics(subset)
    metrics["recommendation_count"] = int(len(subset))
    metrics["inside_band_count"] = int((subset["interval_position"] == "inside_band").sum())
    metrics["outside_band_count"] = int(subset["interval_position"].isin(["below_band", "above_band"]).sum())
    inside = subset[subset["interval_position"].eq("inside_band")]
    outside = subset[subset["interval_position"].isin(["below_band", "above_band"])]
    for prefix, group in [("inside_band", inside), ("outside_band", outside)]:
        sub = subgroup_metrics(group)
        metrics[f"{prefix}_ok_rate"] = sub["ok_rate"]
        metrics[f"{prefix}_high_rate"] = sub["high_rate"]
        metrics[f"{prefix}_low_rate"] = sub["low_rate"]
        metrics[f"{prefix}_out_spec_rate"] = sub["out_spec_rate"]
    return metrics


def rule_decision(metrics: dict[str, float]) -> str:
    band = metrics.get("band_accuracy", math.nan)
    direction = metrics.get("direction_accuracy", math.nan)
    test_n = metrics.get("sample_count", 0)
    inside_n = metrics.get("inside_band_count", 0)
    outside_n = metrics.get("outside_band_count", 0)
    inside_high = metrics.get("inside_band_high_rate", math.nan)
    outside_high = metrics.get("outside_band_high_rate", math.nan)
    inside_low = metrics.get("inside_band_low_rate", math.nan)
    outside_low = metrics.get("outside_band_low_rate", math.nan)
    high_worse = outside_n > 0 and np.isfinite(inside_high) and np.isfinite(outside_high) and inside_high > outside_high + 0.02
    low_worse = outside_n > 0 and np.isfinite(inside_low) and np.isfinite(outside_low) and inside_low > outside_low + 0.005
    if (np.isfinite(band) and np.isfinite(direction) and band < 0.70 and direction < 0.70) or high_worse or low_worse:
        return "reject_or_refine"
    accuracy_pass = (np.isfinite(band) and band >= 0.70) and (np.isfinite(direction) and direction >= 0.70)
    if accuracy_pass and test_n >= 10 and inside_n >= 5:
        if outside_n == 0 or (not high_worse and not low_worse):
            if outside_n >= 5:
                return "monitor_chain_candidate"
            return "manual_review_only"
    if accuracy_pass:
        return "manual_review_only"
    return "reject_or_refine"


def build_rule_audit(replay: pd.DataFrame, rules: pd.DataFrame, manual_candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    work = replay.copy()
    work["interval_position"] = work.apply(interval_position, axis=1)
    exploded = explode_selected_rules(work)
    test = exploded[exploded["split"].eq("test_like")]
    rule_map = {str(row["rule_id"]): row for _, row in rules.iterrows()}
    manual_map = {str(row["rule_id"]): row for _, row in manual_candidates.iterrows()} if not manual_candidates.empty and "rule_id" in manual_candidates.columns else {}
    rows = []
    for rule_id, subset in test.groupby("selected_rule_id", sort=True):
        metrics = rule_accuracy_metrics(subset)
        rule = rule_map.get(str(rule_id), pd.Series(dtype=object))
        decision = rule_decision(metrics)
        confidence_distribution = subset["confidence_level"].value_counts(dropna=False).to_dict()
        row = {
            "rule_id": str(rule_id),
            "regime_feature": rule.get("regime_feature", ""),
            "regime_bin": rule.get("regime_bin", ""),
            "recommended_ca_consumption_min": rule.get("recommended_dose_min", math.nan),
            "recommended_ca_consumption_max": rule.get("recommended_dose_max", math.nan),
            "rule_grade": rule.get("rule_grade", ""),
            "rule_status": rule.get("rule_status", ""),
            "test_like_sample_count": metrics["sample_count"],
            "recommendation_count": metrics["recommendation_count"],
            "inside_band_count": metrics["inside_band_count"],
            "outside_band_count": metrics["outside_band_count"],
            "band_accuracy": metrics["band_accuracy"],
            "direction_accuracy": metrics["direction_accuracy"],
            "target_accuracy_5pct": metrics["target_accuracy_5pct"],
            "ok_rate": metrics["ok_rate"],
            "high_rate": metrics["high_rate"],
            "low_rate": metrics["low_rate"],
            "out_spec_rate": metrics["out_spec_rate"],
            "inside_band_ok_rate": metrics["inside_band_ok_rate"],
            "inside_band_high_rate": metrics["inside_band_high_rate"],
            "inside_band_low_rate": metrics["inside_band_low_rate"],
            "inside_band_out_spec_rate": metrics["inside_band_out_spec_rate"],
            "outside_band_ok_rate": metrics["outside_band_ok_rate"],
            "outside_band_high_rate": metrics["outside_band_high_rate"],
            "outside_band_low_rate": metrics["outside_band_low_rate"],
            "outside_band_out_spec_rate": metrics["outside_band_out_spec_rate"],
            "high_dose_avoidance_candidate": bool(rule.get("high_dose_avoidance_candidate", False)),
            "confidence_distribution": json.dumps(as_jsonable(confidence_distribution), ensure_ascii=False),
            "recommended_decision": decision,
        }
        rows.append(row)
    audit = pd.DataFrame(rows)
    if audit.empty:
        manual = pd.DataFrame()
        summary = {"monitor_chain_candidate_count": 0, "manual_review_only_count": 0, "reject_or_refine_count": 0}
        return audit, manual, summary
    manual_rows = []
    for idx, row in audit.iterrows():
        rule_id = str(row["rule_id"])
        manual_source = manual_map.get(rule_id, pd.Series(dtype=object))
        decision = row["recommended_decision"]
        manual_rows.append(
            {
                "review_id": f"review_{idx + 1:03d}",
                "rule_id": rule_id,
                "regime_feature": row["regime_feature"],
                "regime_bin": row["regime_bin"],
                "recommended_ca_consumption_min": row["recommended_ca_consumption_min"],
                "recommended_ca_consumption_max": row["recommended_ca_consumption_max"],
                "rule_grade": row["rule_grade"],
                "rule_status": row["rule_status"],
                "test_like_sample_count": row["test_like_sample_count"],
                "band_accuracy": row["band_accuracy"],
                "direction_accuracy": row["direction_accuracy"],
                "target_accuracy_5pct": row["target_accuracy_5pct"],
                "inside_band_sample_count": row["inside_band_count"],
                "inside_band_ok_rate": row["inside_band_ok_rate"],
                "inside_band_high_rate": row["inside_band_high_rate"],
                "inside_band_low_rate": row["inside_band_low_rate"],
                "outside_band_sample_count": row["outside_band_count"],
                "outside_band_ok_rate": row["outside_band_ok_rate"],
                "outside_band_high_rate": row["outside_band_high_rate"],
                "outside_band_low_rate": row["outside_band_low_rate"],
                "high_dose_avoidance_candidate": row["high_dose_avoidance_candidate"],
                "monitor_chain_candidate": decision == "monitor_chain_candidate",
                "manual_review_only": decision == "manual_review_only",
                "reject_or_refine": decision == "reject_or_refine",
                "engineering_review_question": manual_source.get("engineering_review_question", engineering_question(row)),
                "suggested_human_decision_options": suggested_options(decision),
            }
        )
    manual = pd.DataFrame(manual_rows)
    summary = {
        "monitor_chain_candidate_count": int((audit["recommended_decision"] == "monitor_chain_candidate").sum()),
        "manual_review_only_count": int((audit["recommended_decision"] == "manual_review_only").sum()),
        "reject_or_refine_count": int((audit["recommended_decision"] == "reject_or_refine").sum()),
        "rule_count": int(len(audit)),
    }
    return audit, manual, summary


def engineering_question(row: pd.Series) -> str:
    if row.get("recommended_decision") == "monitor_chain_candidate":
        return "该规则在验证集准确率和区间内风险上是否符合工艺机理，可否进入仅监测链路？"
    if row.get("outside_band_count", 0) < 5:
        return "该规则验证集区间外样本不足，是否需要更多样本后再判断？"
    if row.get("recommended_decision") == "reject_or_refine":
        return "该规则准确率或风险对比不达标，是否应拒绝或重新定义分层？"
    return "该规则可保留人工复核，但暂不进入监测链路。"


def suggested_options(decision: str) -> str:
    options = ["accept_for_monitor_chain", "monitor_only_no_action_hint", "keep_rule_but_hide_action_hint", "reject_rule", "needs_more_samples"]
    if decision == "monitor_chain_candidate":
        return ";".join(options)
    if decision == "manual_review_only":
        return ";".join(["monitor_only_no_action_hint", "keep_rule_but_hide_action_hint", "needs_more_samples", "reject_rule"])
    return ";".join(["reject_rule", "needs_more_samples"])


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


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
    title = section_title(doc_path, 21, "钙单耗区间推荐器部署前风险审计与人工复核表")
    lines = [
        "",
        title,
        "",
        "本阶段在区间推荐器 MVP 之后执行部署前审计。由于 Stage 20 推荐覆盖率接近 100%，test_like 中 no_recommendation 样本过少，因此不再把 no_recommendation 作为主要风险基线，而改用实际钙单耗在推荐区间内 vs 区间外的风险对比。",
        "",
        "### 审计结果",
        f"- no_recommendation baseline：{report['baseline_validity']}。",
        f"- inside/outside 风险摘要：{report['inside_outside_risk_summary']}。",
        f"- 动作类型摘要：{report['action_type_summary']}。",
        f"- monitor_chain_candidate_count：{report['monitor_chain_candidate_count']}。",
        f"- manual_review_only_count：{report['manual_review_only_count']}。",
        f"- reject_or_refine_count：{report['reject_or_refine_count']}。",
        f"- risk_guardrail_status：{report['risk_guardrail_status']}。",
        f"- readiness_status：`{report['readiness_status']}`。",
        f"- recommended_next_step：`{report['recommended_next_step']}`。",
        "",
        "### 局限",
        "- 仍为离线代理验证，不构成因果证明。",
        "- 不训练模型，不执行自动控制，不写入 DCS，不推荐影子试验。",
        "- 所有规则仍需工程人工复核。",
        "",
    ]
    with doc_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def decide_readiness(
    stage20: dict[str, object],
    inside_outside: dict[str, object],
    action_summary: dict[str, object],
    rule_summary: dict[str, object],
) -> tuple[str, str]:
    band = float(stage20.get("test_like_band_accuracy") or math.nan)
    direction = float(stage20.get("test_like_direction_accuracy") or math.nan)
    accuracy_pass = np.isfinite(band) and np.isfinite(direction) and band >= 0.70 and direction >= 0.70
    risk_pass = bool(inside_outside.get("risk_guardrail_pass"))
    support_pass = bool(inside_outside.get("support_pass"))
    monitor_count = int(rule_summary.get("monitor_chain_candidate_count", 0))
    reject_count = int(rule_summary.get("reject_or_refine_count", 0))
    rule_count = int(rule_summary.get("rule_count", 0))
    many_fail = rule_count > 0 and reject_count / rule_count > 0.5
    if accuracy_pass and risk_pass and support_pass and monitor_count >= 5:
        return "ready_for_monitor_chain_after_manual_review", "prepare_monitor_chain_interface"
    if accuracy_pass and (not risk_pass or not support_pass) and not many_fail:
        return "manual_review_required_before_monitor_chain", "human_review_rule_sheet"
    if accuracy_pass and (many_fail or action_summary["flags"].get("unsafe_increase_hint") or action_summary["flags"].get("unsafe_decrease_hint")):
        return "refine_recommender_before_monitor_chain", "refine_interval_recommender_rules"
    return "stop_until_more_data", "collect_more_data"


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    assumptions = [
        "Recommendation accuracy is judged against validation oracle bands/directions, not T90 qualification rate.",
        "No-recommendation rows are not used as primary risk baseline when support is too small.",
        "No models are trained and the recommender artifact is not modified.",
        "No automatic control, DCS writeback, or shadow-trial recommendation is made.",
    ]
    replay = load_replay(args.replay)
    recommender_report = load_json(args.recommender_report)
    artifact = load_json(args.artifact)
    rules = pd.read_csv(args.rules) if args.rules.exists() else pd.DataFrame()
    manual_candidates = pd.read_csv(args.manual_review_candidates) if args.manual_review_candidates.exists() else pd.DataFrame()
    test = replay[replay["split"].eq("test_like")]
    no_rec_test_count = int((test["recommendation_status"] != "recommended").sum())
    baseline_validity = {
        "recommendation_status_counts": replay["recommendation_status"].value_counts(dropna=False).to_dict(),
        "test_like_no_recommendation_count": no_rec_test_count,
        "no_recommendation_baseline_unreliable": bool(no_rec_test_count < 30),
        "note": "Do not use no_recommendation as main risk baseline when count < 30.",
    }
    risk_audit, inside_outside_summary, action_type_summary, interval_counts = build_risk_audit(replay)
    rule_audit, manual_review, rule_summary = build_rule_audit(replay, rules, manual_candidates)
    write_csv(args.risk_audit_output, risk_audit)
    write_csv(args.rule_audit_output, rule_audit)
    write_csv(args.manual_review_output, manual_review)

    readiness_status, next_step = decide_readiness(
        recommender_report,
        inside_outside_summary,
        action_type_summary,
        rule_summary,
    )
    risk_guardrail_status = {
        "inside_vs_outside_support_pass": inside_outside_summary["support_pass"],
        "inside_vs_outside_guardrail_pass": inside_outside_summary["risk_guardrail_pass"],
        "action_flags": action_type_summary["flags"],
    }
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "replay_path": str(args.replay),
        "recommender_report_path": str(args.recommender_report),
        "artifact_path": str(args.artifact),
        "risk_audit_output_path": str(args.risk_audit_output),
        "rule_audit_output_path": str(args.rule_audit_output),
        "manual_review_output_path": str(args.manual_review_output),
        "stage20_summary": {
            "artifact_rule_count": recommender_report.get("artifact_rule_count"),
            "test_like_recommendation_coverage": recommender_report.get("test_like_recommendation_coverage"),
            "test_like_band_accuracy": recommender_report.get("test_like_band_accuracy"),
            "test_like_direction_accuracy": recommender_report.get("test_like_direction_accuracy"),
            "mvp_status": recommender_report.get("mvp_status"),
            "recommended_next_step": recommender_report.get("recommended_next_step"),
        },
        "baseline_validity": baseline_validity,
        "inside_outside_risk_summary": inside_outside_summary,
        "action_type_summary": action_type_summary,
        "rule_level_summary": rule_summary,
        "monitor_chain_candidate_count": int(rule_summary.get("monitor_chain_candidate_count", 0)),
        "manual_review_only_count": int(rule_summary.get("manual_review_only_count", 0)),
        "reject_or_refine_count": int(rule_summary.get("reject_or_refine_count", 0)),
        "risk_guardrail_status": risk_guardrail_status,
        "readiness_status": readiness_status,
        "warnings": warnings,
        "assumptions": assumptions,
        "recommended_next_step": next_step,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(report), handle, ensure_ascii=False, indent=2)
    append_docs(args.doc, report)

    print("Calcium interval recommender readiness audit summary")
    print(f"No-recommendation baseline validity: {baseline_validity}")
    print(f"Inside/outside risk summary: {inside_outside_summary}")
    print(f"Action type summary: {action_type_summary}")
    print(f"Monitor-chain candidates: {report['monitor_chain_candidate_count']}")
    print(f"Manual-review-only: {report['manual_review_only_count']}")
    print(f"Reject/refine: {report['reject_or_refine_count']}")
    print(f"Readiness status: {readiness_status}")
    print(f"Recommended next step: {next_step}")
    print(f"Documentation appended: {args.doc}")


if __name__ == "__main__":
    main()
