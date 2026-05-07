from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from discover_ca_t90_relationships import as_jsonable, ensure_targets, load_json, make_quantile_bins, section_title, write_csv


T90_LOW = 8.20
T90_HIGH = 8.70
PRIMARY_DOSE_FALLBACK = "ca_per_rubber_flow_win_60_mean"
IR_LAG_FEATURE = "output_ir_corrected_offset_20_win_15_std"
PRIORITY_CONTEXT_VARIABLES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Define interpretable regime-specific calcium band rules with IR-lag diagnostics.")
    parser.add_argument("--features", type=Path, default=Path("data/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--feature-report", type=Path, default=Path("data/t90_ca_feature_report.json"))
    parser.add_argument("--regime-dose-response", type=Path, default=Path("data/ca_regime_dose_response_ir_lag.csv"))
    parser.add_argument("--interaction-screen", type=Path, default=Path("data/ca_context_interaction_screen_ir_lag.csv"))
    parser.add_argument("--ir-strat-response", type=Path, default=Path("data/ca_ir_lag_stratified_dose_response.csv"))
    parser.add_argument("--mediation-diagnostic", type=Path, default=Path("data/ir_lag_mediation_diagnostic.csv"))
    parser.add_argument("--band-map", type=Path, default=Path("data/ca_regime_optimal_band_map_ir_lag.csv"))
    parser.add_argument("--relationship-report", type=Path, default=Path("data/ca_t90_relationship_discovery_ir_lag_report.json"))
    parser.add_argument("--data-with-ir", type=Path, default=Path("data/data_clean_with_ir.parquet"))
    parser.add_argument("--rules-output", type=Path, default=Path("data/ca_regime_calcium_band_rules_ir_lag.csv"))
    parser.add_argument("--validation-output", type=Path, default=Path("data/ca_regime_calcium_band_rule_validation_ir_lag.csv"))
    parser.add_argument("--manual-review-output", type=Path, default=Path("data/ca_regime_calcium_band_manual_review_candidates.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/ca_regime_calcium_band_rules_ir_lag_report.json"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    return parser.parse_args()


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    return pd.read_csv(path)


def rolling_std_feature(values: pd.Series, window: str = "15min") -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rolling(window, min_periods=2).std()


def derive_ir_lag_feature(samples: pd.DataFrame, data_with_ir_path: Path, warnings: list[str]) -> pd.DataFrame:
    if IR_LAG_FEATURE in samples.columns:
        return samples
    result = samples.copy()
    if not data_with_ir_path.exists():
        warnings.append(f"data-with-ir is missing: {data_with_ir_path}; continuing without IR-lag context.")
        result[IR_LAG_FEATURE] = np.nan
        return result
    columns = pd.read_parquet(data_with_ir_path, columns=None).columns.tolist()
    if IR_LAG_FEATURE in columns:
        ir = pd.read_parquet(data_with_ir_path, columns=["time", IR_LAG_FEATURE])
        ir["time"] = pd.to_datetime(ir["time"], errors="coerce")
        ir = ir.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last")
        return result.merge(ir, on="time", how="left")
    if "output_ir_corrected" not in columns:
        warnings.append("data-with-ir does not contain output_ir_corrected; continuing without IR-lag context.")
        result[IR_LAG_FEATURE] = np.nan
        return result
    ir = pd.read_parquet(data_with_ir_path, columns=["time", "output_ir_corrected"])
    ir["time"] = pd.to_datetime(ir["time"], errors="coerce")
    ir = ir.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last").sort_values("time")
    indexed = ir.set_index("time")
    indexed[IR_LAG_FEATURE] = rolling_std_feature(indexed["output_ir_corrected"], "15min")
    lookup = result[["time"]].copy()
    lookup["ir_lookup_time"] = lookup["time"] - pd.to_timedelta(20, unit="m")
    feature_timeline = indexed[[IR_LAG_FEATURE]].reset_index()
    merged_values = lookup.merge(feature_timeline, left_on="ir_lookup_time", right_on="time", how="left", suffixes=("", "_ir"))
    result[IR_LAG_FEATURE] = pd.to_numeric(merged_values[IR_LAG_FEATURE], errors="coerce")
    return result


def load_samples(args: argparse.Namespace, relationship_report: dict[str, object], warnings: list[str]) -> tuple[pd.DataFrame, str]:
    if not args.features.exists():
        raise FileNotFoundError(f"Feature parquet does not exist: {args.features}")
    data = pd.read_parquet(args.features)
    if "time" not in data.columns or "t90" not in data.columns:
        raise ValueError("Feature dataset must contain time and t90 columns.")
    data = data.copy()
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    if data["time"].isna().any():
        raise ValueError("Feature dataset contains invalid time values.")
    data = ensure_targets(data)
    data = data[data["t90"].notna()].sort_values("time").reset_index(drop=True)
    primary = relationship_report.get("primary_dose_feature")
    if not isinstance(primary, str) or primary not in data.columns:
        primary = PRIMARY_DOSE_FALLBACK
    if primary not in data.columns:
        raise ValueError(f"Primary dose feature is missing: {primary}")
    data = derive_ir_lag_feature(data, args.data_with_ir, warnings)
    return data, primary


def interaction_audit(interaction: pd.DataFrame, regime: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    if interaction.empty:
        return pd.DataFrame(), {"passed_count": 0, "stable_candidate_count": 0, "suspicious_large_delta_count": 0}
    stable_regime_features = set(
        regime.loc[
            (regime.get("support_level", "") == "enough_support")
            & (pd.to_numeric(regime.get("ok_rate", np.nan), errors="coerce") >= 0.90),
            "regime_feature",
        ]
    ) if not regime.empty else set()
    for _, row in interaction.iterrows():
        passed = bool(row.get("passed_interaction_screen", False))
        suspicious = bool(row.get("suspicious_large_delta", False)) or (
            pd.to_numeric(pd.Series([row.get("delta_auc")]), errors="coerce").iloc[0] > 0.25
            or pd.to_numeric(pd.Series([row.get("delta_ap")]), errors="coerce").iloc[0] > 0.25
        )
        pos_test = pd.to_numeric(pd.Series([row.get("positive_rate_test")]), errors="coerce").iloc[0]
        test_n = pd.to_numeric(pd.Series([row.get("test_sample_count")]), errors="coerce").iloc[0]
        positive_count_test = pos_test * test_n if np.isfinite(pos_test) and np.isfinite(test_n) else math.nan
        insufficient = bool(np.isfinite(positive_count_test) and positive_count_test < 10)
        supported_direction = row.get("context_feature") in stable_regime_features
        if passed and not insufficient and (not suspicious or supported_direction):
            status = "stable_candidate"
        elif suspicious:
            status = "suspicious_large_delta"
        elif insufficient:
            status = "insufficient_positive_support"
        else:
            status = "rejected"
        rows.append(
            {
                "context_feature": row.get("context_feature"),
                "target": row.get("target"),
                "delta_auc": row.get("delta_auc"),
                "delta_ap": row.get("delta_ap"),
                "positive_count_test": positive_count_test,
                "passed_interaction_screen": passed,
                "suspicious_large_delta": suspicious,
                "insufficient_positive_support": insufficient,
                "supported_by_regime_dose_evidence": supported_direction,
                "interaction_audit_status": status,
            }
        )
    audit = pd.DataFrame(rows)
    summary = {
        "passed_count": int(audit["passed_interaction_screen"].sum()),
        "stable_candidate_count": int((audit["interaction_audit_status"] == "stable_candidate").sum()),
        "suspicious_large_delta_count": int(audit["suspicious_large_delta"].sum()),
        "insufficient_positive_support_count": int(audit["insufficient_positive_support"].sum()),
        "rejected_count": int((audit["interaction_audit_status"] == "rejected").sum()),
    }
    return audit, summary


def make_regime_labels(data: pd.DataFrame, feature: str) -> pd.Series:
    ids = make_quantile_bins(data[feature], 3)
    return ids.map({0: "low", 1: "mid", 2: "high"}).astype("object")


def overall_rates(data: pd.DataFrame) -> dict[str, float]:
    return {
        "overall_ok_rate": float(data["y_ok"].mean()),
        "overall_low_rate": float(data["y_low"].mean()),
        "overall_high_rate": float(data["y_high"].mean()),
        "overall_out_spec_rate": float(data["y_out_spec"].mean()),
    }


def high_dose_risk_for_regime(regime_response: pd.DataFrame, feature: str, regime_bin: str, best_high: float, overall_high: float) -> tuple[bool, bool]:
    subset = regime_response[(regime_response["regime_feature"] == feature) & (regime_response["regime_bin"] == regime_bin)].copy()
    if subset.empty:
        return False, False
    subset["dose_bin_numeric"] = pd.to_numeric(subset["dose_bin"], errors="coerce")
    high_row = subset.sort_values("dose_bin_numeric").tail(1).iloc[0]
    enough = high_row.get("support_level") == "enough_support" and int(high_row.get("sample_count", 0)) >= 20
    high_risk = float(high_row.get("high_rate", 0.0)) >= overall_high + 0.05 and float(high_row.get("high_rate", 0.0)) >= best_high + 0.05
    return bool(high_risk), bool(high_risk and enough)


def ir_context_for_regime(data: pd.DataFrame, feature: str, regime_bin: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    if feature not in data.columns or IR_LAG_FEATURE not in data.columns:
        return {
            "ir_lag_context_available": False,
            "ir_lag_risk_context_useful": False,
            "ir_lag_low_support": True,
            "ir_lag_available_rate": math.nan,
            "ir_lag_mean": math.nan,
            "ir_lag_median": math.nan,
            "ir_lag_note": "IR-lag feature unavailable",
        }, rows
    labels = make_regime_labels(data, feature)
    subset = data[labels == regime_bin].copy()
    if subset.empty:
        return {
            "ir_lag_context_available": False,
            "ir_lag_risk_context_useful": False,
            "ir_lag_low_support": True,
            "ir_lag_available_rate": 0.0,
            "ir_lag_mean": math.nan,
            "ir_lag_median": math.nan,
            "ir_lag_note": "empty regime subset",
        }, rows
    available_rate = float(subset[IR_LAG_FEATURE].notna().mean())
    values = pd.to_numeric(subset[IR_LAG_FEATURE], errors="coerce")
    subset["ir_lag_tertile"] = make_quantile_bins(values, 3).map({0: "low", 1: "mid", 2: "high"}).astype("object")
    rates = []
    min_count = math.inf
    shares = {}
    for tertile, group in subset.dropna(subset=["ir_lag_tertile"]).groupby("ir_lag_tertile", sort=True):
        shares[str(tertile)] = float(len(group) / len(subset))
        min_count = min(min_count, len(group))
        rates.append((float(group["high_rate"].mean()) if "high_rate" in group.columns else float(group["y_high"].mean()), float(group["y_out_spec"].mean())))
        for target in ["y_ok", "y_high", "y_low", "y_out_spec"]:
            rows.append(
                validation_row(
                    "ir_lag_tertile_outcome",
                    "",
                    IR_LAG_FEATURE,
                    target,
                    f"{tertile}_rate",
                    float(group[target].mean()),
                    "diagnostic",
                    f"IR-lag tertile outcome inside {feature}={regime_bin}; n={len(group)}",
                )
            )
    if not rates:
        high_spread = math.nan
        out_spread = math.nan
        min_count = 0
    else:
        high_spread = max(item[0] for item in rates) - min(item[0] for item in rates)
        out_spread = max(item[1] for item in rates) - min(item[1] for item in rates)
    context_available = available_rate >= 0.30
    risk_useful = context_available and min_count >= 10 and (high_spread >= 0.05 or out_spread >= 0.05)
    note = f"availability={available_rate:.3f}; high_spread={high_spread if np.isfinite(high_spread) else None}; out_spread={out_spread if np.isfinite(out_spread) else None}; shares={shares}"
    return {
        "ir_lag_context_available": bool(context_available),
        "ir_lag_risk_context_useful": bool(risk_useful),
        "ir_lag_low_support": not bool(risk_useful),
        "ir_lag_available_rate": available_rate,
        "ir_lag_mean": float(values.mean()) if values.notna().any() else math.nan,
        "ir_lag_median": float(values.median()) if values.notna().any() else math.nan,
        "ir_lag_note": note,
    }, rows


def validation_row(validation_type: str, rule_id: str, feature: str, target: str, metric: str, value: object, status: str, note: str) -> dict[str, object]:
    return {
        "validation_type": validation_type,
        "rule_id": rule_id,
        "feature": feature,
        "target": target,
        "metric": metric,
        "value": value,
        "status": status,
        "note": note,
    }


def rule_grade_and_status(
    sample_count: int,
    ok_lift: float,
    high_delta: float,
    low_delta: float,
    suspicious_interaction: bool,
    risk_note: str,
) -> tuple[str, str, str]:
    reject_reason = None
    if sample_count < 50:
        reject_reason = "insufficient support"
    elif high_delta > 0.05:
        reject_reason = "high-T90 risk worsens"
    elif low_delta > 0.03:
        reject_reason = "low-T90 risk worsens"
    elif "high_t90_risk" in str(risk_note) or "low_t90_risk" in str(risk_note):
        reject_reason = "risk note indicates elevated risk"
    if reject_reason:
        return "Reject", "reject", reject_reason
    if sample_count >= 100 and ok_lift >= 0.05 and high_delta <= 0 and low_delta <= 0.01 and not suspicious_interaction:
        return "A", "accept_for_manual_case_review", "strong historical regime band"
    if sample_count >= 80 and ok_lift >= 0.03 and high_delta <= 0.02 and low_delta <= 0.02:
        return "B", "accept_for_manual_case_review", "moderate historical regime band"
    if sample_count >= 50 and ok_lift > 0 and high_delta <= 0.05 and low_delta <= 0.03:
        return "C", "monitor_only", "positive but weaker historical regime band"
    return "Reject", "reject", "unstable or contradictory evidence"


def split_validation_for_rule(data: pd.DataFrame, rule: dict[str, object], overall_split_rates: dict[str, dict[str, float]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    labels = make_regime_labels(data, str(rule["regime_feature"]))
    dose = pd.to_numeric(data[str(rule["primary_dose_feature"])], errors="coerce")
    mask = (
        (labels == rule["regime_bin"])
        & (dose >= float(rule["recommended_dose_min"]))
        & (dose <= float(rule["recommended_dose_max"]))
    )
    work = data[mask].copy()
    split_index = int(len(data) * 0.8)
    data_split = pd.Series(np.where(data.index < split_index, "train_like", "test_like"), index=data.index)
    rows = []
    split_metrics = {}
    for split in ["train_like", "test_like"]:
        subset = work[data_split.loc[work.index] == split]
        metrics = {
            "sample_count": int(len(subset)),
            "ok_rate": float(subset["y_ok"].mean()) if len(subset) else math.nan,
            "high_rate": float(subset["y_high"].mean()) if len(subset) else math.nan,
            "low_rate": float(subset["y_low"].mean()) if len(subset) else math.nan,
            "out_spec_rate": float(subset["y_out_spec"].mean()) if len(subset) else math.nan,
        }
        split_metrics[split] = metrics
        for metric, value in metrics.items():
            rows.append(validation_row("time_split_rule_outcome", str(rule["rule_id"]), str(rule["regime_feature"]), "rule_subset", f"{split}_{metric}", value, "diagnostic", "rule subset outcome by time split"))
    test = split_metrics["test_like"]
    overall_test = overall_split_rates["test_like"]
    low_support = test["sample_count"] < 10
    unstable = (
        not low_support
        and np.isfinite(test["ok_rate"])
        and test["ok_rate"] < overall_test["ok_rate"]
        and (
            (np.isfinite(test["high_rate"]) and test["high_rate"] > overall_test["high_rate"])
            or (np.isfinite(test["low_rate"]) and test["low_rate"] > overall_test["low_rate"])
        )
    )
    time_stable = (
        not low_support
        and not unstable
        and np.isfinite(test["ok_rate"])
        and test["ok_rate"] >= overall_test["ok_rate"]
        and test["high_rate"] <= overall_test["high_rate"] + 0.02
        and test["low_rate"] <= overall_test["low_rate"] + 0.02
    )
    return {
        "time_stable": bool(time_stable),
        "low_support_in_test": bool(low_support),
        "unstable": bool(unstable),
        "train_like": split_metrics["train_like"],
        "test_like": split_metrics["test_like"],
    }, rows


def overall_split_rates(data: pd.DataFrame) -> dict[str, dict[str, float]]:
    split_index = int(len(data) * 0.8)
    result = {}
    for name, subset in [("train_like", data.iloc[:split_index]), ("test_like", data.iloc[split_index:])]:
        result[name] = {
            "ok_rate": float(subset["y_ok"].mean()),
            "high_rate": float(subset["y_high"].mean()),
            "low_rate": float(subset["y_low"].mean()),
            "out_spec_rate": float(subset["y_out_spec"].mean()),
        }
    return result


def engineering_question(row: pd.Series) -> str:
    if bool(row.get("low_support_in_test")):
        return "该规则测试期样本不足，只能作为监测候选。"
    if bool(row.get("high_dose_avoidance_candidate")):
        return "该工况下最高钙单耗分箱高 T90 风险偏高，是否需要设为人工复核条件？"
    if str(row.get("rule_status")) == "accept_for_manual_case_review":
        return "该工况下中等钙单耗区间历史合格率更高，是否符合工艺机理？"
    return "该规则证据较弱，是否仅作为趋势监测而不进入规则库？"


def build_rules(
    data: pd.DataFrame,
    primary: str,
    band_map: pd.DataFrame,
    regime_response: pd.DataFrame,
    interaction_audit_df: pd.DataFrame,
    overall: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rules = []
    validation_rows = []
    split_rates = overall_split_rates(data)
    suspicious_features = set(
        interaction_audit_df.loc[interaction_audit_df["suspicious_large_delta"].astype(bool), "context_feature"]
    ) if not interaction_audit_df.empty else set()
    band_map = band_map.copy()
    priority_rank = {feature: idx for idx, feature in enumerate(PRIORITY_CONTEXT_VARIABLES)}
    band_map["priority_rank"] = band_map["regime_feature"].map(priority_rank).fillna(999).astype(int)
    band_map = band_map.sort_values(["priority_rank", "regime_feature", "regime_bin"]).reset_index(drop=True)
    for idx, row in band_map.iterrows():
        rule_id = f"ca_regime_rule_{idx + 1:03d}"
        feature = str(row["regime_feature"])
        regime_bin = str(row["regime_bin"])
        best_ok = float(row["best_ok_rate"])
        best_low = float(row["best_low_rate"])
        best_high = float(row["best_high_rate"])
        best_out = float(row["best_out_spec_rate"])
        sample_count = int(row["sample_count"])
        ok_lift = best_ok - overall["overall_ok_rate"]
        high_delta = best_high - overall["overall_high_rate"]
        low_delta = best_low - overall["overall_low_rate"]
        out_delta = best_out - overall["overall_out_spec_rate"]
        high_dose_risk, high_dose_avoid = high_dose_risk_for_regime(
            regime_response, feature, regime_bin, best_high, overall["overall_high_rate"]
        )
        suspicious_interaction = feature in suspicious_features
        grade, status, evidence_note = rule_grade_and_status(
            sample_count, ok_lift, high_delta, low_delta, suspicious_interaction, str(row.get("risk_note", ""))
        )
        ir_context, ir_rows = ir_context_for_regime(data, feature, regime_bin)
        base_rule = {
            "rule_id": rule_id,
            "regime_feature": feature,
            "regime_bin": regime_bin,
            "primary_dose_feature": primary,
            "recommended_dose_min": float(row["best_dose_min"]),
            "recommended_dose_max": float(row["best_dose_max"]),
            "sample_count": sample_count,
            "best_ok_rate": best_ok,
            "best_low_rate": best_low,
            "best_high_rate": best_high,
            "best_out_spec_rate": best_out,
            **overall,
            "ok_lift_vs_overall": ok_lift,
            "high_delta_vs_overall": high_delta,
            "low_delta_vs_overall": low_delta,
            "out_spec_delta_vs_overall": out_delta,
            "rule_grade": grade,
            "rule_status": status,
            "high_dose_high_t90_risk": high_dose_risk,
            "high_dose_avoidance_candidate": high_dose_avoid,
            "time_stable": False,
            "low_support_in_test": False,
            "ir_lag_context_available": ir_context["ir_lag_context_available"],
            "ir_lag_risk_context_useful": ir_context["ir_lag_risk_context_useful"],
            "ir_lag_low_support": ir_context["ir_lag_low_support"],
            "risk_note": row.get("risk_note", ""),
            "evidence_note": evidence_note,
            "suspicious_interaction": suspicious_interaction,
            "ir_lag_available_rate": ir_context["ir_lag_available_rate"],
            "ir_lag_mean": ir_context["ir_lag_mean"],
            "ir_lag_median": ir_context["ir_lag_median"],
            "ir_lag_note": ir_context["ir_lag_note"],
        }
        time_validation, split_rows = split_validation_for_rule(data, base_rule, split_rates)
        base_rule.update(
            {
                "time_stable": time_validation["time_stable"],
                "low_support_in_test": time_validation["low_support_in_test"],
                "unstable": time_validation["unstable"],
            }
        )
        if base_rule["unstable"] and base_rule["rule_status"] == "accept_for_manual_case_review":
            base_rule["rule_status"] = "monitor_only"
            base_rule["rule_grade"] = "C"
            base_rule["evidence_note"] += "; downgraded due to test-like instability"
        if base_rule["low_support_in_test"] and base_rule["rule_status"] == "accept_for_manual_case_review":
            base_rule["rule_status"] = "monitor_only"
            base_rule["evidence_note"] += "; downgraded due to low test support"
        rules.append(base_rule)
        for ir_row in ir_rows:
            ir_row["rule_id"] = rule_id
            validation_rows.append(ir_row)
        validation_rows.extend(split_rows)
        validation_rows.extend(
            [
                validation_row("rule_quality", rule_id, feature, "y_ok", "ok_lift_vs_overall", ok_lift, status, evidence_note),
                validation_row("rule_quality", rule_id, feature, "y_high", "high_delta_vs_overall", high_delta, status, evidence_note),
                validation_row("rule_quality", rule_id, feature, "y_low", "low_delta_vs_overall", low_delta, status, evidence_note),
                validation_row("high_dose_risk", rule_id, feature, "y_high", "high_dose_high_t90_risk", high_dose_risk, "diagnostic", "highest calcium dose bin risk check"),
                validation_row("ir_lag_context", rule_id, IR_LAG_FEATURE, "y_out_spec", "ir_lag_risk_context_useful", ir_context["ir_lag_risk_context_useful"], "diagnostic", ir_context["ir_lag_note"]),
            ]
        )
    rules_df = pd.DataFrame(rules)
    validation_df = pd.DataFrame(validation_rows)
    candidates = rules_df[rules_df["rule_status"].isin(["accept_for_manual_case_review", "monitor_only"])].copy()
    if not candidates.empty:
        candidates["engineering_review_question"] = candidates.apply(engineering_question, axis=1)
    return rules_df, validation_df, candidates


def append_docs(doc_path: Path, report: dict[str, object]) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    title = section_title(doc_path, 19, "分工况钙单耗区间规则定义与 IR-lag 辅助验证")
    top_rules = report.get("top_rules", [])
    top_text = "无"
    if top_rules:
        top_text = "；".join(
            f"{item['rule_id']} {item['regime_feature']}={item['regime_bin']} dose=[{item['recommended_dose_min']}, {item['recommended_dose_max']}] grade={item['rule_grade']}"
            for item in top_rules[:5]
        )
    lines = [
        "",
        title,
        "",
        "本阶段承接 Stage 18 的关系发现结果，将稳定的分工况钙单耗区间转化为可解释、机器可读的规则候选表。该输出仅用于后续人工工程复核，不构成自动控制，不推荐影子试验。",
        "",
        "### 规则输入与边界",
        f"- 主钙单耗特征：`{report['primary_dose_feature']}`。",
        f"- 优先工况变量：{', '.join(report['priority_context_variables'])}。",
        f"- IR-lag 特征：`{report['ir_lag_feature']}`，仅作为辅助上下文/诊断元数据，不作为规则主驱动。",
        "",
        "### 审计结果",
        f"- 交互稳定审计：{report['interaction_audit_summary']}。",
        f"- suspicious large-delta 交互数：{report['interaction_audit_summary']['suspicious_large_delta_count']}。",
        f"- 规则等级统计：{report['rule_counts_by_grade']}。",
        f"- 规则状态统计：{report['rule_counts_by_status']}。",
        f"- accepted / monitor / rejected：{report['accepted_rule_count']} / {report['monitor_only_rule_count']} / {report['rejected_rule_count']}。",
        f"- 高剂量高 T90 避免候选数：{report['high_dose_avoidance_candidate_count']}。",
        f"- 时间稳定规则数：{report['time_stable_rule_count']}。",
        f"- IR-lag 有用上下文规则数：{report['ir_lag_context_useful_rule_count']}。",
        f"- 人工复核候选数：{report['manual_review_candidate_count']}。",
        f"- Top 规则：{top_text}。",
        f"- recommended_next_step：`{report['recommended_next_step']}`。",
        "",
        "### 局限",
        "- 仍为观察性历史数据，不构成因果证明。",
        "- LIMS 标签稀疏，部分规则测试期支持不足。",
        "- IR-lag 覆盖率有限，只能作为辅助风险背景。",
        "- 所有规则必须经过人工工程复核后，才能考虑后续更严格的离线验证。",
        "",
    ]
    with doc_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    assumptions = [
        "Rules are interpretable historical relationship summaries for manual engineering review only.",
        "IR-lag is auxiliary context/diagnostic metadata and is not the sole rule driver.",
        "No automatic control, setpoint recommendation, shadow trial, or policy grid search is performed.",
        "Calcium dose values are not imputed for rule validation.",
    ]
    relationship_report = load_json(args.relationship_report)
    _feature_report = load_json(args.feature_report)
    regime_response = read_csv_required(args.regime_dose_response)
    interaction = read_csv_required(args.interaction_screen)
    _ir_strat = read_csv_required(args.ir_strat_response)
    _mediation = read_csv_required(args.mediation_diagnostic)
    band_map = read_csv_required(args.band_map)
    samples, primary = load_samples(args, relationship_report, warnings)
    overall = overall_rates(samples)
    interaction_audit_df, interaction_summary = interaction_audit(interaction, regime_response)
    rules_df, validation_df, manual_df = build_rules(
        samples,
        primary,
        band_map,
        regime_response,
        interaction_audit_df,
        overall,
    )

    required_rule_columns = [
        "rule_id",
        "regime_feature",
        "regime_bin",
        "primary_dose_feature",
        "recommended_dose_min",
        "recommended_dose_max",
        "sample_count",
        "best_ok_rate",
        "best_low_rate",
        "best_high_rate",
        "best_out_spec_rate",
        "overall_ok_rate",
        "overall_low_rate",
        "overall_high_rate",
        "overall_out_spec_rate",
        "ok_lift_vs_overall",
        "high_delta_vs_overall",
        "low_delta_vs_overall",
        "out_spec_delta_vs_overall",
        "rule_grade",
        "rule_status",
        "high_dose_high_t90_risk",
        "high_dose_avoidance_candidate",
        "time_stable",
        "low_support_in_test",
        "ir_lag_context_available",
        "ir_lag_risk_context_useful",
        "ir_lag_low_support",
        "risk_note",
        "evidence_note",
    ]
    extra_rule_columns = [column for column in rules_df.columns if column not in required_rule_columns]
    rules_df = rules_df[required_rule_columns + extra_rule_columns]

    manual_columns = [
        "rule_id",
        "rule_grade",
        "rule_status",
        "regime_feature",
        "regime_bin",
        "primary_dose_feature",
        "recommended_dose_min",
        "recommended_dose_max",
        "sample_count",
        "best_ok_rate",
        "best_high_rate",
        "best_low_rate",
        "ok_lift_vs_overall",
        "high_delta_vs_overall",
        "low_delta_vs_overall",
        "high_dose_high_t90_risk",
        "high_dose_avoidance_candidate",
        "time_stable",
        "low_support_in_test",
        "ir_lag_context_available",
        "ir_lag_risk_context_useful",
        "ir_lag_note",
        "engineering_review_question",
    ]
    if not manual_df.empty:
        manual_df = manual_df[[column for column in manual_columns if column in manual_df.columns]]
    write_csv(args.rules_output, rules_df)
    write_csv(args.validation_output, validation_df)
    write_csv(args.manual_review_output, manual_df)

    rule_counts_by_grade = rules_df["rule_grade"].value_counts(dropna=False).to_dict()
    rule_counts_by_status = rules_df["rule_status"].value_counts(dropna=False).to_dict()
    accepted = int((rules_df["rule_status"] == "accept_for_manual_case_review").sum())
    monitor = int((rules_df["rule_status"] == "monitor_only").sum())
    rejected = int((rules_df["rule_status"] == "reject").sum())
    high_dose_avoid = int(rules_df["high_dose_avoidance_candidate"].sum())
    time_stable = int(rules_df["time_stable"].sum())
    ir_useful = int(rules_df["ir_lag_risk_context_useful"].sum())
    if accepted >= 5 and time_stable >= 3:
        next_step = "prepare_regime_rule_manual_review"
    elif accepted > 0:
        next_step = "audit_regime_rule_cases"
    elif interaction_summary["suspicious_large_delta_count"] >= max(5, interaction_summary["stable_candidate_count"]):
        next_step = "refine_relationship_discovery"
    else:
        next_step = "collect_more_data_or_new_features"

    top_rules = (
        rules_df[rules_df["rule_status"] == "accept_for_manual_case_review"]
        .sort_values(["rule_grade", "ok_lift_vs_overall", "sample_count"], ascending=[True, False, False])
        .head(10)
        .to_dict(orient="records")
    )
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features_path": str(args.features),
        "feature_report_path": str(args.feature_report),
        "relationship_report_path": str(args.relationship_report),
        "regime_dose_response_path": str(args.regime_dose_response),
        "interaction_screen_path": str(args.interaction_screen),
        "band_map_path": str(args.band_map),
        "data_with_ir_path": str(args.data_with_ir),
        "rules_output_path": str(args.rules_output),
        "validation_output_path": str(args.validation_output),
        "manual_review_output_path": str(args.manual_review_output),
        "primary_dose_feature": primary,
        "ir_lag_feature": IR_LAG_FEATURE,
        "ir_policy": {
            "role": "auxiliary_context_and_diagnostic_metadata_only",
            "not_rule_driver": True,
            "not_action_trigger": True,
            "not_direct_t90_measurement": True,
        },
        "priority_context_variables": PRIORITY_CONTEXT_VARIABLES,
        "interaction_audit_summary": interaction_summary,
        "rule_counts_by_grade": rule_counts_by_grade,
        "rule_counts_by_status": rule_counts_by_status,
        "accepted_rule_count": accepted,
        "monitor_only_rule_count": monitor,
        "rejected_rule_count": rejected,
        "high_dose_avoidance_candidate_count": high_dose_avoid,
        "time_stable_rule_count": time_stable,
        "ir_lag_context_useful_rule_count": ir_useful,
        "manual_review_candidate_count": int(len(manual_df)),
        "top_rules": top_rules,
        "warnings": warnings,
        "assumptions": assumptions,
        "recommended_next_step": next_step,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(report), handle, ensure_ascii=False, indent=2)
    append_docs(args.doc, report)

    print("Regime calcium band rule summary")
    print(f"Primary dose feature: {primary}")
    print(f"IR-lag feature: {IR_LAG_FEATURE}")
    print(f"Accepted rules: {accepted}")
    print(f"Monitor-only rules: {monitor}")
    print(f"Rejected rules: {rejected}")
    print(f"High-dose avoidance candidates: {high_dose_avoid}")
    print(f"Time-stable rules: {time_stable}")
    print(f"IR-lag useful context rules: {ir_useful}")
    print(f"Suspicious interactions: {interaction_summary['suspicious_large_delta_count']}")
    print(f"Manual review candidates: {len(manual_df)}")
    print(f"Recommended next step: {next_step}")
    print(f"Documentation appended: {args.doc}")


if __name__ == "__main__":
    main()
