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


T90_LOW = 8.20
T90_HIGH = 8.70
FINAL_STRATEGY = "median_aggregation_baseline"
PRODUCT_POSITIONING = "stable_safe_band_mvp"
CONTROL_MODE = "monitor_only"
ARTIFACT_VERSION = "1.0.0"
PRIMARY_DOSE_FEATURE = "ca_per_rubber_flow_win_60_mean"

ACTION_VISIBILITY_POLICY = {
    "inside_band": {
        "final_action_hint": "hold_in_band",
        "action_visibility": "monitor_only",
        "explanation_cn": "当前钙单耗处于推荐安全区间内，建议维持观察。",
    },
    "above_band": {
        "final_action_hint": "above_band_manual_review",
        "action_visibility": "manual_review_required",
        "explanation_cn": "当前钙单耗高于推荐安全区间，历史数据中高 T90 风险偏高，建议人工复核是否需要小幅降钙。",
    },
    "below_band": {
        "final_action_hint": "below_band_diagnostic_only",
        "action_visibility": "diagnostic_only",
        "explanation_cn": "当前钙单耗低于推荐安全区间，仅作诊断展示；当前 MVP 不给出加钙操作建议。",
    },
    "missing": {
        "final_action_hint": "no_recommendation_missing_input",
        "action_visibility": "no_recommendation",
        "explanation_cn": "关键输入缺失，无法生成推荐区间。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize stable calcium-consumption safe-band MVP.")
    parser.add_argument("--replay", type=Path, default=Path("runs/ca_interval_recommender_replay.parquet"))
    parser.add_argument("--rules", type=Path, default=Path("runs/ca_regime_calcium_band_rules_ir_lag.csv"))
    parser.add_argument("--rule-audit", type=Path, default=Path("runs/ca_interval_recommender_rule_audit.csv"))
    parser.add_argument("--manual-review", type=Path, default=Path("runs/ca_interval_recommender_manual_review_sheet.csv"))
    parser.add_argument("--readiness-report", type=Path, default=Path("runs/ca_interval_recommender_readiness_report.json"))
    parser.add_argument("--aggregation-report", type=Path, default=Path("runs/ca_interval_aggregation_strategy_test/ca_interval_aggregation_strategy_report.json"))
    parser.add_argument("--diversity-report", type=Path, default=Path("runs/ca_interval_diversity_audit/ca_interval_diversity_audit_report.json"))
    parser.add_argument("--source-artifact", type=Path, default=Path("models/ca_interval_recommender/rule_artifact.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ca_safe_band_mvp"))
    parser.add_argument("--artifact-output", type=Path, default=Path("models/ca_safe_band_mvp/safe_band_artifact.json"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--final-strategy", type=str, default=FINAL_STRATEGY)
    parser.add_argument("--strategy-filter", type=str, default=None, help="If replay contains a strategy column, keep only this strategy before finalization.")
    return parser.parse_args()


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
        raise FileNotFoundError(f"Required input file not found: {path}. Searched {[str(p) for p in search_roots]}.")
    warnings.append(f"Optional input file not found: {path}.")
    return None


def read_table(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_json(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def numeric_series(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")
    return pd.to_numeric(data[column], errors="coerce")


def boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def ensure_targets(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "t90" not in data.columns:
        return data
    t90 = numeric_series(data, "t90")
    if "y_ok" not in data.columns:
        data["y_ok"] = ((t90 >= T90_LOW) & (t90 <= T90_HIGH)).astype(int)
    if "y_low" not in data.columns:
        data["y_low"] = (t90 < T90_LOW).astype(int)
    if "y_high" not in data.columns:
        data["y_high"] = (t90 > T90_HIGH).astype(int)
    if "y_out_spec" not in data.columns:
        data["y_out_spec"] = ((t90 < T90_LOW) | (t90 > T90_HIGH)).astype(int)
    return data


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


def prepare_replay(replay: pd.DataFrame) -> pd.DataFrame:
    data = ensure_targets(replay.copy())
    if "time" in data.columns:
        data["time"] = pd.to_datetime(data["time"], errors="coerce")
        data = data.sort_values("time").reset_index(drop=True)
    if "recommended_ca_consumption_target" not in data.columns:
        data["recommended_ca_consumption_target"] = (
            numeric_series(data, "recommended_ca_consumption_min") + numeric_series(data, "recommended_ca_consumption_max")
        ) / 2.0
    if "interval_position" not in data.columns:
        data["interval_position"] = derive_interval_position(data)
    return data


def final_action_fields(position: object) -> dict[str, str]:
    key = str(position) if str(position) in ACTION_VISIBILITY_POLICY else "missing"
    return ACTION_VISIBILITY_POLICY[key]


def build_monitor_dry_run(replay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in replay.iterrows():
        action = final_action_fields(row.get("interval_position"))
        warning_flags: list[str] = []
        if row.get("interval_position") == "above_band":
            warning_flags.append("high_t90_risk_manual_review")
        if row.get("interval_position") == "below_band":
            warning_flags.append("increase_hint_hidden_diagnostic_only")
        if row.get("interval_position") == "missing":
            warning_flags.append("missing_required_input")
        rows.append({
            "time": row.get("time"),
            "split": row.get("split"),
            "current_ca_consumption": row.get("current_ca_consumption"),
            "recommended_ca_consumption_min": row.get("recommended_ca_consumption_min"),
            "recommended_ca_consumption_max": row.get("recommended_ca_consumption_max"),
            "recommended_ca_consumption_target": row.get("recommended_ca_consumption_target"),
            "interval_position": row.get("interval_position"),
            "final_action_hint": action["final_action_hint"],
            "action_visibility": action["action_visibility"],
            "confidence_level": row.get("confidence_level"),
            "matched_rule_ids": row.get("matched_rule_ids"),
            "selected_rule_ids": row.get("selected_rule_ids"),
            "t90": row.get("t90"),
            "y_ok": row.get("y_ok"),
            "y_low": row.get("y_low"),
            "y_high": row.get("y_high"),
            "y_out_spec": row.get("y_out_spec"),
            "explanation_cn": action["explanation_cn"],
            "engineering_review_required": bool(row.get("interval_position") == "above_band"),
            "warning_flags": ";".join(warning_flags),
        })
    return pd.DataFrame(rows)


def accepted_rules_with_review(rules: pd.DataFrame, rule_audit: pd.DataFrame, manual_review: pd.DataFrame) -> pd.DataFrame:
    data = rules.copy()
    if "rule_status" in data.columns:
        data = data.loc[data["rule_status"].astype(str) == "accept_for_manual_case_review"].copy()
    if "rule_grade" in data.columns:
        data = data.loc[data["rule_grade"].astype(str).isin(["A", "B"])].copy()
    if not rule_audit.empty and "rule_id" in rule_audit.columns:
        keep_cols = [
            "rule_id",
            "recommended_decision",
            "test_like_sample_count",
            "band_accuracy",
            "direction_accuracy",
            "target_accuracy_5pct",
        ]
        data = data.merge(rule_audit[[c for c in keep_cols if c in rule_audit.columns]], on="rule_id", how="left")
    if not manual_review.empty and "rule_id" in manual_review.columns:
        keep_cols = [
            "rule_id",
            "monitor_chain_candidate",
            "manual_review_only",
            "reject_or_refine",
            "engineering_review_question",
            "suggested_human_decision_options",
        ]
        data = data.merge(manual_review[[c for c in keep_cols if c in manual_review.columns]], on="rule_id", how="left")
    for col in ["monitor_chain_candidate", "manual_review_only", "reject_or_refine"]:
        if col not in data.columns:
            if col == "monitor_chain_candidate" and "recommended_decision" in data.columns:
                data[col] = data["recommended_decision"].astype(str).eq("monitor_chain_candidate")
            else:
                data[col] = False
        data[col] = data[col].map(boolish)
    data["recommended_dose_target"] = (
        numeric_series(data, "recommended_dose_min") + numeric_series(data, "recommended_dose_max")
    ) / 2.0
    return data


def final_rule_summary(rules: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rule_id",
        "regime_feature",
        "regime_bin",
        "recommended_dose_min",
        "recommended_dose_max",
        "recommended_dose_target",
        "rule_grade",
        "rule_status",
        "sample_count",
        "monitor_chain_candidate",
        "manual_review_only",
        "reject_or_refine",
        "high_dose_avoidance_candidate",
        "best_ok_rate",
        "best_high_rate",
        "best_low_rate",
        "engineering_review_question",
    ]
    for col in columns:
        if col not in rules.columns:
            rules[col] = np.nan
    return rules[columns].copy()


def manual_review_sheet(final_rules: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in final_rules.iterrows():
        interval = f"[{row.get('recommended_dose_min')}, {row.get('recommended_dose_max')}]"
        if boolish(row.get("monitor_chain_candidate")):
            visibility = "monitor_only_candidate"
        elif boolish(row.get("manual_review_only")):
            visibility = "manual_review_only"
        else:
            visibility = "rule_review_required"
        question = row.get("engineering_review_question")
        if pd.isna(question) or not str(question).strip():
            question = "该稳定钙单耗安全带是否符合当前工艺机理和人工复核要求？"
        rows.append({
            "review_id": f"safe_band_review_{idx + 1:03d}",
            "rule_id": row.get("rule_id"),
            "regime_feature": row.get("regime_feature"),
            "regime_bin": row.get("regime_bin"),
            "recommended_ca_consumption_interval": interval,
            "rule_grade": row.get("rule_grade"),
            "sample_count": row.get("sample_count"),
            "historical_ok_rate": row.get("best_ok_rate"),
            "historical_high_rate": row.get("best_high_rate"),
            "historical_low_rate": row.get("best_low_rate"),
            "monitor_chain_candidate": row.get("monitor_chain_candidate"),
            "manual_review_only": row.get("manual_review_only"),
            "reject_or_refine": row.get("reject_or_refine"),
            "suggested_visibility": visibility,
            "engineering_review_question": question,
            "human_decision_options": "accept_for_monitor_only;show_band_only;hide_action_hint;reject_rule;needs_more_samples",
        })
    return pd.DataFrame(rows)


def rate(frame: pd.DataFrame, col: str) -> float | None:
    if frame.empty or col not in frame.columns:
        return None
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def risk_rows(dry_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    splits = ["all"]
    if "split" in dry_run.columns:
        splits += [s for s in ["train_like", "test_like"] if (dry_run["split"].astype(str) == s).any()]
    positions = ["inside_band", "above_band", "below_band", "missing", "outside_band"]
    for split in splits:
        split_data = dry_run if split == "all" else dry_run.loc[dry_run["split"].astype(str) == split]
        for position in positions:
            if position == "outside_band":
                frame = split_data.loc[split_data["interval_position"].isin(["above_band", "below_band"])]
            else:
                frame = split_data.loc[split_data["interval_position"].astype(str) == position]
            if frame.empty:
                continue
            rows.append({
                "split": split,
                "interval_position": position,
                "sample_count": int(len(frame)),
                "ok_rate": rate(frame, "y_ok"),
                "high_rate": rate(frame, "y_high"),
                "low_rate": rate(frame, "y_low"),
                "out_spec_rate": rate(frame, "y_out_spec"),
                "mean_t90": rate(frame, "t90"),
            })
    return pd.DataFrame(rows)


def add_delta_rows(risk: pd.DataFrame) -> pd.DataFrame:
    rows = risk.to_dict("records")
    for split in risk["split"].dropna().unique():
        subset = risk.loc[risk["split"] == split].set_index("interval_position")
        if "inside_band" not in subset.index:
            continue
        inside = subset.loc["inside_band"]
        for other in ["outside_band", "above_band", "below_band"]:
            if other not in subset.index:
                continue
            comp = subset.loc[other]
            for metric in ["ok_rate", "high_rate", "low_rate", "out_spec_rate"]:
                rows.append({
                    "split": split,
                    "interval_position": f"{other}_minus_inside",
                    "sample_count": int(comp.get("sample_count", 0)),
                    "ok_rate": None,
                    "high_rate": None,
                    "low_rate": None,
                    "out_spec_rate": None,
                    "mean_t90": None,
                    "delta_metric": metric,
                    "delta_value": None if pd.isna(comp.get(metric)) or pd.isna(inside.get(metric)) else float(comp.get(metric) - inside.get(metric)),
                })
    return pd.DataFrame(rows)


def scalar_metric(data: pd.DataFrame, column: str, split: str = "test_like") -> float | None:
    if column not in data.columns:
        return None
    frame = data.loc[data["split"].astype(str) == split] if "split" in data.columns else data
    values = frame[column].dropna()
    if values.empty:
        return None
    if values.dtype == bool or values.astype(str).isin(["True", "False", "true", "false"]).all():
        return float(values.map(boolish).mean())
    return float(pd.to_numeric(values, errors="coerce").mean())


def validation_summary(replay: pd.DataFrame, risk: pd.DataFrame, dry_run: pd.DataFrame) -> pd.DataFrame:
    test = dry_run.loc[dry_run["split"].astype(str) == "test_like"] if "split" in dry_run.columns else dry_run
    inside = risk.loc[(risk["split"] == "test_like") & (risk["interval_position"] == "inside_band")]
    above = risk.loc[(risk["split"] == "test_like") & (risk["interval_position"] == "above_band")]
    below = risk.loc[(risk["split"] == "test_like") & (risk["interval_position"] == "below_band")]
    rows = [
        {"metric": "band_accuracy", "value": scalar_metric(replay, "band_hit"), "interpretation_cn": "验证集推荐区间与 oracle 合理区间的重叠准确率。"},
        {"metric": "direction_accuracy", "value": scalar_metric(replay, "direction_hit"), "interpretation_cn": "验证集推荐方向与 oracle 方向的一致率，仅作离线准确性指标。"},
        {"metric": "target_accuracy_5pct", "value": scalar_metric(replay, "target_hit_5pct"), "interpretation_cn": "推荐中心值与 oracle 中心值 5% 相对误差内比例。"},
        {"metric": "recommendation_coverage", "value": float((test["action_visibility"] != "no_recommendation").mean()) if len(test) else None, "interpretation_cn": "测试集可展示推荐安全带的覆盖率。"},
        {"metric": "final_strategy", "value": FINAL_STRATEGY, "interpretation_cn": "最终锁定中位数聚合。"},
        {"metric": "product_positioning", "value": PRODUCT_POSITIONING, "interpretation_cn": "产品定位为稳定安全带 MVP。"},
    ]
    if not inside.empty:
        row = inside.iloc[0]
        rows += [
            {"metric": "inside_band_ok_rate", "value": row.get("ok_rate"), "interpretation_cn": "实际钙单耗处于安全带内时的 T90 合格率。"},
            {"metric": "inside_band_high_rate", "value": row.get("high_rate"), "interpretation_cn": "实际钙单耗处于安全带内时的高 T90 风险。"},
            {"metric": "inside_band_low_rate", "value": row.get("low_rate"), "interpretation_cn": "实际钙单耗处于安全带内时的低 T90 风险。"},
        ]
    if not above.empty:
        rows.append({"metric": "above_band_high_rate", "value": above.iloc[0].get("high_rate"), "interpretation_cn": "实际钙单耗高于安全带时的高 T90 风险。"})
    if not below.empty:
        rows.append({"metric": "below_band_low_rate", "value": below.iloc[0].get("low_rate"), "interpretation_cn": "实际钙单耗低于安全带时的低 T90 风险。"})
    return pd.DataFrame(rows)


def plot_coverage(dry_run: pd.DataFrame, path: Path) -> None:
    test = dry_run.loc[dry_run["split"].astype(str) == "test_like"].copy() if "split" in dry_run.columns else dry_run.copy()
    test = test.reset_index(drop=True)
    x = np.arange(len(test))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.fill_between(
        x,
        numeric_series(test, "recommended_ca_consumption_min"),
        numeric_series(test, "recommended_ca_consumption_max"),
        color="#90CAF9",
        alpha=0.45,
        label="推荐安全带",
    )
    ax.plot(x, numeric_series(test, "recommended_ca_consumption_target"), color="#1565C0", linestyle="--", linewidth=1.1, label="安全带中心")
    colors = {"inside_band": "#2E7D32", "above_band": "#C62828", "below_band": "#EF6C00", "missing": "#757575"}
    for position, frame in test.groupby("interval_position", dropna=False):
        idx = frame.index.to_numpy()
        ax.scatter(idx, numeric_series(frame, "current_ca_consumption"), s=16, alpha=0.75, color=colors.get(str(position), "#757575"), label=str(position))
    ax.set_title("测试集稳定钙单耗安全带覆盖图")
    ax.set_xlabel("测试集样本序号")
    ax.set_ylabel("钙单耗")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_risk(risk: pd.DataFrame, path: Path) -> None:
    test = risk.loc[(risk["split"] == "test_like") & (risk["interval_position"].isin(["inside_band", "above_band", "below_band", "outside_band"]))].copy()
    metrics = ["ok_rate", "high_rate", "low_rate", "out_spec_rate"]
    x = np.arange(len(test))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.18
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, pd.to_numeric(test[metric], errors="coerce"), width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(test["interval_position"], rotation=20, ha="right")
    ax.set_title("钙单耗区间内外 T90 风险对比")
    ax.set_ylabel("比例")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_visibility(dry_run: pd.DataFrame, path: Path) -> None:
    test = dry_run.loc[dry_run["split"].astype(str) == "test_like"].copy() if "split" in dry_run.columns else dry_run.copy()
    counts = test["action_visibility"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#5E7CE2", alpha=0.85)
    ax.set_title("监测模式动作可见性分布")
    ax.set_xlabel("动作可见性")
    ax.set_ylabel("样本数")
    ax.tick_params(axis="x", rotation=20)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_target_distribution(dry_run: pd.DataFrame, path: Path) -> None:
    test = dry_run.loc[dry_run["split"].astype(str) == "test_like"].copy() if "split" in dry_run.columns else dry_run.copy()
    values = numeric_series(test, "recommended_ca_consumption_target").dropna()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(values, bins=min(30, max(5, values.nunique())), color="#00897B", alpha=0.82, edgecolor="white")
    if not values.empty:
        ax.axvline(values.median(), color="#C62828", linestyle="--", linewidth=1.4, label=f"中位数 {values.median():.6f}")
    ax.set_title("最终推荐钙单耗中心值分布")
    ax.set_xlabel("推荐钙单耗中心值")
    ax.set_ylabel("样本数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_artifact(
    source_artifact: dict[str, object],
    final_rules: pd.DataFrame,
    validation: pd.DataFrame,
    source_artifact_path: Path | None,
    warnings: list[str],
    assumptions: list[str],
) -> dict[str, object]:
    final_rules_json = []
    for _, row in final_rules.iterrows():
        final_rules_json.append({
            "rule_id": row.get("rule_id"),
            "regime_feature": row.get("regime_feature"),
            "regime_bin": row.get("regime_bin"),
            "recommended_dose_min": row.get("recommended_dose_min"),
            "recommended_dose_max": row.get("recommended_dose_max"),
            "recommended_dose_target": row.get("recommended_dose_target"),
            "rule_grade": row.get("rule_grade"),
            "rule_status": row.get("rule_status"),
            "sample_count": row.get("sample_count"),
            "best_ok_rate": row.get("best_ok_rate"),
            "best_high_rate": row.get("best_high_rate"),
            "best_low_rate": row.get("best_low_rate"),
            "ok_lift_vs_overall": row.get("ok_lift_vs_overall"),
            "high_delta_vs_overall": row.get("high_delta_vs_overall"),
            "low_delta_vs_overall": row.get("low_delta_vs_overall"),
            "high_dose_avoidance_candidate": row.get("high_dose_avoidance_candidate"),
            "time_stable": row.get("time_stable"),
            "monitor_chain_candidate": row.get("monitor_chain_candidate"),
            "manual_review_only": row.get("manual_review_only"),
            "reject_or_refine": row.get("reject_or_refine"),
            "engineering_review_question": row.get("engineering_review_question"),
        })
    validation_map = {str(row["metric"]): row["value"] for _, row in validation.iterrows()}
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_version": ARTIFACT_VERSION,
        "product_positioning": PRODUCT_POSITIONING,
        "final_strategy": FINAL_STRATEGY,
        "primary_dose_feature": source_artifact.get("primary_dose_feature", PRIMARY_DOSE_FEATURE),
        "calcium_feed_conversion_formula": source_artifact.get(
            "calcium_feed_conversion_formula",
            "recommended_ca_feed = recommended_ca_consumption * rubber_flow_2_win_60_mean",
        ),
        "source_artifact_path": str(source_artifact_path) if source_artifact_path else None,
        "accepted_rule_count": int(len(final_rules)),
        "monitor_candidate_rule_count": int(pd.Series(final_rules.get("monitor_chain_candidate", [])).map(boolish).sum()) if len(final_rules) else 0,
        "final_rules": final_rules_json,
        "regime_boundaries": source_artifact.get("regime_boundary_method", {}),
        "aggregation_policy": {
            "strategy": FINAL_STRATEGY,
            "description": f"Use {FINAL_STRATEGY} because the latest aggregation strategy test selected it for this finalized monitor-only artifact.",
        },
        "action_visibility_policy": ACTION_VISIBILITY_POLICY,
        "required_input_features": [
            "current_ca_consumption",
            "recommended_ca_consumption_min",
            "recommended_ca_consumption_max",
        ],
        "optional_input_features": [
            "recommended_ca_consumption_target",
            "confidence_level",
            "matched_rule_ids",
            "selected_rule_ids",
            "rubber_flow_2_win_60_mean",
            "ir_lag_context_value",
        ],
        "output_schema": [
            "recommended_ca_consumption_min",
            "recommended_ca_consumption_max",
            "recommended_ca_consumption_target",
            "interval_position",
            "final_action_hint",
            "action_visibility",
            "explanation_cn",
            "warning_flags",
        ],
        "validation_summary": validation_map,
        "known_limitations": [
            "Offline validation only.",
            "No causal proof.",
            "No guarantee that T90 will be qualified when calcium consumption is inside the band.",
            "Engineering human review is required before any monitor-chain use.",
        ],
        "safety_constraints": {
            "monitor_only": True,
            "automatic_control": False,
            "dcs_writeback": False,
            "increase_hint_hidden": True,
            "engineering_review_required": True,
            "no_guarantee_t90_qualified": True,
            "shadow_trial_recommended": False,
        },
        "warnings": warnings,
        "assumptions": assumptions,
    }


def append_doc(
    doc_path: Path,
    artifact_path: Path,
    output_dir: Path,
    manual_sheet: Path,
    risk_summary: dict[str, object],
    recommended_next_step: str,
) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    existing = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""
    section_no = 25
    while f"## {section_no}." in existing:
        section_no += 1
    section = f"""

## {section_no}. 稳定钙单耗安全带 MVP 定版与监测接口准备

### {section_no}.1 定版原因

Stage 24 对比了中位数聚合、最高优先级规则、加权平均和交集策略。中位数聚合保持了最高的推荐区间准确率和较好的方向准确率，且风险护栏通过；Top-rule-only 和交集策略没有恢复有效多样性，准确率下降；加权平均虽然增加区间差异，但方向准确率不足。因此本阶段锁定 `median_aggregation_baseline`。

产品定位为 `stable_safe_band_mvp`：它不是强动态分工况处方系统，而是稳定钙单耗安全带监测 MVP。其含义是：把实际钙单耗控制在历史安全带内，历史上更可能提高 T90 合格概率，但不保证 T90 必然合格。

### {section_no}.2 动作可见性策略

- inside_band：仅监测展示，提示“当前钙单耗处于推荐安全区间内，建议维持观察”。
- above_band：人工复核必需，提示“当前钙单耗高于推荐安全区间，历史数据中高 T90 风险偏高，建议人工复核是否需要小幅降钙”。
- below_band：仅诊断展示，隐藏加钙操作建议。
- missing：关键输入缺失，不生成推荐。

该 MVP 不提供自动控制、不做 DCS 写回、不推荐影子试验。

### {section_no}.3 输出

- 定版 artifact：`{artifact_path}`
- dry-run 表：`{output_dir / 'final_monitor_dry_run.parquet'}` 与 `{output_dir / 'final_monitor_dry_run.csv'}`
- 规则汇总：`{output_dir / 'final_rule_summary.csv'}`
- 人工复核表：`{manual_sheet}`

### {section_no}.4 风险摘要

- inside_band ok/high/low：{risk_summary.get('inside_ok_rate')} / {risk_summary.get('inside_high_rate')} / {risk_summary.get('inside_low_rate')}
- above_band high_rate：{risk_summary.get('above_high_rate')}
- below_band low_rate：{risk_summary.get('below_low_rate')}

推荐下一步：`{recommended_next_step}`。

局限性：离线验证；非因果证明；无自动控制；无 DCS 写回；必须经过工程人工复核。
"""
    with doc_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(section)


def main() -> None:
    args = parse_args()
    global FINAL_STRATEGY
    FINAL_STRATEGY = args.final_strategy
    configure_matplotlib()
    warnings: list[str] = []
    assumptions = [
        f"The final MVP locks {FINAL_STRATEGY} from the latest aggregation strategy comparison.",
        "Below-band conditions are diagnostic-only; no operational increase hint is exposed.",
        "Above-band conditions require manual engineering review and are not automatic decrease commands.",
    ]
    runs_root = Path("runs")
    models_root = Path("models")
    replay_path = resolve_path(args.replay, required=True, search_roots=[runs_root], warnings=warnings)
    rules_path = resolve_path(args.rules, required=True, search_roots=[runs_root], warnings=warnings)
    rule_audit_path = resolve_path(args.rule_audit, required=False, search_roots=[runs_root], warnings=warnings)
    manual_review_path = resolve_path(args.manual_review, required=False, search_roots=[runs_root], warnings=warnings)
    readiness_report_path = resolve_path(args.readiness_report, required=False, search_roots=[runs_root], warnings=warnings)
    aggregation_report_path = resolve_path(args.aggregation_report, required=True, search_roots=[runs_root], warnings=warnings)
    diversity_report_path = resolve_path(args.diversity_report, required=False, search_roots=[runs_root], warnings=warnings)
    source_artifact_path = resolve_path(args.source_artifact, required=True, search_roots=[models_root, runs_root], warnings=warnings)

    replay_raw = read_table(replay_path)
    if args.strategy_filter and "strategy" in replay_raw.columns:
        before = len(replay_raw)
        replay_raw = replay_raw.loc[replay_raw["strategy"].astype(str) == args.strategy_filter].copy()
        warnings.append(f"Filtered strategy replay from {before} rows to {len(replay_raw)} rows for strategy={args.strategy_filter}.")
    if "strategy_selected_rule_ids" in replay_raw.columns and "selected_rule_ids" not in replay_raw.columns:
        replay_raw["selected_rule_ids"] = replay_raw["strategy_selected_rule_ids"]
    replay = prepare_replay(replay_raw)
    rules = read_table(rules_path)
    rule_audit = read_table(rule_audit_path)
    manual_review = read_table(manual_review_path)
    readiness_report = load_json(readiness_report_path)
    aggregation_report = load_json(aggregation_report_path)
    diversity_report = load_json(diversity_report_path)
    source_artifact = load_json(source_artifact_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    final_rules_full = accepted_rules_with_review(rules, rule_audit, manual_review)
    final_rules = final_rule_summary(final_rules_full)
    dry_run = build_monitor_dry_run(replay)
    risk = add_delta_rows(risk_rows(dry_run))
    validation = validation_summary(replay, risk, dry_run)
    review_sheet = manual_review_sheet(final_rules)

    dry_run_parquet = args.output_dir / "final_monitor_dry_run.parquet"
    dry_run_csv = args.output_dir / "final_monitor_dry_run.csv"
    rule_summary_csv = args.output_dir / "final_rule_summary.csv"
    risk_csv = args.output_dir / "final_risk_summary.csv"
    validation_csv = args.output_dir / "final_validation_summary.csv"
    report_json = args.output_dir / "ca_safe_band_mvp_finalization_report.json"
    manual_sheet_path = args.table_dir / "ca_safe_band_mvp_manual_review_sheet.csv"

    dry_run.to_parquet(dry_run_parquet, index=False)
    dry_run.to_csv(dry_run_csv, index=False, encoding="utf-8-sig")
    final_rules.to_csv(rule_summary_csv, index=False, encoding="utf-8-sig")
    risk.to_csv(risk_csv, index=False, encoding="utf-8-sig")
    validation.to_csv(validation_csv, index=False, encoding="utf-8-sig")
    review_sheet.to_csv(manual_sheet_path, index=False, encoding="utf-8-sig")

    artifact = build_artifact(source_artifact, final_rules_full, validation, source_artifact_path, warnings, assumptions)
    with args.artifact_output.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(artifact), handle, ensure_ascii=False, indent=2)

    figures = [
        args.figure_dir / "ca_safe_band_mvp_test_like_coverage.png",
        args.figure_dir / "ca_safe_band_mvp_interval_position_risk.png",
        args.figure_dir / "ca_safe_band_mvp_action_visibility_distribution.png",
        args.figure_dir / "ca_safe_band_mvp_recommended_target_distribution.png",
    ]
    plot_coverage(dry_run, figures[0])
    plot_risk(risk, figures[1])
    plot_visibility(dry_run, figures[2])
    plot_target_distribution(dry_run, figures[3])

    recommended_next_step = "human_review_safe_band_mvp"
    test_risk = risk.loc[risk["split"] == "test_like"]
    inside = test_risk.loc[test_risk["interval_position"] == "inside_band"]
    above = test_risk.loc[test_risk["interval_position"] == "above_band"]
    below = test_risk.loc[test_risk["interval_position"] == "below_band"]
    risk_summary = {
        "inside_ok_rate": None if inside.empty else inside.iloc[0].get("ok_rate"),
        "inside_high_rate": None if inside.empty else inside.iloc[0].get("high_rate"),
        "inside_low_rate": None if inside.empty else inside.iloc[0].get("low_rate"),
        "above_high_rate": None if above.empty else above.iloc[0].get("high_rate"),
        "below_low_rate": None if below.empty else below.iloc[0].get("low_rate"),
    }

    generated_outputs = {
        "artifact": str(args.artifact_output),
        "machine": [
            str(dry_run_parquet),
            str(dry_run_csv),
            str(rule_summary_csv),
            str(risk_csv),
            str(validation_csv),
            str(report_json),
        ],
        "tables": [str(manual_sheet_path)],
        "figures": [str(path) for path in figures],
    }
    finalization_report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_paths": {
            "replay": str(replay_path),
            "rules": str(rules_path),
            "rule_audit": str(rule_audit_path) if rule_audit_path else None,
            "manual_review": str(manual_review_path) if manual_review_path else None,
            "readiness_report": str(readiness_report_path) if readiness_report_path else None,
            "aggregation_report": str(aggregation_report_path),
            "diversity_report": str(diversity_report_path) if diversity_report_path else None,
            "source_artifact": str(source_artifact_path),
        },
        "output_dir": str(args.output_dir),
        "artifact_output_path": str(args.artifact_output),
        "figure_dir": str(args.figure_dir),
        "table_dir": str(args.table_dir),
        "final_strategy": FINAL_STRATEGY,
        "product_positioning": PRODUCT_POSITIONING,
        "action_visibility_policy": ACTION_VISIBILITY_POLICY,
        "artifact_summary": {
            "artifact_version": ARTIFACT_VERSION,
            "accepted_rule_count": int(len(final_rules)),
            "monitor_candidate_rule_count": int(final_rules["monitor_chain_candidate"].map(boolish).sum()) if len(final_rules) else 0,
            "source_artifact_primary_dose_feature": source_artifact.get("primary_dose_feature"),
        },
        "final_rule_summary": {
            "rule_count": int(len(final_rules)),
            "monitor_chain_candidate_count": int(final_rules["monitor_chain_candidate"].map(boolish).sum()) if len(final_rules) else 0,
            "manual_review_only_count": int(final_rules["manual_review_only"].map(boolish).sum()) if len(final_rules) else 0,
            "reject_or_refine_count": int(final_rules["reject_or_refine"].map(boolish).sum()) if len(final_rules) else 0,
        },
        "final_monitor_dry_run_summary": {
            "row_count": int(len(dry_run)),
            "test_like_row_count": int((dry_run["split"].astype(str) == "test_like").sum()) if "split" in dry_run.columns else None,
            "interval_position_counts": dry_run["interval_position"].value_counts(dropna=False).to_dict(),
            "action_visibility_counts": dry_run["action_visibility"].value_counts(dropna=False).to_dict(),
        },
        "final_risk_summary": risk_summary,
        "final_validation_summary": validation.to_dict("records"),
        "source_stage_summaries": {
            "readiness_status": readiness_report.get("readiness_status"),
            "readiness_next_step": readiness_report.get("recommended_next_step"),
            "aggregation_best_strategy": aggregation_report.get("best_strategy"),
            "aggregation_switch_recommendation": aggregation_report.get("switch_recommendation"),
            "aggregation_next_step": aggregation_report.get("recommended_next_step"),
            "diversity_classification": diversity_report.get("interpretation_classification"),
            "diversity_reason": diversity_report.get("likely_reason_for_stable_interval"),
        },
        "generated_outputs": generated_outputs,
        "warnings": warnings,
        "assumptions": assumptions,
        "recommended_next_step": recommended_next_step,
    }
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(finalization_report), handle, ensure_ascii=False, indent=2)

    append_doc(args.doc, args.artifact_output, args.output_dir, manual_sheet_path, risk_summary, recommended_next_step)

    counts = dry_run.loc[dry_run["split"].astype(str) == "test_like", "interval_position"].value_counts() if "split" in dry_run.columns else dry_run["interval_position"].value_counts()
    print("Calcium safe-band MVP finalization summary")
    print(f"final_strategy: {FINAL_STRATEGY}")
    print(f"product_positioning: {PRODUCT_POSITIONING}")
    print(f"artifact path: {args.artifact_output}")
    print(f"final rule count: {len(final_rules)}")
    print(f"monitor dry-run row count: {len(dry_run)}")
    print(f"inside/above/below counts: {int(counts.get('inside_band', 0))} / {int(counts.get('above_band', 0))} / {int(counts.get('below_band', 0))}")
    print(f"inside_band ok/high/low: {risk_summary.get('inside_ok_rate')} / {risk_summary.get('inside_high_rate')} / {risk_summary.get('inside_low_rate')}")
    print(f"above_band high_rate: {risk_summary.get('above_high_rate')}")
    print("Generated report/table/figure paths:")
    for path in [args.artifact_output] + [Path(p) for p in generated_outputs["machine"] + generated_outputs["tables"] + generated_outputs["figures"]]:
        print(f"  {path}")
    print(f"recommended_next_step: {recommended_next_step}")
    print(f"Documentation appended: {args.doc}")
    print("No generated outputs were written under data/.")


if __name__ == "__main__":
    main()
