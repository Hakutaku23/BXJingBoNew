from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline

try:
    from sklearn.ensemble import HistGradientBoostingClassifier

    HAS_HGB = True
except Exception:
    HAS_HGB = False


RANDOM_SEED = 42
T90_LOW = 8.20
T90_HIGH = 8.70
IR_FEATURES = [
    "output_ir_corrected",
    "output_ir_corrected_lag_0",
    "output_ir_corrected_win_5_mean",
    "output_ir_corrected_win_15_mean",
    "output_ir_corrected_win_30_mean",
    "output_ir_corrected_win_15_std",
    "output_ir_corrected_win_30_std",
    "output_ir_corrected_win_15_slope",
]
CALCIUM_FEATURES = [
    "ca_per_rubber_flow_win_60_mean",
    "ca_per_rubber_flow_lag_165",
    "ca_win_60_mean",
    "ca_lag_165",
    "ca_win_30_mean",
    "ca_win_120_mean",
    "ca_delta_30",
    "ca_delta_60",
]
TARGETS = ["y_high", "y_out_spec", "y_low"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate outlet corrected IR as a mechanism-related quality proxy.")
    parser.add_argument("--input", type=Path, default=Path("data/data_clean_with_ir.parquet"))
    parser.add_argument("--ca-feature-input", type=Path, default=Path("data/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/output_ir_proxy_evaluation.csv"))
    parser.add_argument("--bin-output", type=Path, default=Path("data/output_ir_proxy_bins.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/output_ir_proxy_evaluation.json"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--n-bins", type=int, default=5)
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


def load_json_if_exists(path: Path) -> dict[str, object]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def ensure_targets(frame: pd.DataFrame) -> pd.DataFrame:
    if "t90" not in frame.columns:
        raise ValueError("Input dataset must contain t90.")
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


def valid_corr(x: pd.Series, y: pd.Series, method: str) -> float | None:
    values = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(values) < 30 or values["x"].nunique() <= 1 or values["y"].nunique() <= 1:
        return None
    corr = values["x"].corr(values["y"], method=method)
    if corr is None or not np.isfinite(corr):
        return None
    return float(corr)


def make_bins(values: pd.Series, n_bins: int) -> tuple[pd.Series, int, list[str]]:
    warnings: list[str] = []
    series = pd.to_numeric(values, errors="coerce")
    usable = series.dropna()
    if usable.empty:
        return pd.Series(pd.NA, index=values.index, dtype="Int64"), 0, ["No usable values for binning."]
    max_bins = min(n_bins, int(usable.nunique()), int(len(usable)))
    if max_bins < 2:
        return pd.Series(0, index=values.index, dtype="Int64").where(series.notna(), pd.NA), 1, [
            "Only one effective bin is possible."
        ]
    for bins in range(max_bins, 1, -1):
        try:
            cut = pd.qcut(series, q=bins, labels=False, duplicates="drop")
            effective = int(pd.Series(cut).dropna().nunique())
            if effective >= 2:
                if effective < n_bins:
                    warnings.append(f"Effective bin count {effective} is lower than requested {n_bins}.")
                return pd.Series(cut, index=values.index, dtype="Int64"), effective, warnings
        except ValueError:
            continue
    ranked = series.rank(method="first")
    cut = pd.qcut(ranked, q=min(n_bins, int(ranked.dropna().nunique())), labels=False, duplicates="drop")
    warnings.append("Used rank-based fallback for binning.")
    return pd.Series(cut, index=values.index, dtype="Int64"), int(pd.Series(cut).dropna().nunique()), warnings


def ir_diagnostics(frame: pd.DataFrame, ir_feature: str) -> dict[str, object]:
    usable = frame[frame[ir_feature].notna()]
    return {
        "usable_sample_count": int(len(usable)),
        "missing_count": int(frame[ir_feature].isna().sum()),
        "missing_rate": float(frame[ir_feature].isna().mean()),
        "spearman_corr_with_t90": valid_corr(frame[ir_feature], frame["t90"], "spearman"),
        "pearson_corr_with_t90": valid_corr(frame[ir_feature], frame["t90"], "pearson"),
        "spearman_corr_with_y_ok": valid_corr(frame[ir_feature], frame["y_ok"], "spearman"),
        "spearman_corr_with_y_low": valid_corr(frame[ir_feature], frame["y_low"], "spearman"),
        "spearman_corr_with_y_high": valid_corr(frame[ir_feature], frame["y_high"], "spearman"),
        "spearman_corr_with_y_out_spec": valid_corr(frame[ir_feature], frame["y_out_spec"], "spearman"),
    }


def bin_table(frame: pd.DataFrame, ir_feature: str, n_bins: int) -> tuple[pd.DataFrame, dict[str, object], list[str]]:
    usable = frame[frame[ir_feature].notna()].copy()
    if usable.empty:
        return pd.DataFrame(), {"effective_bin_count": 0, "high_rate_spread": None, "out_spec_rate_spread": None}, []
    bins, effective, warnings = make_bins(usable[ir_feature], n_bins)
    usable["bin_id"] = bins
    rows = []
    for bin_id, group in usable[usable["bin_id"].notna()].groupby("bin_id", sort=True):
        values = pd.to_numeric(group[ir_feature], errors="coerce")
        row = {
            "ir_feature": ir_feature,
            "bin_id": int(bin_id),
            "bin_label": f"{int(bin_id)}: [{values.min():.6g}, {values.max():.6g}]",
            "sample_count": int(len(group)),
            "ir_min": float(values.min()),
            "ir_max": float(values.max()),
            "ir_mean": float(values.mean()),
            "ir_median": float(values.median()),
            "t90_mean": float(group["t90"].mean()),
            "t90_median": float(group["t90"].median()),
            "ok_count": int(group["y_ok"].sum()),
            "ok_rate": float(group["y_ok"].mean()),
            "low_count": int(group["y_low"].sum()),
            "low_rate": float(group["y_low"].mean()),
            "high_count": int(group["y_high"].sum()),
            "high_rate": float(group["y_high"].mean()),
            "out_spec_count": int(group["y_out_spec"].sum()),
            "out_spec_rate": float(group["y_out_spec"].mean()),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    summary = {
        "effective_bin_count": int(effective),
        "high_rate_spread": float(table["high_rate"].max() - table["high_rate"].min()) if len(table) else None,
        "out_spec_rate_spread": float(table["out_spec_rate"].max() - table["out_spec_rate"].min()) if len(table) else None,
        "low_rate_spread": float(table["low_rate"].max() - table["low_rate"].min()) if len(table) else None,
        "min_bin_sample_count": int(table["sample_count"].min()) if len(table) else 0,
    }
    return table, summary, warnings


def load_ca_features(path: Path, warnings: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        warnings.append(f"Calcium feature input is missing: {path}; skipping calcium-related diagnostics.")
        return None
    ca = pd.read_parquet(path)
    if "time" not in ca.columns:
        warnings.append("Calcium feature input lacks time; skipping calcium-related diagnostics.")
        return None
    ca = ca.copy()
    ca["time"] = pd.to_datetime(ca["time"], errors="coerce")
    return ca.sort_values("time").reset_index(drop=True)


def calcium_to_ir(frame: pd.DataFrame, calcium_features: list[str], ir_features: list[str], n_bins: int) -> dict[str, object]:
    diagnostics: dict[str, object] = {}
    for ca_feature in calcium_features:
        for ir_feature in ir_features:
            values = pd.DataFrame({"ca": frame[ca_feature], "ir": frame[ir_feature]}).dropna()
            if len(values) < 30:
                continue
            bins, _, _ = make_bins(values["ca"], n_bins)
            tmp = values.copy()
            tmp["bin_id"] = bins.dropna().to_numpy() if len(bins.dropna()) == len(tmp) else pd.Series(bins, index=values.index)
            by_bin = tmp[tmp["bin_id"].notna()].groupby("bin_id")["ir"].mean().sort_index()
            trend = float(by_bin.iloc[-1] - by_bin.iloc[0]) if len(by_bin) >= 2 else None
            key = f"{ca_feature}__{ir_feature}"
            diagnostics[key] = {
                "calcium_feature": ca_feature,
                "ir_feature": ir_feature,
                "usable_sample_count": int(len(values)),
                "spearman_corr_ca_ir": valid_corr(values["ca"], values["ir"], "spearman"),
                "pearson_corr_ca_ir": valid_corr(values["ca"], values["ir"], "pearson"),
                "calcium_bin_mean_ir": {str(int(k)): float(v) for k, v in by_bin.items()},
                "calcium_bin_ir_trend_lowest_to_highest": trend,
            }
    return diagnostics


def make_classifier(warnings: list[str], target: str) -> object:
    if HAS_HGB:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, random_state=RANDOM_SEED),
        )
        return model
    return make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=2, random_state=RANDOM_SEED),
    )


def fit_predict_probability(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, target: str, warnings: list[str]) -> np.ndarray | None:
    if y_train.nunique() < 2:
        warnings.append(f"{target}: training split has only one class; skipping model.")
        return None
    model = make_classifier(warnings, target)
    try:
        model.fit(x_train, y_train)
    except Exception as exc:
        if HAS_HGB:
            warnings.append(f"{target}: HistGradientBoosting failed with {repr(exc)}; using GradientBoosting fallback.")
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=2, random_state=RANDOM_SEED),
            )
            model.fit(x_train, y_train)
        else:
            raise
    proba = model.predict_proba(x_test)
    return proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(x_test))


def classification_metrics(y_true: pd.Series, probability: np.ndarray | None) -> dict[str, float | None]:
    if probability is None or y_true.nunique() < 2:
        return {"test_ap": None, "test_auc": None, "test_brier": None}
    y = y_true.astype(int).to_numpy()
    return {
        "test_ap": float(average_precision_score(y, probability)),
        "test_auc": float(roc_auc_score(y, probability)),
        "test_brier": float(brier_score_loss(y, probability)),
    }


def incremental_tests(frame: pd.DataFrame, calcium_features: list[str], ir_features: list[str], warnings: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    flat: list[dict[str, object]] = []
    if not calcium_features:
        warnings.append("No calcium baseline features available; skipping incremental value test.")
        return rows, flat
    data = frame.sort_values("time").reset_index(drop=True).copy()
    split = int(len(data) * 0.8)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()
    for target in TARGETS:
        if target == "y_low" and (train[target].sum() < 5 or test[target].sum() < 5):
            warnings.append("Skipping y_low incremental test because positives are too few in train or test.")
            continue
        base_probability = fit_predict_probability(train[calcium_features], train[target], test[calcium_features], target, warnings)
        base_metrics = classification_metrics(test[target], base_probability)
        for ir_feature in ir_features:
            model_features = calcium_features + [ir_feature]
            ir_probability = fit_predict_probability(train[model_features], train[target], test[model_features], target, warnings)
            ir_metrics = classification_metrics(test[target], ir_probability)
            result = {
                "target": target,
                "ir_feature": ir_feature,
                "model": "calcium_plus_one_ir",
                **ir_metrics,
                "baseline_test_ap": base_metrics["test_ap"],
                "baseline_test_auc": base_metrics["test_auc"],
                "baseline_test_brier": base_metrics["test_brier"],
                "delta_ap": None if base_metrics["test_ap"] is None or ir_metrics["test_ap"] is None else ir_metrics["test_ap"] - base_metrics["test_ap"],
                "delta_auc": None if base_metrics["test_auc"] is None or ir_metrics["test_auc"] is None else ir_metrics["test_auc"] - base_metrics["test_auc"],
                "delta_brier": None if base_metrics["test_brier"] is None or ir_metrics["test_brier"] is None else ir_metrics["test_brier"] - base_metrics["test_brier"],
                "test_sample_count": int(len(test)),
            }
            rows.append(result)
            for metric in ["test_ap", "test_auc", "test_brier", "delta_ap", "delta_auc", "delta_brier"]:
                flat.append(
                    {
                        "diagnostic_type": "incremental_value_test",
                        "target_or_pair": target,
                        "feature": ir_feature,
                        "metric": metric,
                        "value": result[metric],
                        "sample_count": int(len(test)),
                    }
                )
    return rows, flat


def target_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    result = {}
    for target in ["y_ok", "y_low", "y_high", "y_out_spec"]:
        result[target] = {str(k): int(v) for k, v in frame[target].value_counts().sort_index().items()}
    return result


def choose_best_ir(
    diagnostics: dict[str, object],
    bin_summaries: dict[str, object],
    incremental: list[dict[str, object]],
) -> tuple[str | None, str, dict[str, object]]:
    candidates = []
    for feature, diag in diagnostics.items():
        usable = int(diag["usable_sample_count"])
        high_corr = diag.get("spearman_corr_with_y_high")
        out_corr = diag.get("spearman_corr_with_y_out_spec")
        summary = bin_summaries.get(feature, {})
        spread = max(
            abs(float(summary.get("high_rate_spread") or 0.0)),
            abs(float(summary.get("out_spec_rate_spread") or 0.0)),
        )
        meaningful = usable >= 500 and (
            (high_corr is not None and abs(float(high_corr)) >= 0.08)
            or (out_corr is not None and abs(float(out_corr)) >= 0.08)
            or spread >= 0.05
        )
        best_delta_ap = -999.0
        best_delta_auc = -999.0
        for row in incremental:
            if row["ir_feature"] == feature and row["target"] in ["y_high", "y_out_spec"]:
                if row.get("delta_ap") is not None:
                    best_delta_ap = max(best_delta_ap, float(row["delta_ap"]))
                if row.get("delta_auc") is not None:
                    best_delta_auc = max(best_delta_auc, float(row["delta_auc"]))
        incremental_good = best_delta_ap >= 0.03 or best_delta_auc >= 0.03
        candidates.append(
            {
                "feature": feature,
                "usable_sample_count": usable,
                "meaningful": meaningful,
                "incremental_good": incremental_good,
                "best_delta_ap": None if best_delta_ap == -999.0 else best_delta_ap,
                "best_delta_auc": None if best_delta_auc == -999.0 else best_delta_auc,
                "spread": spread,
                "min_bin_sample_count": summary.get("min_bin_sample_count", 0),
                "score": (int(meaningful) * 10) + (int(incremental_good) * 10) + min(usable / 10000.0, 1.0) + spread,
            }
        )
    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
    best = candidates[0] if candidates else {}
    if best and best["meaningful"] and best["incremental_good"]:
        return best["feature"], "use_ir_in_walk_forward_policy_rewrite", best
    if best and best["meaningful"]:
        return best["feature"], "keep_ir_as_monitoring_only", best
    return (best.get("feature") if best else None), "ir_not_useful_stop_integration", best


def next_doc_section_number(doc_path: Path) -> int:
    text = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""
    numbers = [int(match.group(1)) for match in re.finditer(r"^##\s+(\d+)\.", text, flags=re.MULTILINE)]
    return max(numbers, default=12) + 1


def append_doc(
    doc_path: Path,
    section: int,
    merge_report: dict[str, object],
    report: dict[str, object],
    best_ir: str | None,
    recommendation: str,
) -> bool:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    calcium_diag = report.get("calcium_to_ir_diagnostics", {})
    ca_summary = "无可用钙剂-IR 诊断"
    if calcium_diag:
        first = next(iter(calcium_diag.values()))
        ca_summary = (
            f"{first['calcium_feature']} 与 {first['ir_feature']} 的 Spearman="
            f"{first.get('spearman_corr_ca_ir')}"
        )
    best_diag = report["per_ir_feature_diagnostics"].get(best_ir, {}) if best_ir else {}
    lines = [
        "",
        f"## {section}. 出口红外矫正特征接入与代理价值评估",
        "",
        "- 本阶段用于在严格 walk-forward 钙剂策略重写前，评估产品出口红外矫正值是否可作为机理相关的中间质量代理。",
        "- T90 为人工 LIMS 检测，记录精度为 0.1，实际误差约 0.1；控制目标应理解为 [8.20, 8.70] 合格区间，而非精确预测 8.45。",
        "- 卤化流程停留时间约 3 小时，且不同单元停留时间不同，因此出口 IR 不被当作直接 T90 测量值，而是作为下游质量状态和潜在中介变量评估。",
        (
            "- `output.csv` 接入规则：第一列作为时间戳，最后一列作为出口红外矫正值，"
            "所有中间列完全忽略，不进入 `data_clean_with_ir.parquet`。"
        ),
        f"- 输入文件：`data/data_clean.parquet`、`data/output.csv`、`data/t90_ca_feature_dataset.parquet`。",
        f"- 输出文件：`data/data_clean_with_ir.parquet`、`data/data_clean_with_ir_report.json`、`data/output_ir_proxy_evaluation.csv`、`data/output_ir_proxy_bins.csv`、`data/output_ir_proxy_evaluation.json`。",
        f"- 检测到的时间戳列：`{merge_report.get('detected_ir_timestamp_column')}`；红外值列：`{merge_report.get('detected_ir_value_column')}`。",
        f"- 被忽略的中间列数量：{merge_report.get('ignored_middle_column_count')}。",
        f"- IR 覆盖率：{merge_report.get('output_ir_non_null_rate')}；重叠时间：{merge_report.get('overlap_time_range')}。",
        f"- 最佳 IR 特征：`{best_ir}`。",
        f"- 钙剂到 IR 关系摘要：{ca_summary}。",
        (
            "- IR 到 T90 风险关系摘要："
            f"usable={best_diag.get('usable_sample_count')}，"
            f"Spearman(y_high)={best_diag.get('spearman_corr_with_y_high')}，"
            f"Spearman(y_out_spec)={best_diag.get('spearman_corr_with_y_out_spec')}。"
        ),
        f"- 分箱结论摘要：{report.get('ir_bin_analysis_summary', {}).get(best_ir, {}) if best_ir else {}}。",
        f"- 描述性中介检查：{report.get('descriptive_mediation_check')}。",
        f"- 增量价值测试：{report.get('incremental_value_test', {}).get('summary')}。",
        f"- recommended_next_step：`{recommendation}`。",
        "- 结论：IR 是否进入 walk-forward 策略重写取决于其相对钙剂特征的增量收益；本阶段不建议 shadow trial，也不实施自动控制。",
    ]
    with doc_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    return True


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    if not args.input.exists():
        raise FileNotFoundError(f"Input parquet does not exist: {args.input}")
    frame = pd.read_parquet(args.input)
    if "time" not in frame.columns:
        raise ValueError("Input must contain time.")
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = ensure_targets(frame)
    target_rows = frame[frame["t90"].notna()].sort_values("time").reset_index(drop=True)
    ir_features = [feature for feature in IR_FEATURES if feature in target_rows.columns]
    if not ir_features:
        raise ValueError("No output_ir_corrected feature is available for evaluation.")

    per_diag = {feature: ir_diagnostics(target_rows, feature) for feature in ir_features}
    bin_tables = []
    bin_summaries = {}
    flat_rows = []
    for feature in ir_features:
        table, summary, bin_warnings = bin_table(target_rows, feature, args.n_bins)
        warnings.extend([f"{feature}: {item}" for item in bin_warnings])
        if not table.empty:
            bin_tables.append(table)
        bin_summaries[feature] = summary
        for metric, value in per_diag[feature].items():
            flat_rows.append(
                {
                    "diagnostic_type": "ir_to_t90_risk",
                    "target_or_pair": metric,
                    "feature": feature,
                    "metric": metric,
                    "value": value,
                    "sample_count": per_diag[feature]["usable_sample_count"],
                }
            )

    ca = load_ca_features(args.ca_feature_input, warnings)
    model_frame = target_rows[["time", "t90", "y_ok", "y_low", "y_high", "y_out_spec"] + ir_features].copy()
    calcium_features: list[str] = []
    if ca is not None:
        calcium_features = [feature for feature in CALCIUM_FEATURES if feature in ca.columns]
        model_frame = model_frame.merge(ca[["time"] + calcium_features], on="time", how="left")
    else:
        calcium_features = [feature for feature in CALCIUM_FEATURES if feature in model_frame.columns]
    ca_ir = calcium_to_ir(model_frame, calcium_features, ir_features, args.n_bins) if calcium_features else {}
    for key, item in ca_ir.items():
        for metric in ["spearman_corr_ca_ir", "pearson_corr_ca_ir", "calcium_bin_ir_trend_lowest_to_highest"]:
            flat_rows.append(
                {
                    "diagnostic_type": "calcium_to_ir",
                    "target_or_pair": key,
                    "feature": item["ir_feature"],
                    "metric": metric,
                    "value": item.get(metric),
                    "sample_count": item["usable_sample_count"],
                }
            )

    inc_rows, inc_flat = incremental_tests(model_frame, calcium_features, ir_features, warnings)
    flat_rows.extend(inc_flat)

    best_ir, recommendation, best_info = choose_best_ir(per_diag, bin_summaries, inc_rows)
    has_ca_ir = any(
        item.get("spearman_corr_ca_ir") is not None and abs(float(item["spearman_corr_ca_ir"])) >= 0.05
        for item in ca_ir.values()
    )
    has_ir_risk = bool(best_info and best_info.get("meaningful"))
    mediation = {
        "descriptive_mediation_possible": bool(has_ca_ir and has_ir_risk),
        "not_causal_proof": True,
        "explanation": "This only checks whether calcium relates to IR and IR stratifies T90 risk; it is not causal proof.",
    }
    report = {
        "input_path": str(args.input),
        "ca_feature_input_path": str(args.ca_feature_input),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": int(len(frame)),
        "t90_non_null_count": int(len(target_rows)),
        "target_counts": target_counts(target_rows),
        "ir_features_evaluated": ir_features,
        "per_ir_feature_diagnostics": per_diag,
        "ir_bin_analysis_summary": bin_summaries,
        "calcium_to_ir_diagnostics": ca_ir,
        "descriptive_mediation_check": mediation,
        "incremental_value_test": {
            "rows": inc_rows,
            "summary": best_info,
            "time_split": "first 80% by time train, last 20% by time test",
            "calcium_baseline_features": calcium_features,
        },
        "best_ir_feature": best_ir,
        "recommended_next_step": recommendation,
        "warnings": warnings,
        "assumptions": [
            "Outlet IR is not treated as a direct T90 measurement.",
            "Only output_ir_corrected and derived output_ir_corrected_* features are evaluated.",
            "No output.csv middle columns are used.",
            "Incremental tests compare calcium-only features against calcium plus exactly one IR feature.",
            "No generic final T90 model or calcium policy rewrite is trained in this script.",
        ],
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat_rows).to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    args.bin_output.parent.mkdir(parents=True, exist_ok=True)
    (pd.concat(bin_tables, ignore_index=True) if bin_tables else pd.DataFrame()).to_csv(
        args.bin_output, index=False, encoding="utf-8-sig"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(as_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")

    merge_report = load_json_if_exists(Path("data/data_clean_with_ir_report.json"))
    section = next_doc_section_number(args.doc)
    doc_appended = append_doc(args.doc, section, merge_report, report, best_ir, recommendation)
    print("Output IR proxy evaluation complete.")
    print(f"  first column used as timestamp: {merge_report.get('detected_ir_timestamp_column')}")
    print(f"  last column used as corrected IR: {merge_report.get('detected_ir_value_column')}")
    print(f"  ignored middle column count: {merge_report.get('ignored_middle_column_count')}")
    print(f"  IR non-null coverage: {merge_report.get('output_ir_non_null_rate')}")
    print(f"  best IR feature: {best_ir}")
    print(f"  calcium-to-IR pairs evaluated: {len(ca_ir)}")
    print(f"  IR-to-T90-risk best summary: {best_info}")
    print(f"  incremental-value recommendation: {recommendation}")
    print(f"  recommended_next_step: {recommendation}")
    print(f"  docs appended: {doc_appended}")


if __name__ == "__main__":
    main()
