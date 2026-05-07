from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
except Exception:  # pragma: no cover - depends on local environment
    GradientBoostingClassifier = None
    HistGradientBoostingClassifier = None
    SimpleImputer = None
    average_precision_score = None
    brier_score_loss = None
    roc_auc_score = None
    make_pipeline = None


T90_LOW = 8.20
T90_HIGH = 8.70
RANDOM_SEED = 20260507
OFFSETS = [-10, -5, 0, 5, 10, 15, 20, 30]
ONLINE_SAFE_OFFSETS = [0, 5, 10]
CALCIUM_BASELINE_FEATURES = [
    "ca_per_rubber_flow_win_60_mean",
    "ca_per_rubber_flow_lag_165",
    "ca_win_60_mean",
    "ca_lag_165",
    "ca_win_30_mean",
    "ca_win_120_mean",
    "ca_delta_30",
    "ca_delta_60",
]
IR_SOURCE_COLUMNS = [
    "output_ir_corrected",
    "output_ir_corrected_lag_0",
    "output_ir_corrected_win_5_mean",
    "output_ir_corrected_win_15_mean",
    "output_ir_corrected_win_30_mean",
    "output_ir_corrected_win_15_std",
    "output_ir_corrected_win_30_std",
    "output_ir_corrected_win_15_slope",
]
VARIANTS = [
    "value",
    "win_5_mean",
    "win_15_mean",
    "win_30_mean",
    "win_15_slope",
    "win_30_slope",
    "win_15_std",
    "win_30_std",
]
TARGETS_REQUIRED = ["y_high", "y_out_spec"]
TARGETS_OPTIONAL = ["y_low"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate small outlet-IR lag sensitivity for T90 risk.")
    parser.add_argument("--input", type=Path, default=Path("data/data_clean_with_ir.parquet"))
    parser.add_argument("--ca-feature-input", type=Path, default=Path("data/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--ir-report", type=Path, default=Path("data/output_ir_proxy_evaluation.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/output_ir_lag_sensitivity.csv"))
    parser.add_argument("--bin-output", type=Path, default=Path("data/output_ir_lag_sensitivity_bins.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/output_ir_lag_sensitivity_report.json"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--n-bins", type=int, default=5)
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


def offset_name(offset: int) -> str:
    return f"m{abs(offset)}" if offset < 0 else str(offset)


def feature_name(offset: int, variant: str) -> str:
    return f"output_ir_corrected_offset_{offset_name(offset)}_{variant}"


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


def rolling_slope(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return math.nan
    y = values[valid]
    x = np.arange(len(values), dtype=float)[valid]
    if len(np.unique(x)) < 2:
        return math.nan
    return float(np.polyfit(x, y, 1)[0])


def build_ir_timeline(frame: pd.DataFrame) -> pd.DataFrame:
    if "output_ir_corrected" not in frame.columns:
        raise ValueError("Input dataset does not contain required IR source column: output_ir_corrected")
    timeline = frame[["time", "output_ir_corrected"]].copy()
    timeline["time"] = pd.to_datetime(timeline["time"], errors="coerce")
    timeline = timeline.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last").sort_values("time")
    timeline["output_ir_corrected"] = pd.to_numeric(timeline["output_ir_corrected"], errors="coerce")
    indexed = timeline.set_index("time")
    indexed["value"] = indexed["output_ir_corrected"]
    indexed["win_5_mean"] = indexed["output_ir_corrected"].rolling("5min", min_periods=1).mean()
    indexed["win_15_mean"] = indexed["output_ir_corrected"].rolling("15min", min_periods=1).mean()
    indexed["win_30_mean"] = indexed["output_ir_corrected"].rolling("30min", min_periods=1).mean()
    indexed["win_15_std"] = indexed["output_ir_corrected"].rolling("15min", min_periods=2).std()
    indexed["win_30_std"] = indexed["output_ir_corrected"].rolling("30min", min_periods=2).std()
    indexed["win_15_slope"] = indexed["output_ir_corrected"].rolling("15min", min_periods=2).apply(rolling_slope, raw=True)
    indexed["win_30_slope"] = indexed["output_ir_corrected"].rolling("30min", min_periods=2).apply(rolling_slope, raw=True)
    return indexed.reset_index()[["time"] + VARIANTS]


def attach_offset_features(supervised: pd.DataFrame, ir_timeline: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = supervised.copy()
    created = []
    for offset in OFFSETS:
        target_time = result[["time"]].copy()
        target_time["ir_lookup_time"] = target_time["time"] - pd.to_timedelta(offset, unit="m")
        merged = target_time.merge(ir_timeline, left_on="ir_lookup_time", right_on="time", how="left", suffixes=("", "_ir"))
        for variant in VARIANTS:
            name = feature_name(offset, variant)
            result[name] = pd.to_numeric(merged[variant], errors="coerce")
            created.append(name)
    return result, created


def make_quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    clean = numeric.dropna()
    if len(clean) < 2 or clean.nunique() < 2:
        return result
    for bins in range(min(n_bins, clean.nunique()), 1, -1):
        try:
            labels = pd.qcut(clean, q=bins, labels=False, duplicates="drop")
            if labels.nunique(dropna=True) >= 2:
                result.loc[labels.index] = labels.astype("Int64")
                return result
        except ValueError:
            continue
    ranks = clean.rank(method="first")
    labels = pd.qcut(ranks, q=min(n_bins, len(clean)), labels=False, duplicates="drop")
    result.loc[labels.index] = labels.astype("Int64")
    return result


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    work = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(work) < 3 or work["x"].nunique() < 2 or work["y"].nunique() < 2:
        return math.nan
    return float(work["x"].corr(work["y"], method=method))


def relationship_diagnostics(data: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    rows = []
    bin_rows = []
    summary: dict[str, dict[str, object]] = {}
    for column in feature_columns:
        parsed = parse_feature_meta(column)
        values = pd.to_numeric(data[column], errors="coerce")
        usable = int(values.notna().sum())
        missing = int(values.isna().sum())
        diagnostics = {
            "offset_minutes": parsed["offset_minutes"],
            "online_safe": parsed["online_safe"],
            "ir_feature_variant": parsed["ir_feature_variant"],
            "usable_sample_count": usable,
            "missing_count": missing,
            "missing_rate": float(missing / len(data)) if len(data) else math.nan,
        }
        corr_targets = ["t90", "y_ok", "y_low", "y_high", "y_out_spec"]
        for target in corr_targets:
            spearman = safe_corr(values, data[target], "spearman")
            metric_name = f"spearman_corr_with_{target}"
            diagnostics[metric_name] = spearman
            rows.append(make_flat_row("correlation", parsed, target, "spearman", spearman, usable, "IR lag-to-risk relationship"))
            if target == "t90":
                pearson = safe_corr(values, data[target], "pearson")
                diagnostics["pearson_corr_with_t90"] = pearson
                rows.append(make_flat_row("correlation", parsed, target, "pearson", pearson, usable, "IR lag-to-T90 linear check"))
        work = data[["t90", "y_ok", "y_low", "y_high", "y_out_spec", column]].dropna(subset=[column]).copy()
        work["ir_bin"] = make_quantile_bins(work[column], 5)
        bin_counts = []
        high_rates = []
        out_rates = []
        for bin_id, group in work.dropna(subset=["ir_bin"]).groupby("ir_bin", sort=True):
            ir_values = pd.to_numeric(group[column], errors="coerce")
            bin_counts.append(len(group))
            high_rates.append(float(group["y_high"].mean()))
            out_rates.append(float(group["y_out_spec"].mean()))
            bin_rows.append(
                {
                    "offset_minutes": parsed["offset_minutes"],
                    "online_safe": parsed["online_safe"],
                    "ir_feature_variant": parsed["ir_feature_variant"],
                    "bin_id": int(bin_id),
                    "bin_label": f"bin_{int(bin_id)}",
                    "sample_count": int(len(group)),
                    "ir_min": float(ir_values.min()),
                    "ir_max": float(ir_values.max()),
                    "ir_mean": float(ir_values.mean()),
                    "ir_median": float(ir_values.median()),
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
            )
        high_spread = float(max(high_rates) - min(high_rates)) if len(high_rates) >= 2 else math.nan
        out_spread = float(max(out_rates) - min(out_rates)) if len(out_rates) >= 2 else math.nan
        min_bin_count = int(min(bin_counts)) if bin_counts else 0
        diagnostics["high_rate_spread"] = high_spread
        diagnostics["out_spec_rate_spread"] = out_spread
        diagnostics["min_bin_sample_count"] = min_bin_count
        rows.append(make_flat_row("bin_spread", parsed, "y_high", "high_rate_spread", high_spread, usable, "Quantile-bin high-T90 spread"))
        rows.append(make_flat_row("bin_spread", parsed, "y_out_spec", "out_spec_rate_spread", out_spread, usable, "Quantile-bin out-spec spread"))
        summary[column] = diagnostics
    return pd.DataFrame(rows), pd.DataFrame(bin_rows), summary


def parse_feature_meta(column: str) -> dict[str, object]:
    prefix = "output_ir_corrected_offset_"
    rest = column[len(prefix):]
    parts = rest.split("_")
    offset_token = parts[0]
    if offset_token.startswith("m"):
        offset = -int(offset_token[1:])
        variant = "_".join(parts[1:])
    else:
        offset = int(offset_token)
        variant = "_".join(parts[1:])
    return {
        "offset_minutes": offset,
        "online_safe": offset >= 0,
        "ir_feature_variant": variant,
        "feature_column": column,
    }


def make_flat_row(diagnostic_type: str, parsed: dict[str, object], target: str, metric: str, value: float, sample_count: int, note: str) -> dict[str, object]:
    return {
        "diagnostic_type": diagnostic_type,
        "offset_minutes": parsed["offset_minutes"],
        "online_safe": parsed["online_safe"],
        "ir_feature_variant": parsed["ir_feature_variant"],
        "target": target,
        "metric": metric,
        "value": value,
        "sample_count": sample_count,
        "note": note,
    }


def load_calcium_features(path: Path, supervised: pd.DataFrame, warnings: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        warnings.append(f"Calcium feature input is missing: {path}; incremental tests will be skipped.")
        return supervised, []
    columns = pd.read_parquet(path, columns=None).columns.tolist()
    use_cols = ["time"] + [column for column in CALCIUM_BASELINE_FEATURES if column in columns]
    if len(use_cols) == 1:
        warnings.append("No calcium baseline features are available for incremental tests.")
        return supervised, []
    ca = pd.read_parquet(path, columns=use_cols)
    ca["time"] = pd.to_datetime(ca["time"], errors="coerce")
    ca = ca.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last")
    merged = supervised.merge(ca, on="time", how="left", suffixes=("", "_ca"))
    return merged, use_cols[1:]


def train_binary_model(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> dict[str, object]:
    if any(obj is None for obj in [SimpleImputer, average_precision_score, roc_auc_score, brier_score_loss, make_pipeline]):
        return {"ap": math.nan, "auc": math.nan, "brier": math.nan, "warning": "sklearn unavailable"}
    train = train.dropna(subset=[target])
    test = test.dropna(subset=[target])
    y_train = train[target].astype(int)
    y_test = test[target].astype(int)
    if len(train) < 100 or len(test) < 30 or y_train.nunique() < 2 or y_test.nunique() < 2 or y_train.sum() < 5 or y_test.sum() < 3:
        return {"ap": math.nan, "auc": math.nan, "brier": math.nan, "warning": "insufficient class support"}
    x_train = train[features]
    x_test = test[features]
    models = []
    if HistGradientBoostingClassifier is not None:
        models.append(make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(random_state=RANDOM_SEED)))
    if GradientBoostingClassifier is not None:
        models.append(make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(random_state=RANDOM_SEED)))
    last_error = None
    for model in models:
        try:
            model.fit(x_train, y_train)
            prob = model.predict_proba(x_test)[:, 1]
            return {
                "ap": float(average_precision_score(y_test, prob)),
                "auc": float(roc_auc_score(y_test, prob)),
                "brier": float(brier_score_loss(y_test, prob)),
                "warning": None,
            }
        except Exception as exc:  # pragma: no cover
            last_error = str(exc)
    return {"ap": math.nan, "auc": math.nan, "brier": math.nan, "warning": last_error or "model failed"}


def incremental_tests(data: pd.DataFrame, calcium_features: list[str], ir_features: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows = []
    summary = {feature: {"best_delta_auc": math.nan, "best_delta_ap": math.nan, "best_target": None} for feature in ir_features}
    if not calcium_features:
        return pd.DataFrame(rows), summary
    split = int(len(data) * 0.8)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()
    targets = list(TARGETS_REQUIRED)
    for target in TARGETS_OPTIONAL:
        if train[target].sum() >= 5 and test[target].sum() >= 3:
            targets.append(target)
    baseline_cache: dict[str, dict[str, object]] = {}
    for target in targets:
        baseline_cache[target] = train_binary_model(train, test, calcium_features, target)
    for feature in ir_features:
        parsed = parse_feature_meta(feature)
        for target in targets:
            base = baseline_cache[target]
            with_ir = train_binary_model(train, test, calcium_features + [feature], target)
            delta_ap = with_ir["ap"] - base["ap"] if np.isfinite(with_ir["ap"]) and np.isfinite(base["ap"]) else math.nan
            delta_auc = with_ir["auc"] - base["auc"] if np.isfinite(with_ir["auc"]) and np.isfinite(base["auc"]) else math.nan
            delta_brier = with_ir["brier"] - base["brier"] if np.isfinite(with_ir["brier"]) and np.isfinite(base["brier"]) else math.nan
            metric_values = {
                "baseline_ap": base["ap"],
                "with_ir_ap": with_ir["ap"],
                "delta_ap": delta_ap,
                "baseline_auc": base["auc"],
                "with_ir_auc": with_ir["auc"],
                "delta_auc": delta_auc,
                "baseline_brier": base["brier"],
                "with_ir_brier": with_ir["brier"],
                "delta_brier": delta_brier,
                "train_sample_count": len(train),
                "test_sample_count": len(test),
            }
            for metric, value in metric_values.items():
                rows.append(make_flat_row("incremental_value", parsed, target, metric, value, len(test), "Calcium-only vs calcium + one IR lag feature screening"))
            if target in TARGETS_REQUIRED:
                current_best_auc = summary[feature]["best_delta_auc"]
                current_best_ap = summary[feature]["best_delta_ap"]
                if not np.isfinite(current_best_auc) or (np.isfinite(delta_auc) and delta_auc > current_best_auc):
                    summary[feature]["best_delta_auc"] = delta_auc
                    summary[feature]["best_target"] = target
                if not np.isfinite(current_best_ap) or (np.isfinite(delta_ap) and delta_ap > current_best_ap):
                    summary[feature]["best_delta_ap"] = delta_ap
    return pd.DataFrame(rows), summary


def alignment_score(diag: dict[str, object], inc: dict[str, object]) -> float:
    rel = max(
        abs(float(diag.get("spearman_corr_with_y_high") or 0.0)),
        abs(float(diag.get("spearman_corr_with_y_out_spec") or 0.0)),
        float(diag.get("high_rate_spread") or 0.0),
        float(diag.get("out_spec_rate_spread") or 0.0),
    )
    inc_score = max(float(inc.get("best_delta_auc") or 0.0), float(inc.get("best_delta_ap") or 0.0), 0.0)
    support_bonus = min(float(diag.get("usable_sample_count") or 0.0) / 2000.0, 1.0) * 0.01
    stable_bonus = 0.01 if int(diag.get("min_bin_sample_count") or 0) >= 30 else 0.0
    return rel + inc_score + support_bonus + stable_bonus


def select_alignment(
    diagnostics: dict[str, dict[str, object]],
    incremental: dict[str, dict[str, object]],
    online_only: bool,
) -> dict[str, object] | None:
    candidates = []
    for feature, diag in diagnostics.items():
        if online_only and not bool(diag["online_safe"]):
            continue
        score = alignment_score(diag, incremental.get(feature, {}))
        relation = (
            abs(float(diag.get("spearman_corr_with_y_high") or 0.0)) >= 0.08
            or abs(float(diag.get("spearman_corr_with_y_out_spec") or 0.0)) >= 0.08
            or float(diag.get("high_rate_spread") or 0.0) >= 0.05
            or float(diag.get("out_spec_rate_spread") or 0.0) >= 0.05
        )
        inc = incremental.get(feature, {})
        incremental_signal = (
            float(inc.get("best_delta_auc") or 0.0) >= 0.03
            or float(inc.get("best_delta_ap") or 0.0) >= 0.03
        )
        candidates.append(
            {
                **diag,
                "feature_column": feature,
                "best_delta_auc": inc.get("best_delta_auc"),
                "best_delta_ap": inc.get("best_delta_ap"),
                "best_incremental_target": inc.get("best_target"),
                "meaningful_relation": bool(relation),
                "incremental_signal": bool(incremental_signal),
                "stable_bin_support": int(diag.get("min_bin_sample_count") or 0) >= 30,
                "selection_score": score,
            }
        )
    if not candidates:
        return None
    eligible = [c for c in candidates if c["usable_sample_count"] >= 500 and c["meaningful_relation"] and c["incremental_signal"] and c["stable_bin_support"]]
    pool = eligible if eligible else candidates
    pool = sorted(
        pool,
        key=lambda x: (
            x["usable_sample_count"] >= 500,
            x["meaningful_relation"],
            x["incremental_signal"],
            x["stable_bin_support"],
            x["selection_score"],
            -abs(int(x["offset_minutes"])),
        ),
        reverse=True,
    )
    return pool[0]


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
    title = section_title(doc_path, 17, "出口 IR 小时滞敏感性评估")
    online = report["best_online_safe_ir_alignment"]
    diagnostic = report["best_diagnostic_ir_alignment"]
    lines = [
        "",
        title,
        "",
        "本阶段用于系统比较出口 IR 与 LIMS T90 标签之间的小时间滞对齐。此前 IR 主要采用同时间对齐、lag_0 以及 5/15/30 分钟尾随窗口；由于出口 IR 位于产品出口附近，理论上不应再引入上游约 3 小时的大滞后，本次重点检查 0、5、10 分钟历史对齐。",
        "",
        "### 方法",
        f"- 测试 offset：{report['offsets_tested']}。",
        "- offset >= 0 为在线安全历史 IR；offset < 0 只用于诊断时间戳错位，不用于在线策略。",
        f"- 输入文件：`{report['input_path']}`、`{report['ca_feature_input_path']}`、`{report['ir_report_path']}`。",
        f"- 输出文件：`{report['output_csv_path']}`、`{report['bin_output_path']}`、`{report['report_path']}`。",
        "",
        "### 结果",
        f"- 最佳在线安全 IR 对齐：{online}。",
        f"- 最佳诊断 IR 对齐：{diagnostic}。",
        f"- timestamp_mismatch_possible：{report['timestamp_mismatch_possible']}。",
        f"- recommended_next_step：`{report['recommended_next_step']}`。",
        "",
        "### 判断",
        "- IR 仍被视为下游质量状态代理、上下文变量或交互候选，不被视为 T90 的直接测量替代。",
        "- 若在线安全 offset 的增量价值稳定，可进入后续关系发现或规则分析；若负 offset 明显更强，应先复核时间戳定义。",
        "- 本阶段不训练生产模型、不生成控制规则、不推荐自动控制或影子试验。",
        "",
    ]
    with doc_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    assumptions = [
        "Outlet IR is a downstream quality-state proxy, not a direct T90 measurement.",
        "Negative offsets are diagnostic-only and are not online-safe.",
        "Incremental models are alignment screening tools only, not production T90 models.",
        "No IR values are imputed before correlation or binning.",
    ]
    ir_report = load_json(args.ir_report)
    if not args.input.exists():
        raise FileNotFoundError(f"Input parquet does not exist: {args.input}")
    raw_columns = pd.read_parquet(args.input, columns=None).columns.tolist()
    needed = ["time", "t90", "y_ok", "y_low", "y_high", "y_out_spec"] + [c for c in IR_SOURCE_COLUMNS if c in raw_columns]
    needed = [c for c in needed if c in raw_columns]
    frame = pd.read_parquet(args.input, columns=needed)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    if frame["time"].isna().any():
        raise ValueError("Input dataset contains invalid time values.")
    frame = ensure_targets(frame)
    frame = frame.sort_values("time").reset_index(drop=True)
    ir_timeline = build_ir_timeline(frame)
    supervised = frame[frame["t90"].notna()].copy().sort_values("time").reset_index(drop=True)
    supervised, created_ir_features = attach_offset_features(supervised, ir_timeline)
    supervised, calcium_features = load_calcium_features(args.ca_feature_input, supervised, warnings)

    flat_relationship, bins, per_feature = relationship_diagnostics(supervised, created_ir_features)
    incremental_rows, incremental_summary = incremental_tests(supervised, calcium_features, created_ir_features)
    flat = pd.concat([flat_relationship, incremental_rows], ignore_index=True)
    write_csv(args.output_csv, flat)
    write_csv(args.bin_output, bins)

    best_online = select_alignment(per_feature, incremental_summary, online_only=True)
    best_diagnostic = select_alignment(per_feature, incremental_summary, online_only=False)
    best_online_score = float(best_online["selection_score"]) if best_online else -math.inf
    best_diag_score = float(best_diagnostic["selection_score"]) if best_diagnostic else -math.inf
    timestamp_mismatch_possible = bool(
        best_diagnostic
        and int(best_diagnostic["offset_minutes"]) < 0
        and best_diag_score >= best_online_score + 0.03
    )
    if timestamp_mismatch_possible:
        recommended_next_step = "review_timestamp_definition_before_using_ir"
    elif best_online and best_online["usable_sample_count"] >= 500 and best_online["meaningful_relation"] and best_online["incremental_signal"]:
        recommended_next_step = "use_best_online_safe_ir_lag_in_relationship_discovery"
    elif best_online and best_online["usable_sample_count"] >= 500 and best_online["meaningful_relation"]:
        recommended_next_step = "keep_ir_lag0_monitoring_only"
    else:
        recommended_next_step = "ir_not_useful_for_t90_risk"

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(args.input),
        "ca_feature_input_path": str(args.ca_feature_input),
        "ir_report_path": str(args.ir_report),
        "previous_ir_best_feature": ir_report.get("best_ir_feature"),
        "output_csv_path": str(args.output_csv),
        "bin_output_path": str(args.bin_output),
        "report_path": str(args.report),
        "row_count": int(len(frame)),
        "t90_non_null_count": int(len(supervised)),
        "target_counts": {target: supervised[target].value_counts(dropna=False).to_dict() for target in ["y_ok", "y_low", "y_high", "y_out_spec"]},
        "offsets_tested": OFFSETS,
        "online_safe_offsets": ONLINE_SAFE_OFFSETS,
        "diagnostic_only_offsets": [offset for offset in OFFSETS if offset < 0],
        "per_offset_feature_diagnostics": per_feature,
        "bin_spread_summary": {
            feature: {
                "high_rate_spread": details.get("high_rate_spread"),
                "out_spec_rate_spread": details.get("out_spec_rate_spread"),
                "min_bin_sample_count": details.get("min_bin_sample_count"),
            }
            for feature, details in per_feature.items()
        },
        "incremental_value_summary": incremental_summary,
        "best_online_safe_ir_alignment": best_online,
        "best_diagnostic_ir_alignment": best_diagnostic,
        "timestamp_mismatch_possible": timestamp_mismatch_possible,
        "warnings": warnings,
        "assumptions": assumptions,
        "recommended_next_step": recommended_next_step,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(report), handle, ensure_ascii=False, indent=2)
    append_docs(args.doc, report)

    best_spread = max(
        [
            float(v.get("high_rate_spread") or 0.0)
            for v in per_feature.values()
        ]
        + [
            float(v.get("out_spec_rate_spread") or 0.0)
            for v in per_feature.values()
        ]
    )
    best_delta_auc = max(float(v.get("best_delta_auc") or 0.0) for v in incremental_summary.values())
    best_delta_ap = max(float(v.get("best_delta_ap") or 0.0) for v in incremental_summary.values())
    print("IR lag sensitivity summary")
    print(f"Offsets tested: {OFFSETS}")
    print(f"Best online-safe IR alignment: {best_online}")
    print(f"Best diagnostic IR alignment: {best_diagnostic}")
    print(f"timestamp_mismatch_possible: {timestamp_mismatch_possible}")
    print(f"Best y_high/y_out_spec spread: {best_spread}")
    print(f"Best incremental delta_auc: {best_delta_auc}")
    print(f"Best incremental delta_ap: {best_delta_ap}")
    print(f"Recommended next step: {recommended_next_step}")
    print(f"Documentation appended: {args.doc}")


if __name__ == "__main__":
    main()
