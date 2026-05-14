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


RANDOM_SEED = 20260508
PRIMARY_DOSE = "ca_per_rubber_flow_win_60_mean"
T90_COL = "t90"
REGIME_FEATURES = [
    "bromine_feed_win_60_mean",
    "rubber_flow_2_win_60_mean",
    "tank_rubber_conc_win_60_mean",
    "esbo_feed_win_60_mean",
    "neutral_alkali_feed_win_60_mean",
    "r510a_temp_win_60_mean",
    "r511a_temp_win_60_mean",
    "r512a_temp_win_60_mean",
    "r513_temp_win_60_mean",
    "r514_temp_win_60_mean",
    "output_ir_corrected_offset_20_win_15_std",
]
PAIR_REGIMES = [
    ("bromine_feed_win_60_mean", "rubber_flow_2_win_60_mean"),
    ("bromine_feed_win_60_mean", "tank_rubber_conc_win_60_mean"),
    ("bromine_feed_win_60_mean", "r514_temp_win_60_mean"),
    ("rubber_flow_2_win_60_mean", "r514_temp_win_60_mean"),
]


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
    parser = argparse.ArgumentParser(description="Analyze calcium-T90 threshold relationship within process regimes.")
    parser.add_argument("--input", type=Path, default=Path("runs/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--global-threshold-report", type=Path, default=Path("runs/ca_t90_threshold_relation/ca_t90_threshold_relation_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/regime_specific_ca_t90_thresholds"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
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


def find_input(path: Path) -> Path:
    if path.exists():
        return path
    if Path("runs").exists():
        matches = sorted(Path("runs").rglob("t90_ca_feature_dataset.parquet"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Input not found and no recursive match available: {path}")


def load_global_report(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    matches = sorted(Path("runs").rglob(path.name)) if Path("runs").exists() else []
    if matches:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    return {}


def ensure_targets(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data[T90_COL] = pd.to_numeric(data[T90_COL], errors="coerce")
    if "y_ok" not in data.columns:
        data["y_ok"] = ((data[T90_COL] >= 8.20) & (data[T90_COL] <= 8.70)).astype(int)
    if "y_low" not in data.columns:
        data["y_low"] = (data[T90_COL] < 8.20).astype(int)
    if "y_high" not in data.columns:
        data["y_high"] = (data[T90_COL] > 8.70).astype(int)
    if "y_out_spec" not in data.columns:
        data["y_out_spec"] = ((data[T90_COL] < 8.20) | (data[T90_COL] > 8.70)).astype(int)
    return data


def corr_pair(x: pd.Series, y: pd.Series, method: str = "spearman") -> tuple[float | None, float | None, int]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 5 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None, None, len(frame)
    try:
        from scipy import stats

        res = stats.spearmanr(frame["x"], frame["y"]) if method == "spearman" else stats.pearsonr(frame["x"], frame["y"])
        return float(res.statistic), float(res.pvalue), len(frame)
    except Exception:
        return float(frame["x"].corr(frame["y"], method=method)), None, len(frame)


def fit_linear(y: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    return beta, float(np.sum((y - pred) ** 2))


def piecewise_search(df: pd.DataFrame) -> pd.DataFrame:
    data = df[[PRIMARY_DOSE, T90_COL, "y_high"]].dropna()
    if len(data) < 80 or data[PRIMARY_DOSE].nunique() < 10:
        return pd.DataFrame()
    x = data[PRIMARY_DOSE].to_numpy(dtype=float)
    y = data[T90_COL].to_numpy(dtype=float)
    _, base_sse = fit_linear(y, np.column_stack([np.ones(len(x)), x]))
    rows = []
    for threshold in np.unique(np.quantile(x, np.linspace(0.15, 0.85, 51))):
        design = np.column_stack([np.ones(len(x)), x, np.maximum(0.0, x - threshold)])
        beta, sse = fit_linear(y, design)
        before = data[data[PRIMARY_DOSE] <= threshold]
        after = data[data[PRIMARY_DOSE] > threshold]
        n = len(data)
        aic = n * math.log(max(sse / n, 1e-12)) + 2 * 3
        base_aic = n * math.log(max(base_sse / n, 1e-12)) + 2 * 2
        rows.append(
            {
                "threshold": float(threshold),
                "sse": sse,
                "baseline_sse": base_sse,
                "delta_aic_like": float(base_aic - aic),
                "slope_before": float(beta[1]),
                "slope_after": float(beta[1] + beta[2]),
                "support_before": int(len(before)),
                "support_after": int(len(after)),
                "high_rate_before": float(before["y_high"].mean()) if len(before) else None,
                "high_rate_after": float(after["y_high"].mean()) if len(after) else None,
                "high_rate_delta": float(after["y_high"].mean() - before["y_high"].mean()) if len(before) and len(after) else None,
            }
        )
    return pd.DataFrame(rows)


def best_threshold(df: pd.DataFrame) -> dict[str, Any] | None:
    candidates = piecewise_search(df)
    if candidates.empty:
        return None
    valid = candidates[(candidates["support_before"] >= 30) & (candidates["support_after"] >= 30)]
    if valid.empty:
        valid = candidates
    return valid.sort_values(["delta_aic_like", "high_rate_delta"], ascending=[False, False]).iloc[0].to_dict()


def tertile_bins(series: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    numeric = pd.to_numeric(series, errors="coerce")
    q33 = float(numeric.quantile(1 / 3))
    q66 = float(numeric.quantile(2 / 3))
    labels = pd.Series(pd.NA, index=series.index, dtype="object")
    labels[numeric <= q33] = "low"
    labels[(numeric > q33) & (numeric <= q66)] = "mid"
    labels[numeric > q66] = "high"
    return labels, {"q33": q33, "q66": q66}


def safe_qcut(values: pd.Series, n_bins: int = 5) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().nunique() < 2:
        return pd.Series(pd.NA, index=values.index, dtype="object")
    for q in range(min(n_bins, numeric.dropna().nunique()), 1, -1):
        try:
            return pd.qcut(numeric, q=q, labels=False, duplicates="drop")
        except Exception:
            continue
    return pd.Series(pd.NA, index=values.index, dtype="object")


def summarize_group(group: pd.DataFrame, regime_feature: str, regime_bin: str, global_threshold: float | None, safe_upper: float | None) -> tuple[dict[str, Any], pd.DataFrame]:
    spearman_t90, p_t90, n_t90 = corr_pair(group[PRIMARY_DOSE], group[T90_COL])
    spearman_high, p_high, n_high = corr_pair(group[PRIMARY_DOSE], group["y_high"])
    threshold = best_threshold(group) if len(group) >= 120 else None
    dose_rows = []
    dose_group = group.dropna(subset=[PRIMARY_DOSE, T90_COL]).copy()
    dose_group["dose_bin"] = safe_qcut(dose_group[PRIMARY_DOSE], 5)
    for bin_id, dfg in dose_group.dropna(subset=["dose_bin"]).groupby("dose_bin", sort=True):
        dose_rows.append(
            {
                "regime_feature": regime_feature,
                "regime_bin": regime_bin,
                "dose_bin": int(bin_id),
                "sample_count": int(len(dfg)),
                "calcium_min": float(dfg[PRIMARY_DOSE].min()),
                "calcium_max": float(dfg[PRIMARY_DOSE].max()),
                "calcium_median": float(dfg[PRIMARY_DOSE].median()),
                "t90_mean": float(dfg[T90_COL].mean()),
                "ok_rate": float(dfg["y_ok"].mean()),
                "high_rate": float(dfg["y_high"].mean()),
                "low_rate": float(dfg["y_low"].mean()),
                "out_spec_rate": float(dfg["y_out_spec"].mean()),
            }
        )
    high_delta = threshold.get("high_rate_delta") if threshold else None
    best_thr = threshold.get("threshold") if threshold else None
    flags = {
        "positive_relation_supported": bool(spearman_t90 is not None and spearman_t90 > 0.05),
        "high_calcium_high_t90_risk_supported": bool(high_delta is not None and high_delta >= 0.05),
        "threshold_supported": bool(threshold and threshold.get("delta_aic_like", 0) > 2 and high_delta is not None and high_delta >= 0.05),
        "threshold_near_global_threshold": bool(best_thr is not None and global_threshold is not None and abs(best_thr - global_threshold) <= 0.0005),
        "threshold_near_safe_band_upper": bool(best_thr is not None and safe_upper is not None and abs(best_thr - safe_upper) <= 0.0005),
        "stable_support": bool(len(group) >= 120),
    }
    row = {
        "regime_feature": regime_feature,
        "regime_bin": regime_bin,
        "sample_count": int(len(group)),
        "calcium_median": float(group[PRIMARY_DOSE].median()),
        "calcium_iqr": float(group[PRIMARY_DOSE].quantile(0.75) - group[PRIMARY_DOSE].quantile(0.25)),
        "t90_mean": float(group[T90_COL].mean()),
        "t90_median": float(group[T90_COL].median()),
        "t90_iqr": float(group[T90_COL].quantile(0.75) - group[T90_COL].quantile(0.25)),
        "ok_rate": float(group["y_ok"].mean()),
        "high_rate": float(group["y_high"].mean()),
        "low_rate": float(group["y_low"].mean()),
        "out_spec_rate": float(group["y_out_spec"].mean()),
        "spearman_ca_t90": spearman_t90,
        "spearman_ca_t90_p_value": p_t90,
        "spearman_ca_y_high": spearman_high,
        "spearman_ca_y_high_p_value": p_high,
        "best_threshold": best_thr,
        "high_rate_before_threshold": threshold.get("high_rate_before") if threshold else None,
        "high_rate_after_threshold": threshold.get("high_rate_after") if threshold else None,
        "high_rate_delta": high_delta,
        "slope_before": threshold.get("slope_before") if threshold else None,
        "slope_after": threshold.get("slope_after") if threshold else None,
        **flags,
    }
    return row, pd.DataFrame(dose_rows)


def analyze_single_regimes(df: pd.DataFrame, features: list[str], global_threshold: float | None, safe_upper: float | None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    dose_frames = []
    boundaries = {}
    for feature in features:
        labels, qs = tertile_bins(df[feature])
        boundaries[feature] = qs
        work = df.copy()
        work["regime_bin"] = labels
        for label in ["low", "mid", "high"]:
            group = work[work["regime_bin"] == label].dropna(subset=[PRIMARY_DOSE, T90_COL])
            if len(group) < 30:
                continue
            row, dose = summarize_group(group, feature, label, global_threshold, safe_upper)
            rows.append(row)
            dose_frames.append(dose)
    return pd.DataFrame(rows), pd.concat(dose_frames, ignore_index=True) if dose_frames else pd.DataFrame(), boundaries


def analyze_pair_regimes(df: pd.DataFrame, global_threshold: float | None, safe_upper: float | None) -> pd.DataFrame:
    rows = []
    for f1, f2 in PAIR_REGIMES:
        if f1 not in df.columns or f2 not in df.columns:
            continue
        labels1, _ = tertile_bins(df[f1])
        labels2, _ = tertile_bins(df[f2])
        work = df.copy()
        work["bin1"] = labels1
        work["bin2"] = labels2
        for b1 in ["low", "mid", "high"]:
            for b2 in ["low", "mid", "high"]:
                group = work[(work["bin1"] == b1) & (work["bin2"] == b2)].dropna(subset=[PRIMARY_DOSE, T90_COL])
                if len(group) < 100:
                    continue
                row, _ = summarize_group(group, f"{f1}__x__{f2}", f"{b1}__x__{b2}", global_threshold, safe_upper)
                rows.append(row)
    return pd.DataFrame(rows)


def synthesize(single: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str]:
    if single.empty:
        return {"relation_type": "insufficient_support"}, [], [], "insufficient_data"
    positive = single[single["positive_relation_supported"] == True]
    highrisk = single[single["high_calcium_high_t90_risk_supported"] == True]
    threshold = single[single["threshold_supported"] == True]
    contradictory = single[(single["sample_count"] >= 120) & (single["spearman_ca_t90"].fillna(0) < -0.05)]
    strongest = single.sort_values(["threshold_supported", "high_rate_delta", "spearman_ca_t90"], ascending=[False, False, False]).head(10)
    if len(threshold) >= 5 and len(positive) >= 10:
        relation_type = "broadly_consistent"
        next_step = "use_relation_as_supporting_evidence_for_safe_band"
    elif len(highrisk) >= 5 or len(threshold) >= 3:
        relation_type = "context_specific"
        next_step = "investigate_context_specific_thresholds"
    elif len(positive) >= 5:
        relation_type = "weak_or_unstable"
        next_step = "update_manual_review_explanation_with_high_calcium_risk"
    else:
        relation_type = "weak_or_unstable"
        next_step = "relation_weak_keep_safe_band_empirical"
    synthesis = {
        "relation_type": relation_type,
        "regime_count": int(len(single)),
        "positive_relation_regime_count": int(len(positive)),
        "high_calcium_high_t90_risk_regime_count": int(len(highrisk)),
        "threshold_evidence_regime_count": int(len(threshold)),
        "contradictory_regime_count": int(len(contradictory)),
    }
    return synthesis, strongest.to_dict(orient="records"), contradictory.to_dict(orient="records"), next_step


def make_figures(single: pd.DataFrame, dose: pd.DataFrame, figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    if not single.empty:
        pivot = single.pivot(index="regime_feature", columns="regime_bin", values="spearman_ca_t90").reindex(columns=["low", "mid", "high"])
        fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.35)))
        im = ax.imshow(pivot.fillna(0).values, aspect="auto", cmap="coolwarm", vmin=-0.4, vmax=0.4)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_title("分工况钙单耗-T90 Spearman 关系")
        fig.colorbar(im, ax=ax)
        out = figure_dir / "regime_ca_t90_spearman_heatmap.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        generated.append(str(out))

        plot = single.sort_values("high_rate_delta", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = plot["regime_feature"] + ":" + plot["regime_bin"]
        ax.barh(labels[::-1], plot["high_rate_delta"].fillna(0)[::-1])
        ax.set_title("分工况阈值前后高 T90 风险变化")
        ax.set_xlabel("high_rate_delta")
        out = figure_dir / "regime_high_rate_delta_by_feature.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        generated.append(str(out))

        plot = single.dropna(subset=["best_threshold"]).sort_values("best_threshold")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(np.arange(len(plot)), plot["best_threshold"], c=plot["threshold_supported"].astype(int), cmap="coolwarm")
        ax.set_title("分工况钙单耗阈值候选")
        ax.set_xlabel("工况候选序号")
        ax.set_ylabel("阈值")
        out = figure_dir / "regime_threshold_candidates.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        generated.append(str(out))

    if not dose.empty:
        top = single.sort_values(["threshold_supported", "high_rate_delta"], ascending=[False, False]).head(4)
        fig, ax = plt.subplots(figsize=(10, 6))
        for _, row in top.iterrows():
            subset = dose[(dose["regime_feature"] == row["regime_feature"]) & (dose["regime_bin"] == row["regime_bin"])]
            if subset.empty:
                continue
            ax.plot(subset["dose_bin"], subset["high_rate"], marker="o", label=f"{row['regime_feature']}:{row['regime_bin']}")
        ax.set_title("重点工况钙单耗分箱高 T90 风险")
        ax.set_xlabel("dose bin")
        ax.set_ylabel("high_rate")
        ax.legend(fontsize=7)
        out = figure_dir / "regime_dose_response_top_regimes.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        generated.append(str(out))
    return generated


def append_doc(path: Path, report: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    section_no = 34
    while f"## {section_no}." in existing:
        section_no += 1
    synth = report.get("global_synthesis", {})
    strongest = report.get("strongest_supporting_regimes", [])[:5]
    contradictory = report.get("contradictory_regimes", [])[:5]
    text = f"""

## {section_no}. 分工况钙单耗-T90 阈值关系复验

本阶段在 Stage 31/32 之后进行：上一轮全局阈值结果显示钙单耗与高 T90 风险有中等强度历史证据，但聚类结果不稳，因此本阶段不依赖聚类，而用关键工况变量的 low/mid/high 三分位工况复验钙单耗-T90 关系。

全局阈值回顾：阈值候选 {report.get('global_threshold_candidate')}，安全带上沿中位数 {report.get('safe_band_upper_median')}。

分工况综合：{synth}。

最强支持工况（前 5）：{strongest}。

矛盾或反向工况（前 5）：{contradictory}。

当前判断：`{synth.get('relation_type')}`。推荐下一步：`{report.get('recommended_next_step')}`。

局限性：该分析仍为离线历史关系，不是因果证明；三分位工况存在样本稀疏；IR-lag 只作为可选工况变量；本阶段不产生自动控制建议。
"""
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    args = parse_args()
    configure_chinese_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    input_path = find_input(args.input)
    global_report = load_global_report(args.global_threshold_report)
    global_threshold = (global_report.get("best_threshold_candidate") or {}).get("threshold")
    safe_upper = (global_report.get("safe_band_consistency") or {}).get("recommended_ca_consumption_max_median")

    df = pd.read_parquet(input_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values("time")
    df = ensure_targets(df)
    usable = df.dropna(subset=[T90_COL, PRIMARY_DOSE]).copy()
    features = [f for f in REGIME_FEATURES if f in usable.columns and usable[f].notna().sum() >= 90]
    single, dose, boundaries = analyze_single_regimes(usable, features, global_threshold, safe_upper)
    pair = analyze_pair_regimes(usable, global_threshold, safe_upper)
    synthesis, strongest, contradictory, next_step = synthesize(single)
    figures = make_figures(single, dose, args.figure_dir)

    single.to_csv(args.output_dir / "regime_threshold_summary.csv", index=False, encoding="utf-8-sig")
    pair.to_csv(args.output_dir / "two_variable_regime_threshold_summary.csv", index=False, encoding="utf-8-sig")
    dose.to_csv(args.output_dir / "regime_dose_bin_response.csv", index=False, encoding="utf-8-sig")
    single.to_csv(args.table_dir / "regime_specific_ca_t90_threshold_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "global_threshold_report_path": str(args.global_threshold_report),
        "sample_count": int(len(df[df[T90_COL].notna()])),
        "usable_count": int(len(usable)),
        "primary_dose_feature": PRIMARY_DOSE,
        "regime_features": features,
        "regime_boundaries": boundaries,
        "global_threshold_candidate": global_threshold,
        "safe_band_upper_median": safe_upper,
        "per_regime_summary": single.to_dict(orient="records"),
        "two_variable_regime_summary": pair.to_dict(orient="records"),
        "strongest_supporting_regimes": strongest,
        "contradictory_regimes": contradictory,
        "global_synthesis": synthesis,
        "generated_figures": figures,
        "warnings": [],
        "limitations": ["offline_historical_analysis", "not_causal_proof", "no_automatic_control", "regime_sample_sparsity"],
        "recommended_next_step": next_step,
    }
    write_json(args.output_dir / "regime_specific_ca_t90_threshold_report.json", report)
    append_doc(args.doc, report)

    print("Regime-specific calcium-T90 threshold summary")
    print(f"usable_count: {len(usable)}")
    print(f"regime_features: {features}")
    print(f"global_synthesis: {synthesis}")
    print(f"recommended_next_step: {next_step}")
    print(f"docs_appended: {args.doc}")
    print("No generated outputs were written under data/.")


if __name__ == "__main__":
    main()
