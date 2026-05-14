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
TARGETS = ["y_ok", "y_low", "y_high", "y_out_spec"]
CONTEXT_FEATURES = [
    "rubber_flow_2_win_60_mean",
    "bromine_feed_win_60_mean",
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
    parser = argparse.ArgumentParser(description="Analyze nonlinear calcium-consumption and T90 threshold relationship.")
    parser.add_argument("--input", type=Path, default=Path("runs/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ca_t90_threshold_relation"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--n-bootstrap", type=int, default=200)
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
    for root in [Path("runs"), Path(".")]:
        if root.exists():
            matches = sorted(root.rglob("t90_ca_feature_dataset.parquet"))
            if matches:
                return matches[0]
    required = {"time", T90_COL, PRIMARY_DOSE}
    for root in [Path("runs"), Path(".")]:
        if not root.exists():
            continue
        for candidate in sorted(root.rglob("*.parquet")):
            try:
                cols = set(pd.read_parquet(candidate, columns=[]).columns)
            except Exception:
                try:
                    cols = set(pd.read_parquet(candidate).columns)
                except Exception:
                    continue
            if required.issubset(cols):
                return candidate
    raise FileNotFoundError(f"No suitable input found for required columns: {sorted(required)}")


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


def quantiles(series: pd.Series) -> dict[str, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {k: None for k in ["min", "q25", "median", "q75", "max", "mean"]}
    return {
        "min": float(s.min()),
        "q25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q75": float(s.quantile(0.75)),
        "max": float(s.max()),
        "mean": float(s.mean()),
    }


def corr_pair(x: pd.Series, y: pd.Series, method: str) -> tuple[float | None, float | None, int]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None, None, len(frame)
    try:
        from scipy import stats

        if method == "pearson":
            res = stats.pearsonr(frame["x"], frame["y"])
        else:
            res = stats.spearmanr(frame["x"], frame["y"])
        return float(res.statistic), float(res.pvalue), len(frame)
    except Exception:
        return float(frame["x"].corr(frame["y"], method=method)), None, len(frame)


def basic_correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    summary: dict[str, Any] = {}
    for target in [T90_COL] + TARGETS:
        summary[target] = {}
        for method in ["pearson", "spearman"]:
            corr, p_value, n = corr_pair(df[PRIMARY_DOSE], df[target], method)
            rows.append(
                {
                    "x": PRIMARY_DOSE,
                    "y": target,
                    "method": method,
                    "correlation": corr,
                    "p_value": p_value,
                    "sample_count": n,
                }
            )
            summary[target][method] = {"correlation": corr, "p_value": p_value, "sample_count": n}
    return pd.DataFrame(rows), summary


def safe_qcut(values: pd.Series, n_bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.nunique() < 2:
        return pd.Series([pd.NA] * len(values), index=values.index, dtype="object")
    for bins in range(min(n_bins, valid.nunique()), 1, -1):
        try:
            return pd.qcut(numeric, q=bins, labels=False, duplicates="drop")
        except Exception:
            continue
    ranked = numeric.rank(method="first")
    try:
        return pd.qcut(ranked, q=min(n_bins, ranked.dropna().nunique()), labels=False, duplicates="drop")
    except Exception:
        return pd.Series([pd.NA] * len(values), index=values.index, dtype="object")


def dose_bin_response(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    summaries: dict[str, Any] = {}
    for n_bins in [5, 7, 10]:
        data = df.dropna(subset=[PRIMARY_DOSE, T90_COL]).copy()
        data["dose_bin_id"] = safe_qcut(data[PRIMARY_DOSE], n_bins)
        data = data.dropna(subset=["dose_bin_id"])
        if data.empty:
            continue
        data["dose_bin_id"] = data["dose_bin_id"].astype(int)
        bin_rows = []
        for bin_id, group in data.groupby("dose_bin_id", sort=True):
            row = {
                "n_bins": n_bins,
                "dose_bin": int(bin_id),
                "sample_count": int(len(group)),
                "calcium_min": float(group[PRIMARY_DOSE].min()),
                "calcium_max": float(group[PRIMARY_DOSE].max()),
                "calcium_median": float(group[PRIMARY_DOSE].median()),
                "t90_mean": float(group[T90_COL].mean()),
                "t90_median": float(group[T90_COL].median()),
                "t90_q25": float(group[T90_COL].quantile(0.25)),
                "t90_q75": float(group[T90_COL].quantile(0.75)),
                "ok_rate": float(group["y_ok"].mean()),
                "high_rate": float(group["y_high"].mean()),
                "low_rate": float(group["y_low"].mean()),
                "out_spec_rate": float(group["y_out_spec"].mean()),
            }
            bin_rows.append(row)
            rows.append(row)
        bdf = pd.DataFrame(bin_rows)
        if not bdf.empty:
            best_ok = bdf.sort_values(["ok_rate", "sample_count"], ascending=[False, False]).iloc[0].to_dict()
            highest_high = bdf.sort_values(["high_rate", "sample_count"], ascending=[False, False]).iloc[0].to_dict()
            last = bdf.sort_values("dose_bin").iloc[-1]
            lower = bdf.sort_values("dose_bin").iloc[:-1]
            summaries[str(n_bins)] = {
                "best_ok_rate_bin": best_ok,
                "highest_high_rate_bin": highest_high,
                "highest_calcium_bin_high_rate": float(last["high_rate"]),
                "non_highest_bins_high_rate_mean": float(lower["high_rate"].mean()) if len(lower) else None,
                "highest_bin_elevated_high_rate": bool(len(lower) and last["high_rate"] >= lower["high_rate"].mean() + 0.05),
            }
    return pd.DataFrame(rows), summaries


def fit_linear(y: np.ndarray, x_design: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    pred = x_design @ beta
    sse = float(np.sum((y - pred) ** 2))
    return beta, pred, sse


def piecewise_search(df: pd.DataFrame, y_col: str = T90_COL) -> pd.DataFrame:
    data = df[[PRIMARY_DOSE, y_col, "y_high"]].dropna().copy()
    x = data[PRIMARY_DOSE].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    if len(data) < 50:
        return pd.DataFrame()
    baseline_design = np.column_stack([np.ones(len(x)), x])
    _, _, baseline_sse = fit_linear(y, baseline_design)
    thresholds = np.unique(np.quantile(x, np.linspace(0.15, 0.85, 71)))
    rows = []
    for threshold in thresholds:
        hinge = np.maximum(0.0, x - threshold)
        design = np.column_stack([np.ones(len(x)), x, hinge])
        beta, _, sse = fit_linear(y, design)
        n = len(x)
        aic_like = n * math.log(max(sse / n, 1e-12)) + 2 * design.shape[1]
        baseline_aic = n * math.log(max(baseline_sse / n, 1e-12)) + 2 * baseline_design.shape[1]
        before = data.loc[data[PRIMARY_DOSE] <= threshold]
        after = data.loc[data[PRIMARY_DOSE] > threshold]
        slope_before = float(beta[1])
        slope_after = float(beta[1] + beta[2])
        rows.append(
            {
                "threshold": float(threshold),
                "sse": float(sse),
                "baseline_sse": float(baseline_sse),
                "sse_improvement": float(baseline_sse - sse),
                "aic_like": float(aic_like),
                "baseline_aic_like": float(baseline_aic),
                "delta_aic_like": float(baseline_aic - aic_like),
                "slope_before": slope_before,
                "slope_after": slope_after,
                "slope_ratio": float(slope_after / max(abs(slope_before), 1e-12)),
                "threshold_support_left": int(len(before)),
                "threshold_support_right": int(len(after)),
                "high_rate_before": float(before["y_high"].mean()) if len(before) else None,
                "high_rate_after": float(after["y_high"].mean()) if len(after) else None,
                "high_rate_delta": float(after["y_high"].mean() - before["y_high"].mean()) if len(before) and len(after) else None,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_thresholds(df: pd.DataFrame, n_bootstrap: int) -> dict[str, Any]:
    data = df[[PRIMARY_DOSE, T90_COL, "y_high"]].dropna().copy()
    if len(data) < 80 or n_bootstrap <= 0:
        return {"bootstrap_count": 0}
    rng = np.random.default_rng(RANDOM_SEED)
    selected: list[float] = []
    for _ in range(n_bootstrap):
        sample = data.iloc[rng.integers(0, len(data), len(data))]
        candidates = piecewise_search(sample)
        candidates = candidates[(candidates["threshold_support_left"] >= 30) & (candidates["threshold_support_right"] >= 30)]
        if not candidates.empty:
            selected.append(float(candidates.sort_values(["aic_like", "sse"]).iloc[0]["threshold"]))
    if not selected:
        return {"bootstrap_count": 0}
    arr = np.array(selected, dtype=float)
    return {
        "bootstrap_count": int(len(arr)),
        "threshold_median": float(np.median(arr)),
        "threshold_q25": float(np.quantile(arr, 0.25)),
        "threshold_q75": float(np.quantile(arr, 0.75)),
        "threshold_std": float(np.std(arr)),
    }


def smooth_curve(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    data = df[[PRIMARY_DOSE, y_col]].dropna().sort_values(PRIMARY_DOSE)
    if len(data) < 10:
        return pd.DataFrame(columns=[PRIMARY_DOSE, y_col + "_smooth"])
    x = data[PRIMARY_DOSE].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        sm = lowess(y, x, frac=0.25, return_sorted=True)
        return pd.DataFrame({PRIMARY_DOSE: sm[:, 0], y_col + "_smooth": sm[:, 1]})
    except Exception:
        window = max(15, int(len(data) * 0.08))
        data[y_col + "_smooth"] = data[y_col].rolling(window=window, center=True, min_periods=max(5, window // 3)).median()
        return data[[PRIMARY_DOSE, y_col + "_smooth"]].dropna()


def context_adjusted(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.Series | None]:
    rows = []
    summary: dict[str, Any] = {"context_features_used": []}
    available = [c for c in CONTEXT_FEATURES if c in df.columns and df[c].notna().sum() >= 50]
    if not available:
        return pd.DataFrame(), {"warning": "no_context_features_available"}, None
    summary["context_features_used"] = available
    data = df[[T90_COL, PRIMARY_DOSE, "y_high"] + available].dropna(subset=[T90_COL, PRIMARY_DOSE]).copy()
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        reg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
        reg.fit(data[available], data[T90_COL])
        residual = data[T90_COL] - reg.predict(data[available])
        corr, p_value, n = corr_pair(data[PRIMARY_DOSE], residual, "spearman")
        rows.append({"diagnostic": "residual_spearman", "target": "residual_t90", "metric": "spearman", "value": corr, "p_value": p_value, "sample_count": n})
        residual_df = data[[PRIMARY_DOSE, "y_high"]].copy()
        residual_df["residual_t90"] = residual
        residual_candidates = piecewise_search(residual_df.rename(columns={"residual_t90": T90_COL}))
        if not residual_candidates.empty:
            best = residual_candidates.sort_values(["aic_like", "sse"]).iloc[0].to_dict()
            summary["residual_threshold_candidate"] = best
            rows.append({"diagnostic": "residual_piecewise", "target": "residual_t90", "metric": "best_threshold", "value": best["threshold"], "p_value": None, "sample_count": len(residual_df)})
        split = int(len(data) * 0.8)
        train = data.iloc[:split]
        test = data.iloc[split:]
        y_train = train["y_high"].astype(int)
        y_test = test["y_high"].astype(int)
        if y_train.nunique() > 1 and y_test.nunique() > 1:
            base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
            plus = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
            base.fit(train[available], y_train)
            plus.fit(train[available + [PRIMARY_DOSE]], y_train)
            pred_base = base.predict_proba(test[available])[:, 1]
            pred_plus = plus.predict_proba(test[available + [PRIMARY_DOSE]])[:, 1]
            for metric_name, func in [("auc", roc_auc_score), ("average_precision", average_precision_score)]:
                b = float(func(y_test, pred_base))
                p = float(func(y_test, pred_plus))
                rows.append({"diagnostic": "context_plus_calcium_classifier", "target": "y_high", "metric": "context_only_" + metric_name, "value": b, "p_value": None, "sample_count": len(test)})
                rows.append({"diagnostic": "context_plus_calcium_classifier", "target": "y_high", "metric": "context_plus_calcium_" + metric_name, "value": p, "p_value": None, "sample_count": len(test)})
                rows.append({"diagnostic": "context_plus_calcium_classifier", "target": "y_high", "metric": "delta_" + metric_name, "value": p - b, "p_value": None, "sample_count": len(test)})
            summary["classifier_context_plus_calcium"] = {"test_sample_count": int(len(test))}
        summary["residual_spearman_with_calcium"] = {"correlation": corr, "p_value": p_value, "sample_count": n}
        return pd.DataFrame(rows), summary, residual.reindex(df.index)
    except Exception as exc:
        return pd.DataFrame(rows), {"context_features_used": available, "warning": f"sklearn_context_adjusted_analysis_skipped: {exc}"}, None


def safe_band_consistency(best_threshold: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False}
    threshold = best_threshold.get("threshold") if best_threshold else None
    for path in [Path("runs/ca_safe_band_mvp/final_monitor_dry_run.parquet"), Path("runs/ca_interval_recommender_replay.parquet")]:
        if path.exists():
            try:
                replay = pd.read_parquet(path)
                if "recommended_ca_consumption_max" in replay.columns:
                    med_max = float(pd.to_numeric(replay["recommended_ca_consumption_max"], errors="coerce").median())
                    result.update(
                        {
                            "available": True,
                            "source": str(path),
                            "recommended_ca_consumption_max_median": med_max,
                            "known_stable_safe_band": [0.02016, 0.02050],
                            "threshold_minus_safe_band_max_median": float(threshold - med_max) if threshold is not None else None,
                            "threshold_near_safe_band_upper_bound": bool(threshold is not None and abs(threshold - med_max) <= 0.0005),
                        }
                    )
                    return result
            except Exception:
                continue
    result["known_stable_safe_band"] = [0.02016, 0.02050]
    if threshold is not None:
        result["threshold_minus_known_upper_bound"] = float(threshold - 0.02050)
        result["threshold_near_safe_band_upper_bound"] = bool(abs(threshold - 0.02050) <= 0.0005)
    return result


def make_figures(df: pd.DataFrame, bins: pd.DataFrame, threshold: dict[str, Any] | None, residual: pd.Series | None, figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    smooth_t90 = smooth_curve(df, T90_COL)
    smooth_high = smooth_curve(df, "y_high")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.scatter(df[PRIMARY_DOSE], df[T90_COL], s=10, alpha=0.25, label="样本")
    if not smooth_t90.empty:
        ax1.plot(smooth_t90[PRIMARY_DOSE], smooth_t90[T90_COL + "_smooth"], color="red", linewidth=2, label="T90 平滑")
    if threshold:
        ax1.axvline(float(threshold["threshold"]), color="black", linestyle="--", label="阈值候选")
    ax1.set_title("钙单耗与 T90 平滑关系")
    ax1.set_xlabel("钙单耗")
    ax1.set_ylabel("T90")
    ax1.legend()
    out = figure_dir / "ca_t90_scatter_smooth.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    use_bins = bins[bins["n_bins"] == 10] if not bins.empty and (bins["n_bins"] == 10).any() else bins
    if not use_bins.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = use_bins["dose_bin"].astype(str)
        ax.plot(labels, use_bins["t90_mean"], marker="o", label="T90 均值")
        ax2 = ax.twinx()
        ax2.plot(labels, use_bins["ok_rate"], marker="s", color="green", label="合格率")
        ax.set_title("钙单耗分箱下的 T90 与合格率")
        ax.set_xlabel("钙单耗分箱")
        ax.set_ylabel("T90 均值")
        ax2.set_ylabel("合格率")
        lines, names = ax.get_legend_handles_labels()
        lines2, names2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, names + names2, loc="best")
        out = figure_dir / "ca_t90_dose_bin_response.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        generated.append(str(out))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(labels, use_bins["high_rate"], color="#b64040")
        ax.set_title("钙单耗分箱下的高 T90 风险")
        ax.set_xlabel("钙单耗分箱")
        ax.set_ylabel("高 T90 风险率")
        out = figure_dir / "ca_t90_high_rate_by_dose_bin.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        generated.append(str(out))

    if threshold:
        data = df[[PRIMARY_DOSE, T90_COL]].dropna().sort_values(PRIMARY_DOSE)
        x = data[PRIMARY_DOSE].to_numpy(dtype=float)
        y = data[T90_COL].to_numpy(dtype=float)
        thr = float(threshold["threshold"])
        design = np.column_stack([np.ones(len(x)), x, np.maximum(0, x - thr)])
        beta, pred, _ = fit_linear(y, design)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(x, y, s=10, alpha=0.25)
        ax.plot(x, pred, color="black", linewidth=2, label="分段拟合")
        ax.axvline(thr, color="red", linestyle="--", label="阈值候选")
        ax.set_title("钙单耗-T90 分段阈值拟合")
        ax.set_xlabel("钙单耗")
        ax.set_ylabel("T90")
        ax.legend()
        out = figure_dir / "ca_t90_piecewise_fit.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        generated.append(str(out))

    if residual is not None:
        frame = df[[PRIMARY_DOSE]].copy()
        frame["residual_t90"] = residual
        frame = frame.dropna()
        if not frame.empty:
            sm = smooth_curve(frame.rename(columns={"residual_t90": T90_COL}), T90_COL)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.scatter(frame[PRIMARY_DOSE], frame["residual_t90"], s=10, alpha=0.25)
            if not sm.empty:
                ax.plot(sm[PRIMARY_DOSE], sm[T90_COL + "_smooth"], color="red", linewidth=2)
            ax.set_title("控制工况后的钙单耗与 T90 残差关系")
            ax.set_xlabel("钙单耗")
            ax.set_ylabel("T90 残差")
            out = figure_dir / "ca_t90_context_adjusted_residual.png"
            fig.tight_layout()
            fig.savefig(out, dpi=160)
            plt.close(fig)
            generated.append(str(out))
    return generated


def append_doc(path: Path, report: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    section_no = 30
    while f"## {section_no}." in existing:
        section_no += 1
    corr = report.get("basic_correlations", {}).get(T90_COL, {}).get("spearman", {})
    best = report.get("best_threshold_candidate") or {}
    flags = report.get("decision_flags", {})
    text = f"""

## {section_no}. 钙单耗与 T90 非线性阈值关系验证

本阶段用于验证历史数据是否支持“钙单耗与 T90 存在正向、非线性阈值关系”的工艺预期，而不是预设该关系成立。输入样本数 {report.get('sample_count')}，可用样本数 {report.get('usable_count')}。

基础相关性：钙单耗与 T90 的 Spearman 相关系数为 {corr.get('correlation')}；该结果只说明历史相关关系，不构成因果证明。

分箱响应与阈值搜索：最优阈值候选为 {best.get('threshold')}，阈值前后高 T90 风险差为 {best.get('high_rate_delta')}。正向关系支持：{flags.get('positive_relation_supported')}；非线性阈值支持：{flags.get('nonlinear_threshold_supported')}；安全平台区支持：{flags.get('flat_safe_region_supported')}；高钙高 T90 风险支持：{flags.get('high_calcium_high_t90_risk_supported')}。

当前安全带一致性：{report.get('safe_band_consistency')}。

证据强度：`{report.get('evidence_strength')}`。推荐下一步：`{report.get('recommended_next_step')}`。

局限性：离线历史关系不等于因果证明；T90 为人工 LIMS 且存在约 0.1 的实际误差；工况混杂仍可能影响钙单耗与 T90 的表观关系；本阶段不产生自动控制建议。
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
    df = pd.read_parquet(input_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values("time")
    required = {"time", T90_COL, PRIMARY_DOSE}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")
    df = ensure_targets(df)
    supervised = df[df[T90_COL].notna()].copy()
    usable = supervised.dropna(subset=[PRIMARY_DOSE, T90_COL]).copy()

    corr_df, corr_summary = basic_correlations(usable)
    bins_df, bin_summary = dose_bin_response(usable)
    candidates = piecewise_search(usable)
    valid_candidates = candidates[(candidates["threshold_support_left"] >= 50) & (candidates["threshold_support_right"] >= 50)] if not candidates.empty else candidates
    best_threshold = valid_candidates.sort_values(["aic_like", "sse"]).iloc[0].to_dict() if not valid_candidates.empty else None
    threshold_stability = bootstrap_thresholds(usable, args.n_bootstrap)
    context_df, context_summary, residual = context_adjusted(usable)
    safe_band = safe_band_consistency(best_threshold)

    corr_df.to_csv(args.output_dir / "ca_t90_basic_correlations.csv", index=False, encoding="utf-8-sig")
    bins_df.to_csv(args.output_dir / "ca_t90_dose_bins.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(args.output_dir / "ca_t90_piecewise_threshold_candidates.csv", index=False, encoding="utf-8-sig")
    context_df.to_csv(args.output_dir / "ca_t90_context_adjusted_results.csv", index=False, encoding="utf-8-sig")

    figures = make_figures(usable, bins_df, best_threshold, residual, args.figure_dir)

    spearman_t90 = corr_summary.get(T90_COL, {}).get("spearman", {}).get("correlation")
    highest_bin_elevated = any(v.get("highest_bin_elevated_high_rate") for v in bin_summary.values())
    nonlinear = bool(best_threshold and best_threshold.get("delta_aic_like", 0) > 2 and best_threshold.get("slope_after", 0) > best_threshold.get("slope_before", 0))
    high_risk = bool(best_threshold and (best_threshold.get("high_rate_delta") is not None) and best_threshold.get("high_rate_delta") >= 0.05)
    middle_stable = False
    if not bins_df.empty:
        use_bins = bins_df[bins_df["n_bins"] == 10] if (bins_df["n_bins"] == 10).any() else bins_df
        middle = use_bins.sort_values("dose_bin").iloc[1:-1]
        middle_stable = bool(not middle.empty and middle["high_rate"].max() <= usable["y_high"].mean() + 0.03)
    residual_corr = context_summary.get("residual_spearman_with_calcium", {}).get("correlation")
    context_effect = bool(residual_corr is not None and residual_corr > 0)
    positive_supported = bool(spearman_t90 is not None and spearman_t90 > 0 and highest_bin_elevated)
    flags = {
        "positive_relation_supported": positive_supported,
        "nonlinear_threshold_supported": nonlinear,
        "flat_safe_region_supported": middle_stable,
        "high_calcium_high_t90_risk_supported": high_risk,
        "context_adjusted_effect_supported": context_effect,
    }
    true_count = sum(bool(v) for v in flags.values())
    if true_count >= 4:
        strength = "strong"
        next_step = "use_threshold_relation_as_supporting_evidence"
    elif true_count >= 3:
        strength = "moderate"
        next_step = "use_threshold_relation_as_supporting_evidence"
    elif positive_supported or nonlinear:
        strength = "weak"
        next_step = "investigate_context_specific_thresholds"
    elif len(usable) < 200:
        strength = "not_supported"
        next_step = "insufficient_data"
    else:
        strength = "not_supported"
        next_step = "relation_weak_keep_safe_band_empirical"

    summary_rows = [
        {"metric": "sample_count", "value": len(supervised), "interpretation_cn": "非空 T90 样本数"},
        {"metric": "usable_count", "value": len(usable), "interpretation_cn": "钙单耗与 T90 均可用样本数"},
        {"metric": "spearman_ca_t90", "value": spearman_t90, "interpretation_cn": "钙单耗与 T90 的单变量秩相关"},
        {"metric": "best_threshold_candidate", "value": best_threshold.get("threshold") if best_threshold else None, "interpretation_cn": "分段线性搜索得到的阈值候选"},
        {"metric": "high_rate_delta_after_threshold", "value": best_threshold.get("high_rate_delta") if best_threshold else None, "interpretation_cn": "阈值后高 T90 风险相对阈值前的变化"},
        {"metric": "evidence_strength", "value": strength, "interpretation_cn": "综合证据强度"},
        {"metric": "recommended_next_step", "value": next_step, "interpretation_cn": "建议下一步"},
    ]
    pd.DataFrame(summary_rows).to_csv(args.table_dir / "ca_t90_threshold_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "sample_count": int(len(supervised)),
        "usable_count": int(len(usable)),
        "primary_dose_feature": PRIMARY_DOSE,
        "t90_column": T90_COL,
        "calcium_distribution": quantiles(usable[PRIMARY_DOSE]),
        "t90_distribution": quantiles(usable[T90_COL]),
        "basic_correlations": corr_summary,
        "dose_bin_summary": bin_summary,
        "best_threshold_candidate": best_threshold,
        "threshold_stability": threshold_stability,
        "context_adjusted_summary": context_summary,
        "safe_band_consistency": safe_band,
        "decision_flags": flags,
        "evidence_strength": strength,
        "generated_figures": figures,
        "warnings": [],
        "limitations": ["offline_historical_relationship_only", "not_causal_proof", "t90_measurement_error", "context_confounding_possible", "no_automatic_control"],
        "recommended_next_step": next_step,
    }
    write_json(args.output_dir / "ca_t90_threshold_relation_report.json", report)
    append_doc(args.doc, report)

    print("Calcium-T90 threshold relation summary")
    print(f"input_path: {input_path}")
    print(f"usable_count: {len(usable)}")
    print(f"spearman_ca_t90: {spearman_t90}")
    print(f"best_threshold_candidate: {best_threshold.get('threshold') if best_threshold else None}")
    print(f"evidence_strength: {strength}")
    print(f"recommended_next_step: {next_step}")
    print(f"figures: {figures}")
    print(f"docs_appended: {args.doc}")
    print("No generated outputs were written under data/.")


if __name__ == "__main__":
    main()
