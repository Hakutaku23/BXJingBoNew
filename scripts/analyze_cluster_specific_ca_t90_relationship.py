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


PRIMARY_DOSE = "ca_per_rubber_flow_win_60_mean"
T90_COL = "t90"
CORE_11 = [
    "ca_per_rubber_flow_win_60_mean",
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
]
PROFILE_FEATURES = CORE_11
SAFE_BAND_UPPER_DEFAULT = 0.0204772882317374
SAFE_BAND_LOWER_DEFAULT = 0.02016


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
    parser = argparse.ArgumentParser(description="Analyze calcium-T90 relationships inside robust operating-regime clusters.")
    parser.add_argument("--input", type=Path, default=Path("runs/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--clustering-report", type=Path, default=Path("runs/t90_clustering_robustness_audit/t90_clustering_robustness_audit_report.json"))
    parser.add_argument("--cluster-assignments", type=Path, default=Path("runs/t90_clustering_robustness_audit/clustering_selected_assignments.parquet"))
    parser.add_argument("--regime-threshold-report", type=Path, default=Path("runs/regime_specific_ca_t90_thresholds/regime_specific_ca_t90_threshold_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cluster_specific_ca_t90_relationship"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--expected-k", type=int, default=5, help="Expected robust cluster count; use 0 to accept the selected k from the report.")
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


def resolve_file(path: Path, required: bool = True) -> Path | None:
    if path.exists():
        return path
    if Path("runs").exists():
        matches = sorted(Path("runs").rglob(path.name))
        if matches:
            return matches[0]
    if required:
        raise FileNotFoundError(f"Required file not found: {path}")
    return None


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def selected_features_from_report(report: dict[str, Any]) -> tuple[list[str], list[str]]:
    selected = report.get("selected_result") or {}
    variant = selected.get("variant")
    features = []
    for item in report.get("feature_variants_tested", []):
        if item.get("variant") == variant:
            features = list(item.get("features") or [])
            break
    dropped = [f for f in CORE_11 if f not in features]
    return features, dropped


def join_assignments(features: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    left = features.copy()
    right = assignments[["time", "cluster"]].copy()
    left["time"] = pd.to_datetime(left["time"], errors="coerce")
    right["time"] = pd.to_datetime(right["time"], errors="coerce")
    merged = left.merge(right, on="time", how="inner")
    if len(merged) != len(assignments):
        # Fall back to row order if duplicated timestamps make an exact merge lossy.
        left = left.reset_index(drop=True)
        right = assignments[["cluster"]].reset_index(drop=True)
        merged = pd.concat([left, right], axis=1)
    return ensure_targets(merged)


def corr_pair(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None, int]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 5 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None, None, len(frame)
    try:
        from scipy import stats

        res = stats.spearmanr(frame["x"], frame["y"])
        return float(res.statistic), float(res.pvalue), len(frame)
    except Exception:
        return float(frame["x"].corr(frame["y"], method="spearman")), None, len(frame)


def fit_linear(y: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    return beta, float(np.sum((y - pred) ** 2))


def piecewise_threshold(group: pd.DataFrame) -> dict[str, Any] | None:
    data = group[[PRIMARY_DOSE, T90_COL, "y_high"]].dropna()
    if len(data) < 150 or data[PRIMARY_DOSE].nunique() < 10:
        return None
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
                "best_threshold": float(threshold),
                "delta_aic_like": float(base_aic - aic),
                "slope_before": float(beta[1]),
                "slope_after": float(beta[1] + beta[2]),
                "support_before": int(len(before)),
                "support_after": int(len(after)),
                "high_rate_before_threshold": float(before["y_high"].mean()) if len(before) else None,
                "high_rate_after_threshold": float(after["y_high"].mean()) if len(after) else None,
                "high_rate_delta": float(after["y_high"].mean() - before["y_high"].mean()) if len(before) and len(after) else None,
            }
        )
    candidates = pd.DataFrame(rows)
    valid = candidates[(candidates["support_before"] >= 30) & (candidates["support_after"] >= 30)]
    if valid.empty:
        valid = candidates
    return valid.sort_values(["delta_aic_like", "high_rate_delta"], ascending=[False, False]).iloc[0].to_dict()


def safe_qcut(values: pd.Series, n_bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().nunique() < 2:
        return pd.Series(pd.NA, index=values.index, dtype="object")
    for q in range(min(n_bins, numeric.dropna().nunique()), 1, -1):
        try:
            return pd.qcut(numeric, q=q, labels=False, duplicates="drop")
        except Exception:
            continue
    return pd.Series(pd.NA, index=values.index, dtype="object")


def cluster_process_profile(df: pd.DataFrame, selected_features: list[str], regime_report: dict[str, Any]) -> pd.DataFrame:
    features = [f for f in PROFILE_FEATURES if f in df.columns]
    global_median = df[features].apply(pd.to_numeric, errors="coerce").median()
    global_std = df[features].apply(pd.to_numeric, errors="coerce").std(ddof=0).replace(0, np.nan)
    q33 = df[features].apply(pd.to_numeric, errors="coerce").quantile(1 / 3)
    q66 = df[features].apply(pd.to_numeric, errors="coerce").quantile(2 / 3)
    rows = []
    supporting = {(r.get("regime_feature"), r.get("regime_bin")) for r in regime_report.get("strongest_supporting_regimes", [])}
    contradictory = {(r.get("regime_feature"), r.get("regime_bin")) for r in regime_report.get("contradictory_regimes", [])}
    for cluster, group in df.groupby("cluster", sort=True):
        row: dict[str, Any] = {"cluster": int(cluster), "sample_count": int(len(group))}
        z_scores = {}
        resemblance_support = []
        resemblance_contra = []
        for feature in features:
            value = pd.to_numeric(group[feature], errors="coerce").median()
            row[f"{feature}_median"] = float(value) if pd.notna(value) else None
            row[f"{feature}_q25"] = float(pd.to_numeric(group[feature], errors="coerce").quantile(0.25))
            row[f"{feature}_q75"] = float(pd.to_numeric(group[feature], errors="coerce").quantile(0.75))
            if pd.isna(value):
                level = "missing"
            elif value <= q33[feature]:
                level = "low"
            elif value <= q66[feature]:
                level = "mid"
            else:
                level = "high"
            row[f"{feature}_vs_global"] = level
            z = (value - global_median[feature]) / global_std[feature] if pd.notna(value) and pd.notna(global_std[feature]) else np.nan
            z_scores[feature] = z
            if (feature, level) in supporting:
                resemblance_support.append(f"{feature}:{level}")
            if (feature, level) in contradictory:
                resemblance_contra.append(f"{feature}:{level}")
        top = sorted(z_scores.items(), key=lambda kv: abs(kv[1]) if pd.notna(kv[1]) else -1, reverse=True)[:6]
        row["top_differentiating_features"] = ";".join([f"{k}:{v:.2f}" for k, v in top if pd.notna(v)])
        row["resembles_supporting_regimes"] = ";".join(resemblance_support)
        row["resembles_contradictory_regimes"] = ";".join(resemblance_contra)
        row["cluster_profile"] = assign_cluster_profile(row)
        row["selected_feature_count"] = len(selected_features)
        rows.append(row)
    return pd.DataFrame(rows)


def assign_cluster_profile(row: dict[str, Any]) -> str:
    bromine = row.get("bromine_feed_win_60_mean_vs_global")
    alkali = row.get("neutral_alkali_feed_win_60_mean_vs_global")
    flow = row.get("rubber_flow_2_win_60_mean_vs_global")
    r512 = row.get("r512a_temp_win_60_mean_vs_global")
    r513 = row.get("r513_temp_win_60_mean_vs_global")
    ca = row.get("ca_per_rubber_flow_win_60_mean_vs_global")
    support = row.get("resembles_supporting_regimes", "")
    contra = row.get("resembles_contradictory_regimes", "")
    if bromine == "high" and alkali == "high":
        return "high_bromine_high_alkali_high_risk"
    if bromine == "low" or flow == "low":
        return "low_bromine_or_low_flow_contradictory"
    if (r512 == "low" or r513 == "low") and ca in {"mid", "high"}:
        return "low_temp_high_calcium_sensitive"
    if support and not contra:
        return "stable_safe_band_low_risk"
    if contra and not support:
        return "mixed_or_unclear"
    return "mixed_or_unclear"


def cluster_t90_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, group in df.groupby("cluster", sort=True):
        rows.append(
            {
                "cluster": int(cluster),
                "sample_count": int(len(group)),
                "t90_mean": float(group[T90_COL].mean()),
                "t90_median": float(group[T90_COL].median()),
                "t90_std": float(group[T90_COL].std(ddof=1)),
                "t90_q25": float(group[T90_COL].quantile(0.25)),
                "t90_q75": float(group[T90_COL].quantile(0.75)),
                "ok_rate": float(group["y_ok"].mean()),
                "high_rate": float(group["y_high"].mean()),
                "low_rate": float(group["y_low"].mean()),
                "out_spec_rate": float(group["y_out_spec"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["high_rate_rank"] = out["high_rate"].rank(ascending=False, method="min").astype(int)
    out["ok_rate_rank"] = out["ok_rate"].rank(ascending=False, method="min").astype(int)
    return out


def cluster_relations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    dose_rows = []
    for cluster, group in df.groupby("cluster", sort=True):
        ca_t90, ca_t90_p, _ = corr_pair(group[PRIMARY_DOSE], group[T90_COL])
        ca_high, ca_high_p, _ = corr_pair(group[PRIMARY_DOSE], group["y_high"])
        n = len(group)
        bins = 5 if n >= 200 else 4 if n >= 120 else 0
        if bins:
            work = group.dropna(subset=[PRIMARY_DOSE, T90_COL]).copy()
            work["dose_bin"] = safe_qcut(work[PRIMARY_DOSE], bins)
            for bin_id, bg in work.dropna(subset=["dose_bin"]).groupby("dose_bin", sort=True):
                dose_rows.append(
                    {
                        "cluster": int(cluster),
                        "dose_bin": int(bin_id),
                        "sample_count": int(len(bg)),
                        "calcium_min": float(bg[PRIMARY_DOSE].min()),
                        "calcium_max": float(bg[PRIMARY_DOSE].max()),
                        "calcium_median": float(bg[PRIMARY_DOSE].median()),
                        "t90_mean": float(bg[T90_COL].mean()),
                        "ok_rate": float(bg["y_ok"].mean()),
                        "high_rate": float(bg["y_high"].mean()),
                        "low_rate": float(bg["y_low"].mean()),
                        "out_spec_rate": float(bg["y_out_spec"].mean()),
                    }
                )
        threshold = piecewise_threshold(group)
        high_delta = threshold.get("high_rate_delta") if threshold else None
        rows.append(
            {
                "cluster": int(cluster),
                "sample_count": int(n),
                "spearman_ca_t90": ca_t90,
                "spearman_ca_t90_p_value": ca_t90_p,
                "spearman_ca_y_high": ca_high,
                "spearman_ca_y_high_p_value": ca_high_p,
                "dose_bin_count": bins,
                "best_threshold": threshold.get("best_threshold") if threshold else None,
                "high_rate_before_threshold": threshold.get("high_rate_before_threshold") if threshold else None,
                "high_rate_after_threshold": threshold.get("high_rate_after_threshold") if threshold else None,
                "high_rate_delta": high_delta,
                "slope_before": threshold.get("slope_before") if threshold else None,
                "slope_after": threshold.get("slope_after") if threshold else None,
                "threshold_supported": bool(threshold and threshold.get("delta_aic_like", 0) > 2 and high_delta is not None and high_delta >= 0.05),
                "positive_relation_supported": bool(ca_t90 is not None and ca_t90 > 0.05),
                "high_calcium_high_t90_risk_supported": bool(high_delta is not None and high_delta >= 0.05),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(dose_rows)


def consistency_with_regimes(process: pd.DataFrame, relation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, p in process.iterrows():
        cluster = p["cluster"]
        rel = relation[relation["cluster"] == cluster].iloc[0].to_dict()
        support = bool(p.get("resembles_supporting_regimes"))
        contra = bool(p.get("resembles_contradictory_regimes"))
        if support and rel.get("positive_relation_supported") and rel.get("high_calcium_high_t90_risk_supported"):
            typ = "supporting"
        elif contra and not rel.get("positive_relation_supported"):
            typ = "contradictory"
        elif support and contra:
            typ = "mixed"
        elif support:
            typ = "mixed"
        elif contra:
            typ = "contradictory"
        else:
            typ = "neutral"
        rows.append(
            {
                "cluster": int(cluster),
                "resembles_supporting_regimes": p.get("resembles_supporting_regimes"),
                "resembles_contradictory_regimes": p.get("resembles_contradictory_regimes"),
                "positive_relation_supported": rel.get("positive_relation_supported"),
                "high_calcium_high_t90_risk_supported": rel.get("high_calcium_high_t90_risk_supported"),
                "cluster_relation_type": typ,
            }
        )
    return pd.DataFrame(rows)


def safe_band_consistency(df: pd.DataFrame, safe_min: float, safe_max: float) -> pd.DataFrame:
    rows = []
    for cluster, group in df.groupby("cluster", sort=True):
        ca = pd.to_numeric(group[PRIMARY_DOSE], errors="coerce")
        inside = group[(ca >= safe_min) & (ca <= safe_max)]
        above = group[ca > safe_max]
        below = group[ca < safe_min]
        total = ca.notna().sum()
        high_above = above["y_high"].mean() if len(above) else None
        rows.append(
            {
                "cluster": int(cluster),
                "sample_count": int(len(group)),
                "safe_band_min": safe_min,
                "safe_band_max": safe_max,
                "inside_band_count": int(len(inside)),
                "above_band_count": int(len(above)),
                "below_band_count": int(len(below)),
                "inside_band_rate": float(len(inside) / total) if total else None,
                "above_band_rate": float(len(above) / total) if total else None,
                "below_band_rate": float(len(below) / total) if total else None,
                "inside_band_ok_rate": float(inside["y_ok"].mean()) if len(inside) else None,
                "inside_band_high_rate": float(inside["y_high"].mean()) if len(inside) else None,
                "inside_band_low_rate": float(inside["y_low"].mean()) if len(inside) else None,
                "above_band_high_rate": float(high_above) if high_above is not None else None,
                "below_band_low_rate": float(below["y_low"].mean()) if len(below) else None,
                "safe_band_applicable": "true" if len(inside) >= 30 and (high_above is None or inside["y_high"].mean() <= high_above) else "uncertain",
                "manual_review_note_cn": make_manual_note(len(inside), len(above), high_above),
            }
        )
    return pd.DataFrame(rows)


def make_manual_note(inside_n: int, above_n: int, above_high: float | None) -> str:
    if inside_n < 30:
        return "该聚类内安全带内样本较少，只能作为人工复核线索。"
    if above_n >= 20 and above_high is not None and above_high >= 0.20:
        return "该聚类内高于安全带样本的高 T90 风险偏高，适合在人工复核说明中提示。"
    return "该聚类可用于解释当前工况背景，但不建议改变运行包规则。"


def smooth_xy(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    data = group[[PRIMARY_DOSE, T90_COL]].dropna().sort_values(PRIMARY_DOSE)
    if len(data) < 10:
        return np.array([]), np.array([])
    x = data[PRIMARY_DOSE].to_numpy(dtype=float)
    y = data[T90_COL].to_numpy(dtype=float)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        sm = lowess(y, x, frac=0.35, return_sorted=True)
        return sm[:, 0], sm[:, 1]
    except Exception:
        win = max(10, int(len(data) * 0.15))
        yy = pd.Series(y).rolling(win, center=True, min_periods=max(4, win // 3)).median()
        keep = yy.notna().to_numpy()
        return x[keep], yy.to_numpy()[keep]


def make_figures(df: pd.DataFrame, process: pd.DataFrame, t90_profile: pd.DataFrame, relations: pd.DataFrame, dose: pd.DataFrame, figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    heat_features = [f for f in PROFILE_FEATURES if f in df.columns]
    if heat_features:
        global_med = df[heat_features].apply(pd.to_numeric, errors="coerce").median()
        global_std = df[heat_features].apply(pd.to_numeric, errors="coerce").std(ddof=0).replace(0, np.nan)
        z_rows = []
        for cluster, group in df.groupby("cluster", sort=True):
            med = group[heat_features].apply(pd.to_numeric, errors="coerce").median()
            z_rows.append(((med - global_med) / global_std).rename(cluster))
        heat = pd.DataFrame(z_rows)
        fig, ax = plt.subplots(figsize=(10, max(4, len(heat_features) * 0.35)))
        im = ax.imshow(heat.T.fillna(0).values, cmap="coolwarm", aspect="auto", vmin=-2, vmax=2)
        ax.set_yticks(np.arange(len(heat_features)))
        ax.set_yticklabels(heat_features)
        ax.set_xticks(np.arange(len(heat.index)))
        ax.set_xticklabels([str(c) for c in heat.index])
        ax.set_title("稳健聚类的工况画像")
        fig.colorbar(im, ax=ax)
        out = figure_dir / "cluster_process_profile_heatmap.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        generated.append(str(out))

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [str(c) for c in sorted(df["cluster"].unique())]
    ax.boxplot([g[T90_COL].dropna().to_numpy(dtype=float) for _, g in df.groupby("cluster", sort=True)], labels=labels)
    ax.set_title("稳健聚类下的 T90 分布")
    ax.set_xlabel("cluster")
    ax.set_ylabel("T90")
    out = figure_dir / "cluster_t90_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(t90_profile))
    width = 0.25
    ax.bar(x - width, t90_profile["ok_rate"], width, label="ok_rate")
    ax.bar(x, t90_profile["high_rate"], width, label="high_rate")
    ax.bar(x + width, t90_profile["low_rate"], width, label="low_rate")
    ax.set_xticks(x)
    ax.set_xticklabels(t90_profile["cluster"].astype(str))
    ax.set_title("稳健聚类下的 T90 风险率")
    ax.legend()
    out = figure_dir / "cluster_high_rate_and_ok_rate.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([g[PRIMARY_DOSE].dropna().to_numpy(dtype=float) for _, g in df.groupby("cluster", sort=True)], labels=labels)
    ax.set_title("稳健聚类下的钙单耗分布")
    ax.set_xlabel("cluster")
    ax.set_ylabel("钙单耗")
    out = figure_dir / "cluster_ca_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster, group in df.groupby("cluster", sort=True):
        x_s, y_s = smooth_xy(group)
        if len(x_s):
            ax.plot(x_s, y_s, label=f"cluster {cluster}")
    ax.set_title("各聚类内钙单耗与 T90 关系")
    ax.set_xlabel("钙单耗")
    ax.set_ylabel("T90")
    ax.legend()
    out = figure_dir / "cluster_ca_t90_smooth.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(relations["cluster"].astype(str), relations["high_rate_delta"].fillna(0), color="#b64040", label="high_rate_delta")
    ax2 = ax1.twinx()
    ax2.plot(relations["cluster"].astype(str), relations["best_threshold"], marker="o", color="black", label="threshold")
    ax1.set_title("各聚类内钙单耗阈值与高 T90 风险")
    ax1.set_xlabel("cluster")
    ax1.set_ylabel("高 T90 风险差")
    ax2.set_ylabel("阈值")
    out = figure_dir / "cluster_threshold_summary.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))
    return generated


def synthesize(process: pd.DataFrame, t90: pd.DataFrame, relations: pd.DataFrame, consistency: pd.DataFrame, safe: pd.DataFrame) -> tuple[dict[str, Any], str]:
    high_range = float(t90["high_rate"].max() - t90["high_rate"].min()) if not t90.empty else 0.0
    ok_range = float(t90["ok_rate"].max() - t90["ok_rate"].min()) if not t90.empty else 0.0
    supported_clusters = int(relations["high_calcium_high_t90_risk_supported"].sum()) if not relations.empty else 0
    contradiction_count = int((consistency["cluster_relation_type"] == "contradictory").sum()) if not consistency.empty else 0
    explanation = bool(high_range >= 0.10 or ok_range >= 0.10)
    cluster_specific = bool(supported_clusters >= 2)
    flags = {
        "cluster_interpretation_useful": explanation,
        "cluster_t90_separation_supported": bool(high_range >= 0.10 or ok_range >= 0.10),
        "cluster_specific_ca_t90_supported": cluster_specific,
        "cluster_specific_contradictions_identified": bool(contradiction_count > 0),
        "safe_band_explanation_enhanced": bool((safe["safe_band_applicable"] == "true").sum() >= 2),
        "runtime_package_update_recommended": False,
    }
    if flags["cluster_interpretation_useful"] and flags["safe_band_explanation_enhanced"]:
        next_step = "add_cluster_context_to_manual_review_explanation"
    elif flags["cluster_interpretation_useful"]:
        next_step = "use_clusters_for_monitoring_dashboard_only"
    elif cluster_specific:
        next_step = "investigate_cluster_specific_rules_later"
    else:
        next_step = "keep_safe_band_without_cluster_layer"
    return flags, next_step


def append_doc(path: Path, report: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    section_no = 35
    while f"## {section_no}." in existing:
        section_no += 1
    clusters = []
    for row in report.get("cluster_t90_profiles", []):
        clusters.append(f"- cluster {row.get('cluster')}: n={row.get('sample_count')}, ok={row.get('ok_rate')}, high={row.get('high_rate')}, low={row.get('low_rate')}")
    text = f"""

## {section_no}. 稳健聚类工况画像与聚类内钙单耗-T90 关系验证

本阶段承接 Stage 33 的稳健 k=5 聚类和 Stage 34 的分工况阈值复验，目标是解释每个聚类的工况画像，并在聚类内部验证钙单耗-T90 关系。T90 未用于聚类，只用于聚类后的质量解释。

稳健聚类结果：算法 `{report.get('selected_algorithm')}`，k={report.get('selected_k')}，最终特征数 {report.get('selected_feature_count')}。最终特征：{report.get('selected_features')}。core_11 中被剔除特征：{report.get('dropped_core_features')}，主要原因是缺失/强相关/筛选后不进入最终核心特征集。

聚类规模与 T90 概况：
{chr(10).join(clusters)}

聚类内钙单耗-T90 关系：{report.get('cluster_ca_t90_relations')}。

聚类与分工况阈值结果的一致性：{report.get('consistency_with_regime_thresholds')}。

安全带解释：聚类层可增强人工复核解释，但本阶段不修改运行包规则，不产生自动控制或 DCS 写回。推荐下一步：`{report.get('recommended_next_step')}`。

局限性：离线历史分析，不是因果证明；T90 只用于后验解释；聚类解释仍需工艺人工复核；不建议直接更新 runtime package。
"""
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    args = parse_args()
    configure_chinese_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    input_path = resolve_file(args.input)
    clustering_report_path = resolve_file(args.clustering_report)
    assignment_path = resolve_file(args.cluster_assignments)
    regime_report_path = resolve_file(args.regime_threshold_report, required=False)
    clustering_report = load_json(clustering_report_path)
    regime_report = load_json(regime_report_path)

    features = pd.read_parquet(input_path)
    assignments = pd.read_parquet(assignment_path)
    if "time" not in features.columns or "time" not in assignments.columns:
        raise ValueError("Both feature dataset and cluster assignments must contain time.")
    df = join_assignments(features, assignments)
    df = df[df[T90_COL].notna()].copy()
    selected_features, dropped_core = selected_features_from_report(clustering_report)
    selected = clustering_report.get("selected_result") or {}
    selected_k = int(selected.get("k", df["cluster"].nunique()))
    min_size = int(df["cluster"].value_counts().min())
    size_threshold = max(30, int(0.05 * len(df)))
    if args.expected_k and selected_k != args.expected_k:
        raise ValueError(f"Expected selected k={args.expected_k} from robust audit, got {selected_k}")
    if min_size < size_threshold:
        raise ValueError(f"Selected cluster assignment violates size threshold: min={min_size}, threshold={size_threshold}")

    process = cluster_process_profile(df, selected_features, regime_report)
    t90_profile = cluster_t90_profile(df)
    relations, dose = cluster_relations(df)
    consistency = consistency_with_regimes(process, relations)
    safe = safe_band_consistency(df, SAFE_BAND_LOWER_DEFAULT, SAFE_BAND_UPPER_DEFAULT)
    flags, next_step = synthesize(process, t90_profile, relations, consistency, safe)

    process.to_csv(args.output_dir / "cluster_process_profile.csv", index=False, encoding="utf-8-sig")
    t90_profile.to_csv(args.output_dir / "cluster_t90_profile.csv", index=False, encoding="utf-8-sig")
    relations.to_csv(args.output_dir / "cluster_ca_t90_relation.csv", index=False, encoding="utf-8-sig")
    dose.to_csv(args.output_dir / "cluster_dose_bin_response.csv", index=False, encoding="utf-8-sig")
    safe.to_csv(args.output_dir / "cluster_safe_band_consistency.csv", index=False, encoding="utf-8-sig")
    t90_profile.merge(relations, on=["cluster", "sample_count"], how="left").merge(consistency, on="cluster", how="left").merge(safe, on=["cluster", "sample_count"], how="left").to_csv(
        args.table_dir / "cluster_specific_ca_t90_summary.csv", index=False, encoding="utf-8-sig"
    )
    figures = make_figures(df, process, t90_profile, relations, dose, args.figure_dir)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "clustering_report_path": str(clustering_report_path),
        "cluster_assignment_path": str(assignment_path),
        "sample_count": int(len(df)),
        "selected_k": selected_k,
        "selected_algorithm": selected.get("algorithm"),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "dropped_core_features": dropped_core,
        "cluster_size_summary": df["cluster"].value_counts().sort_index().to_dict(),
        "cluster_process_profiles": process.to_dict(orient="records"),
        "cluster_t90_profiles": t90_profile.to_dict(orient="records"),
        "cluster_ca_t90_relations": relations.to_dict(orient="records"),
        "consistency_with_regime_thresholds": consistency.to_dict(orient="records"),
        "safe_band_consistency": safe.to_dict(orient="records"),
        "decision_flags": flags,
        "generated_figures": figures,
        "warnings": [],
        "limitations": ["offline_analysis", "not_causal_proof", "t90_not_used_for_clustering", "no_automatic_control", "no_dcs_writeback", "runtime_package_unchanged"],
        "recommended_next_step": next_step,
    }
    write_json(args.output_dir / "cluster_specific_ca_t90_relationship_report.json", report)
    append_doc(args.doc, report)

    print("Cluster-specific calcium-T90 relationship summary")
    print(f"selected_features: {selected_features}")
    print(f"dropped_core_features: {dropped_core}")
    print(f"cluster_sizes: {report['cluster_size_summary']}")
    print(f"decision_flags: {flags}")
    print(f"recommended_next_step: {next_step}")
    print(f"docs_appended: {args.doc}")
    print("No generated outputs were written under data/.")


if __name__ == "__main__":
    main()
