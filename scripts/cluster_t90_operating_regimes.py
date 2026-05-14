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
LEAKAGE_COLUMNS = {
    "time",
    "t90",
    "t90_C",
    "t90_D",
    "t90_E",
    "t90_label_count",
    "y_ok",
    "y_low",
    "y_high",
    "y_out_spec",
}
CORE_FEATURES = [
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
    "output_ir_corrected_offset_20_win_15_std",
]
ENGINEERED_SUFFIXES = ("_win_60_mean", "_win_60_std", "_win_120_mean", "_win_120_std", "_slope", "_diff", "_lag")


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
    parser = argparse.ArgumentParser(description="Cluster operating regimes and interpret post-cluster T90 distributions.")
    parser.add_argument("--input", type=Path, default=Path("runs/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/t90_operating_regime_clustering"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--allow-high-missing", action="store_true")
    parser.add_argument("--max-features", type=int, default=60)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
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
    for root in [Path("runs"), Path(".")]:
        if not root.exists():
            continue
        for candidate in sorted(root.rglob("*.parquet")):
            try:
                df = pd.read_parquet(candidate)
            except Exception:
                continue
            if "t90" in df.columns and any(c.endswith("_win_60_mean") for c in df.columns):
                return candidate
    raise FileNotFoundError("No suitable parquet file found for T90 operating regime clustering.")


def ensure_targets(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["t90"] = pd.to_numeric(data["t90"], errors="coerce")
    if "y_ok" not in data.columns:
        data["y_ok"] = ((data["t90"] >= 8.20) & (data["t90"] <= 8.70)).astype(int)
    if "y_low" not in data.columns:
        data["y_low"] = (data["t90"] < 8.20).astype(int)
    if "y_high" not in data.columns:
        data["y_high"] = (data["t90"] > 8.70).astype(int)
    if "y_out_spec" not in data.columns:
        data["y_out_spec"] = ((data["t90"] < 8.20) | (data["t90"] > 8.70)).astype(int)
    return data


def candidate_features(df: pd.DataFrame, allow_high_missing: bool, max_features: int) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    numeric = []
    for col in df.columns:
        if col in LEAKAGE_COLUMNS:
            continue
        if col in CORE_FEATURES or col.endswith(ENGINEERED_SUFFIXES):
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().sum() >= 30 and series.nunique(dropna=True) > 1:
                numeric.append(col)
    before = len(numeric)
    if not allow_high_missing:
        numeric = [c for c in numeric if pd.to_numeric(df[c], errors="coerce").isna().mean() <= 0.40]
    missing_filtered = before - len(numeric)
    frame = df[numeric].apply(pd.to_numeric, errors="coerce")
    medians = frame.median(numeric_only=True)
    frame = frame.fillna(medians)
    std = frame.std(ddof=0)
    numeric = [c for c in numeric if std.get(c, 0) > 1e-12]
    frame = frame[numeric]
    if frame.empty:
        raise ValueError("No usable non-leakage numeric engineered features available for clustering.")

    corr = frame.corr().abs()
    drop: set[str] = set()
    for i, col in enumerate(corr.columns):
        if col in drop:
            continue
        for other in corr.columns[i + 1 :]:
            if other in drop:
                continue
            if corr.loc[col, other] > 0.95:
                keep_col = col
                drop_col = other
                if other in CORE_FEATURES and col not in CORE_FEATURES:
                    keep_col, drop_col = other, col
                drop.add(drop_col)
    numeric = [c for c in numeric if c not in drop]
    frame = frame[numeric]

    if len(numeric) > max_features:
        variances = frame.var(ddof=0)
        priority = pd.Series(0.0, index=numeric)
        for c in numeric:
            if c in CORE_FEATURES:
                priority[c] += 100.0
            if c.endswith("_win_60_mean"):
                priority[c] += 10.0
        ranked = (priority + variances.rank(pct=True)).sort_values(ascending=False)
        numeric = list(ranked.head(max_features).index)
        frame = frame[numeric]
    info = {
        "feature_count_before_filter": before,
        "feature_count_after_missing_filter": before - missing_filtered,
        "missing_filtered_count": missing_filtered,
        "high_correlation_dropped_count": len(drop),
        "final_feature_count": len(numeric),
    }
    return numeric, frame, info


def standardize_and_reduce(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    z = scaler.fit_transform(frame)
    pca = PCA(random_state=RANDOM_SEED)
    pcs = pca.fit_transform(z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(min(20, max(2, np.searchsorted(cum, 0.90) + 1)))
    if frame.shape[1] > 40:
        cluster_input = pcs[:, :n_components]
        mode = "pca_components"
    else:
        cluster_input = z
        mode = "standardized_features"
    info = {
        "clustering_input_mode": mode,
        "pca_components_for_90pct": n_components,
        "pca_explained_variance_90pct": float(cum[n_components - 1]) if len(cum) >= n_components else None,
        "pca_first_two_explained_variance": [float(x) for x in pca.explained_variance_ratio_[:2]],
    }
    return cluster_input, pcs[:, :2], info


def evaluate_k(cluster_input: np.ndarray, df: pd.DataFrame, k_min: int, k_max: int) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    rows = []
    n = len(df)
    for k in range(k_min, min(k_max, n - 1) + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
        labels = model.fit_predict(cluster_input)
        counts = pd.Series(labels).value_counts()
        enriched = df.copy()
        enriched["cluster"] = labels
        summary = enriched.groupby("cluster").agg(t90_mean=("t90", "mean"), high_rate=("y_high", "mean"), ok_rate=("y_ok", "mean"))
        rows.append(
            {
                "algorithm": "KMeans",
                "k": k,
                "silhouette_score": float(silhouette_score(cluster_input, labels)) if len(set(labels)) > 1 else None,
                "calinski_harabasz_score": float(calinski_harabasz_score(cluster_input, labels)) if len(set(labels)) > 1 else None,
                "davies_bouldin_score": float(davies_bouldin_score(cluster_input, labels)) if len(set(labels)) > 1 else None,
                "cluster_size_min": int(counts.min()),
                "cluster_size_max": int(counts.max()),
                "cluster_size_imbalance": float(counts.max() / max(counts.min(), 1)),
                "t90_mean_range_between_clusters": float(summary["t90_mean"].max() - summary["t90_mean"].min()),
                "high_rate_range_between_clusters": float(summary["high_rate"].max() - summary["high_rate"].min()),
                "ok_rate_range_between_clusters": float(summary["ok_rate"].max() - summary["ok_rate"].min()),
            }
        )
    return pd.DataFrame(rows)


def choose_k(metrics: pd.DataFrame, n: int) -> tuple[int, str]:
    if metrics.empty:
        raise ValueError("No k-selection metrics available.")
    min_size = max(30, int(0.05 * n))
    valid = metrics[
        (metrics["cluster_size_min"] >= min_size)
        & ((metrics["t90_mean_range_between_clusters"] >= 0.05) | (metrics["high_rate_range_between_clusters"] >= 0.05))
    ].copy()
    if valid.empty:
        valid = metrics[metrics["cluster_size_min"] >= min_size].copy()
    if valid.empty:
        valid = metrics.copy()
    valid["rank_score"] = (
        valid["silhouette_score"].rank(ascending=False, pct=True)
        + valid["calinski_harabasz_score"].rank(ascending=False, pct=True)
        + valid["davies_bouldin_score"].rank(ascending=True, pct=True)
        + valid["high_rate_range_between_clusters"].rank(ascending=False, pct=True)
    )
    best = valid.sort_values(["rank_score", "silhouette_score"], ascending=[False, False]).iloc[0]
    reason = (
        "selected by balanced clustering metrics with minimum cluster size and post-cluster T90/high-risk separation; "
        f"min_size_threshold={min_size}"
    )
    return int(best["k"]), reason


def fit_kmeans(cluster_input: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
    return model.fit_predict(cluster_input)


def cluster_summary(assignments: pd.DataFrame, feature_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    rows = []
    profile_rows = []
    interpretations = []
    global_mean = feature_frame.mean()
    global_std = feature_frame.std(ddof=0).replace(0, np.nan)
    overall_high = assignments["y_high"].mean()
    overall_low = assignments["y_low"].mean()
    for cluster, group in assignments.groupby("cluster", sort=True):
        idx = group.index
        feature_diff = ((feature_frame.loc[idx].mean() - global_mean) / global_std).replace([np.inf, -np.inf], np.nan).dropna()
        top_features = feature_diff.abs().sort_values(ascending=False).head(6).index.tolist()
        feature_notes = []
        for feat in top_features:
            z = float(feature_diff[feat])
            profile_rows.append({"cluster": int(cluster), "feature": feat, "z_vs_global_mean": z, "cluster_mean": float(feature_frame.loc[idx, feat].mean()), "global_mean": float(global_mean[feat])})
            feature_notes.append(f"{feat}:{z:.2f}")
        profile = interpret_cluster(group, feature_diff, overall_high, overall_low)
        interpretations.append({"cluster": int(cluster), "profile": profile, "top_differentiating_features": feature_notes})
        rows.append(
            {
                "cluster": int(cluster),
                "sample_count": int(len(group)),
                "t90_mean": float(group["t90"].mean()),
                "t90_median": float(group["t90"].median()),
                "t90_std": float(group["t90"].std(ddof=1)),
                "t90_q25": float(group["t90"].quantile(0.25)),
                "t90_q75": float(group["t90"].quantile(0.75)),
                "t90_min": float(group["t90"].min()),
                "t90_max": float(group["t90"].max()),
                "ok_rate": float(group["y_ok"].mean()),
                "low_rate": float(group["y_low"].mean()),
                "high_rate": float(group["y_high"].mean()),
                "out_spec_rate": float(group["y_out_spec"].mean()),
                "ca_consumption_mean": float(group[PRIMARY_DOSE].mean()) if PRIMARY_DOSE in group else None,
                "ca_consumption_median": float(group[PRIMARY_DOSE].median()) if PRIMARY_DOSE in group else None,
                "ca_consumption_q25": float(group[PRIMARY_DOSE].quantile(0.25)) if PRIMARY_DOSE in group else None,
                "ca_consumption_q75": float(group[PRIMARY_DOSE].quantile(0.75)) if PRIMARY_DOSE in group else None,
                "top_differentiating_features": ";".join(feature_notes),
                "regime_interpretation": profile,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(profile_rows), interpretations


def interpret_cluster(group: pd.DataFrame, feature_diff: pd.Series, overall_high: float, overall_low: float) -> str:
    high_rate = group["y_high"].mean()
    low_rate = group["y_low"].mean()
    ca_z = feature_diff.get(PRIMARY_DOSE, 0.0)
    flow_z = feature_diff.get("rubber_flow_2_win_60_mean", 0.0)
    bromine_z = feature_diff.get("bromine_feed_win_60_mean", 0.0)
    temp_z = max([feature_diff.get(c, 0.0) for c in ["r510a_temp_win_60_mean", "r511a_temp_win_60_mean", "r512a_temp_win_60_mean", "r513_temp_win_60_mean", "r514_temp_win_60_mean"]])
    if ca_z > 0.5 and high_rate >= overall_high + 0.05:
        return "high_calcium_high_t90_risk"
    if ca_z < -0.5 and low_rate >= overall_low + 0.03:
        return "low_calcium_low_t90_risk"
    if group["y_ok"].mean() >= 0.90 and high_rate <= overall_high and low_rate <= overall_low + 0.01:
        return "stable_safe_band_low_risk"
    if flow_z > 0.8:
        return "high_flow_dilution_regime"
    if bromine_z > 0.8:
        return "high_bromine_regime"
    if temp_z > 0.8:
        return "high_temperature_regime"
    return "mixed_or_unclear"


def between_cluster_tests(assignments: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    groups = [g["t90"].dropna().to_numpy(dtype=float) for _, g in assignments.groupby("cluster", sort=True)]
    tests: dict[str, Any] = {}
    try:
        from scipy import stats

        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            tests["anova"] = {"statistic": float(stats.f_oneway(*groups).statistic), "p_value": float(stats.f_oneway(*groups).pvalue)}
            kw = stats.kruskal(*groups)
            tests["kruskal_wallis"] = {"statistic": float(kw.statistic), "p_value": float(kw.pvalue)}
    except Exception as exc:
        tests["warning"] = f"scipy_tests_skipped: {exc}"
    pair_rows = []
    summary = assignments.groupby("cluster")["t90"].mean()
    for c1 in summary.index:
        for c2 in summary.index:
            if c1 >= c2:
                continue
            pair_rows.append({"cluster_a": int(c1), "cluster_b": int(c2), "t90_mean_delta_a_minus_b": float(summary.loc[c1] - summary.loc[c2])})
    return tests, pd.DataFrame(pair_rows)


def make_figures(metrics: pd.DataFrame, assignments: pd.DataFrame, pcs2: np.ndarray, cluster_stats: pd.DataFrame, profiles: pd.DataFrame, figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(metrics["k"], metrics["silhouette_score"], marker="o")
    axes[0].set_title("Silhouette")
    axes[1].plot(metrics["k"], metrics["calinski_harabasz_score"], marker="o")
    axes[1].set_title("Calinski-Harabasz")
    axes[2].plot(metrics["k"], metrics["davies_bouldin_score"], marker="o")
    axes[2].set_title("Davies-Bouldin")
    fig.suptitle("聚类数量选择指标")
    for ax in axes:
        ax.set_xlabel("k")
    out = figure_dir / "t90_cluster_k_selection.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(pcs2[:, 0], pcs2[:, 1], c=assignments["cluster"], cmap="tab10", s=14, alpha=0.75)
    ax.set_title("工况聚类 PCA 投影")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="cluster")
    out = figure_dir / "t90_cluster_pca_scatter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    fig, ax = plt.subplots(figsize=(9, 5))
    box_data = [g["t90"].dropna().to_numpy(dtype=float) for _, g in assignments.groupby("cluster", sort=True)]
    labels = [str(c) for c in sorted(assignments["cluster"].unique())]
    ax.boxplot(box_data, labels=labels)
    ax.set_title("不同聚类类别的 T90 分布")
    ax.set_xlabel("聚类类别")
    ax.set_ylabel("T90")
    out = figure_dir / "t90_cluster_t90_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cluster_stats))
    width = 0.25
    ax.bar(x - width, cluster_stats["ok_rate"], width, label="ok_rate")
    ax.bar(x, cluster_stats["high_rate"], width, label="high_rate")
    ax.bar(x + width, cluster_stats["low_rate"], width, label="low_rate")
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_stats["cluster"].astype(str))
    ax.set_title("不同聚类类别的 T90 风险率")
    ax.set_xlabel("聚类类别")
    ax.set_ylabel("比例")
    ax.legend()
    out = figure_dir / "t90_cluster_risk_rates.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    fig, ax = plt.subplots(figsize=(9, 5))
    box_data = [g[PRIMARY_DOSE].dropna().to_numpy(dtype=float) for _, g in assignments.groupby("cluster", sort=True)]
    ax.boxplot(box_data, labels=labels)
    ax.set_title("不同聚类类别的钙单耗分布")
    ax.set_xlabel("聚类类别")
    ax.set_ylabel("钙单耗")
    out = figure_dir / "t90_cluster_ca_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    if not profiles.empty:
        top_features = profiles.groupby("feature")["z_vs_global_mean"].apply(lambda s: s.abs().max()).sort_values(ascending=False).head(12).index.tolist()
        pivot = profiles[profiles["feature"].isin(top_features)].pivot(index="feature", columns="cluster", values="z_vs_global_mean").fillna(0)
        fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.35)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_title("不同聚类类别的关键工况特征画像")
        ax.set_xlabel("聚类类别")
        fig.colorbar(im, ax=ax, label="z vs global")
        out = figure_dir / "t90_cluster_feature_heatmap.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        generated.append(str(out))
    return generated


def append_doc(path: Path, report: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    section_no = 31
    while f"## {section_no}." in existing:
        section_no += 1
    cluster_lines = []
    for row in report.get("cluster_t90_summary", []):
        cluster_lines.append(
            f"- cluster {row.get('cluster')}: n={row.get('sample_count')}, T90均值={row.get('t90_mean')}, "
            f"ok={row.get('ok_rate')}, high={row.get('high_rate')}, profile={row.get('regime_interpretation')}"
        )
    text = f"""

## {section_no}. 基于工况与衍生特征的聚类分析及 T90 分布解释

本阶段用非泄漏工况与衍生特征做无监督聚类，T90 与目标标签不参与聚类，只在聚类完成后用于解释各类工况的质量分布。

使用特征数：{report.get('feature_count_after_filter')}；排除泄漏列：{', '.join(report.get('excluded_leakage_columns', []))}。

k 搜索范围：{report.get('k_search_range')}；最终选择算法 `{report.get('selected_algorithm')}`，k={report.get('selected_k')}。选择理由：{report.get('selected_k_reason')}。

聚类 T90 概况：
{chr(10).join(cluster_lines)}

组间 T90 检验：{report.get('between_cluster_t90_tests')}。

推荐下一步：`{report.get('recommended_next_step')}`。

局限性：该聚类为无监督历史分析，T90 未用于聚类；聚类差异不等于因果证明；结果只适合辅助后续分工况监测或钙-T90 关系分析，不构成自动控制策略。
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
    if "t90" not in df.columns:
        raise ValueError("Input must contain t90 for post-cluster interpretation.")
    df = ensure_targets(df)
    supervised = df[df["t90"].notna()].copy().reset_index(drop=True)
    if len(supervised) < 100:
        raise ValueError("Insufficient supervised rows for clustering analysis.")

    features, feature_frame, feature_info = candidate_features(supervised, args.allow_high_missing, args.max_features)
    cluster_input, pcs2, dim_info = standardize_and_reduce(feature_frame)
    metrics = evaluate_k(cluster_input, supervised, args.k_min, args.k_max)
    selected_k, selected_reason = choose_k(metrics, len(supervised))
    labels = fit_kmeans(cluster_input, selected_k)
    assignments = supervised[["time", "t90", "y_ok", "y_low", "y_high", "y_out_spec"] + ([PRIMARY_DOSE] if PRIMARY_DOSE in supervised.columns else [])].copy()
    assignments["cluster"] = labels.astype(int)
    cluster_stats, profiles, interpretations = cluster_summary(assignments, feature_frame)
    tests, pairwise = between_cluster_tests(assignments)

    feature_list = pd.DataFrame({"feature": features, "used_for_clustering": True})
    feature_list.to_csv(args.output_dir / "clustering_feature_list.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output_dir / "clustering_k_selection_metrics.csv", index=False, encoding="utf-8-sig")
    assignments.to_parquet(args.output_dir / "clustering_assignments.parquet", index=False)
    assignments.to_csv(args.output_dir / "clustering_assignments.csv", index=False, encoding="utf-8-sig")
    cluster_stats.to_csv(args.output_dir / "cluster_t90_summary.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(args.output_dir / "cluster_feature_profiles.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(args.output_dir / "cluster_pairwise_t90_differences.csv", index=False, encoding="utf-8-sig")
    cluster_stats.to_csv(args.table_dir / "t90_cluster_summary.csv", index=False, encoding="utf-8-sig")

    figures = make_figures(metrics, assignments, pcs2, cluster_stats, profiles, args.figure_dir)

    high_range = float(cluster_stats["high_rate"].max() - cluster_stats["high_rate"].min()) if not cluster_stats.empty else 0.0
    t90_range = float(cluster_stats["t90_mean"].max() - cluster_stats["t90_mean"].min()) if not cluster_stats.empty else 0.0
    if high_range >= 0.10 or t90_range >= 0.10:
        next_step = "use_clusters_for_context_specific_ca_t90_analysis"
    elif high_range >= 0.05 or t90_range >= 0.05:
        next_step = "use_clusters_for_regime_monitoring"
    else:
        next_step = "clustering_not_stable_keep_rule_based_safe_band"

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "sample_count": int(len(supervised)),
        "feature_count_before_filter": int(feature_info["feature_count_before_filter"]),
        "feature_count_after_filter": int(len(features)),
        "feature_filter_summary": feature_info,
        "dimensionality_summary": dim_info,
        "clustering_features": features,
        "excluded_leakage_columns": sorted(LEAKAGE_COLUMNS),
        "k_search_range": [args.k_min, args.k_max],
        "k_selection_metrics": metrics.to_dict(orient="records"),
        "selected_algorithm": "KMeans",
        "selected_k": selected_k,
        "selected_k_reason": selected_reason,
        "cluster_t90_summary": cluster_stats.to_dict(orient="records"),
        "cluster_feature_profiles": profiles.to_dict(orient="records"),
        "between_cluster_t90_tests": tests,
        "cluster_interpretations": interpretations,
        "generated_figures": figures,
        "warnings": [],
        "limitations": ["unsupervised_clustering", "t90_not_used_for_clustering", "offline_historical_analysis", "not_causal_proof", "no_automatic_control"],
        "recommended_next_step": next_step,
    }
    write_json(args.output_dir / "t90_operating_regime_clustering_report.json", report)
    append_doc(args.doc, report)

    print("Operating regime clustering summary")
    print(f"input_path: {input_path}")
    print(f"sample_count: {len(supervised)}")
    print(f"feature_count_after_filter: {len(features)}")
    print(f"selected_k: {selected_k}")
    print(f"high_rate_range: {high_range}")
    print(f"t90_mean_range: {t90_range}")
    print(f"recommended_next_step: {next_step}")
    print(f"figures: {figures}")
    print(f"docs_appended: {args.doc}")
    print("No generated outputs were written under data/.")


if __name__ == "__main__":
    main()
