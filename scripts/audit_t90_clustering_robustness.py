from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


RANDOM_STATES = [7, 17, 29, 42, 101]
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
IR_FEATURE = "output_ir_corrected_offset_20_win_15_std"
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
    parser = argparse.ArgumentParser(description="Audit robustness of T90 operating regime clustering.")
    parser.add_argument("--input", type=Path, default=Path("runs/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--previous-report", type=Path, default=Path("runs/t90_operating_regime_clustering/t90_operating_regime_clustering_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/t90_clustering_robustness_audit"))
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
    for root in [Path("runs"), Path(".")]:
        if root.exists():
            matches = sorted(root.rglob("t90_ca_feature_dataset.parquet"))
            if matches:
                return matches[0]
    for root in [Path("runs"), Path(".")]:
        if root.exists():
            for candidate in sorted(root.rglob("*.parquet")):
                try:
                    df = pd.read_parquet(candidate)
                except Exception:
                    continue
                if "t90" in df.columns and any(c.endswith("_win_60_mean") for c in df.columns):
                    return candidate
    raise FileNotFoundError("No suitable clustering input parquet found.")


def load_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        matches = sorted(Path("runs").rglob(path.name)) if Path("runs").exists() else []
        if matches:
            path = matches[0]
        else:
            return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def previous_consistency(report: dict[str, Any] | None, n: int) -> dict[str, Any]:
    if not report:
        return {"previous_report_available": False, "previous_selection_inconsistent": None}
    sizes = [int(row.get("sample_count", 0)) for row in report.get("cluster_t90_summary", [])]
    previous_min = min(sizes) if sizes else None
    threshold = None
    reason = report.get("selected_k_reason", "")
    match = re.search(r"min_size_threshold=(\d+)", reason)
    if match:
        threshold = int(match.group(1))
    else:
        threshold = max(30, int(0.05 * n))
    return {
        "previous_report_available": True,
        "previous_selected_k": report.get("selected_k"),
        "previous_selected_algorithm": report.get("selected_algorithm"),
        "previous_cluster_sizes": sizes,
        "previous_min_cluster_size": previous_min,
        "previous_min_size_threshold": threshold,
        "previous_selection_inconsistent": bool(previous_min is not None and previous_min < threshold),
    }


def candidate_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in LEAKAGE_COLUMNS:
            continue
        if col in CORE_11 or col == IR_FEATURE or col.endswith(ENGINEERED_SUFFIXES):
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() >= 30 and s.nunique(dropna=True) > 1:
                cols.append(col)
    return cols


def feature_audit(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    corr_group = {}
    frame = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = frame.corr().abs()
    group_id = 0
    assigned: set[str] = set()
    for col in cols:
        if col in assigned:
            continue
        group = [c for c in cols if c != col and corr.loc[col, c] > 0.95]
        if group:
            group_id += 1
            for c in [col] + group:
                corr_group[c] = f"corr_group_{group_id}"
                assigned.add(c)
    for col in cols:
        s = frame[col]
        valid = s.dropna()
        med = valid.median() if len(valid) else np.nan
        mad = (valid - med).abs().median() if len(valid) else np.nan
        robust_z = ((valid - med).abs() / (1.4826 * mad)) if mad and mad > 0 else pd.Series([], dtype=float)
        rows.append(
            {
                "feature": col,
                "missing_rate": float(s.isna().mean()),
                "mean": float(valid.mean()) if len(valid) else None,
                "std": float(valid.std(ddof=1)) if len(valid) else None,
                "min": float(valid.min()) if len(valid) else None,
                "q01": float(valid.quantile(0.01)) if len(valid) else None,
                "q25": float(valid.quantile(0.25)) if len(valid) else None,
                "median": float(valid.median()) if len(valid) else None,
                "q75": float(valid.quantile(0.75)) if len(valid) else None,
                "q99": float(valid.quantile(0.99)) if len(valid) else None,
                "max": float(valid.max()) if len(valid) else None,
                "near_constant": bool(valid.nunique() <= 1 or valid.std(ddof=0) < 1e-12),
                "outlier_ratio": float((robust_z > 5).mean()) if len(robust_z) else 0.0,
                "correlation_group": corr_group.get(col),
            }
        )
    return pd.DataFrame(rows)


def prepare_matrix(df: pd.DataFrame, cols: list[str], *, pca: bool = False, max_features: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    frame = df[cols].apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[:, frame.isna().mean() <= 0.40]
    frame = frame.loc[:, frame.nunique(dropna=True) > 1]
    frame = frame.fillna(frame.median(numeric_only=True))
    for col in frame.columns:
        q01, q99 = frame[col].quantile([0.01, 0.99])
        frame[col] = frame[col].clip(q01, q99)
    std = frame.std(ddof=0)
    frame = frame.loc[:, std > 1e-12]
    corr = frame.corr().abs()
    drop: set[str] = set()
    for i, col in enumerate(corr.columns):
        if col in drop:
            continue
        for other in corr.columns[i + 1 :]:
            if other not in drop and corr.loc[col, other] > 0.95:
                drop.add(other if col in CORE_11 or other not in CORE_11 else col)
    frame = frame.drop(columns=sorted(drop), errors="ignore")
    if max_features and frame.shape[1] > max_features:
        z_tmp = StandardScaler().fit_transform(frame)
        variances = pd.Series(z_tmp.var(axis=0), index=frame.columns)
        priority = pd.Series(0.0, index=frame.columns)
        for c in frame.columns:
            if c in CORE_11 or c == IR_FEATURE:
                priority[c] += 100.0
            if c.endswith("_win_60_mean"):
                priority[c] += 10.0
        keep = (priority + variances.rank(pct=True)).sort_values(ascending=False).head(max_features).index.tolist()
        frame = frame[keep]
    scaler = StandardScaler()
    z = scaler.fit_transform(frame)
    pca2 = PCA(n_components=2, random_state=RANDOM_STATES[0]).fit_transform(z)
    info: dict[str, Any] = {"feature_count": int(frame.shape[1]), "features": list(frame.columns), "dropped_high_corr_count": len(drop)}
    if pca:
        p = PCA(random_state=RANDOM_STATES[0])
        pcs = p.fit_transform(z)
        cum = np.cumsum(p.explained_variance_ratio_)
        n_comp = min(15, max(2, int(np.searchsorted(cum, 0.90) + 1)))
        info["pca_components"] = int(n_comp)
        info["pca_explained_variance"] = float(cum[n_comp - 1])
        return pcs[:, :n_comp], pca2, list(frame.columns), info
    return z, pca2, list(frame.columns), info


def make_variants(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    all_cols = candidate_columns(df)
    variants: dict[str, dict[str, Any]] = {}
    core = [c for c in CORE_11 if c in df.columns]
    variants["core_11_features"] = {"cols": core, "pca": False, "max_features": None}
    core_ir = core + ([IR_FEATURE] if IR_FEATURE in df.columns else [])
    variants["core_11_plus_ir"] = {"cols": core_ir, "pca": False, "max_features": None}
    variants["filtered_engineered_features"] = {"cols": all_cols, "pca": False, "max_features": 30}
    variants["pca_90pct_from_filtered_engineered"] = {"cols": all_cols, "pca": True, "max_features": 60}
    return {k: v for k, v in variants.items() if len(v["cols"]) >= 2}


def cluster_once(matrix: np.ndarray, algorithm: str, k: int, seed: int) -> np.ndarray:
    if algorithm == "KMeans":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(matrix)
    if algorithm == "GaussianMixture":
        from sklearn.mixture import GaussianMixture

        return GaussianMixture(n_components=k, random_state=seed, covariance_type="full").fit_predict(matrix)
    if algorithm == "AgglomerativeClustering":
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(n_clusters=k).fit_predict(matrix)
    raise ValueError(algorithm)


def evaluate_labels(matrix: np.ndarray, labels: np.ndarray, df: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    counts = pd.Series(labels).value_counts()
    enriched = df[["t90", "y_ok", "y_low", "y_high", "y_out_spec", PRIMARY_DOSE]].copy()
    enriched["cluster"] = labels
    grp = enriched.groupby("cluster").agg(t90_mean=("t90", "mean"), high_rate=("y_high", "mean"), ok_rate=("y_ok", "mean"))
    return {
        "silhouette_score": float(silhouette_score(matrix, labels)) if len(counts) > 1 and min(counts) > 1 else None,
        "calinski_harabasz_score": float(calinski_harabasz_score(matrix, labels)) if len(counts) > 1 else None,
        "davies_bouldin_score": float(davies_bouldin_score(matrix, labels)) if len(counts) > 1 else None,
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "cluster_size_imbalance": float(counts.max() / max(counts.min(), 1)),
        "t90_mean_range": float(grp["t90_mean"].max() - grp["t90_mean"].min()),
        "high_rate_range": float(grp["high_rate"].max() - grp["high_rate"].min()),
        "ok_rate_range": float(grp["ok_rate"].max() - grp["ok_rate"].min()),
    }


def evaluate_variant(matrix: np.ndarray, df: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    from sklearn.metrics import adjusted_rand_score

    rows = []
    saved: dict[tuple[str, int], np.ndarray] = {}
    algorithms = ["KMeans", "GaussianMixture", "AgglomerativeClustering"]
    tiny_threshold = max(30, int(0.05 * len(df)))
    for algorithm in algorithms:
        for k in range(2, 11):
            labels_list = []
            metric_rows = []
            seeds = RANDOM_STATES if algorithm != "AgglomerativeClustering" else [RANDOM_STATES[0]]
            for seed in seeds:
                try:
                    labels = cluster_once(matrix, algorithm, k, seed)
                    labels_list.append(labels)
                    metric_rows.append(evaluate_labels(matrix, labels, df))
                except Exception as exc:
                    metric_rows.append({"error": str(exc)})
            valid_metrics = [m for m in metric_rows if "error" not in m]
            if not valid_metrics:
                rows.append({"variant": variant, "algorithm": algorithm, "k": k, "error": metric_rows[0].get("error")})
                continue
            aris = []
            for a, b in itertools.combinations(labels_list, 2):
                aris.append(adjusted_rand_score(a, b))
            avg = pd.DataFrame(valid_metrics).mean(numeric_only=True).to_dict()
            worst_min = min(m["min_cluster_size"] for m in valid_metrics)
            tiny = worst_min < tiny_threshold
            stability = float(np.mean(aris)) if aris else 1.0
            row = {
                "variant": variant,
                "algorithm": algorithm,
                "k": k,
                "seed_count": len(valid_metrics),
                "stability_ari": stability,
                "tiny_cluster_flag": bool(tiny),
                "tiny_cluster_threshold": tiny_threshold,
                "min_cluster_size_worst_seed": int(worst_min),
            }
            row.update({k2: float(v) for k2, v in avg.items()})
            rows.append(row)
            if not tiny:
                saved[(algorithm, k)] = labels_list[0]
    return pd.DataFrame(rows), saved


def cluster_summary(df: pd.DataFrame, labels: np.ndarray, matrix_features: pd.DataFrame | None = None) -> pd.DataFrame:
    work = df[["time", "t90", "y_ok", "y_low", "y_high", "y_out_spec", PRIMARY_DOSE]].copy()
    work["cluster"] = labels
    rows = []
    for cluster, group in work.groupby("cluster", sort=True):
        rows.append(
            {
                "cluster": int(cluster),
                "sample_count": int(len(group)),
                "t90_mean": float(group["t90"].mean()),
                "t90_median": float(group["t90"].median()),
                "t90_q25": float(group["t90"].quantile(0.25)),
                "t90_q75": float(group["t90"].quantile(0.75)),
                "ok_rate": float(group["y_ok"].mean()),
                "high_rate": float(group["y_high"].mean()),
                "low_rate": float(group["y_low"].mean()),
                "out_spec_rate": float(group["y_out_spec"].mean()),
                "ca_consumption_median": float(group[PRIMARY_DOSE].median()),
                "interpretation": interpret_cluster(group),
            }
        )
    return pd.DataFrame(rows)


def interpret_cluster(group: pd.DataFrame) -> str:
    if group["y_ok"].mean() >= 0.90 and group["y_high"].mean() <= 0.12:
        return "stable_safe_band_low_risk"
    if group[PRIMARY_DOSE].median() > 0.021 and group["y_high"].mean() >= 0.22:
        return "high_calcium_high_t90_risk"
    if group[PRIMARY_DOSE].median() < 0.019 and group["y_low"].mean() >= 0.04:
        return "low_calcium_low_t90_risk"
    return "mixed_or_unclear"


def select_result(metrics: pd.DataFrame, labels_by_variant: dict[str, dict[tuple[str, int], np.ndarray]]) -> tuple[dict[str, Any] | None, str]:
    usable = metrics[
        (metrics.get("tiny_cluster_flag") == False)
        & (metrics.get("stability_ari", 0) >= 0.70)
        & ((metrics.get("t90_mean_range", 0) >= 0.05) | (metrics.get("high_rate_range", 0) >= 0.05))
        & (metrics.get("silhouette_score", 0).fillna(0) > 0.03)
    ].copy()
    if usable.empty:
        return None, "No tested clustering result passed tiny-cluster, stability, separation, and silhouette criteria."
    usable["score"] = (
        usable["silhouette_score"].fillna(0).rank(pct=True)
        + usable["stability_ari"].fillna(0).rank(pct=True)
        + usable["high_rate_range"].fillna(0).rank(pct=True)
        + usable["t90_mean_range"].fillna(0).rank(pct=True)
        - usable["davies_bouldin_score"].fillna(999).rank(pct=True)
    )
    best = usable.sort_values(["score", "silhouette_score"], ascending=[False, False]).iloc[0].to_dict()
    return best, "Selected because it passed robustness gates and had the best combined stability, separation, and clustering metric score."


def make_figures(feature_audit_df: pd.DataFrame, metrics: pd.DataFrame, selected_assignments: pd.DataFrame, pca_points: np.ndarray | None, figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    top_missing = feature_audit_df.sort_values("missing_rate", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_missing["feature"], top_missing["missing_rate"])
    ax.set_title("聚类候选特征缺失率审计")
    ax.set_xlabel("缺失率")
    out = figure_dir / "t90_clustering_feature_missingness.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    fig, ax = plt.subplots(figsize=(10, 5))
    for variant, group in metrics[metrics["algorithm"] == "KMeans"].groupby("variant"):
        ax.plot(group["k"], group["silhouette_score"], marker="o", label=variant)
    ax.set_title("不同特征集的 KMeans 聚类指标")
    ax.set_xlabel("k")
    ax.set_ylabel("silhouette")
    ax.legend(fontsize=8)
    out = figure_dir / "t90_clustering_k_metrics_by_variant.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    fig, ax = plt.subplots(figsize=(10, 5))
    view = metrics[metrics["algorithm"] == "KMeans"]
    ax.scatter(view["k"], view["min_cluster_size_worst_seed"], c=view["tiny_cluster_flag"].astype(int), cmap="coolwarm", alpha=0.7)
    if "tiny_cluster_threshold" in view:
        ax.axhline(float(view["tiny_cluster_threshold"].dropna().iloc[0]), color="black", linestyle="--", label="min size threshold")
    ax.set_title("聚类最小簇规模审计")
    ax.set_xlabel("k")
    ax.set_ylabel("最小簇样本数")
    ax.legend()
    out = figure_dir / "t90_clustering_cluster_size_audit.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    generated.append(str(out))

    if not selected_assignments.empty and selected_assignments["cluster"].nunique() > 1:
        labels = [str(c) for c in sorted(selected_assignments["cluster"].unique())]
        data = [g["t90"].dropna().to_numpy(dtype=float) for _, g in selected_assignments.groupby("cluster", sort=True)]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data, labels=labels)
        ax.set_title("稳健聚类选中结果的 T90 分布")
        ax.set_xlabel("cluster")
        ax.set_ylabel("T90")
        out = figure_dir / "t90_clustering_selected_t90_distribution.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        generated.append(str(out))
        if pca_points is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(pca_points[:, 0], pca_points[:, 1], c=selected_assignments["cluster"], cmap="tab10", s=14, alpha=0.75)
            ax.set_title("稳健聚类选中结果 PCA 投影")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            fig.colorbar(sc, ax=ax)
            out = figure_dir / "t90_clustering_selected_pca_scatter.png"
            fig.tight_layout()
            fig.savefig(out, dpi=160)
            plt.close(fig)
            generated.append(str(out))
    else:
        for name in ["t90_clustering_selected_t90_distribution.png", "t90_clustering_selected_pca_scatter.png"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "无通过稳健性门槛的聚类结果", ha="center", va="center")
            ax.axis("off")
            out = figure_dir / name
            fig.savefig(out, dpi=160)
            plt.close(fig)
            generated.append(str(out))
    return generated


def append_doc(path: Path, report: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    section_no = 33
    while f"## {section_no}." in existing:
        section_no += 1
    text = f"""

## {section_no}. 聚类特征审计与稳健聚类复验

本阶段复核上一轮聚类结果的稳健性，重点检查 k=2 结果是否违反最小簇规模门槛。上一轮最小簇样本数为 {report.get('previous_min_cluster_size')}，门槛为 {report.get('previous_min_size_threshold')}，不一致标记为 {report.get('previous_selection_inconsistent')}。

本次审计测试的特征集包括 core_11、core_11_plus_ir、filtered_engineered_features 与 pca_90pct_from_filtered_engineered。候选特征审计与各 k 指标已写入 runs。

是否找到稳健聚类结果：{report.get('selected_result') is not None}。选中结果：{report.get('selected_result')}。理由：{report.get('selected_result_reason')}。

推荐下一步：`{report.get('recommended_next_step')}`。

局限性：聚类为无监督离线分析，T90 不参与聚类；不同算法和特征集可能给出不同划分；若没有稳定且非小簇的结果，应继续保留规则型安全带而不是强行解释聚类。
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
    df = ensure_targets(df)
    supervised = df[df["t90"].notna()].copy().reset_index(drop=True)
    prev_report = load_json_optional(args.previous_report)
    prev = previous_consistency(prev_report, len(supervised))

    cols = candidate_columns(supervised)
    audit_df = feature_audit(supervised, cols)
    audit_df.to_csv(args.output_dir / "clustering_feature_audit.csv", index=False, encoding="utf-8-sig")

    all_metrics = []
    label_store: dict[str, dict[tuple[str, int], np.ndarray]] = {}
    pca_store: dict[str, np.ndarray] = {}
    variant_infos = []
    for variant, spec in make_variants(supervised).items():
        matrix, pca2, used_features, info = prepare_matrix(supervised, spec["cols"], pca=spec["pca"], max_features=spec["max_features"])
        info.update({"variant": variant})
        variant_infos.append(info)
        metrics, labels = evaluate_variant(matrix, supervised, variant)
        metrics["feature_count"] = info["feature_count"]
        all_metrics.append(metrics)
        label_store[variant] = labels
        pca_store[variant] = pca2
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.to_csv(args.output_dir / "clustering_variant_k_metrics.csv", index=False, encoding="utf-8-sig")

    selected, reason = select_result(metrics_df, label_store)
    if selected:
        variant = selected["variant"]
        labels = label_store[variant][(selected["algorithm"], int(selected["k"]))]
        assignments = supervised[["time", "t90", "y_ok", "y_low", "y_high", "y_out_spec", PRIMARY_DOSE]].copy()
        assignments["cluster"] = labels.astype(int)
        summary = cluster_summary(supervised, labels)
        pca_points = pca_store.get(variant)
        next_step = "use_clusters_for_context_specific_ca_t90_analysis"
    else:
        assignments = supervised[["time", "t90", "y_ok", "y_low", "y_high", "y_out_spec", PRIMARY_DOSE]].copy()
        assignments["cluster"] = -1
        summary = pd.DataFrame()
        pca_points = None
        next_step = "fix_clustering_selection_logic" if prev.get("previous_selection_inconsistent") else "clustering_not_stable_keep_rule_based_safe_band"

    assignments.to_parquet(args.output_dir / "clustering_selected_assignments.parquet", index=False)
    summary.to_csv(args.output_dir / "clustering_selected_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.table_dir / "t90_clustering_robustness_summary.csv", index=False, encoding="utf-8-sig")
    figures = make_figures(audit_df, metrics_df, assignments if selected else pd.DataFrame(), pca_points, args.figure_dir)

    rejected = {
        "tiny_cluster_rows": int(metrics_df["tiny_cluster_flag"].fillna(False).sum()),
        "non_tiny_rows": int((~metrics_df["tiny_cluster_flag"].fillna(True)).sum()),
        "stable_non_tiny_rows": int(((~metrics_df["tiny_cluster_flag"].fillna(True)) & (metrics_df["stability_ari"].fillna(0) >= 0.70)).sum()),
    }
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "previous_report_path": str(args.previous_report),
        **prev,
        "feature_audit_summary": {
            "candidate_feature_count": int(len(cols)),
            "high_missing_feature_count": int((audit_df["missing_rate"] > 0.40).sum()),
            "near_constant_feature_count": int(audit_df["near_constant"].sum()),
            "high_outlier_feature_count": int((audit_df["outlier_ratio"] > 0.05).sum()),
        },
        "feature_variants_tested": variant_infos,
        "k_metrics_summary": {
            "row_count": int(len(metrics_df)),
            "best_non_tiny_silhouette": sanitize(metrics_df.loc[~metrics_df["tiny_cluster_flag"].fillna(True), "silhouette_score"].max()),
        },
        "rejected_results_summary": rejected,
        "selected_result": selected,
        "selected_result_reason": reason,
        "selected_cluster_t90_summary": summary.to_dict(orient="records"),
        "generated_figures": figures,
        "warnings": [],
        "limitations": ["unsupervised_clustering", "t90_not_used_for_clustering", "offline_historical_analysis", "not_causal_proof", "small_clusters_rejected"],
        "recommended_next_step": next_step,
    }
    write_json(args.output_dir / "t90_clustering_robustness_audit_report.json", report)
    append_doc(args.doc, report)

    print("T90 clustering robustness audit summary")
    print(f"previous_selection_inconsistent: {prev.get('previous_selection_inconsistent')}")
    print(f"previous_min_cluster_size: {prev.get('previous_min_cluster_size')}")
    print(f"previous_min_size_threshold: {prev.get('previous_min_size_threshold')}")
    print(f"selected_result: {selected}")
    print(f"recommended_next_step: {next_step}")
    print(f"docs_appended: {args.doc}")
    print("No generated outputs were written under data/.")


if __name__ == "__main__":
    main()
