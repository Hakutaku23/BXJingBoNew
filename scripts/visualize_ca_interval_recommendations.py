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


POSITION_ORDER = ["inside_band", "below_band", "above_band", "missing"]
POSITION_COLORS = {
    "inside_band": "#2E7D32",
    "below_band": "#1565C0",
    "above_band": "#C62828",
    "missing": "#757575",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize test-like calcium interval recommendations.")
    parser.add_argument("--replay", type=Path, default=Path("data/ca_interval_recommender_replay.parquet"))
    parser.add_argument("--metrics", type=Path, default=Path("data/ca_interval_recommender_metrics.csv"))
    parser.add_argument("--report-input", type=Path, default=Path("data/ca_interval_recommender_report.json"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--data-output", type=Path, default=Path("data/ca_interval_recommendation_visualization_table.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/ca_interval_recommendation_visualization_report.json"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    return parser.parse_args()


def as_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
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


def derive_interval_position(row: pd.Series) -> str:
    current = pd.to_numeric(pd.Series([row.get("current_ca_consumption")]), errors="coerce").iloc[0]
    rec_min = pd.to_numeric(pd.Series([row.get("recommended_ca_consumption_min")]), errors="coerce").iloc[0]
    rec_max = pd.to_numeric(pd.Series([row.get("recommended_ca_consumption_max")]), errors="coerce").iloc[0]
    if not np.isfinite(current) or not np.isfinite(rec_min) or not np.isfinite(rec_max):
        return "missing"
    if current < rec_min:
        return "below_band"
    if current > rec_max:
        return "above_band"
    return "inside_band"


def load_test_like(args: argparse.Namespace, warnings: list[str]) -> pd.DataFrame:
    if not args.replay.exists():
        raise FileNotFoundError(f"Replay parquet does not exist: {args.replay}")
    frame = pd.read_parquet(args.replay)
    missing = [column for column in ["time", "current_ca_consumption", "recommended_ca_consumption_min", "recommended_ca_consumption_max"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Replay data is missing required columns: {missing}")
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.sort_values("time").reset_index(drop=True)
    if "split" in frame.columns:
        test = frame[frame["split"].eq("test_like")].copy()
    else:
        warnings.append("split column is missing; using last 20% by time as test_like.")
        start = int(len(frame) * 0.8)
        test = frame.iloc[start:].copy()
        test["split"] = "test_like"
    if test.empty:
        raise ValueError("No test_like rows are available for visualization.")
    if "interval_position" not in test.columns:
        test["interval_position"] = test.apply(derive_interval_position, axis=1)
    else:
        test["interval_position"] = test["interval_position"].fillna(test.apply(derive_interval_position, axis=1))
    test["interval_position"] = test["interval_position"].where(test["interval_position"].isin(POSITION_ORDER), test.apply(derive_interval_position, axis=1))
    test = test.reset_index(drop=True)
    test["row_index"] = np.arange(len(test))
    return test


def visualization_table(test: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "row_index": test["row_index"],
            "time": test["time"],
            "actual_ca_consumption": test["current_ca_consumption"],
            "recommended_ca_consumption_min": test["recommended_ca_consumption_min"],
            "recommended_ca_consumption_max": test["recommended_ca_consumption_max"],
            "recommended_ca_consumption_interval": test.apply(
                lambda row: f"[{row['recommended_ca_consumption_min']}, {row['recommended_ca_consumption_max']}]",
                axis=1,
            ),
            "recommended_ca_consumption_target": test.get("recommended_ca_consumption_target", pd.Series(np.nan, index=test.index)),
            "interval_position": test["interval_position"],
            "action_hint": test.get("action_hint", pd.Series("", index=test.index)),
            "confidence_level": test.get("confidence_level", pd.Series("", index=test.index)),
            "recommendation_status": test.get("recommendation_status", pd.Series("", index=test.index)),
            "band_hit": test.get("band_hit", pd.Series(pd.NA, index=test.index)),
            "direction_hit": test.get("direction_hit", pd.Series(pd.NA, index=test.index)),
            "target_hit_5pct": test.get("target_hit_5pct", pd.Series(pd.NA, index=test.index)),
            "t90": test.get("t90", pd.Series(np.nan, index=test.index)),
            "y_ok": test.get("y_ok", pd.Series(pd.NA, index=test.index)),
            "y_low": test.get("y_low", pd.Series(pd.NA, index=test.index)),
            "y_high": test.get("y_high", pd.Series(pd.NA, index=test.index)),
            "y_out_spec": test.get("y_out_spec", pd.Series(pd.NA, index=test.index)),
            "selected_rule_ids": test.get("selected_rule_ids", pd.Series("", index=test.index)),
        }
    )
    return table


def plot_coverage(test: pd.DataFrame, path: Path, time_axis: bool = False, title: str = "", subset: pd.DataFrame | None = None) -> None:
    data = test if subset is None else subset
    x = data["time"] if time_axis else data["row_index"]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(
        x,
        pd.to_numeric(data["recommended_ca_consumption_min"], errors="coerce"),
        pd.to_numeric(data["recommended_ca_consumption_max"], errors="coerce"),
        color="#FFCC80",
        alpha=0.45,
        label="推荐区间",
        step=None,
    )
    if "recommended_ca_consumption_target" in data.columns:
        ax.plot(x, data["recommended_ca_consumption_target"], color="#EF6C00", linestyle="--", linewidth=1.0, label="推荐目标中点")
    ax.plot(x, data["current_ca_consumption"], color="#263238", linewidth=0.9, alpha=0.65, label="真实钙单耗")
    for position in POSITION_ORDER:
        group = data[data["interval_position"].eq(position)]
        if group.empty:
            continue
        gx = group["time"] if time_axis else group["row_index"]
        ax.scatter(
            gx,
            group["current_ca_consumption"],
            s=16 if len(data) <= 500 else 9,
            alpha=0.8 if len(data) <= 500 else 0.55,
            color=POSITION_COLORS[position],
            label=position,
        )
    ax.set_title(title)
    ax.set_xlabel("时间" if time_axis else "测试集样本序号")
    ax.set_ylabel("钙单耗")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2, fontsize=9)
    if time_axis:
        fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_position_distribution(test: pd.DataFrame, path: Path) -> None:
    counts = test["interval_position"].value_counts().reindex(POSITION_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=[POSITION_COLORS[key] for key in counts.index])
    ax.set_title("测试集钙单耗相对推荐区间的位置分布")
    ax.set_xlabel("区间位置")
    ax.set_ylabel("样本数")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_risk_by_position(test: pd.DataFrame, path: Path) -> bool:
    required = ["y_ok", "y_high", "y_low", "y_out_spec"]
    if not all(column in test.columns for column in required):
        return False
    grouped = test.groupby("interval_position")[required].mean().reindex(POSITION_ORDER)
    grouped = grouped.rename(
        columns={
            "y_ok": "ok_rate",
            "y_high": "high_rate",
            "y_low": "low_rate",
            "y_out_spec": "out_spec_rate",
        }
    )
    if grouped.dropna(how="all").empty:
        return False
    fig, ax = plt.subplots(figsize=(10, 5.5))
    grouped.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("测试集不同区间位置的 T90 风险对比")
    ax.set_xlabel("区间位置")
    ax.set_ylabel("比例")
    ax.set_ylim(0, min(1.0, max(0.2, float(np.nanmax(grouped.to_numpy())) * 1.2)))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def above_band_focus_subset(test: pd.DataFrame) -> pd.DataFrame:
    above_indices = set(test.loc[test["interval_position"].eq("above_band"), "row_index"].astype(int).tolist())
    if not above_indices:
        return pd.DataFrame()
    focus = set()
    for idx in above_indices:
        for j in range(max(0, idx - 3), min(len(test), idx + 4)):
            focus.add(j)
    return test[test["row_index"].isin(sorted(focus))].copy()


def bool_mean(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return math.nan
    return float(valid.astype(bool).mean())


def summary_metrics(test: pd.DataFrame) -> dict[str, object]:
    counts = test["interval_position"].value_counts().to_dict()
    rec_width = pd.to_numeric(test["recommended_ca_consumption_max"], errors="coerce") - pd.to_numeric(test["recommended_ca_consumption_min"], errors="coerce")
    actual = pd.to_numeric(test["current_ca_consumption"], errors="coerce")
    summary: dict[str, object] = {
        "test_like_sample_count": int(len(test)),
        "recommended_sample_count": int((test.get("recommendation_status", "") == "recommended").sum()) if "recommendation_status" in test.columns else int(len(test)),
        "inside_band_count": int(counts.get("inside_band", 0)),
        "below_band_count": int(counts.get("below_band", 0)),
        "above_band_count": int(counts.get("above_band", 0)),
        "missing_count": int(counts.get("missing", 0)),
        "inside_band_rate": float(counts.get("inside_band", 0) / len(test)),
        "below_band_rate": float(counts.get("below_band", 0) / len(test)),
        "above_band_rate": float(counts.get("above_band", 0) / len(test)),
        "band_accuracy": bool_mean(test["band_hit"]) if "band_hit" in test.columns else None,
        "direction_accuracy": bool_mean(test["direction_hit"]) if "direction_hit" in test.columns else None,
        "target_accuracy_5pct": bool_mean(test["target_hit_5pct"]) if "target_hit_5pct" in test.columns else None,
        "actual_calcium_min": float(actual.min()) if actual.notna().any() else None,
        "actual_calcium_median": float(actual.median()) if actual.notna().any() else None,
        "actual_calcium_max": float(actual.max()) if actual.notna().any() else None,
        "recommended_interval_width_mean": float(rec_width.mean()) if rec_width.notna().any() else None,
        "recommended_interval_width_median": float(rec_width.median()) if rec_width.notna().any() else None,
        "recommended_interval_width_max": float(rec_width.max()) if rec_width.notna().any() else None,
        "actual_inside_interval_mean_boolean": float((test["interval_position"] == "inside_band").mean()),
    }
    if all(column in test.columns for column in ["y_ok", "y_high", "y_low", "y_out_spec"]):
        risk = {}
        for position, group in test.groupby("interval_position", sort=True):
            risk[position] = {
                "sample_count": int(len(group)),
                "ok_rate": float(group["y_ok"].mean()),
                "high_rate": float(group["y_high"].mean()),
                "low_rate": float(group["y_low"].mean()),
                "out_spec_rate": float(group["y_out_spec"].mean()),
            }
        summary["risk_by_interval_position"] = risk
    return summary


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
    title = section_title(doc_path, 22, "测试集钙单耗推荐区间覆盖可视化")
    summary = report["summary_metrics"]
    figures = report["generated_figures"]
    tables = report["generated_tables"]
    lines = [
        "",
        title,
        "",
        "本阶段仅用于离线可视化测试集真实钙单耗与推荐钙单耗区间的覆盖关系，不训练模型，不改变推荐规则，不生成自动控制建议。",
        "",
        "### 输入与输出",
        f"- 输入文件：`{report['replay_path']}`。",
        f"- 生成图像：{', '.join(figures)}。",
        f"- 生成表格：{', '.join(tables)}。",
        "",
        "### 摘要",
        f"- test_like 样本数：{report['test_like_sample_count']}。",
        f"- inside/below/above/missing：{summary['inside_band_count']} / {summary['below_band_count']} / {summary['above_band_count']} / {summary['missing_count']}。",
        f"- band accuracy：{summary.get('band_accuracy')}。",
        f"- direction accuracy：{summary.get('direction_accuracy')}。",
        "- 真实钙单耗落在推荐区间内，表示当前操作位于历史关系中推荐的钙单耗带；高于推荐区间的样本应作为可能高 T90 风险条件进行人工检查；低于推荐区间仅是诊断提示，不能直接变成自动增加指令。",
        "",
        "### 局限",
        "- 这是离线可视化，不构成因果证明。",
        "- 不执行自动控制，不写入 DCS，不推荐影子试验。",
        "",
    ]
    with doc_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    assumptions = [
        "This is offline visualization only.",
        "Recommended interval is not a fixed setpoint and not an automatic command.",
        "Below-band diagnostics must not become automatic increase commands.",
        "No DCS writeback or shadow trial is recommended.",
    ]
    configure_matplotlib()
    test = load_test_like(args, warnings)
    table = visualization_table(test)
    write_csv(args.data_output, table)
    table_copy = args.table_dir / "ca_interval_recommendation_visualization_table.csv"
    write_csv(table_copy, table)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    generated_figures: list[str] = []
    fig1 = args.figure_dir / "ca_interval_test_like_coverage.png"
    plot_coverage(test, fig1, time_axis=False, title="测试集真实钙单耗与推荐钙单耗区间覆盖图")
    generated_figures.append(str(fig1))
    fig2 = args.figure_dir / "ca_interval_test_like_coverage_time.png"
    plot_coverage(test, fig2, time_axis=True, title="测试集真实钙单耗与推荐钙单耗区间覆盖图（时间轴）")
    generated_figures.append(str(fig2))
    fig3 = args.figure_dir / "ca_interval_position_distribution.png"
    plot_position_distribution(test, fig3)
    generated_figures.append(str(fig3))
    fig4 = args.figure_dir / "ca_interval_position_t90_risk.png"
    if plot_risk_by_position(test, fig4):
        generated_figures.append(str(fig4))
    else:
        warnings.append("T90 target columns are missing; skipped risk-by-position figure.")
    fig5 = args.figure_dir / "ca_interval_test_like_coverage_zoom_first150.png"
    plot_coverage(test, fig5, time_axis=False, title="测试集前150个样本真实钙单耗与推荐区间覆盖图", subset=test.head(150).copy())
    generated_figures.append(str(fig5))
    focus = above_band_focus_subset(test)
    if not focus.empty:
        fig6 = args.figure_dir / "ca_interval_above_band_focus.png"
        plot_coverage(test, fig6, time_axis=False, title="above_band样本及邻近样本钙单耗覆盖聚焦图", subset=focus)
        generated_figures.append(str(fig6))
    else:
        warnings.append("No above_band samples; skipped above-band focus figure.")

    summary = summary_metrics(test)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "replay_path": str(args.replay),
        "metrics_path": str(args.metrics),
        "report_input_path": str(args.report_input),
        "figure_dir": str(args.figure_dir),
        "table_dir": str(args.table_dir),
        "data_output_path": str(args.data_output),
        "test_like_sample_count": int(len(test)),
        "summary_metrics": summary,
        "generated_figures": generated_figures,
        "generated_tables": [str(args.data_output), str(table_copy)],
        "warnings": warnings,
        "assumptions": assumptions,
        "recommended_next_step": "review_interval_coverage_visualizations",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(report), handle, ensure_ascii=False, indent=2)
    append_docs(args.doc, report)

    print("Calcium interval visualization summary")
    print(f"test_like sample count: {len(test)}")
    print(f"inside_band_count: {summary['inside_band_count']}")
    print(f"below_band_count: {summary['below_band_count']}")
    print(f"above_band_count: {summary['above_band_count']}")
    print("Generated figures:")
    for figure in generated_figures:
        print(f"  {figure}")
    print("Generated tables:")
    print(f"  {args.data_output}")
    print(f"  {table_copy}")
    print(f"Documentation appended: {args.doc}")


if __name__ == "__main__":
    main()
