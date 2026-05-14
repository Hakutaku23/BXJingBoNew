from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


T90_LOW = 8.20
T90_HIGH = 8.70
CLEAR_LOW = 8.10
CLEAR_OK_LOW = 8.30
CLEAR_OK_HIGH = 8.60
CLEAR_HIGH = 8.80

RAW_POINT_MAPPING = {
    "rubber_flow_2": {"tag": "B4-FIC-C51001.PV.F_CV", "output": "rubber_flow_2_win_60_mean", "required": True},
    "bromine_feed": {"tag": "B4-FIC-C51004.PV.CV", "output": "bromine_feed_win_60_mean", "required": True},
    "tank_rubber_conc": {"tag": "B4-AT-C50002A-BIIR.PV.CV", "output": "tank_rubber_conc_win_60_mean", "required": True},
    "r510a_temp": {"tag": "B4-TI-C51007A_S.PV.CV", "output": "r510a_temp_win_60_mean", "required": True},
    "r511a_temp": {"tag": "B4-TI-C51101A_S.PV.CV", "output": "r511a_temp_win_60_mean", "required": True},
    "r512a_temp": {"tag": "B4-TI-C51702A.PV.F_CV", "output": "r512a_temp_win_60_mean", "required": True},
    "ca_feed": {"tag": "B4-FIC-C51401.PV.F_CV", "output": "ca_feed", "required": True},
    "esbo_feed": {"tag": "B4-FIC-C51801.PV.F_CV", "output": "esbo_feed_win_60_mean", "required": True},
    "neutral_alkali_feed": {"tag": "B4-FIC-C51605.PV.F_CV", "output": "neutral_alkali_feed_win_60_mean", "required": True},
    "r513_temp": {"tag": "B4-TI-C51301_S.PV.CV", "output": "r513_temp_win_60_mean", "required": True},
    "r514_temp": {"tag": "B4-TI-C51401_S.PV.CV", "output": "r514_temp_win_60_mean", "required": True},
}

TIME_CANDIDATES = ["time", "时间", "日期", "采样时间", "检测时间", "分析时间", "化验时间", "取样时间", "sample_time", "test_time", "datetime"]
RUBBER_CANDIDATES = ["橡胶种类", "胶种", "橡胶类型", "产品类型", "产品名称", "物料名称", "牌号", "rubber_type", "material", "product"]
T90_CANDIDATES = ["t90", "T90", "T_90", "t 90", "门尼T90", "硫化T90", "T90值", "t´c(90),min", "t'c(90),min"]
LINE_CANDIDATES = ["线别", "生产线", "装置", "C/D/E", "line"]


# Override candidate lists with UTF-8 names as well as legacy mojibake names that
# may appear when older console output was captured.
TIME_CANDIDATES = ["time", "时间", "日期", "采样时间", "检测时间", "分析时间", "化验时间", "取样时间", "鏃堕棿", "鏃ユ湡", "閲囨牱鏃堕棿", "sample_time", "test_time", "datetime"]
RUBBER_CANDIDATES = ["橡胶种类", "胶种", "橡胶类型", "产品类型", "产品名称", "样品名称", "物料名称", "牌号", "姗¤兌绉嶇被", "鑳剁", "浜у搧鍚嶇О", "rubber_type", "material", "product", "sample_name"]
T90_CANDIDATES = ["t90", "T90", "T_90", "t 90", "门尼T90", "硫化T90", "T90值", "t´c(90),min", "t'c(90),min", "t麓c(90),min"]
LINE_CANDIDATES = ["线别", "生产线", "装置", "绾垮埆", "C/D/E", "line"]


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
    parser = argparse.ArgumentParser(description="Validate frozen V1 calcium safe-band package on future raw holdout data.")
    parser.add_argument("--future-dir", type=Path, default=Path("data/future"))
    parser.add_argument("--future-t90-files", type=Path, nargs="*", default=None)
    parser.add_argument("--target-rubber-type", type=str, default="卤化橡胶")
    parser.add_argument("--historical-reference", type=Path, default=Path("runs/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--deploy-dir", type=Path, default=Path("deploy/ca_safe_band_mvp"))
    parser.add_argument("--residence-minutes", type=int, default=174)
    parser.add_argument("--t90-match-tolerance-minutes", type=int, default=90)
    parser.add_argument("--eval-frequency-minutes", type=int, default=10)
    parser.add_argument("--min-valid-points", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/future_holdout_v1_validation"))
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--method-doc", type=Path, default=Path("docs/ca_safe_band_mvp_method_and_dataflow.md"))
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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def repair_mojibake_text(value: str) -> str:
    if value and ("鍗" in value or "姗" in value or "兌" in value):
        return "卤化橡胶"
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def normalize_tag(text: Any) -> str:
    value = str(text).strip().replace("/", ".").replace("-", "_").replace(".", "_")
    value = re.sub(r"_+", "_", value).upper().strip("_")
    return value


def tag_to_friendly() -> dict[str, str]:
    mapping = {}
    for friendly, meta in RAW_POINT_MAPPING.items():
        mapping[normalize_tag(meta["tag"])] = friendly
    return mapping


def read_txt_point(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    errors = []
    for encoding in ["utf-8", "gbk", "gb18030"]:
        try:
            raw = pd.read_csv(path, sep="\t", header=None, encoding=encoding, engine="python")
            if raw.shape[1] < 3:
                raw = pd.read_csv(path, sep=None, header=None, encoding=encoding, engine="python")
            if raw.shape[1] < 2:
                raise ValueError("fewer than two columns")
            if raw.shape[1] >= 3:
                tag_col, time_col, value_col = 0, 1, 2
            else:
                tag_col, time_col, value_col = None, 0, 1
            data = pd.DataFrame(
                {
                    "time": pd.to_datetime(raw.iloc[:, time_col], errors="coerce"),
                    "value": pd.to_numeric(raw.iloc[:, value_col], errors="coerce"),
                }
            )
            tag_value = str(raw.iloc[0, tag_col]) if tag_col is not None and len(raw) else path.stem
            data = data.dropna(subset=["time"]).sort_values("time")
            return data, {
                "file": str(path),
                "encoding": encoding,
                "row_count": int(len(data)),
                "tag_value": tag_value,
                "time_min": data["time"].min().isoformat() if len(data) else None,
                "time_max": data["time"].max().isoformat() if len(data) else None,
                "parse_error": None,
            }
        except Exception as exc:
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")
    return pd.DataFrame(columns=["time", "value"]), {"file": str(path), "parse_error": " | ".join(errors), "row_count": 0}


def parse_future_raw(future_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    txt_files = sorted(future_dir.glob("*.txt"))
    name_map = tag_to_friendly()
    frames = []
    parsed_meta = []
    unparsed = []
    detected = []
    for path in txt_files:
        data, meta = read_txt_point(path)
        parsed_meta.append(meta)
        if data.empty:
            unparsed.append(str(path))
            continue
        normalized_stem = normalize_tag(path.stem)
        friendly = name_map.get(normalized_stem)
        if friendly is None:
            normalized_tag = normalize_tag(meta.get("tag_value", ""))
            friendly = name_map.get(normalized_tag)
        if friendly is None:
            continue
        detected.append(friendly)
        point = data.groupby("time", as_index=False)["value"].mean().rename(columns={"value": friendly})
        frames.append(point)
    if not frames:
        merged = pd.DataFrame(columns=["time"])
    else:
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="time", how="outer")
        duplicate_count = int(merged["time"].duplicated().sum())
        merged = merged.groupby("time", as_index=False).mean(numeric_only=True).sort_values("time").reset_index(drop=True)
    required = [name for name, meta in RAW_POINT_MAPPING.items() if meta["required"]]
    missing_required = sorted(set(required) - set(detected))
    point_rows = []
    for friendly in sorted(set(detected)):
        series = pd.to_numeric(merged.get(friendly, pd.Series(dtype=float)), errors="coerce")
        point_rows.append(
            {
                "point": friendly,
                "non_null_count": int(series.notna().sum()),
                "missing_rate": float(series.isna().mean()) if len(merged) else None,
                "min": float(series.min()) if series.notna().any() else None,
                "median": float(series.median()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
            }
        )
    point_quality = pd.DataFrame(point_rows)
    if len(merged) > 1:
        diffs = merged["time"].sort_values().diff().dropna().dt.total_seconds() / 60.0
        sampling = {
            "median_minutes": float(diffs.median()),
            "q25_minutes": float(diffs.quantile(0.25)),
            "q75_minutes": float(diffs.quantile(0.75)),
            "max_minutes": float(diffs.max()),
        }
    else:
        sampling = {}
    report = {
        "file_count": len(txt_files),
        "parsed_file_count": int(sum(1 for m in parsed_meta if not m.get("parse_error"))),
        "unparsed_files": unparsed,
        "detected_points": sorted(set(detected)),
        "missing_required_points": missing_required,
        "time_min": merged["time"].min().isoformat() if len(merged) else None,
        "time_max": merged["time"].max().isoformat() if len(merged) else None,
        "row_count": int(len(merged)),
        "duplicate_timestamp_count": duplicate_count if frames else 0,
        "sampling_interval_summary": sampling,
        "missing_rate_by_point": {row["point"]: row["missing_rate"] for row in point_rows},
        "future_raw_quality_pass": bool(len(missing_required) == 0 and len(merged) > 0),
        "parsed_file_details": parsed_meta,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_dir / "future_raw_merged.parquet", index=False)
    merged.to_csv(output_dir / "future_raw_merged.csv", index=False, encoding="utf-8-sig")
    point_quality.to_csv(output_dir / "future_raw_point_quality.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "future_raw_quality_report.json", report)
    return merged, report, point_quality


def resolve_t90_files(future_dir: Path, requested: list[Path] | None) -> list[Path]:
    if requested:
        result = []
        for path in requested:
            if path.exists():
                result.append(path)
                continue
            fallback = None
            for suffix in [".xlsx", ".csv", ".xls"]:
                candidate = future_dir / f"{path.stem}{suffix}"
                if candidate.exists():
                    fallback = candidate
                    break
            result.append(fallback if fallback is not None else path)
        return result
    result = []
    for stem in ["2026.1", "2026.2", "2026.3C"]:
        for suffix in [".xlsx", ".csv", ".xls"]:
            path = future_dir / f"{stem}{suffix}"
            if path.exists():
                result.append(path)
                break
    return result


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value).strip())


def pick_column(columns: list[Any], candidates: list[str], fuzzy: bool = True) -> Any | None:
    normalized = {normalize_text(c).lower(): c for c in columns}
    for cand in candidates:
        key = normalize_text(cand).lower()
        if key in normalized:
            return normalized[key]
    if fuzzy:
        for col in columns:
            col_norm = normalize_text(col).lower()
            for cand in candidates:
                cand_norm = normalize_text(cand).lower()
                if cand_norm and cand_norm in col_norm:
                    return col
    return None


def read_t90_file(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {"file": str(path), "parsed": False, "error": None, "xls_engine_missing": False}
    try:
        if path.suffix.lower() == ".csv":
            last_error = None
            for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
                try:
                    df = pd.read_csv(path, encoding=enc)
                    meta["encoding"] = enc
                    meta["parsed"] = True
                    return df, meta
                except Exception as exc:
                    last_error = exc
            raise last_error if last_error else ValueError("csv parse failed")
        df = pd.read_excel(path)
        meta["parsed"] = True
        return df, meta
    except ImportError as exc:
        if path.suffix.lower() == ".xls" and "xlrd" in str(exc).lower():
            meta["xls_engine_missing"] = True
            meta["error"] = "当前环境无法读取 .xls，请将 2026.1.xls / 2026.2.xls / 2026.3C.xls 另存为 .xlsx 或 .csv 后重跑，或在厂方允许范围内补充读取依赖。"
        else:
            meta["error"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(), meta
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(), meta


def parse_future_t90(files: list[Path], target_rubber: str, output_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    all_rows = []
    halogen_rows = []
    details = []
    selected_time: dict[str, str | None] = {}
    selected_rubber: dict[str, str | None] = {}
    selected_t90: dict[str, str | None] = {}
    rubber_counts: dict[str, int] = {}
    missing_rubber = False
    missing_t90 = False
    xls_engine_missing = False
    warnings: list[str] = []
    for path in files:
        df, meta = read_t90_file(path)
        details.append(meta)
        if meta.get("xls_engine_missing"):
            xls_engine_missing = True
        if df.empty:
            continue
        time_col = pick_column(list(df.columns), TIME_CANDIDATES)
        rubber_col = pick_column(list(df.columns), RUBBER_CANDIDATES)
        t90_col = pick_column(list(df.columns), T90_CANDIDATES)
        selected_time[str(path)] = str(time_col) if time_col is not None else None
        selected_rubber[str(path)] = str(rubber_col) if rubber_col is not None else None
        selected_t90[str(path)] = str(t90_col) if t90_col is not None else None
        if rubber_col is None:
            missing_rubber = True
            warnings.append(f"missing_rubber_type_column: {path}")
            continue
        if t90_col is None:
            missing_t90 = True
            warnings.append(f"missing_t90_column: {path}")
            continue
        if time_col is None:
            warnings.append(f"missing_time_column: {path}")
            continue
        part = pd.DataFrame(
            {
                "source_file": str(path),
                "time": pd.to_datetime(df[time_col], errors="coerce"),
                "rubber_type": df[rubber_col].map(normalize_text),
                "t90": pd.to_numeric(df[t90_col], errors="coerce"),
            }
        )
        part = part.dropna(subset=["time"])
        all_rows.append(part)
        for value, count in part["rubber_type"].value_counts(dropna=False).items():
            rubber_counts[str(value)] = rubber_counts.get(str(value), 0) + int(count)
        target_norm = normalize_text(target_rubber)
        keep = part["rubber_type"].str.contains(target_norm, na=False) & ~part["rubber_type"].str.contains("氯丁基", na=False)
        keep &= part["t90"].notna()
        halogen_rows.append(part.loc[keep].copy())
    all_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["source_file", "time", "rubber_type", "t90"])
    halogen_df = pd.concat(halogen_rows, ignore_index=True) if halogen_rows else pd.DataFrame(columns=["source_file", "time", "rubber_type", "t90"])
    if not halogen_df.empty:
        halogen_df = halogen_df.sort_values("time").reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(output_dir / "future_t90_raw_all.parquet", index=False)
    all_df.to_csv(output_dir / "future_t90_raw_all.csv", index=False, encoding="utf-8-sig")
    if not halogen_df.empty:
        halogen_df.to_parquet(output_dir / "future_t90_halogen_only.parquet", index=False)
        halogen_df.to_csv(output_dir / "future_t90_halogen_only.csv", index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(
        [
            {"metric": "total_t90_rows", "value": len(all_df)},
            {"metric": "halogen_t90_rows", "value": len(halogen_df)},
            {"metric": "excluded_non_halogen_rows", "value": max(len(all_df) - len(halogen_df), 0)},
            {"metric": "future_t90_filter_pass", "value": bool(len(halogen_df) > 0 and not missing_rubber)},
        ]
    )
    summary_df.to_csv(table_dir / "future_t90_halogen_filter_summary.csv", index=False, encoding="utf-8-sig")
    q = halogen_df["t90"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict() if not halogen_df.empty else {}
    report = {
        "future_t90_files_requested": [str(p) for p in files],
        "future_t90_files_found": [str(p) for p in files if p.exists()],
        "future_t90_files_parsed": [d["file"] for d in details if d.get("parsed")],
        "future_t90_files_failed": [d for d in details if not d.get("parsed")],
        "xls_engine_missing": xls_engine_missing,
        "selected_time_column_by_file": selected_time,
        "selected_rubber_type_column_by_file": selected_rubber,
        "selected_t90_column_by_file": selected_t90,
        "total_t90_rows": int(len(all_df)),
        "halogen_t90_rows": int(len(halogen_df)),
        "excluded_non_halogen_rows": int(max(len(all_df) - len(halogen_df), 0)),
        "rubber_type_value_counts": rubber_counts,
        "t90_time_min": halogen_df["time"].min().isoformat() if len(halogen_df) else None,
        "t90_time_max": halogen_df["time"].max().isoformat() if len(halogen_df) else None,
        "t90_min": float(q.get(0.0)) if q else None,
        "t90_q25": float(q.get(0.25)) if q else None,
        "t90_median": float(q.get(0.5)) if q else None,
        "t90_q75": float(q.get(0.75)) if q else None,
        "t90_max": float(q.get(1.0)) if q else None,
        "missing_rubber_type_column": missing_rubber,
        "missing_t90_column": missing_t90,
        "future_t90_available": bool(len(halogen_df) > 0 and not missing_t90),
        "future_t90_filter_pass": bool(len(halogen_df) > 0 and not missing_rubber),
        "warnings": warnings,
    }
    write_json(output_dir / "future_t90_parse_report.json", report)
    return all_df, halogen_df, report


def build_future_runtime_features(raw: pd.DataFrame, eval_freq_min: int, min_valid: int, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if raw.empty or "time" not in raw.columns:
        raise ValueError("Future raw data is empty or missing time.")
    data = raw.copy().sort_values("time").drop_duplicates("time", keep="last").set_index("time")
    data = data.apply(pd.to_numeric, errors="coerce")
    start = data.index.min() + pd.Timedelta(minutes=60)
    end = data.index.max()
    eval_times = pd.date_range(start=start.ceil(f"{eval_freq_min}min"), end=end.floor(f"{eval_freq_min}min"), freq=f"{eval_freq_min}min")
    features = pd.DataFrame(index=data.index)
    insufficient_counts: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    rolling_count = data.rolling("60min", min_periods=1).count()
    for friendly, meta in RAW_POINT_MAPPING.items():
        if friendly == "ca_feed":
            continue
        out_col = meta["output"]
        if friendly not in data.columns:
            features[out_col] = np.nan
            missing_counts[out_col] = len(eval_times)
            continue
        mean = data[friendly].rolling("60min", min_periods=min_valid).mean()
        features[out_col] = mean
        insufficient_counts[out_col] = int((rolling_count[friendly].reindex(eval_times, method="nearest") < min_valid).sum()) if len(eval_times) else 0
    if "ca_feed" in data.columns and "rubber_flow_2" in data.columns:
        ratio = data["ca_feed"].where(data["rubber_flow_2"].notna() & (data["rubber_flow_2"] != 0)) / data["rubber_flow_2"].where(data["rubber_flow_2"].notna() & (data["rubber_flow_2"] != 0))
        features["ca_per_rubber_flow_win_60_mean"] = ratio.rolling("60min", min_periods=min_valid).mean()
        features["current_ca_consumption"] = features["ca_per_rubber_flow_win_60_mean"]
        insufficient_counts["ca_per_rubber_flow_win_60_mean"] = int((ratio.rolling("60min", min_periods=1).count().reindex(eval_times, method="nearest") < min_valid).sum()) if len(eval_times) else 0
    else:
        features["ca_per_rubber_flow_win_60_mean"] = np.nan
        features["current_ca_consumption"] = np.nan
        missing_counts["ca_per_rubber_flow_win_60_mean"] = len(eval_times)
    features["output_ir_corrected_offset_20_win_15_std"] = np.nan
    runtime = features.reindex(eval_times, method="nearest").reset_index().rename(columns={"index": "time"})
    runtime["time"] = pd.to_datetime(runtime["time"])
    required_feature_cols = [meta["output"] for name, meta in RAW_POINT_MAPPING.items() if name != "ca_feed"] + ["ca_per_rubber_flow_win_60_mean"]
    missing_required = runtime[required_feature_cols].isna()
    runtime["feature_quality"] = np.where(missing_required.any(axis=1), "incomplete", "ok")
    runtime["missing_raw_columns"] = ""
    runtime["insufficient_window_features"] = missing_required.apply(lambda row: ";".join(row.index[row].tolist()), axis=1)
    runtime["warning_flags"] = np.where(missing_required.any(axis=1), "insufficient_window_features;optional_ir_missing", "optional_ir_missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime.to_parquet(output_dir / "future_runtime_features.parquet", index=False)
    runtime.to_csv(output_dir / "future_runtime_features.csv", index=False, encoding="utf-8-sig")
    quality_by_time = runtime[["time", "feature_quality", "insufficient_window_features", "warning_flags"]].copy()
    quality_by_time.to_csv(output_dir / "future_feature_quality_by_time.csv", index=False, encoding="utf-8-sig")
    report = {
        "evaluation_row_count": int(len(runtime)),
        "feature_valid_row_count": int((runtime["feature_quality"] == "ok").sum()),
        "invalid_row_count": int((runtime["feature_quality"] != "ok").sum()),
        "missing_feature_counts": missing_required.sum().astype(int).to_dict(),
        "insufficient_window_counts": insufficient_counts,
        "optional_ir_available_rate": 0.0,
        "feature_valid_rate": float((runtime["feature_quality"] == "ok").mean()) if len(runtime) else 0.0,
        "feature_quality_pass": bool(len(runtime) > 0 and (runtime["feature_quality"] == "ok").mean() >= 0.80),
    }
    write_json(output_dir / "future_feature_quality_report.json", report)
    return runtime, report, quality_by_time


def load_recommender(deploy_dir: Path) -> Any:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(deploy_dir.resolve()))
    spec = importlib.util.spec_from_file_location("future_safe_band_interface", deploy_dir / "interface.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load interface.py from {deploy_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SafeBandRecommender(deploy_dir, mode="production").load()


def score_future(runtime: pd.DataFrame, deploy_dir: Path, output_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    recommender = load_recommender(deploy_dir)
    rows = runtime.to_dict(orient="records")
    preds = recommender.predict_batch(rows, mode="production")
    pred_df = pd.DataFrame(preds)
    replay = pd.concat([runtime.reset_index(drop=True), pred_df.add_prefix("pred_")], axis=1)
    rename = {
        "pred_recommended_ca_consumption_min": "recommended_ca_consumption_min",
        "pred_recommended_ca_consumption_max": "recommended_ca_consumption_max",
        "pred_recommended_ca_consumption_target": "recommended_ca_consumption_target",
        "pred_interval_position": "interval_position",
        "pred_action_hint": "action_hint",
        "pred_action_visibility": "action_visibility",
        "pred_engineering_review_required": "engineering_review_required",
        "pred_input_valid": "input_valid",
        "pred_missing_required_features": "missing_required_features",
        "pred_warning_flags": "runtime_warning_flags",
    }
    replay = replay.rename(columns=rename)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay.to_parquet(output_dir / "future_v1_recommendation_replay.parquet", index=False)
    replay.to_csv(output_dir / "future_v1_recommendation_replay.csv", index=False, encoding="utf-8-sig")
    valid = replay["recommended_ca_consumption_min"].notna() if "recommended_ca_consumption_min" in replay else pd.Series(False, index=replay.index)
    pos_counts = replay.get("interval_position", pd.Series(dtype=object)).value_counts(dropna=False).to_dict()
    vis_counts = replay.get("action_visibility", pd.Series(dtype=object)).value_counts(dropna=False).to_dict()
    summary_df = pd.DataFrame(
        [
            {"metric": "scored_row_count", "value": len(replay)},
            {"metric": "recommendation_coverage", "value": float(valid.mean()) if len(replay) else None},
            {"metric": "inside_band_count", "value": int(pos_counts.get("inside_band", 0))},
            {"metric": "above_band_count", "value": int(pos_counts.get("above_band", 0))},
            {"metric": "below_band_count", "value": int(pos_counts.get("below_band", 0))},
            {"metric": "manual_review_required_count", "value": int(vis_counts.get("manual_review_required", 0))},
        ]
    )
    table_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(table_dir / "future_v1_recommendation_distribution_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "scored_row_count": int(len(replay)),
        "recommendation_coverage": float(valid.mean()) if len(replay) else None,
        "no_recommendation_count": int((replay.get("action_visibility", "") == "no_recommendation").sum()) if "action_visibility" in replay else None,
        "input_invalid_count": int((~replay.get("input_valid", pd.Series(True, index=replay.index)).astype(bool)).sum()) if len(replay) else 0,
        "inside_band_count": int(pos_counts.get("inside_band", 0)),
        "above_band_count": int(pos_counts.get("above_band", 0)),
        "below_band_count": int(pos_counts.get("below_band", 0)),
        "manual_review_required_count": int(vis_counts.get("manual_review_required", 0)),
        "diagnostic_only_count": int(vis_counts.get("diagnostic_only", 0)),
        "monitor_only_count": int(vis_counts.get("monitor_only", 0)),
        "missing_required_features_summary": replay.get("missing_required_features", pd.Series(dtype=object)).astype(str).value_counts().head(20).to_dict(),
        "warning_flags_summary": replay.get("runtime_warning_flags", pd.Series(dtype=object)).astype(str).value_counts().head(20).to_dict(),
        "recommended_ca_consumption_distribution": distribution(replay.get("recommended_ca_consumption_target")),
        "current_ca_consumption_distribution": distribution(replay.get("current_ca_consumption")),
        "future_replay_pass": bool(len(replay) > 0 and valid.mean() >= 0.50),
    }
    write_json(output_dir / "future_v1_recommendation_distribution_report.json", report)
    return replay, report


def distribution(series: pd.Series | None) -> dict[str, float | None]:
    if series is None:
        return {}
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None}
    return {k: float(v) for k, v in {"min": s.min(), "q25": s.quantile(0.25), "median": s.median(), "q75": s.quantile(0.75), "max": s.max()}.items()}


def add_t90_targets(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    t90 = pd.to_numeric(data["t90"], errors="coerce")
    data["y_ok"] = ((t90 >= T90_LOW) & (t90 <= T90_HIGH)).astype(int)
    data["y_low"] = (t90 < T90_LOW).astype(int)
    data["y_high"] = (t90 > T90_HIGH).astype(int)
    data["y_out_spec"] = ((t90 < T90_LOW) | (t90 > T90_HIGH)).astype(int)
    data["clear_label"] = np.select(
        [t90.between(CLEAR_OK_LOW, CLEAR_OK_HIGH, inclusive="both"), t90 <= CLEAR_LOW, t90 >= CLEAR_HIGH],
        ["clear_ok", "clear_low", "clear_high"],
        default="uncertain_boundary",
    )
    return data


def summarize_position_risk(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["outside_group"] = np.where(work["interval_position"].isin(["above_band", "below_band"]), "outside_band", work["interval_position"])
    for pos in ["inside_band", "outside_band", "above_band", "below_band"]:
        sub = work[work["outside_group"].eq(pos)] if pos == "outside_band" else work[work["interval_position"].eq(pos)]
        if sub.empty:
            continue
        rows.append(
            {
                "interval_position": pos,
                "sample_count": int(len(sub)),
                "ok_rate": float(sub["y_ok"].mean()),
                "high_rate": float(sub["y_high"].mean()),
                "low_rate": float(sub["y_low"].mean()),
                "out_spec_rate": float(sub["y_out_spec"].mean()),
                "mean_t90": float(sub["t90"].mean()),
            }
        )
    return pd.DataFrame(rows)


def align_t90(replay: pd.DataFrame, t90: pd.DataFrame, t90_report: dict[str, Any], residence: int, tolerance: int, output_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if t90.empty:
        status = "t90_parse_failed" if (t90_report.get("xls_engine_missing") or t90_report.get("future_t90_files_failed")) else "pending_lims_labels"
        report = {"future_t90_available": False, "future_t90_validation_status": status, "aligned_sample_count": 0}
        write_json(output_dir / "future_t90_backfill_validation_report.json", report)
        pd.DataFrame().to_csv(table_dir / "future_t90_backfill_validation_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(table_dir / "future_t90_clear_label_validation_summary.csv", index=False, encoding="utf-8-sig")
        return pd.DataFrame(), report, pd.DataFrame(), pd.DataFrame()
    rec = replay.copy()
    rec["time"] = pd.to_datetime(rec["time"], errors="coerce")
    rec["quality_time"] = rec["time"] + pd.to_timedelta(residence, unit="m")
    labels = t90[["time", "t90", "rubber_type", "source_file"]].copy().sort_values("time")
    aligned = pd.merge_asof(
        rec.sort_values("quality_time"),
        labels.rename(columns={"time": "t90_time"}).sort_values("t90_time"),
        left_on="quality_time",
        right_on="t90_time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance),
    )
    aligned = aligned.dropna(subset=["t90"]).copy()
    aligned["t90_match_delta_minutes"] = (aligned["t90_time"] - aligned["quality_time"]).dt.total_seconds().abs() / 60.0
    aligned = add_t90_targets(aligned)
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(output_dir / "future_t90_backfill_aligned.parquet", index=False)
    aligned.to_csv(output_dir / "future_t90_backfill_aligned.csv", index=False, encoding="utf-8-sig")
    risk = summarize_position_risk(aligned)
    risk.to_csv(table_dir / "future_t90_backfill_validation_summary.csv", index=False, encoding="utf-8-sig")
    clear = aligned[aligned["clear_label"].isin(["clear_ok", "clear_low", "clear_high"])].copy()
    clear_risk = summarize_position_risk(clear) if len(clear) >= 20 else pd.DataFrame()
    clear_risk.to_csv(table_dir / "future_t90_clear_label_validation_summary.csv", index=False, encoding="utf-8-sig")
    inside = risk[risk["interval_position"].eq("inside_band")]
    outside = risk[risk["interval_position"].eq("outside_band")]
    aligned_count = len(aligned)
    inside_count = int(inside["sample_count"].iloc[0]) if not inside.empty else 0
    outside_count = int(outside["sample_count"].iloc[0]) if not outside.empty else 0
    guardrail = bool(
        aligned_count >= 30
        and inside_count >= 10
        and outside_count >= 10
        and not inside.empty
        and not outside.empty
        and float(inside["high_rate"].iloc[0]) <= float(outside["high_rate"].iloc[0])
        and float(inside["out_spec_rate"].iloc[0]) <= float(outside["out_spec_rate"].iloc[0])
    )
    report = {
        "future_t90_available": True,
        "future_t90_validation_status": "available",
        "aligned_sample_count": int(aligned_count),
        "inside_band_count": inside_count,
        "outside_band_count": outside_count,
        "risk_by_interval_position": risk.to_dict(orient="records"),
        "inside_vs_outside_high_rate_delta": rate_delta(risk, "inside_band", "outside_band", "high_rate"),
        "inside_vs_outside_out_spec_rate_delta": rate_delta(risk, "inside_band", "outside_band", "out_spec_rate"),
        "above_vs_inside_high_rate_delta": rate_delta(risk, "above_band", "inside_band", "high_rate"),
        "recommendation_coverage_on_aligned": float(aligned["recommended_ca_consumption_min"].notna().mean()) if aligned_count else None,
        "future_holdout_risk_guardrail_pass": guardrail,
        "clear_sample_count": int(len(clear)),
        "uncertain_boundary_rate": float((aligned["clear_label"] == "uncertain_boundary").mean()) if aligned_count else None,
        "clear_label_risk_by_interval_position": clear_risk.to_dict(orient="records"),
    }
    write_json(output_dir / "future_t90_backfill_validation_report.json", report)
    return aligned, report, risk, clear_risk


def rate_delta(risk: pd.DataFrame, left: str, right: str, metric: str) -> float | None:
    l = risk[risk["interval_position"].eq(left)]
    r = risk[risk["interval_position"].eq(right)]
    if l.empty or r.empty:
        return None
    return float(l[metric].iloc[0] - r[metric].iloc[0])


def resolve_historical(path: Path) -> Path | None:
    if path.exists():
        return path
    if Path("runs").exists():
        matches = sorted(Path("runs").rglob("t90_ca_feature_dataset.parquet"))
        if matches:
            return matches[0]
        matches = sorted(Path("runs").rglob("final_monitor_dry_run.parquet"))
        if matches:
            return matches[0]
    return None


def compare_historical(runtime: pd.DataFrame, historical_path: Path | None, output_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = [
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
    if historical_path is None:
        drift = pd.DataFrame()
        report = {"historical_reference_available": False, "future_within_historical_support": None}
    else:
        hist = pd.read_parquet(historical_path)
        rows = []
        for feature in features:
            if feature not in runtime.columns or feature not in hist.columns:
                continue
            h = pd.to_numeric(hist[feature], errors="coerce").dropna()
            f = pd.to_numeric(runtime[feature], errors="coerce").dropna()
            if h.empty or f.empty:
                continue
            hq = h.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            fq = f.quantile([0.25, 0.5, 0.75])
            out_rate = float(((f < hq.loc[0.01]) | (f > hq.loc[0.99])).mean())
            rows.append(
                {
                    "feature": feature,
                    "historical_median": float(hq.loc[0.5]),
                    "future_median": float(fq.loc[0.5]),
                    "historical_q25": float(hq.loc[0.25]),
                    "future_q25": float(fq.loc[0.25]),
                    "historical_q75": float(hq.loc[0.75]),
                    "future_q75": float(fq.loc[0.75]),
                    "out_of_historical_range_rate": out_rate,
                    "psi_like_drift_score": psi_like(h, f),
                    "future_within_historical_support": bool(out_rate <= 0.20),
                }
            )
        drift = pd.DataFrame(rows)
        report = {
            "historical_reference_available": True,
            "historical_reference_path": str(historical_path),
            "feature_count_compared": int(len(drift)),
            "max_out_of_historical_range_rate": float(drift["out_of_historical_range_rate"].max()) if not drift.empty else None,
            "max_psi_like_drift_score": float(drift["psi_like_drift_score"].max()) if not drift.empty else None,
            "future_within_historical_support": bool(not drift.empty and drift["future_within_historical_support"].mean() >= 0.70),
        }
    drift.to_csv(output_dir / "future_vs_historical_feature_drift.csv", index=False, encoding="utf-8-sig")
    drift.to_csv(table_dir / "future_vs_historical_feature_drift_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "future_vs_historical_feature_drift_report.json", report)
    return drift, report


def psi_like(hist: pd.Series, fut: pd.Series, bins: int = 10) -> float | None:
    try:
        edges = np.unique(np.quantile(hist, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return None
        h_counts, _ = np.histogram(hist, bins=edges)
        f_counts, _ = np.histogram(fut, bins=edges)
        h_pct = np.maximum(h_counts / max(h_counts.sum(), 1), 1e-6)
        f_pct = np.maximum(f_counts / max(f_counts.sum(), 1), 1e-6)
        return float(np.sum((f_pct - h_pct) * np.log(f_pct / h_pct)))
    except Exception:
        return None


def plot_figures(raw_quality: pd.DataFrame, replay: pd.DataFrame, drift: pd.DataFrame, risk: pd.DataFrame, t90_all: pd.DataFrame, figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    test = replay.copy().reset_index(drop=True)
    x = np.arange(len(test))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(x, pd.to_numeric(test["recommended_ca_consumption_min"], errors="coerce"), pd.to_numeric(test["recommended_ca_consumption_max"], errors="coerce"), alpha=0.35, color="#90CAF9", label="推荐安全带")
    ax.plot(x, pd.to_numeric(test["current_ca_consumption"], errors="coerce"), color="#333333", linewidth=0.8, label="实际钙单耗")
    ax.set_title("future 新数据钙单耗与推荐安全带覆盖图")
    ax.set_xlabel("future 样本序号")
    ax.set_ylabel("钙单耗")
    ax.legend()
    fig.tight_layout()
    path = figure_dir / "future_v1_ca_band_coverage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path))

    counts = replay["interval_position"].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#4E79A7")
    ax.set_title("future 新数据区间位置分布")
    ax.tick_params(axis="x", rotation=20)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())), ha="center", va="bottom")
    fig.tight_layout()
    path = figure_dir / "future_v1_interval_position_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path))

    if not raw_quality.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        q = raw_quality.sort_values("missing_rate", ascending=False)
        ax.bar(q["point"], q["missing_rate"], color="#F28E2B")
        ax.set_title("future 新数据关键点位缺失率")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("缺失率")
        fig.tight_layout()
        path = figure_dir / "future_v1_feature_missingness.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))

    if not drift.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        idx = np.arange(len(drift))
        ax.bar(idx - 0.2, drift["historical_median"], width=0.4, label="历史中位数")
        ax.bar(idx + 0.2, drift["future_median"], width=0.4, label="future中位数")
        ax.set_xticks(idx)
        ax.set_xticklabels(drift["feature"], rotation=45, ha="right")
        ax.set_title("future 与历史工况特征分布对比")
        ax.legend()
        fig.tight_layout()
        path = figure_dir / "future_vs_historical_feature_drift.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))

    if not risk.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        idx = np.arange(len(risk))
        width = 0.22
        for offset, metric in [(-width, "ok_rate"), (0, "high_rate"), (width, "low_rate")]:
            ax.bar(idx + offset, risk[metric], width=width, label=metric)
        ax.set_xticks(idx)
        ax.set_xticklabels(risk["interval_position"], rotation=20)
        ax.set_title("future 卤化橡胶 T90 回填下区间内外风险对比")
        ax.legend()
        fig.tight_layout()
        path = figure_dir / "future_t90_backfill_risk_summary.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))

    if not t90_all.empty:
        counts = t90_all["rubber_type"].value_counts().head(12)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(counts.index.astype(str), counts.values, color="#59A14F")
        ax.set_title("future T90 胶种过滤统计")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        path = figure_dir / "future_t90_rubber_type_filter_summary.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def update_method_doc(path: Path) -> None:
    title = "## future holdout 新数据与卤化橡胶 T90 回填验证方案"
    section = f"""

{title}

历史数据仅作为上线前参考库，用于理解规则、边界和特征分布；`data/future/` 下的新 DCS 与 LIMS 数据是独立 future holdout，不能在评分前更新 artifact、q33/q66 边界、安全带或推荐逻辑。

future DCS 原始 txt 文件先合并为时间序列，再按在线运行语义构造 trailing `[t_now-60min, t_now]` 窗口特征；钙单耗由 `ca_feed / rubber_flow_2` 在 60 分钟窗口内逐点计算后取均值。IR 输入为可选项，缺失不阻断评分。

future T90 来自 `2026.1.xls`、`2026.2.xls`、`2026.3C.xls` 或其转换后的 `.xlsx/.csv`。T90 验证只保留胶种为 `卤化橡胶` 的记录，排除 `氯丁基橡胶` 和其他胶种。若无法识别胶种列，不能静默使用 T90。

回填验证使用推荐时刻加停留时间进行对齐：`quality_time = recommendation_time + 174min`，在容差窗口内匹配最近的后续 LIMS T90。由于 T90 人工测量和记录精度约为 0.1，同时报告清晰标签指标：`8.30-8.60` 为 clear_ok，`<=8.10` 为 clear_low，`>=8.80` 为 clear_high，其余为边界不确定。

该流程只验证冻结 V1 监测链路，不自动控制、不写回 DCS、不改变推荐规则。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else "# 稳定钙单耗安全带 MVP 方法与数据流\n"
    if title in text:
        text = text.split(title)[0].rstrip() + section
    else:
        text = text.rstrip() + section
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def next_section_number(doc_path: Path, preferred: int) -> int:
    if not doc_path.exists():
        return preferred
    used = set()
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            num = line[3:].split(".", 1)[0].strip()
            if num.isdigit():
                used.add(int(num))
    n = preferred
    while n in used:
        n += 1
    return n


def append_doc(path: Path, report: dict[str, Any]) -> None:
    n = next_section_number(path, 40)
    section = f"""

## {n}. future holdout 新数据与卤化橡胶 T90 回填验证

本阶段使用 `data/future/` 作为完全未见过的 future holdout，验证冻结 V1 monitor-only 钙单耗安全带运行链路。T90 文件包括 `2026.1.xls`、`2026.2.xls`、`2026.3C.xls`；验证原则是只使用胶种为 `卤化橡胶` 的 T90，排除 `氯丁基橡胶` 及其他胶种。

原始 DCS 解析：{report.get('raw_quality_summary')}。

T90 解析与过滤：{report.get('t90_parse_summary')}。

运行特征质量：{report.get('feature_quality_summary')}。

推荐 replay 摘要：{report.get('recommendation_distribution_summary')}。

停留时间回填验证：{report.get('future_t90_backfill_validation_summary')}。

清晰标签不确定性结果：{report.get('clear_label_validation_summary')}。

future 与历史特征漂移：{report.get('future_vs_historical_drift_summary')}。

validation_mode：`{report.get('validation_mode')}`；recommended_next_step：`{report.get('recommended_next_step')}`。

局限性：`.xls` 读取可能需要转换为 `.xlsx/.csv` 或厂方允许的读取依赖；T90 测量误差约 0.1；future raw 点位映射依赖文件命名和格式；本阶段仅 monitor-only，不自动控制，不写回 DCS。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(section)


def decide(raw_report: dict[str, Any], t90_report: dict[str, Any], feature_report: dict[str, Any], rec_report: dict[str, Any], backfill: dict[str, Any], drift: dict[str, Any]) -> tuple[str, str]:
    if not raw_report.get("future_raw_quality_pass"):
        return "failed_raw_parsing", "fix_future_data_mapping"
    if not rec_report.get("future_replay_pass"):
        return "failed_runtime_scoring", "stop_due_to_runtime_failure"
    if feature_report.get("feature_valid_rate", 0.0) < 0.50:
        return "failed_feature_construction", "fix_future_data_mapping"
    if t90_report.get("xls_engine_missing") or (not t90_report.get("future_t90_available")):
        return "runtime_plus_t90_parse_failed", "fix_future_t90_parsing_or_convert_xls"
    if drift.get("future_within_historical_support") is False:
        return "runtime_plus_t90_backfill", "investigate_future_distribution_shift"
    if backfill.get("future_holdout_risk_guardrail_pass"):
        return "runtime_plus_t90_backfill", "prepare_monitor_only_factory_test_with_future_evidence"
    return "runtime_plus_t90_backfill", "human_review_future_runtime_replay"


def main() -> None:
    configure_chinese_font()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    target_rubber = repair_mojibake_text(args.target_rubber_type)

    raw, raw_report, point_quality = parse_future_raw(args.future_dir, args.output_dir)
    t90_files = resolve_t90_files(args.future_dir, args.future_t90_files)
    t90_all, t90_halogen, t90_report = parse_future_t90(t90_files, target_rubber, args.output_dir, args.table_dir)
    runtime, feature_report, _ = build_future_runtime_features(raw, args.eval_frequency_minutes, args.min_valid_points, args.output_dir)
    replay, rec_report = score_future(runtime, args.deploy_dir, args.output_dir, args.table_dir)
    aligned, backfill_report, risk_table, clear_risk = align_t90(replay, t90_halogen, t90_report, args.residence_minutes, args.t90_match_tolerance_minutes, args.output_dir, args.table_dir)
    hist_path = resolve_historical(args.historical_reference)
    drift, drift_report = compare_historical(runtime, hist_path, args.output_dir, args.table_dir)
    figures = plot_figures(point_quality, replay, drift, risk_table, t90_all, args.figure_dir)

    validation_mode, next_step = decide(raw_report, t90_report, feature_report, rec_report, backfill_report, drift_report)
    safety = {
        "monitor_only": True,
        "automatic_control": False,
        "dcs_writeback": False,
        "future_data_updates_artifact": False,
        "future_data_updates_boundaries": False,
    }
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "future_dir": str(args.future_dir),
        "future_t90_files": [str(p) for p in t90_files],
        "target_rubber_type": target_rubber,
        "deploy_dir": str(args.deploy_dir),
        "historical_reference_path": str(hist_path) if hist_path else None,
        "output_dir": str(args.output_dir),
        "raw_quality_summary": raw_report,
        "t90_parse_summary": t90_report,
        "feature_quality_summary": feature_report,
        "recommendation_distribution_summary": rec_report,
        "future_t90_available": t90_report.get("future_t90_available"),
        "future_t90_filter_pass": t90_report.get("future_t90_filter_pass"),
        "future_t90_backfill_validation_summary": backfill_report,
        "clear_label_validation_summary": {
            "clear_sample_count": backfill_report.get("clear_sample_count"),
            "uncertain_boundary_rate": backfill_report.get("uncertain_boundary_rate"),
            "risk_by_interval_position": clear_risk.to_dict(orient="records") if not clear_risk.empty else [],
        },
        "future_vs_historical_drift_summary": drift_report,
        "safety_check_summary": safety,
        "validation_mode": validation_mode,
        "holdout_principle": "Historical data is prelaunch reference only; future holdout data does not update frozen V1 before scoring.",
        "limitations": [
            ".xls reading may require conversion to xlsx/csv if xlrd is unavailable.",
            "T90 measurement error is about 0.1, so clear-label metrics are reported separately when labels are available.",
            "Future raw point mapping depends on file naming and txt format.",
            "Monitor-only validation; no automatic control and no DCS writeback.",
        ],
        "generated_figures": figures,
        "recommended_next_step": next_step,
    }
    write_json(args.output_dir / "future_holdout_v1_validation_report.json", report)
    update_method_doc(args.method_doc)
    append_doc(args.doc, report)

    print("Future holdout V1 validation summary")
    print("raw_quality_pass:", raw_report.get("future_raw_quality_pass"))
    print("missing_required_points:", raw_report.get("missing_required_points"))
    print("future_t90_available:", t90_report.get("future_t90_available"))
    print("future_t90_filter_pass:", t90_report.get("future_t90_filter_pass"))
    print("halogen_t90_rows:", t90_report.get("halogen_t90_rows"))
    print("feature_quality_pass:", feature_report.get("feature_quality_pass"))
    print("recommendation_coverage:", rec_report.get("recommendation_coverage"))
    print("inside/above/below:", rec_report.get("inside_band_count"), rec_report.get("above_band_count"), rec_report.get("below_band_count"))
    print("validation_mode:", validation_mode)
    print("recommended_next_step:", next_step)
    print("No generated outputs were written under data/.")


if __name__ == "__main__":
    main()
