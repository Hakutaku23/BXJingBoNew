from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import math
import re
import shutil
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

TIME_CANDIDATES = ["time", "\u65f6\u95f4", "\u65e5\u671f", "\u91c7\u6837\u65f6\u95f4", "\u68c0\u6d4b\u65f6\u95f4", "\u5206\u6790\u65f6\u95f4", "\u5316\u9a8c\u65f6\u95f4", "\u53d6\u6837\u65f6\u95f4", "sample_time", "test_time", "datetime"]
RUBBER_CANDIDATES = ["\u6a61\u80f6\u79cd\u7c7b", "\u80f6\u79cd", "\u6a61\u80f6\u7c7b\u578b", "\u4ea7\u54c1\u7c7b\u578b", "\u4ea7\u54c1\u540d\u79f0", "\u6837\u54c1\u540d\u79f0", "\u7269\u6599\u540d\u79f0", "\u724c\u53f7", "rubber_type", "material", "product", "sample_name"]
T90_CANDIDATES = ["t90", "T90", "T_90", "t 90", "\u95e8\u5c3cT90", "\u786b\u5316T90", "T90\u503c", "t\u00b4c(90),min", "t'c(90),min"]
LINE_CANDIDATES = ["\u7ebf\u522b", "\u751f\u4ea7\u7ebf", "\u88c5\u7f6e", "C/D/E", "line", "\u91c7\u6837\u70b9\u63cf\u8ff0"]

POINT_VAR_CANDIDATES = ["\u53d8\u91cf", "\u53d8\u91cf\u540d", "\u540d\u79f0", "\u4f4d\u53f7\u540d\u79f0", "\u70b9\u4f4d\u540d\u79f0", "\u4e2d\u6587\u540d"]
POINT_TAG_CANDIDATES = ["\u70b9\u4f4d", "\u4f4d\u53f7", "tag", "Tag", "DCS\u70b9\u4f4d", "\u6d4b\u70b9", "\u70b9\u540d", "\u6587\u4ef6\u540d"]
POINT_LOW_CANDIDATES = ["\u4e0b\u9650", "\u6b63\u5e38\u4e0b\u9650", "\u4f4e\u9650", "\u6700\u5c0f\u503c", "lower", "low", "min", "LSL"]
POINT_HIGH_CANDIDATES = ["\u4e0a\u9650", "\u6b63\u5e38\u4e0a\u9650", "\u9ad8\u9650", "\u6700\u5927\u503c", "upper", "high", "max", "USL"]
POINT_UNIT_CANDIDATES = ["\u5355\u4f4d", "unit"]
POINT_ENABLED_CANDIDATES = ["\u662f\u5426\u4f7f\u7528", "\u4f7f\u7528", "\u542f\u7528", "include"]


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
    parser = argparse.ArgumentParser(description="Validate frozen V1 package on future raw data after point-bound cleaning.")
    parser.add_argument("--future-dir", type=Path, default=Path("data/future"))
    parser.add_argument("--point-config", type=Path, default=Path("data") / "\u526f\u672c\u5364\u5316\u5de5\u6bb5\u6570\u636e\u70b9\u4f4d.xlsx")
    parser.add_argument("--future-t90-files", type=Path, nargs="*", default=None)
    parser.add_argument("--target-rubber-type", type=str, default="\u5364\u5316\u6a61\u80f6")
    parser.add_argument("--historical-reference", type=Path, default=Path("runs/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--deploy-dir", type=Path, default=Path("deploy/ca_safe_band_mvp"))
    parser.add_argument("--residence-minutes", type=int, default=174)
    parser.add_argument("--t90-match-tolerance-minutes", type=int, default=90)
    parser.add_argument("--eval-frequency-minutes", type=int, default=10)
    parser.add_argument("--min-valid-points", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/future_holdout_v1_cleaned_validation"))
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value).strip())


def repair_mojibake_text(value: str) -> str:
    if value and ("鍗" in value or "姗" in value or "兌" in value):
        return "\u5364\u5316\u6a61\u80f6"
    return value


def normalize_tag(text: Any) -> str:
    value = str(text).strip().replace("/", ".").replace("-", "_").replace(".", "_")
    value = re.sub(r"_+", "_", value).upper().strip("_")
    return value


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


def point_name_map() -> dict[str, str]:
    result = {}
    for friendly, meta in RAW_POINT_MAPPING.items():
        result[normalize_tag(meta["tag"])] = friendly
    return result


def resolve_point_config(path: Path) -> Path:
    if path.exists():
        return path
    for candidate in Path("data").glob("*.xlsx"):
        if "\u6570\u636e\u70b9\u4f4d" in candidate.name:
            return candidate
    return path


def resolve_t90_files(future_dir: Path, requested: list[Path] | None) -> list[Path]:
    if requested:
        return requested
    result = []
    for stem in ["2026.1", "2026.2", "2026.3C"]:
        xlsx = future_dir / f"{stem}.xlsx"
        xls = future_dir / f"{stem}.xls"
        if xlsx.exists():
            result.append(xlsx)
        elif xls.exists():
            result.append(xls)
    return result


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
            data = pd.DataFrame({"time": pd.to_datetime(raw.iloc[:, time_col], errors="coerce"), "value": pd.to_numeric(raw.iloc[:, value_col], errors="coerce")})
            tag_value = str(raw.iloc[0, tag_col]) if tag_col is not None and len(raw) else path.stem
            data = data.dropna(subset=["time"]).sort_values("time")
            return data, {"file": str(path), "encoding": encoding, "row_count": int(len(data)), "tag_value": tag_value, "parse_error": None}
        except Exception as exc:
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")
    return pd.DataFrame(columns=["time", "value"]), {"file": str(path), "parse_error": " | ".join(errors), "row_count": 0}


def parse_future_raw_before_cleaning(future_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    txt_files = sorted(future_dir.glob("*.txt"))
    tag_map = point_name_map()
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
        friendly = tag_map.get(normalize_tag(path.stem)) or tag_map.get(normalize_tag(meta.get("tag_value", "")))
        if friendly is None:
            continue
        detected.append(friendly)
        frames.append(data.groupby("time", as_index=False)["value"].mean().rename(columns={"value": friendly}))
    if frames:
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="time", how="outer")
        duplicate_count = int(merged["time"].duplicated().sum())
        merged = merged.groupby("time", as_index=False).mean(numeric_only=True).sort_values("time").reset_index(drop=True)
    else:
        duplicate_count = 0
        merged = pd.DataFrame(columns=["time"])
    required = [name for name, meta in RAW_POINT_MAPPING.items() if meta["required"]]
    missing_required = sorted(set(required) - set(detected))
    point_rows = []
    for friendly in sorted(set(detected)):
        series = pd.to_numeric(merged.get(friendly, pd.Series(dtype=float)), errors="coerce")
        point_rows.append({"point": friendly, "non_null_count": int(series.notna().sum()), "missing_rate": float(series.isna().mean()) if len(merged) else None, "min": float(series.min()) if series.notna().any() else None, "median": float(series.median()) if series.notna().any() else None, "max": float(series.max()) if series.notna().any() else None})
    point_quality = pd.DataFrame(point_rows)
    if len(merged) > 1:
        diffs = merged["time"].sort_values().diff().dropna().dt.total_seconds() / 60.0
        sampling = {"median_minutes": float(diffs.median()), "q25_minutes": float(diffs.quantile(0.25)), "q75_minutes": float(diffs.quantile(0.75)), "max_minutes": float(diffs.max())}
    else:
        sampling = {}
    report = {"file_count": len(txt_files), "parsed_file_count": int(sum(1 for m in parsed_meta if not m.get("parse_error"))), "unparsed_files": unparsed, "detected_points": sorted(set(detected)), "missing_required_points": missing_required, "time_min": merged["time"].min().isoformat() if len(merged) else None, "time_max": merged["time"].max().isoformat() if len(merged) else None, "row_count": int(len(merged)), "duplicate_timestamp_count": duplicate_count, "sampling_interval_summary": sampling, "missing_rate_by_point": {row["point"]: row["missing_rate"] for row in point_rows}, "future_raw_quality_pass": bool(len(missing_required) == 0 and len(merged) > 0), "parsed_file_details": parsed_meta}
    merged.to_parquet(output_dir / "future_raw_merged_before_cleaning.parquet", index=False)
    merged.to_csv(output_dir / "future_raw_merged_before_cleaning.csv", index=False, encoding="utf-8-sig")
    point_quality.to_csv(output_dir / "future_raw_point_quality_before_cleaning.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "future_raw_quality_before_cleaning_report.json", report)
    return merged, report, point_quality


def parse_point_bounds(point_config: Path, output_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    point_config = resolve_point_config(point_config)
    warnings = []
    if not point_config.exists():
        report = {"point_config_path": str(point_config), "sheets_inspected": [], "selected_sheet": None, "selected_columns": {}, "required_tag_count": len(RAW_POINT_MAPPING), "bounds_found_count": 0, "bounds_missing_tags": [m["tag"] for m in RAW_POINT_MAPPING.values()], "bounds_used": [], "warnings": ["point_config_missing"], "point_bounds_parse_pass": False}
        bounds = pd.DataFrame()
        write_json(output_dir / "point_bounds_parse_report.json", report)
        bounds.to_csv(output_dir / "point_bounds_used.csv", index=False, encoding="utf-8-sig")
        bounds.to_csv(table_dir / "future_point_bounds_used.csv", index=False, encoding="utf-8-sig")
        return bounds, report
    xl = pd.ExcelFile(point_config)
    best = None
    inspected = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(point_config, sheet_name=sheet)
        if df.empty:
            inspected.append({"sheet": sheet, "row_count": 0, "score": 0})
            continue
        columns = list(df.columns)
        tag_col = pick_column(columns, POINT_TAG_CANDIDATES)
        low_col = pick_column(columns, POINT_LOW_CANDIDATES)
        high_col = pick_column(columns, POINT_HIGH_CANDIDATES)
        var_col = pick_column(columns, POINT_VAR_CANDIDATES)
        unit_col = pick_column(columns, POINT_UNIT_CANDIDATES)
        enabled_col = pick_column(columns, POINT_ENABLED_CANDIDATES)
        required_tags = {normalize_tag(m["tag"]) for m in RAW_POINT_MAPPING.values()}
        matched = 0
        if tag_col is not None:
            matched = int(df[tag_col].map(normalize_tag).isin(required_tags).sum())
        score = matched * 10 + int(low_col is not None) + int(high_col is not None) + int(var_col is not None)
        inspected.append({"sheet": sheet, "row_count": int(len(df)), "score": int(score), "matched_required_tags": matched})
        if best is None or score > best["score"]:
            best = {"sheet": sheet, "df": df, "score": score, "columns": {"tag": tag_col, "low": low_col, "high": high_col, "variable": var_col, "unit": unit_col, "enabled": enabled_col}}
    rows = []
    if best is not None and best["columns"]["tag"] is not None:
        df = best["df"]
        cols = best["columns"]
        lookup = {normalize_tag(str(row[cols["tag"]])): row for _, row in df.iterrows()}
        for friendly, meta in RAW_POINT_MAPPING.items():
            row = lookup.get(normalize_tag(meta["tag"]))
            low = high = unit = variable = None
            if row is not None:
                low = pd.to_numeric(pd.Series([row[cols["low"]]]) if cols["low"] is not None else pd.Series([np.nan]), errors="coerce").iloc[0]
                high = pd.to_numeric(pd.Series([row[cols["high"]]]) if cols["high"] is not None else pd.Series([np.nan]), errors="coerce").iloc[0]
                unit = str(row[cols["unit"]]) if cols["unit"] is not None and pd.notna(row[cols["unit"]]) else None
                variable = str(row[cols["variable"]]) if cols["variable"] is not None and pd.notna(row[cols["variable"]]) else None
            if pd.notna(low) and pd.notna(high) and float(low) > float(high):
                warnings.append(f"swap_bounds_for_{meta['tag']}")
                low, high = high, low
            rows.append({"friendly_name": friendly, "dcs_tag": meta["tag"], "runtime_feature": meta["output"], "variable_name": variable, "lower_bound": float(low) if pd.notna(low) else None, "upper_bound": float(high) if pd.notna(high) else None, "unit": unit, "bound_found": bool(pd.notna(low) or pd.notna(high))})
    bounds = pd.DataFrame(rows)
    missing = bounds.loc[~bounds["bound_found"], "dcs_tag"].tolist() if not bounds.empty else [m["tag"] for m in RAW_POINT_MAPPING.values()]
    report = {"point_config_path": str(point_config), "sheets_inspected": inspected, "selected_sheet": best["sheet"] if best else None, "selected_columns": {k: str(v) if v is not None else None for k, v in (best["columns"] if best else {}).items()}, "required_tag_count": len(RAW_POINT_MAPPING), "bounds_found_count": int(bounds["bound_found"].sum()) if not bounds.empty else 0, "bounds_missing_tags": missing, "bounds_used": bounds.to_dict(orient="records"), "warnings": warnings, "point_bounds_parse_pass": bool(not bounds.empty and bounds["bound_found"].sum() >= max(1, int(len(RAW_POINT_MAPPING) * 0.7)))}
    bounds.to_csv(output_dir / "point_bounds_used.csv", index=False, encoding="utf-8-sig")
    bounds.to_csv(table_dir / "future_point_bounds_used.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "point_bounds_parse_report.json", report)
    return bounds, report


def compute_ca_ratio(df: pd.DataFrame) -> pd.Series:
    ca = pd.to_numeric(df.get("ca_feed"), errors="coerce")
    flow = pd.to_numeric(df.get("rubber_flow_2"), errors="coerce")
    ratio = ca.where(ca >= 0) / flow.where(flow > 0)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio = ratio.where(ratio >= 0)
    return ratio


def compute_raw_ca_ratio(df: pd.DataFrame) -> pd.Series:
    ca = pd.to_numeric(df.get("ca_feed"), errors="coerce")
    flow = pd.to_numeric(df.get("rubber_flow_2"), errors="coerce")
    ratio = ca / flow.where(flow != 0)
    return ratio.replace([np.inf, -np.inf], np.nan)


def clean_raw_with_bounds(raw: pd.DataFrame, bounds: pd.DataFrame, output_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cleaned = raw.copy()
    out_rows = []
    invalid_masks: dict[str, pd.Series] = {}
    before_ratio = compute_raw_ca_ratio(raw)
    bound_lookup = bounds.set_index("friendly_name").to_dict(orient="index") if not bounds.empty else {}
    for friendly in RAW_POINT_MAPPING:
        if friendly not in cleaned.columns:
            continue
        series = pd.to_numeric(cleaned[friendly], errors="coerce")
        low = bound_lookup.get(friendly, {}).get("lower_bound")
        high = bound_lookup.get(friendly, {}).get("upper_bound")
        mask = pd.Series(False, index=cleaned.index)
        reason = pd.Series("", index=cleaned.index, dtype=object)
        if low is not None and not pd.isna(low):
            m = series < float(low)
            mask |= m
            reason.loc[m] = "below_lower_bound"
        if high is not None and not pd.isna(high):
            m = series > float(high)
            mask |= m
            reason.loc[m] = np.where(reason.loc[m].eq(""), "above_upper_bound", reason.loc[m] + ";above_upper_bound")
        if friendly == "ca_feed":
            m = series < 0
            mask |= m
            reason.loc[m] = np.where(reason.loc[m].eq(""), "negative_ca_feed", reason.loc[m] + ";negative_ca_feed")
        if friendly == "rubber_flow_2":
            m = series <= 0
            mask |= m
            reason.loc[m] = np.where(reason.loc[m].eq(""), "invalid_rubber_flow", reason.loc[m] + ";invalid_rubber_flow")
        invalid_masks[friendly] = mask
        if mask.any():
            part = pd.DataFrame({"time": cleaned.loc[mask, "time"], "point": friendly, "raw_value": series.loc[mask], "reason": reason.loc[mask], "lower_bound": low, "upper_bound": high})
            out_rows.append(part)
            cleaned.loc[mask, friendly] = np.nan
    after_ratio = compute_ca_ratio(cleaned)
    required_cols = [name for name in RAW_POINT_MAPPING if name in cleaned.columns]
    invalid_required_count = pd.DataFrame({k: v for k, v in invalid_masks.items() if k in required_cols}).sum(axis=1) if invalid_masks else pd.Series(0, index=cleaned.index)
    possible_shutdown = invalid_required_count.ge(3)
    if "rubber_flow_2" in invalid_masks:
        possible_shutdown |= invalid_masks["rubber_flow_2"]
    cleaned["possible_shutdown_or_invalid_operation"] = possible_shutdown
    out_df = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=["time", "point", "raw_value", "reason", "lower_bound", "upper_bound"])
    summary_rows = []
    for friendly in RAW_POINT_MAPPING:
        count = int(invalid_masks.get(friendly, pd.Series(False, index=cleaned.index)).sum())
        summary_rows.append({"point": friendly, "out_of_bound_count": count, "out_of_bound_rate": float(count / len(cleaned)) if len(cleaned) else None, "lower_bound": bound_lookup.get(friendly, {}).get("lower_bound"), "upper_bound": bound_lookup.get(friendly, {}).get("upper_bound")})
    summary = pd.DataFrame(summary_rows)
    negative_ca_feed_count = int(((pd.to_numeric(raw.get("ca_feed"), errors="coerce") < 0).sum()) if "ca_feed" in raw.columns else 0)
    invalid_rubber_flow_count = int(((pd.to_numeric(raw.get("rubber_flow_2"), errors="coerce") <= 0).sum()) if "rubber_flow_2" in raw.columns else 0)
    impossible_before = before_ratio.notna() & ((before_ratio < 0) | ~np.isfinite(before_ratio))
    abnormal_ca_removed_count = int(impossible_before.sum() + max(before_ratio.notna().sum() - after_ratio.notna().sum(), 0))
    report = {
        "bounds_applied_count": int(summary["lower_bound"].notna().sum() + summary["upper_bound"].notna().sum()),
        "bounds_missing_count": int((summary["lower_bound"].isna() & summary["upper_bound"].isna()).sum()),
        "total_rows": int(len(cleaned)),
        "out_of_bound_count_by_point": summary.set_index("point")["out_of_bound_count"].astype(int).to_dict(),
        "out_of_bound_rate_by_point": summary.set_index("point")["out_of_bound_rate"].to_dict(),
        "negative_ca_feed_count": negative_ca_feed_count,
        "invalid_rubber_flow_count": invalid_rubber_flow_count,
        "possible_shutdown_timestamp_count": int(possible_shutdown.sum()),
        "before_cleaning_ca_consumption_min": float(before_ratio.min()) if before_ratio.notna().any() else None,
        "after_cleaning_ca_consumption_min": float(after_ratio.min()) if after_ratio.notna().any() else None,
        "before_cleaning_ca_consumption_max": float(before_ratio.max()) if before_ratio.notna().any() else None,
        "after_cleaning_ca_consumption_max": float(after_ratio.max()) if after_ratio.notna().any() else None,
        "abnormal_ca_removed_count": abnormal_ca_removed_count,
        "cleaning_pass": bool((after_ratio.dropna() >= 0).all() and np.isfinite(after_ratio.dropna()).all()),
        "warnings": [],
    }
    cleaned.to_parquet(output_dir / "future_raw_merged_cleaned.parquet", index=False)
    cleaned.to_csv(output_dir / "future_raw_merged_cleaned.csv", index=False, encoding="utf-8-sig")
    out_df.to_csv(output_dir / "future_dcs_out_of_bound_rows.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "future_dcs_cleaning_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "future_dcs_cleaning_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "future_dcs_cleaning_report.json", report)
    return cleaned, out_df, summary, report


def parse_future_t90(files: list[Path], target_rubber: str, output_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    all_parts = []
    halogen_parts = []
    selected_time = {}
    selected_rubber = {}
    selected_t90 = {}
    selected_line = {}
    failed = []
    parsed = []
    sheet_summary = []
    missing_rubber = False
    missing_t90 = False
    warnings = []
    for path in files:
        if not path.exists():
            failed.append({"file": str(path), "error": "file_missing"})
            continue
        if path.suffix.lower() == ".xls":
            failed.append({"file": str(path), "error": "xlsx_preferred_xls_not_read"})
            warnings.append(f"xlsx_preferred_for_{path.name}")
            continue
        try:
            sheets = pd.read_excel(path, sheet_name=None)
        except Exception as exc:
            failed.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        parsed.append(str(path))
        for sheet_name, df in sheets.items():
            if df.empty:
                continue
            time_col = pick_column(list(df.columns), TIME_CANDIDATES)
            rubber_col = pick_column(list(df.columns), RUBBER_CANDIDATES)
            t90_col = pick_column(list(df.columns), T90_CANDIDATES)
            line_col = pick_column(list(df.columns), LINE_CANDIDATES)
            key = f"{path}:{sheet_name}"
            selected_time[key] = str(time_col) if time_col is not None else None
            selected_rubber[key] = str(rubber_col) if rubber_col is not None else None
            selected_t90[key] = str(t90_col) if t90_col is not None else None
            selected_line[key] = str(line_col) if line_col is not None else None
            if rubber_col is None:
                missing_rubber = True
                warnings.append(f"missing_rubber_type_column:{key}")
                continue
            if t90_col is None:
                missing_t90 = True
                warnings.append(f"missing_t90_column:{key}")
                continue
            if time_col is None:
                warnings.append(f"missing_time_column:{key}")
                continue
            part = pd.DataFrame({
                "source_file": str(path),
                "sheet_name": sheet_name,
                "time": pd.to_datetime(df[time_col], errors="coerce"),
                "rubber_type": df[rubber_col].map(normalize_text),
                "t90": pd.to_numeric(df[t90_col], errors="coerce"),
                "line": df[line_col].map(normalize_text) if line_col is not None else None,
            })
            all_parts.append(part)
            target = normalize_text(target_rubber)
            hal_mask = part["rubber_type"].str.contains(target, na=False) & ~part["rubber_type"].str.contains("\u6c2f\u4e01\u57fa", na=False)
            hal_raw = int(hal_mask.sum())
            valid = hal_mask & part["time"].notna() & part["t90"].notna() & (part["t90"] > 0)
            halogen_parts.append(part.loc[valid].copy())
            sheet_summary.append({"file": str(path), "sheet": sheet_name, "rows": int(len(part)), "halogen_raw_rows": hal_raw, "halogen_valid_t90_rows": int(valid.sum())})
    all_df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame(columns=["source_file", "sheet_name", "time", "rubber_type", "t90", "line"])
    halogen = pd.concat(halogen_parts, ignore_index=True) if halogen_parts else pd.DataFrame(columns=all_df.columns)
    if not halogen.empty:
        halogen = halogen.sort_values("time").reset_index(drop=True)
    total = len(all_df)
    hal_raw_mask = all_df["rubber_type"].str.contains(normalize_text(target_rubber), na=False) & ~all_df["rubber_type"].str.contains("\u6c2f\u4e01\u57fa", na=False) if not all_df.empty else pd.Series(dtype=bool)
    hal_raw_count = int(hal_raw_mask.sum()) if not all_df.empty else 0
    dropped_time = int((hal_raw_mask & all_df["time"].isna()).sum()) if not all_df.empty else 0
    dropped_t90 = int((hal_raw_mask & (all_df["time"].notna()) & (all_df["t90"].isna() | (all_df["t90"] <= 0))).sum()) if not all_df.empty else 0
    q = halogen["t90"].quantile([0, 0.25, 0.5, 0.75, 1.0]).to_dict() if not halogen.empty else {}
    rubber_counts = all_df["rubber_type"].value_counts(dropna=False).to_dict() if not all_df.empty else {}
    all_df.to_parquet(output_dir / "future_t90_raw_all.parquet", index=False)
    all_df.to_csv(output_dir / "future_t90_raw_all.csv", index=False, encoding="utf-8-sig")
    halogen.to_parquet(output_dir / "future_t90_halogen_only.parquet", index=False)
    halogen.to_csv(output_dir / "future_t90_halogen_only.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([
        {"metric": "total_t90_rows", "value": total},
        {"metric": "halogen_raw_rows_before_t90_cleaning", "value": hal_raw_count},
        {"metric": "halogen_t90_rows_after_time_and_t90_cleaning", "value": len(halogen)},
        {"metric": "excluded_non_halogen_rows", "value": max(total - hal_raw_count, 0)},
        {"metric": "future_t90_filter_pass", "value": bool(len(halogen) > 0 and not missing_rubber)},
    ])
    summary.to_csv(table_dir / "future_t90_halogen_filter_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "future_t90_files_requested": [str(p) for p in files],
        "future_t90_files_found": [str(p) for p in files if p.exists()],
        "future_t90_files_parsed": parsed,
        "future_t90_files_failed": failed,
        "selected_time_column_by_file": selected_time,
        "selected_rubber_type_column_by_file": selected_rubber,
        "selected_t90_column_by_file": selected_t90,
        "selected_line_column_by_file": selected_line,
        "sheet_summary_by_file": sheet_summary,
        "total_t90_rows": int(total),
        "halogen_raw_rows_before_t90_cleaning": hal_raw_count,
        "halogen_t90_rows_after_time_and_t90_cleaning": int(len(halogen)),
        "excluded_non_halogen_rows": int(max(total - hal_raw_count, 0)),
        "dropped_missing_time_rows": dropped_time,
        "dropped_missing_or_nonnumeric_t90_rows": dropped_t90,
        "rubber_type_value_counts": {str(k): int(v) for k, v in rubber_counts.items()},
        "t90_time_min": halogen["time"].min().isoformat() if len(halogen) else None,
        "t90_time_max": halogen["time"].max().isoformat() if len(halogen) else None,
        "t90_min": float(q.get(0)) if q else None,
        "t90_q25": float(q.get(0.25)) if q else None,
        "t90_median": float(q.get(0.5)) if q else None,
        "t90_q75": float(q.get(0.75)) if q else None,
        "t90_max": float(q.get(1.0)) if q else None,
        "missing_rubber_type_column": missing_rubber,
        "missing_t90_column": missing_t90,
        "future_t90_available": bool(len(halogen) > 0),
        "future_t90_filter_pass": bool(len(halogen) > 0 and not missing_rubber),
        "warnings": warnings,
    }
    write_json(output_dir / "future_t90_parse_report.json", report)
    return all_df, halogen, report


def build_runtime_features(cleaned: pd.DataFrame, eval_freq: int, min_valid: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if cleaned.empty:
        runtime = pd.DataFrame()
        report = {"evaluation_row_count": 0, "feature_valid_row_count": 0, "invalid_row_count": 0, "feature_quality_pass": False}
        write_json(output_dir / "future_feature_quality_report.json", report)
        return runtime, pd.DataFrame(), report
    data = cleaned.copy().sort_values("time").set_index("time")
    start = data.index.min()
    end = data.index.max()
    eval_times = pd.date_range(start=start.ceil(f"{eval_freq}min"), end=end.floor(f"{eval_freq}min"), freq=f"{eval_freq}min")
    features = pd.DataFrame(index=data.index)
    counts = data.rolling("60min", min_periods=1).count()
    insufficient_counts = {}
    for friendly, meta in RAW_POINT_MAPPING.items():
        if friendly == "ca_feed":
            continue
        out_col = meta["output"]
        if friendly not in data.columns:
            features[out_col] = np.nan
            insufficient_counts[out_col] = len(eval_times)
            continue
        features[out_col] = data[friendly].rolling("60min", min_periods=min_valid).mean()
        insufficient_counts[out_col] = int((counts[friendly].reindex(eval_times, method="nearest") < min_valid).sum()) if len(eval_times) else 0
    ratio = compute_ca_ratio(data.reset_index()).set_axis(data.index)
    features["ca_per_rubber_flow_win_60_mean"] = ratio.rolling("60min", min_periods=min_valid).mean()
    features["current_ca_consumption"] = features["ca_per_rubber_flow_win_60_mean"]
    insufficient_counts["ca_per_rubber_flow_win_60_mean"] = int((ratio.rolling("60min", min_periods=1).count().reindex(eval_times, method="nearest") < min_valid).sum()) if len(eval_times) else 0
    features["output_ir_corrected_offset_20_win_15_std"] = np.nan
    possible_shutdown = data.get("possible_shutdown_or_invalid_operation", pd.Series(False, index=data.index)).rolling("60min", min_periods=1).max()
    runtime = features.reindex(eval_times, method="nearest").reset_index().rename(columns={"index": "time"})
    runtime["possible_shutdown_or_invalid_operation"] = possible_shutdown.reindex(eval_times, method="nearest").fillna(False).astype(bool).to_numpy() if len(eval_times) else []
    required_cols = [meta["output"] for name, meta in RAW_POINT_MAPPING.items() if name != "ca_feed"] + ["ca_per_rubber_flow_win_60_mean"]
    missing_required = runtime[required_cols].isna()
    runtime["feature_quality"] = np.where(missing_required.any(axis=1), "incomplete", "ok")
    runtime["missing_raw_columns"] = ""
    runtime["insufficient_window_features"] = missing_required.apply(lambda row: ";".join(row.index[row].tolist()), axis=1)
    runtime["warning_flags"] = np.where(missing_required.any(axis=1), "insufficient_window_features;optional_ir_missing", "optional_ir_missing")
    quality = runtime[["time", "feature_quality", "insufficient_window_features", "warning_flags", "possible_shutdown_or_invalid_operation"]].copy()
    runtime.to_parquet(output_dir / "future_runtime_features.parquet", index=False)
    runtime.to_csv(output_dir / "future_runtime_features.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(output_dir / "future_feature_quality_by_time.csv", index=False, encoding="utf-8-sig")
    valid_rate = float((runtime["feature_quality"] == "ok").mean()) if len(runtime) else 0.0
    report = {"evaluation_row_count": int(len(runtime)), "feature_valid_row_count": int((runtime["feature_quality"] == "ok").sum()), "invalid_row_count": int((runtime["feature_quality"] != "ok").sum()), "missing_feature_counts": missing_required.sum().astype(int).to_dict(), "insufficient_window_counts": insufficient_counts, "optional_ir_available_rate": 0.0, "invalid_due_to_shutdown_or_out_of_bound_count": int(runtime["possible_shutdown_or_invalid_operation"].sum()), "feature_valid_rate": valid_rate, "feature_quality_pass": bool(valid_rate >= 0.80)}
    write_json(output_dir / "future_feature_quality_report.json", report)
    return runtime, quality, report


def load_recommender(deploy_dir: Path) -> Any:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(deploy_dir.resolve()))
    spec = importlib.util.spec_from_file_location("cleaned_safe_band_interface", deploy_dir / "interface.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load interface.py from {deploy_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SafeBandRecommender(deploy_dir, mode="production").load()


def score_future(runtime: pd.DataFrame, deploy_dir: Path, output_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    recommender = load_recommender(deploy_dir)
    preds = recommender.predict_batch(runtime.to_dict(orient="records"), mode="production")
    pred_df = pd.DataFrame(preds)
    replay = pd.concat([runtime.reset_index(drop=True), pred_df.add_prefix("pred_")], axis=1)
    rename = {"pred_recommended_ca_consumption_min": "recommended_ca_consumption_min", "pred_recommended_ca_consumption_max": "recommended_ca_consumption_max", "pred_recommended_ca_consumption_target": "recommended_ca_consumption_target", "pred_interval_position": "interval_position", "pred_action_hint": "action_hint", "pred_action_visibility": "action_visibility", "pred_engineering_review_required": "engineering_review_required", "pred_input_valid": "input_valid", "pred_recommendation_status": "recommendation_status"}
    replay = replay.rename(columns=rename)
    for col in ["recommended_ca_consumption_min", "recommended_ca_consumption_max", "recommended_ca_consumption_target", "interval_position", "action_visibility", "input_valid", "recommendation_status"]:
        if col not in replay.columns:
            replay[col] = np.nan
    valid = replay["recommended_ca_consumption_min"].notna() & replay["recommended_ca_consumption_max"].notna()
    inside = replay["interval_position"].eq("inside_band")
    above = replay["interval_position"].eq("above_band")
    below = replay["interval_position"].eq("below_band")
    current = pd.to_numeric(replay["current_ca_consumption"], errors="coerce")
    impossible = current.notna() & ((current < 0) | ~np.isfinite(current))
    rec_dist = pd.to_numeric(replay.loc[valid, "recommended_ca_consumption_target"], errors="coerce").quantile([0, 0.25, 0.5, 0.75, 1]).to_dict() if valid.any() else {}
    cur_dist = current.dropna().quantile([0, 0.25, 0.5, 0.75, 1]).to_dict() if current.notna().any() else {}
    replay.to_parquet(output_dir / "future_v1_recommendation_replay.parquet", index=False)
    replay.to_csv(output_dir / "future_v1_recommendation_replay.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([
        {"metric": "scored_row_count", "value": len(replay)},
        {"metric": "recommendation_coverage", "value": float(valid.mean()) if len(replay) else None},
        {"metric": "inside_band_count", "value": int(inside.sum())},
        {"metric": "above_band_count", "value": int(above.sum())},
        {"metric": "below_band_count", "value": int(below.sum())},
        {"metric": "impossible_current_ca_consumption_count", "value": int(impossible.sum())},
    ])
    summary.to_csv(table_dir / "future_v1_recommendation_distribution_summary.csv", index=False, encoding="utf-8-sig")
    report = {"scored_row_count": int(len(replay)), "recommendation_coverage": float(valid.mean()) if len(replay) else 0.0, "no_recommendation_count": int((~valid).sum()), "input_invalid_count": int((~replay.get("input_valid", pd.Series(True, index=replay.index)).astype(bool)).sum()) if len(replay) else 0, "inside_band_count": int(inside.sum()), "above_band_count": int(above.sum()), "below_band_count": int(below.sum()), "manual_review_required_count": int(replay["action_visibility"].eq("manual_review_required").sum()), "diagnostic_only_count": int(replay["action_visibility"].eq("diagnostic_only").sum()), "monitor_only_count": int(replay["action_visibility"].eq("monitor_only").sum()), "missing_required_features_summary": replay.get("pred_missing_required_features", pd.Series([], dtype=object)).astype(str).value_counts().to_dict(), "warning_flags_summary": replay.get("pred_warning_flags", pd.Series([], dtype=object)).astype(str).value_counts().to_dict(), "recommended_ca_consumption_distribution": {"min": rec_dist.get(0), "q25": rec_dist.get(0.25), "median": rec_dist.get(0.5), "q75": rec_dist.get(0.75), "max": rec_dist.get(1)}, "current_ca_consumption_distribution": {"min": cur_dist.get(0), "q25": cur_dist.get(0.25), "median": cur_dist.get(0.5), "q75": cur_dist.get(0.75), "max": cur_dist.get(1)}, "impossible_current_ca_consumption_count": int(impossible.sum()), "future_replay_pass": bool(len(replay) > 0 and int(impossible.sum()) == 0 and valid.mean() > 0)}
    write_json(output_dir / "future_v1_recommendation_distribution_report.json", report)
    return replay, report


def add_t90_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["y_ok"] = ((out["t90"] >= T90_LOW) & (out["t90"] <= T90_HIGH)).astype(int)
    out["y_low"] = (out["t90"] < T90_LOW).astype(int)
    out["y_high"] = (out["t90"] > T90_HIGH).astype(int)
    out["y_out_spec"] = ((out["t90"] < T90_LOW) | (out["t90"] > T90_HIGH)).astype(int)
    out["clear_label"] = np.select([out["t90"].between(CLEAR_OK_LOW, CLEAR_OK_HIGH, inclusive="both"), out["t90"] <= CLEAR_LOW, out["t90"] >= CLEAR_HIGH], ["clear_ok", "clear_low", "clear_high"], default="uncertain_boundary")
    return out


def summarize_position_risk(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["interval_position", "sample_count", "ok_rate", "high_rate", "low_rate", "out_spec_rate", "mean_t90"])
    rows = []
    for label, subset in [("inside_band", df[df["interval_position"].eq("inside_band")]), ("outside_band", df[df["interval_position"].isin(["above_band", "below_band"])]), ("above_band", df[df["interval_position"].eq("above_band")]), ("below_band", df[df["interval_position"].eq("below_band")])]:
        if subset.empty:
            continue
        rows.append({"interval_position": label, "sample_count": int(len(subset)), "ok_rate": float(subset["y_ok"].mean()), "high_rate": float(subset["y_high"].mean()), "low_rate": float(subset["y_low"].mean()), "out_spec_rate": float(subset["y_out_spec"].mean()), "mean_t90": float(subset["t90"].mean())})
    return pd.DataFrame(rows)


def rate_delta(risk: pd.DataFrame, left: str, right: str, metric: str) -> float | None:
    l = risk[risk["interval_position"].eq(left)]
    r = risk[risk["interval_position"].eq(right)]
    if l.empty or r.empty:
        return None
    return float(l[metric].iloc[0] - r[metric].iloc[0])


def strategy_report(aligned: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if aligned.empty:
        return pd.DataFrame(), pd.DataFrame(), {"strategy": strategy, "aligned_sample_count": 0, "risk_guardrail_pass": False}
    aligned = add_t90_targets(aligned)
    risk = summarize_position_risk(aligned)
    clear = aligned[aligned["clear_label"].isin(["clear_ok", "clear_low", "clear_high"])].copy()
    clear_risk = summarize_position_risk(clear) if len(clear) >= 20 else pd.DataFrame()
    inside = risk[risk["interval_position"].eq("inside_band")]
    outside = risk[risk["interval_position"].eq("outside_band")]
    inside_count = int(inside["sample_count"].iloc[0]) if not inside.empty else 0
    outside_count = int(outside["sample_count"].iloc[0]) if not outside.empty else 0
    guardrail = bool(len(aligned) >= 30 and inside_count >= 10 and outside_count >= 10 and not inside.empty and not outside.empty and float(inside["high_rate"].iloc[0]) <= float(outside["high_rate"].iloc[0]) and float(inside["out_spec_rate"].iloc[0]) <= float(outside["out_spec_rate"].iloc[0]))
    report = {"strategy": strategy, "aligned_sample_count": int(len(aligned)), "unique_t90_count": int(aligned["t90_time"].nunique()) if "t90_time" in aligned else int(len(aligned)), "unique_recommendation_count": int(aligned["time"].nunique()) if "time" in aligned else int(len(aligned)), "duplicate_t90_match_rate": float(1 - aligned["t90_time"].nunique() / len(aligned)) if "t90_time" in aligned and len(aligned) else 0.0, "risk_by_interval_position": risk.to_dict(orient="records"), "inside_vs_outside_high_rate_delta": rate_delta(risk, "inside_band", "outside_band", "high_rate"), "inside_vs_outside_out_spec_rate_delta": rate_delta(risk, "inside_band", "outside_band", "out_spec_rate"), "above_vs_inside_high_rate_delta": rate_delta(risk, "above_band", "inside_band", "high_rate"), "risk_guardrail_pass": guardrail, "clear_sample_count": int(len(clear)), "uncertain_boundary_rate": float((aligned["clear_label"] == "uncertain_boundary").mean()) if len(aligned) else None, "clear_label_risk_by_interval_position": clear_risk.to_dict(orient="records") if not clear_risk.empty else []}
    return risk, clear_risk, report


def align_t90_strategies(replay: pd.DataFrame, t90: pd.DataFrame, residence: int, tolerance: int, output_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if t90.empty:
        report = {"future_t90_available": False, "future_t90_validation_status": "missing_halogen_t90", "many_predictions_to_nearest_t90": {"aligned_sample_count": 0}, "one_t90_to_nearest_prediction": {"aligned_sample_count": 0}}
        write_json(output_dir / "future_t90_backfill_validation_report.json", report)
        return pd.DataFrame(), pd.DataFrame(), report, pd.DataFrame(), pd.DataFrame()
    rec = replay.copy()
    rec["time"] = pd.to_datetime(rec["time"], errors="coerce")
    rec["quality_time"] = rec["time"] + pd.to_timedelta(residence, unit="m")
    labels = t90[["time", "t90", "rubber_type", "source_file"]].copy().sort_values("time").rename(columns={"time": "t90_time"})
    many = pd.merge_asof(rec.sort_values("quality_time"), labels.sort_values("t90_time"), left_on="quality_time", right_on="t90_time", direction="nearest", tolerance=pd.Timedelta(minutes=tolerance)).dropna(subset=["t90"]).copy()
    many["t90_match_delta_minutes"] = (many["t90_time"] - many["quality_time"]).dt.total_seconds().abs() / 60.0
    labels2 = labels.copy()
    labels2["expected_recommendation_time"] = labels2["t90_time"] - pd.to_timedelta(residence, unit="m")
    one = pd.merge_asof(labels2.sort_values("expected_recommendation_time"), rec.sort_values("time"), left_on="expected_recommendation_time", right_on="time", direction="nearest", tolerance=pd.Timedelta(minutes=tolerance)).dropna(subset=["recommended_ca_consumption_min"]).copy()
    one["t90_match_delta_minutes"] = (one["time"] - one["expected_recommendation_time"]).dt.total_seconds().abs() / 60.0
    many.to_parquet(output_dir / "future_t90_backfill_aligned_many.parquet", index=False)
    many.to_csv(output_dir / "future_t90_backfill_aligned_many.csv", index=False, encoding="utf-8-sig")
    one.to_parquet(output_dir / "future_t90_backfill_aligned_one_to_one.parquet", index=False)
    one.to_csv(output_dir / "future_t90_backfill_aligned_one_to_one.csv", index=False, encoding="utf-8-sig")
    many_risk, many_clear, many_report = strategy_report(many, "many_predictions_to_nearest_t90")
    one_risk, one_clear, one_report = strategy_report(one, "one_t90_to_nearest_prediction")
    compare = pd.DataFrame([{"strategy": "many_predictions_to_nearest_t90", **{k: v for k, v in many_report.items() if k not in ["risk_by_interval_position", "clear_label_risk_by_interval_position"]}}, {"strategy": "one_t90_to_nearest_prediction", **{k: v for k, v in one_report.items() if k not in ["risk_by_interval_position", "clear_label_risk_by_interval_position"]}}])
    compare.to_csv(table_dir / "future_backfill_alignment_strategy_comparison.csv", index=False, encoding="utf-8-sig")
    risk_rows = []
    for strategy, risk in [("many_predictions_to_nearest_t90", many_risk), ("one_t90_to_nearest_prediction", one_risk)]:
        if not risk.empty:
            part = risk.copy()
            part.insert(0, "strategy", strategy)
            risk_rows.append(part)
    risk_table = pd.concat(risk_rows, ignore_index=True) if risk_rows else pd.DataFrame()
    risk_table.to_csv(table_dir / "future_t90_backfill_validation_summary.csv", index=False, encoding="utf-8-sig")
    clear_rows = []
    for strategy, risk in [("many_predictions_to_nearest_t90", many_clear), ("one_t90_to_nearest_prediction", one_clear)]:
        if not risk.empty:
            part = risk.copy()
            part.insert(0, "strategy", strategy)
            clear_rows.append(part)
    clear_table = pd.concat(clear_rows, ignore_index=True) if clear_rows else pd.DataFrame()
    clear_table.to_csv(table_dir / "future_t90_clear_label_validation_summary.csv", index=False, encoding="utf-8-sig")
    report = {"future_t90_available": True, "future_t90_validation_status": "available", "many_predictions_to_nearest_t90": many_report, "one_t90_to_nearest_prediction": one_report}
    write_json(output_dir / "future_t90_backfill_validation_report.json", report)
    return many, one, report, risk_table, clear_table


def resolve_historical(path: Path) -> Path | None:
    if path.exists():
        return path
    for name in ["t90_ca_feature_dataset.parquet", "final_monitor_dry_run.parquet"]:
        matches = sorted(Path("runs").rglob(name)) if Path("runs").exists() else []
        if matches:
            return matches[0]
    return None


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


def compare_historical(runtime: pd.DataFrame, historical_path: Path | None, output_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = ["ca_per_rubber_flow_win_60_mean", "rubber_flow_2_win_60_mean", "bromine_feed_win_60_mean", "tank_rubber_conc_win_60_mean", "esbo_feed_win_60_mean", "neutral_alkali_feed_win_60_mean", "r510a_temp_win_60_mean", "r511a_temp_win_60_mean", "r512a_temp_win_60_mean", "r513_temp_win_60_mean", "r514_temp_win_60_mean"]
    rows = []
    if historical_path is not None:
        hist = pd.read_parquet(historical_path)
        for feature in features:
            if feature not in runtime.columns or feature not in hist.columns:
                continue
            h = pd.to_numeric(hist[feature], errors="coerce").dropna()
            f = pd.to_numeric(runtime[feature], errors="coerce").dropna()
            if h.empty or f.empty:
                continue
            hq = h.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
            fq = f.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
            out_rate = float(((f < hq.loc[0.01]) | (f > hq.loc[0.99])).mean())
            rows.append({"feature": feature, "historical_min": hq.loc[0], "historical_q01": hq.loc[0.01], "historical_q05": hq.loc[0.05], "historical_q25": hq.loc[0.25], "historical_median": hq.loc[0.5], "historical_q75": hq.loc[0.75], "historical_q95": hq.loc[0.95], "historical_q99": hq.loc[0.99], "historical_max": hq.loc[1], "future_min": fq.loc[0], "future_q01": fq.loc[0.01], "future_q05": fq.loc[0.05], "future_q25": fq.loc[0.25], "future_median": fq.loc[0.5], "future_q75": fq.loc[0.75], "future_q95": fq.loc[0.95], "future_q99": fq.loc[0.99], "future_max": fq.loc[1], "out_of_historical_range_rate": out_rate, "psi_like_drift_score": psi_like(h, f), "future_within_historical_support": bool(out_rate <= 0.20)})
    drift = pd.DataFrame(rows)
    top = drift.sort_values("psi_like_drift_score", ascending=False).head(5)["feature"].tolist() if not drift.empty and "psi_like_drift_score" in drift else []
    report = {"historical_reference_available": historical_path is not None, "historical_reference_path": str(historical_path) if historical_path else None, "feature_count_compared": int(len(drift)), "max_out_of_historical_range_rate": float(drift["out_of_historical_range_rate"].max()) if not drift.empty else None, "max_psi_like_drift_score": float(drift["psi_like_drift_score"].max()) if not drift.empty else None, "future_within_historical_support": bool(not drift.empty and drift["future_within_historical_support"].mean() >= 0.70), "top_drift_features": top}
    drift.to_csv(output_dir / "future_vs_historical_feature_drift.csv", index=False, encoding="utf-8-sig")
    drift.to_csv(table_dir / "future_vs_historical_feature_drift_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "future_vs_historical_feature_drift_report.json", report)
    return drift, report


def monthly_summary(replay: pd.DataFrame, cleaned: pd.DataFrame, out_bounds: pd.DataFrame, t90: pd.DataFrame, one: pd.DataFrame, drift: pd.DataFrame, output_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    months = sorted(pd.to_datetime(replay["time"]).dt.to_period("M").astype(str).unique()) if not replay.empty else []
    rows = []
    for month in months:
        r = replay[pd.to_datetime(replay["time"]).dt.to_period("M").astype(str).eq(month)]
        raw_month = cleaned[pd.to_datetime(cleaned["time"]).dt.to_period("M").astype(str).eq(month)]
        out_month = out_bounds[pd.to_datetime(out_bounds["time"]).dt.to_period("M").astype(str).eq(month)] if not out_bounds.empty else pd.DataFrame()
        t90_month = t90[pd.to_datetime(t90["time"]).dt.to_period("M").astype(str).eq(month)] if not t90.empty else pd.DataFrame()
        one_month = one[pd.to_datetime(one["t90_time"]).dt.to_period("M").astype(str).eq(month)] if not one.empty and "t90_time" in one else pd.DataFrame()
        risk = summarize_position_risk(add_t90_targets(one_month)) if not one_month.empty else pd.DataFrame()
        inside = risk[risk["interval_position"].eq("inside_band")]
        above = risk[risk["interval_position"].eq("above_band")]
        below = risk[risk["interval_position"].eq("below_band")]
        outside = risk[risk["interval_position"].eq("outside_band")]
        inside_count = int(inside["sample_count"].iloc[0]) if not inside.empty else 0
        outside_count = int(outside["sample_count"].iloc[0]) if not outside.empty else 0
        month_evidence_sufficient = bool(len(one_month) >= 30 and inside_count >= 10 and outside_count >= 10)
        guard = bool(month_evidence_sufficient and not inside.empty and not outside.empty and inside["high_rate"].iloc[0] <= outside["high_rate"].iloc[0] and inside["out_spec_rate"].iloc[0] <= outside["out_spec_rate"].iloc[0])
        cell_denominator = max(len(raw_month) * len(RAW_POINT_MAPPING), 1)
        rows.append({"month": month, "raw_row_count": int(len(raw_month)), "out_of_bound_rate": float(len(out_month) / cell_denominator), "feature_valid_rate": float(r["feature_quality"].eq("ok").mean()) if len(r) else None, "recommendation_coverage": float(r["recommended_ca_consumption_min"].notna().mean()) if len(r) else None, "inside_band_count": int(r["interval_position"].eq("inside_band").sum()), "above_band_count": int(r["interval_position"].eq("above_band").sum()), "below_band_count": int(r["interval_position"].eq("below_band").sum()), "t90_halogen_count": int(len(t90_month)), "one_to_one_aligned_sample_count": int(len(one_month)), "monthly_evidence_sufficient": month_evidence_sufficient, "inside_high_rate": float(inside["high_rate"].iloc[0]) if not inside.empty else None, "above_high_rate": float(above["high_rate"].iloc[0]) if not above.empty else None, "below_low_rate": float(below["low_rate"].iloc[0]) if not below.empty else None, "risk_guardrail_pass": guard})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(output_dir / "future_monthly_validation_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(table_dir / "future_monthly_validation_summary.csv", index=False, encoding="utf-8-sig")
    sufficient = monthly[monthly["monthly_evidence_sufficient"].fillna(False)] if not monthly.empty else monthly
    report = {
        "months": monthly.to_dict(orient="records"),
        "sufficient_month_count": int(len(sufficient)),
        "insufficient_month_count": int(len(monthly) - len(sufficient)) if not monthly.empty else 0,
        "monthly_risk_separation_stable": bool(not sufficient.empty and sufficient["risk_guardrail_pass"].fillna(False).all()),
    }
    return monthly, report


def split_doc_sections(text: str) -> tuple[str, list[dict[str, Any]]]:
    lines = text.splitlines()
    sections = []
    preamble = []
    current = None
    heading_re = re.compile(r"^##\s*(?:(\d+)\s*\.?\s*)?(.*)$")
    for line in lines:
        m = heading_re.match(line)
        if m:
            if current is not None:
                sections.append(current)
            elif preamble:
                pass
            number = int(m.group(1)) if m.group(1) else None
            title = m.group(2).strip()
            current = {"heading": line, "number": number, "title": title, "body_lines": []}
        elif current is None:
            preamble.append(line)
        else:
            current["body_lines"].append(line)
    if current is not None:
        sections.append(current)
    return "\n".join(preamble).rstrip(), sections


def norm_body(lines: list[str]) -> str:
    text = "\n".join(line.rstrip() for line in lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def dedup_experiment_doc(doc_path: Path, audit_dir: Path) -> dict[str, Any]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    backup = audit_dir / "Experimental_Procedure_cn.before_stage41_dedup.md"
    text = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""
    backup.write_text(text, encoding="utf-8")
    preamble, sections = split_doc_sections(text)
    inventory = []
    duplicates = []
    conflicts = []
    kept = []
    seen: list[dict[str, Any]] = []
    for idx, sec in enumerate(sections):
        body = norm_body(sec["body_lines"])
        title_key = re.sub(r"\s+", " ", sec["title"]).strip()
        record = {"original_index": idx, "section_number": sec["number"], "title": sec["title"], "body_hash": str(hash(body)), "body_length": len(body)}
        inventory.append(record)
        duplicate_of = None
        conflict = None
        for prev in seen:
            sim = difflib.SequenceMatcher(None, body, prev["body"]).ratio() if body or prev["body"] else 1.0
            if title_key == prev["title_key"] and body == prev["body"]:
                duplicate_of = (prev, "exact_duplicate", sim)
                break
            if title_key == prev["title_key"] and sim >= 0.98 and abs(len(body) - len(prev["body"])) <= 20:
                duplicate_of = (prev, "near_duplicate", sim)
                break
            if sec["number"] is not None and sec["number"] == prev["number"] and (title_key != prev["title_key"] or sim < 0.98):
                conflict = {"conflict_type": "section_number_conflict", "section_number": sec["number"], "title": sec["title"], "previous_title": prev["title"], "similarity": sim}
            elif title_key == prev["title_key"] and sim < 0.98:
                conflict = {"conflict_type": "title_conflict", "section_number": sec["number"], "title": sec["title"], "previous_section_number": prev["number"], "similarity": sim}
        if duplicate_of is not None:
            prev, dtype, sim = duplicate_of
            duplicates.append({"removed_index": idx, "kept_index": prev["index"], "duplicate_type": dtype, "title": sec["title"], "similarity": sim})
            continue
        if conflict is not None:
            conflicts.append(conflict)
        kept.append(sec)
        seen.append({"index": idx, "number": sec["number"], "title": sec["title"], "title_key": title_key, "body": body})
    new_parts = []
    if preamble:
        new_parts.append(preamble)
    for sec in kept:
        new_parts.append(sec["heading"] + "\n" + "\n".join(sec["body_lines"]).rstrip())
    cleaned = "\n\n".join(part.rstrip() for part in new_parts if part is not None).rstrip() + "\n"
    doc_path.write_text(cleaned, encoding="utf-8")
    inventory_df = pd.DataFrame(inventory)
    dup_df = pd.DataFrame(duplicates)
    conflict_df = pd.DataFrame(conflicts)
    inventory_df.to_csv(audit_dir / "experiment_doc_section_inventory.csv", index=False, encoding="utf-8-sig")
    dup_df.to_csv(audit_dir / "experiment_doc_duplicate_sections.csv", index=False, encoding="utf-8-sig")
    conflict_df.to_csv(audit_dir / "experiment_doc_conflict_sections.csv", index=False, encoding="utf-8-sig")
    report = {"doc_path": str(doc_path), "backup_path": str(backup), "original_section_count": len(sections), "cleaned_section_count": len(kept), "exact_duplicate_count": int(sum(1 for d in duplicates if d["duplicate_type"] == "exact_duplicate")), "near_duplicate_count": int(sum(1 for d in duplicates if d["duplicate_type"] == "near_duplicate")), "removed_duplicate_count": len(duplicates), "section_number_conflict_count": int(sum(1 for c in conflicts if c["conflict_type"] == "section_number_conflict")), "title_conflict_count": int(sum(1 for c in conflicts if c["conflict_type"] == "title_conflict")), "conflicts_require_manual_review": bool(len(conflicts) > 0), "cleaned_doc_written": True, "stage41_appended_after_dedup": False, "warnings": []}
    write_json(audit_dir / "experiment_doc_dedup_report.json", report)
    return report


def next_section_number(doc_path: Path, preferred: int) -> int:
    used = set()
    if doc_path.exists():
        for line in doc_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^##\s*(\d+)", line)
            if m:
                used.add(int(m.group(1)))
    n = preferred
    while n in used:
        n += 1
    return n


def append_stage41(doc_path: Path, report: dict[str, Any], dedup_report_path: Path) -> None:
    n = next_section_number(doc_path, 41)
    cleaning = report.get("dcs_cleaning_summary", {})
    rec = report.get("recommendation_distribution_summary", {})
    t90 = report.get("t90_parse_summary", {})
    one = report.get("one_to_one_backfill_summary", {})
    drift = report.get("future_vs_historical_drift_summary", {})
    monthly = report.get("monthly_stability_summary", {})
    section = f"""

## {n}. 基于点位上下限清洗的 future holdout V1 回放与卤化橡胶 T90 复验

本阶段针对 future holdout 重新验证冻结 V1 monitor-only 钙单耗安全带。由于上一轮 raw DCS 直接入模后出现 `current_ca_consumption=-44` 一类不可能值，本阶段先使用 `data/副本卤化工段数据点位.xlsx` 中的点位正常上下限清洗 DCS：低于下限或高于上限的值置为缺失，不做裁剪和插值；胶液流量无效或小于等于 0 时，对应时刻钙单耗置为缺失。

实验记录文档已先备份并去重，去重报告见 `{dedup_report_path}`。DCS 清洗摘要：清洗前钙单耗最小值 `{cleaning.get('before_cleaning_ca_consumption_min')}`，清洗后最小值 `{cleaning.get('after_cleaning_ca_consumption_min')}`，清洗后最大值 `{cleaning.get('after_cleaning_ca_consumption_max')}`，possible shutdown/invalid operation 时间戳数 `{cleaning.get('possible_shutdown_timestamp_count')}`。

T90 文件使用 `2026.1.xlsx`、`2026.2.xlsx`、`2026.3C.xlsx`，仅保留胶种为 `卤化橡胶` 的记录，排除 `氯丁基橡胶` 等其他胶种。卤化橡胶有效 T90 行数为 `{t90.get('halogen_t90_rows_after_time_and_t90_cleaning')}`。

清洗后运行特征质量：`{report.get('feature_quality_summary')}`。推荐回放覆盖率 `{rec.get('recommendation_coverage')}`，inside/above/below 数量分别为 `{rec.get('inside_band_count')}`、`{rec.get('above_band_count')}`、`{rec.get('below_band_count')}`，不可能钙单耗数量 `{rec.get('impossible_current_ca_consumption_count')}`。

更严格的一对一 T90 回填结果：`{one}`。清晰标签不确定性与边界样本结果见 clear-label 报告。future 与历史参考漂移摘要：`{drift}`。分月稳定性摘要：`{monthly}`。

validation_mode：`{report.get('validation_mode')}`；recommended_next_step：`{report.get('recommended_next_step')}`。

局限性：点位上下限质量依赖配置表；T90 测量误差约 0.1；future raw 点位映射依赖文件命名和格式；本阶段仍为 monitor-only，不自动控制，不写回 DCS。
"""
    with doc_path.open("a", encoding="utf-8") as handle:
        handle.write(section)
    if dedup_report_path.exists():
        payload = json.loads(dedup_report_path.read_text(encoding="utf-8"))
        payload["stage41_appended_after_dedup"] = True
        write_json(dedup_report_path, payload)


def update_method_doc(path: Path) -> None:
    title = "## future holdout 原始 DCS 上下限清洗与卤化橡胶 T90 回填验证"
    section = f"""

{title}

本阶段使用 `data/副本卤化工段数据点位.xlsx` 中的正常上下限对 future DCS 原始值做上线前验证清洗。清洗只用于数据质量控制：越界值置为缺失，不裁剪、不插值，也不更新 safe-band artifact、q33/q66 边界或推荐规则。异常值可能来自设备故障、通信异常、停车或非正常操作。

清洗后的 raw DCS 再构造在线运行特征：过程变量使用 `[t_now-60min, t_now]` trailing 窗口均值；钙单耗先逐点计算 `ca_feed / rubber_flow_2`，再在 60 分钟窗口内取均值。胶液流量无效、低于下限或小于等于 0 时，该时刻钙单耗为缺失；窗口有效点不足时不生成推荐。

future T90 来自 `2026.1.xlsx`、`2026.2.xlsx`、`2026.3C.xlsx`，只保留 `卤化橡胶`，排除 `氯丁基橡胶` 及其他胶种。回填验证同时报告 many-predictions-to-nearest-T90 和 one-T90-to-nearest-prediction；其中一对一策略更适合作为可信度验证。T90 ±0.1 的测量不确定性用 clear-label 指标单独报告。

本阶段还在追加 Stage 41 前对 `docs/Experimental_Procedure_cn.md` 做备份和安全去重：只自动删除 exact/near duplicate，非一致冲突保留并报告。全流程不自动控制，不写回 DCS。
"""
    text = path.read_text(encoding="utf-8") if path.exists() else "# 稳定钙单耗安全带 MVP 方法与数据流\n"
    if title in text:
        text = text.split(title)[0].rstrip() + section
    else:
        text = text.rstrip() + section
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def plot_figures(raw: pd.DataFrame, cleaned: pd.DataFrame, out_summary: pd.DataFrame, replay: pd.DataFrame, drift: pd.DataFrame, risk_table: pd.DataFrame, monthly: pd.DataFrame, figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    x = np.arange(len(replay))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(x, pd.to_numeric(replay["recommended_ca_consumption_min"], errors="coerce"), pd.to_numeric(replay["recommended_ca_consumption_max"], errors="coerce"), alpha=0.35, color="#90CAF9", label="推荐安全带")
    ax.plot(x, pd.to_numeric(replay["current_ca_consumption"], errors="coerce"), color="#333333", linewidth=0.8, label="实际钙单耗")
    ax.set_title("清洗后 future 新数据钙单耗与推荐安全带覆盖图")
    ax.legend()
    fig.tight_layout()
    path = figure_dir / "future_cleaned_v1_ca_band_coverage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path))

    counts = replay["interval_position"].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(counts.index.astype(str), counts.values, color="#4E79A7")
    ax.set_title("清洗后 future 新数据区间位置分布")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = figure_dir / "future_cleaned_v1_interval_position_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path))

    if not out_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        s = out_summary.sort_values("out_of_bound_rate", ascending=False)
        ax.bar(s["point"], s["out_of_bound_rate"], color="#F28E2B")
        ax.set_title("future 原始 DCS 点位越界率")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = figure_dir / "future_dcs_out_of_bound_rates.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))

    before = compute_ca_ratio(raw)
    after = compute_ca_ratio(cleaned)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(before.dropna(), bins=80, alpha=0.45, label="清洗前", color="#E15759")
    ax.hist(after.dropna(), bins=80, alpha=0.55, label="清洗后", color="#59A14F")
    ax.set_title("future 清洗前后钙单耗分布对比")
    ax.legend()
    fig.tight_layout()
    path = figure_dir / "future_cleaned_vs_raw_ca_consumption.png"
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
        ax.set_title("清洗后 future 与历史工况特征分布对比")
        ax.legend()
        fig.tight_layout()
        path = figure_dir / "future_cleaned_vs_historical_feature_drift.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))

    one_risk = risk_table[risk_table["strategy"].eq("one_t90_to_nearest_prediction")] if not risk_table.empty else pd.DataFrame()
    if not one_risk.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        idx = np.arange(len(one_risk))
        width = 0.22
        for offset, metric in [(-width, "ok_rate"), (0, "high_rate"), (width, "low_rate")]:
            ax.bar(idx + offset, one_risk[metric], width=width, label=metric)
        ax.set_xticks(idx)
        ax.set_xticklabels(one_risk["interval_position"], rotation=20)
        ax.set_title("清洗后 future 卤化橡胶 T90 回填下区间内外风险对比")
        ax.legend()
        fig.tight_layout()
        path = figure_dir / "future_cleaned_t90_backfill_risk_summary.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))

    if not monthly.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(monthly["month"], monthly["inside_high_rate"], marker="o", label="inside high")
        ax.plot(monthly["month"], monthly["above_high_rate"], marker="o", label="above high")
        ax.plot(monthly["month"], monthly["below_low_rate"], marker="o", label="below low")
        ax.set_title("清洗后 future 分月风险分离稳定性")
        ax.legend()
        fig.tight_layout()
        path = figure_dir / "future_cleaned_monthly_risk_stability.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def decide(dedup: dict[str, Any], bounds: dict[str, Any], raw: dict[str, Any], clean: dict[str, Any], t90: dict[str, Any], feature: dict[str, Any], rec: dict[str, Any], backfill: dict[str, Any], drift: dict[str, Any], monthly: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    one = backfill.get("one_t90_to_nearest_prediction", {})
    flags = {
        "experiment_doc_dedup_completed": bool(dedup.get("cleaned_doc_written")),
        "experiment_doc_conflicts_require_manual_review": bool(dedup.get("conflicts_require_manual_review")),
        "point_bounds_loaded": bool(bounds.get("point_bounds_parse_pass")),
        "dcs_cleaning_pass": bool(clean.get("cleaning_pass")),
        "impossible_ca_removed": bool(rec.get("impossible_current_ca_consumption_count") == 0 and clean.get("after_cleaning_ca_consumption_min") is not None and clean.get("after_cleaning_ca_consumption_min") >= 0),
        "future_t90_filter_pass": bool(t90.get("future_t90_filter_pass")),
        "one_to_one_backfill_confirms_risk_separation": bool(one.get("risk_guardrail_pass")),
        "monthly_risk_separation_stable": bool(monthly.get("monthly_risk_separation_stable")),
        "future_within_historical_support": bool(drift.get("future_within_historical_support")),
    }
    flags["factory_test_ready_with_cleaned_future_evidence"] = bool(flags["point_bounds_loaded"] and flags["dcs_cleaning_pass"] and flags["impossible_ca_removed"] and flags["future_t90_filter_pass"] and flags["one_to_one_backfill_confirms_risk_separation"] and not flags["experiment_doc_conflicts_require_manual_review"])
    if raw.get("missing_required_points"):
        return "failed_raw_parsing", "fix_future_data_mapping", flags
    if not bounds.get("point_bounds_parse_pass"):
        return "failed_point_bounds_loading", "fix_point_bounds_config", flags
    if not rec.get("future_replay_pass"):
        return "failed_runtime_scoring", "stop_due_to_runtime_failure", flags
    if not t90.get("future_t90_available"):
        return "failed_t90_parsing", "collect_more_future_lims_t90_for_backfill_validation", flags
    if dedup.get("conflicts_require_manual_review") and dedup.get("section_number_conflict_count", 0) > 0:
        return "cleaned_runtime_plus_t90_backfill", "fix_experiment_doc_conflicts", flags
    if flags["one_to_one_backfill_confirms_risk_separation"] and flags["impossible_ca_removed"] and flags["monthly_risk_separation_stable"]:
        if drift.get("future_within_historical_support") is False:
            return "cleaned_runtime_plus_t90_backfill", "investigate_future_distribution_shift", flags
        return "cleaned_runtime_plus_t90_backfill", "prepare_V1_monitor_only_factory_test_with_cleaned_future_evidence", flags
    return "cleaned_runtime_plus_t90_backfill", "keep_future_result_as_preliminary_only", flags


def main() -> None:
    configure_chinese_font()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    target_rubber = repair_mojibake_text(args.target_rubber_type)
    dedup_dir = args.output_dir / "doc_dedup_audit"
    dedup_report = dedup_experiment_doc(args.doc, dedup_dir)
    bounds, bounds_report = parse_point_bounds(args.point_config, args.output_dir, args.table_dir)
    raw, raw_report, raw_quality = parse_future_raw_before_cleaning(args.future_dir, args.output_dir)
    cleaned, out_rows, clean_summary, clean_report = clean_raw_with_bounds(raw, bounds, args.output_dir, args.table_dir)
    t90_files = resolve_t90_files(args.future_dir, args.future_t90_files)
    t90_all, t90_halogen, t90_report = parse_future_t90(t90_files, target_rubber, args.output_dir, args.table_dir)
    runtime, feature_quality, feature_report = build_runtime_features(cleaned, args.eval_frequency_minutes, args.min_valid_points, args.output_dir)
    replay, rec_report = score_future(runtime, args.deploy_dir, args.output_dir, args.table_dir)
    many, one, backfill_report, risk_table, clear_table = align_t90_strategies(replay, t90_halogen, args.residence_minutes, args.t90_match_tolerance_minutes, args.output_dir, args.table_dir)
    hist_path = resolve_historical(args.historical_reference)
    drift, drift_report = compare_historical(runtime, hist_path, args.output_dir, args.table_dir)
    monthly, monthly_report = monthly_summary(replay, cleaned, out_rows, t90_halogen, one, drift, args.output_dir, args.table_dir)
    figures = plot_figures(raw, cleaned, clean_summary, replay, drift, risk_table, monthly, args.figure_dir)
    validation_mode, next_step, flags = decide(dedup_report, bounds_report, raw_report, clean_report, t90_report, feature_report, rec_report, backfill_report, drift_report, monthly_report)
    safety = {"monitor_only": True, "automatic_control": False, "dcs_writeback": False, "future_data_updates_artifact": False, "future_data_updates_boundaries": False}
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "future_dir": str(args.future_dir),
        "point_config_path": str(resolve_point_config(args.point_config)),
        "future_t90_files": [str(p) for p in t90_files],
        "target_rubber_type": target_rubber,
        "deploy_dir": str(args.deploy_dir),
        "historical_reference_path": str(hist_path) if hist_path else None,
        "output_dir": str(args.output_dir),
        "doc_dedup_summary": dedup_report,
        "point_bounds_summary": bounds_report,
        "raw_quality_before_cleaning_summary": raw_report,
        "dcs_cleaning_summary": clean_report,
        "t90_parse_summary": t90_report,
        "feature_quality_summary": feature_report,
        "recommendation_distribution_summary": rec_report,
        "t90_backfill_validation_summary": backfill_report,
        "one_to_one_backfill_summary": backfill_report.get("one_t90_to_nearest_prediction", {}),
        "clear_label_validation_summary": {"many_and_one_to_one_tables": str(args.table_dir / "future_t90_clear_label_validation_summary.csv")},
        "future_vs_historical_drift_summary": drift_report,
        "monthly_stability_summary": monthly_report,
        "safety_check_summary": safety,
        "validation_mode": validation_mode,
        "holdout_principle": "Historical data is prelaunch reference only; cleaned future holdout data does not update frozen V1 before scoring.",
        "final_decision_flags": flags,
        "limitations": ["Normal bounds depend on point-config quality.", "T90 measurement error is about 0.1.", "Future raw point mapping depends on file naming and format.", "Monitor-only validation; no automatic control and no DCS writeback."],
        "generated_figures": figures,
        "recommended_next_step": next_step,
    }
    write_json(args.output_dir / "future_holdout_v1_cleaned_validation_report.json", report)
    append_stage41(args.doc, report, dedup_dir / "experiment_doc_dedup_report.json")
    dedup_report = json.loads((dedup_dir / "experiment_doc_dedup_report.json").read_text(encoding="utf-8"))
    report["doc_dedup_summary"] = dedup_report
    write_json(args.output_dir / "future_holdout_v1_cleaned_validation_report.json", report)
    update_method_doc(args.method_doc)
    print("Future cleaned holdout V1 validation summary")
    print("doc_removed_duplicates:", dedup_report.get("removed_duplicate_count"))
    print("doc_conflicts_require_manual_review:", dedup_report.get("conflicts_require_manual_review"))
    print("point_bounds_found:", bounds_report.get("bounds_found_count"))
    print("cleaning_pass:", clean_report.get("cleaning_pass"))
    print("before/after ca min:", clean_report.get("before_cleaning_ca_consumption_min"), clean_report.get("after_cleaning_ca_consumption_min"))
    print("future_t90_rows:", t90_report.get("halogen_t90_rows_after_time_and_t90_cleaning"))
    print("feature_valid_rate:", feature_report.get("feature_valid_rate"))
    print("recommendation_coverage:", rec_report.get("recommendation_coverage"))
    print("inside/above/below:", rec_report.get("inside_band_count"), rec_report.get("above_band_count"), rec_report.get("below_band_count"))
    print("one_to_one_guardrail:", backfill_report.get("one_t90_to_nearest_prediction", {}).get("risk_guardrail_pass"))
    print("validation_mode:", validation_mode)
    print("recommended_next_step:", next_step)
    print("No generated outputs were written under data/.")


if __name__ == "__main__":
    main()
