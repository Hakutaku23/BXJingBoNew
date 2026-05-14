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
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_v1_on_future_cleaned_raw_data as base


T90_LOW = 8.20
T90_HIGH = 8.70
C_LINE_DEPLOY_NAME = "ca_safe_band_mvp_c_line"
OLD_MERGED_DEPLOY_NAME = "ca_safe_band_mvp"
TARGET_RUBBER_DEFAULT = "\u5364\u5316\u6a61\u80f6"

TIME_CANDIDATES = [
    "time",
    "\u65f6\u95f4",
    "\u65e5\u671f",
    "\u91c7\u6837\u65f6\u95f4",
    "\u68c0\u6d4b\u65f6\u95f4",
    "\u5206\u6790\u65f6\u95f4",
    "\u5316\u9a8c\u65f6\u95f4",
    "\u53d6\u6837\u65f6\u95f4",
    "sample_time",
    "test_time",
    "datetime",
]
RUBBER_CANDIDATES = [
    "\u6a61\u80f6\u79cd\u7c7b",
    "\u80f6\u79cd",
    "\u6a61\u80f6\u7c7b\u578b",
    "\u4ea7\u54c1\u7c7b\u578b",
    "\u4ea7\u54c1\u540d\u79f0",
    "\u6837\u54c1\u540d\u79f0",
    "\u7269\u6599\u540d\u79f0",
    "\u724c\u53f7",
    "rubber_type",
    "material",
    "product",
    "sample_name",
]
LINE_CANDIDATES = [
    "\u7ebf\u522b",
    "\u751f\u4ea7\u7ebf",
    "\u88c5\u7f6e",
    "\u7ebf\u53f7",
    "\u751f\u4ea7\u88c5\u7f6e",
    "C/D/E",
    "line",
    "train",
    "unit",
    "\u91c7\u6837\u70b9\u63cf\u8ff0",
    "\u91c7\u6837\u70b9",
]
T90_CANDIDATES = [
    "t90",
    "T90",
    "T_90",
    "t 90",
    "\u95e8\u5c3cT90",
    "\u786b\u5316T90",
    "T90\u503c",
    "t\u00b4c(90),min",
    "t'c(90),min",
]

FEATURES_FOR_DRIFT = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate frozen C-line-only V1 calcium safe-band package on cleaned future holdout data."
    )
    parser.add_argument("--future-dir", type=Path, default=Path("data/future"))
    parser.add_argument("--point-config", type=Path, default=Path("data") / "\u526f\u672c\u5364\u5316\u5de5\u6bb5\u6570\u636e\u70b9\u4f4d.xlsx")
    parser.add_argument("--future-t90-files", type=Path, nargs="*", default=None)
    parser.add_argument("--target-rubber-type", type=str, default=TARGET_RUBBER_DEFAULT)
    parser.add_argument("--target-line", type=str, default="C")
    parser.add_argument("--historical-reference", type=Path, default=Path("runs/c_line_revalidation/t90_ca_feature_dataset_c_line.parquet"))
    parser.add_argument("--c-line-revalidation-dir", type=Path, default=Path("runs/c_line_revalidation"))
    parser.add_argument("--deploy-dir", type=Path, default=Path("deploy/ca_safe_band_mvp_c_line"))
    parser.add_argument("--residence-minutes", type=int, default=174)
    parser.add_argument("--t90-match-tolerance-minutes", type=int, default=90)
    parser.add_argument("--eval-frequency-minutes", type=int, default=10)
    parser.add_argument("--min-valid-points", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/c_line_future_holdout_v1_cleaned_validation"))
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--method-doc", type=Path, default=Path("docs/ca_safe_band_mvp_method_and_dataflow.md"))
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    base.write_json(path, payload)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def qstats(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None}
    q = values.quantile([0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": float(q.loc[0]),
        "q25": float(q.loc[0.25]),
        "median": float(q.loc[0.5]),
        "q75": float(q.loc[0.75]),
        "max": float(q.loc[1.0]),
    }


def safe_ratio_min_max(df: pd.DataFrame) -> tuple[Any, Any]:
    if df.empty or "ca_feed" not in df.columns or "rubber_flow_2" not in df.columns:
        return None, None
    ratio = base.compute_raw_ca_ratio(df)
    ratio = pd.to_numeric(ratio, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return None, None
    return float(ratio.min()), float(ratio.max())


def ensure_output_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def resolve_t90_files(future_dir: Path, requested: list[Path] | None) -> list[Path]:
    if requested:
        return requested
    result: list[Path] = []
    for stem in ["2026.1", "2026.2", "2026.3C"]:
        xlsx = future_dir / f"{stem}.xlsx"
        xls = future_dir / f"{stem}.xls"
        if xlsx.exists():
            result.append(xlsx)
        elif xls.exists():
            result.append(xls)
    return result


def resolve_c_line_historical_reference(path: Path, c_line_revalidation_dir: Path) -> tuple[Path | None, list[str]]:
    warnings: list[str] = []
    if path.exists():
        return path, warnings
    if c_line_revalidation_dir.exists():
        direct = c_line_revalidation_dir / "t90_ca_feature_dataset_c_line.parquet"
        if direct.exists():
            return direct, warnings
        matches = sorted(c_line_revalidation_dir.rglob("*c_line*t90_ca_feature*.parquet"))
        if matches:
            return matches[0], warnings
        fallback = c_line_revalidation_dir / "ca_safe_band_mvp" / "final_monitor_dry_run.parquet"
        if fallback.exists():
            warnings.append("using_c_line_final_monitor_dry_run_as_reference_metadata")
            return fallback, warnings
    warnings.append("c_line_historical_reference_not_found; old merged-line reference not used")
    return None, warnings


def create_supersession_outputs(output_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = [
        (Path("runs/future_holdout_v1_validation"), "future_holdout_validation_old_merged_line"),
        (Path("runs/future_holdout_v1_cleaned_validation"), "future_holdout_cleaned_validation_old_merged_line"),
        (Path("runs/future_holdout_validation_audit"), "future_holdout_validation_audit_old_merged_line"),
        (Path("runs/v1_factory_test_readiness_pack"), "factory_test_readiness_old_merged_line"),
    ]
    rows = []
    for path, evidence_type in candidates:
        rows.append(
            {
                "old_evidence_path": str(path),
                "path_exists": bool(path.exists()),
                "evidence_type": evidence_type,
                "used_old_package": True,
                "old_package_path": "deploy/ca_safe_band_mvp",
                "superseded_for_c_line_deployment": True,
                "reason_cn": "\u65e7 V1 \u8bc1\u636e\u57fa\u4e8e C/D/E \u5408\u5e76\u7ebf\u5047\u8bbe\uff0c\u4e0d\u80fd\u4f5c\u4e3a C \u7ebf\u90e8\u7f72 Go/No-Go \u8bc1\u636e\u3002",
                "allowed_future_use_cn": "\u4ec5\u53ef\u4f5c\u4e3a\u65b9\u6cd5\u5f00\u53d1\u6216\u5386\u53f2\u53c2\u8003\uff0c\u4e0d\u53ef\u4f5c\u4e3a C \u7ebf\u90e8\u7f72\u51b3\u7b56\u8bc1\u636e\u3002",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "old_merged_line_evidence_supersession.csv", index=False, encoding="utf-8-sig")
    df.to_csv(table_dir / "c_line_old_merged_line_evidence_supersession.csv", index=False, encoding="utf-8-sig")
    report = {
        "old_merged_line_evidence_superseded": True,
        "old_package_path": "deploy/ca_safe_band_mvp",
        "c_line_package_required": "deploy/ca_safe_band_mvp_c_line",
        "superseded_paths": df.to_dict(orient="records"),
        "deletion_performed": False,
    }
    write_json(output_dir / "old_merged_line_evidence_supersession_report.json", report)
    return df, report


def write_c_line_bound_table(bounds: pd.DataFrame, report: dict[str, Any], table_dir: Path) -> None:
    if bounds.empty:
        bounds.to_csv(table_dir / "c_line_future_point_bounds_used.csv", index=False, encoding="utf-8-sig")
        return
    bounds.to_csv(table_dir / "c_line_future_point_bounds_used.csv", index=False, encoding="utf-8-sig")


def normalize_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value).strip())


def rubber_matches(value: Any, target: str) -> bool:
    text = normalize_string(value)
    target_text = normalize_string(target)
    if not text:
        return False
    if "\u6c2f\u4e01\u57fa" in text:
        return False
    return text == target_text or target_text in text


def line_matches(value: Any, target_line: str) -> bool:
    text = normalize_string(value).upper()
    target = normalize_string(target_line).upper()
    if not text or not target:
        return False
    accepted = {target, f"{target}\u7ebf"}
    if text in accepted:
        return True
    if text.startswith(f"{target}-") or text.startswith(f"{target}_"):
        return True
    if f"{target}\u7ebf" in text:
        return True
    return False


def choose_column(columns: list[Any], candidates: list[str]) -> Any | None:
    return base.pick_column(columns, candidates)


def parse_future_t90_c_line(
    files: list[Path],
    target_rubber_type: str,
    target_line: str,
    output_dir: Path,
    table_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_rows: list[pd.DataFrame] = []
    sheet_rows: list[dict[str, Any]] = []
    parsed_files: list[str] = []
    failed_files: list[dict[str, str]] = []
    selected_time: dict[str, Any] = {}
    selected_rubber: dict[str, Any] = {}
    selected_line: dict[str, Any] = {}
    selected_t90: dict[str, Any] = {}
    warnings: list[str] = []

    for requested_path in files:
        path = requested_path
        if not path.exists() and path.suffix.lower() == ".xls":
            xlsx = path.with_suffix(".xlsx")
            if xlsx.exists():
                path = xlsx
        if not path.exists():
            failed_files.append({"file": str(requested_path), "error": "file_not_found"})
            continue
        try:
            if path.suffix.lower() == ".csv":
                for encoding in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
                    try:
                        frame = pd.read_csv(path, encoding=encoding)
                        break
                    except Exception:
                        frame = None
                if frame is None:
                    raise RuntimeError("csv_read_failed")
                sheet_map = {"csv": frame}
            else:
                xls = pd.ExcelFile(path)
                sheet_map = {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in xls.sheet_names}
            parsed_files.append(str(path))
        except Exception as exc:
            failed_files.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue

        file_had_line_column = False
        for sheet_name, frame in sheet_map.items():
            original_count = int(len(frame))
            if frame.empty:
                sheet_rows.append({"file": str(path), "sheet": sheet_name, "row_count": 0, "valid_output_rows": 0})
                continue
            columns = list(frame.columns)
            time_col = choose_column(columns, TIME_CANDIDATES)
            rubber_col = choose_column(columns, RUBBER_CANDIDATES)
            line_col = choose_column(columns, LINE_CANDIDATES)
            t90_col = choose_column(columns, T90_CANDIDATES)
            key = f"{path.name}:{sheet_name}"
            selected_time[key] = str(time_col) if time_col is not None else None
            selected_rubber[key] = str(rubber_col) if rubber_col is not None else None
            selected_line[key] = str(line_col) if line_col is not None else None
            selected_t90[key] = str(t90_col) if t90_col is not None else None
            if line_col is not None:
                file_had_line_column = True
            if time_col is None or rubber_col is None or t90_col is None:
                sheet_rows.append(
                    {
                        "file": str(path),
                        "sheet": sheet_name,
                        "row_count": original_count,
                        "valid_output_rows": 0,
                        "missing_required_columns": True,
                    }
                )
                continue
            out = pd.DataFrame(
                {
                    "source_file": str(path),
                    "source_sheet": sheet_name,
                    "time": pd.to_datetime(frame[time_col], errors="coerce"),
                    "rubber_type": frame[rubber_col].astype(str),
                    "t90": pd.to_numeric(frame[t90_col], errors="coerce"),
                }
            )
            if line_col is not None:
                out["line_value"] = frame[line_col].astype(str)
                out["line_filter_source"] = str(line_col)
            elif "3C" in path.stem.upper():
                out["line_value"] = target_line
                out["line_filter_source"] = "filename_3C_hint"
                warnings.append(f"filename_line_hint_used_for_{path.name}_{sheet_name}")
            else:
                out["line_value"] = ""
                out["line_filter_source"] = "missing_line_column"
                warnings.append(f"missing_line_column_for_{path.name}_{sheet_name}; rows_not_silently_treated_as_C")
            out["rubber_filter_match"] = out["rubber_type"].map(lambda x: rubber_matches(x, target_rubber_type))
            out["c_line_filter_match"] = out["line_value"].map(lambda x: line_matches(x, target_line))
            all_rows.append(out)
            sheet_rows.append(
                {
                    "file": str(path),
                    "sheet": sheet_name,
                    "row_count": original_count,
                    "rubber_match_rows": int(out["rubber_filter_match"].sum()),
                    "c_line_match_rows": int(out["c_line_filter_match"].sum()),
                    "halogen_c_line_rows": int((out["rubber_filter_match"] & out["c_line_filter_match"]).sum()),
                }
            )
        if not file_had_line_column and "3C" not in path.stem.upper():
            warnings.append(f"missing_line_column_for_file_{path.name}")

    raw_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if raw_all.empty:
        halogen_c = pd.DataFrame(columns=["time", "t90", "rubber_type", "line_value", "source_file", "source_sheet"])
    else:
        raw_all["time_valid"] = raw_all["time"].notna()
        raw_all["t90_valid"] = pd.to_numeric(raw_all["t90"], errors="coerce").gt(0)
        halogen_c = raw_all[raw_all["rubber_filter_match"] & raw_all["c_line_filter_match"]].copy()
        halogen_c = halogen_c[halogen_c["time_valid"] & halogen_c["t90_valid"]].copy()
        halogen_c = halogen_c.sort_values("time")

    raw_all.to_parquet(output_dir / "future_t90_raw_all.parquet", index=False)
    raw_all.to_csv(output_dir / "future_t90_raw_all.csv", index=False, encoding="utf-8-sig")
    halogen_c.to_parquet(output_dir / "future_t90_halogen_c_line_only.parquet", index=False)
    halogen_c.to_csv(output_dir / "future_t90_halogen_c_line_only.csv", index=False, encoding="utf-8-sig")

    if not raw_all.empty:
        rubber_counts = raw_all["rubber_type"].fillna("").astype(str).value_counts(dropna=False).to_dict()
        line_counts = raw_all["line_value"].fillna("").astype(str).value_counts(dropna=False).to_dict()
    else:
        rubber_counts = {}
        line_counts = {}
    halogen_raw = int(raw_all["rubber_filter_match"].sum()) if "rubber_filter_match" in raw_all else 0
    c_line_raw = int(raw_all["c_line_filter_match"].sum()) if "c_line_filter_match" in raw_all else 0
    halogen_c_raw = int((raw_all["rubber_filter_match"] & raw_all["c_line_filter_match"]).sum()) if "rubber_filter_match" in raw_all else 0
    dropped_missing_time = int((raw_all["rubber_filter_match"] & raw_all["c_line_filter_match"] & ~raw_all["time_valid"]).sum()) if "time_valid" in raw_all else 0
    dropped_bad_t90 = int((raw_all["rubber_filter_match"] & raw_all["c_line_filter_match"] & raw_all["time_valid"] & ~raw_all["t90_valid"]).sum()) if "t90_valid" in raw_all else 0
    missing_rubber_col = bool(not selected_rubber or all(v is None for v in selected_rubber.values()))
    missing_line_col = bool(not selected_line or all(v is None for v in selected_line.values()))
    missing_t90_col = bool(not selected_t90 or all(v is None for v in selected_t90.values()))
    report = {
        "future_t90_files_requested": [str(p) for p in files],
        "future_t90_files_found": [str(p) for p in files if p.exists() or p.with_suffix(".xlsx").exists()],
        "future_t90_files_parsed": parsed_files,
        "future_t90_files_failed": failed_files,
        "selected_time_column_by_file": selected_time,
        "selected_rubber_type_column_by_file": selected_rubber,
        "selected_line_column_by_file": selected_line,
        "selected_t90_column_by_file": selected_t90,
        "sheet_summary_by_file": sheet_rows,
        "total_t90_rows": int(len(raw_all)),
        "halogen_raw_rows_before_t90_cleaning": halogen_raw,
        "c_line_raw_rows_before_t90_cleaning": c_line_raw,
        "halogen_c_line_rows_before_t90_cleaning": halogen_c_raw,
        "halogen_c_line_t90_rows_after_time_and_t90_cleaning": int(len(halogen_c)),
        "excluded_non_halogen_rows": int(len(raw_all) - halogen_raw) if not raw_all.empty else 0,
        "excluded_non_c_line_rows": int(halogen_raw - halogen_c_raw) if not raw_all.empty else 0,
        "dropped_missing_time_rows": dropped_missing_time,
        "dropped_missing_or_nonnumeric_t90_rows": dropped_bad_t90,
        "rubber_type_value_counts": rubber_counts,
        "line_value_counts": line_counts,
        "t90_time_min": halogen_c["time"].min().isoformat() if not halogen_c.empty else None,
        "t90_time_max": halogen_c["time"].max().isoformat() if not halogen_c.empty else None,
        **{f"t90_{k}": v for k, v in qstats(halogen_c["t90"] if "t90" in halogen_c else pd.Series(dtype=float)).items()},
        "missing_rubber_type_column": missing_rubber_col,
        "missing_line_column": missing_line_col,
        "missing_t90_column": missing_t90_col,
        "future_t90_available": bool(len(halogen_c) > 0 and not missing_t90_col),
        "future_t90_filter_pass": bool(not missing_rubber_col and halogen_raw > 0),
        "future_t90_c_line_filter_pass": bool(not missing_line_col and halogen_c_raw > 0),
        "warnings": sorted(set(warnings)),
    }
    write_json(output_dir / "future_t90_parse_report.json", report)
    summary = pd.DataFrame(sheet_rows)
    summary.to_csv(table_dir / "c_line_future_t90_halogen_c_line_filter_summary.csv", index=False, encoding="utf-8-sig")
    return halogen_c, report


def load_c_line_recommender(deploy_dir: Path) -> Any:
    if not deploy_dir.exists():
        raise FileNotFoundError(f"C-line deploy package not found: {deploy_dir}")
    if deploy_dir.name != C_LINE_DEPLOY_NAME:
        raise RuntimeError(f"Refusing to score with non-C-line deploy dir: {deploy_dir}")
    old_path = Path("deploy") / OLD_MERGED_DEPLOY_NAME
    if deploy_dir.resolve() == old_path.resolve():
        raise RuntimeError("Refusing to use old merged-line deploy package.")
    interface_path = deploy_dir / "interface.py"
    if not interface_path.exists():
        raise FileNotFoundError(f"C-line interface.py not found: {interface_path}")
    sys.dont_write_bytecode = True
    deploy_abs = str(deploy_dir.resolve())
    old_sys_path = list(sys.path)
    try:
        if deploy_abs not in sys.path:
            sys.path.insert(0, deploy_abs)
        spec = importlib.util.spec_from_file_location("ca_safe_band_mvp_c_line_interface", interface_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load C-line interface from {interface_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.init(model_dir=deploy_dir, mode="production")
    finally:
        sys.path = old_sys_path


def score_c_line_future(features: pd.DataFrame, deploy_dir: Path, output_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    recommender = load_c_line_recommender(deploy_dir)
    preds = recommender.predict_batch(features, mode="production")
    if not isinstance(preds, pd.DataFrame):
        preds = pd.DataFrame(preds)
    for col in [
        "time",
        "feature_quality",
        "missing_raw_columns",
        "insufficient_window_features",
        "warning_flags",
        "ca_per_rubber_flow_win_60_mean",
        "rubber_flow_2_win_60_mean",
    ]:
        if col in features.columns and col not in preds.columns:
            preds[col] = features[col].values
    if "time" in features.columns and "time" in preds.columns:
        preds["time"] = pd.to_datetime(preds["time"], errors="coerce")
    artifact = read_json(deploy_dir / "safe_band_artifact.json")
    final_strategy = artifact.get("final_strategy") or artifact.get("strategy")
    current = pd.to_numeric(preds.get("current_ca_consumption"), errors="coerce")
    impossible = int(((current < 0) | np.isinf(current)).sum())
    scored_count = int(len(preds))
    valid_rec = preds.get("recommended_ca_consumption_min", pd.Series(index=preds.index, dtype=float)).notna()
    pos_counts = preds.get("interval_position", pd.Series(index=preds.index, dtype=object)).value_counts(dropna=False).to_dict()
    action_counts = preds.get("action_visibility", pd.Series(index=preds.index, dtype=object)).value_counts(dropna=False).to_dict()
    report = {
        "scored_row_count": scored_count,
        "recommendation_coverage": float(valid_rec.mean()) if scored_count else 0.0,
        "no_recommendation_count": int(scored_count - valid_rec.sum()),
        "input_invalid_count": int((preds.get("input_valid", pd.Series(True, index=preds.index)) == False).sum()) if scored_count else 0,
        "inside_band_count": int(pos_counts.get("inside_band", 0)),
        "above_band_count": int(pos_counts.get("above_band", 0)),
        "below_band_count": int(pos_counts.get("below_band", 0)),
        "manual_review_required_count": int(action_counts.get("manual_review_required", 0)),
        "diagnostic_only_count": int(action_counts.get("diagnostic_only", 0)),
        "monitor_only_count": int(action_counts.get("monitor_only", 0)),
        "missing_required_features_summary": preds.get("missing_required_features", pd.Series(index=preds.index, dtype=object)).astype(str).value_counts().head(20).to_dict(),
        "warning_flags_summary": preds.get("warning_flags", pd.Series(index=preds.index, dtype=object)).astype(str).value_counts().head(20).to_dict(),
        "recommended_ca_consumption_min_distribution": qstats(preds.get("recommended_ca_consumption_min", pd.Series(dtype=float))),
        "recommended_ca_consumption_target_distribution": qstats(preds.get("recommended_ca_consumption_target", pd.Series(dtype=float))),
        "recommended_ca_consumption_max_distribution": qstats(preds.get("recommended_ca_consumption_max", pd.Series(dtype=float))),
        "current_ca_consumption_distribution": qstats(current),
        "impossible_current_ca_consumption_count": impossible,
        "final_strategy": final_strategy,
        "package_path_used": str(deploy_dir),
        "old_merged_package_used": False,
        "future_replay_pass": bool(scored_count > 0 and impossible == 0 and not (current.dropna() < 0).any()),
    }
    preds.to_parquet(output_dir / "future_c_line_v1_recommendation_replay.parquet", index=False)
    preds.to_csv(output_dir / "future_c_line_v1_recommendation_replay.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "future_c_line_v1_recommendation_distribution_report.json", report)
    pd.DataFrame([report]).to_csv(table_dir / "c_line_future_v1_recommendation_distribution_summary.csv", index=False, encoding="utf-8-sig")
    return preds, report


def align_c_line_t90_strategies(
    replay: pd.DataFrame,
    t90: pd.DataFrame,
    residence: int,
    tolerance: int,
    output_dir: Path,
    table_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if t90.empty:
        report = {
            "future_t90_available": False,
            "future_t90_validation_status": "missing_c_line_halogen_t90",
            "many_predictions_to_nearest_t90": {"aligned_sample_count": 0},
            "one_t90_to_nearest_prediction": {"aligned_sample_count": 0},
        }
        write_json(output_dir / "future_t90_backfill_validation_report.json", report)
        return pd.DataFrame(), pd.DataFrame(), report, pd.DataFrame(), pd.DataFrame()
    rec = replay.copy()
    rec["time"] = pd.to_datetime(rec["time"], errors="coerce")
    rec = rec.dropna(subset=["time"]).sort_values("time")
    rec["quality_time"] = rec["time"] + pd.to_timedelta(residence, unit="m")
    labels = t90[["time", "t90", "rubber_type", "line_value", "source_file", "source_sheet"]].copy()
    labels["time"] = pd.to_datetime(labels["time"], errors="coerce")
    labels = labels.dropna(subset=["time", "t90"]).sort_values("time").rename(columns={"time": "t90_time"})
    labels["t90_id"] = np.arange(len(labels))
    many = pd.merge_asof(
        rec.sort_values("quality_time"),
        labels.sort_values("t90_time"),
        left_on="quality_time",
        right_on="t90_time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance),
    ).dropna(subset=["t90"]).copy()
    if not many.empty:
        many["t90_match_delta_minutes"] = (many["t90_time"] - many["quality_time"]).dt.total_seconds().abs() / 60.0
    labels2 = labels.copy()
    labels2["expected_recommendation_time"] = labels2["t90_time"] - pd.to_timedelta(residence, unit="m")
    one = pd.merge_asof(
        labels2.sort_values("expected_recommendation_time"),
        rec.sort_values("time"),
        left_on="expected_recommendation_time",
        right_on="time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance),
    ).dropna(subset=["recommended_ca_consumption_min"]).copy()
    if not one.empty:
        one["t90_match_delta_minutes"] = (one["time"] - one["expected_recommendation_time"]).dt.total_seconds().abs() / 60.0
    many.to_parquet(output_dir / "future_t90_backfill_aligned_many.parquet", index=False)
    many.to_csv(output_dir / "future_t90_backfill_aligned_many.csv", index=False, encoding="utf-8-sig")
    one.to_parquet(output_dir / "future_t90_backfill_aligned_one_to_one.parquet", index=False)
    one.to_csv(output_dir / "future_t90_backfill_aligned_one_to_one.csv", index=False, encoding="utf-8-sig")
    many_risk, many_clear, many_report = base.strategy_report(many, "many_predictions_to_nearest_t90")
    one_risk, one_clear, one_report = base.strategy_report(one, "one_t90_to_nearest_prediction")
    compare = pd.DataFrame(
        [
            {"strategy": "many_predictions_to_nearest_t90", **{k: v for k, v in many_report.items() if k not in ["risk_by_interval_position", "clear_label_risk_by_interval_position"]}},
            {"strategy": "one_t90_to_nearest_prediction", **{k: v for k, v in one_report.items() if k not in ["risk_by_interval_position", "clear_label_risk_by_interval_position"]}},
        ]
    )
    compare.to_csv(table_dir / "c_line_future_backfill_alignment_strategy_comparison.csv", index=False, encoding="utf-8-sig")
    risk_rows = []
    for strategy, risk in [("many_predictions_to_nearest_t90", many_risk), ("one_t90_to_nearest_prediction", one_risk)]:
        if not risk.empty:
            part = risk.copy()
            part.insert(0, "strategy", strategy)
            risk_rows.append(part)
    risk_table = pd.concat(risk_rows, ignore_index=True) if risk_rows else pd.DataFrame()
    risk_table.to_csv(table_dir / "c_line_future_t90_backfill_validation_summary.csv", index=False, encoding="utf-8-sig")
    clear_rows = []
    for strategy, risk in [("many_predictions_to_nearest_t90", many_clear), ("one_t90_to_nearest_prediction", one_clear)]:
        if not risk.empty:
            part = risk.copy()
            part.insert(0, "strategy", strategy)
            clear_rows.append(part)
    clear_table = pd.concat(clear_rows, ignore_index=True) if clear_rows else pd.DataFrame()
    clear_table.to_csv(table_dir / "c_line_future_t90_clear_label_validation_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "future_t90_available": True,
        "future_t90_validation_status": "available_c_line_halogen",
        "many_predictions_to_nearest_t90": many_report,
        "one_t90_to_nearest_prediction": one_report,
    }
    write_json(output_dir / "future_t90_backfill_validation_report.json", report)
    return many, one, report, risk_table, clear_table


def psi_like(hist: pd.Series, fut: pd.Series, bins: int = 10) -> float | None:
    return base.psi_like(hist, fut, bins=bins)


def compare_c_line_historical(runtime: pd.DataFrame, historical_path: Path | None, output_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    if historical_path is not None and historical_path.exists():
        hist = pd.read_parquet(historical_path)
        for feature in FEATURES_FOR_DRIFT:
            if feature not in runtime.columns or feature not in hist.columns:
                continue
            h = pd.to_numeric(hist[feature], errors="coerce").dropna()
            f = pd.to_numeric(runtime[feature], errors="coerce").dropna()
            if h.empty or f.empty:
                continue
            hq = h.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
            fq = f.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
            out_rate = float(((f < hq.loc[0.01]) | (f > hq.loc[0.99])).mean())
            rows.append(
                {
                    "feature": feature,
                    "historical_c_line_min": hq.loc[0],
                    "historical_c_line_q01": hq.loc[0.01],
                    "historical_c_line_q05": hq.loc[0.05],
                    "historical_c_line_q25": hq.loc[0.25],
                    "historical_c_line_median": hq.loc[0.5],
                    "historical_c_line_q75": hq.loc[0.75],
                    "historical_c_line_q95": hq.loc[0.95],
                    "historical_c_line_q99": hq.loc[0.99],
                    "historical_c_line_max": hq.loc[1],
                    "future_cleaned_min": fq.loc[0],
                    "future_cleaned_q01": fq.loc[0.01],
                    "future_cleaned_q05": fq.loc[0.05],
                    "future_cleaned_q25": fq.loc[0.25],
                    "future_cleaned_median": fq.loc[0.5],
                    "future_cleaned_q75": fq.loc[0.75],
                    "future_cleaned_q95": fq.loc[0.95],
                    "future_cleaned_q99": fq.loc[0.99],
                    "future_cleaned_max": fq.loc[1],
                    "future_median_minus_historical_median": float(fq.loc[0.5] - hq.loc[0.5]),
                    "out_of_historical_range_rate": out_rate,
                    "psi_like_drift_score": psi_like(h, f),
                    "future_within_c_line_historical_support": bool(out_rate <= 0.20),
                }
            )
    drift = pd.DataFrame(rows)
    top = drift.sort_values("psi_like_drift_score", ascending=False).head(5)["feature"].tolist() if not drift.empty and "psi_like_drift_score" in drift else []
    report = {
        "c_line_historical_reference_available": historical_path is not None and historical_path.exists(),
        "historical_reference_path": str(historical_path) if historical_path else None,
        "old_merged_line_reference_used": False,
        "feature_count_compared": int(len(drift)),
        "max_out_of_historical_range_rate": float(drift["out_of_historical_range_rate"].max()) if not drift.empty else None,
        "max_psi_like_drift_score": float(drift["psi_like_drift_score"].max()) if not drift.empty else None,
        "future_within_c_line_historical_support": bool(not drift.empty and drift["future_within_c_line_historical_support"].mean() >= 0.70),
        "top_drift_features": top,
    }
    drift.to_csv(output_dir / "future_vs_c_line_historical_feature_drift.csv", index=False, encoding="utf-8-sig")
    drift.to_csv(table_dir / "c_line_future_vs_historical_feature_drift_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "future_vs_c_line_historical_feature_drift_report.json", report)
    return drift, report


def monthly_c_line_summary(
    replay: pd.DataFrame,
    cleaned: pd.DataFrame,
    out_bounds: pd.DataFrame,
    t90: pd.DataFrame,
    one: pd.DataFrame,
    drift: pd.DataFrame,
    output_dir: Path,
    table_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    months = sorted(pd.to_datetime(replay["time"]).dt.to_period("M").astype(str).dropna().unique()) if not replay.empty else []
    rows = []
    for month in months:
        r = replay[pd.to_datetime(replay["time"]).dt.to_period("M").astype(str).eq(month)]
        raw_month = cleaned[pd.to_datetime(cleaned["time"]).dt.to_period("M").astype(str).eq(month)] if not cleaned.empty else pd.DataFrame()
        out_month = out_bounds[pd.to_datetime(out_bounds["time"]).dt.to_period("M").astype(str).eq(month)] if not out_bounds.empty and "time" in out_bounds else pd.DataFrame()
        t90_month = t90[pd.to_datetime(t90["time"]).dt.to_period("M").astype(str).eq(month)] if not t90.empty else pd.DataFrame()
        one_month = one[pd.to_datetime(one["t90_time"]).dt.to_period("M").astype(str).eq(month)] if not one.empty and "t90_time" in one else pd.DataFrame()
        risk = base.summarize_position_risk(base.add_t90_targets(one_month)) if not one_month.empty else pd.DataFrame()
        inside = risk[risk["interval_position"].eq("inside_band")] if not risk.empty else pd.DataFrame()
        above = risk[risk["interval_position"].eq("above_band")] if not risk.empty else pd.DataFrame()
        below = risk[risk["interval_position"].eq("below_band")] if not risk.empty else pd.DataFrame()
        outside = risk[risk["interval_position"].eq("outside_band")] if not risk.empty else pd.DataFrame()
        inside_count = int(inside["sample_count"].iloc[0]) if not inside.empty else 0
        outside_count = int(outside["sample_count"].iloc[0]) if not outside.empty else 0
        enough = bool(len(one_month) >= 30 and inside_count >= 10 and outside_count >= 10)
        guard = bool(
            enough
            and not inside.empty
            and not outside.empty
            and inside["high_rate"].iloc[0] <= outside["high_rate"].iloc[0]
            and inside["out_spec_rate"].iloc[0] <= outside["out_spec_rate"].iloc[0]
        )
        rows.append(
            {
                "month": month,
                "raw_row_count": int(len(raw_month)),
                "out_of_bound_rate": float(len(out_month) / max(len(raw_month) * len(base.RAW_POINT_MAPPING), 1)),
                "feature_valid_rate": float(r["feature_quality"].eq("ok").mean()) if len(r) and "feature_quality" in r else None,
                "recommendation_coverage": float(r["recommended_ca_consumption_min"].notna().mean()) if len(r) and "recommended_ca_consumption_min" in r else None,
                "inside_band_count": int(r["interval_position"].eq("inside_band").sum()) if "interval_position" in r else 0,
                "above_band_count": int(r["interval_position"].eq("above_band").sum()) if "interval_position" in r else 0,
                "below_band_count": int(r["interval_position"].eq("below_band").sum()) if "interval_position" in r else 0,
                "c_line_halogen_t90_count": int(len(t90_month)),
                "one_to_one_aligned_sample_count": int(len(one_month)),
                "inside_high_rate": float(inside["high_rate"].iloc[0]) if not inside.empty else None,
                "above_high_rate": float(above["high_rate"].iloc[0]) if not above.empty else None,
                "below_low_rate": float(below["low_rate"].iloc[0]) if not below.empty else None,
                "risk_guardrail_pass": guard,
                "monthly_evidence_sufficient": enough,
            }
        )
    monthly = pd.DataFrame(rows)
    monthly.to_csv(output_dir / "future_monthly_validation_summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(table_dir / "c_line_future_monthly_validation_summary.csv", index=False, encoding="utf-8-sig")
    sufficient = monthly[monthly["monthly_evidence_sufficient"].fillna(False)] if not monthly.empty and "monthly_evidence_sufficient" in monthly else pd.DataFrame()
    report = {
        "months": monthly.to_dict(orient="records"),
        "sufficient_month_count": int(len(sufficient)),
        "insufficient_month_count": int(len(monthly) - len(sufficient)) if not monthly.empty else 0,
        "monthly_risk_separation_stable": bool(not sufficient.empty and sufficient["risk_guardrail_pass"].fillna(False).all()),
    }
    return monthly, report


def build_readiness_update(
    deploy_dir: Path,
    point_report: dict[str, Any],
    cleaning_report: dict[str, Any],
    t90_report: dict[str, Any],
    rec_report: dict[str, Any],
    backfill_report: dict[str, Any],
    clear_table: pd.DataFrame,
    monthly_report: dict[str, Any],
    drift_report: dict[str, Any],
    output_dir: Path,
    table_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    one = backfill_report.get("one_t90_to_nearest_prediction", {})
    clear_pass = False
    if not clear_table.empty:
        one_clear = clear_table[clear_table["strategy"].eq("one_t90_to_nearest_prediction")]
        inside = one_clear[one_clear["interval_position"].eq("inside_band")]
        outside = one_clear[one_clear["interval_position"].eq("outside_band")]
        clear_pass = bool(not inside.empty and not outside.empty and inside["high_rate"].iloc[0] <= outside["high_rate"].iloc[0])
    rows = [
        ("C-line package used", "pass" if deploy_dir.name == C_LINE_DEPLOY_NAME else "fail", str(deploy_dir), "\u786e\u8ba4\u53ea\u4f7f\u7528 C \u7ebf\u5305\u3002", ""),
        ("Old merged-line package not used", "pass" if not rec_report.get("old_merged_package_used") else "fail", "deploy/ca_safe_band_mvp not imported for scoring", "\u65e7\u5408\u5e76\u7ebf\u5305\u4e0d\u53ef\u4f5c\u4e3a C \u7ebf\u8bc1\u636e\u3002", ""),
        ("C-line top_rule_only strategy confirmed", "pass" if rec_report.get("final_strategy") == "top_rule_only" else "warning", str(rec_report.get("final_strategy")), "\u4eba\u5de5\u590d\u6838 C \u7ebf final_strategy\u3002", ""),
        ("DCS bounds cleaning applied", "pass" if cleaning_report.get("cleaning_pass") else "fail", f"bounds={cleaning_report.get('bounds_applied_count')}", "\u8d8a\u754c\u503c\u8bbe\u4e3a\u7f3a\u5931\uff0c\u4e0d\u526a\u88c1\u3002", ""),
        ("T90 filtered to \u5364\u5316\u6a61\u80f6", "pass" if t90_report.get("future_t90_filter_pass") else "fail", f"rows={t90_report.get('halogen_raw_rows_before_t90_cleaning')}", "\u4ec5\u4f7f\u7528\u5364\u5316\u6a61\u80f6 T90\u3002", ""),
        ("T90 filtered to C line", "pass" if t90_report.get("future_t90_c_line_filter_pass") else "fail", f"rows={t90_report.get('halogen_c_line_t90_rows_after_time_and_t90_cleaning')}", "\u4ec5\u4f7f\u7528 C \u7ebf T90\u3002", ""),
        ("Impossible calcium values removed", "pass" if rec_report.get("impossible_current_ca_consumption_count") == 0 else "fail", str(rec_report.get("current_ca_consumption_distribution")), "\u4e0d\u5141\u8bb8\u8d1f\u503c\u6216\u65e0\u7a77\u94d9\u5355\u8017\u8fdb\u5165\u8bc4\u5206\u7ed3\u679c\u3002", ""),
        ("One-to-one C-line T90 risk separation", "pass" if one.get("risk_guardrail_pass") else "warning", str(one.get("risk_by_interval_position")), "\u4e00 T90 \u5bf9\u4e00\u9884\u6d4b\u662f\u66f4\u4e25\u683c\u7684\u4fe1\u5ea6\u9a8c\u8bc1\u3002", ""),
        ("Clear-label risk separation", "pass" if clear_pass else "warning", f"clear_rows={len(clear_table)}", "\u8fb9\u754c\u4e0d\u786e\u5b9a\u6837\u672c\u9700\u5355\u72ec\u89e3\u8bfb\u3002", ""),
        ("Monthly stability", "pass" if monthly_report.get("monthly_risk_separation_stable") else "warning", str(monthly_report.get("months")), "\u5206\u6708\u6837\u672c\u4e0d\u8db3\u65f6\u9700\u7ee7\u7eed\u7d2f\u79ef\u3002", ""),
        ("C-line readiness remains human_review_required", "pending_human_review", "C-line rebuild readiness was stop_until_more_data", "\u5373\u4f7f future \u590d\u9a8c\u901a\u8fc7\uff0c\u4ecd\u9700\u5de5\u827a\u4e0e\u8fd0\u884c\u4eba\u5458\u590d\u6838\u3002", ""),
        ("stop_until_more_data status considered", "warning", "only 9 monitor-chain candidates in C-line rebuild", "\u4e0d\u76f4\u63a5\u5347\u7ea7\u4e3a\u5df2\u6279\u51c6\u90e8\u7f72\u3002", ""),
        ("No automatic control", "pass", "monitor-only", "\u672c\u9636\u6bb5\u4e0d\u5b9e\u65bd\u81ea\u52a8\u63a7\u5236\u3002", ""),
        ("No DCS writeback", "pass", "no writeback", "\u672c\u9636\u6bb5\u4e0d\u5199\u56de DCS\u3002", ""),
    ]
    table = pd.DataFrame(rows, columns=["item", "status", "evidence", "action_required_cn", "reviewer_note_cn"])
    table.to_csv(output_dir / "c_line_factory_readiness_update.csv", index=False, encoding="utf-8-sig")
    table.to_csv(table_dir / "c_line_factory_readiness_update.csv", index=False, encoding="utf-8-sig")
    report = {
        "final_status": "C_line_monitor_only_candidate_for_human_review",
        "pass_count": int(table["status"].eq("pass").sum()),
        "warning_count": int(table["status"].eq("warning").sum()),
        "fail_count": int(table["status"].eq("fail").sum()),
        "pending_human_review_count": int(table["status"].eq("pending_human_review").sum()),
        "items": table.to_dict(orient="records"),
    }
    return table, report


def plot_outputs(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    replay: pd.DataFrame,
    out_bounds: pd.DataFrame,
    drift: pd.DataFrame,
    risk_table: pd.DataFrame,
    monthly: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    if not replay.empty:
        x = pd.to_datetime(replay["time"])
        plt.plot(x, replay["current_ca_consumption"], label="\u5f53\u524d\u94d9\u5355\u8017", linewidth=1.2)
        plt.plot(x, replay["recommended_ca_consumption_min"], label="\u63a8\u8350\u4e0b\u9650", linewidth=1.0)
        plt.plot(x, replay["recommended_ca_consumption_max"], label="\u63a8\u8350\u4e0a\u9650", linewidth=1.0)
        plt.legend()
    plt.title("C\u7ebf\u6e05\u6d17\u540e future \u65b0\u6570\u636e\u94d9\u5355\u8017\u4e0e\u63a8\u8350\u5b89\u5168\u5e26\u8986\u76d6\u56fe")
    plt.tight_layout()
    plt.savefig(figure_dir / "c_line_future_cleaned_v1_ca_band_coverage.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    if "interval_position" in replay:
        replay["interval_position"].value_counts().plot(kind="bar")
    plt.title("C\u7ebf\u6e05\u6d17\u540e future \u533a\u95f4\u4f4d\u7f6e\u5206\u5e03")
    plt.tight_layout()
    plt.savefig(figure_dir / "c_line_future_cleaned_interval_position_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    if not out_bounds.empty and "friendly_name" in out_bounds:
        rates = out_bounds["friendly_name"].value_counts() / max(len(raw), 1)
        rates.sort_values(ascending=False).plot(kind="bar")
    plt.title("C\u7ebf future \u539f\u59cb DCS \u70b9\u4f4d\u8d8a\u754c\u7387")
    plt.tight_layout()
    plt.savefig(figure_dir / "c_line_future_dcs_out_of_bound_rates.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    before = base.compute_raw_ca_ratio(raw) if not raw.empty else pd.Series(dtype=float)
    after = base.compute_raw_ca_ratio(cleaned) if not cleaned.empty else pd.Series(dtype=float)
    plt.hist(pd.to_numeric(before, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(), bins=80, alpha=0.45, label="\u6e05\u6d17\u524d")
    plt.hist(pd.to_numeric(after, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(), bins=80, alpha=0.55, label="\u6e05\u6d17\u540e")
    plt.legend()
    plt.title("C\u7ebf future \u6e05\u6d17\u524d\u540e\u94d9\u5355\u8017\u5206\u5e03\u5bf9\u6bd4")
    plt.tight_layout()
    plt.savefig(figure_dir / "c_line_future_cleaned_vs_raw_ca_consumption.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    if not drift.empty:
        drift.sort_values("psi_like_drift_score", ascending=False).head(10).plot(
            x="feature", y="psi_like_drift_score", kind="bar", ax=plt.gca(), legend=False
        )
    plt.title("C\u7ebf future \u4e0e C\u7ebf\u5386\u53f2\u5de5\u51b5\u7279\u5f81\u5206\u5e03\u5bf9\u6bd4")
    plt.tight_layout()
    plt.savefig(figure_dir / "c_line_future_vs_historical_feature_drift.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    if not risk_table.empty:
        one = risk_table[risk_table["strategy"].eq("one_t90_to_nearest_prediction")]
        if not one.empty:
            labels = one["interval_position"].tolist()
            x = np.arange(len(labels))
            plt.bar(x - 0.2, one["high_rate"], width=0.4, label="\u9ad8 T90 \u7387")
            plt.bar(x + 0.2, one["out_spec_rate"], width=0.4, label="\u51fa\u89c4\u7387")
            plt.xticks(x, labels, rotation=20)
            plt.legend()
    plt.title("C\u7ebf future \u5364\u5316\u6a61\u80f6 T90 \u56de\u586b\u4e0b\u533a\u95f4\u5185\u5916\u98ce\u9669\u5bf9\u6bd4")
    plt.tight_layout()
    plt.savefig(figure_dir / "c_line_future_t90_backfill_risk_summary.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    if not monthly.empty:
        x = np.arange(len(monthly))
        plt.bar(x - 0.2, monthly["inside_high_rate"].fillna(0), width=0.4, label="inside high")
        plt.bar(x + 0.2, monthly["above_high_rate"].fillna(0), width=0.4, label="above high")
        plt.xticks(x, monthly["month"], rotation=20)
        plt.legend()
    plt.title("C\u7ebf future \u5206\u6708\u98ce\u9669\u5206\u79bb\u7a33\u5b9a\u6027")
    plt.tight_layout()
    plt.savefig(figure_dir / "c_line_future_monthly_risk_stability.png", dpi=150)
    plt.close()


def append_method_doc(path: Path, summary: dict[str, Any]) -> None:
    section_title = "## C\u7ebf\u4e13\u7528\u5305\u4e0e\u65e7 C/D/E \u5408\u5e76\u7ebf\u8bc1\u636e\u4fee\u6b63"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    body = f"""

{section_title}

\u65e7\u7684 `deploy/ca_safe_band_mvp` \u57fa\u4e8e C/D/E \u5408\u5e76\u7ebf\u903b\u8f91\uff0c\u56e0\u6b64\u4e0d\u518d\u4f5c\u4e3a C \u7ebf\u90e8\u7f72\u8bc1\u636e\u3002C \u7ebf\u90e8\u7f72\u590d\u9a8c\u5fc5\u987b\u4f7f\u7528 `deploy/ca_safe_band_mvp_c_line` \u548c `models/ca_safe_band_mvp_c_line/safe_band_artifact.json`\u3002\u65e7 future \u56de\u653e\u548c Go/No-Go \u8f93\u51fa\u5df2\u6807\u8bb0\u4e3a\u4e0d\u9002\u7528\u4e8e C \u7ebf\u90e8\u7f72\u51b3\u7b56\uff0c\u4ec5\u53ef\u4f5c\u5386\u53f2\u6216\u65b9\u6cd5\u53c2\u8003\u3002

C \u7ebf\u5305\u7684\u6700\u7ec8\u7b56\u7565\u4e3a `top_rule_only`\u3002future \u6570\u636e\u4ec5\u4f5c holdout \u9a8c\u8bc1\uff0c\u4e0d\u66f4\u65b0 artifact\u3001q33/q66 \u8fb9\u754c\u3001\u89c4\u5219\u6216\u5b89\u5168\u5e26\u533a\u95f4\u3002\u539f\u59cb DCS \u70b9\u4f4d\u5728\u6784\u9020 60min \u7a97\u53e3\u7279\u5f81\u524d\uff0c\u5148\u7528 `data/\u526f\u672c\u5364\u5316\u5de5\u6bb5\u6570\u636e\u70b9\u4f4d.xlsx` \u7684\u6b63\u5e38\u4e0a\u4e0b\u9650\u6e05\u6d17\uff1a\u8d8a\u754c\u503c\u8bbe\u4e3a\u7f3a\u5931\uff0c\u4e0d\u526a\u88c1\u3001\u4e0d\u63d2\u503c\u3002

future T90 \u53ea\u4fdd\u7559\u80f6\u79cd\u4e3a\u5364\u5316\u6a61\u80f6\u4e14\u7ebf\u522b\u4e3a C \u7ebf\u7684\u8bb0\u5f55\u3002T90 \u56de\u586b\u9a8c\u8bc1\u4f7f\u7528\u63a8\u8350\u65f6\u523b +174min \u7684\u4f4f\u7559\u65f6\u95f4\u5bf9\u9f50\uff0c\u5176\u4e2d one-T90-one-prediction \u662f\u66f4\u4e25\u683c\u7684\u53ef\u4fe1\u5ea6\u9a8c\u8bc1\u3002\u5f53\u524d C \u7ebf\u5305\u4ecd\u662f monitor-only \u5019\u9009\uff0c\u9700\u4eba\u5de5\u590d\u6838\uff1b\u672c\u9636\u6bb5\u4e0d\u5b9e\u65bd\u81ea\u52a8\u63a7\u5236\uff0c\u4e0d\u5199\u56de DCS\u3002
"""
    if section_title in text:
        text = text.rstrip() + "\n" + body
    else:
        text = text.rstrip() + "\n" + body
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_experiment_doc(path: Path, summary: dict[str, Any]) -> str:
    title = "C\u7ebf\u4e13\u7528 future holdout \u6e05\u6d17\u590d\u9a8c\u4e0e\u65e7\u5408\u5e76\u7ebf\u8bc1\u636e\u4fee\u6b63"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    heading_re = re.compile(r"^##\s*(\d+)\s*\.?\s*", re.MULTILINE)
    existing_numbers = [int(m.group(1)) for m in heading_re.finditer(text)]
    number = 45 if 45 not in existing_numbers else max(existing_numbers + [45]) + 1
    heading = f"## {number}. {title}"
    body = f"""

{heading}

- \u4fee\u6b63\u539f\u56e0\uff1a\u65e7 V1 monitor-only \u56de\u653e\u4f7f\u7528\u4e86 C/D/E \u5408\u5e76\u7ebf\u5305\uff0c\u4e0d\u80fd\u4f5c\u4e3a C \u7ebf\u90e8\u7f72\u8bc1\u636e\u3002
- C \u7ebf\u5305\uff1a`deploy/ca_safe_band_mvp_c_line`\uff1bC \u7ebf artifact\uff1a`models/ca_safe_band_mvp_c_line/safe_band_artifact.json`\u3002
- \u65e7\u5408\u5e76\u7ebf\u8bc1\u636e\uff1a\u5df2\u6807\u8bb0\u4e3a superseded for C-line deployment evidence\uff0c\u672a\u5220\u9664\u3002
- future \u6570\u636e\u8def\u5f84\uff1a`{summary.get('future_dir')}`\u3002
- \u70b9\u4f4d\u4e0a\u4e0b\u9650\u6765\u6e90\uff1a`{summary.get('point_config_path')}`\uff1b\u6e05\u6d17\u89c4\u5219\u4e3a\u8d8a\u754c\u503c\u8bbe\u4e3a\u7f3a\u5931\uff0c\u4e0d\u526a\u88c1\u3001\u4e0d\u63d2\u503c\u3002
- T90 \u6587\u4ef6\uff1a2026.1.xlsx\u30012026.2.xlsx\u30012026.3C.xlsx\uff1b\u8fc7\u6ee4\u6761\u4ef6\uff1a\u80f6\u79cd = \u5364\u5316\u6a61\u80f6\uff0c\u7ebf\u522b = C\u3002
- DCS \u6e05\u6d17\uff1abounds_applied={summary.get('dcs_cleaning_summary', {}).get('bounds_applied_count')}\uff0cpossible_shutdown_timestamp_count={summary.get('dcs_cleaning_summary', {}).get('possible_shutdown_timestamp_count')}\u3002
- \u5f02\u5e38\u94d9\u5355\u8017\uff1a\u6e05\u6d17\u524d min={summary.get('dcs_cleaning_summary', {}).get('before_cleaning_ca_consumption_min')}\uff0c\u6e05\u6d17\u540e min={summary.get('dcs_cleaning_summary', {}).get('after_cleaning_ca_consumption_min')}\u3002
- C \u7ebf\u63a8\u8350\u56de\u653e\uff1acoverage={summary.get('recommendation_distribution_summary', {}).get('recommendation_coverage')}\uff0cinside={summary.get('recommendation_distribution_summary', {}).get('inside_band_count')}\uff0cabove={summary.get('recommendation_distribution_summary', {}).get('above_band_count')}\uff0cbelow={summary.get('recommendation_distribution_summary', {}).get('below_band_count')}\u3002
- one-to-one T90 \u56de\u586b\uff1aaligned={summary.get('one_to_one_backfill_summary', {}).get('aligned_sample_count')}\uff0crisk_guardrail_pass={summary.get('one_to_one_backfill_summary', {}).get('risk_guardrail_pass')}\u3002
- clear-label \u4e0d\u786e\u5b9a\uff1auncertain_boundary_rate={summary.get('one_to_one_backfill_summary', {}).get('uncertain_boundary_rate')}\u3002
- C \u7ebf\u5386\u53f2\u5bf9\u6bd4\uff1afuture_within_c_line_historical_support={summary.get('future_vs_c_line_historical_drift_summary', {}).get('future_within_c_line_historical_support')}\u3002
- \u5206\u6708\u7a33\u5b9a\u6027\uff1amonthly_risk_separation_stable={summary.get('monthly_stability_summary', {}).get('monthly_risk_separation_stable')}\u3002
- validation_mode\uff1a{summary.get('validation_mode')}\u3002
- recommended_next_step\uff1a{summary.get('recommended_next_step')}\u3002
- \u9650\u5236\uff1aC \u7ebf rebuild \u9636\u6bb5\u4ecd\u4e3a stop_until_more_data\uff1b\u5f53\u524d\u4ec5\u662f monitor-only candidate\uff1b\u9700\u4eba\u5de5\u590d\u6838\uff1bT90 \u6d4b\u91cf\u8bef\u5dee\u7ea6 0.1\uff1bfuture \u70b9\u4f4d\u6620\u5c04\u4f9d\u8d56\u6587\u4ef6\u547d\u540d/\u683c\u5f0f\uff1b\u4e0d\u5b9e\u65bd\u81ea\u52a8\u63a7\u5236\uff0c\u4e0d\u5199\u56de DCS\u3002
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n" + body.rstrip() + "\n", encoding="utf-8")
    return heading


def decide_final(
    deploy_dir: Path,
    point_report: dict[str, Any],
    cleaning_report: dict[str, Any],
    t90_report: dict[str, Any],
    rec_report: dict[str, Any],
    backfill_report: dict[str, Any],
    drift_report: dict[str, Any],
    monthly_report: dict[str, Any],
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    one = backfill_report.get("one_t90_to_nearest_prediction", {})
    flags = {
        "old_merged_line_results_not_valid_for_c_line_deployment": True,
        "c_line_package_used": bool(deploy_dir.exists() and deploy_dir.name == C_LINE_DEPLOY_NAME),
        "old_merged_package_not_used": bool(not rec_report.get("old_merged_package_used", True)),
        "point_bounds_loaded": bool(point_report.get("point_bounds_parse_pass")),
        "dcs_cleaning_pass": bool(cleaning_report.get("cleaning_pass")),
        "impossible_ca_removed": bool(rec_report.get("impossible_current_ca_consumption_count") == 0),
        "future_t90_halogen_filter_pass": bool(t90_report.get("future_t90_filter_pass")),
        "future_t90_c_line_filter_pass": bool(t90_report.get("future_t90_c_line_filter_pass")),
        "one_to_one_backfill_confirms_risk_separation": bool(one.get("risk_guardrail_pass")),
        "monthly_risk_separation_stable": bool(monthly_report.get("monthly_risk_separation_stable")),
        "future_within_c_line_historical_support": bool(drift_report.get("future_within_c_line_historical_support")),
        "c_line_monitor_only_candidate_for_human_review": False,
        "c_line_factory_test_ready_after_human_review": False,
    }
    if not flags["c_line_package_used"]:
        return "failed_c_line_package_missing", "stop_due_to_runtime_failure", flags, {"monitor_only": True, "automatic_control": False, "dcs_writeback": False}
    if not flags["future_t90_c_line_filter_pass"]:
        return "failed_c_line_t90_filtering", "fix_c_line_t90_filtering", flags, {"monitor_only": True, "automatic_control": False, "dcs_writeback": False}
    if not rec_report.get("future_replay_pass"):
        return "failed_runtime_scoring", "stop_due_to_runtime_failure", flags, {"monitor_only": True, "automatic_control": False, "dcs_writeback": False}
    mode = "c_line_cleaned_runtime_plus_t90_backfill" if t90_report.get("future_t90_available") else "c_line_cleaned_runtime_only_no_t90"
    if flags["one_to_one_backfill_confirms_risk_separation"] and flags["impossible_ca_removed"]:
        flags["c_line_monitor_only_candidate_for_human_review"] = True
        next_step = "human_review_c_line_monitor_only_candidate"
    elif not flags["future_t90_halogen_filter_pass"]:
        next_step = "fix_c_line_t90_filtering"
    elif not flags["future_within_c_line_historical_support"]:
        next_step = "investigate_c_line_future_distribution_shift"
    else:
        next_step = "keep_c_line_result_as_preliminary_only"
    safety = {
        "monitor_only": True,
        "automatic_control": False,
        "dcs_writeback": False,
        "no_operational_increase_hint": True,
        "old_merged_package_used": False,
    }
    return mode, next_step, flags, safety


def main() -> int:
    args = parse_args()
    base.configure_chinese_font()
    ensure_output_dirs(args.output_dir, args.table_dir, args.figure_dir)

    if not args.deploy_dir.exists():
        report = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "validation_mode": "failed_c_line_package_missing",
            "deploy_dir": str(args.deploy_dir),
            "recommended_next_step": "stop_due_to_runtime_failure",
        }
        write_json(args.output_dir / "c_line_future_holdout_v1_cleaned_validation_report.json", report)
        raise FileNotFoundError(f"C-line deploy package not found: {args.deploy_dir}")
    if args.deploy_dir.name != C_LINE_DEPLOY_NAME:
        raise RuntimeError(f"Refusing non-C-line deploy package: {args.deploy_dir}")

    supersession_df, supersession_report = create_supersession_outputs(args.output_dir, args.table_dir)

    bounds, point_report = base.parse_point_bounds(args.point_config, args.output_dir, args.table_dir)
    write_c_line_bound_table(bounds, point_report, args.table_dir)

    raw_before, raw_report, raw_point_quality = base.parse_future_raw_before_cleaning(args.future_dir, args.output_dir)
    cleaned, out_bounds, cleaning_summary, cleaning_report = base.clean_raw_with_bounds(raw_before, bounds, args.output_dir, args.table_dir)
    # C-line named table alias.
    cleaning_summary_path = args.output_dir / "future_dcs_cleaning_summary.csv"
    if cleaning_summary_path.exists():
        pd.read_csv(cleaning_summary_path).to_csv(args.table_dir / "c_line_future_dcs_cleaning_summary.csv", index=False, encoding="utf-8-sig")

    t90_files = resolve_t90_files(args.future_dir, args.future_t90_files)
    t90_c_line, t90_report = parse_future_t90_c_line(t90_files, args.target_rubber_type, args.target_line, args.output_dir, args.table_dir)

    runtime_features, feature_quality_by_time, feature_report = base.build_runtime_features(cleaned, args.eval_frequency_minutes, args.min_valid_points, args.output_dir)
    replay, rec_report = score_c_line_future(runtime_features, args.deploy_dir, args.output_dir, args.table_dir)
    many, one, backfill_report, risk_table, clear_table = align_c_line_t90_strategies(
        replay,
        t90_c_line,
        args.residence_minutes,
        args.t90_match_tolerance_minutes,
        args.output_dir,
        args.table_dir,
    )

    hist_path, hist_warnings = resolve_c_line_historical_reference(args.historical_reference, args.c_line_revalidation_dir)
    drift, drift_report = compare_c_line_historical(runtime_features, hist_path, args.output_dir, args.table_dir)
    if hist_warnings:
        drift_report["warnings"] = hist_warnings
        write_json(args.output_dir / "future_vs_c_line_historical_feature_drift_report.json", drift_report)

    monthly, monthly_report = monthly_c_line_summary(replay, cleaned, out_bounds, t90_c_line, one, drift, args.output_dir, args.table_dir)
    readiness_table, readiness_report = build_readiness_update(
        args.deploy_dir,
        point_report,
        cleaning_report,
        t90_report,
        rec_report,
        backfill_report,
        clear_table,
        monthly_report,
        drift_report,
        args.output_dir,
        args.table_dir,
    )
    plot_outputs(raw_before, cleaned, replay, out_bounds, drift, risk_table, monthly, args.figure_dir)

    validation_mode, recommended_next_step, flags, safety = decide_final(
        args.deploy_dir,
        point_report,
        cleaning_report,
        t90_report,
        rec_report,
        backfill_report,
        drift_report,
        monthly_report,
    )
    c_line_artifact_path = Path("models/ca_safe_band_mvp_c_line/safe_band_artifact.json")
    final_report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "correction_reason": "Previous V1 monitor-only package used old C/D/E merged-line evidence; this run uses the C-line-only package.",
        "old_merged_line_evidence_superseded": supersession_report,
        "future_dir": str(args.future_dir),
        "point_config_path": str(args.point_config),
        "future_t90_files": [str(p) for p in t90_files],
        "target_rubber_type": args.target_rubber_type,
        "target_line": args.target_line,
        "deploy_dir": str(args.deploy_dir),
        "c_line_artifact_path": str(c_line_artifact_path),
        "c_line_revalidation_dir": str(args.c_line_revalidation_dir),
        "historical_reference_path": str(hist_path) if hist_path else None,
        "output_dir": str(args.output_dir),
        "point_bounds_summary": point_report,
        "raw_quality_before_cleaning_summary": raw_report,
        "dcs_cleaning_summary": cleaning_report,
        "t90_parse_summary": t90_report,
        "feature_quality_summary": feature_report,
        "recommendation_distribution_summary": rec_report,
        "t90_backfill_validation_summary": backfill_report,
        "one_to_one_backfill_summary": backfill_report.get("one_t90_to_nearest_prediction", {}),
        "clear_label_validation_summary": clear_table.to_dict(orient="records") if not clear_table.empty else [],
        "future_vs_c_line_historical_drift_summary": drift_report,
        "monthly_stability_summary": monthly_report,
        "c_line_factory_readiness_update": readiness_report,
        "safety_check_summary": safety,
        "validation_mode": validation_mode,
        "holdout_principle": {
            "future_data_updates_artifact": False,
            "future_data_updates_rules": False,
            "future_data_updates_q33_q66_boundaries": False,
            "future_data_updates_safe_band_interval": False,
            "future_data_is_holdout_only": True,
        },
        "final_decision_flags": flags,
        "limitations": [
            "C-line rebuild readiness was stop_until_more_data and remains subject to human review.",
            "Only 9 monitor-chain candidates were available in the C-line rebuild.",
            "T90 measurement uncertainty is approximately 0.1.",
            "Future raw point parsing depends on file naming and text format.",
            "This is monitor-only evidence, not automatic control.",
        ],
        "recommended_next_step": recommended_next_step,
    }
    write_json(args.output_dir / "c_line_future_holdout_v1_cleaned_validation_report.json", final_report)
    append_method_doc(args.method_doc, final_report)
    heading = append_experiment_doc(args.doc, final_report)
    final_report["experiment_doc_section_appended"] = heading
    write_json(args.output_dir / "c_line_future_holdout_v1_cleaned_validation_report.json", final_report)

    print("repaired_for_c_line=true")
    print(f"old_merged_line_evidence_superseded={True}")
    print(f"package_path_used={args.deploy_dir}")
    print(f"old_merged_package_used={rec_report.get('old_merged_package_used')}")
    print(f"bounds_found_count={point_report.get('bounds_found_count')}")
    print(f"cleaning_pass={cleaning_report.get('cleaning_pass')}")
    print(f"halogen_c_line_t90_rows={t90_report.get('halogen_c_line_t90_rows_after_time_and_t90_cleaning')}")
    print(f"recommendation_coverage={rec_report.get('recommendation_coverage')}")
    print(f"one_to_one_risk_guardrail_pass={backfill_report.get('one_t90_to_nearest_prediction', {}).get('risk_guardrail_pass')}")
    print(f"validation_mode={validation_mode}")
    print(f"recommended_next_step={recommended_next_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
