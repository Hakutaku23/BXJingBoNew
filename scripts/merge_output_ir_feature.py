from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


IR_COLUMN = "output_ir_corrected"
DERIVED_IR_FEATURES = [
    "output_ir_corrected_lag_0",
    "output_ir_corrected_win_5_mean",
    "output_ir_corrected_win_15_mean",
    "output_ir_corrected_win_30_mean",
    "output_ir_corrected_win_15_std",
    "output_ir_corrected_win_30_std",
    "output_ir_corrected_win_15_slope",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge product-outlet corrected IR value into cleaned DCS data.")
    parser.add_argument("--process-input", type=Path, default=Path("data/data_clean.parquet"))
    parser.add_argument("--ir-input", type=Path, default=Path("output.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/data_clean_with_ir.parquet"))
    parser.add_argument("--report", type=Path, default=Path("data/data_clean_with_ir_report.json"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    return parser.parse_args()


def as_jsonable(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): as_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [as_jsonable(item) for item in value]
    return value


def resolve_ir_input(path: Path, warnings: list[str]) -> Path:
    if path.exists():
        return path
    fallback = Path("data") / path.name
    if path.name.lower() == "output.csv" and fallback.exists():
        warnings.append(f"IR input {path} does not exist; using fallback {fallback}.")
        return fallback
    raise FileNotFoundError(f"IR input CSV does not exist: {path}")


def read_header(path: Path, encoding: str) -> list[str]:
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def read_ir_csv(path: Path, warnings: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    last_error: Exception | None = None
    for encoding in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            header = read_header(path, encoding)
            if len(header) < 2:
                raise ValueError("output.csv must contain at least two columns.")
            timestamp_col = header[0]
            value_col = header[-1]
            usecols = [0, len(header) - 1]
            raw = pd.read_csv(path, encoding=encoding, usecols=usecols, low_memory=False)
            raw = raw.rename(columns={raw.columns[0]: "ir_timestamp_raw", raw.columns[-1]: IR_COLUMN})
            timestamps = pd.to_datetime(raw["ir_timestamp_raw"], errors="coerce")
            values = pd.to_numeric(raw[IR_COLUMN], errors="coerce")
            timestamp_success = float(timestamps.notna().mean()) if len(raw) else 0.0
            value_success = float(values.notna().mean()) if len(raw) else 0.0
            if timestamp_success < 0.80:
                raise ValueError(f"Timestamp parse success rate {timestamp_success:.3f} is below 0.80.")
            if value_success < 0.80:
                raise ValueError(f"IR numeric conversion success rate {value_success:.3f} is below 0.80.")
            ir = pd.DataFrame({"time": timestamps.dt.floor("min"), IR_COLUMN: values})
            ir = ir[ir["time"].notna()].copy()
            ir = ir.groupby("time", as_index=False)[IR_COLUMN].mean()
            meta = {
                "csv_encoding_used": encoding,
                "detected_ir_timestamp_column": str(timestamp_col),
                "detected_ir_value_column": str(value_col),
                "output_csv_total_columns": int(len(header)),
                "ignored_middle_column_count": int(max(0, len(header) - 2)),
                "ignored_middle_column_names_sample": [str(item) for item in header[1:-1][:20]],
                "raw_ir_row_count": int(len(raw)),
                "timestamp_parse_success_rate": timestamp_success,
                "ir_numeric_conversion_success_rate": value_success,
            }
            return ir.sort_values("time").reset_index(drop=True), meta
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            if encoding == "gbk":
                break
            continue
    raise ValueError(f"Failed to read output.csv using utf-8-sig, utf-8, or gbk: {last_error}")


def add_derived_ir_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("time").reset_index(drop=True).copy()
    series = pd.to_numeric(frame[IR_COLUMN], errors="coerce")
    frame["output_ir_corrected_lag_0"] = series
    for window in [5, 15, 30]:
        rolling = series.rolling(window=window, min_periods=max(2, window // 3))
        frame[f"output_ir_corrected_win_{window}_mean"] = rolling.mean()
        if window in [15, 30]:
            frame[f"output_ir_corrected_win_{window}_std"] = rolling.std()
    frame["output_ir_corrected_win_15_slope"] = (series - series.shift(14)) / 14.0
    return frame


def missing_summary(frame: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    rates = frame[columns].isna().mean().sort_values(ascending=False)
    return {
        "missing_rate_min": float(rates.min()),
        "missing_rate_median": float(rates.median()),
        "missing_rate_mean": float(rates.mean()),
        "missing_rate_max": float(rates.max()),
        "per_feature": {str(k): float(v) for k, v in rates.items()},
    }


def range_dict(series: pd.Series) -> dict[str, str | None]:
    return {
        "min": series.min().isoformat() if len(series) and pd.notna(series.min()) else None,
        "max": series.max().isoformat() if len(series) and pd.notna(series.max()) else None,
    }


def append_doc(doc_path: Path, report: dict[str, object]) -> bool:
    # The full interpretation section is appended by evaluate_output_ir_proxy.py.
    # This merge script only records machine-readable outputs to avoid duplicate narrative sections.
    return False


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    if not args.process_input.exists():
        raise FileNotFoundError(f"Process input parquet does not exist: {args.process_input}")
    process = pd.read_parquet(args.process_input)
    if "time" not in process.columns:
        raise ValueError("Process input must contain a time column.")
    process = process.copy()
    process["time"] = pd.to_datetime(process["time"], errors="coerce")
    if process["time"].isna().any():
        raise ValueError("Process input contains invalid time values.")
    process["time"] = process["time"].dt.floor("min")
    process = process.sort_values("time").reset_index(drop=True)

    ir_path = resolve_ir_input(args.ir_input, warnings)
    ir, meta = read_ir_csv(ir_path, warnings)
    merged = process.merge(ir, on="time", how="left", validate="many_to_one")
    merged = add_derived_ir_features(merged)
    merged = merged.sort_values("time").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)

    overlap_min = max(process["time"].min(), ir["time"].min()) if len(ir) else pd.NaT
    overlap_max = min(process["time"].max(), ir["time"].max()) if len(ir) else pd.NaT
    report = {
        "process_input_path": str(args.process_input),
        "ir_input_path": str(ir_path),
        "requested_ir_input_path": str(args.ir_input),
        "output_path": str(args.output),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **meta,
        "strict_column_rule_applied": True,
        "process_row_count": int(len(process)),
        "ir_row_count": int(len(ir)),
        "merged_row_count": int(len(merged)),
        "process_time_range": range_dict(process["time"]),
        "ir_time_range": range_dict(ir["time"]),
        "overlap_time_range": {
            "min": overlap_min.isoformat() if pd.notna(overlap_min) else None,
            "max": overlap_max.isoformat() if pd.notna(overlap_max) else None,
        },
        "output_ir_non_null_count": int(merged[IR_COLUMN].notna().sum()),
        "output_ir_non_null_rate": float(merged[IR_COLUMN].notna().mean()),
        "derived_ir_features": DERIVED_IR_FEATURES,
        "missing_rate_summary": missing_summary(merged, [IR_COLUMN] + DERIVED_IR_FEATURES),
        "warnings": warnings,
        "assumptions": [
            "Only the first output.csv column is used as timestamp.",
            "Only the last output.csv column is used as corrected outlet IR value.",
            "All middle output.csv columns are ignored and never merged.",
            "IR values are not imputed or forward-filled before trailing window features.",
            "Trailing IR windows end at the current timestamp and do not use future values.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(as_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    doc_appended = append_doc(args.doc, report)
    print("Output IR merge complete.")
    print(f"  first column used as timestamp: {meta['detected_ir_timestamp_column']}")
    print(f"  last column used as corrected IR: {meta['detected_ir_value_column']}")
    print(f"  ignored middle column count: {meta['ignored_middle_column_count']}")
    print(f"  IR non-null coverage: {report['output_ir_non_null_rate']}")
    print(f"  output: {args.output}")
    print(f"  report: {args.report}")
    print(f"  docs appended: {doc_appended}")


if __name__ == "__main__":
    main()
