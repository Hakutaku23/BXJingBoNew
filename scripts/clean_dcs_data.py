from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_t90_file(data_dir: Path) -> Path:
    candidates = [path for path in data_dir.glob("*.xlsx") if not path.name.startswith("~$") and path.name.startswith("t90-")]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one t90 workbook, found {len(candidates)}: {candidates}")
    return candidates[0]


def longest_true_run(mask: pd.Series) -> int:
    best = 0
    current = 0
    for value in mask.fillna(False).to_numpy():
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def detect_isolated_spikes(series: pd.Series, window: int, sigma: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    rolling_median = values.rolling(window=window, center=True, min_periods=max(5, window // 2)).median()
    absolute_deviation = (values - rolling_median).abs()
    rolling_mad = absolute_deviation.rolling(window=window, center=True, min_periods=max(5, window // 2)).median()
    robust_sigma = 1.4826 * rolling_mad

    diff_q999 = values.diff().abs().quantile(0.999)
    value_range = values.quantile(0.99) - values.quantile(0.01)
    absolute_floor = max(float(diff_q999 * 3) if pd.notna(diff_q999) else 0.0, float(value_range * 0.05))

    threshold = np.maximum(sigma * robust_sigma, absolute_floor)
    spike = values.notna() & rolling_median.notna() & (robust_sigma > 0) & (absolute_deviation > threshold)

    previous_close = (values.shift(1) - rolling_median).abs() <= threshold
    next_close = (values.shift(-1) - rolling_median).abs() <= threshold
    return spike & previous_close.fillna(False) & next_close.fillna(False)


def clean_dcs_frame(
    frame: pd.DataFrame,
    interpolate_limit: int,
    spike_window: int,
    spike_sigma: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, int | float]]]:
    cleaned = frame.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True).copy()
    feature_cols = [column for column in cleaned.columns if column != "time"]
    report: dict[str, dict[str, int | float]] = {}

    for column in feature_cols:
        original = pd.to_numeric(cleaned[column], errors="coerce")
        missing_before = int(original.isna().sum())

        spike_mask = detect_isolated_spikes(original, window=spike_window, sigma=spike_sigma)
        cleaned[f"{column}_was_missing"] = original.isna()
        cleaned[f"{column}_spike_flag"] = spike_mask.fillna(False)

        work = original.mask(spike_mask)
        missing_after_spike = int(work.isna().sum())

        interpolated = work.interpolate(method="linear", limit=interpolate_limit, limit_area="inside")
        filled_by_interpolation = int(work.isna().sum() - interpolated.isna().sum())
        cleaned[column] = interpolated

        report[column] = {
            "missing_before_cleaning": missing_before,
            "missing_rate_before_cleaning": float(missing_before / len(cleaned)) if len(cleaned) else 0.0,
            "isolated_spikes_to_nan": int(spike_mask.sum()),
            "missing_after_spike_filter": missing_after_spike,
            "filled_by_short_gap_interpolation": filled_by_interpolation,
            "missing_after_cleaning": int(cleaned[column].isna().sum()),
            "missing_rate_after_cleaning": float(cleaned[column].isna().mean()) if len(cleaned) else 0.0,
            "longest_missing_run_after_cleaning": longest_true_run(cleaned[column].isna()),
        }

    return cleaned, report


def load_lims_t90_labels(t90_file: Path, start_time: pd.Timestamp, end_time: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, object]]:
    workbook = pd.ExcelFile(t90_file)
    sheet_frames: list[pd.DataFrame] = []
    sheet_report: dict[str, dict[str, int | float]] = {}

    for sheet_name in workbook.sheet_names:
        raw = pd.read_excel(t90_file, sheet_name=sheet_name)
        time_col = raw.columns[1]
        t90_col = raw.columns[3]
        labels = pd.DataFrame(
            {
                "time": pd.to_datetime(raw[time_col], errors="coerce").dt.round("min"),
                f"t90_{sheet_name}": pd.to_numeric(raw[t90_col], errors="coerce"),
            }
        ).dropna()
        labels = labels[(labels["time"] >= start_time) & (labels["time"] <= end_time)]
        labels = labels.groupby("time", as_index=False).mean(numeric_only=True)
        sheet_frames.append(labels)
        sheet_report[sheet_name] = {
            "labels_in_range": int(len(labels)),
            "t90_mean": float(labels[f"t90_{sheet_name}"].mean()) if len(labels) else None,
            "t90_min": float(labels[f"t90_{sheet_name}"].min()) if len(labels) else None,
            "t90_max": float(labels[f"t90_{sheet_name}"].max()) if len(labels) else None,
        }

    merged = sheet_frames[0] if sheet_frames else pd.DataFrame(columns=["time"])
    for labels in sheet_frames[1:]:
        merged = merged.merge(labels, on="time", how="outer")

    t90_cols = [column for column in merged.columns if column.startswith("t90_")]
    if t90_cols:
        merged["t90"] = merged[t90_cols].mean(axis=1, skipna=True)
        merged["t90_label_count"] = merged[t90_cols].notna().sum(axis=1)
    else:
        merged["t90"] = np.nan
        merged["t90_label_count"] = 0

    merged = merged.sort_values("time").reset_index(drop=True)
    report = {
        "t90_file": str(t90_file),
        "sheet_report": sheet_report,
        "combined_label_rows": int(merged["t90"].notna().sum()),
        "combined_label_unique_times": int(merged["time"].nunique()),
    }
    return merged, report


def attach_labels(cleaned: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    return cleaned.merge(labels, on="time", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean DCS data and attach sparse LIMS t90 labels.")
    parser.add_argument("--input", default=Path("data/data_new.parquet"), type=Path)
    parser.add_argument("--output", default=Path("data/data_clean.parquet"), type=Path)
    parser.add_argument("--report", default=Path("data/data_clean_report.json"), type=Path)
    parser.add_argument("--data-dir", default=Path("data"), type=Path)
    parser.add_argument("--interpolate-limit", default=5, type=int)
    parser.add_argument("--spike-window", default=11, type=int)
    parser.add_argument("--spike-sigma", default=8.0, type=float)
    args = parser.parse_args()

    source = pd.read_parquet(args.input)
    source["time"] = pd.to_datetime(source["time"], errors="coerce")
    source = source.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    cleaned, variable_report = clean_dcs_frame(
        source,
        interpolate_limit=args.interpolate_limit,
        spike_window=args.spike_window,
        spike_sigma=args.spike_sigma,
    )

    t90_file = find_t90_file(args.data_dir)
    labels, label_report = load_lims_t90_labels(t90_file, cleaned["time"].min(), cleaned["time"].max())
    cleaned_with_labels = attach_labels(cleaned, labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned_with_labels.to_parquet(args.output, index=False)

    feature_cols = [column for column in source.columns if column != "time"]
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": int(len(cleaned_with_labels)),
        "time_min": cleaned_with_labels["time"].min().isoformat() if len(cleaned_with_labels) else None,
        "time_max": cleaned_with_labels["time"].max().isoformat() if len(cleaned_with_labels) else None,
        "cleaning_methods": {
            "isolated_spike_filter": {
                "method": "centered rolling median and MAD",
                "window": args.spike_window,
                "sigma": args.spike_sigma,
                "note": "Only isolated points are flagged; step changes are retained.",
            },
            "short_gap_interpolation": {
                "method": "linear interpolation",
                "limit_consecutive_rows": args.interpolate_limit,
                "note": "Only internal short gaps are filled; leading, trailing, and long gaps stay NaN.",
            },
            "missing_indicators": "For each DCS variable, *_was_missing and *_spike_flag columns are added.",
            "lims_labels": "Sparse t90 labels are attached at exact rounded-minute sample times.",
        },
        "feature_columns": feature_cols,
        "variable_report": variable_report,
        "label_report": label_report,
        "label_non_null_counts": {
            column: int(cleaned_with_labels[column].notna().sum())
            for column in cleaned_with_labels.columns
            if column.startswith("t90")
        },
        "quality_checks": {
            "duplicate_time_rows": int(cleaned_with_labels["time"].duplicated().sum()),
            "rows_with_any_feature_value": int(cleaned_with_labels[feature_cols].notna().any(axis=1).sum()),
            "rows_with_all_feature_values": int(cleaned_with_labels[feature_cols].notna().all(axis=1).sum()),
            "total_dcs_missing_before": int(source[feature_cols].isna().sum().sum()),
            "total_dcs_missing_after": int(cleaned_with_labels[feature_cols].isna().sum().sum()),
            "total_isolated_spikes_flagged": int(
                sum(item["isolated_spikes_to_nan"] for item in variable_report.values())
            ),
            "total_short_gaps_filled": int(
                sum(item["filled_by_short_gap_interpolation"] for item in variable_report.values())
            ),
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
