from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


T90_LOW = 8.20
T90_HIGH = 8.70
CALCIUM_COLUMN = "硬脂酸钙加注量"
RUBBER_FLOW_COLUMN = "卤化工段胶液总量2"
MINUTE_NS = 60 * 1_000_000_000

LABEL_COLUMNS = {
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

LEAKAGE_EXCLUDED_COLUMNS = [
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
]

CALCIUM_CORE_FEATURES = [
    "ca_lag_165",
    "ca_win_15_mean",
    "ca_win_30_mean",
    "ca_win_60_mean",
    "ca_win_120_mean",
    "ca_win_60_std",
    "ca_win_60_min",
    "ca_win_60_max",
    "ca_win_60_range",
    "ca_win_60_slope",
    "ca_delta_15",
    "ca_delta_30",
    "ca_delta_60",
    "ca_missing_rate_60",
    "ca_per_rubber_flow_lag_165",
    "ca_per_rubber_flow_win_60_mean",
    "ca_per_rubber_flow_win_60_slope",
]

RECOMMENDED_USAGE = {
    "dose_response": "Use calcium_core_features only, especially ca_per_rubber_flow and ca_win_* features.",
    "control_model": "Use calcium_core_features plus process_context_features, but perform train-only feature selection before modeling.",
    "jitl_search": "Use a smaller selected subset fitted on training data only.",
    "do_not": "Do not blindly use all 237 candidate features for interpretation or policy decisions.",
}


PROCESS_LAG_PRIORS = {
    "硬脂酸钙加注量": 165,
    "ESBO加注量": 165,
    "中和碱液添加量": 165,
    "R513温度": 165,
    "R514温度": 164,
    "R512A温度": 166,
    "R511A温度": 173,
    "R510A温度": 174,
    "卤化工段胶液总量2": 174,
    "反应溴添加量": 174,
    "储罐胶浓在线检测": 174,
}

SAFE_NAMES = {
    "硬脂酸钙加注量": "ca",
    "ESBO加注量": "esbo_feed",
    "中和碱液添加量": "neutral_alkali_feed",
    "R513温度": "r513_temp",
    "R514温度": "r514_temp",
    "R512A温度": "r512a_temp",
    "R511A温度": "r511a_temp",
    "R510A温度": "r510a_temp",
    "卤化工段胶液总量2": "rubber_flow_2",
    "反应溴添加量": "bromine_feed",
    "储罐胶浓在线检测": "tank_rubber_conc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a supervised calcium-focused feature dataset for T90 qualification."
    )
    parser.add_argument("--input", type=Path, default=Path("data/data_clean.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--report", type=Path, default=Path("data/t90_ca_feature_report.json"))
    return parser.parse_args()


def as_jsonable(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): as_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [as_jsonable(item) for item in value]
    return value


def safe_name(column: str) -> str:
    return SAFE_NAMES.get(column, column)


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")


def load_clean_data(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input parquet does not exist: {path}")

    frame = pd.read_parquet(path)
    require_columns(frame, ["time", "t90"])

    warnings: list[str] = []
    raw_rows = len(frame)
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    invalid_time_count = int(frame["time"].isna().sum())
    if invalid_time_count:
        warnings.append(f"Removed {invalid_time_count} rows with invalid time values.")
        frame = frame[frame["time"].notna()].copy()

    frame = frame.sort_values("time").reset_index(drop=True)
    duplicate_time_count = int(frame["time"].duplicated().sum())
    if duplicate_time_count:
        warnings.append(
            f"Input contains {duplicate_time_count} duplicate timestamps; window features keep all duplicate rows."
        )
    if len(frame) != raw_rows:
        warnings.append(f"Raw row count changed from {raw_rows} to {len(frame)} after time parsing.")
    return frame, warnings


def exact_value_at(time_ns: np.ndarray, values: np.ndarray, anchor_ns: int) -> float:
    index = int(np.searchsorted(time_ns, anchor_ns, side="left"))
    if index < len(time_ns) and int(time_ns[index]) == anchor_ns:
        value = values[index]
        return float(value) if np.isfinite(value) else math.nan
    return math.nan


def window_slice(time_ns: np.ndarray, anchor_ns: int, window_minutes: int) -> slice:
    start_ns = anchor_ns - window_minutes * MINUTE_NS
    start = int(np.searchsorted(time_ns, start_ns, side="right"))
    end = int(np.searchsorted(time_ns, anchor_ns, side="right"))
    return slice(start, end)


def window_stats(time_ns: np.ndarray, values: np.ndarray, anchor_ns: int, window_minutes: int) -> dict[str, float]:
    rows = window_slice(time_ns, anchor_ns, window_minutes)
    window_values = values[rows]
    window_times = time_ns[rows]
    valid_mask = np.isfinite(window_values)
    valid_values = window_values[valid_mask]

    expected_count = max(1, int(window_minutes))
    non_null_count = int(valid_mask.sum())
    denominator = max(expected_count, len(window_values))
    missing_rate = 1.0 - non_null_count / denominator
    missing_rate = float(min(1.0, max(0.0, missing_rate)))

    if non_null_count == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "range": math.nan,
            "slope": math.nan,
            "missing_rate": missing_rate,
        }

    minimum = float(np.min(valid_values))
    maximum = float(np.max(valid_values))
    std = float(np.std(valid_values, ddof=1)) if non_null_count >= 2 else math.nan
    slope = math.nan
    if non_null_count >= 2:
        x = (window_times[valid_mask] - window_times[valid_mask][0]) / MINUTE_NS
        if np.ptp(x) > 0:
            slope = float(np.polyfit(x.astype(float), valid_values.astype(float), deg=1)[0])

    return {
        "mean": float(np.mean(valid_values)),
        "std": std,
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "slope": slope,
        "missing_rate": missing_rate,
    }


def calculate_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    valid = num.notna() & den.notna() & np.isfinite(num) & np.isfinite(den) & (den != 0)
    ratio = pd.Series(np.nan, index=numerator.index, dtype="float64")
    ratio.loc[valid] = num.loc[valid] / den.loc[valid]
    return ratio


def add_target_columns(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["y_ok"] = ((labels["t90"] >= T90_LOW) & (labels["t90"] <= T90_HIGH)).astype(int)
    labels["y_low"] = (labels["t90"] < T90_LOW).astype(int)
    labels["y_high"] = (labels["t90"] > T90_HIGH).astype(int)
    labels["y_out_spec"] = ((labels["t90"] < T90_LOW) | (labels["t90"] > T90_HIGH)).astype(int)
    return labels


def build_calcium_features(
    sample_time: pd.Timestamp,
    time_ns: np.ndarray,
    calcium_values: np.ndarray,
    ratio_values: np.ndarray | None,
) -> dict[str, float]:
    anchor_ns = int(sample_time.value - PROCESS_LAG_PRIORS[CALCIUM_COLUMN] * MINUTE_NS)
    features: dict[str, float] = {
        "ca_lag_165": exact_value_at(time_ns, calcium_values, anchor_ns),
    }

    for window in [15, 30, 60, 120]:
        stats = window_stats(time_ns, calcium_values, anchor_ns, window)
        if window in {15, 30, 60, 120}:
            features[f"ca_win_{window}_mean"] = stats["mean"]
        if window == 60:
            features["ca_win_60_std"] = stats["std"]
            features["ca_win_60_min"] = stats["min"]
            features["ca_win_60_max"] = stats["max"]
            features["ca_win_60_range"] = stats["range"]
            features["ca_win_60_slope"] = stats["slope"]
            features["ca_missing_rate_60"] = stats["missing_rate"]

    for delta_minutes in [15, 30, 60]:
        prior_value = exact_value_at(time_ns, calcium_values, anchor_ns - delta_minutes * MINUTE_NS)
        current_value = features["ca_lag_165"]
        features[f"ca_delta_{delta_minutes}"] = (
            current_value - prior_value if np.isfinite(current_value) and np.isfinite(prior_value) else math.nan
        )

    if ratio_values is not None:
        features["ca_per_rubber_flow_lag_165"] = exact_value_at(time_ns, ratio_values, anchor_ns)
        ratio_stats = window_stats(time_ns, ratio_values, anchor_ns, 60)
        features["ca_per_rubber_flow_win_60_mean"] = ratio_stats["mean"]
        features["ca_per_rubber_flow_win_60_slope"] = ratio_stats["slope"]
    else:
        features["ca_per_rubber_flow_lag_165"] = math.nan
        features["ca_per_rubber_flow_win_60_mean"] = math.nan
        features["ca_per_rubber_flow_win_60_slope"] = math.nan

    return features


def build_process_features(
    sample_time: pd.Timestamp,
    column: str,
    time_ns: np.ndarray,
    values: np.ndarray,
) -> dict[str, float]:
    delay = PROCESS_LAG_PRIORS[column]
    anchor_ns = int(sample_time.value - delay * MINUTE_NS)
    prefix = safe_name(column)
    features: dict[str, float] = {
        f"{prefix}_lag_{delay}": exact_value_at(time_ns, values, anchor_ns),
    }
    for window in [15, 30, 60]:
        stats = window_stats(time_ns, values, anchor_ns, window)
        for stat_name in ["mean", "std", "min", "max", "range", "slope", "missing_rate"]:
            features[f"{prefix}_win_{window}_{stat_name}"] = stats[stat_name]
    return features


def build_feature_dataset(frame: pd.DataFrame, warnings: list[str]) -> tuple[pd.DataFrame, list[str]]:
    missing_expected_columns = [column for column in PROCESS_LAG_PRIORS if column not in frame.columns]
    if missing_expected_columns:
        warnings.append(f"Missing expected process columns: {missing_expected_columns}")

    t90_non_null = frame[frame["t90"].notna()].copy()
    label_columns = [column for column in ["time", "t90", "t90_C", "t90_D", "t90_E", "t90_label_count"] if column in frame.columns]
    labels = add_target_columns(t90_non_null[label_columns].copy())

    time_ns = frame["time"].astype("int64").to_numpy()
    numeric_series = {
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
        for column in PROCESS_LAG_PRIORS
        if column in frame.columns
    }

    ratio_values: np.ndarray | None = None
    if CALCIUM_COLUMN in frame.columns and RUBBER_FLOW_COLUMN in frame.columns:
        ratio = calculate_ratio(frame[CALCIUM_COLUMN], frame[RUBBER_FLOW_COLUMN])
        ratio_values = ratio.to_numpy(dtype="float64")
    else:
        warnings.append(
            "Cannot create normalized calcium consumption because calcium or rubber-flow column is missing."
        )

    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    for _, label_row in labels.iterrows():
        sample_time = pd.Timestamp(label_row["time"])
        output_row = label_row.to_dict()

        if CALCIUM_COLUMN in numeric_series:
            output_row.update(
                build_calcium_features(sample_time, time_ns, numeric_series[CALCIUM_COLUMN], ratio_values)
            )
        else:
            for column in [
                "ca_lag_165",
                "ca_win_15_mean",
                "ca_win_30_mean",
                "ca_win_60_mean",
                "ca_win_120_mean",
                "ca_win_60_std",
                "ca_win_60_min",
                "ca_win_60_max",
                "ca_win_60_range",
                "ca_win_60_slope",
                "ca_delta_15",
                "ca_delta_30",
                "ca_delta_60",
                "ca_missing_rate_60",
                "ca_per_rubber_flow_lag_165",
                "ca_per_rubber_flow_win_60_mean",
                "ca_per_rubber_flow_win_60_slope",
            ]:
                output_row[column] = math.nan

        for column, values in numeric_series.items():
            if column == CALCIUM_COLUMN:
                continue
            output_row.update(build_process_features(sample_time, column, time_ns, values))
        rows.append(output_row)

    dataset = pd.DataFrame(rows)
    dataset = dataset.sort_values("time").reset_index(drop=True)
    return dataset, missing_expected_columns


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataset.columns
        if column not in LABEL_COLUMNS and not column.startswith("t90_")
    ]


def build_feature_groups(features: list[str]) -> dict[str, object]:
    feature_set = set(features)
    calcium_core = [feature for feature in CALCIUM_CORE_FEATURES if feature in feature_set]
    context_prefixes = [
        f"{safe_name(column)}_"
        for column in PROCESS_LAG_PRIORS
        if column != CALCIUM_COLUMN
    ]
    process_context = [
        feature
        for feature in features
        if any(feature.startswith(prefix) for prefix in context_prefixes)
    ]
    return {
        "calcium_core_features": calcium_core,
        "process_context_features": process_context,
        "all_candidate_features": features,
        "leakage_excluded_columns": LEAKAGE_EXCLUDED_COLUMNS,
        "recommended_usage": RECOMMENDED_USAGE,
    }


def summarize_missing_rates(dataset: pd.DataFrame, features: list[str]) -> dict[str, object]:
    if not features:
        return {
            "feature_missing_rate_min": None,
            "feature_missing_rate_median": None,
            "feature_missing_rate_mean": None,
            "feature_missing_rate_max": None,
            "top_missing_features": [],
        }
    missing_rates = dataset[features].isna().mean().sort_values(ascending=False)
    return {
        "feature_missing_rate_min": float(missing_rates.min()),
        "feature_missing_rate_median": float(missing_rates.median()),
        "feature_missing_rate_mean": float(missing_rates.mean()),
        "feature_missing_rate_max": float(missing_rates.max()),
        "top_missing_features": [
            {"feature": str(feature), "missing_rate": float(rate)}
            for feature, rate in missing_rates.head(20).items()
        ],
    }


def target_counts(dataset: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for column in ["y_ok", "y_low", "y_high", "y_out_spec"]:
        counts = dataset[column].value_counts(dropna=False).sort_index()
        result[column] = {str(key): int(value) for key, value in counts.items()}
    return result


def build_report(
    input_path: Path,
    output_path: Path,
    raw_frame: pd.DataFrame,
    dataset: pd.DataFrame,
    features: list[str],
    missing_expected_columns: list[str],
    warnings: list[str],
) -> dict[str, object]:
    time_min = raw_frame["time"].min()
    time_max = raw_frame["time"].max()
    leakage_columns = sorted(set(dataset.columns) & LABEL_COLUMNS)
    feature_groups = build_feature_groups(features)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_row_count": int(len(raw_frame)),
        "supervised_row_count": int(len(dataset)),
        "time_range": {
            "min": time_min.isoformat() if pd.notna(time_min) else None,
            "max": time_max.isoformat() if pd.notna(time_max) else None,
        },
        "t90_non_null_count": int(raw_frame["t90"].notna().sum()),
        "target_counts": target_counts(dataset),
        "feature_count": int(len(features)),
        "feature_columns": features,
        "feature_groups": feature_groups,
        "missing_rate_summary": summarize_missing_rates(dataset, features),
        "missing_expected_columns": missing_expected_columns,
        "warnings": warnings,
        "assumptions": [
            "All DCS features are computed from historical windows ending at sample_time minus the configured process lag.",
            "Lag features require an exact timestamp match at the process-aligned anchor; missing anchor values remain NaN.",
            "Window features use (anchor - window_minutes, anchor] and do not use values later than the anchor.",
            "NaN values are preserved during feature construction; no imputation, forward fill, or denominator zero fill is applied.",
            "Normalized calcium consumption is calculated only when both calcium feed and rubber flow are present and the denominator is finite and non-zero.",
            "Feature columns exclude time, t90, line-specific t90 columns, label count, and derived target columns.",
        ],
        "leakage_excluded_columns_present": leakage_columns,
        "process_lag_priors_min": PROCESS_LAG_PRIORS,
        "t90_qualified_interval": [T90_LOW, T90_HIGH],
    }


def print_summary(report: dict[str, object]) -> None:
    missing_summary = report["missing_rate_summary"]
    feature_groups = report.get("feature_groups", {})
    calcium_core = feature_groups.get("calcium_core_features", []) if isinstance(feature_groups, dict) else []
    process_context = feature_groups.get("process_context_features", []) if isinstance(feature_groups, dict) else []
    leakage_excluded = feature_groups.get("leakage_excluded_columns", []) if isinstance(feature_groups, dict) else []
    print("T90 calcium feature dataset built.")
    print(f"  input: {report['input_path']}")
    print(f"  output: {report['output_path']}")
    print(f"  raw rows: {report['raw_row_count']}")
    print(f"  supervised rows: {report['supervised_row_count']}")
    print(f"  t90 non-null rows: {report['t90_non_null_count']}")
    print(f"  feature count: {report['feature_count']}")
    print(f"  calcium core feature count: {len(calcium_core)}")
    print(f"  process context feature count: {len(process_context)}")
    print(f"  leakage excluded columns: {leakage_excluded}")
    if isinstance(missing_summary, dict):
        print(f"  median feature missing rate: {missing_summary.get('feature_missing_rate_median')}")
        print(f"  max feature missing rate: {missing_summary.get('feature_missing_rate_max')}")
    warnings = report.get("warnings") or []
    if warnings:
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")


def main() -> None:
    args = parse_args()
    frame, warnings = load_clean_data(args.input)
    dataset, missing_expected_columns = build_feature_dataset(frame, warnings)
    features = feature_columns(dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output, index=False)

    report = build_report(
        input_path=args.input,
        output_path=args.output,
        raw_frame=frame,
        dataset=dataset,
        features=features,
        missing_expected_columns=missing_expected_columns,
        warnings=warnings,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(as_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)


if __name__ == "__main__":
    main()
