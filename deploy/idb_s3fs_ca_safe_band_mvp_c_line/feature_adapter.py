from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Optional

import pandas as pd


RAW_POINT_MAPPING = {
    "rubber_flow_2": {"friendly": "rubber_flow_2", "tag": "B4-FIC-C51001.PV.F_CV", "output": "rubber_flow_2_win_60_mean"},
    "bromine_feed": {"friendly": "bromine_feed", "tag": "B4-FIC-C51004.PV.CV", "output": "bromine_feed_win_60_mean"},
    "tank_rubber_conc": {"friendly": "tank_rubber_conc", "tag": "B4-AT-C50002A-BIIR.PV.CV", "output": "tank_rubber_conc_win_60_mean"},
    "r510a_temp": {"friendly": "r510a_temp", "tag": "B4-TI-C51007A_S.PV.CV", "output": "r510a_temp_win_60_mean"},
    "r511a_temp": {"friendly": "r511a_temp", "tag": "B4-TI-C51101A_S.PV.CV", "output": "r511a_temp_win_60_mean"},
    "r512a_temp": {"friendly": "r512a_temp", "tag": "B4-TI-C51702A.PV.F_CV", "output": "r512a_temp_win_60_mean"},
    "ca_feed": {"friendly": "ca_feed", "tag": "B4-FIC-C51401.PV.F_CV", "output": "ca_per_rubber_flow_win_60_mean"},
    "esbo_feed": {"friendly": "esbo_feed", "tag": "B4-FIC-C51801.PV.F_CV", "output": "esbo_feed_win_60_mean"},
    "neutral_alkali_feed": {"friendly": "neutral_alkali_feed", "tag": "B4-FIC-C51605.PV.F_CV", "output": "neutral_alkali_feed_win_60_mean"},
    "r513_temp": {"friendly": "r513_temp", "tag": "B4-TI-C51301_S.PV.CV", "output": "r513_temp_win_60_mean"},
    "r514_temp": {"friendly": "r514_temp", "tag": "B4-TI-C51401_S.PV.CV", "output": "r514_temp_win_60_mean"},
}
IR_CANDIDATES = ["output_ir_corrected", "Y_cal", "output_ir"]


def normalize_columns(df: pd.DataFrame, column_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    data = df.copy()
    rename: Dict[str, str] = {}
    for _, meta in RAW_POINT_MAPPING.items():
        friendly = meta["friendly"]
        tag = meta["tag"]
        if tag in data.columns and friendly not in data.columns:
            rename[tag] = friendly
    if column_mapping:
        for source, target in column_mapping.items():
            if source in data.columns:
                rename[source] = target
    return data.rename(columns=rename)


def infer_time_column(df: pd.DataFrame, preferred: str = "time") -> str:
    if preferred in df.columns:
        return preferred
    for candidate in ["timestamp", "datetime", "date_time", "Time", "TIME"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("No timestamp column found; provide time_col.")


def _prepare_time(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    return data.dropna(subset=[time_col]).sort_values(time_col)


def safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _bound_lookup(point_bounds: Any) -> Dict[str, Dict[str, Optional[float]]]:
    lookup: Dict[str, Dict[str, Optional[float]]] = {}
    if not point_bounds:
        return lookup
    for row in point_bounds:
        if not isinstance(row, dict):
            continue
        name = row.get("friendly_name") or row.get("point") or row.get("friendly")
        if not name:
            continue
        low = row.get("lower_bound")
        high = row.get("upper_bound")
        try:
            low = float(low) if low is not None and str(low).strip() != "" else None
        except Exception:
            low = None
        try:
            high = float(high) if high is not None and str(high).strip() != "" else None
        except Exception:
            high = None
        lookup[str(name)] = {"lower_bound": low, "upper_bound": high}
    return lookup


def apply_point_bounds_cleaning(df: pd.DataFrame, point_bounds: Any = None) -> pd.DataFrame:
    data = df.copy()
    lookup = _bound_lookup(point_bounds)
    for key in RAW_POINT_MAPPING:
        if key not in data.columns:
            continue
        values = safe_numeric_series(data[key])
        invalid = pd.Series(False, index=data.index)
        low = lookup.get(key, {}).get("lower_bound")
        high = lookup.get(key, {}).get("upper_bound")
        if low is not None:
            invalid |= values < low
        if high is not None:
            invalid |= values > high
        if key == "ca_feed":
            invalid |= values < 0
        if key == "rubber_flow_2":
            invalid |= values <= 0
        data.loc[invalid, key] = None
    return data


def filter_window(df: pd.DataFrame, end_time: Any, minutes: int) -> pd.DataFrame:
    time_col = infer_time_column(df, "time")
    data = _prepare_time(df, time_col)
    end = pd.to_datetime(end_time)
    start = end - timedelta(minutes=minutes)
    return data.loc[(data[time_col] >= start) & (data[time_col] <= end)].copy()


def calc_window_mean(df: pd.DataFrame, column: str, end_time: Any, minutes: int = 60, min_valid_points: int = 30) -> Optional[float]:
    time_col = infer_time_column(df, "time")
    data = _prepare_time(df, time_col)
    end = pd.to_datetime(end_time)
    start = end - timedelta(minutes=minutes)
    if column not in data.columns:
        return None
    values = safe_numeric_series(data.loc[(data[time_col] >= start) & (data[time_col] <= end), column]).dropna()
    if len(values) < min_valid_points:
        return None
    return float(values.mean())


def calc_ca_consumption_window_mean(df: pd.DataFrame, ca_col: str, rubber_flow_col: str, end_time: Any, minutes: int = 60, min_valid_points: int = 30) -> Optional[float]:
    time_col = infer_time_column(df, "time")
    data = _prepare_time(df, time_col)
    end = pd.to_datetime(end_time)
    start = end - timedelta(minutes=minutes)
    if ca_col not in data.columns or rubber_flow_col not in data.columns:
        return None
    frame = data.loc[(data[time_col] >= start) & (data[time_col] <= end), [ca_col, rubber_flow_col]].copy()
    ca = safe_numeric_series(frame[ca_col])
    flow = safe_numeric_series(frame[rubber_flow_col])
    ratio = ca.where(ca >= 0) / flow.where(flow > 0)
    ratio = ratio.replace([float("inf"), float("-inf")], pd.NA).dropna()
    ratio = ratio.where(ratio >= 0).dropna()
    if len(ratio) < min_valid_points:
        return None
    return float(ratio.mean())


def calc_ir_lag_std(df: pd.DataFrame, ir_col: str, end_time: Any, offset_minutes: int = 20, window_minutes: int = 15, min_valid_points: int = 5) -> Optional[float]:
    time_col = infer_time_column(df, "time")
    data = _prepare_time(df, time_col)
    end = pd.to_datetime(end_time) - timedelta(minutes=offset_minutes)
    start = end - timedelta(minutes=window_minutes)
    if ir_col not in data.columns:
        return None
    values = safe_numeric_series(data.loc[(data[time_col] >= start) & (data[time_col] <= end), ir_col]).dropna()
    if len(values) < min_valid_points:
        return None
    return float(values.std(ddof=1))


def _find_ir_column(data: pd.DataFrame, column_mapping: Optional[Dict[str, str]] = None) -> Optional[str]:
    if column_mapping:
        for source, target in column_mapping.items():
            if target == "output_ir_corrected" and source in data.columns:
                return source
            if source in data.columns and source in IR_CANDIDATES:
                return source
    for candidate in IR_CANDIDATES:
        if candidate in data.columns:
            return candidate
    return None


def build_runtime_features_from_dataframe(
    df: pd.DataFrame,
    end_time: Any = None,
    time_col: str = "time",
    column_mapping: Optional[Dict[str, str]] = None,
    min_valid_points: int = 30,
    include_optional_ir: bool = True,
    point_bounds: Any = None,
) -> Dict[str, Any]:
    data = normalize_columns(df, column_mapping=column_mapping)
    actual_time_col = infer_time_column(data, time_col)
    if actual_time_col != "time":
        data = data.rename(columns={actual_time_col: "time"})
    data = apply_point_bounds_cleaning(data, point_bounds=point_bounds)
    data = _prepare_time(data, "time")
    if data.empty:
        raise ValueError("Raw DataFrame has no valid timestamp rows.")
    if end_time is None:
        end_time = data["time"].max()
    end_time = pd.to_datetime(end_time)
    result: Dict[str, Any] = {"time": end_time.isoformat()}
    missing_raw: List[str] = []
    insufficient: List[str] = []

    for key, meta in RAW_POINT_MAPPING.items():
        if key == "ca_feed":
            continue
        output = meta["output"]
        if key not in data.columns:
            result[output] = None
            missing_raw.append(key)
            continue
        value = calc_window_mean(data, key, end_time, minutes=60, min_valid_points=min_valid_points)
        result[output] = value
        if value is None:
            insufficient.append(output)

    if "ca_feed" not in data.columns:
        missing_raw.append("ca_feed")
    if "rubber_flow_2" not in data.columns:
        missing_raw.append("rubber_flow_2")
    ca_value = calc_ca_consumption_window_mean(data, "ca_feed", "rubber_flow_2", end_time, minutes=60, min_valid_points=min_valid_points)
    result["ca_per_rubber_flow_win_60_mean"] = ca_value
    result["current_ca_consumption"] = ca_value
    if ca_value is None:
        insufficient.append("ca_per_rubber_flow_win_60_mean")

    if include_optional_ir:
        ir_col = _find_ir_column(data, column_mapping)
        if ir_col:
            result["output_ir_corrected_offset_20_win_15_std"] = calc_ir_lag_std(data, ir_col, end_time)
            if result["output_ir_corrected_offset_20_win_15_std"] is None:
                insufficient.append("output_ir_corrected_offset_20_win_15_std")
        else:
            result["output_ir_corrected_offset_20_win_15_std"] = None
    else:
        result["output_ir_corrected_offset_20_win_15_std"] = None

    result["missing_raw_columns"] = sorted(set(missing_raw))
    result["insufficient_window_features"] = sorted(set(insufficient))
    warnings: List[str] = []
    if "output_ir_corrected_offset_20_win_15_std" in insufficient or result.get("output_ir_corrected_offset_20_win_15_std") is None:
        warnings.append("optional_ir_missing")
    if insufficient:
        warnings.append("insufficient_window_features")
    if missing_raw:
        warnings.append("missing_raw_columns")
    required_insufficient = [item for item in insufficient if item != "output_ir_corrected_offset_20_win_15_std"]
    result["feature_quality"] = "ok" if not required_insufficient and not missing_raw else "incomplete"
    result["warning_flags"] = warnings
    return result


def build_batch_runtime_features_from_dataframe(
    df: pd.DataFrame,
    evaluation_times: Any,
    time_col: str = "time",
    column_mapping: Optional[Dict[str, str]] = None,
    min_valid_points: int = 30,
    include_optional_ir: bool = True,
    point_bounds: Any = None,
) -> pd.DataFrame:
    rows = [
        build_runtime_features_from_dataframe(
            df,
            end_time=end_time,
            time_col=time_col,
            column_mapping=column_mapping,
            min_valid_points=min_valid_points,
            include_optional_ir=include_optional_ir,
            point_bounds=point_bounds,
        )
        for end_time in evaluation_times
    ]
    return pd.DataFrame(rows)
