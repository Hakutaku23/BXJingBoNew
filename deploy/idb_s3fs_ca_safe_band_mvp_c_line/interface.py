from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:
    from . import feature_adapter
    from . import package
    from .idb_s3fs_asset_loader import load_runtime_assets
except Exception:
    import feature_adapter  # type: ignore
    import package  # type: ignore
    from idb_s3fs_asset_loader import load_runtime_assets  # type: ignore


CA_FEED_POINT_LOWER_BOUND = 700.0
CA_FEED_POINT_UPPER_BOUND = 1300.0


def _sem_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "nan", "null"}:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if out != out or out in {float("inf"), float("-inf")}:
        return None
    return out


def _append_warning(existing: Any, warning: str) -> str:
    values: list[str] = []
    if isinstance(existing, str) and existing.strip():
        values.extend([part.strip() for part in existing.split(";") if part.strip()])
    elif isinstance(existing, list):
        values.extend([str(part).strip() for part in existing if str(part).strip()])
    if warning not in values:
        values.append(warning)
    return ";".join(sorted(set(values)))


def _clip_ca_feed_to_point_bounds(value: Any) -> tuple[Optional[float], bool]:
    numeric = _sem_to_float(value)
    if numeric is None:
        return None, False
    clipped = min(max(numeric, CA_FEED_POINT_LOWER_BOUND), CA_FEED_POINT_UPPER_BOUND)
    return clipped, clipped != numeric


def postprocess_output_semantics(output: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Add user-facing feed aliases and T90 risk-warning fields without changing rule decisions."""
    result = dict(output or {})
    state = dict(state or {})
    flow = _sem_to_float(state.get("rubber_flow_2_win_60_mean"))
    current_consumption = _sem_to_float(result.get("current_ca_consumption"))
    if flow is not None and current_consumption is not None:
        result["current_ca_feed"] = current_consumption * flow
    else:
        result.setdefault("current_ca_feed", None)
    conversion_ok = flow is not None and flow > 0
    for suffix in ("min", "max", "target"):
        feed_key = f"recommended_ca_feed_{suffix}"
        cons_key = f"recommended_ca_consumption_{suffix}"
        if result.get(feed_key) is None and conversion_ok:
            value = _sem_to_float(result.get(cons_key))
            result[feed_key] = None if value is None else value * flow
        result[f"recommended_ca_feed_unbounded_{suffix}"] = result.get(feed_key)
    if any(result.get(f"recommended_ca_feed_{suffix}") is None for suffix in ("min", "max", "target")):
        result["recommended_ca_feed_conversion_status"] = "unavailable"
        result["recommended_ca_feed_bounds_status"] = "unavailable"
        result["ca_feed_point_lower_bound"] = CA_FEED_POINT_LOWER_BOUND
        result["ca_feed_point_upper_bound"] = CA_FEED_POINT_UPPER_BOUND
        if not conversion_ok:
            result["warning_flags"] = _append_warning(result.get("warning_flags"), "feed_conversion_unavailable")
    else:
        result["recommended_ca_feed_conversion_status"] = "ok"
        clipped_any = False
        for suffix in ("min", "max", "target"):
            feed_key = f"recommended_ca_feed_{suffix}"
            clipped_value, was_clipped = _clip_ca_feed_to_point_bounds(result.get(feed_key))
            result[feed_key] = clipped_value
            clipped_any = clipped_any or was_clipped
        result["ca_feed_point_lower_bound"] = CA_FEED_POINT_LOWER_BOUND
        result["ca_feed_point_upper_bound"] = CA_FEED_POINT_UPPER_BOUND
        if clipped_any:
            result["recommended_ca_feed_bounds_status"] = "clipped_to_point_bounds"
            result["warning_flags"] = _append_warning(result.get("warning_flags"), "feed_recommendation_clipped_to_point_bounds")
        else:
            result["recommended_ca_feed_bounds_status"] = "within_point_bounds"
    position = str(result.get("interval_position") or "missing")
    if position == "above_band":
        result["t90_risk_level"] = "high_t90_risk"
        result["t90_high_risk_warning"] = True
        result["t90_low_risk_warning"] = False
    elif position == "below_band":
        result["t90_risk_level"] = "low_t90_risk"
        result["t90_high_risk_warning"] = False
        result["t90_low_risk_warning"] = True
    elif position == "inside_band":
        result["t90_risk_level"] = "low_risk_reference"
        result["t90_high_risk_warning"] = False
        result["t90_low_risk_warning"] = False
    else:
        result["t90_risk_level"] = "unknown"
        result["t90_high_risk_warning"] = False
        result["t90_low_risk_warning"] = False
    if result.get("recommendation_status", "").startswith("no_recommendation"):
        result["t90_risk_level"] = "unknown"
    result["prediction_type"] = "risk_warning_not_t90_value_prediction"
    result["recommendation_target"] = "calcium_stearate_feed"
    result["internal_normalized_metric"] = "calcium_consumption"
    result["control_mode"] = "guidance_monitor_only"
    result["automatic_control"] = False
    result["dcs_setpoint_writeback"] = False
    if result.get("selected_rule_id") is None:
        ids = str(result.get("selected_rule_ids") or "").split(";")
        result["selected_rule_id"] = ids[0] if ids and ids[0] else None
    result.setdefault("timestamp", state.get("time") or state.get("timestamp"))
    result.setdefault("residence_time_minutes_used", 174)
    timestamp = result.get("timestamp")
    if timestamp:
        try:
            import pandas as pd
            result.setdefault("estimated_quality_time", (pd.to_datetime(timestamp) + pd.Timedelta(minutes=174)).isoformat())
        except Exception:
            result.setdefault("estimated_quality_time", None)
    else:
        result.setdefault("estimated_quality_time", None)
    if not result.get("input_valid", True):
        result.setdefault("error_code", result.get("recommendation_status") or "no_recommendation")
        result.setdefault("error_message", "关键输入缺失或窗口有效点不足，未生成推荐。")
    else:
        result.setdefault("error_code", None)
        result.setdefault("error_message", None)
    return result

class SafeBandRecommender:
    def __init__(self, model_dir: Optional[Any] = None, mode: str = "production", config_path: Optional[Any] = None, config: Optional[Dict[str, Any]] = None):
        self.model_dir = Path(model_dir) if model_dir is not None else Path(__file__).resolve().parent
        self.mode = mode
        self.config_path = config_path
        self.config = dict(config or {})
        self.artifact = None
        self.support = None
        self.schema = None
        self.asset_paths = None

    def load(self) -> "SafeBandRecommender":
        loaded = load_runtime_assets(config_path=self.config_path, config=self.config, base_dir=self.model_dir)
        self.artifact = loaded["artifact"]
        self.support = loaded["support"]
        self.schema = loaded["schema"]
        self.asset_paths = loaded["paths"]
        return self

    def _ensure_loaded(self) -> None:
        if self.artifact is None or self.support is None or self.schema is None:
            self.load()

    def predict_one(self, state: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, Any]:
        self._ensure_loaded()
        return postprocess_output_semantics(package.recommend_one(state, self.artifact, self.support, schema=self.schema, mode=mode or self.mode), state)

    def predict_batch(self, input_data: Any, mode: Optional[str] = None) -> Any:
        self._ensure_loaded()
        try:
            import pandas as pd
        except Exception:
            pd = None  # type: ignore
        if pd is not None and isinstance(input_data, pd.DataFrame):
            rows = input_data.to_dict(orient="records")
            result = package.recommend_batch(rows, self.artifact, self.support, schema=self.schema, mode=mode or self.mode)
            return pd.DataFrame([postprocess_output_semantics(item, row) for item, row in zip(result, rows)])
        if isinstance(input_data, list):
            return [postprocess_output_semantics(item, row) for item, row in zip(package.recommend_batch(input_data, self.artifact, self.support, schema=self.schema, mode=mode or self.mode), input_data)]
        raise TypeError("predict_batch expects list[dict] or pandas.DataFrame when pandas is available.")

    def predict_from_raw_dataframe(
        self,
        df: Any,
        end_time: Any = None,
        time_col: str = "time",
        column_mapping: Optional[Dict[str, str]] = None,
        min_valid_points: int = 30,
        include_optional_ir: bool = False,
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        state = feature_adapter.build_runtime_features_from_dataframe(
            df,
            end_time=end_time,
            time_col=time_col,
            column_mapping=column_mapping,
            min_valid_points=min_valid_points,
            include_optional_ir=include_optional_ir,
        )
        pred = self.predict_one(state, mode="production")
        pred["adapter_feature_quality"] = state.get("feature_quality")
        pred["adapter_warning_flags"] = state.get("warning_flags")
        pred["adapter_missing_raw_columns"] = state.get("missing_raw_columns")
        pred["adapter_insufficient_window_features"] = state.get("insufficient_window_features")
        pred["adapter_time"] = state.get("time")
        return pred


def init(model_dir: Optional[Any] = None, mode: str = "production", config_path: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> SafeBandRecommender:
    return SafeBandRecommender(model_dir=model_dir, mode=mode, config_path=config_path, config=config).load()
