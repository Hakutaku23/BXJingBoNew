from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd
import requests

try:
    from interface import (
        init as _runtime_init,
        predict_one as _predict_one,
        predict_batch as _predict_batch,
        run_once as _runtime_run_once,
    )
except Exception:  # pragma: no cover
    from .interface import (  # type: ignore
        init as _runtime_init,
        predict_one as _predict_one,
        predict_batch as _predict_batch,
        run_once as _runtime_run_once,
    )

# ============================================================================
# 现场固定配置区：IDB 平台无法读取环境变量时，只改这里。
# 现场配置全部写在本文件内，不依赖进程环境变量。
# ============================================================================
DATA_HUB_HOST = "http://10.4.0.211:30805"
LOOKBACK_MINUTES = 90
HISTORY_INTERVAL_SECONDS = 60
MIN_VALID_POINTS = 30
REQUEST_TIMEOUT_SECONDS = 10.0

# 二选一：
# 1) 若现场通过 s3fs API 下载，配置 MODEL_S3_DIR；
# 2) 若现场已经把 s3fs 挂载成本地目录，配置 S3FS_ASSET_DIR，并可把 MODEL_S3_DIR 留空。
# 注意：若你已经把 MODEL_S3_DIR 改成真实路径，保留你的真实路径即可。
MODEL_S3_DIR = "s3://data/t90_Ca"
S3FS_ASSET_DIR = ""  # 示例："/data/idb_algos/ca_safe_band_mvp_c_line/1.0.0-strict-safe"
CACHE_ROOT = "/tmp/idb_algos"

# 胶浓为百分数：例如 500_ZHUJI.P1 返回 18.0 表示 18%，计算干胶量时必须换算为 0.18。
RUBBER_CONC_SCALE = 0.01

# 已确认的读取点位：实际 IDB tag -> 算法内部字段。
# rubber_conc 仅用于回写前按实时干胶量计算钙单耗上下限，不参与算法规则匹配。
IDB_TAG_TO_FIELD = {
    "FIC-C51001": "rubber_flow_2",
    "FIC-C51004": "bromine_feed",
    "AT-C5002A-BIIR": "tank_rubber_conc",
    "TI-C51007A_S": "r510a_temp",
    "TI-C51101_S": "r511a_temp",
    "TI-C51702A": "r512a_temp",
    "FIC-C51401": "ca_feed",
    "FIC-C51801": "esbo_feed",
    "FIC-C51605": "neutral_alkali_feed",
    "TI-C51301_S": "r513_temp",
    "TI-C51401_S": "r514_temp",
    "500_ZHUJI.P1": "rubber_conc",
}

# 回写点位：算法输出字段 -> IDB 输出 tag。
# 这里仅回写按“实时干胶量”换算后的钙单耗上限/下限。
# 钙单耗 = 硬脂酸钙加注量 / 干胶量；干胶量 = 卤化工段胶液总量2 * 胶浓。
OUTPUT_TAG_MAP: Dict[str, str] = {
    "ca_consumption_upper_writeback": "Cal_Cadanhao_UP",
    "ca_consumption_lower_writeback": "Cal_Cadanhao_LOW",
}

POSITION_CODE = {"inside_band": 0, "above_band": 1, "below_band": -1, "missing": 99}
_DEFAULT_RUNTIME = None


def _runtime_config(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "model_s3_dir": MODEL_S3_DIR,
        "asset_dir": S3FS_ASSET_DIR or None,
        "cache_root": CACHE_ROOT,
    }
    if extra:
        cfg.update({k: v for k, v in extra.items() if v is not None})
    return cfg


def init(config_path: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> Any:
    global _DEFAULT_RUNTIME
    _DEFAULT_RUNTIME = _runtime_init(config_path=config_path, config=_runtime_config(config), mode="production")
    return _DEFAULT_RUNTIME


def predict_one(
    row: Dict[str, Any],
    config_path: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if _DEFAULT_RUNTIME is None or config_path is not None or config is not None:
        init(config_path=config_path, config=config)
    return _predict_one(row)


def predict_batch(
    rows: Any,
    config_path: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    if _DEFAULT_RUNTIME is None or config_path is not None or config is not None:
        init(config_path=config_path, config=config)
    return _predict_batch(rows)


def _safe_output_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _append_warning(result: Dict[str, Any], flag: str) -> None:
    old = result.get("warning_flags")
    if old is None or old == "":
        result["warning_flags"] = flag
    elif isinstance(old, list):
        if flag not in old:
            old.append(flag)
        result["warning_flags"] = old
    else:
        parts = [x for x in str(old).split(";") if x]
        if flag not in parts:
            parts.append(flag)
        result["warning_flags"] = ";".join(parts)


def fail_safe_result(code: str, message: str) -> Dict[str, Any]:
    return {
        "recommendation_status": "no_recommendation_runtime_error",
        "input_valid": False,
        "interval_position": "missing",
        "interval_position_code": 99,
        "algo_status_code": 100,
        "error_code": code,
        "error_message": message,
        "automatic_control": False,
        "dcs_setpoint_writeback": False,
        "control_mode": "guidance_monitor_only",
        "recommendation_target": "calcium_stearate_feed",
        "prediction_type": "risk_warning_not_t90_value_prediction",
    }


def read_hisdata(
    data_hub_host: str,
    tag_names: list[str],
    time_begin: str,
    time_end: str,
    interval: int,
) -> pd.DataFrame:
    url = f"{data_hub_host.rstrip('/')}/api/tag-value/getHistoryValue"
    param = {
        "data": {
            "begTime": time_begin,
            "endTime": time_end,
            "interval": interval,
            "isSecond": True,
            "option": 0,
            "tagNames": tag_names,
        },
        "requestBase": {"page": "0-0", "sort": "-appTime"},
    }
    resp = requests.post(url, json=param, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    res = resp.json()
    if not res.get("isSuccess"):
        raise RuntimeError(f"IDB getHistoryValue failed: {res}")
    records = res.get("content", {}).get("records") or []
    if not records:
        return pd.DataFrame(columns=["tagTime", "tagName", "tagValue"])
    return pd.DataFrame(records)


def _resolve_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    time_col = next((c for c in ["tagTime", "time", "timestamp", "appTime", "dataTime"] if c in df.columns), None)
    tag_col = next((c for c in ["tagName", "tag", "pointName", "name"] if c in df.columns), None)
    value_col = next((c for c in ["tagValue", "value", "val"] if c in df.columns), None)
    if not time_col or not tag_col or not value_col:
        raise ValueError(f"history dataframe missing required columns; got {list(df.columns)}")
    return time_col, tag_col, value_col


def history_long_to_wide(df: pd.DataFrame, tag_to_field: Dict[str, str] = IDB_TAG_TO_FIELD) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["time", *sorted(set(tag_to_field.values()))])
    time_col, tag_col, value_col = _resolve_columns(df)
    work = df[[time_col, tag_col, value_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work[tag_col] = work[tag_col].astype(str).str.strip()
    work["field"] = work[tag_col].map(tag_to_field)
    work = work.dropna(subset=[time_col, "field"])
    if work.empty:
        return pd.DataFrame(columns=["time", *sorted(set(tag_to_field.values()))])
    wide = work.pivot_table(index=time_col, columns="field", values=value_col, aggfunc="mean").reset_index()
    return wide.rename(columns={time_col: "time"}).sort_values("time")


def read_raw_window(end_time: Optional[Any] = None) -> pd.DataFrame:
    if end_time is None:
        end_dt = datetime.now()
    else:
        end_dt = pd.to_datetime(end_time).to_pydatetime()
    start_dt = end_dt - timedelta(minutes=LOOKBACK_MINUTES)
    raw = read_hisdata(
        DATA_HUB_HOST,
        list(IDB_TAG_TO_FIELD.keys()),
        start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        HISTORY_INTERVAL_SECONDS,
    )
    return history_long_to_wide(raw)


def _latest_finite_value(df: pd.DataFrame, column: str) -> Optional[float]:
    if df is None or df.empty or column not in df.columns:
        return None
    work = df.copy()
    if "time" in work.columns:
        work["time"] = pd.to_datetime(work["time"], errors="coerce")
        work = work.dropna(subset=["time"]).sort_values("time")
    values = pd.to_numeric(work[column], errors="coerce").dropna()
    if values.empty:
        return None
    value = float(values.iloc[-1])
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def enrich_realtime_ca_consumption_writeback(result: Dict[str, Any], raw_df: Optional[Any]) -> Dict[str, Any]:
    """按实时干胶量换算待回写钙单耗上下限。

    计算口径：
      干胶量 = 卤化工段胶液总量2实时值 * 胶浓实时值 * RUBBER_CONC_SCALE
      钙单耗下限 = recommended_ca_feed_min / 干胶量
      钙单耗上限 = recommended_ca_feed_max / 干胶量

    注意：recommended_ca_feed_min/max 是已经反归一化并经安全截断后的硬脂酸钙加注量边界。
    """
    if raw_df is None:
        _append_warning(result, "missing_raw_df_for_realtime_ca_consumption_writeback")
        result["ca_consumption_upper_writeback"] = None
        result["ca_consumption_lower_writeback"] = None
        return result

    data = pd.DataFrame(raw_df).copy()
    rubber_flow = _latest_finite_value(data, "rubber_flow_2")
    rubber_conc = _latest_finite_value(data, "rubber_conc")
    current_ca_feed = _latest_finite_value(data, "ca_feed")

    result["realtime_rubber_flow_2"] = rubber_flow
    result["realtime_rubber_conc"] = rubber_conc
    result["rubber_conc_scale"] = RUBBER_CONC_SCALE

    if rubber_flow is None or rubber_flow <= 0:
        _append_warning(result, "invalid_realtime_rubber_flow_for_ca_consumption_writeback")
        result["dry_rubber_realtime"] = None
        result["ca_consumption_upper_writeback"] = None
        result["ca_consumption_lower_writeback"] = None
        return result
    if rubber_conc is None or rubber_conc <= 0:
        _append_warning(result, "invalid_realtime_rubber_conc_for_ca_consumption_writeback")
        result["dry_rubber_realtime"] = None
        result["ca_consumption_upper_writeback"] = None
        result["ca_consumption_lower_writeback"] = None
        return result

    dry_rubber = rubber_flow * rubber_conc * RUBBER_CONC_SCALE
    result["dry_rubber_realtime"] = dry_rubber
    if dry_rubber <= 0 or math.isnan(dry_rubber) or math.isinf(dry_rubber):
        _append_warning(result, "invalid_realtime_dry_rubber_for_ca_consumption_writeback")
        result["ca_consumption_upper_writeback"] = None
        result["ca_consumption_lower_writeback"] = None
        return result

    feed_min = _safe_output_value(result.get("recommended_ca_feed_min"))
    feed_max = _safe_output_value(result.get("recommended_ca_feed_max"))

    result["ca_consumption_lower_writeback"] = None if feed_min is None else feed_min / dry_rubber
    result["ca_consumption_upper_writeback"] = None if feed_max is None else feed_max / dry_rubber
    result["current_ca_consumption_realtime"] = None if current_ca_feed is None else current_ca_feed / dry_rubber
    return result


def run_once(
    raw_df: Optional[Any] = None,
    end_time: Optional[Any] = None,
    row: Optional[Dict[str, Any]] = None,
    config_path: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if _DEFAULT_RUNTIME is None or config_path is not None or config is not None:
        init(config_path=config_path, config=config)

    wide_df: Optional[pd.DataFrame] = None
    if row is not None:
        result = _predict_one(row)
    else:
        if raw_df is None:
            wide_df = read_raw_window(end_time=end_time)
        else:
            wide_df = pd.DataFrame(raw_df).copy()
        result = _runtime_run_once(
            raw_df=wide_df,
            end_time=end_time,
            min_valid_points=MIN_VALID_POINTS,
            include_optional_ir=False,
        )
        result = enrich_realtime_ca_consumption_writeback(result, wide_df)

    result["interval_position_code"] = POSITION_CODE.get(str(result.get("interval_position") or "missing"), 99)
    result["algo_status_code"] = 0 if result.get("recommendation_status") == "recommended" and result.get("input_valid") else 10
    result["automatic_control"] = False
    result["dcs_setpoint_writeback"] = False
    return result


def configured_output_tags(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    out = {str(result_key): str(tag).strip() for result_key, tag in OUTPUT_TAG_MAP.items() if str(tag).strip()}
    if extra:
        for result_key, tag in extra.items():
            if tag:
                out[str(result_key)] = str(tag).strip()
    return out


def write_monitor_outputs(
    result: Dict[str, Any],
    data_hub_host: str = DATA_HUB_HOST,
    output_tag_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    tag_map = configured_output_tags(output_tag_map)
    values: Dict[str, float] = {}
    skipped: Dict[str, Any] = {}
    for result_key, tag in tag_map.items():
        value = _safe_output_value(result.get(result_key))
        if value is not None:
            values[tag] = value
        else:
            skipped[result_key] = result.get(result_key)
    if not values:
        return {"written": False, "reason": "no_valid_output_values", "values": {}, "skipped": skipped}
    url = f"{data_hub_host.rstrip('/')}/api/tag-value/writeTagValues"
    payload = {"data": {"values": values}, "requestBase": {"page": "1-10", "sort": "-createTime"}}
    resp = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json", "Connection": "keep-alive"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return {"written": True, "values": values, "skipped": skipped, "response": resp.json() if resp.content else None}


def main(payload: Optional[Any] = None) -> Any:
    output_tag_map = None
    runtime_config = None
    try:
        if isinstance(payload, dict):
            output_tag_map = payload.get("output_tag_map") if isinstance(payload.get("output_tag_map"), dict) else None
            runtime_config = payload.get("runtime_config") if isinstance(payload.get("runtime_config"), dict) else None
            if "row" in payload:
                result = run_once(row=payload["row"], config=runtime_config)
            elif "raw_df" in payload:
                result = run_once(raw_df=payload["raw_df"], end_time=payload.get("end_time"), config=runtime_config)
            else:
                result = run_once(row=payload, config=runtime_config)
        else:
            result = run_once()
        result["writeback"] = write_monitor_outputs(result, output_tag_map=output_tag_map)
    except Exception as exc:
        result = fail_safe_result(exc.__class__.__name__, str(exc))
        try:
            result["writeback"] = write_monitor_outputs(result, output_tag_map=output_tag_map)
        except Exception as write_exc:
            result["writeback"] = {"written": False, "error": str(write_exc)}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
