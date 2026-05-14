from __future__ import annotations

import json
from typing import Any, Dict, Optional

try:
    from .interface import init as _init
except Exception:
    from interface import init as _init  # type: ignore


_DEFAULT_RECOMMENDER = None


def init(config_path: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> Any:
    global _DEFAULT_RECOMMENDER
    _DEFAULT_RECOMMENDER = _init(mode="production", config_path=config_path, config=config)
    return _DEFAULT_RECOMMENDER


def _rec(config_path: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> Any:
    global _DEFAULT_RECOMMENDER
    if _DEFAULT_RECOMMENDER is None or config_path is not None or config is not None:
        return init(config_path=config_path, config=config)
    return _DEFAULT_RECOMMENDER


def predict_one(row: Dict[str, Any], config_path: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _rec(config_path=config_path, config=config).predict_one(row, mode="production")


def predict_batch(rows: Any, config_path: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> Any:
    return _rec(config_path=config_path, config=config).predict_batch(rows, mode="production")


def run_once(raw_df: Optional[Any] = None, end_time: Optional[Any] = None, row: Optional[Dict[str, Any]] = None, config_path: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    recommender = _rec(config_path=config_path, config=config)
    if raw_df is not None:
        return recommender.predict_from_raw_dataframe(raw_df, end_time=end_time)
    if row is not None:
        return recommender.predict_one(row, mode="production")
    raise ValueError("Provide raw_df or engineered feature row.")


def main(payload: Optional[Any] = None) -> Any:
    if payload is None:
        out = {"status": "ready", "mode": "guidance_monitor_only"}
        print(json.dumps(out, ensure_ascii=False))
        return out
    result = run_once(row=payload)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
