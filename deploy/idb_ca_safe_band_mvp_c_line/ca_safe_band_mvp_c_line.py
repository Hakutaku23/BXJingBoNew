from __future__ import annotations

import json
from typing import Any, Dict, Optional

try:
    from .interface import init
except Exception:
    from interface import init  # type: ignore


def run_once(
    input_data: Optional[Any] = None,
    raw_dataframe: Optional[Any] = None,
    end_time: Optional[Any] = None,
    time_col: str = "time",
    column_mapping: Optional[Dict[str, str]] = None,
    min_valid_points: int = 30,
) -> Any:
    recommender = init(mode="production", use_embedded_assets=True)
    if raw_dataframe is not None:
        return recommender.predict_from_raw_dataframe(
            raw_dataframe,
            end_time=end_time,
            time_col=time_col,
            column_mapping=column_mapping,
            min_valid_points=min_valid_points,
            include_optional_ir=True,
        )
    if isinstance(input_data, dict):
        return recommender.predict_one(input_data, mode="production")
    if isinstance(input_data, list):
        return recommender.predict_batch(input_data, mode="production")
    raise ValueError("Provide engineered feature dict/list or raw_dataframe.")


def main(payload: Optional[Any] = None) -> Any:
    if payload is None:
        print(json.dumps({"status": "ready", "mode": "monitor_only_guidance"}, ensure_ascii=False))
        return {"status": "ready", "mode": "monitor_only_guidance"}
    result = run_once(input_data=payload)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
