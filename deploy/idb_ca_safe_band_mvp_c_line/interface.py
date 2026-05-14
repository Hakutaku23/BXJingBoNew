from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from . import feature_adapter
    from . import package
    from .runtime_assets_embedded import MODEL_METADATA, SAFE_BAND_ARTIFACT, SCHEMA, SUPPORT
except Exception:
    import feature_adapter  # type: ignore
    import package  # type: ignore
    from runtime_assets_embedded import MODEL_METADATA, SAFE_BAND_ARTIFACT, SCHEMA, SUPPORT  # type: ignore


def _load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class SafeBandRecommender:
    def __init__(self, model_dir: Optional[Any] = None, mode: str = "production", use_embedded_assets: bool = True):
        self.model_dir = Path(model_dir) if model_dir is not None else Path(__file__).resolve().parent
        self.mode = mode
        self.use_embedded_assets = use_embedded_assets
        self.artifact = None
        self.support = None
        self.schema = None

    def load(self) -> "SafeBandRecommender":
        if self.use_embedded_assets:
            self.artifact = SAFE_BAND_ARTIFACT
            self.support = SUPPORT
            self.schema = SCHEMA
            return self
        self.artifact = _load_json_file(self.model_dir / "safe_band_artifact.json")
        self.support = _load_json_file(self.model_dir / "support.json")
        self.schema = _load_json_file(self.model_dir / "schema.json")
        return self

    def _ensure_loaded(self) -> None:
        if self.artifact is None or self.support is None or self.schema is None:
            self.load()

    def predict_one(self, state: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, Any]:
        self._ensure_loaded()
        return package.recommend_one(state, self.artifact, self.support, schema=self.schema, mode=mode or self.mode)

    def predict_batch(self, input_data: Any, mode: Optional[str] = None) -> Any:
        self._ensure_loaded()
        try:
            import pandas as pd
        except Exception:
            pd = None  # type: ignore
        if pd is not None and isinstance(input_data, pd.DataFrame):
            rows = input_data.to_dict(orient="records")
            result = package.recommend_batch(rows, self.artifact, self.support, schema=self.schema, mode=mode or self.mode)
            return pd.DataFrame(result)
        if isinstance(input_data, list):
            return package.recommend_batch(input_data, self.artifact, self.support, schema=self.schema, mode=mode or self.mode)
        raise TypeError("predict_batch expects list[dict] or pandas.DataFrame when pandas is available.")

    def predict_from_raw_dataframe(
        self,
        df: Any,
        end_time: Any = None,
        time_col: str = "time",
        column_mapping: Optional[Dict[str, str]] = None,
        min_valid_points: int = 30,
        include_optional_ir: bool = True,
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        point_bounds = self.support.get("point_bounds") if isinstance(self.support, dict) else None
        state = feature_adapter.build_runtime_features_from_dataframe(
            df,
            end_time=end_time,
            time_col=time_col,
            column_mapping=column_mapping,
            min_valid_points=min_valid_points,
            include_optional_ir=include_optional_ir,
            point_bounds=point_bounds,
        )
        pred = self.predict_one(state, mode="production")
        pred["adapter_feature_quality"] = state.get("feature_quality")
        pred["adapter_warning_flags"] = state.get("warning_flags")
        pred["adapter_missing_raw_columns"] = state.get("missing_raw_columns")
        pred["adapter_insufficient_window_features"] = state.get("insufficient_window_features")
        pred["adapter_time"] = state.get("time")
        return pred

    def predict_batch_from_raw_dataframe(
        self,
        df: Any,
        evaluation_times: Any = None,
        time_col: str = "time",
        column_mapping: Optional[Dict[str, str]] = None,
        min_valid_points: int = 30,
        include_optional_ir: bool = True,
    ) -> Any:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError("Raw DataFrame feature adapter requires pandas.") from exc
        self._ensure_loaded()
        point_bounds = self.support.get("point_bounds") if isinstance(self.support, dict) else None
        if evaluation_times is None:
            state = feature_adapter.build_runtime_features_from_dataframe(
                df,
                end_time=None,
                time_col=time_col,
                column_mapping=column_mapping,
                min_valid_points=min_valid_points,
                include_optional_ir=include_optional_ir,
                point_bounds=point_bounds,
            )
            states = pd.DataFrame([state])
        else:
            states = feature_adapter.build_batch_runtime_features_from_dataframe(
                df,
                evaluation_times=evaluation_times,
                time_col=time_col,
                column_mapping=column_mapping,
                min_valid_points=min_valid_points,
                include_optional_ir=include_optional_ir,
                point_bounds=point_bounds,
            )
        preds = self.predict_batch(states, mode="production")
        for col in ["feature_quality", "warning_flags", "missing_raw_columns", "insufficient_window_features", "time"]:
            if col in states.columns:
                preds["adapter_" + col] = states[col].values
        return preds

    def metadata(self) -> Dict[str, Any]:
        return dict(MODEL_METADATA)


def init(model_dir: Optional[Any] = None, mode: str = "production", use_embedded_assets: bool = True) -> SafeBandRecommender:
    return SafeBandRecommender(model_dir=model_dir, mode=mode, use_embedded_assets=use_embedded_assets).load()
