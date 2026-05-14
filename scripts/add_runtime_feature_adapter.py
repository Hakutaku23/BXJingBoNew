from __future__ import annotations

import argparse
import ast
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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
STANDARD_IMPORTS = {"__future__", "argparse", "ast", "csv", "datetime", "json", "math", "os", "pathlib", "re", "statistics", "sys", "typing", "warnings"}
IMPORT_TO_PACKAGE = {"pandas": "pandas", "pyarrow": "pyarrow", "numpy": "numpy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add raw platform DataFrame feature adapter to safe-band runtime package.")
    parser.add_argument("--requirements", type=Path, default=Path("IDB_requirements.txt"))
    parser.add_argument("--deploy-dir", type=Path, default=Path("deploy/ca_safe_band_mvp"))
    parser.add_argument("--final-dry-run", type=Path, default=Path("runs/ca_safe_band_mvp/final_monitor_dry_run.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ca_safe_band_feature_adapter"))
    parser.add_argument("--experiment-doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    return parser.parse_args()


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def parse_requirements(path: Path) -> set[str]:
    packages = set()
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("package") or set(line) <= {"-", " "}:
            continue
        token = re.split(r"\s+|==|>=|<=|~=|>|<", line, maxsplit=1)[0].strip()
        if token and re.match(r"^[A-Za-z0-9_.-]+$", token):
            packages.add(normalize_package_name(token))
    return packages


def find_by_name(name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        if root.exists():
            matches = sorted(root.rglob(name))
            if matches:
                return matches[0]
    return None


def resolve_path(path: Path, *, required: bool, roots: list[Path], warnings: list[str]) -> Path | None:
    if path.exists():
        return path
    found = find_by_name(path.name, roots)
    if found:
        warnings.append(f"Input {path} not found; using recursive match {found}.")
        return found
    if required:
        raise FileNotFoundError(f"Required input file not found: {path}")
    warnings.append(f"Optional input file not found: {path}")
    return None


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if hasattr(value, "item"):
        try:
            return sanitize(value.item())
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def feature_adapter_source() -> str:
    return r'''from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional

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
    for key, meta in RAW_POINT_MAPPING.items():
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


def filter_window(df: pd.DataFrame, end_time: Any, minutes: int) -> pd.DataFrame:
    time_col = infer_time_column(df, "time")
    data = _prepare_time(df, time_col)
    end = pd.to_datetime(end_time)
    start = end - timedelta(minutes=minutes)
    return data.loc[(data[time_col] >= start) & (data[time_col] <= end)].copy()


def safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


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
    ratio = ca.where(flow.notna() & (flow != 0)) / flow.where(flow.notna() & (flow != 0))
    ratio = ratio.dropna()
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


def _find_column(data: pd.DataFrame, friendly: str, explicit: Optional[str] = None) -> Optional[str]:
    if explicit and explicit in data.columns:
        return explicit
    if friendly in data.columns:
        return friendly
    return None


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
) -> Dict[str, Any]:
    data = normalize_columns(df, column_mapping=column_mapping)
    actual_time_col = infer_time_column(data, time_col)
    if actual_time_col != "time":
        data = data.rename(columns={actual_time_col: "time"})
    data = _prepare_time(data, "time")
    if end_time is None:
        end_time = data["time"].max()
    end_time = pd.to_datetime(end_time)
    result: Dict[str, Any] = {"time": end_time.isoformat()}
    missing_raw: List[str] = []
    insufficient: List[str] = []

    for key, meta in RAW_POINT_MAPPING.items():
        if key in {"ca_feed"}:
            continue
        friendly = meta["friendly"]
        output = meta["output"]
        col = _find_column(data, friendly)
        if col is None:
            result[output] = None
            missing_raw.append(friendly)
            continue
        value = calc_window_mean(data, col, end_time, minutes=60, min_valid_points=min_valid_points)
        result[output] = value
        if value is None:
            insufficient.append(output)

    ca_col = _find_column(data, "ca_feed")
    flow_col = _find_column(data, "rubber_flow_2")
    if ca_col is None:
        missing_raw.append("ca_feed")
    if flow_col is None:
        missing_raw.append("rubber_flow_2")
    ca_value = calc_ca_consumption_window_mean(data, ca_col, flow_col, end_time, minutes=60, min_valid_points=min_valid_points) if ca_col and flow_col else None
    result["ca_per_rubber_flow_win_60_mean"] = ca_value
    result["current_ca_consumption"] = ca_value
    if ca_value is None:
        insufficient.append("ca_per_rubber_flow_win_60_mean")

    warning_flags: List[str] = []
    if include_optional_ir:
        ir_col = _find_ir_column(data, column_mapping=column_mapping)
        if ir_col is None:
            result["output_ir_corrected_offset_20_win_15_std"] = None
            warning_flags.append("optional_ir_missing")
        else:
            ir_value = calc_ir_lag_std(data, ir_col, end_time, offset_minutes=20, window_minutes=15, min_valid_points=5)
            result["output_ir_corrected_offset_20_win_15_std"] = ir_value
            if ir_value is None:
                warning_flags.append("optional_ir_insufficient_window")

    if missing_raw:
        warning_flags.append("missing_raw_columns")
    if insufficient:
        warning_flags.append("insufficient_window_features")
    result["feature_quality"] = "ok" if not insufficient else "incomplete"
    result["missing_raw_columns"] = sorted(set(missing_raw))
    result["insufficient_window_features"] = sorted(set(insufficient))
    result["warning_flags"] = ";".join(sorted(set(warning_flags)))
    return result


def build_batch_runtime_features_from_dataframe(
    df: pd.DataFrame,
    evaluation_times: Iterable[Any],
    time_col: str = "time",
    column_mapping: Optional[Dict[str, str]] = None,
    min_valid_points: int = 30,
    include_optional_ir: bool = True,
) -> pd.DataFrame:
    rows = [
        build_runtime_features_from_dataframe(
            df,
            end_time=end_time,
            time_col=time_col,
            column_mapping=column_mapping,
            min_valid_points=min_valid_points,
            include_optional_ir=include_optional_ir,
        )
        for end_time in evaluation_times
    ]
    return pd.DataFrame(rows)
'''


def interface_source() -> str:
    return r'''from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from . import feature_adapter
    from . import package
except Exception:
    import feature_adapter  # type: ignore
    import package  # type: ignore


class SafeBandRecommender:
    def __init__(self, model_dir: Optional[Any] = None, mode: str = "production"):
        self.model_dir = Path(model_dir) if model_dir is not None else Path(__file__).resolve().parent
        self.mode = mode
        self.artifact = None
        self.support = None
        self.schema = None

    def load(self) -> "SafeBandRecommender":
        with (self.model_dir / "safe_band_artifact.json").open("r", encoding="utf-8") as handle:
            self.artifact = json.load(handle)
        with (self.model_dir / "support.json").open("r", encoding="utf-8") as handle:
            self.support = json.load(handle)
        with (self.model_dir / "schema.json").open("r", encoding="utf-8") as handle:
            self.schema = json.load(handle)
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
        if evaluation_times is None:
            state = feature_adapter.build_runtime_features_from_dataframe(
                df,
                end_time=None,
                time_col=time_col,
                column_mapping=column_mapping,
                min_valid_points=min_valid_points,
                include_optional_ir=include_optional_ir,
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
            )
        preds = self.predict_batch(states, mode="production")
        for col in ["feature_quality", "warning_flags", "missing_raw_columns", "insufficient_window_features", "time"]:
            if col in states.columns:
                preds["adapter_" + col] = states[col].values
        return preds


def init(model_dir: Optional[Any] = None, mode: str = "production") -> SafeBandRecommender:
    return SafeBandRecommender(model_dir=model_dir, mode=mode).load()
'''


def main_source() -> str:
    return r'''from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from .interface import SafeBandRecommender
except Exception:
    from interface import SafeBandRecommender  # type: ignore


def parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor-only calcium safe-band MVP runtime example.")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mode", choices=["production", "replay"], default="production")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--input-parquet", type=Path)
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--raw-input-csv", type=Path)
    parser.add_argument("--raw-input-parquet", type=Path)
    parser.add_argument("--raw-time-col", default="time")
    parser.add_argument("--end-time")
    parser.add_argument("--min-valid-points", type=int, default=30)
    parser.add_argument("--include-optional-ir", default="true")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-parquet", type=Path)
    return parser.parse_args()


def read_json_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("rows")
        return rows if isinstance(rows, list) else [payload]
    raise ValueError("JSON input must be an object, list of objects, or {'rows': [...]} structure.")


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_engineered_input(args: argparse.Namespace) -> Any:
    if args.input_json:
        return read_json_rows(args.input_json)
    if args.input_csv:
        return read_csv_rows(args.input_csv)
    if args.input_parquet:
        import pandas as pd
        return pd.read_parquet(args.input_parquet)
    raise ValueError("Provide engineered input or raw input.")


def read_raw_input(args: argparse.Namespace) -> Any:
    import pandas as pd
    if args.raw_input_csv:
        return pd.read_csv(args.raw_input_csv)
    if args.raw_input_parquet:
        return pd.read_parquet(args.raw_input_parquet)
    raise ValueError("Provide --raw-input-csv or --raw-input-parquet.")


def write_outputs(result: Any, args: argparse.Namespace) -> None:
    rows = result.to_dict(orient="records") if hasattr(result, "to_dict") else result
    if isinstance(rows, dict):
        rows = [rows]
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
        with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    if args.output_parquet:
        import pandas as pd
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        frame = result if hasattr(result, "to_parquet") else pd.DataFrame(rows)
        frame.to_parquet(args.output_parquet, index=False)


def main() -> None:
    # Plant DCS fetch logic should be implemented by the plant adapter owner.
    # Plant writeback logic should be implemented by the plant adapter owner.
    # Current script does not write DCS and does not perform automatic control.
    args = parse_args()
    recommender = SafeBandRecommender(args.model_dir, mode=args.mode).load()
    if args.raw_input_csv or args.raw_input_parquet:
        raw_df = read_raw_input(args)
        if args.end_time:
            result = recommender.predict_from_raw_dataframe(
                raw_df,
                end_time=args.end_time,
                time_col=args.raw_time_col,
                min_valid_points=args.min_valid_points,
                include_optional_ir=parse_bool(args.include_optional_ir),
            )
        else:
            result = recommender.predict_batch_from_raw_dataframe(
                raw_df,
                evaluation_times=None,
                time_col=args.raw_time_col,
                min_valid_points=args.min_valid_points,
                include_optional_ir=parse_bool(args.include_optional_ir),
            )
    else:
        input_data = read_engineered_input(args)
        result = recommender.predict_batch(input_data, mode=args.mode)
    write_outputs(result, args)
    count = len(result) if hasattr(result, "__len__") else 0
    print("Scored rows: {}".format(count))
    print("Mode: {}; monitor-only; no DCS writeback; no automatic control.".format(args.mode))


if __name__ == "__main__":
    main()
'''


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")


def update_schema_and_support(deploy_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    schema_path = deploy_dir / "schema.json"
    support_path = deploy_dir / "support.json"
    schema = load_json(schema_path)
    support = load_json(support_path)
    schema["runtime_input_modes"] = ["engineered_feature_state", "raw_platform_dataframe"]
    schema["feature_window_definitions"] = {
        "process_context_window_minutes": 60,
        "ca_consumption_window_minutes": 60,
        "ir_lag_offset_minutes": 20,
        "ir_lag_window_minutes": 15,
        "online_runtime_note": "Online runtime uses trailing windows ending at current time, not an additional 165min historical shift.",
    }
    schema["raw_point_mapping"] = RAW_POINT_MAPPING
    optional = set(schema.get("optional_features", []))
    optional.add("output_ir_corrected_offset_20_win_15_std")
    schema["optional_features"] = sorted(optional)
    schema["required_features"] = [f for f in schema.get("required_features", []) if f != "output_ir_corrected_offset_20_win_15_std"]
    features = support.setdefault("features", {})
    if "output_ir_corrected_offset_20_win_15_std" in features:
        features["output_ir_corrected_offset_20_win_15_std"]["required"] = False
        features["output_ir_corrected_offset_20_win_15_std"]["feature_role"] = "diagnostic"
    support["raw_point_mapping"] = RAW_POINT_MAPPING
    support["feature_window_definitions"] = schema["feature_window_definitions"]
    write_json(schema_path, schema)
    write_json(support_path, support)
    return schema, support


def detect_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.module in {None, "package", "interface", "feature_adapter"}:
                continue
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def dependency_check(requirements: Path, deploy_dir: Path, available: set[str]) -> dict[str, Any]:
    files = [deploy_dir / n for n in ["package.py", "interface.py", "main.py", "feature_adapter.py"]]
    third_party = {}
    package_third = []
    for path in files:
        for imp in sorted(detect_imports(path)):
            if imp in STANDARD_IMPORTS or imp in {"package", "interface", "feature_adapter"}:
                continue
            third_party.setdefault(imp, []).append(str(path))
            if path.name == "package.py":
                package_third.append(imp)
    missing = sorted({imp for imp in third_party if normalize_package_name(IMPORT_TO_PACKAGE.get(imp, imp)) not in available})
    return {
        "requirements_path": str(requirements),
        "deploy_files_checked": [str(p) for p in files],
        "third_party_imports_detected": third_party,
        "imports_not_in_requirements": missing,
        "package_py_standard_library_only": not package_third,
        "package_py_third_party_imports": sorted(package_third),
        "dependency_policy_pass": not missing and not package_third,
        "warnings": [],
    }


def synthetic_raw_dataframe(deploy_dir: Path, dry_run: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    support = load_json(deploy_dir / "support.json")
    end = pd.Timestamp("2026-01-01 12:00:00")
    times = pd.date_range(end - pd.Timedelta(minutes=70), end, freq="1min")
    row = dry_run.dropna(subset=["current_ca_consumption"]).iloc[0]
    flow = float(row.get("rubber_flow_2_win_60_mean", 48000) or 48000)
    current = float(row["current_ca_consumption"])
    data = pd.DataFrame({"time": times})
    missing_raw_warning = ["synthetic_raw_like_dataframe_used_because_true_raw_dcs_sample_unavailable"]
    defaults = {
        "rubber_flow_2": flow,
        "ca_feed": current * flow,
        "output_ir_corrected": 1.0,
    }
    for key, meta in RAW_POINT_MAPPING.items():
        friendly = meta["friendly"]
        if friendly in defaults:
            data[friendly] = defaults[friendly]
        else:
            output = meta["output"]
            q66 = support.get("features", {}).get(output, {}).get("q66")
            q33 = support.get("features", {}).get(output, {}).get("q33")
            data[friendly] = float(q66 if q66 is not None else q33 if q33 is not None else 1.0) + 0.01
    data["output_ir_corrected"] = [1.0 + (i % 7) * 0.001 for i in range(len(data))]
    return data, missing_raw_warning


def run_smoke_tests(deploy_dir: Path, dry_run: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("adapter_interface", deploy_dir / "interface.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(deploy_dir))
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        try:
            sys.path.remove(str(deploy_dir))
        except ValueError:
            pass
    rec = module.SafeBandRecommender(deploy_dir, mode="production").load()
    engineered = dry_run.iloc[0].to_dict()
    engineered_result = rec.predict_one(engineered, mode="replay")
    raw_df, raw_warnings = synthetic_raw_dataframe(deploy_dir, dry_run)
    raw_result = rec.predict_from_raw_dataframe(raw_df, min_valid_points=30, include_optional_ir=True)
    raw_input = output_dir / "synthetic_raw_input.parquet"
    raw_df.to_parquet(raw_input, index=False)
    raw_cli_csv = output_dir / "raw_cli_smoke_output.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(deploy_dir / "main.py"),
            "--model-dir",
            str(deploy_dir),
            "--raw-input-parquet",
            str(raw_input),
            "--output-csv",
            str(raw_cli_csv),
            "--mode",
            "production",
        ],
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
    )
    rows = [
        {"test_name": "engineered_predict_one", "passed": bool(engineered_result.get("action_visibility")), "details": engineered_result.get("action_visibility")},
        {"test_name": "raw_dataframe_predict", "passed": bool(raw_result.get("action_visibility")), "details": raw_result.get("adapter_feature_quality")},
        {"test_name": "main_cli_raw_dataframe", "passed": completed.returncode == 0 and raw_cli_csv.exists(), "details": completed.stdout.strip() or completed.stderr.strip()},
    ]
    smoke = pd.DataFrame(rows)
    report = {
        "engineered_predict_smoke_pass": bool(rows[0]["passed"]),
        "raw_dataframe_smoke_pass": bool(rows[1]["passed"]),
        "main_cli_raw_smoke_pass": bool(rows[2]["passed"]),
        "missing_raw_columns_in_smoke": raw_result.get("adapter_missing_raw_columns"),
        "warnings": raw_warnings,
    }
    return smoke, report, raw_df


def append_doc(path: Path, report: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    section_no = 29
    while f"## {section_no}." in existing:
        section_no += 1
    section = f"""

## {section_no}. 运行包实时特征适配器与 IR 可选输入支持

### {section_no}.1 阶段目的

此前运行包默认接收工程化特征。厂内集成通常拿到的是带时间戳的原始平台 DataFrame，因此本阶段新增 `feature_adapter.py`，将原始点位列转换为运行包所需的当前特征状态。

### {section_no}.2 在线窗口策略

在线运行使用当前时刻 `t_now` 之前的尾随窗口：工况变量采用 `[t_now-60min, t_now]` 的均值；钙单耗采用该窗口内 `ca_feed / rubber_flow_2` 的均值。离线标签对齐曾使用停留时间；在线推荐不再额外向前平移 165min，因为当前上游操作影响的是未来产品质量。后续 LIMS 回填验证应按停留时间把当前输出与未来 T90 标签比较。

### {section_no}.3 IR-lag

IR-lag `output_ir_corrected_offset_20_win_15_std` 为可选输入。若存在原始 IR，则计算 `[t_now-35min, t_now-20min]` 的 15 分钟标准差；若缺失，不阻断推荐，只记录 `optional_ir_missing`。

### {section_no}.4 接口更新

- `interface.py` 新增 `predict_from_raw_dataframe` 和 `predict_batch_from_raw_dataframe`。
- `main.py` 新增 `--raw-input-csv`、`--raw-input-parquet`、`--raw-time-col`、`--end-time`、`--min-valid-points` 和 `--include-optional-ir`。
- `schema.json/support.json` 增加原始点位映射和窗口定义。

### {section_no}.5 烟测结果

- engineered predict_one：{report.get('engineered_predict_smoke_pass')}
- raw dataframe predict：{report.get('raw_dataframe_smoke_pass')}
- main.py raw CLI：{report.get('main_cli_raw_smoke_pass')}
- 依赖策略：{report.get('dependency_policy_pass')}
- IR 可选确认：{report.get('ir_optional_confirmed')}

推荐下一步：`{report.get('recommended_next_step')}`。

局限性：本阶段使用合成 raw-like 数据验证接口路径，仍需厂方提供真实原始平台 DataFrame 做最终适配器验收；无 DCS 写回；无自动控制。
"""
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(section)


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requirements = resolve_path(args.requirements, required=True, roots=[Path(".")], warnings=warnings)
    deploy_dir = resolve_path(args.deploy_dir, required=True, roots=[Path("deploy")], warnings=warnings)
    dry_run_path = resolve_path(args.final_dry_run, required=True, roots=[Path("runs")], warnings=warnings)
    available = parse_requirements(requirements)
    if "pandas" not in available:
        raise RuntimeError("pandas is required for feature_adapter.py but is absent from IDB_requirements.txt.")
    dry_run = pd.read_parquet(dry_run_path)

    (deploy_dir / "feature_adapter.py").write_text(feature_adapter_source(), encoding="utf-8")
    (deploy_dir / "interface.py").write_text(interface_source(), encoding="utf-8")
    (deploy_dir / "main.py").write_text(main_source(), encoding="utf-8")
    schema, support = update_schema_and_support(deploy_dir)

    dep = dependency_check(requirements, deploy_dir, available)
    smoke, smoke_report, raw_df = run_smoke_tests(deploy_dir, dry_run, args.output_dir)
    smoke.to_csv(args.output_dir / "feature_adapter_smoke_test.csv", index=False, encoding="utf-8-sig")
    report = {
        "feature_adapter_created": True,
        "interface_updated": True,
        "main_updated": True,
        "schema_updated": True,
        "support_updated": True,
        "ir_optional_confirmed": not support.get("features", {}).get("output_ir_corrected_offset_20_win_15_std", {}).get("required", True),
        "package_py_standard_library_only": dep["package_py_standard_library_only"],
        "dependency_policy_pass": dep["dependency_policy_pass"],
        **smoke_report,
        "warnings": warnings + smoke_report.get("warnings", []),
    }
    if dep["dependency_policy_pass"] and report["engineered_predict_smoke_pass"] and report["raw_dataframe_smoke_pass"] and report["main_cli_raw_smoke_pass"]:
        next_step = "human_review_feature_adapter_contract"
    else:
        next_step = "fix_feature_adapter"
    report["recommended_next_step"] = next_step
    write_json(args.output_dir / "feature_adapter_smoke_test_report.json", smoke_report)
    write_json(args.output_dir / "dependency_check_feature_adapter.json", dep)
    write_json(args.output_dir / "feature_adapter_update_report.json", report)
    append_doc(args.experiment_doc, report)

    print("Runtime feature adapter update summary")
    print("feature_adapter_created: True")
    print("interface_updated: True")
    print("main_updated: True")
    print(f"ir_optional_confirmed: {report['ir_optional_confirmed']}")
    print(f"package_py_standard_library_only: {report['package_py_standard_library_only']}")
    print(f"dependency_policy_pass: {report['dependency_policy_pass']}")
    print(f"engineered_predict_smoke_pass: {report['engineered_predict_smoke_pass']}")
    print(f"raw_dataframe_smoke_pass: {report['raw_dataframe_smoke_pass']}")
    print(f"main_cli_raw_smoke_pass: {report['main_cli_raw_smoke_pass']}")
    print(f"missing_raw_columns_in_smoke: {report['missing_raw_columns_in_smoke']}")
    print(f"recommended_next_step: {next_step}")
    print(f"Documentation appended: {args.experiment_doc}")
    print("No generated outputs were written under data/.")


if __name__ == "__main__":
    main()
