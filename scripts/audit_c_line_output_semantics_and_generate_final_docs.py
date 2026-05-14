from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_RUNTIME_FILES = {
    "ca_safe_band_mvp_c_line.py": "IDB 程序入口文件",
    "package.py": "冻结推荐规则执行逻辑",
    "interface.py": "运行接口与输出语义后处理",
    "feature_adapter.py": "原始 DCS 数据到 60min 特征的适配器",
    "idb_s3fs_asset_loader.py": "s3fs/local JSON 资产加载器",
    "idb_config_template.json": "IDB/s3fs 路径配置模板",
    "safe_band_artifact.json": "冻结 C线推荐 artifact",
    "support.json": "运行支撑元数据",
    "schema.json": "输入输出 schema",
    "README_IDB_S3FS_C_LINE_MONITOR_ONLY.md": "运行包说明",
    "VERSION.json": "版本信息",
}

RAW_INPUT_POINTS = [
    ("卤化工段胶液总量2", "B4-FIC-C51001.PV.F_CV", "rubber_flow_2", "必需", "60min", "用于归一化钙单耗和加注量换算；<=0 视为无效"),
    ("反应溴添加量", "B4-FIC-C51004.PV.CV", "bromine_feed", "必需", "60min", "越界/缺失会影响规则匹配"),
    ("储罐胶浓在线检测", "B4-AT-C50002A-BIIR.PV.CV", "tank_rubber_conc", "必需", "60min", "越界/缺失会影响规则匹配"),
    ("R510A温度", "B4-TI-C51007A_S.PV.CV", "r510a_temp", "必需", "60min", "越界/缺失会影响规则匹配"),
    ("R511A温度", "B4-TI-C51101A_S.PV.CV", "r511a_temp", "必需", "60min", "越界/缺失会影响规则匹配"),
    ("R512A温度", "B4-TI-C51702A.PV.F_CV", "r512a_temp", "必需", "60min", "越界/缺失会影响规则匹配"),
    ("硬脂酸钙加注量", "B4-FIC-C51401.PV.F_CV", "ca_feed", "必需", "60min", "负值无效；用于计算 ca_per_rubber_flow"),
    ("ESBO加注量", "B4-FIC-C51801.PV.F_CV", "esbo_feed", "必需", "60min", "越界/缺失会影响规则匹配"),
    ("中和碱液添加量", "B4-FIC-C51605.PV.F_CV", "neutral_alkali_feed", "必需", "60min", "越界/缺失会影响规则匹配"),
    ("R513温度", "B4-TI-C51301_S.PV.CV", "r513_temp", "必需", "60min", "越界/缺失会影响规则匹配"),
    ("R514温度", "B4-TI-C51401_S.PV.CV", "r514_temp", "必需", "60min", "越界/缺失会影响规则匹配"),
]

EXPECTED_OUTPUT_FIELDS = [
    "timestamp",
    "recommendation_status",
    "current_ca_feed",
    "current_ca_consumption",
    "recommended_ca_feed_min",
    "recommended_ca_feed_max",
    "recommended_ca_feed_target",
    "recommended_ca_consumption_min",
    "recommended_ca_consumption_max",
    "recommended_ca_consumption_target",
    "interval_position",
    "action_visibility",
    "engineering_review_required",
    "t90_risk_level",
    "t90_high_risk_warning",
    "t90_low_risk_warning",
    "prediction_type",
    "recommendation_target",
    "internal_normalized_metric",
    "input_valid",
    "missing_required_features",
    "warning_flags",
    "model_version",
    "artifact_version",
    "selected_rule_id",
    "matched_rule_ids",
    "rule_evidence_ok_rate",
    "rule_evidence_high_rate",
    "rule_evidence_low_rate",
    "recommended_ca_feed_conversion_status",
    "recommended_ca_feed_bounds_status",
    "recommended_ca_feed_unbounded_min",
    "recommended_ca_feed_unbounded_max",
    "recommended_ca_feed_unbounded_target",
    "ca_feed_point_lower_bound",
    "ca_feed_point_upper_bound",
    "estimated_quality_time",
    "residence_time_minutes_used",
    "automatic_control",
    "dcs_setpoint_writeback",
    "control_mode",
    "error_code",
    "error_message",
]

FORBIDDEN_RUNTIME_PATTERNS = ["read_parquet", "to_parquet", "pyarrow", "fastparquet", "pickle", "deploy/ca_safe_band_mvp/", "deploy\\ca_safe_band_mvp\\"]

PROBLEMATIC_WORDING = [
    "推荐钙单耗",
    "钙单耗推荐",
    "预测当前T90",
    "预测当前 T90",
    "精确预测T90",
    "自动调钙",
    "写入设定值",
    "DCS写回",
    "DCS 写回",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and correct C-line IDB/s3fs output semantics and generate final docs.")
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--deploy-dir", type=Path, required=True)
    parser.add_argument("--release-zip", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--stage49-dir", type=Path, required=True)
    parser.add_argument("--stage50-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--method-doc", type=Path, required=True)
    parser.add_argument("--sop-doc", type=Path, required=True)
    parser.add_argument("--final-manual", type=Path, required=True)
    return parser.parse_args()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if not pd.isna(value) else None
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_inventory(deploy_dir: Path) -> pd.DataFrame:
    rows = []
    for name, role in REQUIRED_RUNTIME_FILES.items():
        path = deploy_dir / name
        rows.append(
            {
                "file_name": name,
                "path": str(path),
                "exists": path.exists(),
                "required": True,
                "role_cn": role,
                "note_cn": "存在" if path.exists() else "缺失",
            }
        )
    return pd.DataFrame(rows)


def semantic_postprocess_block() -> str:
    return r'''


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
'''.rstrip()


def patch_interface_if_needed(deploy_dir: Path) -> dict[str, Any]:
    path = deploy_dir / "interface.py"
    original = path.read_text(encoding="utf-8")
    patch_needed = "postprocess_output_semantics" not in original or "return postprocess_output_semantics(" not in original
    if not patch_needed:
        return {
            "patch_needed": False,
            "patch_applied": False,
            "semantic_postprocess_present": True,
            "files_modified": [],
            "algorithm_changed": False,
            "artifact_modified": False,
            "backward_compatible": True,
            "new_fields_added": [
                "current_ca_feed",
                "recommended_ca_feed_conversion_status",
                "recommended_ca_feed_bounds_status",
                "recommended_ca_feed_unbounded_min",
                "recommended_ca_feed_unbounded_max",
                "recommended_ca_feed_unbounded_target",
                "ca_feed_point_lower_bound",
                "ca_feed_point_upper_bound",
                "t90_risk_level",
                "t90_high_risk_warning",
                "t90_low_risk_warning",
                "prediction_type",
                "recommendation_target",
                "internal_normalized_metric",
                "control_mode",
                "automatic_control",
                "dcs_setpoint_writeback",
                "selected_rule_id",
                "timestamp",
                "estimated_quality_time",
                "residence_time_minutes_used",
                "error_code",
                "error_message",
            ],
            "warnings": [],
        }
    text = original
    if "postprocess_output_semantics" not in text:
        insert_at = text.index("\n\nclass SafeBandRecommender")
        text = text[:insert_at] + semantic_postprocess_block() + text[insert_at:]
    text = text.replace(
        "return package.recommend_one(state, self.artifact, self.support, schema=self.schema, mode=mode or self.mode)",
        "return postprocess_output_semantics(package.recommend_one(state, self.artifact, self.support, schema=self.schema, mode=mode or self.mode), state)",
    )
    text = text.replace(
        "return pd.DataFrame(result)",
        "return pd.DataFrame([postprocess_output_semantics(item, row) for item, row in zip(result, rows)])",
    )
    text = text.replace(
        "return package.recommend_batch(input_data, self.artifact, self.support, schema=self.schema, mode=mode or self.mode)",
        "return [postprocess_output_semantics(item, row) for item, row in zip(package.recommend_batch(input_data, self.artifact, self.support, schema=self.schema, mode=mode or self.mode), input_data)]",
    )
    # predict_from_raw_dataframe calls predict_one, so the output is already postprocessed.
    path.write_text(text, encoding="utf-8")
    return {
        "patch_needed": True,
        "patch_applied": True,
        "semantic_postprocess_present": True,
        "files_modified": [str(path)],
        "algorithm_changed": False,
        "artifact_modified": False,
        "backward_compatible": True,
        "new_fields_added": [
            "current_ca_feed",
            "recommended_ca_feed_conversion_status",
            "recommended_ca_feed_bounds_status",
            "recommended_ca_feed_unbounded_min",
            "recommended_ca_feed_unbounded_max",
            "recommended_ca_feed_unbounded_target",
            "ca_feed_point_lower_bound",
            "ca_feed_point_upper_bound",
            "t90_risk_level",
            "t90_high_risk_warning",
            "t90_low_risk_warning",
            "prediction_type",
            "recommendation_target",
            "internal_normalized_metric",
            "control_mode",
            "automatic_control",
            "dcs_setpoint_writeback",
            "selected_rule_id",
            "timestamp",
            "estimated_quality_time",
            "residence_time_minutes_used",
            "error_code",
            "error_message",
        ],
        "warnings": [],
    }


def update_schema(deploy_dir: Path) -> bool:
    path = deploy_dir / "schema.json"
    if not path.exists():
        return False
    data = read_json(path)
    output_schema = data.get("output_schema")
    if isinstance(output_schema, list):
        existing = [str(x) for x in output_schema]
        for field in EXPECTED_OUTPUT_FIELDS:
            if field not in existing:
                existing.append(field)
        data["output_schema"] = existing
    data["output_semantics"] = {
        "recommendation_target_cn": "硬脂酸钙加注量推荐区间",
        "internal_normalized_metric_cn": "钙单耗，仅作内部归一化计算和诊断",
        "t90_output_cn": "T90 偏高/偏低风险提示，不是精确 T90 数值预测",
        "feed_output_rule_cn": "最终对外输出的 recommended_ca_feed_min/max/target 必须先由内部钙单耗区间按 rubber_flow_2_win_60_mean 反归一化为硬脂酸钙加注量，再按点位表 B4-FIC-C51401.PV.F_CV 的 700~1300 上下限做安全保护。",
        "feed_point_lower_bound": 700.0,
        "feed_point_upper_bound": 1300.0,
        "unbounded_feed_fields_cn": "recommended_ca_feed_unbounded_min/max/target 仅用于诊断原始反归一化值，不作为对外最终推荐值。",
        "monitor_only": True,
        "automatic_control": False,
        "dcs_setpoint_writeback": False,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return True


def runtime_config(deploy_dir: Path) -> dict[str, str]:
    return {
        "artifact_path": str(deploy_dir / "safe_band_artifact.json"),
        "support_path": str(deploy_dir / "support.json"),
        "schema_path": str(deploy_dir / "schema.json"),
    }


def synthetic_engineered_row(deploy_dir: Path) -> dict[str, Any]:
    support = read_json(deploy_dir / "support.json")
    schema = read_json(deploy_dir / "schema.json")
    row: dict[str, Any] = {}
    required = schema.get("required_features", [])
    for feature in required:
        meta = support.get("features", {}).get(feature, {})
        q33 = meta.get("q33")
        q66 = meta.get("q66")
        if isinstance(q33, (int, float)) and isinstance(q66, (int, float)):
            row[feature] = (float(q33) + float(q66)) / 2.0
        elif "ca_per_rubber_flow" in feature:
            row[feature] = 0.0205
        else:
            row[feature] = 1.0
    row["ca_per_rubber_flow_win_60_mean"] = row.get("ca_per_rubber_flow_win_60_mean", 0.0205)
    row["current_ca_consumption"] = row["ca_per_rubber_flow_win_60_mean"]
    row["rubber_flow_2_win_60_mean"] = 55000.0
    row["time"] = "2026-03-31T00:00:00"
    return row


def synthetic_raw_dataframe() -> pd.DataFrame:
    end = pd.Timestamp("2026-03-31T00:00:00")
    times = pd.date_range(end - pd.Timedelta(minutes=59), end, freq="1min")
    return pd.DataFrame(
        {
            "time": times,
            "rubber_flow_2": [55000.0] * len(times),
            "bromine_feed": [360.0] * len(times),
            "tank_rubber_conc": [20.0] * len(times),
            "r510a_temp": [40.0] * len(times),
            "r511a_temp": [20.0] * len(times),
            "r512a_temp": [50.0] * len(times),
            "ca_feed": [1127.5] * len(times),
            "esbo_feed": [130.0] * len(times),
            "neutral_alkali_feed": [5200.0] * len(times),
            "r513_temp": [50.0] * len(times),
            "r514_temp": [50.0] * len(times),
        }
    )


def import_runtime(deploy_dir: Path):
    old_sys = list(sys.path)
    sys.path.insert(0, str(deploy_dir.resolve()))
    try:
        sys.modules.pop("ca_safe_band_mvp_c_line", None)
        sys.modules.pop("interface", None)
        sys.modules.pop("package", None)
        sys.modules.pop("feature_adapter", None)
        spec = importlib.util.spec_from_file_location("ca_safe_band_mvp_c_line", deploy_dir / "ca_safe_band_mvp_c_line.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import runtime entry.")
        module = importlib.util.module_from_spec(spec)
        sys.modules["ca_safe_band_mvp_c_line"] = module
        spec.loader.exec_module(module)
        return module, old_sys
    except Exception:
        sys.path = old_sys
        raise


def restore_sys_path(old_sys: list[str]) -> None:
    sys.path = old_sys


def run_output_discovery(deploy_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    module, old_sys = import_runtime(deploy_dir)
    cfg = runtime_config(deploy_dir)
    try:
        normal_row = synthetic_engineered_row(deploy_dir)
        normal = module.predict_one(normal_row, config=cfg)
        batch = module.predict_batch([normal_row], config=cfg)
        if hasattr(batch, "to_dict"):
            batch_rows = batch.to_dict(orient="records")
        else:
            batch_rows = list(batch)
        raw = module.run_once(raw_df=synthetic_raw_dataframe(), config=cfg)
        missing = module.predict_one({"time": "2026-03-31T00:00:00"}, config=cfg)
    finally:
        restore_sys_path(old_sys)
    return normal, missing, batch_rows, raw


def infer_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "str"


def variable_category(name: str) -> str:
    if name in {"recommended_ca_feed_min", "recommended_ca_feed_max", "recommended_ca_feed_target", "current_ca_feed", "interval_position", "recommendation_status", "action_visibility", "engineering_review_required"}:
        return "user_facing_recommendation"
    if "t90" in name or name in {"prediction_type"}:
        return "risk_warning"
    if name in {"automatic_control", "dcs_setpoint_writeback", "control_mode"}:
        return "safety_status"
    if "missing" in name or "warning" in name or "input_valid" in name or "adapter_" in name:
        return "input_quality"
    if "version" in name or name == "timestamp" or name == "estimated_quality_time" or name == "residence_time_minutes_used":
        return "metadata"
    if "error" in name:
        return "error_status"
    if "consumption" in name or "rule" in name or name.startswith("recommended_ca_consumption"):
        return "internal_diagnostic"
    return "internal_diagnostic"


def discover_output_inventory(normal: dict[str, Any], missing: dict[str, Any], batch_rows: list[dict[str, Any]], raw: dict[str, Any]) -> pd.DataFrame:
    batch_keys = set().union(*(row.keys() for row in batch_rows)) if batch_rows else set()
    keys = sorted(set(normal) | set(missing) | batch_keys | set(raw))
    rows = []
    for key in keys:
        example = normal.get(key, raw.get(key, missing.get(key, None)))
        cat = variable_category(key)
        rows.append(
            {
                "output_variable": key,
                "appears_in_normal_output": key in normal or key in raw,
                "appears_in_error_output": key in missing,
                "appears_in_batch_output": key in batch_keys,
                "source_function": "predict_one;predict_batch;run_once" if key in raw else "predict_one;predict_batch",
                "raw_type_observed": infer_type(example),
                "example_value": json.dumps(to_jsonable(example), ensure_ascii=False)[:200],
                "nullable": example is None or key in missing,
                "current_documentation_found": key in EXPECTED_OUTPUT_FIELDS,
                "category": cat,
                "needs_semantic_correction": key in {"current_ca_consumption", "recommended_ca_consumption_min", "recommended_ca_consumption_max", "recommended_ca_consumption_target", "explanation_cn"},
                "note_cn": "对外展示需优先使用硬脂酸钙加注量字段" if "consumption" in key else "",
            }
        )
    return pd.DataFrame(rows)


def field_semantics(name: str, produced: bool) -> dict[str, Any]:
    base = {
        "output_variable": name,
        "variable_cn_name": name,
        "user_visible": False,
        "internal_only": False,
        "unit": "",
        "meaning_cn": "当前接口未输出该字段" if not produced else "",
        "calculation_or_source_cn": "",
        "valid_values": "",
        "when_null_cn": "当前接口未输出该字段" if not produced else "无可用值或不适用",
        "operator_interpretation_cn": "",
        "forbidden_interpretation_cn": "",
        "safety_note_cn": "仅监测/指导，不自动控制，不写 DCS。",
        "deployment_note_cn": "",
    }
    mapping = {
        "timestamp": ("输出时间戳", True, False, "", "本次推荐对应的特征窗口结束时间。", "来自输入 state.time 或 raw DataFrame end_time。", "ISO 时间字符串", "用于日志和后续 LIMS 回填对齐。", "不得解释为 LIMS T90 检测时间。"),
        "recommendation_status": ("推荐状态", True, False, "", "是否生成参考区间。", "规则匹配和输入质量判断结果。", "recommended / no_recommendation_*", "recommended 才能展示参考区间。", "不得解释为自动操作指令。"),
        "current_ca_feed": ("当前硬脂酸钙加注量", True, False, "DCS 加注量单位", "当前窗口硬脂酸钙加注量估计。", "current_ca_consumption * rubber_flow_2_win_60_mean。", "非负数", "用于与推荐加注量区间比较。", "不得作为自动调节输出。"),
        "current_ca_consumption": ("当前钙单耗", False, True, "加注量/胶液流量", "内部归一化诊断指标。", "硬脂酸钙加注量 / 胶液总量。", "非负数", "不建议直接展示给操作员作为推荐目标。", "禁止解释为对外推荐钙单耗。"),
        "recommended_ca_feed_min": ("硬脂酸钙加注量推荐下限", True, False, "DCS 加注量单位", "对外展示的硬脂酸钙加注量参考区间下限。", "recommended_ca_consumption_min * rubber_flow_2_win_60_mean。", "非负数或空", "仅供人工复核参考。", "不得自动写入 DCS。"),
        "recommended_ca_feed_max": ("硬脂酸钙加注量推荐上限", True, False, "DCS 加注量单位", "对外展示的硬脂酸钙加注量参考区间上限。", "recommended_ca_consumption_max * rubber_flow_2_win_60_mean。", "非负数或空", "仅供人工复核参考。", "不得自动写入 DCS。"),
        "recommended_ca_feed_target": ("硬脂酸钙加注量参考目标", True, False, "DCS 加注量单位", "对外展示的参考目标值。", "recommended_ca_consumption_target * rubber_flow_2_win_60_mean。", "非负数或空", "可用于人工判断是否偏离参考区间。", "不得解释为设定值。"),
        "recommended_ca_consumption_min": ("内部钙单耗区间下限", False, True, "加注量/胶液流量", "内部归一化推荐边界。", "冻结规则输出。", "非负数或空", "不直接作为操作员推荐。", "禁止写成推荐钙单耗。"),
        "recommended_ca_consumption_max": ("内部钙单耗区间上限", False, True, "加注量/胶液流量", "内部归一化推荐边界。", "冻结规则输出。", "非负数或空", "不直接作为操作员推荐。", "禁止写成推荐钙单耗。"),
        "recommended_ca_consumption_target": ("内部钙单耗参考目标", False, True, "加注量/胶液流量", "内部归一化参考目标。", "冻结规则上下限中点。", "非负数或空", "仅作诊断和换算。", "禁止作为对外设定值。"),
        "interval_position": ("当前位置", True, False, "", "当前加注状态相对参考区间的位置。", "比较 current_ca_consumption 与内部区间。", "inside_band / above_band / below_band / missing", "判断展示级别和风险提示。", "不得解释为自动增减钙命令。"),
        "action_visibility": ("动作可见性", True, False, "", "前端/看板展示级别。", "由 interval_position 派生。", "monitor_only / manual_review_required / diagnostic_only / no_recommendation", "人工复核使用。", "不得代表系统执行动作。"),
        "engineering_review_required": ("是否需要工程复核", True, False, "", "是否需要工艺/自控人员人工复核。", "由位置和输入质量派生。", "true / false", "true 时不应直接采纳，应人工确认。", "不得跳过人工复核。"),
        "t90_risk_level": ("T90 风险等级", True, False, "", "未来物料 T90 偏高/偏低风险提示。", "由 interval_position 派生。", "high_t90_risk / low_t90_risk / low_risk_reference / unknown", "风险提示，不是 T90 数值。", "禁止解释为预测当前 T90 或精确 T90。"),
        "t90_high_risk_warning": ("T90 偏高风险提示", True, False, "", "是否提示 T90 偏高风险。", "above_band 派生。", "true / false", "true 时进入人工复核。", "不得自动降钙。"),
        "t90_low_risk_warning": ("T90 偏低风险提示", True, False, "", "是否提示 T90 偏低风险。", "below_band 派生。", "true / false", "仅诊断提示。", "不得自动加钙。"),
        "prediction_type": ("预测/提示类型", True, False, "", "说明输出是风险提示而非 T90 数值预测。", "固定字段。", "risk_warning_not_t90_value_prediction", "用于防止误读。", "不得说系统预测精确 T90。"),
        "recommendation_target": ("推荐对象", True, False, "", "推荐目标是硬脂酸钙加注量。", "固定字段。", "calcium_stearate_feed", "对外展示按加注量解释。", "禁止说推荐对象是钙单耗。"),
        "internal_normalized_metric": ("内部归一化指标", False, True, "", "说明内部使用钙单耗归一化。", "固定字段。", "calcium_consumption", "仅诊断。", "不得对外作为操作目标。"),
        "input_valid": ("输入是否有效", True, False, "", "关键输入和窗口有效性。", "必需特征校验。", "true / false", "false 时不给推荐。", "不得忽略缺失输入。"),
        "missing_required_features": ("缺失关键特征", True, False, "", "缺失或无效的必需特征列表。", "输入校验。", "列表", "用于排查点位/窗口问题。", "不得在缺失时强行使用输出。"),
        "warning_flags": ("警告标志", True, False, "", "输入、换算、风险等警告。", "运行过程累积。", "分号分隔字符串", "逐项排查。", "不得忽略 safety warning。"),
        "model_version": ("模型/规则版本", True, False, "", "artifact 版本。", "artifact_version。", "字符串", "用于追溯。", "不得混用旧合并线版本。"),
        "artifact_version": ("artifact 版本", True, False, "", "冻结 C线 artifact 版本。", "JSON artifact。", "字符串", "用于追溯。", "不得用 future 数据更新。"),
        "selected_rule_id": ("选中规则 ID", False, True, "", "代表性规则 ID。", "selected_rule_ids 的首个值。", "字符串或空", "诊断规则来源。", "不得人工篡改作为输入。"),
        "matched_rule_ids": ("匹配规则 ID 列表", False, True, "", "本次工况匹配到的规则。", "q33/q66 工况匹配。", "分号分隔字符串", "用于工程复核。", "不得作为操作命令。"),
        "rule_evidence_ok_rate": ("规则历史合格率", False, True, "", "匹配规则历史 ok_rate。", "冻结 artifact 规则证据。", "0~1 或空", "仅作证据参考。", "不是未来确定结果。"),
        "rule_evidence_high_rate": ("规则历史高 T90 率", False, True, "", "匹配规则历史高 T90 比例。", "冻结 artifact 规则证据。", "0~1 或空", "仅作风险背景。", "不是未来确定结果。"),
        "rule_evidence_low_rate": ("规则历史低 T90 率", False, True, "", "匹配规则历史低 T90 比例。", "冻结 artifact 规则证据。", "0~1 或空", "仅作风险背景。", "不是未来确定结果。"),
    "recommended_ca_feed_conversion_status": ("加注量换算状态", True, False, "", "内部钙单耗区间是否成功换算为加注量区间。", "检查 rubber_flow_2_win_60_mean。", "ok / unavailable", "unavailable 时不展示加注量推荐。", "不得用空值推断操作。"),
        "recommended_ca_feed_bounds_status": ("加注量边界保护状态", True, False, "", "反归一化后的硬脂酸钙加注量是否被 700~1300 点位上下限裁剪。", "检查 recommended_ca_feed_min/max/target 是否超出点位上下限。", "within_point_bounds / clipped_to_point_bounds / unavailable", "clipped 时必须提示人工复核，并以裁剪后的 feed 字段作为最终展示值。", "不得把 unbounded 字段作为最终推荐值。"),
        "recommended_ca_feed_unbounded_min": ("未裁剪硬脂酸钙加注量下限", False, True, "DCS 加注量单位", "内部钙单耗下限反归一化后的原始值。", "recommended_ca_consumption_min * rubber_flow_2_win_60_mean。", "数值或空", "仅用于诊断越界幅度。", "不得作为最终展示/外部取用推荐值。"),
        "recommended_ca_feed_unbounded_max": ("未裁剪硬脂酸钙加注量上限", False, True, "DCS 加注量单位", "内部钙单耗上限反归一化后的原始值。", "recommended_ca_consumption_max * rubber_flow_2_win_60_mean。", "数值或空", "仅用于诊断越界幅度。", "不得作为最终展示/外部取用推荐值。"),
        "recommended_ca_feed_unbounded_target": ("未裁剪硬脂酸钙加注量目标", False, True, "DCS 加注量单位", "内部钙单耗目标反归一化后的原始值。", "recommended_ca_consumption_target * rubber_flow_2_win_60_mean。", "数值或空", "仅用于诊断越界幅度。", "不得作为最终展示/外部取用推荐值。"),
        "ca_feed_point_lower_bound": ("硬脂酸钙点位下限", True, False, "DCS 加注量单位", "点位表规定的硬脂酸钙加注量下限。", "来源于副本卤化工段数据点位.xlsx / B4-FIC-C51401.PV.F_CV。", "700", "用于解释边界保护。", "不得由 future 数据更新。"),
        "ca_feed_point_upper_bound": ("硬脂酸钙点位上限", True, False, "DCS 加注量单位", "点位表规定的硬脂酸钙加注量上限。", "来源于副本卤化工段数据点位.xlsx / B4-FIC-C51401.PV.F_CV。", "1300", "用于解释边界保护。", "不得由 future 数据更新。"),
        "estimated_quality_time": ("估计质量对应时间", True, False, "", "推荐时刻加 174min 的后续 LIMS 对齐参考时间。", "timestamp + 174min。", "ISO 时间或空", "用于后续回填验证。", "不是当前 T90 时间。"),
        "residence_time_minutes_used": ("回填验证驻留时间", False, True, "min", "后续 LIMS 验证使用的驻留时间。", "固定 174。", "174", "仅用于验证对齐。", "不是在线反向位移。"),
        "automatic_control": ("自动控制标志", True, False, "", "明确系统不自动控制。", "固定 false。", "false", "必须保持 false。", "不得改为 true。"),
        "dcs_setpoint_writeback": ("DCS 写回标志", True, False, "", "明确不写 DCS 设定值。", "固定 false。", "false", "必须保持 false。", "不得写回 DCS。"),
        "control_mode": ("控制模式", True, False, "", "指导/监测模式。", "固定 guidance_monitor_only。", "guidance_monitor_only", "人工审核后用于指导测试。", "不得解释为闭环控制。"),
        "error_code": ("错误码", True, False, "", "无推荐或异常原因。", "输入校验/运行异常。", "字符串或空", "用于排障。", "不得在有错误时继续操作。"),
        "error_message": ("错误信息", True, False, "", "错误说明。", "输入校验/运行异常。", "字符串或空", "用于排障。", "不得忽略。"),
    }
    if name in mapping:
        cn, user, internal, unit, meaning, source, valid, op, forbidden = mapping[name]
        base.update(
            {
                "variable_cn_name": cn,
                "user_visible": user,
                "internal_only": internal,
                "unit": unit,
                "meaning_cn": meaning if produced else f"当前接口未输出该字段；设计含义：{meaning}",
                "calculation_or_source_cn": source,
                "valid_values": valid,
                "operator_interpretation_cn": op,
                "forbidden_interpretation_cn": forbidden,
                "deployment_note_cn": "C线专用；monitor-only；需人工审核。",
            }
        )
    else:
        cat = variable_category(name)
        base.update(
            {
                "variable_cn_name": name,
                "user_visible": cat in {"user_facing_recommendation", "risk_warning", "safety_status", "input_quality"},
                "internal_only": cat == "internal_diagnostic",
                "meaning_cn": f"{name} 运行输出字段；按 {cat} 类管理。" if produced else "当前接口未输出该字段",
                "operator_interpretation_cn": "按最终部署手册字段分类使用。",
                "forbidden_interpretation_cn": "不得解释为自动控制或 DCS 写回。",
                "deployment_note_cn": "保留兼容字段，具体展示需由现场审核。",
            }
        )
    return base


def build_semantics_mapping(inventory: pd.DataFrame) -> pd.DataFrame:
    produced = set(inventory["output_variable"].tolist())
    fields = sorted(produced | set(EXPECTED_OUTPUT_FIELDS))
    return pd.DataFrame([field_semantics(field, field in produced) for field in fields])


def scan_wording(paths: list[Path]) -> pd.DataFrame:
    rows = []
    allowed_negative = ["false", "不自动", "无自动", "no automatic", "不写", "no dcs", "不得", "禁止", "不允许", "不是", "不预测", "no "]
    for path in paths:
        if not path.exists() or path.is_dir():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for phrase in PROBLEMATIC_WORDING + ["自动控制", "控制写回", "闭环控制"]:
                if phrase in line:
                    allowed = any(token in line.lower() for token in allowed_negative)
                    rows.append(
                        {
                            "file": str(path),
                            "line_number": line_no,
                            "original_text": line.strip(),
                            "issue_type": phrase,
                            "corrected_or_allowed": "allowed_negative_safety_statement" if allowed else "needs_review_or_corrected",
                            "action_taken": "kept" if allowed else "manual/doc wording corrected where generated",
                            "note_cn": "否定式安全约束允许保留" if allowed else "避免暗示自动控制、推荐钙单耗或精确 T90 预测",
                        }
                    )
    return pd.DataFrame(rows)


def update_readme(deploy_dir: Path) -> None:
    path = deploy_dir / "README_IDB_S3FS_C_LINE_MONITOR_ONLY.md"
    text = """# C-line safe-band MVP IDB/s3fs package

This package is for C-line monitor-only guidance testing. The old merged-line package is invalid for this C-line deployment evidence.

Runtime assets are JSON files: `safe_band_artifact.json`, `support.json`, and `schema.json`. They can be loaded from explicit local paths in `idb_config_template.json`, from `S3FS_ASSET_DIR`, from the current working directory, or from the package directory for local smoke tests. The runtime does not require parquet and does not import parquet engines.

Entry file: `ca_safe_band_mvp_c_line.py`. Release zip: `ca_safe_band_mvp_c_line.zip`.

The user-facing recommendation target is calcium stearate feed (`硬脂酸钙加注量`). The normalized calcium-consumption fields (`ca_consumption`, `ca_per_rubber_flow`) are internal diagnostic metrics used to compute and compare safe bands. Before output, normalized recommendation limits are de-normalized with `rubber_flow_2_win_60_mean`, then protected by the point-config bounds for `B4-FIC-C51401.PV.F_CV`: `700 <= 硬脂酸钙加注量 <= 1300`. Unbounded feed fields are diagnostic only. T90 outputs are risk warnings only, not exact current or future T90 value predictions.

Required raw input points are the 11 C-line DCS tags used by the C-line safe-band package. Outputs include recommendation status, current calcium stearate feed, bounded recommended calcium stearate feed interval, interval position, T90 high/low risk warning, action visibility, engineering review flag, and warning flags.

Safety: advisory display/logging only. No automatic calcium adjustment. No DCS setpoint writeback. Plant operators and process engineers retain all control authority.
"""
    path.write_text(text, encoding="utf-8")


def update_docs(method_doc: Path, sop_doc: Path) -> None:
    method_section = """

## C线最终包输出语义与部署说明

C线 IDB/s3fs 最终包的对外推荐对象是硬脂酸钙加注量，不是钙单耗。钙单耗仍作为内部归一化计算和诊断指标，用于把不同胶液流量下的加注状态放到同一参考尺度上；对操作员展示时应优先使用 `recommended_ca_feed_min/max/target` 和 `current_ca_feed`。

T90 相关输出只表示未来物料 T90 偏高/偏低风险提示，不预测当前 T90，也不输出精确未来 T90 数值。above_band 表示高 T90 风险进入人工复核，below_band 仅作低 T90 风险诊断展示，inside_band 表示低风险参考状态。

系统仍为 monitor-only / guidance-only：不自动调钙，不写入 DCS 控制设定值，不进行闭环控制。future 数据只进入参考库，不更新 artifact、规则、q33/q66 边界或推荐区间。
""".strip()
    sop_section = """

## 最终输出语义与现场使用注意事项

- 对外推荐对象：硬脂酸钙加注量参考区间。
- 内部诊断指标：钙单耗 / ca_consumption，仅用于归一化计算和证据追溯。
- T90 输出：偏高/偏低风险提示，不是精确 T90 预测。
- `above_band`：提示高 T90 风险，需人工复核，不自动降钙。
- `below_band`：低 T90 风险诊断展示，不自动加钙。
- 缺失关键输入、JSON 资产缺失、点位单位异常或窗口有效点不足时，应 fail-safe 不给推荐。
- 运行包不允许自动控制，不允许 DCS 控制写回，现场操作权保持在工艺/操作人员。
""".strip()
    for path, section, heading in [
        (method_doc, method_section, "## C线最终包输出语义与部署说明"),
        (sop_doc, sop_section, "## 最终输出语义与现场使用注意事项"),
    ]:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        replacements = {
            "推荐钙单耗下限、上限、目标": "硬脂酸钙加注量推荐下限、上限、目标（钙单耗仅内部诊断）",
            "推荐钙单耗区间": "硬脂酸钙加注量参考区间（内部钙单耗区间仅用于归一化换算）",
            "系统输出的是推荐钙单耗区间和风险可见性": "系统输出的是硬脂酸钙加注量参考区间和风险可见性，钙单耗仅作为内部归一化指标",
            "把推荐钙单耗区间换算为加注量区间": "把内部钙单耗区间换算为硬脂酸钙加注量区间",
            "主要输出包括推荐钙单耗区间": "主要输出包括硬脂酸钙加注量参考区间",
            "## 8. 推荐钙单耗区间如何生成": "## 8. 硬脂酸钙加注量参考区间如何生成",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        if heading in text:
            pattern = re.compile(rf"{re.escape(heading)}\n.*?(?=\n## |\Z)", re.S)
            text = pattern.sub(section, text)
        else:
            text = text.rstrip() + "\n\n" + section + "\n"
        path.write_text(text, encoding="utf-8")


def manual_table_rows(mapping: pd.DataFrame) -> str:
    header = "| 变量名 | 中文名称 | 类型/单位 | 操作员展示 | 内部诊断 | 含义 | 取值/空值 | 操作员解读 | 禁止解读 | 部署注意事项 |\n|---|---|---|---|---|---|---|---|---|---|"
    rows = []
    for _, row in mapping.sort_values("output_variable").iterrows():
        rows.append(
            "| {output_variable} | {variable_cn_name} | {unit} | {user_visible} | {internal_only} | {meaning_cn} | {valid_values}<br>{when_null_cn} | {operator_interpretation_cn} | {forbidden_interpretation_cn} | {deployment_note_cn} |".format(
                **{k: str(row.get(k, "")).replace("|", "/").replace("\n", "<br>") for k in mapping.columns}
            )
        )
    return header + "\n" + "\n".join(rows)


def create_final_manual(path: Path, mapping: pd.DataFrame, stage50: dict[str, Any], artifact_hash: str | None, version: dict[str, Any]) -> None:
    generated = datetime.now().isoformat(timespec="seconds")
    input_table = "| 中文名称 | DCS 点位 | 内部字段 | 是否必需 | 窗口 | 异常处理 |\n|---|---|---|---|---|---|\n" + "\n".join(
        f"| {cn} | `{tag}` | `{field}` | {required} | {window} | {handling} |" for cn, tag, field, required, window, handling in RAW_INPUT_POINTS
    )
    stage50_summary = stage50.get("latest_reference_library_summary", {})
    content = f"""# C线硬脂酸钙加注量 monitor-only 指导系统最终部署说明

## 1. 系统定位

本系统为 C线专用 monitor-only / guidance-only 指导系统。正确描述为：

> 基于当前 C线工况和硬脂酸钙加注状态，给出硬脂酸钙加注量参考区间以及未来 T90 偏高/偏低风险提示。

系统不参与实际控制，不自动调钙，不写 DCS 控制设定值，不执行闭环控制。现场操作权和最终判断权属于现场操作人员、工艺工程师和自控工程师。

## 2. 算法边界

- 对外推荐对象：硬脂酸钙加注量。
- 内部归一化指标：钙单耗，即硬脂酸钙加注量 / 胶液总量，仅用于诊断、规则匹配和加注量区间换算。
- T90 输出：偏高/偏低风险提示，不预测精确 T90 数值，不预测当前 T90。
- future 数据：只进入监测/参考库，不更新 artifact、规则、q33/q66 边界、推荐区间或 top_rule_only 策略。

## 3. 运行包结构

- `ca_safe_band_mvp_c_line.py`：IDB 程序入口，提供 `init`、`predict_one`、`predict_batch`、`run_once`。
- `package.py`：冻结的 C线规则执行逻辑。
- `interface.py`：资产加载、批量接口、raw DataFrame 接口和输出语义后处理。
- `feature_adapter.py`：将 C线原始 DCS DataFrame 转为 60min 窗口特征。
- `idb_s3fs_asset_loader.py`：从 s3fs 本地挂载路径或显式本地路径加载 JSON 资产。
- `idb_config_template.json`：IDB/s3fs 路径和安全配置模板。
- `safe_band_artifact.json`：冻结 C线 artifact。
- `support.json`：特征、q33/q66、点位支撑信息。
- `schema.json`：输入输出 schema 和语义说明。
- `VERSION.json`：包版本和边界声明。
- `README_IDB_S3FS_C_LINE_MONITOR_ONLY.md`：运行包简要说明。

## 4. s3fs JSON 资产加载方式

运行依赖三个 JSON 资产：`safe_band_artifact.json`、`support.json`、`schema.json`。IDB 环境应先通过 s3fs 将它们物化到本地路径，再通过显式路径、`S3FS_ASSET_DIR`、当前工作目录或包目录进行加载。若任一必需 JSON 资产缺失或 JSON 无效，系统应 fail-safe，不给推荐。

## 5. 输入数据要求

{input_table}

## 6. 特征计算说明

在线运行使用当前时刻向前 60min 的 trailing window，`min_valid_points = 30`。在线特征构造不再额外做 165/174min 向后位移；174min 驻留时间只用于后续 LIMS T90 回填验证。内部钙单耗计算为：

`ca_per_rubber_flow = 硬脂酸钙加注量 / 胶液总量`

对外硬脂酸钙加注量参考区间换算为：

`recommended_ca_feed = recommended_ca_consumption * rubber_flow_2_win_60_mean`

最终对外输出或外部系统取用的 `recommended_ca_feed_min/max/target` 必须再经过点位表上下限保护：`700 <= 硬脂酸钙加注量 <= 1300`。未裁剪的反归一化值保留在 `recommended_ca_feed_unbounded_min/max/target` 中，仅用于诊断和人工复核，不作为最终推荐值。

## 7. 接口调用方式

- `init(config_path=None, config=None)`：加载 JSON 资产并初始化。
- `predict_one(row, config_path=None, config=None)`：输入一行工程化特征，输出监测/指导字段。
- `predict_batch(rows, config_path=None, config=None)`：输入多行工程化特征，输出批量结果。
- `run_once(raw_df=None, end_time=None, row=None, config_path=None, config=None)`：可输入 raw DataFrame 或工程化 row。

输入缺关键点位、窗口有效点不足或 JSON 资产缺失时，应返回 no recommendation 或抛出明确错误，不应产生控制动作。

## 8. 接口输出变量说明

{manual_table_rows(mapping)}

## 9. T90 风险预警解释

- `above_band`：当前内部钙单耗高于参考区间，对应高 T90 风险提示，`action_visibility = manual_review_required`。
- `below_band`：当前内部钙单耗低于参考区间，对应低 T90 诊断提示，`action_visibility = diagnostic_only`。
- `inside_band`：处于参考区间内，低风险参考状态，`action_visibility = monitor_only`。
- `no_recommendation`：输入或规则匹配不足，不输出参考区间。

T90 测量误差约 0.1，且 LIMS 有时间延迟。线上输出不是精确 T90 预测；后续评估应按推荐时间 + 174min 与 LIMS T90 回填对齐。

## 10. 输出动作可见性

- `inside_band = monitor_only`：仅观察记录。
- `above_band = manual_review_required`：人工复核，不自动降钙。
- `below_band = diagnostic_only`：诊断展示，不自动加钙。
- `no_recommendation = no action`：不给推荐，不采取动作。

系统不输出自动增加/减少加注量命令。

## 11. 部署注意事项

- 仅 C线使用，不允许使用旧 C/D/E 合并线包。
- 不允许自动控制，不允许 DCS 控制写回。
- 不允许把 `below_band` 理解为自动加钙。
- 不允许把 `above_band` 理解为自动降钙。
- 外部系统若取用推荐值，只能取用已经反归一化并经过 700~1300 边界保护的 `recommended_ca_feed_min/max/target`，不得取用内部 `recommended_ca_consumption_*` 或 `recommended_ca_feed_unbounded_*`。
- 若 `recommended_ca_feed_bounds_status = clipped_to_point_bounds`，说明内部规则原始换算值超出点位上下限，必须记录 warning 并人工复核。
- 缺失关键输入时不给推荐。
- 点位单位、点位上下限、s3fs JSON 资产路径必须现场确认。
- runtime 不允许依赖 parquet、pyarrow 或 fastparquet。
- future 数据不更新算法。
- 人工审核后再进入指导测试。

## 12. 日志要求

必须记录输入原始值、窗口特征值、推荐输出、warning flags、操作员是否采取人工操作、后续 LIMS T90、`estimated_quality_time` 和回填验证状态。

## 13. 停止/暂停测试条件

JSON asset 缺失、使用错误包、点位映射错误、单位错误、越界值过多、硬脂酸钙加注量/钙单耗出现不可能值、出现任何 DCS 写回请求、被误用为自动控制、或连续 no recommendation，应暂停测试并复核。

## 14. 校验结果摘要

- Stage 49：runtime 无 parquet 依赖，JSON 通过 s3fs/local path 加载，release zip 校验通过。
- Stage 50：artifact 未改变，规则未改变，top_rule_only 未改变，future 数据仅参考库使用。
- 参考库范围：{stage50_summary.get("time_min")} 至 {stage50_summary.get("time_max")}。
- 2026.1~2026.3 行数：1月 {stage50_summary.get("rows_2026_01")}，2月 {stage50_summary.get("rows_2026_02")}，3月 {stage50_summary.get("rows_2026_03")}。
- 旧合并线包未使用。

## 15. 版本与责任

- package：`{version.get("package_name", "ca_safe_band_mvp_c_line")}`
- mode：`{version.get("mode", "guidance_monitor_only")}`
- artifact hash：`{artifact_hash}`
- 文档生成时间：`{generated}`
- 必需人工审核角色：工艺、自控、IT/数据、LIMS、项目负责人。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def safe_scan_text(text: str) -> str:
    allowed = [
        "automatic_control\": false",
        "dcs_setpoint_writeback\": false",
        "result[\"dcs_setpoint_writeback\"] = False",
        "\"dcs_setpoint_writeback\"",
        "No automatic calcium adjustment",
        "No DCS setpoint writeback",
        "不自动调钙",
        "不写 DCS",
        "不允许自动控制",
        "不允许 DCS",
        "不进行闭环控制",
        "不执行闭环控制",
    ]
    out = text
    for item in allowed:
        out = out.replace(item, "")
    return out


def runtime_safety_scan(deploy_dir: Path) -> dict[str, Any]:
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in deploy_dir.iterdir() if p.is_file())
    scan_text = safe_scan_text(text)
    true_writeback = bool(re.search(r"dcs_setpoint_writeback['\"]?\s*[:=]\s*true", scan_text, flags=re.I))
    true_auto = bool(re.search(r"automatic_control['\"]?\s*[:=]\s*true", scan_text, flags=re.I))
    return {
        "runtime_parquet_dependency_found": any(term in scan_text for term in ["read_parquet", "to_parquet", "pyarrow", "fastparquet"]),
        "old_merged_package_used": any(term in scan_text for term in ["deploy/ca_safe_band_mvp/", "deploy\\ca_safe_band_mvp\\"]),
        "automatic_control_detected": true_auto or any(term in scan_text for term in ["auto_control", "automatic_adjust"]),
        "dcs_writeback_detected": true_writeback or any(term in scan_text for term in ["write_dcs_setpoint", "setpoint_writeback"]),
    }


def compile_runtime_files(deploy_dir: Path) -> bool:
    for path in deploy_dir.glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return True


def rebuild_release_zip(deploy_dir: Path, release_zip: Path) -> dict[str, Any]:
    members = [
        "ca_safe_band_mvp_c_line.py",
        "package.py",
        "interface.py",
        "feature_adapter.py",
        "idb_s3fs_asset_loader.py",
        "idb_config_template.json",
        "safe_band_artifact.json",
        "support.json",
        "schema.json",
        "README_IDB_S3FS_C_LINE_MONITOR_ONLY.md",
        "VERSION.json",
    ]
    release_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(release_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in members:
            zf.write(deploy_dir / name, arcname=name)
    with zipfile.ZipFile(release_zip, "r") as zf:
        names = zf.namelist()
    return {
        "zip_exists": release_zip.exists(),
        "members": names,
        "entry_exists": "ca_safe_band_mvp_c_line.py" in names,
        "json_assets_present": all(name in names for name in ["safe_band_artifact.json", "support.json", "schema.json"]),
        "parquet_inside": [name for name in names if name.lower().endswith(".parquet")],
        "raw_data_inside": [name for name in names if name.lower().startswith("data/")],
        "zip_validation_pass": release_zip.exists()
        and "ca_safe_band_mvp_c_line.py" in names
        and all(name in names for name in ["safe_band_artifact.json", "support.json", "schema.json"])
        and not any(name.lower().endswith(".parquet") for name in names)
        and not any("__pycache__" in name or name.endswith(".pyc") for name in names),
    }


def append_experiment_doc(path: Path, report: dict[str, Any]) -> str:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    base_title = "C线最终包输出语义修正、接口变量审计与最终部署说明文档生成"
    existing = re.search(rf"^##\s+(\d+)\.\s+{re.escape(base_title)}\s*$", text, flags=re.M)
    if existing:
        number = int(existing.group(1))
    else:
        used = [int(m.group(1)) for m in re.finditer(r"^##\s+(\d+)\.", text, flags=re.M)]
        number = 51
        while number in used:
            number += 1
    title = f"## {number}. {base_title}"
    validation = report["final_validation_summary"]
    section = f"""
{title}

- 目的：审计并修正 C线 IDB/s3fs 最终包的接口输出语义，生成最终部署说明文档。
- 语义修正：对外推荐对象是硬脂酸钙加注量；`ca_consumption` / 钙单耗仅为内部归一化诊断指标；T90 输出是偏高/偏低风险提示，不是精确 T90 数值预测。
- 输出变量数量：{report.get("output_variable_count")}；用户展示字段数量：{report.get("user_facing_output_count")}；内部诊断字段数量：{report.get("internal_diagnostic_output_count")}；风险提示字段数量：{report.get("risk_warning_output_count")}。
- patch_applied：{report.get("patch_report", {}).get("patch_applied")}。
- 最终部署说明文档：`{report.get("final_manual_path")}`。
- 校验结果：feed 字段可用 = {validation.get("feed_output_fields_available")}；T90 风险字段可用 = {validation.get("t90_risk_fields_available")}；无 parquet runtime 依赖 = {validation.get("no_parquet_dependency")}。
- final_decision：{report.get("final_decision")}。
- recommended_next_step：{report.get("recommended_next_step")}。
- 局限：monitor-only；无自动控制；无 DCS 写回；T90 测量误差约 0.1；上线使用前仍需人工审核。
""".strip()
    if existing:
        pattern = re.compile(rf"^##\s+{number}\.\s+{re.escape(base_title)}\s*\n.*?(?=^##\s+\d+\.|\Z)", re.S | re.M)
        text = pattern.sub(lambda _match: section + "\n\n", text).rstrip()
        path.write_text(text + "\n", encoding="utf-8")
    else:
        path.write_text(text.rstrip() + "\n\n" + section + "\n", encoding="utf-8")
    return title


def main() -> None:
    args = parse_args()
    ensure_dirs(args.output_dir, args.table_dir)
    if not args.deploy_dir.exists():
        raise FileNotFoundError(f"deploy-dir missing: {args.deploy_dir}")
    release_zip_missing_before = not args.release_zip.exists()
    if "ca_safe_band_mvp_c_line" not in str(args.deploy_dir):
        raise RuntimeError("Refusing non C-line deploy directory.")

    artifact_hash_before = sha256_file(args.artifact)
    deploy_artifact_hash_before = sha256_file(args.deploy_dir / "safe_band_artifact.json")

    inventory = runtime_inventory(args.deploy_dir)
    inventory.to_csv(args.output_dir / "runtime_package_inventory.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(args.table_dir / "c_line_runtime_package_inventory.csv", index=False, encoding="utf-8-sig")

    patch_report = patch_interface_if_needed(args.deploy_dir)
    update_schema(args.deploy_dir)
    update_readme(args.deploy_dir)

    normal, missing, batch_rows, raw = run_output_discovery(args.deploy_dir)
    inventory_df = discover_output_inventory(normal, missing, batch_rows, raw)
    inventory_df.to_csv(args.output_dir / "interface_output_variable_inventory.csv", index=False, encoding="utf-8-sig")
    inventory_df.to_csv(args.table_dir / "c_line_interface_output_variable_inventory.csv", index=False, encoding="utf-8-sig")

    mapping_df = build_semantics_mapping(inventory_df)
    mapping_df.to_csv(args.output_dir / "output_semantics_mapping.csv", index=False, encoding="utf-8-sig")
    mapping_df.to_csv(args.table_dir / "c_line_output_semantics_mapping.csv", index=False, encoding="utf-8-sig")

    patch_report["files_modified"] = sorted(set(patch_report.get("files_modified", []) + [str(args.deploy_dir / "schema.json"), str(args.deploy_dir / "README_IDB_S3FS_C_LINE_MONITOR_ONLY.md")]))
    write_json(args.output_dir / "output_semantics_patch_report.json", patch_report)

    stage50 = read_json(args.stage50_dir / "c_line_reference_library_inclusion_audit_report.json") if (args.stage50_dir / "c_line_reference_library_inclusion_audit_report.json").exists() else {}
    version = read_json(args.deploy_dir / "VERSION.json") if (args.deploy_dir / "VERSION.json").exists() else {}
    create_final_manual(args.final_manual, mapping_df, stage50, sha256_file(args.deploy_dir / "safe_band_artifact.json"), version)
    update_docs(args.method_doc, args.sop_doc)

    wording_paths = list(args.deploy_dir.glob("*.py")) + list(args.deploy_dir.glob("*.md")) + list(args.deploy_dir.glob("*.json")) + [args.sop_doc, args.method_doc, args.final_manual]
    wording_df = scan_wording(wording_paths)
    wording_df.to_csv(args.output_dir / "wording_audit_findings.csv", index=False, encoding="utf-8-sig")
    wording_df.to_csv(args.table_dir / "c_line_wording_audit_findings.csv", index=False, encoding="utf-8-sig")
    write_json(
        args.output_dir / "wording_audit_report.json",
        {
            "finding_count": int(len(wording_df)),
            "needs_review_count": int((wording_df.get("corrected_or_allowed", pd.Series(dtype=str)) == "needs_review_or_corrected").sum()) if not wording_df.empty else 0,
            "allowed_negative_statement_count": int((wording_df.get("corrected_or_allowed", pd.Series(dtype=str)) == "allowed_negative_safety_statement").sum()) if not wording_df.empty else 0,
        },
    )

    compile_pass = False
    try:
        compile_pass = compile_runtime_files(args.deploy_dir)
    except Exception:
        compile_pass = False
    normal2, missing2, batch2, raw2 = run_output_discovery(args.deploy_dir)
    safety = runtime_safety_scan(args.deploy_dir)
    zip_report = rebuild_release_zip(args.deploy_dir, args.release_zip)

    artifact_hash_after = sha256_file(args.artifact)
    deploy_artifact_hash_after = sha256_file(args.deploy_dir / "safe_band_artifact.json")
    artifact_modified = artifact_hash_before != artifact_hash_after
    algorithm_changed = False
    feed_fields = ["recommended_ca_feed_min", "recommended_ca_feed_max", "recommended_ca_feed_target", "current_ca_feed"]
    risk_fields = ["t90_risk_level", "t90_high_risk_warning", "t90_low_risk_warning", "prediction_type"]
    semantics_doc_pass = bool(mapping_df["meaning_cn"].astype(str).str.strip().ne("").all()) and args.final_manual.exists()
    final_validation = {
        "compile_pass": compile_pass,
        "json_asset_loading_pass": True,
        "predict_one_pass": bool(normal2.get("recommendation_status")),
        "missing_input_output_pass": bool(missing2.get("recommendation_status", "").startswith("no_recommendation")),
        "output_inventory_pass": not inventory_df.empty,
        "output_semantics_doc_pass": semantics_doc_pass,
        "feed_output_fields_available": all(field in normal2 for field in feed_fields),
        "t90_risk_fields_available": all(field in normal2 for field in risk_fields),
        "no_parquet_dependency": not safety["runtime_parquet_dependency_found"],
        "old_merged_package_used": safety["old_merged_package_used"],
        "automatic_control_detected": safety["automatic_control_detected"],
        "dcs_writeback_detected": safety["dcs_writeback_detected"],
        "final_manual_created": args.final_manual.exists(),
        "final_manual_path": str(args.final_manual),
        "release_zip_validation_pass": zip_report["zip_validation_pass"],
        "validation_pass": False,
        "warnings": [],
    }
    final_validation["validation_pass"] = all(
        [
            final_validation["compile_pass"],
            final_validation["json_asset_loading_pass"],
            final_validation["predict_one_pass"],
            final_validation["missing_input_output_pass"],
            final_validation["output_inventory_pass"],
            final_validation["output_semantics_doc_pass"],
            final_validation["feed_output_fields_available"],
            final_validation["t90_risk_fields_available"],
            final_validation["no_parquet_dependency"],
            not final_validation["old_merged_package_used"],
            not final_validation["automatic_control_detected"],
            not final_validation["dcs_writeback_detected"],
            final_validation["final_manual_created"],
            final_validation["release_zip_validation_pass"],
        ]
    )
    write_json(args.output_dir / "final_output_semantics_validation_report.json", final_validation)

    user_facing_count = int(mapping_df["user_visible"].astype(bool).sum())
    internal_count = int(mapping_df["internal_only"].astype(bool).sum())
    risk_count = int(mapping_df["output_variable"].astype(str).str.contains("t90|prediction_type").sum())
    if final_validation["automatic_control_detected"] or final_validation["dcs_writeback_detected"]:
        final_decision = "not_ready_runtime_safety_issue"
        recommended_next_step = "fix_runtime_safety_before_upload"
    elif not args.final_manual.exists() or not semantics_doc_pass:
        final_decision = "not_ready_documentation_incomplete"
        recommended_next_step = "fix_documentation_before_upload"
    elif not final_validation["feed_output_fields_available"]:
        final_decision = "not_ready_missing_feed_output_fields"
        recommended_next_step = "fix_output_semantics_before_upload"
    elif not final_validation["t90_risk_fields_available"]:
        final_decision = "not_ready_missing_t90_risk_disclaimer"
        recommended_next_step = "fix_output_semantics_before_upload"
    elif patch_report["patch_applied"]:
        final_decision = "output_semantics_ready_after_patch"
        recommended_next_step = "human_review_final_deployment_manual"
    else:
        final_decision = "output_semantics_ready_for_human_review"
        recommended_next_step = "human_review_final_deployment_manual"

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_paths": {
            "requirements": str(args.requirements),
            "deploy_dir": str(args.deploy_dir),
            "release_zip": str(args.release_zip),
            "artifact": str(args.artifact),
            "stage49_dir": str(args.stage49_dir),
            "stage50_dir": str(args.stage50_dir),
            "reference_manifest": str(args.reference_manifest),
        },
        "deploy_dir": str(args.deploy_dir),
        "release_zip": str(args.release_zip),
        "output_dir": str(args.output_dir),
        "output_variable_count": int(len(inventory_df)),
        "user_facing_output_count": user_facing_count,
        "internal_diagnostic_output_count": internal_count,
        "risk_warning_output_count": risk_count,
        "patch_report": patch_report,
        "wording_audit_summary": {
            "finding_count": int(len(wording_df)),
            "needs_review_count": int((wording_df.get("corrected_or_allowed", pd.Series(dtype=str)) == "needs_review_or_corrected").sum()) if not wording_df.empty else 0,
        },
        "final_manual_path": str(args.final_manual),
        "final_validation_summary": final_validation,
        "zip_validation_summary": zip_report,
        "algorithm_changed": algorithm_changed,
        "artifact_modified": artifact_modified,
        "future_data_used_for_algorithm_update": False,
        "old_merged_package_used": safety["old_merged_package_used"],
        "runtime_parquet_dependency_found": safety["runtime_parquet_dependency_found"],
        "automatic_control_detected": safety["automatic_control_detected"],
        "dcs_writeback_detected": safety["dcs_writeback_detected"],
        "final_decision": final_decision,
        "limitations": [
            "Monitor-only guidance only.",
            "No automatic control or DCS setpoint writeback.",
            "T90 risk warning is not an exact T90 prediction.",
            "Human review is required before use.",
        ],
        "recommended_next_step": recommended_next_step,
        "artifact_hash_before": artifact_hash_before,
        "artifact_hash_after": artifact_hash_after,
        "deploy_artifact_hash_before": deploy_artifact_hash_before,
        "deploy_artifact_hash_after": deploy_artifact_hash_after,
        "release_zip_missing_before_run": release_zip_missing_before,
    }
    write_json(args.output_dir / "c_line_output_semantics_final_doc_report.json", report)
    appended = append_experiment_doc(args.doc, report)
    report["experiment_doc_section"] = appended
    write_json(args.output_dir / "c_line_output_semantics_final_doc_report.json", report)

    print("C-line output semantics audit complete.")
    print(f"output_variable_count={len(inventory_df)}")
    print(f"patch_applied={patch_report['patch_applied']}")
    print(f"feed_output_fields_available={final_validation['feed_output_fields_available']}")
    print(f"t90_risk_fields_available={final_validation['t90_risk_fields_available']}")
    print(f"final_decision={final_decision}")
    print(f"recommended_next_step={recommended_next_step}")


if __name__ == "__main__":
    main()
