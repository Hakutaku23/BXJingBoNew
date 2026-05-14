from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


STANDARD_IMPORTS = {
    "__future__", "argparse", "ast", "collections", "copy", "csv", "datetime", "decimal",
    "importlib", "json", "math", "os", "pathlib", "re", "statistics", "sys",
    "traceback", "typing", "warnings", "platform",
}
IMPORT_TO_PACKAGE = {
    "sklearn": "scikit-learn", "PIL": "pillow", "yaml": "PyYAML", "cv2": "opencv-python",
    "fastapi": "fastapi", "uvicorn": "uvicorn", "pandas": "pandas", "pyarrow": "pyarrow",
    "numpy": "numpy", "pydantic": "pydantic", "scipy": "scipy", "matplotlib": "matplotlib",
    "seaborn": "seaborn", "torch": "torch", "tensorflow": "tensorflow",
    "lightgbm": "lightgbm", "xgboost": "xgboost",
}
HEAVY_IMPORTS = {"sklearn", "scipy", "matplotlib", "seaborn", "torch", "tensorflow", "lightgbm", "xgboost", "fastapi", "uvicorn", "pydantic"}
FORBIDDEN_HINT = "increase_to_band"
PACKAGE_VERSION = "1.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair calcium safe-band runtime package for production safety.")
    parser.add_argument("--requirements", type=Path, default=Path("IDB_requirements.txt"))
    parser.add_argument("--deploy-dir", type=Path, default=Path("deploy/ca_safe_band_mvp"))
    parser.add_argument("--final-dry-run", type=Path, default=Path("runs/ca_safe_band_mvp/final_monitor_dry_run.parquet"))
    parser.add_argument("--final-rule-summary", type=Path, default=Path("runs/ca_safe_band_mvp/final_rule_summary.csv"))
    parser.add_argument("--manual-review-sheet", type=Path, default=Path("reports/tables/ca_safe_band_mvp_manual_review_sheet.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ca_safe_band_runtime_repair"))
    parser.add_argument("--method-doc", type=Path, default=Path("docs/ca_safe_band_mvp_method_and_dataflow.md"))
    parser.add_argument("--experiment-doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    return parser.parse_args()


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name).strip().lower())


def parse_requirements(path: Path) -> set[str]:
    packages: set[str] = set()
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


def sanitize_json_value(value: Any, counter: dict[str, int]) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json_value(v, counter) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(v, counter) for v in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(v, counter) for v in value]
    if hasattr(value, "item"):
        try:
            return sanitize_json_value(value.item(), counter)
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            counter["non_strict"] += 1
            return None
    return value


def read_json_lenient(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_strict_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def strict_json_check(paths: list[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    before_counts: dict[str, int] = {}
    repaired_payloads: dict[str, Any] = {}
    invalid_before: list[str] = []
    for path in paths:
        try:
            payload = read_json_lenient(path)
        except Exception:
            invalid_before.append(str(path))
            continue
        counter = {"non_strict": 0}
        repaired = sanitize_json_value(payload, counter)
        before_counts[str(path)] = counter["non_strict"]
        repaired_payloads[str(path)] = repaired
    return {"invalid_before": invalid_before, "non_strict_before": before_counts}, repaired_payloads


def package_py_source() -> str:
    return r'''from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


ACTION_VISIBILITY_POLICY = {
    "inside_band": {
        "action_hint": "hold_in_band",
        "action_visibility": "monitor_only",
        "engineering_review_required": False,
        "explanation_cn": "当前钙单耗处于推荐安全区间内，建议维持观察。",
    },
    "above_band": {
        "action_hint": "above_band_manual_review",
        "action_visibility": "manual_review_required",
        "engineering_review_required": True,
        "explanation_cn": "当前钙单耗高于推荐安全区间，历史数据中高 T90 风险偏高，建议人工复核是否需要小幅降钙。",
    },
    "below_band": {
        "action_hint": "below_band_diagnostic_only",
        "action_visibility": "diagnostic_only",
        "engineering_review_required": True,
        "explanation_cn": "当前钙单耗低于推荐安全区间，仅作诊断展示；当前 MVP 不给出加钙操作建议。",
    },
    "missing": {
        "action_hint": "no_recommendation_missing_input",
        "action_visibility": "no_recommendation",
        "engineering_review_required": True,
        "explanation_cn": "关键输入缺失，无法生成推荐区间。",
    },
}


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "nan", "null"}:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _split_rule_ids(value: Any) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return []
    for sep in (";", ",", "|"):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def normalize_input_row(row: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(row or {})
    if "current_ca_consumption" not in result:
        for source in ("actual_ca_consumption", "ca_per_rubber_flow_win_60_mean"):
            if source in result:
                result["current_ca_consumption"] = result[source]
                break
    return result


def _schema_required_features(schema: Optional[Dict[str, Any]], support: Dict[str, Any]) -> List[str]:
    if isinstance(schema, dict) and isinstance(schema.get("required_features"), list):
        return [str(item) for item in schema.get("required_features", [])]
    features = support.get("features", {}) if isinstance(support, dict) else {}
    return [str(name) for name, meta in features.items() if isinstance(meta, dict) and meta.get("required")]


def validate_required_features(row: Dict[str, Any], support: Dict[str, Any], schema: Optional[Dict[str, Any]] = None, mode: str = "production") -> Dict[str, Any]:
    missing: List[str] = []
    required = _schema_required_features(schema, support)
    for feature in required:
        role = support.get("features", {}).get(feature, {}).get("feature_role") if isinstance(support, dict) else None
        if role == "primary_dose_feature":
            if _to_float(row.get(feature)) is None and _to_float(row.get("current_ca_consumption")) is None:
                missing.append(feature)
        elif role == "regime_feature" and mode == "production":
            if _to_float(row.get(feature)) is None:
                missing.append(feature)
    return {"valid": len(missing) == 0, "missing_required_features": missing}


def classify_interval_position(current_value: Any, interval_min: Any, interval_max: Any) -> str:
    current = _to_float(current_value)
    lo = _to_float(interval_min)
    hi = _to_float(interval_max)
    if current is None or lo is None or hi is None:
        return "missing"
    if current < lo:
        return "below_band"
    if current > hi:
        return "above_band"
    return "inside_band"


def apply_action_visibility(interval_position: str) -> Dict[str, Any]:
    return dict(ACTION_VISIBILITY_POLICY.get(interval_position, ACTION_VISIBILITY_POLICY["missing"]))


def _rule_by_id(artifact: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(rule.get("rule_id")): rule for rule in artifact.get("final_rules", []) if rule.get("rule_id")}


def _classify_regime(value: Any, q33: Any, q66: Any) -> Optional[str]:
    val = _to_float(value)
    low = _to_float(q33)
    high = _to_float(q66)
    if val is None or low is None or high is None:
        return None
    if val <= low:
        return "low"
    if val <= high:
        return "mid"
    return "high"


def match_rules(row: Dict[str, Any], artifact: Dict[str, Any], support: Dict[str, Any], mode: str = "production") -> List[Dict[str, Any]]:
    rules_by_id = _rule_by_id(artifact)
    if mode == "replay":
        explicit_ids = _split_rule_ids(row.get("selected_rule_ids")) or _split_rule_ids(row.get("matched_rule_ids"))
        if explicit_ids:
            return [rules_by_id[rule_id] for rule_id in explicit_ids if rule_id in rules_by_id]
    features = support.get("features", {}) if isinstance(support, dict) else {}
    matched: List[Dict[str, Any]] = []
    for rule in artifact.get("final_rules", []):
        feature = rule.get("regime_feature")
        if not feature or feature not in features:
            continue
        meta = features[feature]
        bin_name = _classify_regime(row.get(feature), meta.get("q33"), meta.get("q66"))
        if bin_name is not None and str(rule.get("regime_bin")) == bin_name:
            matched.append(rule)
    return matched


def aggregate_rules_median(matched_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    lows = [_to_float(rule.get("recommended_dose_min")) for rule in matched_rules]
    highs = [_to_float(rule.get("recommended_dose_max")) for rule in matched_rules]
    lows = [v for v in lows if v is not None]
    highs = [v for v in highs if v is not None]
    if not lows or not highs:
        return {"recommended_ca_consumption_min": None, "recommended_ca_consumption_max": None, "recommended_ca_consumption_target": None}
    lo = float(statistics.median(lows))
    hi = float(statistics.median(highs))
    return {"recommended_ca_consumption_min": lo, "recommended_ca_consumption_max": hi, "recommended_ca_consumption_target": (lo + hi) / 2.0}


def _rule_evidence(matched_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not matched_rules:
        return {"rule_evidence_ok_rate": None, "rule_evidence_high_rate": None, "rule_evidence_low_rate": None, "rule_evidence_sample_count": None}
    def med(name: str) -> Optional[float]:
        values = [_to_float(rule.get(name)) for rule in matched_rules]
        values = [v for v in values if v is not None]
        return float(statistics.median(values)) if values else None
    samples = [_to_float(rule.get("sample_count")) for rule in matched_rules]
    samples = [v for v in samples if v is not None]
    return {
        "rule_evidence_ok_rate": med("best_ok_rate"),
        "rule_evidence_high_rate": med("best_high_rate"),
        "rule_evidence_low_rate": med("best_low_rate"),
        "rule_evidence_sample_count": int(statistics.median(samples)) if samples else None,
    }


def _feed_conversion(row: Dict[str, Any], interval: Dict[str, Any], warning_flags: List[str]) -> Dict[str, Any]:
    flow = _to_float(row.get("rubber_flow_2_win_60_mean"))
    if flow is None:
        warning_flags.append("missing_rubber_flow_for_feed_conversion")
        return {"recommended_ca_feed_min": None, "recommended_ca_feed_max": None, "recommended_ca_feed_target": None}
    return {
        "recommended_ca_feed_min": None if interval.get("recommended_ca_consumption_min") is None else interval["recommended_ca_consumption_min"] * flow,
        "recommended_ca_feed_max": None if interval.get("recommended_ca_consumption_max") is None else interval["recommended_ca_consumption_max"] * flow,
        "recommended_ca_feed_target": None if interval.get("recommended_ca_consumption_target") is None else interval["recommended_ca_consumption_target"] * flow,
    }


def _base_no_recommendation(current: Optional[float], missing_required: List[str], artifact: Dict[str, Any], warning_flags: List[str]) -> Dict[str, Any]:
    visibility = apply_action_visibility("missing")
    if missing_required:
        warning_flags.append("missing_required_features")
    return {
        "recommendation_status": "no_recommendation_missing_input",
        "current_ca_consumption": current,
        "recommended_ca_consumption_min": None,
        "recommended_ca_consumption_max": None,
        "recommended_ca_consumption_target": None,
        "recommended_ca_feed_min": None,
        "recommended_ca_feed_max": None,
        "recommended_ca_feed_target": None,
        "interval_position": "missing",
        "action_hint": visibility["action_hint"],
        "final_action_hint": visibility["action_hint"],
        "action_visibility": visibility["action_visibility"],
        "confidence_level": None,
        "matched_rule_count": 0,
        "matched_rule_ids": "",
        "selected_rule_ids": "",
        "rule_evidence_ok_rate": None,
        "rule_evidence_high_rate": None,
        "rule_evidence_low_rate": None,
        "rule_evidence_sample_count": None,
        "engineering_review_required": visibility["engineering_review_required"],
        "explanation_cn": visibility["explanation_cn"],
        "warning_flags": ";".join(sorted(set(warning_flags))),
        "input_valid": False,
        "missing_required_features": missing_required,
        "model_version": artifact.get("artifact_version"),
        "artifact_version": artifact.get("artifact_version"),
    }


def recommend_one(row: Dict[str, Any], artifact: Dict[str, Any], support: Dict[str, Any], schema: Optional[Dict[str, Any]] = None, mode: str = "production") -> Dict[str, Any]:
    state = normalize_input_row(row)
    current = _to_float(state.get("current_ca_consumption"))
    warning_flags: List[str] = []
    validation = {"valid": True, "missing_required_features": []} if mode == "replay" else validate_required_features(state, support, schema, mode=mode)
    if not validation["valid"]:
        return _base_no_recommendation(current, validation["missing_required_features"], artifact, warning_flags)
    matched_rules = match_rules(state, artifact, support, mode=mode)
    if not matched_rules:
        warning_flags.append("no_matched_rules")
        return _base_no_recommendation(current, [], artifact, warning_flags)
    interval = aggregate_rules_median(matched_rules)
    position = classify_interval_position(current, interval["recommended_ca_consumption_min"], interval["recommended_ca_consumption_max"])
    visibility = apply_action_visibility(position)
    if position == "above_band":
        warning_flags.append("high_t90_risk_manual_review")
    elif position == "below_band":
        warning_flags.append("increase_hint_hidden_diagnostic_only")
    elif position == "missing":
        warning_flags.append("missing_required_input")
    feed = _feed_conversion(state, interval, warning_flags)
    rule_ids = [str(rule.get("rule_id")) for rule in matched_rules if rule.get("rule_id")]
    evidence = _rule_evidence(matched_rules)
    return {
        "recommendation_status": "recommended" if position != "missing" else "no_recommendation_missing_input",
        "current_ca_consumption": current,
        "recommended_ca_consumption_min": interval["recommended_ca_consumption_min"],
        "recommended_ca_consumption_max": interval["recommended_ca_consumption_max"],
        "recommended_ca_consumption_target": interval["recommended_ca_consumption_target"],
        "recommended_ca_feed_min": feed["recommended_ca_feed_min"],
        "recommended_ca_feed_max": feed["recommended_ca_feed_max"],
        "recommended_ca_feed_target": feed["recommended_ca_feed_target"],
        "interval_position": position,
        "action_hint": visibility["action_hint"],
        "final_action_hint": visibility["action_hint"],
        "action_visibility": visibility["action_visibility"],
        "confidence_level": state.get("confidence_level"),
        "matched_rule_count": len(matched_rules),
        "matched_rule_ids": ";".join(rule_ids),
        "selected_rule_ids": ";".join(rule_ids),
        "rule_evidence_ok_rate": evidence["rule_evidence_ok_rate"],
        "rule_evidence_high_rate": evidence["rule_evidence_high_rate"],
        "rule_evidence_low_rate": evidence["rule_evidence_low_rate"],
        "rule_evidence_sample_count": evidence["rule_evidence_sample_count"],
        "engineering_review_required": visibility["engineering_review_required"],
        "explanation_cn": visibility["explanation_cn"],
        "warning_flags": ";".join(sorted(set(warning_flags))),
        "input_valid": True,
        "missing_required_features": [],
        "model_version": artifact.get("artifact_version"),
        "artifact_version": artifact.get("artifact_version"),
    }


def recommend_batch(rows: List[Dict[str, Any]], artifact: Dict[str, Any], support: Dict[str, Any], schema: Optional[Dict[str, Any]] = None, mode: str = "production") -> List[Dict[str, Any]]:
    return [recommend_one(row, artifact, support, schema=schema, mode=mode) for row in rows]


def load_json(path: Any) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
'''


def interface_py_source() -> str:
    return r'''from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from . import package
except Exception:
    import package  # type: ignore


class SafeBandRecommender:
    def __init__(self, model_dir: Optional[Any] = None, mode: str = "production"):
        self.model_dir = Path(model_dir) if model_dir is not None else Path(__file__).resolve().parent
        self.mode = mode
        self.artifact = None  # type: Optional[Dict[str, Any]]
        self.support = None  # type: Optional[Dict[str, Any]]
        self.schema = None  # type: Optional[Dict[str, Any]]

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
        assert self.artifact is not None and self.support is not None and self.schema is not None
        return package.recommend_one(state, self.artifact, self.support, schema=self.schema, mode=mode or self.mode)

    def predict_batch(self, input_data: Any, mode: Optional[str] = None) -> Any:
        self._ensure_loaded()
        assert self.artifact is not None and self.support is not None and self.schema is not None
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


def init(model_dir: Optional[Any] = None, mode: str = "production") -> SafeBandRecommender:
    return SafeBandRecommender(model_dir=model_dir, mode=mode).load()
'''


def main_py_source() -> str:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor-only calcium safe-band MVP runtime example.")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mode", choices=["production", "replay"], default="production")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--input-parquet", type=Path)
    parser.add_argument("--input-json", type=Path)
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


def read_input(args: argparse.Namespace) -> Any:
    if args.input_json:
        return read_json_rows(args.input_json)
    if args.input_csv:
        return read_csv_rows(args.input_csv)
    if args.input_parquet:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError("Parquet input requires pandas and pyarrow in the runtime environment.") from exc
        return pd.read_parquet(args.input_parquet)
    raise ValueError("Provide one of --input-json, --input-csv, or --input-parquet.")


def write_outputs(result: Any, args: argparse.Namespace) -> None:
    rows = result.to_dict(orient="records") if hasattr(result, "to_dict") else result
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
        with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    if args.output_parquet:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError("Parquet output requires pandas and pyarrow in the runtime environment.") from exc
        args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        frame = result if hasattr(result, "to_parquet") else pd.DataFrame(rows)
        frame.to_parquet(args.output_parquet, index=False)


def main() -> None:
    # Plant DCS fetch logic should be implemented by the plant adapter owner.
    # Plant writeback logic should be implemented by the plant adapter owner.
    # Current script does not write DCS and does not perform automatic control.
    args = parse_args()
    recommender = SafeBandRecommender(args.model_dir, mode=args.mode).load()
    input_data = read_input(args)
    result = recommender.predict_batch(input_data, mode=args.mode)
    write_outputs(result, args)
    count = len(result) if hasattr(result, "__len__") else 0
    print("Scored rows: {}".format(count))
    print("Mode: {}; monitor-only; no DCS writeback; no automatic control.".format(args.mode))


if __name__ == "__main__":
    main()
'''


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name).strip().lower())


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


def detect_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.module in {None, "package", "interface"}:
                continue
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def package_for_import(import_name: str) -> str:
    return normalize_package_name(IMPORT_TO_PACKAGE.get(import_name, import_name))


def dependency_check(requirements_path: Path, deploy_files: list[Path], available: set[str]) -> dict[str, Any]:
    third_party = {}
    package_third = []
    for path in deploy_files:
        if path.suffix != ".py":
            continue
        for imp in sorted(detect_imports(path)):
            if imp in STANDARD_IMPORTS or imp in {"package", "interface"}:
                continue
            third_party.setdefault(imp, []).append(str(path))
            if path.name == "package.py":
                package_third.append(imp)
    missing = sorted({imp for imp in third_party if package_for_import(imp) not in available})
    heavy = sorted({imp for imp in third_party if imp in HEAVY_IMPORTS})
    return {
        "requirements_path": str(requirements_path),
        "parsed_available_packages": sorted(available),
        "deploy_files_checked": [str(path) for path in deploy_files if path.suffix == ".py"],
        "third_party_imports_detected": third_party,
        "imports_not_in_requirements": missing,
        "unnecessary_heavy_runtime_imports": heavy,
        "package_py_standard_library_only": not package_third,
        "package_py_third_party_imports": sorted(package_third),
        "dependency_policy_pass": not missing and not package_third,
        "warnings": [f"Heavy runtime imports detected but allowed if listed: {heavy}"] if heavy else [],
    }


def build_support_from_existing(path: Path) -> dict[str, Any]:
    support = read_json_lenient(path)
    if not isinstance(support, dict):
        return {"created_at": datetime.now().isoformat(timespec="seconds"), "features": {}}
    features = support.get("features", {})
    for banned in ["t90", "y_ok", "y_low", "y_high", "y_out_spec"]:
        if isinstance(features, dict):
            features.pop(banned, None)
    if isinstance(features, dict) and "rubber_flow_2_win_60_mean" in features:
        features["rubber_flow_2_win_60_mean"]["feature_role"] = "feed_conversion_feature"
        features["rubber_flow_2_win_60_mean"]["required"] = False
    support["features"] = features
    return support


def build_schema(artifact: dict[str, Any], support: dict[str, Any]) -> dict[str, Any]:
    features = support.get("features", {}) if isinstance(support, dict) else {}
    required = [name for name, meta in features.items() if isinstance(meta, dict) and meta.get("required")]
    optional = [name for name, meta in features.items() if isinstance(meta, dict) and not meta.get("required")]
    output_schema = [
        "recommendation_status", "current_ca_consumption", "recommended_ca_consumption_min",
        "recommended_ca_consumption_max", "recommended_ca_consumption_target",
        "recommended_ca_feed_min", "recommended_ca_feed_max", "recommended_ca_feed_target",
        "interval_position", "action_hint", "action_visibility", "confidence_level",
        "matched_rule_count", "matched_rule_ids", "selected_rule_ids", "rule_evidence_ok_rate",
        "rule_evidence_high_rate", "rule_evidence_low_rate", "rule_evidence_sample_count",
        "engineering_review_required", "explanation_cn", "warning_flags", "input_valid",
        "missing_required_features", "model_version", "artifact_version",
    ]
    return {
        "package_name": "ca_safe_band_mvp",
        "package_version": PACKAGE_VERSION,
        "artifact_version": artifact.get("artifact_version"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_compatibility": "Python >= 3.8; package.py avoids PEP 604 union syntax.",
        "required_features": required,
        "optional_features": optional,
        "output_schema": output_schema,
        "action_visibility_policy": artifact.get("action_visibility_policy"),
        "safety_constraints": artifact.get("safety_constraints"),
        "dependency_policy": {
            "package_py_standard_library_only_expected": True,
            "third_party_dependencies_must_exist_in_IDB_requirements": True,
            "no_pickle": True,
        },
        "runtime_modes": {
            "production": "Default. Ignores input matched_rule_ids and selected_rule_ids; matches rules from current feature values and q33/q66 boundaries.",
            "replay": "For equivalence testing only. May use matched_rule_ids/selected_rule_ids from historical replay rows.",
        },
    }


def import_interface(deploy_dir: Path):
    spec = importlib.util.spec_from_file_location("repaired_ca_safe_band_interface", deploy_dir / "interface.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import generated interface.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(deploy_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(deploy_dir))
        except ValueError:
            pass
    return module


def close_num(a: Any, b: Any, tol: float = 1e-9) -> bool:
    av = None if pd.isna(a) else float(a) if str(a).strip().lower() not in {"none", "nan"} else None
    bv = None if pd.isna(b) else float(b) if str(b).strip().lower() not in {"none", "nan"} else None
    if av is None and bv is None:
        return True
    if av is None or bv is None:
        return False
    return abs(av - bv) <= tol


def same_str(a: Any, b: Any) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return str(a) == str(b)


def replay_equivalence(deploy_dir: Path, dry_run: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    module = import_interface(deploy_dir)
    rec = module.SafeBandRecommender(model_dir=deploy_dir, mode="replay").load()
    pred = rec.predict_batch(dry_run, mode="replay")
    rows = []
    for idx, expected in dry_run.reset_index(drop=True).iterrows():
        got = pred.iloc[idx]
        checks = {
            "interval_min_match": close_num(got.get("recommended_ca_consumption_min"), expected.get("recommended_ca_consumption_min")),
            "interval_max_match": close_num(got.get("recommended_ca_consumption_max"), expected.get("recommended_ca_consumption_max")),
            "interval_target_match": close_num(got.get("recommended_ca_consumption_target"), expected.get("recommended_ca_consumption_target")),
            "interval_position_match": same_str(got.get("interval_position"), expected.get("interval_position")),
            "action_hint_match": same_str(got.get("action_hint"), expected.get("final_action_hint")),
            "action_visibility_match": same_str(got.get("action_visibility"), expected.get("action_visibility")),
        }
        checks["all_core_fields_match"] = all(checks.values())
        checks["row_index"] = idx
        rows.append(checks)
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "runtime_equivalence_after_repair.csv", index=False, encoding="utf-8-sig")
    rate = float(audit["all_core_fields_match"].mean()) if len(audit) else 0.0
    report = {
        "tested_rows": int(len(audit)),
        "all_core_fields_match_rate": rate,
        "interval_min_match_rate": float(audit["interval_min_match"].mean()) if len(audit) else 0.0,
        "interval_max_match_rate": float(audit["interval_max_match"].mean()) if len(audit) else 0.0,
        "interval_target_match_rate": float(audit["interval_target_match"].mean()) if len(audit) else 0.0,
        "interval_position_match_rate": float(audit["interval_position_match"].mean()) if len(audit) else 0.0,
        "action_hint_match_rate": float(audit["action_hint_match"].mean()) if len(audit) else 0.0,
        "action_visibility_match_rate": float(audit["action_visibility_match"].mean()) if len(audit) else 0.0,
        "pass_runtime_equivalence": rate >= 0.99,
        "warnings": [],
    }
    write_strict_json(output_dir / "runtime_equivalence_after_repair_report.json", sanitize_json_value(report, {"non_strict": 0}))
    return report


def production_sanity(deploy_dir: Path, dry_run: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    module = import_interface(deploy_dir)
    rec = module.SafeBandRecommender(model_dir=deploy_dir, mode="production").load()
    pred = rec.predict_batch(dry_run, mode="production")
    forbidden_blob = json.dumps(sanitize_json_value(pred.to_dict("records"), {"non_strict": 0}), ensure_ascii=False)
    required_cols = ["recommendation_status", "interval_position", "action_hint", "action_visibility", "input_valid", "missing_required_features"]
    valid = pred[required_cols].notna().all(axis=1) if all(c in pred.columns for c in required_cols) else pd.Series(False, index=pred.index)
    valid_output_rate = float(valid.mean()) if len(valid) else 0.0
    report = {
        "tested_rows": int(len(pred)),
        "valid_output_rate": valid_output_rate,
        "production_ignored_input_rule_ids": True,
        "no_increase_to_band_operational_hint_detected": FORBIDDEN_HINT not in forbidden_blob,
        "no_automatic_control_wording_detected": "automatic control" not in forbidden_blob.lower() and "自动控制" not in forbidden_blob,
        "pass_production_sanity": valid_output_rate >= 0.99 and FORBIDDEN_HINT not in forbidden_blob,
        "action_visibility_counts": pred["action_visibility"].value_counts(dropna=False).to_dict() if "action_visibility" in pred.columns else {},
    }
    write_strict_json(output_dir / "production_mode_sanity_report.json", sanitize_json_value(report, {"non_strict": 0}))
    return report


def write_method_doc(path: Path) -> None:
    text = """# 稳定钙单耗安全带 MVP 方法与数据流说明

## 1. 项目目标

本项目目标是围绕溴化工段 T90 合格风险，构建一个用于人工监测和工程复核的钙单耗安全带 MVP。系统输出的是推荐钙单耗区间和风险可见性，不输出固定设定值，也不保证 T90 必然合格。

## 2. 当前版本定位

当前版本定位为 `stable_safe_band_mvp`。它来自历史离线实验和验证，最终采用 `median_aggregation_baseline`。该版本不是强动态分工况处方系统，而是稳定安全带监测工具。

## 3. 方法概述

离线阶段识别出历史上 T90 合格率更高、低/高 T90 风险更低的钙单耗区间。运行阶段根据当前工况匹配规则，并对多个匹配规则的推荐区间做中位数聚合，形成最终推荐区间。

## 4. 数据流

离线数据流：原始 DCS/LIMS/IR 数据 -> 清洗与滞后对齐 -> 钙单耗特征 `ca_per_rubber_flow_win_60_mean` -> 分工况规则 -> `safe_band_artifact.json` / `support.json` / `schema.json`。

运行数据流：厂方当前过程状态 -> `interface.py` -> `package.py` -> 规则匹配 -> 中位数区间聚合 -> 区间位置判断 -> 监测/人工复核输出 -> 厂方适配器决定如何展示或存储。

## 5. 输入字段

核心输入包括当前钙单耗 `ca_per_rubber_flow_win_60_mean` 或 `current_ca_consumption`，以及规则使用的工况变量。`rubber_flow_2_win_60_mean` 可用于把推荐钙单耗区间换算为加注量区间。

## 6. 输出字段

主要输出包括推荐钙单耗区间、推荐中心值、区间位置、动作可见性、人工复核标记、规则证据、加注量换算结果、输入有效性和缺失字段列表。

## 7. 当前所属工况如何判定

每条规则包含 `regime_feature` 和 `regime_bin`。`regime_bin` 为 low/mid/high。边界来自 `support.json` 中的 q33/q66：value <= q33 为 low，q33 < value <= q66 为 mid，value > q66 为 high。生产模式忽略输入的 `selected_rule_ids` 和 `matched_rule_ids`。

## 8. 推荐钙单耗区间如何生成

匹配规则包含 `recommended_dose_min` 和 `recommended_dose_max`。最终区间取所有匹配规则区间上下限的中位数，推荐中心值为最终区间中点。当前钙单耗与该区间比较后得到 inside/above/below/missing。

## 9. 历史窗口和滞后如何使用

主钙单耗特征为 `ca_per_rubber_flow_win_60_mean`，代表历史 60 分钟窗口的归一化钙单耗，不是瞬时值。钙单耗由硬脂酸钙加注量除以胶液流量得到。离线特征构造中，上游 DCS 变量已按约 3 小时总停留时间和各设备滞后先验对齐。运行包不重新计算原始滞后特征，要求厂方适配器提供已经准备好的当前特征值。

## 10. IR-lag 如何使用

IR-lag 特征 `output_ir_corrected_offset_20_win_15_std` 表示 T-20min 对齐、尾随 15 分钟出口红外标准差，是出口质量状态波动代理。它仅作为诊断或上下文信息，不作为独立动作触发变量。

## 11. 动作可见性策略

inside_band 输出 `hold_in_band`，仅监测展示。above_band 输出 `above_band_manual_review`，需要人工复核。below_band 输出 `below_band_diagnostic_only`，仅诊断展示，不给出加钙操作建议。missing 不生成推荐。

## 12. 为什么不是自动控制

该系统基于历史观察数据，不是因果证明；T90 是稀疏 LIMS 标签且有测量误差；因此输出只用于监测和人工复核，不执行自动控制、不写 DCS、不推荐影子试验。

## 13. 为什么上线前可以使用全量历史数据作为参考

离线验证完成后，全量既有历史数据可作为上线前参考支持，用于规则边界、特征范围、规则证据和工程复核。但它不能被重新描述为验证性能。验证证据仍以 Stage 20-24 的历史离线验证为准。

## 14. 如何接入厂方 main.py

厂方适配器负责从实时系统读取并计算当前特征，调用 `interface.SafeBandRecommender.predict_one` 或 `predict_batch`。`main.py` 只是示例 CLI，支持 JSON/CSV/parquet 输入输出，不包含 DCS 读取和写回逻辑。

## 15. 运行包目录结构

`package.py` 包含纯推荐逻辑；`interface.py` 提供公共 API；`main.py` 是示例入口；`safe_band_artifact.json` 保存规则；`support.json` 和 `support.parquet` 保存特征支持信息；`schema.json` 保存输入输出契约。

## 16. 安全约束和限制

monitor_only = true；automatic_control = false；dcs_writeback = false；increase_hint_hidden = true；engineering_review_required = true；no_guarantee_t90_qualified = true。该包仍需工程人工复核和在线监测验证。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_experiment_doc(path: Path, report: dict[str, Any], method_doc: Path) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    section_no = 28
    while f"## {section_no}." in existing:
        section_no += 1
    section = f"""

## {section_no}. 运行包生产安全修复与方法说明文档固化

### {section_no}.1 修复原因

本阶段针对稳定钙单耗安全带 MVP 运行包做生产安全修复：严格 JSON、生产模式不信任输入规则 ID、必需特征校验、输出 schema 扩展、加注量换算、Python 3.8+ 兼容和方法文档固化。

### {section_no}.2 修复结果

- 严格 JSON：{report.get('strict_json_pass')}
- 依赖策略：{report.get('dependency_policy_pass')}
- package.py 标准库-only：{report.get('package_py_standard_library_only')}
- replay 等价测试：{report.get('replay_pass_runtime_equivalence')}
- 生产模式有效输出率：{report.get('production_mode_valid_output_rate')}
- 生产模式禁用输入 rule-id override：{report.get('rule_id_override_disabled_in_production')}
- 输出 schema 扩展：{report.get('output_schema_expanded')}
- 加注量换算：{report.get('feed_conversion_enabled')}

方法说明文档：`{method_doc}`。

推荐下一步：`{report.get('recommended_next_step')}`。

局限性：仍需工程人工复核；厂方实时适配器尚未实现；尚无在线验证；该安全带关系不是因果证明。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(section)


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requirements = resolve_path(args.requirements, required=True, roots=[Path(".")], warnings=warnings)
    deploy_dir = resolve_path(args.deploy_dir, required=True, roots=[Path("deploy")], warnings=warnings)
    dry_run_path = resolve_path(args.final_dry_run, required=True, roots=[Path("runs")], warnings=warnings)
    final_rule_path = resolve_path(args.final_rule_summary, required=True, roots=[Path("runs")], warnings=warnings)
    manual_review_path = resolve_path(args.manual_review_sheet, required=False, roots=[Path("reports"), Path("runs")], warnings=warnings)

    available = parse_requirements(requirements)
    if "pandas" not in available:
        raise RuntimeError("pandas is required by the repair script and absent from IDB_requirements.txt")

    artifact_path = deploy_dir / "safe_band_artifact.json"
    support_path = deploy_dir / "support.json"
    schema_path = deploy_dir / "schema.json"
    json_paths = [artifact_path, support_path, schema_path]
    before, payloads = strict_json_check(json_paths)

    artifact = payloads[str(artifact_path)]
    support = build_support_from_existing(support_path)
    schema = build_schema(artifact, support)
    artifact["artifact_version"] = artifact.get("artifact_version") or "1.0.0"
    artifact["runtime_policy"] = {
        "default_mode": "production",
        "production_ignores_input_rule_ids": True,
        "replay_mode_allows_rule_id_override_for_equivalence_only": True,
    }

    write_strict_json(artifact_path, sanitize_json_value(artifact, {"non_strict": 0}))
    write_strict_json(support_path, sanitize_json_value(support, {"non_strict": 0}))
    write_strict_json(schema_path, sanitize_json_value(schema, {"non_strict": 0}))

    (deploy_dir / "package.py").write_text(package_py_source(), encoding="utf-8")
    (deploy_dir / "interface.py").write_text(interface_py_source(), encoding="utf-8")
    (deploy_dir / "main.py").write_text(main_py_source(), encoding="utf-8")

    final_rules = pd.read_csv(final_rule_path)
    support_rows = []
    for name, meta in support.get("features", {}).items():
        row = {"feature_name": name}
        row.update(meta)
        support_rows.append(row)
    pd.DataFrame(support_rows).to_parquet(deploy_dir / "support.parquet", index=False)

    after, _ = strict_json_check(json_paths)
    invalid_after = []
    after_counts = {}
    for p in json_paths:
        try:
            txt = p.read_text(encoding="utf-8")
            json.loads(txt)
            json.dumps(json.loads(txt), allow_nan=False)
            after_counts[str(p)] = 0
        except Exception:
            invalid_after.append(str(p))
    strict_report = {
        "files_checked": [str(p) for p in json_paths],
        "invalid_json_files": invalid_after,
        "non_strict_values_found_before_repair": before["non_strict_before"],
        "non_strict_values_found_after_repair": after_counts,
        "strict_json_pass": not invalid_after and all(v == 0 for v in after_counts.values()),
    }
    write_strict_json(args.output_dir / "strict_json_check_report.json", strict_report)

    dep = dependency_check(requirements, [deploy_dir / "package.py", deploy_dir / "interface.py", deploy_dir / "main.py"], available)
    write_strict_json(args.output_dir / "dependency_check_after_repair.json", sanitize_json_value(dep, {"non_strict": 0}))

    dry_run = pd.read_parquet(dry_run_path)
    replay_report = replay_equivalence(deploy_dir, dry_run, args.output_dir)
    prod_report = production_sanity(deploy_dir, dry_run, args.output_dir)
    write_method_doc(args.method_doc)

    forbidden_blob = "\n".join((deploy_dir / f).read_text(encoding="utf-8") for f in ["package.py", "interface.py", "main.py"])
    no_forbidden = FORBIDDEN_HINT not in forbidden_blob
    recommended = "human_review_repaired_runtime_package"
    if not dep["dependency_policy_pass"]:
        recommended = "fix_dependency_policy"
    elif not strict_report["strict_json_pass"] or not replay_report["pass_runtime_equivalence"] or not prod_report["pass_production_sanity"] or not no_forbidden:
        recommended = "fix_runtime_repair"
    repair_report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "deploy_dir": str(deploy_dir),
        "output_dir": str(args.output_dir),
        "repaired_files": [str(deploy_dir / name) for name in ["package.py", "interface.py", "main.py", "safe_band_artifact.json", "support.json", "schema.json"]],
        "strict_json_check_report_path": str(args.output_dir / "strict_json_check_report.json"),
        "dependency_check_report_path": str(args.output_dir / "dependency_check_after_repair.json"),
        "runtime_equivalence_report_path": str(args.output_dir / "runtime_equivalence_after_repair_report.json"),
        "production_mode_sanity_report_path": str(args.output_dir / "production_mode_sanity_report.json"),
        "package_py_standard_library_only": dep["package_py_standard_library_only"],
        "dependency_policy_pass": dep["dependency_policy_pass"],
        "strict_json_pass": strict_report["strict_json_pass"],
        "replay_pass_runtime_equivalence": replay_report["pass_runtime_equivalence"],
        "production_mode_valid_output_rate": prod_report["valid_output_rate"],
        "required_feature_validation_enabled": True,
        "rule_id_override_disabled_in_production": True,
        "output_schema_expanded": True,
        "feed_conversion_enabled": True,
        "safety_constraints": artifact.get("safety_constraints"),
        "warnings": warnings + dep.get("warnings", []),
        "assumptions": [
            "Replay mode is only for equivalence testing.",
            "Production mode ignores input rule IDs and requires current regime features.",
            "Full-history support is reference data, not new validation performance.",
        ],
        "recommended_next_step": recommended,
        "python_version": sys.version,
        "platform": sys.platform,
    }
    write_strict_json(args.output_dir / "runtime_repair_report.json", sanitize_json_value(repair_report, {"non_strict": 0}))
    append_experiment_doc(args.experiment_doc, repair_report, args.method_doc)

    print("Calcium safe-band runtime repair summary")
    print(f"repaired files: {repair_report['repaired_files']}")
    print(f"strict_json_pass: {repair_report['strict_json_pass']}")
    print(f"dependency_policy_pass: {repair_report['dependency_policy_pass']}")
    print(f"package_py_standard_library_only: {repair_report['package_py_standard_library_only']}")
    print(f"replay all_core_fields_match_rate: {replay_report['all_core_fields_match_rate']}")
    print(f"production valid_output_rate: {prod_report['valid_output_rate']}")
    print("required_feature_validation_enabled: True")
    print("rule_id_override_disabled_in_production: True")
    print("output_schema_expanded: True")
    print("feed_conversion_enabled: True")
    print(f"method doc path: {args.method_doc}")
    print(f"recommended_next_step: {recommended}")
    print(f"Documentation appended: {args.experiment_doc}")
    print("No generated outputs were written under data/.")


if __name__ == "__main__":
    main()
