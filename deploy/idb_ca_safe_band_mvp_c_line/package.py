from __future__ import annotations

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
