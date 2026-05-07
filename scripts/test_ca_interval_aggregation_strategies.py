from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


T90_LOW = 8.20
T90_HIGH = 8.70
STRATEGIES = [
    "median_aggregation_baseline",
    "top_rule_only",
    "weighted_rule_average",
    "narrow_intersection_if_overlap",
]
POSITION_ORDER = ["inside_band", "below_band", "above_band", "missing"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare calcium interval aggregation strategies.")
    parser.add_argument("--replay", type=Path, default=Path("runs/ca_interval_recommender_replay.parquet"))
    parser.add_argument("--oracle", type=Path, default=Path("runs/ca_interval_recommender_validation_oracle.csv"))
    parser.add_argument("--rules", type=Path, default=Path("runs/ca_regime_calcium_band_rules_ir_lag.csv"))
    parser.add_argument("--rule-audit", type=Path, default=Path("runs/ca_interval_recommender_rule_audit.csv"))
    parser.add_argument("--diversity-report", type=Path, default=Path("runs/ca_interval_diversity_audit/ca_interval_diversity_audit_report.json"))
    parser.add_argument("--artifact", type=Path, default=Path("models/ca_interval_recommender/rule_artifact.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ca_interval_aggregation_strategy_test"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    return parser.parse_args()


def as_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [as_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return None if math.isnan(val) else val
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130


def find_by_name(name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None


def resolve_path(path: Path, *, required: bool, search_roots: list[Path], warnings: list[str]) -> Path | None:
    if path.exists():
        return path
    found = find_by_name(path.name, search_roots)
    if found:
        warnings.append(f"Input {path} not found; using recursive match {found}.")
        return found
    if required:
        raise FileNotFoundError(f"Required input file not found: {path}. Searched {[str(p) for p in search_roots]}.")
    warnings.append(f"Optional input file not found: {path}.")
    return None


def load_json(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_table(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def numeric_series(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")
    return pd.to_numeric(data[column], errors="coerce")


def ensure_targets(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "t90" not in data.columns:
        return data
    t90 = numeric_series(data, "t90")
    if "y_ok" not in data.columns:
        data["y_ok"] = ((t90 >= T90_LOW) & (t90 <= T90_HIGH)).astype(int)
    if "y_low" not in data.columns:
        data["y_low"] = (t90 < T90_LOW).astype(int)
    if "y_high" not in data.columns:
        data["y_high"] = (t90 > T90_HIGH).astype(int)
    if "y_out_spec" not in data.columns:
        data["y_out_spec"] = ((t90 < T90_LOW) | (t90 > T90_HIGH)).astype(int)
    return data


def parse_rule_ids(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    for sep in [";", ",", "|"]:
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def grade_rank(value: object) -> int:
    return {"A": 0, "B": 1, "C": 2}.get(str(value), 9)


def build_rule_map(rules: pd.DataFrame) -> dict[str, dict[str, object]]:
    data = rules.copy()
    data["recommended_target"] = (numeric_series(data, "recommended_dose_min") + numeric_series(data, "recommended_dose_max")) / 2.0
    return {str(row["rule_id"]): row.to_dict() for _, row in data.iterrows() if "rule_id" in data.columns and pd.notna(row.get("rule_id"))}


def priority_key(rule: dict[str, object]) -> tuple[object, ...]:
    return (
        grade_rank(rule.get("rule_grade")),
        0 if boolish(rule.get("time_stable")) else 1,
        -float(rule.get("sample_count", 0) or 0),
        -float(rule.get("ok_lift_vs_overall", 0) or 0),
        float(rule.get("high_delta_vs_overall", 0) or 0),
        float(rule.get("low_delta_vs_overall", 0) or 0),
        float(rule.get("out_spec_delta_vs_overall", 0) or 0),
        str(rule.get("rule_id")),
    )


def selected_rule_ids(row: pd.Series) -> list[str]:
    ids = parse_rule_ids(row.get("selected_rule_ids"))
    if ids:
        return ids
    return parse_rule_ids(row.get("matched_rule_ids"))


def rule_interval(rule: dict[str, object]) -> tuple[float, float] | None:
    lo = pd.to_numeric(pd.Series([rule.get("recommended_dose_min")]), errors="coerce").iloc[0]
    hi = pd.to_numeric(pd.Series([rule.get("recommended_dose_max")]), errors="coerce").iloc[0]
    if pd.notna(lo) and pd.notna(hi):
        return float(lo), float(hi)
    return None


def top_rule(rules: list[dict[str, object]]) -> dict[str, object] | None:
    if not rules:
        return None
    return sorted(rules, key=priority_key)[0]


def weighted_interval(rules: list[dict[str, object]]) -> tuple[float, float] | None:
    weights = []
    lows = []
    highs = []
    for rule in rules:
        interval = rule_interval(rule)
        if interval is None:
            continue
        grade_weight = {"A": 3.0, "B": 2.0, "C": 1.0}.get(str(rule.get("rule_grade")), 0.5)
        sample_weight = math.log1p(max(float(rule.get("sample_count", 0) or 0), 0.0))
        lift_weight = max(float(rule.get("ok_lift_vs_overall", 0) or 0), 0.001)
        high_penalty = max(float(rule.get("high_delta_vs_overall", 0) or 0), 0.0) * 10.0
        low_penalty = max(float(rule.get("low_delta_vs_overall", 0) or 0), 0.0) * 20.0
        weight = grade_weight * sample_weight * lift_weight / (1.0 + high_penalty + low_penalty)
        weights.append(max(weight, 1e-9))
        lows.append(interval[0])
        highs.append(interval[1])
    if not weights:
        return None
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float(np.dot(w, np.asarray(lows))), float(np.dot(w, np.asarray(highs)))


def intersection_interval(rules: list[dict[str, object]]) -> tuple[float, float] | None:
    intervals = [rule_interval(rule) for rule in rules]
    intervals = [item for item in intervals if item is not None]
    if not intervals:
        return None
    lo = max(item[0] for item in intervals)
    hi = min(item[1] for item in intervals)
    if lo <= hi:
        return float(lo), float(hi)
    top = top_rule(rules)
    return rule_interval(top) if top else None


def action_from_interval(current: float, lo: float, hi: float) -> str:
    if not np.isfinite(current) or not np.isfinite(lo) or not np.isfinite(hi):
        return "no_recommendation"
    if current < lo:
        return "increase_to_band"
    if current > hi:
        return "decrease_to_band"
    return "hold_in_band"


def position_from_interval(current: float, lo: float, hi: float) -> str:
    if not np.isfinite(current) or not np.isfinite(lo) or not np.isfinite(hi):
        return "missing"
    if current < lo:
        return "below_band"
    if current > hi:
        return "above_band"
    return "inside_band"


def parse_acceptable_bands(value: object, fallback_min: float, fallback_max: float) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    if isinstance(value, str) and value.strip() and value.strip().lower() not in {"nan", "none"}:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        lo = pd.to_numeric(pd.Series([item.get("dose_min")]), errors="coerce").iloc[0]
                        hi = pd.to_numeric(pd.Series([item.get("dose_max")]), errors="coerce").iloc[0]
                        if pd.notna(lo) and pd.notna(hi):
                            bands.append((float(lo), float(hi)))
        except Exception:
            pass
    if not bands and pd.notna(fallback_min) and pd.notna(fallback_max):
        bands.append((float(fallback_min), float(fallback_max)))
    return bands


def overlap_ratio(lo: float, hi: float, oracle_lo: float, oracle_hi: float) -> float:
    if not all(np.isfinite([lo, hi, oracle_lo, oracle_hi])) or oracle_hi <= oracle_lo:
        return np.nan
    overlap = max(0.0, min(hi, oracle_hi) - max(lo, oracle_lo))
    return overlap / (oracle_hi - oracle_lo)


def calculate_hits(row: pd.Series, lo: float, hi: float, target: float, action: str) -> dict[str, object]:
    oracle_min = pd.to_numeric(pd.Series([row.get("oracle_ca_band_min")]), errors="coerce").iloc[0]
    oracle_max = pd.to_numeric(pd.Series([row.get("oracle_ca_band_max")]), errors="coerce").iloc[0]
    oracle_target = pd.to_numeric(pd.Series([row.get("oracle_ca_target")]), errors="coerce").iloc[0]
    bands = parse_acceptable_bands(row.get("oracle_acceptable_bands"), oracle_min, oracle_max)
    ratios = [overlap_ratio(lo, hi, band_lo, band_hi) for band_lo, band_hi in bands]
    valid_ratios = [ratio for ratio in ratios if pd.notna(ratio)]
    band_hit = max(valid_ratios) >= 0.50 if valid_ratios else None
    relaxed = max(valid_ratios) > 0.0 if valid_ratios else None
    oracle_action = row.get("oracle_action")
    direction_hit = action == oracle_action if isinstance(oracle_action, str) and oracle_action else None
    if pd.notna(target) and pd.notna(oracle_target) and oracle_target != 0:
        rel_err = abs(float(target) - float(oracle_target)) / abs(float(oracle_target))
        target_3 = rel_err <= 0.03
        target_5 = rel_err <= 0.05
        target_10 = rel_err <= 0.10
    else:
        target_3 = target_5 = target_10 = None
    return {
        "band_hit": band_hit,
        "relaxed_band_hit": relaxed,
        "direction_hit": direction_hit,
        "target_hit_3pct": target_3,
        "target_hit_5pct": target_5,
        "target_hit_10pct": target_10,
    }


def merge_oracle(replay: pd.DataFrame, oracle: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    data = replay.copy()
    if oracle.empty:
        warnings.append("Oracle CSV unavailable; accuracy metrics will use oracle columns already present in replay only.")
        return data
    if "time" in data.columns and "time" in oracle.columns:
        data["time"] = pd.to_datetime(data["time"], errors="coerce")
        oracle = oracle.copy()
        oracle["time"] = pd.to_datetime(oracle["time"], errors="coerce")
        add_cols = [col for col in oracle.columns if col not in data.columns or col == "oracle_acceptable_bands"]
        data = data.merge(oracle[["time"] + [c for c in add_cols if c != "time"]], on="time", how="left", suffixes=("", "_oracle"))
        if "oracle_acceptable_bands_oracle" in data.columns and "oracle_acceptable_bands" not in data.columns:
            data["oracle_acceptable_bands"] = data["oracle_acceptable_bands_oracle"]
    elif len(oracle) == int((data.get("split", pd.Series("", index=data.index)).astype(str) == "test_like").sum()):
        test_idx = data.index[data["split"].astype(str) == "test_like"] if "split" in data.columns else data.tail(len(oracle)).index
        for col in oracle.columns:
            if col not in data.columns or col == "oracle_acceptable_bands":
                data.loc[test_idx, col] = oracle[col].to_numpy()
        warnings.append("Oracle joined by test_like row order because time columns were unavailable.")
    else:
        warnings.append("Oracle could not be joined by time or row index; accuracy metrics may be incomplete.")
    return data


def build_strategy_replay(replay: pd.DataFrame, rule_map: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for _, row in replay.iterrows():
        ids = selected_rule_ids(row)
        rules = [rule_map[rid] for rid in ids if rid in rule_map]
        current = pd.to_numeric(pd.Series([row.get("current_ca_consumption")]), errors="coerce").iloc[0]
        for strategy in STRATEGIES:
            source = "existing_replay"
            source_ids = ids
            if strategy == "median_aggregation_baseline":
                lo = pd.to_numeric(pd.Series([row.get("recommended_ca_consumption_min")]), errors="coerce").iloc[0]
                hi = pd.to_numeric(pd.Series([row.get("recommended_ca_consumption_max")]), errors="coerce").iloc[0]
            elif strategy == "top_rule_only":
                rule = top_rule(rules)
                interval = rule_interval(rule) if rule else None
                lo, hi = interval if interval else (np.nan, np.nan)
                source = "top_priority_rule"
                source_ids = [str(rule.get("rule_id"))] if rule else []
            elif strategy == "weighted_rule_average":
                interval = weighted_interval(rules)
                lo, hi = interval if interval else (np.nan, np.nan)
                source = "weighted_selected_rules"
            else:
                interval = intersection_interval(rules)
                lo, hi = interval if interval else (np.nan, np.nan)
                source = "intersection_or_top_rule"
                if rules:
                    inter = [rule_interval(rule) for rule in rules]
                    inter = [item for item in inter if item is not None]
                    if inter and max(item[0] for item in inter) <= min(item[1] for item in inter):
                        source_ids = ids
                    else:
                        rule = top_rule(rules)
                        source_ids = [str(rule.get("rule_id"))] if rule else []
            target = (lo + hi) / 2.0 if pd.notna(lo) and pd.notna(hi) else np.nan
            action = action_from_interval(float(current) if pd.notna(current) else np.nan, float(lo) if pd.notna(lo) else np.nan, float(hi) if pd.notna(hi) else np.nan)
            position = position_from_interval(float(current) if pd.notna(current) else np.nan, float(lo) if pd.notna(lo) else np.nan, float(hi) if pd.notna(hi) else np.nan)
            hits = calculate_hits(row, float(lo) if pd.notna(lo) else np.nan, float(hi) if pd.notna(hi) else np.nan, float(target) if pd.notna(target) else np.nan, action)
            out = {
                "strategy": strategy,
                "time": row.get("time"),
                "split": row.get("split"),
                "current_ca_consumption": current,
                "recommended_ca_consumption_min": lo,
                "recommended_ca_consumption_max": hi,
                "recommended_ca_consumption_target": target,
                "interval_width": hi - lo if pd.notna(lo) and pd.notna(hi) else np.nan,
                "interval_position": position,
                "action_hint": action,
                "matched_rule_count": len(ids),
                "strategy_selected_rule_ids": ";".join(source_ids),
                "strategy_rule_source": source,
                "confidence_level": row.get("confidence_level"),
            }
            for col in [
                "t90",
                "y_ok",
                "y_low",
                "y_high",
                "y_out_spec",
                "oracle_ca_band_min",
                "oracle_ca_band_max",
                "oracle_ca_target",
                "oracle_action",
                "oracle_acceptable_bands",
            ]:
                if col in row.index:
                    out[col] = row.get(col)
            out.update(hits)
            rows.append(out)
    return pd.DataFrame(rows)


def bool_mean(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.astype(bool).mean())


def rate(frame: pd.DataFrame, col: str) -> float | None:
    if col not in frame.columns or frame.empty:
        return None
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def strategy_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    split_values = ["all"]
    if "split" in data.columns:
        split_values.extend([split for split in ["train_like", "test_like"] if (data["split"].astype(str) == split).any()])
    for strategy in STRATEGIES:
        strategy_data = data.loc[data["strategy"] == strategy].copy()
        for split in split_values:
            frame = strategy_data if split == "all" else strategy_data.loc[strategy_data["split"].astype(str) == split]
            if frame.empty:
                continue
            has_rec = numeric_series(frame, "recommended_ca_consumption_min").notna() & numeric_series(frame, "recommended_ca_consumption_max").notna()
            rec = frame.loc[has_rec].copy()
            interval_keys = numeric_series(rec, "recommended_ca_consumption_min").round(9).astype(str) + " - " + numeric_series(rec, "recommended_ca_consumption_max").round(9).astype(str)
            top_counts = interval_keys.value_counts()
            target = numeric_series(rec, "recommended_ca_consumption_target")
            inside = rec.loc[rec["interval_position"] == "inside_band"]
            outside = rec.loc[rec["interval_position"].isin(["below_band", "above_band"])]
            above = rec.loc[rec["interval_position"] == "above_band"]
            below = rec.loc[rec["interval_position"] == "below_band"]
            inside_count = int(len(inside))
            outside_count = int(len(outside))
            guardrail = (
                inside_count >= 30
                and outside_count >= 30
                and (rate(inside, "y_high") is not None and rate(outside, "y_high") is not None and rate(inside, "y_high") <= rate(outside, "y_high"))
                and (rate(inside, "y_low") is not None and rate(outside, "y_low") is not None and rate(inside, "y_low") <= rate(outside, "y_low") + 0.005)
                and (rate(inside, "y_out_spec") is not None and rate(outside, "y_out_spec") is not None and rate(inside, "y_out_spec") <= rate(outside, "y_out_spec"))
            )
            rows.append({
                "strategy": strategy,
                "split": split,
                "sample_count": int(len(frame)),
                "recommendation_coverage": float(has_rec.mean()) if len(frame) else None,
                "unique_interval_count": int(interval_keys.nunique(dropna=True)),
                "top_5_interval_coverage": float(top_counts.head(5).sum() / len(rec)) if len(rec) else None,
                "recommended_target_median": float(target.median()) if target.notna().any() else None,
                "recommended_target_iqr": float(target.quantile(0.75) - target.quantile(0.25)) if target.notna().any() else None,
                "recommended_target_range": float(target.max() - target.min()) if target.notna().any() else None,
                "interval_width_mean": float(numeric_series(rec, "interval_width").mean()) if len(rec) else None,
                "interval_width_median": float(numeric_series(rec, "interval_width").median()) if len(rec) else None,
                "band_accuracy": bool_mean(rec["band_hit"]) if "band_hit" in rec.columns else None,
                "relaxed_band_accuracy": bool_mean(rec["relaxed_band_hit"]) if "relaxed_band_hit" in rec.columns else None,
                "direction_accuracy": bool_mean(rec["direction_hit"]) if "direction_hit" in rec.columns else None,
                "target_accuracy_3pct": bool_mean(rec["target_hit_3pct"]) if "target_hit_3pct" in rec.columns else None,
                "target_accuracy_5pct": bool_mean(rec["target_hit_5pct"]) if "target_hit_5pct" in rec.columns else None,
                "target_accuracy_10pct": bool_mean(rec["target_hit_10pct"]) if "target_hit_10pct" in rec.columns else None,
                "inside_band_count": inside_count,
                "below_band_count": int((rec["interval_position"] == "below_band").sum()),
                "above_band_count": int((rec["interval_position"] == "above_band").sum()),
                "inside_band_ok_rate": rate(inside, "y_ok"),
                "outside_band_ok_rate": rate(outside, "y_ok"),
                "inside_band_high_rate": rate(inside, "y_high"),
                "outside_band_high_rate": rate(outside, "y_high"),
                "inside_band_low_rate": rate(inside, "y_low"),
                "outside_band_low_rate": rate(outside, "y_low"),
                "inside_band_out_spec_rate": rate(inside, "y_out_spec"),
                "outside_band_out_spec_rate": rate(outside, "y_out_spec"),
                "above_band_high_rate": rate(above, "y_high"),
                "below_band_low_rate": rate(below, "y_low"),
                "risk_guardrail_pass": bool(guardrail),
            })
    return pd.DataFrame(rows)


def compare_strategies(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], str, str, str]:
    test = metrics.loc[metrics["split"] == "test_like"].copy()
    base = test.loc[test["strategy"] == "median_aggregation_baseline"]
    base_row = base.iloc[0].to_dict() if not base.empty else {}
    rows = []
    for _, row in test.iterrows():
        base_iqr = float(base_row.get("recommended_target_iqr") or 0)
        base_unique = float(base_row.get("unique_interval_count") or 0)
        iqr = float(row.get("recommended_target_iqr") or 0)
        unique = float(row.get("unique_interval_count") or 0)
        band_drop = float(base_row.get("band_accuracy") or 0) - float(row.get("band_accuracy") or 0)
        direction_drop = float(base_row.get("direction_accuracy") or 0) - float(row.get("direction_accuracy") or 0)
        diversity_recovered = (
            (base_iqr > 0 and iqr >= 1.5 * base_iqr) or (base_unique > 0 and unique >= 1.5 * base_unique)
        ) and band_drop <= 0.05 and direction_drop <= 0.05 and bool(row.get("risk_guardrail_pass"))
        rows.append({
            "strategy": row["strategy"],
            "split": "test_like",
            "target_iqr_ratio_vs_baseline": iqr / base_iqr if base_iqr > 0 else None,
            "unique_interval_ratio_vs_baseline": unique / base_unique if base_unique > 0 else None,
            "band_accuracy_delta_vs_baseline": float(row.get("band_accuracy") or 0) - float(base_row.get("band_accuracy") or 0),
            "direction_accuracy_delta_vs_baseline": float(row.get("direction_accuracy") or 0) - float(base_row.get("direction_accuracy") or 0),
            "diversity_recovered": bool(diversity_recovered),
            "risk_guardrail_pass": bool(row.get("risk_guardrail_pass")),
        })
    comparison = pd.DataFrame(rows)
    candidates = test.loc[
        (test["band_accuracy"].fillna(0) >= 0.70)
        & (test["direction_accuracy"].fillna(0) >= 0.70)
        & (test["risk_guardrail_pass"].fillna(False))
    ].copy()
    if not candidates.empty:
        candidates["sort_key"] = candidates["recommended_target_iqr"].fillna(0)
        best = candidates.sort_values(
            by=["sort_key", "band_accuracy", "direction_accuracy"],
            ascending=[False, False, False],
        ).iloc[0]
        best_strategy = str(best["strategy"])
    elif not test.empty:
        best = test.sort_values(by=["risk_guardrail_pass", "band_accuracy", "direction_accuracy"], ascending=[False, False, False]).iloc[0]
        best_strategy = str(best["strategy"])
    else:
        best_strategy = "insufficient_data"

    top_row = test.loc[test["strategy"] == "top_rule_only"]
    top_recovers = bool((comparison.loc[comparison["strategy"] == "top_rule_only", "diversity_recovered"].any()) if not comparison.empty else False)
    base_guardrail = bool(base_row.get("risk_guardrail_pass")) if base_row else False
    if best_strategy == "top_rule_only" and top_recovers:
        switch = "switch_to_top_rule_only"
        next_step = "update_monitor_artifact_with_selected_aggregation"
    elif not top_row.empty and top_recovers is False and base_guardrail:
        switch = "keep_median_aggregation"
        next_step = "keep_stable_safe_band_mvp"
    elif best_strategy == "weighted_rule_average":
        switch = "test_weighted_aggregation_further"
        next_step = "refine_rule_priority_logic"
    elif best_strategy == "narrow_intersection_if_overlap":
        switch = "use_hybrid_top_rule_for_high_confidence_only"
        next_step = "refine_rule_priority_logic"
    elif not comparison["diversity_recovered"].any() if not comparison.empty else True:
        switch = "build_multivariate_regime_rules"
        next_step = "build_multivariate_regime_rules"
    else:
        switch = "insufficient_data"
        next_step = "collect_more_diverse_regime_data"
    summary = {
        "baseline_test_like": base_row,
        "comparison_rows": rows,
        "diversity_recovered_strategies": comparison.loc[comparison.get("diversity_recovered", pd.Series(False, index=comparison.index)).fillna(False), "strategy"].tolist() if not comparison.empty else [],
    }
    return comparison, summary, best_strategy, switch, next_step


def plot_target_distribution(data: pd.DataFrame, path: Path) -> None:
    test = data.loc[data["split"].astype(str) == "test_like"].copy() if "split" in data.columns else data
    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy in STRATEGIES:
        values = numeric_series(test.loc[test["strategy"] == strategy], "recommended_ca_consumption_target").dropna()
        if values.empty:
            continue
        ax.hist(values, bins=24, alpha=0.38, label=strategy)
    ax.set_title("不同聚合策略的推荐钙单耗中心值分布")
    ax.set_xlabel("推荐钙单耗中心值")
    ax.set_ylabel("样本数")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_width_distribution(data: pd.DataFrame, path: Path) -> None:
    test = data.loc[data["split"].astype(str) == "test_like"].copy() if "split" in data.columns else data
    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy in STRATEGIES:
        values = numeric_series(test.loc[test["strategy"] == strategy], "interval_width").dropna()
        if values.empty:
            continue
        ax.hist(values, bins=24, alpha=0.38, label=strategy)
    ax.set_title("不同聚合策略的推荐区间宽度分布")
    ax.set_xlabel("推荐区间宽度")
    ax.set_ylabel("样本数")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_accuracy(metrics: pd.DataFrame, path: Path) -> None:
    test = metrics.loc[metrics["split"] == "test_like"].copy()
    x = np.arange(len(test))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.18, numeric_series(test, "band_accuracy"), width=0.36, label="band_accuracy")
    ax.bar(x + 0.18, numeric_series(test, "direction_accuracy"), width=0.36, label="direction_accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(test["strategy"], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("不同聚合策略的验证集推荐准确率")
    ax.set_ylabel("准确率")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_risk(metrics: pd.DataFrame, path: Path) -> None:
    test = metrics.loc[metrics["split"] == "test_like"].copy()
    x = np.arange(len(test))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 0.27, numeric_series(test, "inside_band_high_rate"), width=0.18, label="inside high_rate")
    ax.bar(x - 0.09, numeric_series(test, "outside_band_high_rate"), width=0.18, label="outside high_rate")
    ax.bar(x + 0.09, numeric_series(test, "inside_band_low_rate"), width=0.18, label="inside low_rate")
    ax.bar(x + 0.27, numeric_series(test, "outside_band_low_rate"), width=0.18, label="outside low_rate")
    ax.set_xticks(x)
    ax.set_xticklabels(test["strategy"], rotation=25, ha="right", fontsize=8)
    ax.set_title("不同聚合策略的区间内外 T90 风险")
    ax.set_ylabel("风险率")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_top_vs_median(data: pd.DataFrame, path: Path) -> None:
    test = data.loc[data["split"].astype(str) == "test_like"].copy() if "split" in data.columns else data
    base = test.loc[test["strategy"] == "median_aggregation_baseline", ["time", "recommended_ca_consumption_target"]].rename(columns={"recommended_ca_consumption_target": "median_target"})
    top = test.loc[test["strategy"] == "top_rule_only", ["time", "recommended_ca_consumption_target"]].rename(columns={"recommended_ca_consumption_target": "top_target"})
    merged = base.merge(top, on="time", how="inner")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(numeric_series(merged, "median_target"), numeric_series(merged, "top_target"), s=18, alpha=0.6)
    if not merged.empty:
        lo = min(merged["median_target"].min(), merged["top_target"].min())
        hi = max(merged["median_target"].max(), merged["top_target"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#C62828")
    ax.set_title("Top Rule 与中位数聚合推荐中心值对比")
    ax.set_xlabel("中位数聚合中心值")
    ax.set_ylabel("Top Rule 中心值")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def append_doc(doc_path: Path, test_metrics: pd.DataFrame, comparison: pd.DataFrame, best_strategy: str, switch: str, next_step: str, outputs: dict[str, list[str]]) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    existing = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""
    section_no = 24
    while f"## {section_no}." in existing:
        section_no += 1
    def metric_line(strategy: str) -> str:
        rows = test_metrics.loc[test_metrics["strategy"] == strategy]
        if rows.empty:
            return f"- {strategy}: 无测试集指标"
        row = rows.iloc[0]
        return (
            f"- {strategy}: band_accuracy={row.get('band_accuracy')}, "
            f"direction_accuracy={row.get('direction_accuracy')}, "
            f"target_iqr={row.get('recommended_target_iqr')}, "
            f"unique_interval_count={row.get('unique_interval_count')}, "
            f"risk_guardrail_pass={row.get('risk_guardrail_pass')}"
        )
    diversity = comparison[["strategy", "diversity_recovered"]].to_dict("records") if not comparison.empty else []
    section = f"""

## {section_no}. 钙单耗推荐区间聚合策略对比实验

### {section_no}.1 实验目的

Stage 23 显示推荐区间稳定的主要原因是多规则中位数聚合压缩。本阶段在不修改规则、不训练模型、不进行策略搜索的前提下，复用同一批匹配规则和验证 oracle，对比中位数聚合、最高优先级规则、加权平均和重叠交集四种输出方式。

### {section_no}.2 验证集指标

{metric_line('median_aggregation_baseline')}
{metric_line('top_rule_only')}
{metric_line('weighted_rule_average')}
{metric_line('narrow_intersection_if_overlap')}

多样性恢复判断：{diversity}

### {section_no}.3 风险与策略判断

最佳策略：`{best_strategy}`。

切换建议：`{switch}`。

推荐下一步：`{next_step}`。

本阶段只做离线 replay 对比。`increase_to_band` 或 `decrease_to_band` 均不能解释为自动控制动作，也不形成 DCS 写回或影子试验建议。

### {section_no}.4 输出文件

- 机器输出：{', '.join(outputs.get('machine', []))}
- 图像输出：{', '.join(outputs.get('figures', []))}
- 人工表格：{', '.join(outputs.get('tables', []))}

局限性：离线验证不能证明因果关系；oracle 来自验证集事实分箱；聚合策略切换仍需人工工程复核。
"""
    with doc_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(section)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    warnings: list[str] = []
    assumptions = [
        "This script only replays existing matched rules and does not modify recommender artifacts.",
        "No model training, policy grid search, automatic control, DCS writeback, or shadow-trial recommendation is performed.",
        "Accuracy metrics use replay/oracle factual validation and remain offline diagnostics.",
    ]
    runs_root = Path("runs")
    models_root = Path("models")
    replay_path = resolve_path(args.replay, required=True, search_roots=[runs_root], warnings=warnings)
    oracle_path = resolve_path(args.oracle, required=False, search_roots=[runs_root], warnings=warnings)
    rules_path = resolve_path(args.rules, required=True, search_roots=[runs_root], warnings=warnings)
    rule_audit_path = resolve_path(args.rule_audit, required=False, search_roots=[runs_root], warnings=warnings)
    diversity_report_path = resolve_path(args.diversity_report, required=False, search_roots=[runs_root], warnings=warnings)
    artifact_path = resolve_path(args.artifact, required=False, search_roots=[models_root, runs_root], warnings=warnings)

    replay = ensure_targets(read_table(replay_path))
    oracle = read_table(oracle_path)
    replay = merge_oracle(replay, oracle, warnings)
    rules = read_table(rules_path)
    _rule_audit = read_table(rule_audit_path)
    diversity_report = load_json(diversity_report_path)
    artifact = load_json(artifact_path)
    rule_map = build_rule_map(rules)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    strategy_replay = build_strategy_replay(replay, rule_map)
    metrics = strategy_metrics(strategy_replay)
    comparison, diversity_summary, best_strategy, switch, next_step = compare_strategies(metrics)

    replay_parquet = args.output_dir / "strategy_recommendation_replay.parquet"
    replay_csv = args.output_dir / "strategy_recommendation_replay.csv"
    metrics_csv = args.output_dir / "strategy_metrics.csv"
    comparison_csv = args.output_dir / "strategy_comparison_summary.csv"
    report_json = args.output_dir / "ca_interval_aggregation_strategy_report.json"
    strategy_replay.to_parquet(replay_parquet, index=False)
    strategy_replay.to_csv(replay_csv, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
    summary_table = args.table_dir / "ca_interval_aggregation_strategy_summary.csv"
    metrics.loc[metrics["split"] == "test_like"].to_csv(summary_table, index=False, encoding="utf-8-sig")

    figures = [
        args.figure_dir / "ca_aggregation_strategy_target_distribution.png",
        args.figure_dir / "ca_aggregation_strategy_interval_width.png",
        args.figure_dir / "ca_aggregation_strategy_accuracy.png",
        args.figure_dir / "ca_aggregation_strategy_risk.png",
        args.figure_dir / "ca_top_rule_vs_median_target_scatter.png",
    ]
    plot_target_distribution(strategy_replay, figures[0])
    plot_width_distribution(strategy_replay, figures[1])
    plot_accuracy(metrics, figures[2])
    plot_risk(metrics, figures[3])
    plot_top_vs_median(strategy_replay, figures[4])

    test_metrics = metrics.loc[metrics["split"] == "test_like"].copy()
    risk_guardrail_summary = {
        row["strategy"]: bool(row.get("risk_guardrail_pass")) for _, row in test_metrics.iterrows()
    }
    accuracy_summary = {
        row["strategy"]: {
            "band_accuracy": row.get("band_accuracy"),
            "direction_accuracy": row.get("direction_accuracy"),
            "target_accuracy_5pct": row.get("target_accuracy_5pct"),
        }
        for _, row in test_metrics.iterrows()
    }
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_paths": {
            "replay": str(replay_path),
            "oracle": str(oracle_path) if oracle_path else None,
            "rules": str(rules_path),
            "rule_audit": str(rule_audit_path) if rule_audit_path else None,
            "diversity_report": str(diversity_report_path) if diversity_report_path else None,
            "artifact": str(artifact_path) if artifact_path else None,
        },
        "output_dir": str(args.output_dir),
        "figure_dir": str(args.figure_dir),
        "table_dir": str(args.table_dir),
        "strategies_tested": STRATEGIES,
        "baseline_strategy": "median_aggregation_baseline",
        "strategy_metrics": test_metrics.to_dict("records"),
        "diversity_recovery_summary": diversity_summary,
        "stage23_context": {
            "classification": diversity_report.get("interpretation_classification"),
            "likely_reason": diversity_report.get("likely_reason_for_stable_interval"),
            "recommended_next_step": diversity_report.get("recommended_next_step"),
        },
        "risk_guardrail_summary": risk_guardrail_summary,
        "accuracy_summary": accuracy_summary,
        "best_strategy": best_strategy,
        "switch_recommendation": switch,
        "artifact_rule_count": artifact.get("accepted_rule_count") if artifact else None,
        "warnings": warnings,
        "assumptions": assumptions,
        "recommended_next_step": next_step,
        "generated_outputs": {
            "machine": [str(replay_parquet), str(replay_csv), str(metrics_csv), str(comparison_csv), str(report_json)],
            "figures": [str(path) for path in figures],
            "tables": [str(summary_table)],
        },
    }
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(report), handle, ensure_ascii=False, indent=2)

    append_doc(
        args.doc,
        test_metrics,
        comparison,
        best_strategy,
        switch,
        next_step,
        {"machine": report["generated_outputs"]["machine"], "figures": report["generated_outputs"]["figures"], "tables": report["generated_outputs"]["tables"]},
    )

    def brief(strategy: str) -> dict[str, object]:
        rows = test_metrics.loc[test_metrics["strategy"] == strategy]
        return rows.iloc[0].to_dict() if not rows.empty else {}

    base = brief("median_aggregation_baseline")
    top = brief("top_rule_only")
    print("Calcium interval aggregation strategy test summary")
    print(f"strategies tested: {', '.join(STRATEGIES)}")
    print(f"baseline band/direction accuracy: {base.get('band_accuracy')} / {base.get('direction_accuracy')}")
    print(f"top_rule_only band/direction accuracy: {top.get('band_accuracy')} / {top.get('direction_accuracy')}")
    print(f"baseline target IQR/range: {base.get('recommended_target_iqr')} / {base.get('recommended_target_range')}")
    print(f"top_rule_only target IQR/range: {top.get('recommended_target_iqr')} / {top.get('recommended_target_range')}")
    print(f"risk guardrail pass by strategy: {risk_guardrail_summary}")
    print(f"best_strategy: {best_strategy}")
    print(f"switch_recommendation: {switch}")
    print(f"recommended_next_step: {next_step}")
    print("Generated output paths:")
    for path in report["generated_outputs"]["machine"] + report["generated_outputs"]["figures"] + report["generated_outputs"]["tables"]:
        print(f"  {path}")
    print(f"Documentation appended: {args.doc}")


if __name__ == "__main__":
    main()
