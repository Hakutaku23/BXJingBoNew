from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


FEATURE_COLUMNS = [
    "ca_per_rubber_flow_win_60_mean",
    "rubber_flow_2_win_60_mean",
    "bromine_feed_win_60_mean",
    "tank_rubber_conc_win_60_mean",
    "esbo_feed_win_60_mean",
    "neutral_alkali_feed_win_60_mean",
    "r510a_temp_win_60_mean",
    "r511a_temp_win_60_mean",
    "r512a_temp_win_60_mean",
    "r513_temp_win_60_mean",
    "r514_temp_win_60_mean",
]

INVENTORY_NAMES = [
    "c_line_reference_feature_summary.csv",
    "c_line_reference_feature_quantiles.csv",
    "c_line_reference_manifest.json",
    "c_line_historical_reference_features.csv",
    "future_runtime_features.csv",
    "future_c_line_v1_recommendation_replay.csv",
    "future_t90_halogen_c_line_only.csv",
    "future_t90_backfill_aligned_one_to_one.csv",
    "future_vs_c_line_historical_feature_drift.csv",
]

TIME_CANDIDATES = [
    "time",
    "timestamp",
    "recommendation_time",
    "sample_time",
    "t90_time",
    "quality_time",
    "matched_t90_time",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit C-line reference library inclusion of 2026 future data.")
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--source-c-line-revalidation-dir", type=Path, required=True)
    parser.add_argument("--future-validation-dir", type=Path, required=True)
    parser.add_argument("--stage49-dir", type=Path, required=True)
    parser.add_argument("--runtime-deploy-dir", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--method-doc", type=Path, required=True)
    return parser.parse_args()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_for_semantic_compare(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_for_semantic_compare(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_semantic_compare(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def semantic_json_equal(path_a: Path | None, path_b: Path | None) -> bool | None:
    if path_a is None or path_b is None or not path_a.exists() or not path_b.exists():
        return None
    try:
        a = sanitize_for_semantic_compare(read_json(path_a))
        b = sanitize_for_semantic_compare(read_json(path_b))
    except Exception:
        return None
    return a == b


def search_file(root: Path, name: str) -> Path | None:
    if not root.exists():
        return None
    exact = root / name
    if exact.exists():
        return exact
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def read_table(path: Path | None, nrows: int | None = None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.DataFrame()


def find_time_column(df: pd.DataFrame) -> str | None:
    lower_map = {str(col).lower(): col for col in df.columns}
    for cand in TIME_CANDIDATES:
        if cand.lower() in lower_map:
            return str(lower_map[cand.lower()])
    for col in df.columns:
        name = str(col).lower()
        if "time" in name or "日期" in str(col) or "时间" in str(col):
            return str(col)
    return None


def time_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "row_count": 0,
            "time_column": None,
            "time_min": None,
            "time_max": None,
            "rows_before_2026": 0,
            "rows_2026_01": 0,
            "rows_2026_02": 0,
            "rows_2026_03": 0,
        }
    time_col = find_time_column(df)
    if time_col is None:
        return {
            "row_count": int(len(df)),
            "time_column": None,
            "time_min": None,
            "time_max": None,
            "rows_before_2026": 0,
            "rows_2026_01": 0,
            "rows_2026_02": 0,
            "rows_2026_03": 0,
        }
    times = pd.to_datetime(df[time_col], errors="coerce")
    valid = times.dropna()
    return {
        "row_count": int(len(df)),
        "time_column": time_col,
        "time_min": valid.min().isoformat() if not valid.empty else None,
        "time_max": valid.max().isoformat() if not valid.empty else None,
        "rows_before_2026": int((times < pd.Timestamp("2026-01-01")).sum()),
        "rows_2026_01": int(((times >= pd.Timestamp("2026-01-01")) & (times < pd.Timestamp("2026-02-01"))).sum()),
        "rows_2026_02": int(((times >= pd.Timestamp("2026-02-01")) & (times < pd.Timestamp("2026-03-01"))).sum()),
        "rows_2026_03": int(((times >= pd.Timestamp("2026-03-01")) & (times < pd.Timestamp("2026-04-01"))).sum()),
    }


def inventory_candidate_files(args: argparse.Namespace) -> pd.DataFrame:
    roots = [
        args.stage49_dir / "c_line_reference_library",
        Path("runs/c_line_idb_final_pre_go_live/c_line_reference_library"),
        args.source_c_line_revalidation_dir,
        args.future_validation_dir,
    ]
    paths: list[Path] = []
    for root in roots:
        for name in INVENTORY_NAMES:
            match = search_file(root, name)
            if match and match not in paths:
                paths.append(match)
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        file_type = path.suffix.lower().lstrip(".")
        df = read_table(path) if file_type in {"csv", "parquet"} else pd.DataFrame()
        summary = time_summary(df)
        text = ""
        if path.suffix.lower() == ".json":
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                payload = read_json(path)
                manifest_summary = payload.get("summary", payload)
                summary.update(
                    {
                        "row_count": manifest_summary.get("historical_source_rows") or manifest_summary.get("row_count") or 0,
                        "time_min": manifest_summary.get("reference_time_min") or manifest_summary.get("time_min"),
                        "time_max": manifest_summary.get("reference_time_max") or manifest_summary.get("time_max"),
                    }
                )
            except Exception:
                pass
        columns = list(df.columns) if not df.empty else []
        path_str = str(path).replace("\\", "/").lower()
        rows.append(
            {
                "file_path": str(path),
                "file_name": path.name,
                "file_type": file_type,
                "exists": path.exists(),
                "row_count": summary.get("row_count"),
                "time_min": summary.get("time_min"),
                "time_max": summary.get("time_max"),
                "columns": ";".join(map(str, columns[:80])),
                "likely_contains_future_2026_data": bool(
                    (summary.get("rows_2026_01", 0) or 0)
                    + (summary.get("rows_2026_02", 0) or 0)
                    + (summary.get("rows_2026_03", 0) or 0)
                    or "2026-03" in text
                    or "future_holdout_rows_added" in text
                ),
                "likely_runtime_decision_asset": False,
                "likely_reference_asset": "reference" in path_str or "future" in path_str,
                "note_cn": "参考库/审计候选文件" if ("reference" in path_str or "future" in path_str) else "C线历史源候选文件",
            }
        )
    return pd.DataFrame(rows)


def recursive_get_first(obj: Any, keys: set[str]) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in keys:
                return value
        for value in obj.values():
            found = recursive_get_first(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = recursive_get_first(value, keys)
            if found is not None:
                return found
    return None


def infer_rule_count(obj: dict[str, Any]) -> int | None:
    candidates = ["final_rules", "rules", "accepted_rules", "selected_rules", "rule_table"]
    for key in candidates:
        value = recursive_get_first(obj, {key})
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            return len(value)
    return None


def runtime_asset_freeze_audit(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    assets = [
        ("model_safe_band_artifact", args.model_artifact, None, True),
        ("deploy_safe_band_artifact", args.runtime_deploy_dir / "safe_band_artifact.json", args.model_artifact, True),
        ("deploy_support", args.runtime_deploy_dir / "support.json", Path("deploy/ca_safe_band_mvp_c_line/support.json"), True),
        ("deploy_schema", args.runtime_deploy_dir / "schema.json", Path("deploy/ca_safe_band_mvp_c_line/schema.json"), True),
    ]
    rows: list[dict[str, Any]] = []
    for name, path, source_path, expected_unchanged in assets:
        data: dict[str, Any] = {}
        warnings: list[str] = []
        if path.exists() and path.suffix.lower() == ".json":
            try:
                data = read_json(path)
            except Exception as exc:
                warnings.append(str(exc))
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        source_hash = sha256_file(source_path) if source_path and source_path.exists() else None
        current_hash = sha256_file(path)
        changed_vs_source = bool(source_hash and current_hash and source_hash != current_hash)
        sem_equal = semantic_json_equal(path, source_path) if source_path else None
        semantic_changed_vs_source = bool(sem_equal is False)
        strategy = data.get("final_strategy") or recursive_get_first(data, {"final_strategy", "strategy"})
        rule_count = infer_rule_count(data)
        contains_future_rows = bool(
            re.search(
                r"future_runtime_features|future_holdout|future_validation|2026[-./年](?:0?[123])(?:[-./月]|$)",
                text,
                flags=re.I,
            )
        )
        old_merged_ref = "deploy/ca_safe_band_mvp/" in text or "deploy\\ca_safe_band_mvp\\" in text
        status = "pass"
        notes: list[str] = []
        if not path.exists():
            status = "fail"
            notes.append("资产缺失")
        if expected_unchanged and changed_vs_source:
            notes.append("与源资产哈希不一致；需结合语义归一化判断")
        if expected_unchanged and semantic_changed_vs_source and name.endswith("safe_band_artifact"):
            status = "fail"
            notes.append("artifact 语义内容与源资产不一致")
        elif expected_unchanged and semantic_changed_vs_source:
            notes.append("非 artifact 运行支撑资产语义内容与源资产不完全一致，按元数据差异记录")
        if name.endswith("safe_band_artifact") and strategy not in {None, "top_rule_only"}:
            status = "fail"
            notes.append("策略不是 top_rule_only")
        if name.endswith("safe_band_artifact") and contains_future_rows:
            status = "fail"
            notes.append("运行决策资产疑似包含 future 行来源")
        if old_merged_ref:
            status = "fail"
            notes.append("疑似引用旧合并线包")
        rows.append(
            {
                "asset_name": name,
                "path": str(path),
                "sha256": current_hash,
                "source_path": str(source_path) if source_path else "",
                "source_sha256": source_hash,
                "exists": path.exists(),
                "strategy": strategy,
                "rule_count": rule_count,
                "changed_vs_source": changed_vs_source,
                "semantic_changed_vs_source": semantic_changed_vs_source,
                "contains_future_reference_rows": contains_future_rows,
                "expected_to_change": False,
                "status": status,
                "note_cn": "；".join(notes) if notes else "运行决策资产冻结状态符合预期",
                "warnings": ";".join(warnings),
            }
        )
    df = pd.DataFrame(rows)
    summary = {
        "asset_count": int(len(df)),
        "all_pass": bool(not df.empty and df["status"].eq("pass").all()),
        "runtime_artifact_unchanged": bool(
            not df[(df["asset_name"] == "deploy_safe_band_artifact") & (df["semantic_changed_vs_source"])]
            .astype(bool)
            .any()
            .any()
        ),
        "strategy_top_rule_only_unchanged": bool(
            df[df["asset_name"].str.contains("safe_band_artifact")]["strategy"].dropna().isin(["top_rule_only"]).all()
        ),
        "old_merged_package_used": bool(df["note_cn"].astype(str).str.contains("旧合并线").any()),
    }
    return df, summary


def load_reference_candidates(args: argparse.Namespace) -> dict[str, tuple[Path | None, pd.DataFrame]]:
    stage49_ref = search_file(args.stage49_dir / "c_line_reference_library", "c_line_historical_reference_features.csv")
    stage48_ref = search_file(Path("runs/c_line_idb_final_pre_go_live/c_line_reference_library"), "c_line_historical_reference_features.csv")
    historical = search_file(args.source_c_line_revalidation_dir, "t90_ca_feature_dataset_c_line.parquet")
    future_features = search_file(args.future_validation_dir, "future_runtime_features.csv") or search_file(args.future_validation_dir, "future_runtime_features.parquet")
    future_replay = search_file(args.future_validation_dir, "future_c_line_v1_recommendation_replay.csv") or search_file(
        args.future_validation_dir, "future_c_line_v1_recommendation_replay.parquet"
    )
    return {
        "stage49_reference_features": (stage49_ref, read_table(stage49_ref)),
        "stage48_reference_features": (stage48_ref, read_table(stage48_ref)),
        "historical_prelaunch_features": (historical, read_table(historical)),
        "future_runtime_features": (future_features, read_table(future_features)),
        "future_recommendation_replay": (future_replay, read_table(future_replay)),
    }


def source_period_counts(df: pd.DataFrame) -> dict[str, int]:
    for col in ["source_period", "reference_source", "source_dataset", "source_role"]:
        if col in df.columns:
            return {str(k): int(v) for k, v in df[col].value_counts(dropna=False).to_dict().items()}
    return {}


def reference_inclusion_audit(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any], dict[str, tuple[Path | None, pd.DataFrame]]]:
    candidates = load_reference_candidates(args)
    manifest_path = search_file(args.stage49_dir / "c_line_reference_library", "c_line_reference_manifest.json")
    manifest = read_json(manifest_path) if manifest_path and manifest_path.exists() else {}
    rows: list[dict[str, Any]] = []
    latest_row: dict[str, Any] | None = None
    for name, (path, df) in candidates.items():
        summary = time_summary(df)
        period_counts = source_period_counts(df)
        future_rows = int(summary["rows_2026_01"] + summary["rows_2026_02"] + summary["rows_2026_03"])
        has_future = future_rows > 0
        latest_data_included = bool(
            name == "stage49_reference_features"
            and has_future
            and summary["time_max"] is not None
            and pd.Timestamp(summary["time_max"]) >= pd.Timestamp("2026-03-01")
            and (
                "future" in " ".join(period_counts.keys()).lower()
                or manifest.get("summary", {}).get("latest_data_included") is True
                or manifest.get("summary", {}).get("future_holdout_rows_added", 0) > 0
            )
        )
        row = {
            "reference_candidate": name,
            "file_path": str(path) if path else "",
            "row_count": summary["row_count"],
            "time_min": summary["time_min"],
            "time_max": summary["time_max"],
            "rows_before_2026": summary["rows_before_2026"],
            "rows_2026_01": summary["rows_2026_01"],
            "rows_2026_02": summary["rows_2026_02"],
            "rows_2026_03": summary["rows_2026_03"],
            "has_source_period_column": any(col in df.columns for col in ["source_period", "reference_source", "source_dataset", "source_role"]),
            "source_period_counts": json.dumps(period_counts, ensure_ascii=False),
            "has_future_2026_data": has_future,
            "future_rows_added_estimate": future_rows,
            "latest_data_included": latest_data_included,
            "note_cn": "Stage49 最新参考库候选" if name == "stage49_reference_features" else "对照源",
        }
        if name == "stage49_reference_features":
            latest_row = row
        rows.append(row)
    df_out = pd.DataFrame(rows)
    latest_included = bool(latest_row and latest_row["latest_data_included"])
    latest_summary = latest_row or {}
    summary = {
        "latest_future_data_included": latest_included,
        "latest_future_data_missing_from_reference_library": not latest_included,
        "latest_reference_library_summary": latest_summary,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_latest_data_included": manifest.get("summary", {}).get("latest_data_included"),
        "manifest_future_holdout_rows_added": manifest.get("summary", {}).get("future_holdout_rows_added"),
    }
    return df_out, summary, candidates


def summary_diff(args: argparse.Namespace, candidates: dict[str, tuple[Path | None, pd.DataFrame]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    latest_summary_path = search_file(args.stage49_dir / "c_line_reference_library", "c_line_reference_feature_summary.csv")
    previous_summary_path = search_file(Path("runs/c_line_idb_final_pre_go_live/c_line_reference_library"), "c_line_reference_feature_summary.csv")
    latest = read_table(latest_summary_path)
    previous = read_table(previous_summary_path)
    rows: list[dict[str, Any]] = []
    latest_ref_path, latest_ref = candidates.get("stage49_reference_features", (None, pd.DataFrame()))
    prev_ref_path, prev_ref = candidates.get("stage48_reference_features", (None, pd.DataFrame()))
    future_path, future = candidates.get("future_runtime_features", (None, pd.DataFrame()))
    latest_ts = time_summary(latest_ref)
    prev_ts = time_summary(prev_ref)
    future_ts = time_summary(future)
    reference_features_identical = False
    if not latest_ref.empty and not prev_ref.empty:
        common_cols = [col for col in latest_ref.columns if col in prev_ref.columns]
        if common_cols and len(latest_ref) == len(prev_ref):
            reference_features_identical = bool(
                latest_ref[common_cols].reset_index(drop=True).equals(prev_ref[common_cols].reset_index(drop=True))
            )
    if previous.empty and not prev_ref.empty:
        previous = feature_quantiles(prev_ref)
        previous_summary_path = prev_ref_path
    if latest.empty and not latest_ref.empty:
        latest = feature_quantiles(latest_ref)
        latest_summary_path = latest_ref_path
    if not latest.empty and not previous.empty and "feature" in latest.columns and "feature" in previous.columns:
        latest_idx = latest.set_index("feature")
        prev_idx = previous.set_index("feature")
        for feature in FEATURE_COLUMNS:
            lrow = latest_idx.loc[feature] if feature in latest_idx.index else pd.Series(dtype=object)
            prow = prev_idx.loc[feature] if feature in prev_idx.index else pd.Series(dtype=object)
            identical = False
            median_delta = q25_delta = q75_delta = None
            if not lrow.empty and not prow.empty:
                common = [col for col in ["count", "min", "q01", "q05", "q25", "median", "q75", "q95", "q99", "max"] if col in latest.columns and col in previous.columns]
                identical = bool(lrow[common].equals(prow[common])) if common else False
                median_delta = numeric_or_none(lrow.get("median")) - numeric_or_none(prow.get("median")) if numeric_or_none(lrow.get("median")) is not None and numeric_or_none(prow.get("median")) is not None else None
                q25_delta = numeric_or_none(lrow.get("q25")) - numeric_or_none(prow.get("q25")) if numeric_or_none(lrow.get("q25")) is not None and numeric_or_none(prow.get("q25")) is not None else None
                q75_delta = numeric_or_none(lrow.get("q75")) - numeric_or_none(prow.get("q75")) if numeric_or_none(lrow.get("q75")) is not None and numeric_or_none(prow.get("q75")) is not None else None
            rows.append(
                {
                    "feature": feature,
                    "old_reference_count": numeric_or_none(prow.get("count")) if not prow.empty else None,
                    "latest_reference_count": numeric_or_none(lrow.get("count")) if not lrow.empty else None,
                    "future_rows_added": int(future_ts["row_count"] or 0),
                    "old_time_max": prev_ts["time_max"],
                    "latest_time_max": latest_ts["time_max"],
                    "quantile_changed": not identical,
                    "median_delta": median_delta,
                    "q25_delta": q25_delta,
                    "q75_delta": q75_delta,
                    "identical_summary_flag": identical,
                    "likely_not_rebuilt_flag": identical and prev_ts["time_max"] == latest_ts["time_max"],
                }
            )
    else:
        for feature in FEATURE_COLUMNS:
            rows.append(
                {
                    "feature": feature,
                    "old_reference_count": None,
                    "latest_reference_count": None,
                    "future_rows_added": int(future_ts["row_count"] or 0),
                    "old_time_max": prev_ts["time_max"],
                    "latest_time_max": latest_ts["time_max"],
                    "quantile_changed": None,
                    "median_delta": None,
                    "q25_delta": None,
                    "q75_delta": None,
                    "identical_summary_flag": None,
                    "likely_not_rebuilt_flag": None,
                }
            )
    out = pd.DataFrame(rows)
    identical_all = bool(not out.empty and out["identical_summary_flag"].fillna(False).all())
    likely_not_rebuilt = bool(not out.empty and out["likely_not_rebuilt_flag"].fillna(False).all())
    return out, {
        "previous_summary_path": str(previous_summary_path) if previous_summary_path else None,
        "latest_summary_path": str(latest_summary_path) if latest_summary_path else None,
        "previous_reference_path": str(prev_ref_path) if prev_ref_path else None,
        "latest_reference_path": str(latest_ref_path) if latest_ref_path else None,
        "future_feature_path": str(future_path) if future_path else None,
        "old_time_max": prev_ts["time_max"],
        "latest_time_max": latest_ts["time_max"],
        "future_rows_available": int(future_ts["row_count"] or 0),
        "identical_summary_all_features": identical_all,
        "reference_features_identical": reference_features_identical,
        "likely_not_rebuilt_flag": likely_not_rebuilt,
        "note_cn": "Stage49 与上一冻结参考摘要完全一致，说明 Stage49 可能复用了已有参考库；若该已有参考库已含 future，则不代表运行资产被更新。",
    }


def numeric_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def assign_source_period(times: pd.Series, default_historical: bool = False) -> pd.Series:
    dt = pd.to_datetime(times, errors="coerce")
    result = pd.Series("future_unknown_month", index=times.index, dtype=object)
    result.loc[dt < pd.Timestamp("2026-01-01")] = "historical_prelaunch"
    result.loc[(dt >= pd.Timestamp("2026-01-01")) & (dt < pd.Timestamp("2026-02-01"))] = "future_2026_01"
    result.loc[(dt >= pd.Timestamp("2026-02-01")) & (dt < pd.Timestamp("2026-03-01"))] = "future_2026_02"
    result.loc[(dt >= pd.Timestamp("2026-03-01")) & (dt < pd.Timestamp("2026-04-01"))] = "future_2026_03"
    if default_historical:
        result.loc[dt.isna()] = "historical_prelaunch"
    return result


def feature_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURE_COLUMNS:
        if feature not in df.columns:
            continue
        s = pd.to_numeric(df[feature], errors="coerce").dropna()
        if s.empty:
            rows.append({"feature": feature, "count": 0})
        else:
            rows.append(
                {
                    "feature": feature,
                    "count": int(s.count()),
                    "min": float(s.min()),
                    "q01": float(s.quantile(0.01)),
                    "q05": float(s.quantile(0.05)),
                    "q25": float(s.quantile(0.25)),
                    "median": float(s.median()),
                    "q75": float(s.quantile(0.75)),
                    "q95": float(s.quantile(0.95)),
                    "q99": float(s.quantile(0.99)),
                    "max": float(s.max()),
                }
            )
    return pd.DataFrame(rows)


def rebuild_reference_library_if_needed(
    args: argparse.Namespace,
    inclusion_summary: dict[str, Any],
    candidates: dict[str, tuple[Path | None, pd.DataFrame]],
) -> tuple[bool, dict[str, str], dict[str, Any]]:
    if not inclusion_summary.get("latest_future_data_missing_from_reference_library"):
        return False, {}, {"reason": "latest future data already included"}
    out_dir = args.output_dir / "corrected_c_line_reference_library"
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path, hist = candidates.get("historical_prelaunch_features", (None, pd.DataFrame()))
    future_path, future = candidates.get("future_runtime_features", (None, pd.DataFrame()))
    if hist.empty or future.empty:
        return False, {}, {"reason": "historical or future source unavailable"}
    keep_cols = ["time"] + [col for col in FEATURE_COLUMNS if col in hist.columns or col in future.columns]
    hist_ref = hist[[col for col in keep_cols if col in hist.columns]].copy()
    fut_ref = future[[col for col in keep_cols if col in future.columns]].copy()
    hist_ref["source_period"] = "historical_prelaunch"
    hist_ref["source_dataset"] = str(hist_path)
    hist_ref["source_role"] = "prelaunch_historical_reference"
    hist_ref["included_for_reference_only"] = True
    hist_ref["used_for_algorithm_update"] = False
    fut_ref["source_period"] = assign_source_period(fut_ref[find_time_column(fut_ref) or "time"])
    fut_ref["source_dataset"] = str(future_path)
    fut_ref["source_role"] = "future_holdout_reference_only"
    fut_ref["included_for_reference_only"] = True
    fut_ref["used_for_algorithm_update"] = False
    combined = pd.concat([hist_ref, fut_ref], ignore_index=True, sort=False)
    features_path = out_dir / "c_line_reference_features_with_source.csv"
    quant_path = out_dir / "c_line_reference_feature_quantiles.csv"
    summary_path = out_dir / "c_line_reference_summary.json"
    manifest_path = out_dir / "c_line_reference_manifest.json"
    combined.to_csv(features_path, index=False, encoding="utf-8-sig")
    quant = feature_quantiles(combined)
    quant.to_csv(quant_path, index=False, encoding="utf-8-sig")
    ts = time_summary(combined)
    source_counts = source_period_counts(combined)
    summary = {
        "historical_source_rows": int(len(hist_ref)),
        "future_holdout_rows_added": int(len(fut_ref)),
        "latest_data_included": bool(
            ts["time_max"] is not None
            and pd.Timestamp(ts["time_max"]) >= pd.Timestamp("2026-03-01")
            and ts["rows_2026_01"] + ts["rows_2026_02"] + ts["rows_2026_03"] > 0
        ),
        "use_for_algorithm_update": False,
        "use_for_reference_only": True,
        "reference_time_min": ts["time_min"],
        "reference_time_max": ts["time_max"],
        "source_period_counts": source_counts,
        "feature_columns": [col for col in FEATURE_COLUMNS if col in combined.columns],
    }
    write_json(summary_path, summary)
    write_json(
        manifest_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "assets": [
                {"asset_name": "c_line_reference_features_with_source", "path": str(features_path), "role": "monitoring_reference_only"},
                {"asset_name": "c_line_reference_feature_quantiles", "path": str(quant_path), "role": "monitoring_reference_only"},
                {"asset_name": "c_line_reference_summary", "path": str(summary_path), "role": "monitoring_reference_only"},
            ],
            "summary": summary,
        },
    )
    manifest_table = pd.DataFrame(
        [
            {
                "asset_name": "corrected_c_line_reference_features_with_source",
                "local_path": str(features_path),
                "suggested_s3fs_path": "/s3fs/ca_safe_band_mvp_c_line/reference/c_line_reference_features_with_source.csv",
                "required_for_basic_scoring": False,
                "required_for_monitoring_reference": True,
                "includes_future_2026_data": True,
                "source_periods": ";".join(source_counts.keys()),
                "file_format": "csv",
                "upload_required": True,
                "note_cn": "仅用于监测/漂移参考，不用于推荐算法更新",
            },
            {
                "asset_name": "corrected_c_line_reference_manifest",
                "local_path": str(manifest_path),
                "suggested_s3fs_path": "/s3fs/ca_safe_band_mvp_c_line/reference/c_line_reference_manifest.json",
                "required_for_basic_scoring": False,
                "required_for_monitoring_reference": True,
                "includes_future_2026_data": True,
                "source_periods": ";".join(source_counts.keys()),
                "file_format": "json",
                "upload_required": True,
                "note_cn": "修正参考库清单",
            },
        ]
    )
    manifest_table.to_csv(args.table_dir / "c_line_corrected_reference_manifest.csv", index=False, encoding="utf-8-sig")
    return True, {"features": str(features_path), "quantiles": str(quant_path), "summary": str(summary_path), "manifest": str(manifest_path)}, summary


def audited_s3fs_manifest(
    args: argparse.Namespace,
    inclusion_summary: dict[str, Any],
    corrected_created: bool,
    corrected_paths: dict[str, str],
) -> pd.DataFrame:
    ref_manifest = search_file(args.stage49_dir / "c_line_reference_library", "c_line_reference_manifest.json")
    latest_ref = search_file(args.stage49_dir / "c_line_reference_library", "c_line_historical_reference_features.csv")
    latest_summary = search_file(args.stage49_dir / "c_line_reference_library", "c_line_reference_feature_summary.csv")
    rows = [
        {
            "asset_name": "safe_band_artifact",
            "local_path": str(args.runtime_deploy_dir / "safe_band_artifact.json"),
            "suggested_s3fs_path": "/s3fs/ca_safe_band_mvp_c_line/safe_band_artifact.json",
            "required_for_basic_scoring": True,
            "required_for_monitoring_reference": False,
            "includes_future_2026_data": False,
            "source_periods": "",
            "file_format": "json",
            "upload_required": True,
            "note_cn": "运行决策资产，保持冻结",
        },
        {
            "asset_name": "support",
            "local_path": str(args.runtime_deploy_dir / "support.json"),
            "suggested_s3fs_path": "/s3fs/ca_safe_band_mvp_c_line/support.json",
            "required_for_basic_scoring": True,
            "required_for_monitoring_reference": False,
            "includes_future_2026_data": False,
            "source_periods": "",
            "file_format": "json",
            "upload_required": True,
            "note_cn": "运行支撑资产，保持冻结",
        },
        {
            "asset_name": "schema",
            "local_path": str(args.runtime_deploy_dir / "schema.json"),
            "suggested_s3fs_path": "/s3fs/ca_safe_band_mvp_c_line/schema.json",
            "required_for_basic_scoring": True,
            "required_for_monitoring_reference": False,
            "includes_future_2026_data": False,
            "source_periods": "",
            "file_format": "json",
            "upload_required": True,
            "note_cn": "运行 schema，保持冻结",
        },
    ]
    if corrected_created:
        rows.append(
            {
                "asset_name": "corrected_c_line_reference_features_with_source",
                "local_path": corrected_paths.get("features", ""),
                "suggested_s3fs_path": "/s3fs/ca_safe_band_mvp_c_line/reference/c_line_reference_features_with_source.csv",
                "required_for_basic_scoring": False,
                "required_for_monitoring_reference": True,
                "includes_future_2026_data": True,
                "source_periods": "historical_prelaunch;future_2026_01;future_2026_02;future_2026_03",
                "file_format": "csv",
                "upload_required": True,
                "note_cn": "修正后参考库，仅用于监测/漂移参考",
            }
        )
    else:
        rows.extend(
            [
                {
                    "asset_name": "latest_c_line_reference_features",
                    "local_path": str(latest_ref) if latest_ref else "",
                    "suggested_s3fs_path": "/s3fs/ca_safe_band_mvp_c_line/reference/c_line_historical_reference_features.csv",
                    "required_for_basic_scoring": False,
                    "required_for_monitoring_reference": True,
                    "includes_future_2026_data": bool(inclusion_summary.get("latest_future_data_included")),
                    "source_periods": "see_reference_source_or_manifest",
                    "file_format": "csv",
                    "upload_required": True,
                    "note_cn": "现有 Stage49 参考库，可用于监测参考",
                },
                {
                    "asset_name": "latest_c_line_reference_manifest",
                    "local_path": str(ref_manifest) if ref_manifest else "",
                    "suggested_s3fs_path": "/s3fs/ca_safe_band_mvp_c_line/reference/c_line_reference_manifest.json",
                    "required_for_basic_scoring": False,
                    "required_for_monitoring_reference": True,
                    "includes_future_2026_data": bool(inclusion_summary.get("latest_future_data_included")),
                    "source_periods": "see_manifest",
                    "file_format": "json",
                    "upload_required": True,
                    "note_cn": "参考库清单",
                },
                {
                    "asset_name": "latest_c_line_reference_feature_summary",
                    "local_path": str(latest_summary) if latest_summary else "",
                    "suggested_s3fs_path": "/s3fs/ca_safe_band_mvp_c_line/reference/c_line_reference_feature_summary.csv",
                    "required_for_basic_scoring": False,
                    "required_for_monitoring_reference": True,
                    "includes_future_2026_data": bool(inclusion_summary.get("latest_future_data_included")),
                    "source_periods": "summary_only",
                    "file_format": "csv",
                    "upload_required": True,
                    "note_cn": "参考分布摘要",
                },
            ]
        )
    return pd.DataFrame(rows)


def update_method_doc(path: Path) -> None:
    section = """

## C线历史参考库纳入性审计

运行推荐所需的 artifact、support 和 schema 应保持冻结，历史参考库与推荐决策资产分离管理。2026.1~2026.3 的 C线真实运行数据只能作为监测/参考数据纳入历史参考库，不更新规则、q33/q66 边界、artifact 或推荐区间。

纳入性核验应同时检查参考库 `time_max`、2026.1~2026.3 行数、source_period/source_dataset 或 manifest 记录、参考库行数增长，以及特征分位数是否发生合理变化。如果最新参考库与旧冻结参考内容完全一致，需要进一步区分：运行资产一致是预期；参考库完全一致则可能说明未重新构建，除非旧参考库本身已经包含 future 数据。

本阶段仅生成审计和必要时的修正参考库。修正参考库也只作为 s3fs 监测参考资产，不复制进运行决策包，不改变 C线 top_rule_only 推荐逻辑，不引入自动控制或 DCS 写回。
""".strip()
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    heading = "## C线历史参考库纳入性审计"
    if heading in text:
        pattern = re.compile(r"## C线历史参考库纳入性审计\n.*?(?=\n## |\Z)", re.S)
        text = pattern.sub(section, text)
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    path.write_text(text, encoding="utf-8")


def append_experiment_doc(path: Path, report: dict[str, Any]) -> str:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    base_title = "C线历史参考库纳入性审计与运行资产边界确认"
    existing_same = re.search(rf"^##\s+(\d+)\.\s+{re.escape(base_title)}\s*$", text, flags=re.M)
    if existing_same:
        number = int(existing_same.group(1))
    else:
        existing_numbers = [int(m.group(1)) for m in re.finditer(r"^##\s+(\d+)\.", text, flags=re.M)]
        number = 50
        while number in existing_numbers:
            number += 1
    title = f"## {number}. {base_title}"
    flags = report["final_decision_flags"]
    latest = report["latest_reference_library_summary"]
    section = f"""

{title}

- 目的：核验 Stage 49 之后 C线历史参考库是否真实纳入 2026.1~2026.3 future 数据，同时确认运行推荐资产仍保持冻结。
- 用户观察：最新生成的历史工况/参考内容看起来与冻结版本完全一致，因此本阶段区分“运行决策资产一致”和“参考库应包含最新 future 行”。
- 运行资产边界：`safe_band_artifact.json`、`support.json`、`schema.json` 用于评分，预期不因 future 数据改变。
- 审计结果：运行 artifact 未变更 = {flags.get("runtime_artifact_unchanged")}；规则未变更 = {flags.get("recommender_rules_unchanged")}；top_rule_only 未变更 = {flags.get("strategy_top_rule_only_unchanged")}。
- 最新参考库时间范围：{latest.get("time_min")} 至 {latest.get("time_max")}。
- 2026 行数：1月 {latest.get("rows_2026_01")}，2月 {latest.get("rows_2026_02")}，3月 {latest.get("rows_2026_03")}。
- latest future 数据是否已纳入参考库：{flags.get("latest_future_data_included_in_reference_library")}。
- 最新参考摘要是否与旧冻结摘要一致：{flags.get("latest_generated_reference_identical_to_frozen_reference")}。
- 是否创建修正参考库：{flags.get("corrected_reference_library_created")}。
- 算法边界：future 数据未用于算法更新 = {not flags.get("future_data_used_for_algorithm_update")}；未使用旧合并线包 = {not flags.get("old_merged_package_used")}。
- recommended_next_step：{report.get("recommended_next_step")}。
- 局限：参考库不是评分 artifact；future 数据仅作监测参考；本阶段不提供因果证明；不实现自动控制；不实现 DCS 写回。
""".strip()
    if existing_same:
        pattern = re.compile(rf"^##\s+{number}\.\s+{re.escape(base_title)}\s*\n.*?(?=^##\s+\d+\.|\Z)", re.S | re.M)
        text = pattern.sub(section + "\n\n", text).rstrip()
        path.write_text(text + "\n", encoding="utf-8")
    else:
        path.write_text(text.rstrip() + "\n\n" + section + "\n", encoding="utf-8")
    return title


def main() -> None:
    args = parse_args()
    ensure_dirs(args.output_dir, args.table_dir)

    inventory = inventory_candidate_files(args)
    inventory.to_csv(args.output_dir / "reference_file_inventory.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(args.table_dir / "c_line_reference_file_inventory.csv", index=False, encoding="utf-8-sig")

    asset_df, asset_summary = runtime_asset_freeze_audit(args)
    asset_df.to_csv(args.output_dir / "runtime_decision_asset_freeze_audit.csv", index=False, encoding="utf-8-sig")
    asset_df.to_csv(args.table_dir / "c_line_runtime_decision_asset_freeze_audit.csv", index=False, encoding="utf-8-sig")
    write_json(args.output_dir / "runtime_decision_asset_freeze_audit.json", asset_summary)

    inclusion_df, inclusion_summary, candidates = reference_inclusion_audit(args)
    inclusion_df.to_csv(args.output_dir / "reference_library_inclusion_audit.csv", index=False, encoding="utf-8-sig")
    inclusion_df.to_csv(args.table_dir / "c_line_reference_library_inclusion_audit.csv", index=False, encoding="utf-8-sig")

    diff_df, diff_summary = summary_diff(args, candidates)
    diff_df.to_csv(args.output_dir / "reference_summary_diff.csv", index=False, encoding="utf-8-sig")
    diff_df.to_csv(args.table_dir / "c_line_reference_summary_diff.csv", index=False, encoding="utf-8-sig")

    corrected_created, corrected_paths, corrected_summary = rebuild_reference_library_if_needed(args, inclusion_summary, candidates)

    s3fs_manifest = audited_s3fs_manifest(args, inclusion_summary, corrected_created, corrected_paths)
    s3fs_manifest.to_csv(args.table_dir / "c_line_s3fs_reference_asset_manifest_audited.csv", index=False, encoding="utf-8-sig")
    s3fs_manifest.to_csv(args.output_dir / "c_line_s3fs_reference_asset_manifest_audited.csv", index=False, encoding="utf-8-sig")

    latest_summary = inclusion_summary.get("latest_reference_library_summary", {})
    runtime_asset_changed = not asset_summary.get("all_pass", False)
    if runtime_asset_changed:
        recommended_next_step = "stop_due_to_runtime_asset_change"
    elif inclusion_summary.get("latest_future_data_included"):
        recommended_next_step = "use_existing_reference_library"
    elif corrected_created:
        recommended_next_step = "use_corrected_reference_library_for_s3fs_monitoring_reference"
    else:
        recommended_next_step = "fix_reference_library_build_logic"

    artifact_rows = asset_df[asset_df["asset_name"].str.contains("safe_band_artifact")]
    rule_counts = [x for x in artifact_rows["rule_count"].dropna().tolist()]
    recommender_rules_unchanged = len(set(rule_counts)) <= 1 and asset_summary.get("all_pass", False)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_paths": {
            "requirements": str(args.requirements),
            "source_c_line_revalidation_dir": str(args.source_c_line_revalidation_dir),
            "future_validation_dir": str(args.future_validation_dir),
            "stage49_dir": str(args.stage49_dir),
            "runtime_deploy_dir": str(args.runtime_deploy_dir),
            "model_artifact": str(args.model_artifact),
        },
        "output_dir": str(args.output_dir),
        "runtime_asset_freeze_summary": asset_summary,
        "reference_file_inventory_summary": {
            "file_count": int(len(inventory)),
            "future_2026_candidates": int(inventory["likely_contains_future_2026_data"].sum()) if not inventory.empty else 0,
        },
        "latest_reference_library_summary": latest_summary,
        "future_data_inclusion_summary": inclusion_summary,
        "reference_summary_diff_summary": diff_summary,
        "corrected_reference_library_created": corrected_created,
        "corrected_reference_library_paths": corrected_paths,
        "corrected_reference_library_summary": corrected_summary,
        "s3fs_reference_manifest_path": str(args.table_dir / "c_line_s3fs_reference_asset_manifest_audited.csv"),
        "final_decision_flags": {
            "runtime_artifact_unchanged": bool(asset_summary.get("runtime_artifact_unchanged")),
            "recommender_rules_unchanged": bool(recommender_rules_unchanged),
            "strategy_top_rule_only_unchanged": bool(asset_summary.get("strategy_top_rule_only_unchanged")),
            "latest_future_data_included_in_reference_library": bool(inclusion_summary.get("latest_future_data_included")),
            "latest_generated_reference_identical_to_frozen_reference": bool(
                diff_summary.get("identical_summary_all_features") or diff_summary.get("reference_features_identical")
            ),
            "corrected_reference_library_created": corrected_created,
            "future_data_used_for_algorithm_update": False,
            "old_merged_package_used": bool(asset_summary.get("old_merged_package_used")),
        },
        "warnings": [
            "Stage49 reference summary is identical to previous reference summary; this is acceptable only if the previous reference already included future rows."
            if diff_summary.get("identical_summary_all_features")
            else ""
        ],
        "limitations": [
            "Reference library is not a runtime scoring artifact.",
            "Future data is reference-only and must not update recommendation rules or thresholds.",
            "Timestamp-based future inclusion inference depends on a valid time column when explicit source_period is absent.",
            "No automatic control or DCS writeback is implemented.",
        ],
        "recommended_next_step": recommended_next_step,
    }
    report["warnings"] = [w for w in report["warnings"] if w]
    write_json(args.output_dir / "c_line_reference_library_inclusion_audit_report.json", report)

    update_method_doc(args.method_doc)
    appended_title = append_experiment_doc(args.doc, report)

    print("C-line reference library inclusion audit complete.")
    print(f"runtime_asset_freeze_pass={asset_summary.get('all_pass')}")
    print(f"latest_future_data_included={inclusion_summary.get('latest_future_data_included')}")
    print(f"corrected_reference_library_created={corrected_created}")
    print(f"recommended_next_step={recommended_next_step}")
    print(f"doc_section={appended_title}")


if __name__ == "__main__":
    main()
