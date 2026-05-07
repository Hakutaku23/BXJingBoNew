from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_TARGETS = ["t90", "y_ok", "y_low", "y_high", "y_out_spec"]
PRIMARY_DOSE_PRIORITY = [
    "ca_per_rubber_flow_win_60_mean",
    "ca_per_rubber_flow_lag_165",
    "ca_win_60_mean",
    "ca_lag_165",
]
SECONDARY_DOSE_FEATURES = [
    "ca_per_rubber_flow_win_60_mean",
    "ca_per_rubber_flow_lag_165",
    "ca_per_rubber_flow_win_60_slope",
    "ca_lag_165",
    "ca_win_15_mean",
    "ca_win_30_mean",
    "ca_win_60_mean",
    "ca_win_120_mean",
    "ca_win_60_slope",
    "ca_delta_15",
    "ca_delta_30",
    "ca_delta_60",
]
LEAKAGE_COLUMNS = {
    "time",
    "t90",
    "t90_C",
    "t90_D",
    "t90_E",
    "t90_label_count",
    "y_ok",
    "y_low",
    "y_high",
    "y_out_spec",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze calcium dose-response for T90 qualification.")
    parser.add_argument("--input", type=Path, default=Path("data/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--feature-report", type=Path, default=Path("data/t90_ca_feature_report.json"))
    parser.add_argument("--bins-output", type=Path, default=Path("data/t90_ca_dose_response_bins.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/t90_ca_dose_response_report.json"))
    parser.add_argument("--n-bins", type=int, default=5)
    return parser.parse_args()


def as_jsonable(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): as_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [as_jsonable(item) for item in value]
    return value


def load_inputs(input_path: Path, feature_report_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet does not exist: {input_path}")
    if not feature_report_path.exists():
        raise FileNotFoundError(f"Feature report JSON does not exist: {feature_report_path}")

    frame = pd.read_parquet(input_path)
    missing_targets = [column for column in REQUIRED_TARGETS if column not in frame.columns]
    if missing_targets:
        raise ValueError(f"Input dataset is missing required targets: {missing_targets}")

    with feature_report_path.open("r", encoding="utf-8") as handle:
        feature_report = json.load(handle)
    return frame, feature_report


def select_dose_features(frame: pd.DataFrame, feature_report: dict[str, object]) -> tuple[str, list[str], list[str]]:
    warnings: list[str] = []
    groups = feature_report.get("feature_groups", {})
    if not isinstance(groups, dict):
        groups = {}
        warnings.append("Feature report does not contain feature_groups; falling back to dataset columns.")

    calcium_core = groups.get("calcium_core_features", [])
    if not isinstance(calcium_core, list):
        calcium_core = []
        warnings.append("feature_groups.calcium_core_features is invalid; falling back to dataset columns.")

    allowed = set(str(feature) for feature in calcium_core) if calcium_core else set(frame.columns)
    allowed = allowed - LEAKAGE_COLUMNS
    available_secondary = [
        feature
        for feature in SECONDARY_DOSE_FEATURES
        if feature in frame.columns and feature in allowed
    ]
    primary = next(
        (
            feature
            for feature in PRIMARY_DOSE_PRIORITY
            if feature in frame.columns and feature in allowed
        ),
        None,
    )
    if primary is None:
        raise ValueError(
            "No primary dose feature is available from the priority list and calcium_core_features."
        )
    if primary not in available_secondary:
        available_secondary.insert(0, primary)

    context_features = set(groups.get("process_context_features", [])) if isinstance(groups.get("process_context_features"), list) else set()
    accidental_context = sorted(set(available_secondary) & context_features)
    if accidental_context:
        raise ValueError(f"Dose feature selection accidentally included process context features: {accidental_context}")

    return primary, available_secondary, warnings


def valid_corr(x: pd.Series, y: pd.Series, method: str) -> float | None:
    values = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(values) < 3 or values["x"].nunique() <= 1 or values["y"].nunique() <= 1:
        return None
    corr = values["x"].corr(values["y"], method=method)
    if corr is None or not np.isfinite(corr):
        return None
    return float(corr)


def make_quantile_bins(values: pd.Series, requested_bins: int) -> tuple[pd.Series, int, list[str]]:
    warnings: list[str] = []
    dose = pd.to_numeric(values, errors="coerce")
    usable = dose.dropna()
    if usable.empty:
        return pd.Series(pd.NA, index=values.index, dtype="Int64"), 0, ["No usable dose values for binning."]

    max_bins = min(requested_bins, int(usable.nunique()), int(len(usable)))
    if max_bins < 2:
        return pd.Series(0, index=values.index, dtype="Int64").where(dose.notna(), pd.NA), 1, [
            "Only one effective dose bin is possible because dose values have too little variation."
        ]

    for bins in range(max_bins, 1, -1):
        try:
            cut = pd.qcut(dose, q=bins, labels=False, duplicates="drop")
            effective = int(pd.Series(cut).dropna().nunique())
            if effective >= 2:
                if effective < requested_bins:
                    warnings.append(
                        f"Effective bin count {effective} is lower than requested {requested_bins} because of duplicate dose values."
                    )
                return pd.Series(cut, index=values.index, dtype="Int64"), effective, warnings
        except ValueError:
            continue

    ranked = dose.rank(method="first")
    fallback_bins = min(requested_bins, int(ranked.dropna().nunique()))
    if fallback_bins < 2:
        return pd.Series(0, index=values.index, dtype="Int64").where(dose.notna(), pd.NA), 1, [
            "Rank-based fallback still produced only one effective bin."
        ]
    cut = pd.qcut(ranked, q=fallback_bins, labels=False, duplicates="drop")
    effective = int(pd.Series(cut).dropna().nunique())
    warnings.append("Used rank-based fallback for quantile binning.")
    if effective < requested_bins:
        warnings.append(f"Effective bin count {effective} is lower than requested {requested_bins}.")
    return pd.Series(cut, index=values.index, dtype="Int64"), effective, warnings


def summarize_bins(frame: pd.DataFrame, dose_feature: str, bin_ids: pd.Series) -> pd.DataFrame:
    work = frame.copy()
    work["dose_feature"] = dose_feature
    work["bin_id"] = bin_ids
    work = work[work["bin_id"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    effective_count = int(work["bin_id"].nunique())
    rows: list[dict[str, object]] = []
    for bin_id, group in work.groupby("bin_id", sort=True):
        bin_id_int = int(bin_id)
        dose = pd.to_numeric(group[dose_feature], errors="coerce")
        row = {
            "dose_feature": dose_feature,
            "bin_id": bin_id_int,
            "bin_label": f"{bin_id_int}: [{dose.min():.6g}, {dose.max():.6g}]",
            "is_lowest_bin": bool(bin_id_int == 0),
            "is_highest_bin": bool(bin_id_int == effective_count - 1),
            "is_interior_bin": bool(0 < bin_id_int < effective_count - 1),
            "sample_count": int(len(group)),
            "dose_min": float(dose.min()),
            "dose_max": float(dose.max()),
            "dose_mean": float(dose.mean()),
            "dose_median": float(dose.median()),
            "t90_mean": float(group["t90"].mean()),
            "t90_median": float(group["t90"].median()),
            "t90_std": float(group["t90"].std(ddof=1)) if len(group) >= 2 else math.nan,
            "ok_count": int(group["y_ok"].sum()),
            "ok_rate": float(group["y_ok"].mean()),
            "low_count": int(group["y_low"].sum()),
            "low_rate": float(group["y_low"].mean()),
            "high_count": int(group["y_high"].sum()),
            "high_rate": float(group["y_high"].mean()),
            "out_spec_count": int(group["y_out_spec"].sum()),
            "out_spec_rate": float(group["y_out_spec"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def get_endpoint(table: pd.DataFrame, column: str, highest: bool) -> float | None:
    if table.empty or column not in table.columns:
        return None
    row = table.sort_values("bin_id").iloc[-1 if highest else 0]
    value = row[column]
    return None if pd.isna(value) else float(value)


def analyze_feature(
    frame: pd.DataFrame,
    dose_feature: str,
    requested_bins: int,
) -> tuple[pd.DataFrame, dict[str, object], list[str]]:
    dose = pd.to_numeric(frame[dose_feature], errors="coerce")
    missing_count = int(dose.isna().sum())
    usable = frame[dose.notna()].copy()
    usable_sample_count = int(len(usable))
    missing_rate = float(missing_count / max(1, len(frame)))

    if usable.empty:
        diagnostics = {
            "usable_sample_count": 0,
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "effective_bin_count": 0,
            "interpretation_flags": {
                "non_monotonic_possible": False,
                "risk_tradeoff_possible": False,
                "weak_univariate_signal": True,
                "direction_conflict": False,
                "low_bin_support": True,
            },
        }
        return pd.DataFrame(), diagnostics, [f"{dose_feature}: no usable samples."]

    bin_ids, effective_bins, bin_warnings = make_quantile_bins(usable[dose_feature], requested_bins)
    bin_table = summarize_bins(usable, dose_feature, bin_ids)
    if bin_table.empty:
        effective_bins = 0

    corr = {
        "spearman_corr_dose_t90": valid_corr(usable[dose_feature], usable["t90"], "spearman"),
        "spearman_corr_dose_y_ok": valid_corr(usable[dose_feature], usable["y_ok"], "spearman"),
        "spearman_corr_dose_y_low": valid_corr(usable[dose_feature], usable["y_low"], "spearman"),
        "spearman_corr_dose_y_high": valid_corr(usable[dose_feature], usable["y_high"], "spearman"),
        "spearman_corr_dose_y_out_spec": valid_corr(usable[dose_feature], usable["y_out_spec"], "spearman"),
        "pearson_corr_dose_t90": valid_corr(usable[dose_feature], usable["t90"], "pearson"),
    }

    if bin_table.empty:
        best_ok_bin = worst_ok_bin = None
        best_ok_rate = worst_ok_rate = None
        best_bin_is_interior = False
        low_bin_support = True
    else:
        best_row = bin_table.sort_values(["ok_rate", "sample_count"], ascending=[False, False]).iloc[0]
        worst_row = bin_table.sort_values(["ok_rate", "sample_count"], ascending=[True, False]).iloc[0]
        best_ok_bin = int(best_row["bin_id"])
        worst_ok_bin = int(worst_row["bin_id"])
        best_ok_rate = float(best_row["ok_rate"])
        worst_ok_rate = float(worst_row["ok_rate"])
        best_bin_is_interior = bool(best_row["is_interior_bin"])
        low_bin_support = bool((bin_table["sample_count"] < 30).any())

    low_rate_lowest = get_endpoint(bin_table, "low_rate", highest=False)
    low_rate_highest = get_endpoint(bin_table, "low_rate", highest=True)
    high_rate_lowest = get_endpoint(bin_table, "high_rate", highest=False)
    high_rate_highest = get_endpoint(bin_table, "high_rate", highest=True)
    low_delta = None if low_rate_lowest is None or low_rate_highest is None else low_rate_highest - low_rate_lowest
    high_delta = None if high_rate_lowest is None or high_rate_highest is None else high_rate_highest - high_rate_lowest

    spearman_values = [
        corr["spearman_corr_dose_t90"],
        corr["spearman_corr_dose_y_ok"],
        corr["spearman_corr_dose_y_low"],
        corr["spearman_corr_dose_y_high"],
        corr["spearman_corr_dose_y_out_spec"],
    ]
    weak_univariate = all(value is None or abs(value) < 0.05 for value in spearman_values)
    risk_tradeoff = bool(
        low_delta is not None
        and high_delta is not None
        and ((high_delta < 0 < low_delta) or (low_delta < 0 < high_delta))
    )
    spearman_low = corr["spearman_corr_dose_y_low"]
    spearman_high = corr["spearman_corr_dose_y_high"]
    sign_conflict = bool(
        spearman_low is not None
        and spearman_high is not None
        and spearman_low * spearman_high < 0
    )
    edge_conflict = bool(
        low_delta is not None
        and high_delta is not None
        and low_delta * high_delta < 0
    )

    edge_low_diff = None if low_delta is None else abs(low_delta)
    edge_high_diff = None if high_delta is None else abs(high_delta)
    clear_relationship = bool(
        (spearman_low is not None and abs(spearman_low) >= 0.08)
        or (spearman_high is not None and abs(spearman_high) >= 0.08)
        or (edge_low_diff is not None and edge_low_diff >= 0.05)
        or (edge_high_diff is not None and edge_high_diff >= 0.05)
    )
    weak_relationship = bool(
        weak_univariate
        and (edge_low_diff is None or edge_low_diff < 0.03)
        and (edge_high_diff is None or edge_high_diff < 0.03)
    )

    diagnostics = {
        "usable_sample_count": usable_sample_count,
        "missing_count": missing_count,
        "missing_rate": missing_rate,
        "effective_bin_count": int(effective_bins),
        **corr,
        "best_ok_rate_bin": best_ok_bin,
        "best_ok_rate": best_ok_rate,
        "worst_ok_rate_bin": worst_ok_bin,
        "worst_ok_rate": worst_ok_rate,
        "best_bin_is_interior": best_bin_is_interior,
        "t90_mean_at_lowest_bin": get_endpoint(bin_table, "t90_mean", highest=False),
        "t90_mean_at_highest_bin": get_endpoint(bin_table, "t90_mean", highest=True),
        "ok_rate_at_lowest_bin": get_endpoint(bin_table, "ok_rate", highest=False),
        "ok_rate_at_highest_bin": get_endpoint(bin_table, "ok_rate", highest=True),
        "low_rate_at_lowest_bin": low_rate_lowest,
        "low_rate_at_highest_bin": low_rate_highest,
        "high_rate_at_lowest_bin": high_rate_lowest,
        "high_rate_at_highest_bin": high_rate_highest,
        "edge_low_rate_diff_highest_minus_lowest": low_delta,
        "edge_high_rate_diff_highest_minus_lowest": high_delta,
        "clear_relationship": clear_relationship,
        "weak_relationship": weak_relationship,
        "interpretation_flags": {
            "non_monotonic_possible": best_bin_is_interior,
            "risk_tradeoff_possible": risk_tradeoff,
            "weak_univariate_signal": weak_univariate,
            "direction_conflict": bool(sign_conflict or edge_conflict),
            "low_bin_support": low_bin_support,
        },
    }
    return bin_table, diagnostics, [f"{dose_feature}: {warning}" for warning in bin_warnings]


def count_targets(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for column in ["y_ok", "y_low", "y_high", "y_out_spec"]:
        counts = frame[column].value_counts(dropna=False).sort_index()
        result[column] = {str(key): int(value) for key, value in counts.items()}
    return result


def overall_rates(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "ok_rate": float(frame["y_ok"].mean()),
        "low_rate": float(frame["y_low"].mean()),
        "high_rate": float(frame["y_high"].mean()),
        "out_spec_rate": float(frame["y_out_spec"].mean()),
        "t90_mean": float(frame["t90"].mean()),
        "t90_median": float(frame["t90"].median()),
    }


def decide_next_step(
    primary_diagnostics: dict[str, object],
    all_diagnostics: dict[str, dict[str, object]],
) -> str:
    enough = int(primary_diagnostics.get("usable_sample_count") or 0) >= 500
    clear = bool(primary_diagnostics.get("clear_relationship"))
    weak = bool(primary_diagnostics.get("weak_relationship"))
    flags = primary_diagnostics.get("interpretation_flags", {})
    non_monotonic = bool(flags.get("non_monotonic_possible")) if isinstance(flags, dict) else False
    all_weak = all(bool(item.get("weak_relationship")) for item in all_diagnostics.values())

    if enough and clear:
        return "proceed_to_control_model_with_train_only_feature_selection"
    if weak and non_monotonic:
        return "inspect_dose_response_manually_before_modeling"
    if not enough or all_weak:
        return "insufficient_calcium_signal_do_not_model_yet"
    return "inspect_dose_response_manually_before_modeling"


def build_report(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    primary_dose_feature: str,
    dose_features: list[str],
    diagnostics: dict[str, dict[str, object]],
    warnings: list[str],
) -> dict[str, object]:
    return {
        "input_path": str(args.input),
        "feature_report_path": str(args.feature_report),
        "bins_output_path": str(args.bins_output),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": int(len(frame)),
        "n_bins_requested": int(args.n_bins),
        "dose_features_analyzed": dose_features,
        "primary_dose_feature": primary_dose_feature,
        "target_counts": count_targets(frame),
        "overall_rates": overall_rates(frame),
        "per_feature_diagnostics": diagnostics,
        "recommended_next_step": decide_next_step(diagnostics[primary_dose_feature], diagnostics),
        "warnings": warnings,
        "assumptions": [
            "Dose-response interpretation uses calcium_core_features only.",
            "process_context_features are intentionally excluded from this univariate analysis stage.",
            "Rows are dropped only for the currently analyzed dose feature when that dose value is missing.",
            "Dose values are not imputed before binning or correlation calculation.",
            "Quantile bins are duplicate-safe and may use fewer effective bins than requested.",
            "This analysis is observational and does not prove a causal calcium control effect.",
        ],
    }


def print_summary(report: dict[str, object]) -> None:
    primary = str(report["primary_dose_feature"])
    diagnostics = report["per_feature_diagnostics"][primary]
    flags = diagnostics.get("interpretation_flags", {})
    rates = report["overall_rates"]
    print("T90 calcium dose-response analysis complete.")
    print(f"  primary dose feature: {primary}")
    print(f"  usable sample count: {diagnostics.get('usable_sample_count')}")
    print(f"  overall ok_rate: {rates.get('ok_rate')}")
    print(f"  overall low_rate: {rates.get('low_rate')}")
    print(f"  overall high_rate: {rates.get('high_rate')}")
    print(f"  overall out_spec_rate: {rates.get('out_spec_rate')}")
    print(f"  best ok-rate bin: {diagnostics.get('best_ok_rate_bin')} ({diagnostics.get('best_ok_rate')})")
    print("  Spearman correlations:")
    for key in [
        "spearman_corr_dose_t90",
        "spearman_corr_dose_y_ok",
        "spearman_corr_dose_y_low",
        "spearman_corr_dose_y_high",
        "spearman_corr_dose_y_out_spec",
    ]:
        print(f"    {key}: {diagnostics.get(key)}")
    print(f"  interpretation flags: {flags}")
    print(f"  recommended next step: {report['recommended_next_step']}")
    print(f"  bins output: {report['bins_output_path']}")
    print(f"  report output: {report.get('report_path', '')}")


def main() -> None:
    args = parse_args()
    if args.n_bins < 2:
        raise ValueError("--n-bins must be at least 2")

    frame, feature_report = load_inputs(args.input, args.feature_report)
    primary_dose_feature, dose_features, warnings = select_dose_features(frame, feature_report)

    all_bins: list[pd.DataFrame] = []
    diagnostics: dict[str, dict[str, object]] = {}
    for dose_feature in dose_features:
        bin_table, feature_diagnostics, feature_warnings = analyze_feature(frame, dose_feature, args.n_bins)
        if not bin_table.empty:
            all_bins.append(bin_table)
        diagnostics[dose_feature] = feature_diagnostics
        warnings.extend(feature_warnings)

    bins_output = pd.concat(all_bins, ignore_index=True) if all_bins else pd.DataFrame()
    args.bins_output.parent.mkdir(parents=True, exist_ok=True)
    bins_output.to_csv(args.bins_output, index=False, encoding="utf-8-sig")

    report = build_report(
        args=args,
        frame=frame,
        primary_dose_feature=primary_dose_feature,
        dose_features=dose_features,
        diagnostics=diagnostics,
        warnings=warnings,
    )
    report["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(as_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)


if __name__ == "__main__":
    main()
