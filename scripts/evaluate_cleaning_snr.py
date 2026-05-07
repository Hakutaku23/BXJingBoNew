from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def robust_mad(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    median = clean.median()
    return float((clean - median).abs().median())


def snr_metrics(frame: pd.DataFrame, column: str, window: int) -> dict[str, float | int]:
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.dropna()
    if len(valid) < max(20, window):
        return {
            "non_null_count": int(valid.size),
            "non_null_rate": float(values.notna().mean()),
            "snr_db": None,
            "residual_mad": None,
            "diff_mad": None,
            "mean": float(valid.mean()) if len(valid) else None,
            "std": float(valid.std()) if len(valid) else None,
        }

    smooth = values.rolling(window=window, center=True, min_periods=max(5, window // 2)).median()
    residual = values - smooth

    signal_var = float(smooth.dropna().var())
    noise_var = float(residual.dropna().var())
    snr_db = None
    if signal_var > 0 and noise_var > 0:
        snr_db = float(10 * np.log10(signal_var / noise_var))

    return {
        "non_null_count": int(valid.size),
        "non_null_rate": float(values.notna().mean()),
        "snr_db": snr_db,
        "residual_mad": robust_mad(residual),
        "diff_mad": robust_mad(values.diff()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate proxy SNR before and after DCS cleaning.")
    parser.add_argument("--before", default=Path("data/data_new.parquet"), type=Path)
    parser.add_argument("--after", default=Path("data/data_clean.parquet"), type=Path)
    parser.add_argument("--output", default=Path("data/data_clean_snr_report.json"), type=Path)
    parser.add_argument("--csv-output", default=Path("data/data_clean_snr_report.csv"), type=Path)
    parser.add_argument("--window", default=31, type=int)
    args = parser.parse_args()

    before = pd.read_parquet(args.before).sort_values("time").reset_index(drop=True)
    after = pd.read_parquet(args.after).sort_values("time").reset_index(drop=True)

    before_columns = [column for column in before.columns if column != "time"]
    after_columns = set(after.columns)
    feature_columns = [column for column in before_columns if column in after_columns]

    rows: list[dict[str, object]] = []
    for column in feature_columns:
        before_metrics = snr_metrics(before, column, args.window)
        after_metrics = snr_metrics(after, column, args.window)

        before_snr = before_metrics["snr_db"]
        after_snr = after_metrics["snr_db"]
        snr_delta = None
        if before_snr is not None and after_snr is not None:
            snr_delta = float(after_snr - before_snr)

        before_residual = before_metrics["residual_mad"]
        after_residual = after_metrics["residual_mad"]
        residual_reduction = None
        if before_residual and after_residual is not None:
            residual_reduction = float((before_residual - after_residual) / before_residual)

        before_diff = before_metrics["diff_mad"]
        after_diff = after_metrics["diff_mad"]
        diff_reduction = None
        if before_diff and after_diff is not None:
            diff_reduction = float((before_diff - after_diff) / before_diff)

        before_mean = before_metrics["mean"]
        after_mean = after_metrics["mean"]
        before_std = before_metrics["std"]
        mean_shift_sigma = None
        if before_mean is not None and after_mean is not None and before_std and before_std > 0:
            mean_shift_sigma = float(abs(after_mean - before_mean) / before_std)

        rows.append(
            {
                "variable": column,
                "before_non_null_rate": before_metrics["non_null_rate"],
                "after_non_null_rate": after_metrics["non_null_rate"],
                "before_snr_db": before_snr,
                "after_snr_db": after_snr,
                "snr_delta_db": snr_delta,
                "before_residual_mad": before_residual,
                "after_residual_mad": after_residual,
                "residual_mad_reduction": residual_reduction,
                "before_diff_mad": before_diff,
                "after_diff_mad": after_diff,
                "diff_mad_reduction": diff_reduction,
                "mean_shift_sigma": mean_shift_sigma,
                "before_non_null_count": before_metrics["non_null_count"],
                "after_non_null_count": after_metrics["non_null_count"],
            }
        )

    result = pd.DataFrame(rows)
    finite_delta = result["snr_delta_db"].dropna()
    finite_residual_reduction = result["residual_mad_reduction"].dropna()
    large_mean_shift = result["mean_shift_sigma"].fillna(0) > 0.05

    summary = {
        "before": str(args.before),
        "after": str(args.after),
        "method": {
            "signal_proxy": f"centered rolling median, window={args.window}",
            "noise_proxy": "residual between raw/cleaned value and rolling median",
            "snr_db": "10 * log10(var(signal_proxy) / var(noise_proxy))",
            "notes": [
                "This is a proxy SNR for DCS process data, not an instrument-certified SNR.",
                "A small SNR change is expected when cleaning is conservative.",
                "Coverage, residual MAD, first-difference MAD, and mean shift are checked together.",
            ],
        },
        "feature_count": int(len(result)),
        "snr_improved_count": int((result["snr_delta_db"] > 0).sum()),
        "snr_unchanged_or_worse_count": int((result["snr_delta_db"] <= 0).sum()),
        "median_snr_delta_db": float(finite_delta.median()) if len(finite_delta) else None,
        "mean_snr_delta_db": float(finite_delta.mean()) if len(finite_delta) else None,
        "median_residual_mad_reduction": float(finite_residual_reduction.median())
        if len(finite_residual_reduction)
        else None,
        "large_mean_shift_count": int(large_mean_shift.sum()),
        "judgement": None,
        "variables": rows,
    }

    if summary["large_mean_shift_count"] == 0 and summary["median_residual_mad_reduction"] is not None:
        if summary["median_residual_mad_reduction"] >= -0.01 and summary["snr_improved_count"] >= summary["feature_count"] // 2:
            summary["judgement"] = "cleaning_is_reasonable_but_conservative"
        elif summary["median_residual_mad_reduction"] >= -0.01:
            summary["judgement"] = "cleaning_is_safe_but_snr_gain_is_limited"
        else:
            summary["judgement"] = "cleaning_may_not_improve_noise_proxy"
    else:
        summary["judgement"] = "review_required_due_to_distribution_shift_or_insufficient_metrics"

    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    result.to_csv(args.csv_output, index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
