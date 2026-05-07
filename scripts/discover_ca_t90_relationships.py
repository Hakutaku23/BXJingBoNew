from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
except Exception:  # pragma: no cover - dependency availability is environment-specific
    GradientBoostingClassifier = None
    HistGradientBoostingClassifier = None
    LogisticRegression = None
    SimpleImputer = None
    average_precision_score = None
    brier_score_loss = None
    roc_auc_score = None
    make_pipeline = None


T90_LOW = 8.20
T90_HIGH = 8.70
RANDOM_SEED = 20260507

PRIMARY_DOSE_PRIORITY = [
    "ca_per_rubber_flow_win_60_mean",
    "ca_per_rubber_flow_lag_165",
    "ca_win_60_mean",
    "ca_lag_165",
]
CONTEXT_CANDIDATES = [
    "rubber_flow_2_win_60_mean",
    "bromine_feed_win_60_mean",
    "tank_rubber_conc_win_60_mean",
    "esbo_feed_win_60_mean",
    "neutral_alkali_feed_win_60_mean",
    "r513_temp_win_60_mean",
    "r514_temp_win_60_mean",
    "r510a_temp_win_60_mean",
    "r511a_temp_win_60_mean",
    "r512a_temp_win_60_mean",
]
IR_PRIORITY = [
    "output_ir_corrected_win_15_slope",
    "output_ir_corrected_win_30_mean",
    "output_ir_corrected_win_15_mean",
    "output_ir_corrected",
    "output_ir_corrected_lag_0",
]
IR_EVALUATION_FEATURES = [
    "output_ir_corrected",
    "output_ir_corrected_lag_0",
    "output_ir_corrected_win_5_mean",
    "output_ir_corrected_win_15_mean",
    "output_ir_corrected_win_30_mean",
    "output_ir_corrected_win_15_std",
    "output_ir_corrected_win_30_std",
    "output_ir_corrected_win_15_slope",
]
TARGETS = ["y_high", "y_low", "y_out_spec", "y_ok"]
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
    parser = argparse.ArgumentParser(description="Discover calcium, context, IR, and T90-risk relationships.")
    parser.add_argument("--features", type=Path, default=Path("data/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--feature-report", type=Path, default=Path("data/t90_ca_feature_report.json"))
    parser.add_argument("--dose-response-report", type=Path, default=Path("data/t90_ca_dose_response_report.json"))
    parser.add_argument("--data-with-ir", type=Path, default=Path("data/data_clean_with_ir.parquet"))
    parser.add_argument("--ir-report", type=Path, default=Path("data/output_ir_proxy_evaluation.json"))
    parser.add_argument("--regime-output", type=Path, default=Path("data/ca_regime_dose_response.csv"))
    parser.add_argument("--interaction-output", type=Path, default=Path("data/ca_context_interaction_screen.csv"))
    parser.add_argument("--ir-strat-output", type=Path, default=Path("data/ca_ir_stratified_dose_response.csv"))
    parser.add_argument("--mediation-output", type=Path, default=Path("data/ir_mediation_diagnostic.csv"))
    parser.add_argument("--band-map-output", type=Path, default=Path("data/ca_regime_optimal_band_map.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/ca_t90_relationship_discovery_report.json"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--n-bins", type=int, default=5)
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


def load_json(path: Path, required: bool = True) -> dict[str, object]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required JSON does not exist: {path}")
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_targets(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["t90"] = pd.to_numeric(frame["t90"], errors="coerce")
    if "y_ok" not in frame.columns:
        frame["y_ok"] = ((frame["t90"] >= T90_LOW) & (frame["t90"] <= T90_HIGH)).astype(int)
    if "y_low" not in frame.columns:
        frame["y_low"] = (frame["t90"] < T90_LOW).astype(int)
    if "y_high" not in frame.columns:
        frame["y_high"] = (frame["t90"] > T90_HIGH).astype(int)
    if "y_out_spec" not in frame.columns:
        frame["y_out_spec"] = ((frame["t90"] < T90_LOW) | (frame["t90"] > T90_HIGH)).astype(int)
    return frame


def is_leakage_column(column: str) -> bool:
    lowered = column.lower()
    return (
        column in LEAKAGE_COLUMNS
        or lowered.startswith("pred_")
        or lowered.startswith("p_")
        or lowered.endswith("_pred")
        or lowered.startswith("target_")
    )


def choose_primary_dose(frame: pd.DataFrame, dose_report: dict[str, object]) -> str:
    primary = dose_report.get("primary_dose_feature")
    if isinstance(primary, str) and primary in frame.columns:
        return primary
    for feature in PRIMARY_DOSE_PRIORITY:
        if feature in frame.columns:
            return feature
    raise ValueError("No primary calcium dose feature is available.")


def choose_ir_feature(frame: pd.DataFrame, ir_report: dict[str, object]) -> str | None:
    best = ir_report.get("best_ir_feature")
    if isinstance(best, str) and best in frame.columns:
        return best
    for feature in IR_PRIORITY:
        if feature in frame.columns:
            return feature
    return None


def load_supervised(args: argparse.Namespace, dose_report: dict[str, object], ir_report: dict[str, object], warnings: list[str]) -> tuple[pd.DataFrame, str, list[str], str | None]:
    if not args.features.exists():
        raise FileNotFoundError(f"Feature parquet does not exist: {args.features}")
    frame = pd.read_parquet(args.features)
    required = ["time", "t90"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {missing}")
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    if frame["time"].isna().any():
        raise ValueError("Feature dataset contains invalid time values.")
    frame = ensure_targets(frame)
    frame = frame[frame["t90"].notna()].sort_values("time").reset_index(drop=True)

    if args.data_with_ir.exists():
        needed = ["time"] + IR_EVALUATION_FEATURES
        available = pd.read_parquet(args.data_with_ir, columns=None).columns.tolist()
        ir_cols = [column for column in needed if column in available]
        if len(ir_cols) > 1:
            ir_frame = pd.read_parquet(args.data_with_ir, columns=ir_cols)
            ir_frame["time"] = pd.to_datetime(ir_frame["time"], errors="coerce")
            ir_frame = ir_frame.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last")
            frame = frame.merge(ir_frame, on="time", how="left")
        else:
            warnings.append("data-with-ir exists but no recognized output_ir_corrected feature columns were found.")
    else:
        warnings.append(f"data-with-ir is missing: {args.data_with_ir}; IR analyses will be skipped.")

    dose_feature = choose_primary_dose(frame, dose_report)
    context_features = [
        feature for feature in CONTEXT_CANDIDATES if feature in frame.columns and not is_leakage_column(feature)
    ]
    ir_feature = choose_ir_feature(frame, ir_report)
    return frame, dose_feature, context_features, ir_feature


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    work = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(work) < 3 or work["x"].nunique() < 2 or work["y"].nunique() < 2:
        return math.nan
    return float(work["x"].corr(work["y"], method=method))


def make_quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    clean = numeric.dropna()
    if len(clean) < 2 or clean.nunique() < 2:
        return result
    for bins in range(min(n_bins, clean.nunique()), 1, -1):
        try:
            labels = pd.qcut(clean, q=bins, labels=False, duplicates="drop")
            if labels.nunique(dropna=True) >= 2:
                result.loc[labels.index] = labels.astype("Int64")
                return result
        except ValueError:
            continue
    ranks = clean.rank(method="first")
    labels = pd.qcut(ranks, q=min(n_bins, len(clean)), labels=False, duplicates="drop")
    result.loc[labels.index] = labels.astype("Int64")
    return result


def make_tertiles(values: pd.Series) -> pd.Series:
    ids = make_quantile_bins(values, 3)
    mapping = {0: "low", 1: "mid", 2: "high"}
    return ids.map(mapping).astype("object")


def summarize_dose_bins(
    frame: pd.DataFrame,
    dose_feature: str,
    group_column: str,
    group_value_column: str,
    n_bins: int,
    source: str,
) -> pd.DataFrame:
    rows = []
    for group_value, subset in frame.dropna(subset=[group_column]).groupby(group_column, sort=True):
        subset = subset[pd.to_numeric(subset[dose_feature], errors="coerce").notna()].copy()
        regime_count = int(len(subset))
        if subset.empty:
            continue
        subset["dose_bin"] = make_quantile_bins(subset[dose_feature], n_bins)
        for dose_bin, bin_df in subset.dropna(subset=["dose_bin"]).groupby("dose_bin", sort=True):
            dose = pd.to_numeric(bin_df[dose_feature], errors="coerce")
            enough = regime_count >= 80 and len(bin_df) >= 20
            rows.append(
                {
                    group_value_column: group_value,
                    "dose_bin": int(dose_bin),
                    "sample_count": int(len(bin_df)),
                    "regime_sample_count": regime_count,
                    "dose_min": float(dose.min()),
                    "dose_max": float(dose.max()),
                    "dose_mean": float(dose.mean()),
                    "t90_mean": float(bin_df["t90"].mean()),
                    "t90_median": float(bin_df["t90"].median()),
                    "ok_count": int(bin_df["y_ok"].sum()),
                    "ok_rate": float(bin_df["y_ok"].mean()),
                    "low_count": int(bin_df["y_low"].sum()),
                    "low_rate": float(bin_df["y_low"].mean()),
                    "high_count": int(bin_df["y_high"].sum()),
                    "high_rate": float(bin_df["y_high"].mean()),
                    "out_spec_count": int(bin_df["y_out_spec"].sum()),
                    "out_spec_rate": float(bin_df["y_out_spec"].mean()),
                    "support_level": "enough_support" if enough else "weak_support",
                    "source": source,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["best_dose_bin_in_regime"] = False
        for group_value, subset in result.groupby(group_value_column, sort=False):
            eligible = subset[subset["support_level"] == "enough_support"]
            if eligible.empty:
                eligible = subset
            if not eligible.empty:
                best_index = eligible.sort_values(["ok_rate", "sample_count"], ascending=[False, False]).index[0]
                result.loc[best_index, "best_dose_bin_in_regime"] = True
    return result


def experiment_a_regime_dose_response(data: pd.DataFrame, dose_feature: str, context_features: list[str], n_bins: int) -> pd.DataFrame:
    outputs = []
    for context in context_features:
        work = data[[context, dose_feature, "t90", "y_ok", "y_low", "y_high", "y_out_spec"]].copy()
        work["regime_bin"] = make_tertiles(work[context])
        table = summarize_dose_bins(work, dose_feature, "regime_bin", "regime_bin", n_bins, source="context_tertile")
        if not table.empty:
            table.insert(0, "regime_feature", context)
            outputs.append(table)
    if outputs:
        return pd.concat(outputs, ignore_index=True)
    columns = [
        "regime_feature", "regime_bin", "dose_bin", "sample_count", "regime_sample_count", "dose_min",
        "dose_max", "dose_mean", "t90_mean", "t90_median", "ok_count", "ok_rate", "low_count",
        "low_rate", "high_count", "high_rate", "out_spec_count", "out_spec_rate", "support_level",
        "source", "best_dose_bin_in_regime",
    ]
    return pd.DataFrame(columns=columns)


def train_screen_model(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> dict[str, object]:
    if any(obj is None for obj in [SimpleImputer, LogisticRegression, average_precision_score, roc_auc_score, brier_score_loss, make_pipeline]):
        return {"warning": "sklearn is unavailable", "ap": math.nan, "auc": math.nan, "brier": math.nan}
    train = train.dropna(subset=[target])
    test = test.dropna(subset=[target])
    y_train = train[target].astype(int)
    y_test = test[target].astype(int)
    if len(train) < 50 or len(test) < 30 or y_train.nunique() < 2 or y_test.nunique() < 2 or y_train.sum() < 5 or y_test.sum() < 3:
        return {"warning": "insufficient class support", "ap": math.nan, "auc": math.nan, "brier": math.nan}
    x_train = train[features]
    x_test = test[features]
    models = [
        make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ]
    if HistGradientBoostingClassifier is not None:
        models.append(make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(random_state=RANDOM_SEED)))
    if GradientBoostingClassifier is not None:
        models.append(make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(random_state=RANDOM_SEED)))
    last_error = None
    for model in models:
        try:
            model.fit(x_train, y_train)
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(x_test)[:, 1]
            else:
                score = model.decision_function(x_test)
                prob = 1.0 / (1.0 + np.exp(-score))
            return {
                "ap": float(average_precision_score(y_test, prob)),
                "auc": float(roc_auc_score(y_test, prob)),
                "brier": float(brier_score_loss(y_test, prob)),
                "warning": None,
            }
        except Exception as exc:  # pragma: no cover - depends on solver/backend
            last_error = str(exc)
            continue
    return {"warning": last_error or "all models failed", "ap": math.nan, "auc": math.nan, "brier": math.nan}


def experiment_b_interaction_screen(data: pd.DataFrame, dose_feature: str, context_features: list[str]) -> pd.DataFrame:
    split = int(len(data) * 0.8)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()
    rows = []
    for context in context_features:
        for target in TARGETS:
            cols = [dose_feature, context, target]
            tr = train[cols].copy()
            te = test[cols].copy()
            dose_median = pd.to_numeric(tr[dose_feature], errors="coerce").median()
            ctx_median = pd.to_numeric(tr[context], errors="coerce").median()
            for frame in [tr, te]:
                frame["dose_context_interaction"] = (
                    pd.to_numeric(frame[dose_feature], errors="coerce") - dose_median
                ) * (pd.to_numeric(frame[context], errors="coerce") - ctx_median)
            base = train_screen_model(tr, te, [dose_feature, context], target)
            inter = train_screen_model(tr, te, [dose_feature, context, "dose_context_interaction"], target)
            train_pos = float(tr[target].mean()) if len(tr) else math.nan
            test_pos = float(te[target].mean()) if len(te) else math.nan
            direction = safe_corr(tr["dose_context_interaction"], tr[target], method="spearman")
            delta_ap = inter["ap"] - base["ap"] if np.isfinite(inter["ap"]) and np.isfinite(base["ap"]) else math.nan
            delta_auc = inter["auc"] - base["auc"] if np.isfinite(inter["auc"]) and np.isfinite(base["auc"]) else math.nan
            delta_brier = inter["brier"] - base["brier"] if np.isfinite(inter["brier"]) and np.isfinite(base["brier"]) else math.nan
            enough = tr[target].sum() >= 5 and te[target].sum() >= 3 and tr[target].nunique() == 2 and te[target].nunique() == 2
            no_instability = enough and len(te) >= 30
            passed = no_instability and ((np.isfinite(delta_auc) and delta_auc >= 0.03) or (np.isfinite(delta_ap) and delta_ap >= 0.03))
            rows.append(
                {
                    "target": target,
                    "context_feature": context,
                    "train_sample_count": int(len(tr)),
                    "test_sample_count": int(len(te)),
                    "positive_rate_train": train_pos,
                    "positive_rate_test": test_pos,
                    "base_ap": base["ap"],
                    "interaction_ap": inter["ap"],
                    "delta_ap": delta_ap,
                    "base_auc": base["auc"],
                    "interaction_auc": inter["auc"],
                    "delta_auc": delta_auc,
                    "base_brier": base["brier"],
                    "interaction_brier": inter["brier"],
                    "delta_brier": delta_brier,
                    "interaction_direction_proxy": direction,
                    "passed_interaction_screen": bool(passed),
                    "warning": base.get("warning") or inter.get("warning"),
                }
            )
    return pd.DataFrame(rows)


def experiment_c_ir_stratified(data: pd.DataFrame, dose_feature: str, ir_feature: str | None, n_bins: int) -> pd.DataFrame:
    columns = [
        "ir_feature", "ir_regime_type", "ir_regime", "dose_bin", "sample_count", "regime_sample_count",
        "dose_min", "dose_max", "dose_mean", "t90_mean", "t90_median", "ok_count", "ok_rate",
        "low_count", "low_rate", "high_count", "high_rate", "out_spec_count", "out_spec_rate",
        "support_level", "source", "best_dose_bin_in_regime",
    ]
    if ir_feature is None or ir_feature not in data.columns:
        return pd.DataFrame(columns=columns)
    outputs = []
    base = data[[ir_feature, dose_feature, "t90", "y_ok", "y_low", "y_high", "y_out_spec"]].copy()
    base["ir_regime"] = make_tertiles(base[ir_feature])
    tertile = summarize_dose_bins(base, dose_feature, "ir_regime", "ir_regime", n_bins, source="ir_tertile")
    if not tertile.empty:
        tertile.insert(0, "ir_feature", ir_feature)
        tertile.insert(1, "ir_regime_type", "tertile")
        outputs.append(tertile)
    if "slope" in ir_feature.lower():
        values = pd.to_numeric(base[ir_feature], errors="coerce")
        threshold = max(float(values.abs().quantile(0.25)) if values.notna().any() else 0.0, 1e-9)
        sign_frame = base.copy()
        sign_frame["ir_regime"] = np.select(
            [values < -threshold, values > threshold],
            ["falling", "rising"],
            default="stable",
        )
        sign_frame.loc[values.isna(), "ir_regime"] = pd.NA
        sign_table = summarize_dose_bins(sign_frame, dose_feature, "ir_regime", "ir_regime", n_bins, source="ir_sign")
        if not sign_table.empty:
            sign_table.insert(0, "ir_feature", ir_feature)
            sign_table.insert(1, "ir_regime_type", "sign")
            outputs.append(sign_table)
    if outputs:
        return pd.concat(outputs, ignore_index=True)
    return pd.DataFrame(columns=columns)


def bin_rate_spread(frame: pd.DataFrame, feature: str, target: str, n_bins: int) -> float:
    work = frame[[feature, target]].dropna().copy()
    if work.empty:
        return math.nan
    work["bin"] = make_quantile_bins(work[feature], n_bins)
    rates = work.dropna(subset=["bin"]).groupby("bin")[target].mean()
    if len(rates) < 2:
        return math.nan
    return float(rates.max() - rates.min())


def experiment_d_mediation(data: pd.DataFrame, dose_feature: str, context_features: list[str], ir_feature: str | None, n_bins: int) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    summary = {
        "calcium_to_ir_signal": False,
        "ir_to_t90_risk_signal": False,
        "ir_incremental_signal": False,
        "calcium_ir_interaction_signal": False,
        "descriptive_mediation_possible": False,
        "not_causal_proof": True,
    }
    if ir_feature is None or ir_feature not in data.columns:
        rows.append({"diagnostic_type": "availability", "target": "", "feature_or_pair": "", "metric": "ir_available", "value": 0, "sample_count": 0, "interpretation": "IR feature unavailable"})
        return pd.DataFrame(rows), summary
    usable = data[[dose_feature, ir_feature, "t90", "y_ok", "y_low", "y_high", "y_out_spec"]].dropna(subset=[ir_feature])
    ca_ir_s = safe_corr(usable[dose_feature], usable[ir_feature], "spearman")
    ca_ir_p = safe_corr(usable[dose_feature], usable[ir_feature], "pearson")
    summary["calcium_to_ir_signal"] = bool(np.isfinite(ca_ir_s) and abs(ca_ir_s) >= 0.08)
    rows.extend(
        [
            {"diagnostic_type": "calcium_to_ir", "target": "output_ir", "feature_or_pair": f"{dose_feature}->{ir_feature}", "metric": "spearman", "value": ca_ir_s, "sample_count": int(len(usable.dropna(subset=[dose_feature]))), "interpretation": "calcium_to_ir_signal" if summary["calcium_to_ir_signal"] else "weak_or_unclear"},
            {"diagnostic_type": "calcium_to_ir", "target": "output_ir", "feature_or_pair": f"{dose_feature}->{ir_feature}", "metric": "pearson", "value": ca_ir_p, "sample_count": int(len(usable.dropna(subset=[dose_feature]))), "interpretation": "linear_relation_check"},
        ]
    )
    work = usable.dropna(subset=[dose_feature]).copy()
    work["dose_bin"] = make_quantile_bins(work[dose_feature], n_bins)
    for dose_bin, group in work.dropna(subset=["dose_bin"]).groupby("dose_bin", sort=True):
        rows.append({"diagnostic_type": "calcium_bin_ir", "target": "output_ir", "feature_or_pair": dose_feature, "metric": f"ir_mean_bin_{int(dose_bin)}", "value": float(group[ir_feature].mean()), "sample_count": int(len(group)), "interpretation": "IR mean by calcium dose bin"})
    for target in ["t90", "y_ok", "y_low", "y_high", "y_out_spec"]:
        corr = safe_corr(usable[ir_feature], usable[target], "spearman")
        rows.append({"diagnostic_type": "ir_to_t90_risk", "target": target, "feature_or_pair": ir_feature, "metric": "spearman", "value": corr, "sample_count": int(len(usable.dropna(subset=[target]))), "interpretation": "IR-to-risk correlation"})
    high_spread = bin_rate_spread(usable, ir_feature, "y_high", n_bins)
    out_spread = bin_rate_spread(usable, ir_feature, "y_out_spec", n_bins)
    summary["ir_to_t90_risk_signal"] = bool(
        (np.isfinite(high_spread) and high_spread >= 0.05)
        or (np.isfinite(out_spread) and out_spread >= 0.05)
        or abs(safe_corr(usable[ir_feature], usable["y_high"], "spearman")) >= 0.08
        or abs(safe_corr(usable[ir_feature], usable["y_out_spec"], "spearman")) >= 0.08
    )
    rows.extend(
        [
            {"diagnostic_type": "ir_bin_risk_spread", "target": "y_high", "feature_or_pair": ir_feature, "metric": "high_rate_spread", "value": high_spread, "sample_count": int(len(usable)), "interpretation": "IR bin risk spread"},
            {"diagnostic_type": "ir_bin_risk_spread", "target": "y_out_spec", "feature_or_pair": ir_feature, "metric": "out_spec_rate_spread", "value": out_spread, "sample_count": int(len(usable)), "interpretation": "IR bin risk spread"},
        ]
    )
    split = int(len(data) * 0.8)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()
    base_features = [dose_feature] + context_features[:5]
    for target in ["y_high", "y_out_spec"]:
        base = train_screen_model(train, test, base_features, target)
        plus_ir = train_screen_model(train, test, base_features + [ir_feature], target)
        delta_ap = plus_ir["ap"] - base["ap"] if np.isfinite(plus_ir["ap"]) and np.isfinite(base["ap"]) else math.nan
        delta_auc = plus_ir["auc"] - base["auc"] if np.isfinite(plus_ir["auc"]) and np.isfinite(base["auc"]) else math.nan
        if (np.isfinite(delta_ap) and delta_ap >= 0.03) or (np.isfinite(delta_auc) and delta_auc >= 0.03):
            summary["ir_incremental_signal"] = True
        rows.extend(
            [
                {"diagnostic_type": "incremental_ir", "target": target, "feature_or_pair": "calcium_context_plus_ir", "metric": "delta_ap", "value": delta_ap, "sample_count": int(len(test)), "interpretation": "IR incremental screening"},
                {"diagnostic_type": "incremental_ir", "target": target, "feature_or_pair": "calcium_context_plus_ir", "metric": "delta_auc", "value": delta_auc, "sample_count": int(len(test)), "interpretation": "IR incremental screening"},
                {"diagnostic_type": "incremental_ir", "target": target, "feature_or_pair": "calcium_context_plus_ir", "metric": "delta_brier", "value": plus_ir["brier"] - base["brier"] if np.isfinite(plus_ir["brier"]) and np.isfinite(base["brier"]) else math.nan, "sample_count": int(len(test)), "interpretation": "negative brier delta is better"},
            ]
        )
        tr = train[[dose_feature, ir_feature, target]].copy()
        te = test[[dose_feature, ir_feature, target]].copy()
        dose_med = pd.to_numeric(tr[dose_feature], errors="coerce").median()
        ir_med = pd.to_numeric(tr[ir_feature], errors="coerce").median()
        for frame in [tr, te]:
            frame["ca_ir_interaction"] = (pd.to_numeric(frame[dose_feature], errors="coerce") - dose_med) * (pd.to_numeric(frame[ir_feature], errors="coerce") - ir_med)
        ca_ir = train_screen_model(tr, te, [dose_feature, ir_feature], target)
        ca_ir_inter = train_screen_model(tr, te, [dose_feature, ir_feature, "ca_ir_interaction"], target)
        d_ap = ca_ir_inter["ap"] - ca_ir["ap"] if np.isfinite(ca_ir_inter["ap"]) and np.isfinite(ca_ir["ap"]) else math.nan
        d_auc = ca_ir_inter["auc"] - ca_ir["auc"] if np.isfinite(ca_ir_inter["auc"]) and np.isfinite(ca_ir["auc"]) else math.nan
        if (np.isfinite(d_ap) and d_ap >= 0.03) or (np.isfinite(d_auc) and d_auc >= 0.03):
            summary["calcium_ir_interaction_signal"] = True
        rows.extend(
            [
                {"diagnostic_type": "calcium_ir_interaction", "target": target, "feature_or_pair": f"{dose_feature}*{ir_feature}", "metric": "delta_ap", "value": d_ap, "sample_count": int(len(te)), "interpretation": "calcium-IR interaction screening"},
                {"diagnostic_type": "calcium_ir_interaction", "target": target, "feature_or_pair": f"{dose_feature}*{ir_feature}", "metric": "delta_auc", "value": d_auc, "sample_count": int(len(te)), "interpretation": "calcium-IR interaction screening"},
            ]
        )
    summary["descriptive_mediation_possible"] = bool(summary["calcium_to_ir_signal"] and summary["ir_to_t90_risk_signal"])
    return pd.DataFrame(rows), summary


def experiment_e_band_map(regime_table: pd.DataFrame, overall_rates: dict[str, float]) -> pd.DataFrame:
    rows = []
    if regime_table.empty:
        return pd.DataFrame(
            columns=[
                "regime_feature", "regime_bin", "best_dose_bin", "best_dose_min", "best_dose_max",
                "best_ok_rate", "best_low_rate", "best_high_rate", "best_out_spec_rate",
                "sample_count", "support_level", "risk_note",
            ]
        )
    valid = regime_table[regime_table["support_level"] == "enough_support"].copy()
    for (feature, regime_bin), subset in valid.groupby(["regime_feature", "regime_bin"], sort=True):
        if subset.empty:
            continue
        best = subset.sort_values(["ok_rate", "sample_count"], ascending=[False, False]).iloc[0]
        notes = []
        if best["high_rate"] > overall_rates["high_rate"] + 0.05:
            notes.append("high_t90_risk")
        if best["low_rate"] > overall_rates["low_rate"] + 0.03:
            notes.append("low_t90_risk")
        if (
            best["ok_rate"] > overall_rates["ok_rate"] + 0.03
            and best["high_rate"] <= overall_rates["high_rate"]
            and best["low_rate"] <= overall_rates["low_rate"]
        ):
            notes.append("stable_candidate")
        if not notes:
            notes.append("candidate_requires_audit")
        rows.append(
            {
                "regime_feature": feature,
                "regime_bin": regime_bin,
                "best_dose_bin": int(best["dose_bin"]),
                "best_dose_min": float(best["dose_min"]),
                "best_dose_max": float(best["dose_max"]),
                "best_ok_rate": float(best["ok_rate"]),
                "best_low_rate": float(best["low_rate"]),
                "best_high_rate": float(best["high_rate"]),
                "best_out_spec_rate": float(best["out_spec_rate"]),
                "sample_count": int(best["sample_count"]),
                "support_level": "enough_support",
                "risk_note": ";".join(notes),
            }
        )
    return pd.DataFrame(rows)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def section_title(doc_path: Path, preferred: int, text: str) -> str:
    if not doc_path.exists():
        return f"## {preferred}. {text}"
    used = []
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            prefix = line[3:].split(".", 1)[0].strip()
            if prefix.isdigit():
                used.append(int(prefix))
    number = preferred
    while number in used:
        number += 1
    return f"## {number}. {text}"


def append_docs(doc_path: Path, report: dict[str, object], strongest_interactions: pd.DataFrame) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    title = section_title(doc_path, 16, "钙单耗、工况与出口 IR 对 T90 风险的关系发现实验")
    interactions_text = "无通过筛查项"
    if not strongest_interactions.empty:
        pieces = []
        for _, row in strongest_interactions.head(5).iterrows():
            pieces.append(f"{row['context_feature']}->{row['target']} (delta_auc={row['delta_auc']}, delta_ap={row['delta_ap']})")
        interactions_text = "；".join(pieces)
    lines = [
        "",
        title,
        "",
        "本阶段用于发现钙单耗、过程工况、出口 IR 代理变量与 T90 风险之间的内部关系。该实验不是新的策略网格搜索，不训练生产模型，也不形成自动控制或影子试验建议。",
        "",
        "### 数据与特征",
        f"- 主钙单耗特征：`{report['primary_dose_feature']}`。",
        f"- 工况特征：{', '.join(report['context_features_used']) if report['context_features_used'] else '无'}。",
        f"- IR 特征：`{report['ir_feature_used']}`；IR 被视为机理相关代理变量和交互候选，不作为直接 T90 测量。",
        "",
        "### 实验摘要",
        f"- 工况分层剂量响应：有效支持的 regime×dose 分组数为 {report['regime_dose_response_summary']['enough_support_group_count']}。",
        f"- 钙×工况交互筛查：通过项数为 {report['interaction_screen_summary']['passed_interaction_count']}；主要项：{interactions_text}。",
        f"- IR 分层剂量响应：有效支持分组数为 {report['ir_stratified_summary']['enough_support_group_count']}。",
        f"- IR 描述性中介/驱动诊断：{report['mediation_diagnostic_summary']}。",
        f"- 最优钙单耗区间映射：稳定候选数为 {report['optimal_band_map_summary']['stable_candidate_count']}。",
        "",
        "### 关键判断",
    ]
    for finding in report["key_findings"]:
        lines.append(f"- {finding}")
    lines.extend(
        [
            f"- recommended_next_step：`{report['recommended_next_step']}`。",
            "",
            "### 局限",
            "- 结果来自离线观察数据，不构成因果证明。",
            "- T90 标签精度为 0.1，且人工测量误差约 0.1。",
            "- LIMS 标签稀疏，部分分层样本支持不足。",
            "- IR 覆盖率有限，当前只能作为代理/诊断变量，不作为控制动作驱动。",
            "- 本阶段不推荐自动控制和影子试验。",
            "",
        ]
    )
    with doc_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    assumptions = [
        "Relationship models are screening diagnostics only, not production T90 models.",
        "Calcium dose and IR values are not imputed before dose-response or IR analyses.",
        "All model comparisons use time-order split.",
        "No automatic control or shadow-trial recommendation is made.",
    ]
    feature_report = load_json(args.feature_report)
    dose_report = load_json(args.dose_response_report)
    ir_report = load_json(args.ir_report, required=False)
    data, dose_feature, context_features, ir_feature = load_supervised(args, dose_report, ir_report, warnings)
    target_counts = {target: data[target].value_counts(dropna=False).to_dict() for target in ["y_ok", "y_low", "y_high", "y_out_spec"]}
    overall_rates = {
        "ok_rate": float(data["y_ok"].mean()),
        "low_rate": float(data["y_low"].mean()),
        "high_rate": float(data["y_high"].mean()),
        "out_spec_rate": float(data["y_out_spec"].mean()),
    }

    regime_table = experiment_a_regime_dose_response(data, dose_feature, context_features, args.n_bins)
    interaction_table = experiment_b_interaction_screen(data, dose_feature, context_features)
    ir_strat_table = experiment_c_ir_stratified(data, dose_feature, ir_feature, args.n_bins)
    mediation_table, mediation_summary = experiment_d_mediation(data, dose_feature, context_features, ir_feature, args.n_bins)
    band_map = experiment_e_band_map(regime_table, overall_rates)

    write_csv(args.regime_output, regime_table)
    write_csv(args.interaction_output, interaction_table)
    write_csv(args.ir_strat_output, ir_strat_table)
    write_csv(args.mediation_output, mediation_table)
    write_csv(args.band_map_output, band_map)

    enough_regime_count = int((regime_table["support_level"] == "enough_support").sum()) if not regime_table.empty else 0
    enough_ir_count = int((ir_strat_table["support_level"] == "enough_support").sum()) if not ir_strat_table.empty else 0
    passed_interactions = interaction_table[interaction_table["passed_interaction_screen"].astype(bool)] if not interaction_table.empty else pd.DataFrame()
    stable_band = band_map[band_map["risk_note"].str.contains("stable_candidate", na=False)] if not band_map.empty else pd.DataFrame()

    strongest_context = []
    if not passed_interactions.empty:
        strongest_context = passed_interactions.sort_values(["delta_auc", "delta_ap"], ascending=[False, False])[
            ["context_feature", "target", "delta_auc", "delta_ap"]
        ].head(10).to_dict(orient="records")
    high_dose_risk_note = "未形成一致高剂量高 T90 风险结论"
    high_rows = regime_table[regime_table["dose_bin"] == regime_table.groupby(["regime_feature", "regime_bin"])["dose_bin"].transform("max")] if not regime_table.empty else pd.DataFrame()
    if not high_rows.empty and (high_rows["high_rate"] > overall_rates["high_rate"] + 0.05).mean() >= 0.3:
        high_dose_risk_note = "多个分层中最高钙单耗分箱的高 T90 风险高于总体水平"

    key_findings = [
        f"共发现 {len(stable_band)} 个稳定候选钙单耗区间。",
        f"交互筛查通过 {len(passed_interactions)} 个 context-target 组合。",
        f"IR 描述性中介可能性为 {mediation_summary['descriptive_mediation_possible']}，但不是因果证明。",
        high_dose_risk_note,
    ]

    if len(stable_band) >= 2 and len(passed_interactions) >= 1:
        recommended_next_step = "define_regime_specific_calcium_band_rules"
    elif len(stable_band) > 0 or len(passed_interactions) > 0 or mediation_summary["descriptive_mediation_possible"]:
        recommended_next_step = "audit_promising_regime_cases"
    elif enough_regime_count < 10 or (ir_feature is not None and float(data[ir_feature].notna().mean()) < 0.5):
        recommended_next_step = "collect_more_data_or_new_features"
    else:
        recommended_next_step = "stop_policy_work_for_now"

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features_path": str(args.features),
        "feature_report_path": str(args.feature_report),
        "dose_response_report_path": str(args.dose_response_report),
        "data_with_ir_path": str(args.data_with_ir),
        "ir_report_path": str(args.ir_report),
        "row_count": int(len(data)),
        "t90_non_null_count": int(data["t90"].notna().sum()),
        "target_counts": target_counts,
        "primary_dose_feature": dose_feature,
        "context_features_used": context_features,
        "ir_feature_used": ir_feature,
        "experiment_outputs": {
            "regime_output": str(args.regime_output),
            "interaction_output": str(args.interaction_output),
            "ir_strat_output": str(args.ir_strat_output),
            "mediation_output": str(args.mediation_output),
            "band_map_output": str(args.band_map_output),
        },
        "regime_dose_response_summary": {
            "row_count": int(len(regime_table)),
            "enough_support_group_count": enough_regime_count,
            "best_bin_rows": int(regime_table["best_dose_bin_in_regime"].sum()) if not regime_table.empty else 0,
        },
        "interaction_screen_summary": {
            "row_count": int(len(interaction_table)),
            "passed_interaction_count": int(len(passed_interactions)),
            "strongest_context_relationships": strongest_context,
        },
        "ir_stratified_summary": {
            "row_count": int(len(ir_strat_table)),
            "enough_support_group_count": enough_ir_count,
            "ir_non_null_rate": float(data[ir_feature].notna().mean()) if ir_feature else None,
        },
        "mediation_diagnostic_summary": mediation_summary,
        "optimal_band_map_summary": {
            "row_count": int(len(band_map)),
            "stable_candidate_count": int(len(stable_band)),
            "high_t90_risk_candidate_count": int(band_map["risk_note"].str.contains("high_t90_risk", na=False).sum()) if not band_map.empty else 0,
            "top_candidates": band_map.sort_values(["best_ok_rate", "sample_count"], ascending=[False, False]).head(10).to_dict(orient="records") if not band_map.empty else [],
        },
        "key_findings": key_findings,
        "warnings": warnings,
        "assumptions": assumptions,
        "recommended_next_step": recommended_next_step,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(report), handle, ensure_ascii=False, indent=2)

    append_docs(args.doc, report, passed_interactions.sort_values(["delta_auc", "delta_ap"], ascending=[False, False]) if not passed_interactions.empty else pd.DataFrame())

    print("Relationship discovery summary")
    print(f"Primary dose feature: {dose_feature}")
    print(f"Context features used: {', '.join(context_features)}")
    print(f"IR feature used: {ir_feature}")
    print(f"Regime groups with enough support: {enough_regime_count}")
    if not passed_interactions.empty:
        top = passed_interactions.sort_values(["delta_auc", "delta_ap"], ascending=[False, False]).head(5)
        print("Strongest interactions:")
        print(top[["context_feature", "target", "delta_auc", "delta_ap"]].to_string(index=False))
    else:
        print("Strongest interactions: none passed")
    print(f"IR mediation possible: {mediation_summary['descriptive_mediation_possible']}")
    print(f"Stable calcium band candidates: {len(stable_band)}")
    print(f"Recommended next step: {recommended_next_step}")
    print(f"Documentation appended: {args.doc}")


if __name__ == "__main__":
    main()
