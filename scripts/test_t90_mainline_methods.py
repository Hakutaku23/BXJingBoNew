from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


LABEL_COLUMNS = {"time", "t90", "t90_C", "t90_D", "t90_E", "t90_label_count"}
T90_TARGET = 8.45
T90_LOW = 8.20
T90_HIGH = 8.70


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(p, eps, 1 - eps)
    return np.log(clipped / (1 - clipped))


def inv_logit(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def process_delays() -> dict[str, int]:
    return {
        "卤化工段胶液总量2": 174,
        "反应溴添加量": 174,
        "储罐胶浓在线检测": 174,
        "R510A温度": 174,
        "R511A温度": 173,
        "R512A温度": 166,
        "硬脂酸钙加注量": 165,
        "ESBO加注量": 165,
        "中和碱液添加量": 165,
        "R513温度": 165,
        "R514温度": 164,
    }


def selected_variables(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in LABEL_COLUMNS and not column.endswith("_was_missing") and not column.endswith("_spike_flag")]


def rolling_features_for_series(series: pd.Series, prefix: str, windows: list[int]) -> pd.DataFrame:
    features = pd.DataFrame(index=series.index)
    for window in windows:
        rolling = series.rolling(window=window, min_periods=max(3, window // 3))
        features[f"{prefix}_w{window}_mean"] = rolling.mean()
        features[f"{prefix}_w{window}_std"] = rolling.std()
        features[f"{prefix}_w{window}_min"] = rolling.min()
        features[f"{prefix}_w{window}_max"] = rolling.max()
        features[f"{prefix}_w{window}_range"] = features[f"{prefix}_w{window}_max"] - features[f"{prefix}_w{window}_min"]
        features[f"{prefix}_w{window}_last"] = series
        features[f"{prefix}_w{window}_slope"] = (series - series.shift(window - 1)) / max(1, window - 1)
    return features


def build_lagged_feature_table(frame: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    variables = selected_variables(frame)
    delays = process_delays()
    pieces: list[pd.DataFrame] = []

    for variable in variables:
        delay = delays[variable]
        values = pd.to_numeric(frame[variable], errors="coerce")
        value_features = rolling_features_for_series(values, variable, windows)

        missing_col = f"{variable}_was_missing"
        spike_col = f"{variable}_spike_flag"
        for window in windows:
            if missing_col in frame.columns:
                value_features[f"{variable}_w{window}_missing_rate"] = (
                    frame[missing_col].astype(float).rolling(window=window, min_periods=max(3, window // 3)).mean()
                )
            if spike_col in frame.columns:
                value_features[f"{variable}_w{window}_spike_rate"] = (
                    frame[spike_col].astype(float).rolling(window=window, min_periods=max(3, window // 3)).mean()
                )

        value_features = value_features.reset_index(drop=True)
        value_features.insert(0, "time", frame["time"].reset_index(drop=True) + pd.Timedelta(minutes=delay))
        pieces.append(value_features)

    feature_table = pieces[0]
    for piece in pieces[1:]:
        feature_table = pd.merge_asof(
            feature_table.sort_values("time"),
            piece.sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=30),
        )
    return feature_table.sort_values("time").reset_index(drop=True)


def desirability(t90: pd.Series, width: float = 0.25, power: float = 4.0) -> pd.Series:
    return 1 / (1 + (np.abs(t90 - T90_TARGET) / width) ** power)


def build_dataset(frame: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    labels = frame[frame["t90"].notna()][["time", "t90", "t90_C", "t90_D", "t90_E", "t90_label_count"]].copy()
    labels["y_out_spec"] = ((labels["t90"] < T90_LOW) | (labels["t90"] > T90_HIGH)).astype(int)
    labels["risk_score"] = 1 - desirability(labels["t90"])

    feature_table = build_lagged_feature_table(frame, windows)
    dataset = pd.merge_asof(
        labels.sort_values("time"),
        feature_table.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=30),
    )
    return dataset.sort_values("time").reset_index(drop=True)


def score_features(train: pd.DataFrame, features: list[str]) -> pd.Series:
    y = train["y_out_spec"].astype(float)
    scores: dict[str, float] = {}
    for feature in features:
        x = pd.to_numeric(train[feature], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 50 or x[valid].nunique() <= 1:
            scores[feature] = 0.0
            continue
        corr = np.corrcoef(x[valid], y[valid])[0, 1]
        scores[feature] = float(abs(corr)) if np.isfinite(corr) else 0.0
    return pd.Series(scores).sort_values(ascending=False)


def probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    unique = np.unique(y_true)
    auc = float(roc_auc_score(y_true, probability)) if len(unique) == 2 else None
    return {
        "brier": float(brier_score_loss(y_true, probability)),
        "ap": float(average_precision_score(y_true, probability)),
        "auc": auc,
    }


def threshold_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probability >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_alarm_rate": float(((pred == 1) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())),
        "alarm_rate": float(pred.mean()),
    }


def scan_thresholds(y_true: np.ndarray, probability: np.ndarray) -> dict[str, dict[str, float]]:
    thresholds = np.linspace(0.02, 0.80, 157)
    rows = [threshold_metrics(y_true, probability, threshold) for threshold in thresholds]
    table = pd.DataFrame(rows)
    best_f1 = table.sort_values(["f1", "recall"], ascending=False).iloc[0].to_dict()
    high_recall = table[table["recall"] >= 0.75]
    if len(high_recall):
        high_recall_best = high_recall.sort_values(["false_alarm_rate", "f1"], ascending=[True, False]).iloc[0].to_dict()
    else:
        high_recall_best = {}
    alarm_014 = threshold_metrics(y_true, probability, 0.14)
    return {"best_f1": best_f1, "high_recall": high_recall_best, "alarm_0_14": alarm_014}


def fit_autogluon(train: pd.DataFrame, test: pd.DataFrame, features: list[str], model_dir: Path, time_limit: int) -> tuple[np.ndarray, dict[str, object]]:
    try:
        from autogluon.tabular import TabularPredictor

        model_dir.mkdir(parents=True, exist_ok=True)
        train_ag = train[features + ["y_out_spec"]].copy()
        test_ag = test[features].copy()
        predictor = TabularPredictor(
            label="y_out_spec",
            problem_type="binary",
            eval_metric="average_precision",
            path=str(model_dir),
            verbosity=0,
        )
        predictor.fit(
            train_ag,
            presets="medium_quality",
            time_limit=time_limit,
            ag_args_fit={"num_cpus": 1},
        )
        proba = predictor.predict_proba(test_ag)
        if isinstance(proba, pd.DataFrame):
            probability = proba[1].to_numpy() if 1 in proba.columns else proba.iloc[:, -1].to_numpy()
        else:
            probability = np.asarray(proba)
        info = {
            "engine": "AutoGluon TabularPredictor",
            "model_dir": str(model_dir),
            "leaderboard": predictor.leaderboard(silent=True).head(10).to_dict(orient="records"),
        }
        return probability, info
    except Exception as exc:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingClassifier(n_estimators=180, learning_rate=0.04, max_depth=2, random_state=42),
        )
        model.fit(train[features], train["y_out_spec"])
        probability = model.predict_proba(test[features])[:, 1]
        return probability, {"engine": "sklearn fallback GradientBoostingClassifier", "error": repr(exc)}


def jitl_probability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    neighbor_max_k: int,
    min_neighbors: int,
    radius_multiplier: float,
) -> tuple[np.ndarray, dict[str, float]]:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[features]))
    x_test = scaler.transform(imputer.transform(test[features]))
    y_train = train["y_out_spec"].to_numpy()
    global_rate = float(y_train.mean())

    probabilities: list[float] = []
    neighbor_counts: list[int] = []
    median_distances: list[float] = []

    for row in x_test:
        distances = np.nanmedian(np.abs(x_train - row), axis=1)
        order = np.argsort(distances)
        sorted_distances = distances[order]
        radius = np.nanmedian(sorted_distances[: max(min_neighbors, min(neighbor_max_k, len(sorted_distances)))]) * radius_multiplier
        candidate_idx = order[sorted_distances <= radius]
        if len(candidate_idx) < min_neighbors:
            candidate_idx = order[: min(neighbor_max_k, len(order))]
        else:
            candidate_idx = candidate_idx[:neighbor_max_k]
        if len(candidate_idx) < min_neighbors:
            probabilities.append(global_rate)
        else:
            probabilities.append(float(y_train[candidate_idx].mean()))
        neighbor_counts.append(int(len(candidate_idx)))
        median_distances.append(float(np.nanmedian(distances[candidate_idx])) if len(candidate_idx) else math.nan)

    diagnostics = {
        "neighbor_count_mean": float(np.mean(neighbor_counts)),
        "neighbor_count_min": int(np.min(neighbor_counts)) if neighbor_counts else 0,
        "neighbor_count_max": int(np.max(neighbor_counts)) if neighbor_counts else 0,
        "median_distance_mean": float(np.nanmean(median_distances)),
        "global_train_out_spec_rate": global_rate,
    }
    return np.asarray(probabilities), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Test T90 mainline probability/JITL methods on selected variables.")
    parser.add_argument("--input", default=Path("data/data_clean.parquet"), type=Path)
    parser.add_argument("--output", default=Path("data/t90_mainline_selected_variables_report.json"), type=Path)
    parser.add_argument("--dataset-output", default=Path("data/t90_mainline_selected_variables_dataset.parquet"), type=Path)
    parser.add_argument("--csv-output", default=Path("data/t90_mainline_selected_variables_metrics.csv"), type=Path)
    parser.add_argument("--model-dir", default=Path("models/t90_mainline_global_autogluon"), type=Path)
    parser.add_argument("--windows", default="15,30,60")
    parser.add_argument("--global-top-n", default=120, type=int)
    parser.add_argument("--search-top-n", default=40, type=int)
    parser.add_argument("--autogluon-time-limit", default=120, type=int)
    parser.add_argument("--alpha", default=0.40, type=float)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input).sort_values("time").reset_index(drop=True)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    windows = [int(item.strip()) for item in args.windows.split(",") if item.strip()]
    dataset = build_dataset(frame, windows)
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.dataset_output, index=False)

    all_features = [
        column
        for column in dataset.columns
        if column not in {"time", "t90", "t90_C", "t90_D", "t90_E", "t90_label_count", "y_out_spec", "risk_score"}
    ]
    split = int(len(dataset) * 0.8)
    train = dataset.iloc[:split].copy()
    test = dataset.iloc[split:].copy()
    feature_scores = score_features(train, all_features)
    global_features = feature_scores.head(min(args.global_top_n, len(feature_scores))).index.tolist()
    search_features = feature_scores.head(min(args.search_top_n, len(feature_scores))).index.tolist()

    p_global, global_info = fit_autogluon(train, test, global_features, args.model_dir, args.autogluon_time_limit)
    p_jitl, jitl_info = jitl_probability(
        train,
        test,
        search_features,
        neighbor_max_k=50,
        min_neighbors=20,
        radius_multiplier=3.0,
    )
    p_final = inv_logit((1 - args.alpha) * logit(p_global) + args.alpha * logit(p_jitl))
    y_test = test["y_out_spec"].to_numpy()

    models = {
        "p_global": p_global,
        "p_jitl": p_jitl,
        "p_final_alpha_0_40": p_final,
    }
    rows = []
    metrics = {}
    for name, probability in models.items():
        probability_metrics_result = probability_metrics(y_test, probability)
        threshold_result = scan_thresholds(y_test, probability)
        metrics[name] = {"probability": probability_metrics_result, "thresholds": threshold_result}
        row = {"model": name}
        row.update(probability_metrics_result)
        row.update({f"alarm_0_14_{k}": v for k, v in threshold_result["alarm_0_14"].items()})
        row.update({f"best_f1_{k}": v for k, v in threshold_result["best_f1"].items()})
        rows.append(row)

    result_table = pd.DataFrame(rows)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(args.csv_output, index=False, encoding="utf-8-sig")

    report = {
        "input": str(args.input),
        "dataset_output": str(args.dataset_output),
        "label_definition": {
            "target": "P(T90 out spec)",
            "t90_target": T90_TARGET,
            "in_spec_range": [T90_LOW, T90_HIGH],
            "y_out_spec": "1 if t90 < 8.20 or t90 > 8.70 else 0",
        },
        "rows": int(len(dataset)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "out_spec_rate_total": float(dataset["y_out_spec"].mean()),
        "out_spec_rate_train": float(train["y_out_spec"].mean()),
        "out_spec_rate_test": float(test["y_out_spec"].mean()),
        "time_split": {
            "train_min": train["time"].min().isoformat(),
            "train_max": train["time"].max().isoformat(),
            "test_min": test["time"].min().isoformat(),
            "test_max": test["time"].max().isoformat(),
        },
        "feature_engineering": {
            "selected_variables": selected_variables(frame),
            "windows": windows,
            "process_delays_min": process_delays(),
            "all_feature_count": int(len(all_features)),
            "global_feature_count": int(len(global_features)),
            "search_feature_count": int(len(search_features)),
            "global_top_features": global_features[:30],
            "search_top_features": search_features[:40],
        },
        "global_model": global_info,
        "jitl": {
            "similarity_metric": "median_l1 over robust-scaled search features",
            "radius_multiplier": 3.0,
            "neighbor_max_k": 50,
            "min_neighbors": 20,
            "diagnostics": jitl_info,
        },
        "fusion": {
            "method": "logit(p_final) = 0.6 * logit(p_global) + 0.4 * logit(p_jitl)",
            "alpha": args.alpha,
        },
        "metrics": metrics,
        "best_by_ap": result_table.sort_values("ap", ascending=False).iloc[0].to_dict(),
        "best_by_brier": result_table.sort_values("brier", ascending=True).iloc[0].to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
