from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline


LABEL_COLUMNS = {"time", "t90", "t90_C", "t90_D", "t90_E", "t90_label_count"}


@dataclass(frozen=True)
class Scenario:
    name: str
    delays: dict[str, int]
    description: str


def base_variable_name(column: str) -> str:
    for suffix in ("_was_missing", "_spike_flag"):
        if column.endswith(suffix):
            return column[: -len(suffix)]
    return column


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in LABEL_COLUMNS]


def build_uniform_scenario(name: str, delay: int, columns: list[str]) -> Scenario:
    return Scenario(
        name=name,
        delays={base_variable_name(column): delay for column in columns},
        description=f"all DCS features shifted by {delay} minutes",
    )


def process_delays() -> dict[str, int]:
    # Derived from the visible process-flow residence-time annotations:
    # R510A 1 min, R511A 7 min, R512A 1 min, R513 1 min, R514 1 min,
    # V530 25 min, V532 50 min, V540 50 min, T300 38 min.
    # Delay means: DCS feature at t-delay is used to predict a LIMS t90 sample at t.
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


def build_shifted_design(frame: pd.DataFrame, labels: pd.DataFrame, columns: list[str], scenario: Scenario) -> pd.DataFrame:
    result = labels[["time", "t90"]].copy()

    for column in columns:
        base = base_variable_name(column)
        delay = scenario.delays.get(base)
        if delay is None:
            continue

        part = frame[["time", column]].copy()
        part["time"] = part["time"] + pd.Timedelta(minutes=delay)
        part = part.rename(columns={column: f"{column}__lag_{delay}m"})
        result = pd.merge_asof(
            result.sort_values("time"),
            part.sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=30),
        )

    return result


def evaluate_design(design: pd.DataFrame) -> dict[str, object]:
    design = design.sort_values("time").reset_index(drop=True)
    feature_cols = [column for column in design.columns if column not in {"time", "t90"}]
    label_data = design[design["t90"].notna()].copy()

    if len(label_data) < 50:
        return {"label_rows": int(len(label_data)), "error": "not enough labeled rows"}

    X = label_data[feature_cols].copy()
    for column in X.columns:
        if X[column].dtype == bool:
            X[column] = X[column].astype("int8")
    y = label_data["t90"].astype(float)

    split = int(len(label_data) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    train_time = label_data["time"].iloc[:split]
    test_time = label_data["time"].iloc[split:]

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(n_estimators=180, learning_rate=0.035, max_depth=2, random_state=42),
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    baseline = np.full(len(y_test), y_train.mean())

    def metrics(values: np.ndarray) -> dict[str, float]:
        return {
            "mae": float(mean_absolute_error(y_test, values)),
            "rmse": float(mean_squared_error(y_test, values) ** 0.5),
            "r2": float(r2_score(y_test, values)),
        }

    model_metrics = metrics(pred)
    baseline_metrics = metrics(baseline)
    available_feature_rate = float(X.notna().mean().mean())

    return {
        "label_rows": int(len(label_data)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "feature_count": int(len(feature_cols)),
        "available_feature_rate": available_feature_rate,
        "train_time_min": train_time.min().isoformat(),
        "train_time_max": train_time.max().isoformat(),
        "test_time_min": test_time.min().isoformat(),
        "test_time_max": test_time.max().isoformat(),
        "mean_baseline": baseline_metrics,
        "model": model_metrics,
        "rmse_improvement_vs_baseline": float(baseline_metrics["rmse"] - model_metrics["rmse"]),
        "mae_improvement_vs_baseline": float(baseline_metrics["mae"] - model_metrics["mae"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate process residence-time lag suitability for sparse LIMS t90.")
    parser.add_argument("--input", default=Path("data/data_clean.parquet"), type=Path)
    parser.add_argument("--output", default=Path("data/residence_time_evaluation.json"), type=Path)
    parser.add_argument("--csv-output", default=Path("data/residence_time_evaluation.csv"), type=Path)
    parser.add_argument("--uniform-lags", default="0,30,60,90,120,150,165,174,180,210,240")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input).sort_values("time").reset_index(drop=True)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    columns = feature_columns(frame)
    labels = frame[["time", "t90"]].dropna().sort_values("time").reset_index(drop=True)

    scenarios: list[Scenario] = []
    for lag in [int(item.strip()) for item in args.uniform_lags.split(",") if item.strip()]:
        scenarios.append(build_uniform_scenario(f"uniform_{lag}min", lag, columns))
    scenarios.append(
        Scenario(
            name="process_flow_piecewise",
            delays=process_delays(),
            description="piecewise delays derived from process-flow residence times",
        )
    )

    rows: list[dict[str, object]] = []
    details: dict[str, object] = {}
    for scenario in scenarios:
        design = build_shifted_design(frame, labels, columns, scenario)
        metrics = evaluate_design(design)
        details[scenario.name] = {"description": scenario.description, "delays": scenario.delays, "metrics": metrics}
        row = {"scenario": scenario.name, "description": scenario.description}
        row.update({key: value for key, value in metrics.items() if not isinstance(value, dict)})
        if "model" in metrics:
            row.update(
                {
                    "model_mae": metrics["model"]["mae"],
                    "model_rmse": metrics["model"]["rmse"],
                    "model_r2": metrics["model"]["r2"],
                    "baseline_mae": metrics["mean_baseline"]["mae"],
                    "baseline_rmse": metrics["mean_baseline"]["rmse"],
                    "baseline_r2": metrics["mean_baseline"]["r2"],
                }
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    ranked = table.dropna(subset=["model_rmse"]).sort_values("model_rmse").reset_index(drop=True)
    best = ranked.iloc[0].to_dict() if len(ranked) else None
    process_row = table[table["scenario"] == "process_flow_piecewise"]
    process_rank = None
    if len(ranked) and len(process_row):
        process_rank = int(ranked.index[ranked["scenario"] == "process_flow_piecewise"][0] + 1) if "process_flow_piecewise" in set(ranked["scenario"]) else None

    summary = {
        "input": str(args.input),
        "label": "t90",
        "label_rows": int(len(labels)),
        "residence_time_from_flow_min": {
            "R510A": 1,
            "R511A": 7,
            "R512A": 1,
            "R513": 1,
            "R514": 1,
            "V530": 25,
            "V532": 50,
            "V540": 50,
            "T300": 38,
            "total_main_path": 174,
        },
        "process_piecewise_rank_by_rmse": process_rank,
        "best_scenario_by_rmse": best,
        "judgement": None,
        "scenarios": details,
    }

    if best and process_rank is not None:
        process_metrics = details["process_flow_piecewise"]["metrics"]
        best_rmse = float(best["model_rmse"])
        process_rmse = float(process_metrics["model"]["rmse"])
        gap = process_rmse - best_rmse
        summary["process_vs_best_rmse_gap"] = gap
        if process_rank <= 3 or gap <= 0.01:
            summary["judgement"] = "process_residence_time_is_reasonable"
        else:
            summary["judgement"] = "process_residence_time_is_not_best_for_instantaneous_features"

    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    table.to_csv(args.csv_output, index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
