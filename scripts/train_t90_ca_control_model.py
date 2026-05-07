from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    HAS_HIST_GRADIENT_BOOSTING = True
except Exception:
    HAS_HIST_GRADIENT_BOOSTING = False


RANDOM_SEED = 42
REQUIRED_COLUMNS = ["time", "t90", "y_ok", "y_low", "y_high", "y_out_spec"]
REGRESSION_TARGET = "t90"
CLASSIFICATION_TARGETS = ["y_ok", "y_low", "y_high", "y_out_spec"]
ALL_TARGETS = [REGRESSION_TARGET] + CLASSIFICATION_TARGETS
LEAKAGE_EXCLUSIONS = {
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
MODEL_FILENAMES = {
    "t90": "model_t90_reg.joblib",
    "y_ok": "model_y_ok_clf.joblib",
    "y_low": "model_y_low_clf.joblib",
    "y_high": "model_y_high_clf.joblib",
    "y_out_spec": "model_y_out_spec_clf.joblib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train T90 calcium-control models with train-only feature selection.")
    parser.add_argument("--input", type=Path, default=Path("data/t90_ca_feature_dataset.parquet"))
    parser.add_argument("--feature-report", type=Path, default=Path("data/t90_ca_feature_report.json"))
    parser.add_argument("--dose-response-report", type=Path, default=Path("data/t90_ca_dose_response_report.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/t90_ca_control"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/t90_ca_control_metrics.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/t90_ca_control_report.json"))
    parser.add_argument("--doc", type=Path, default=Path("docs/Experimental_Procedure_cn.md"))
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--global-top-n", type=int, default=120)
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


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input parquet does not exist: {path}")
    frame = pd.read_parquet(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    invalid_time_count = int(frame["time"].isna().sum())
    if invalid_time_count:
        raise ValueError(f"Input dataset contains {invalid_time_count} invalid time values.")
    return frame.sort_values("time").reset_index(drop=True)


def is_leakage_column(column: str, leakage_columns: set[str]) -> bool:
    lowered = column.lower()
    return (
        column in leakage_columns
        or lowered.startswith("pred_")
        or lowered.startswith("p_")
        or lowered.endswith("_pred")
        or lowered.startswith("target_")
        or lowered.endswith("_target")
    )


def get_feature_groups(feature_report: dict[str, object], frame: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    groups = feature_report.get("feature_groups", {})
    if not isinstance(groups, dict):
        raise ValueError("Feature report does not contain feature_groups.")

    calcium_core = groups.get("calcium_core_features", [])
    process_context = groups.get("process_context_features", [])
    all_candidate = groups.get("all_candidate_features", [])
    leakage_report = groups.get("leakage_excluded_columns", [])
    if not all(isinstance(item, list) for item in [calcium_core, process_context, all_candidate, leakage_report]):
        raise ValueError("Feature report feature groups must be lists.")

    leakage_columns = set(str(column) for column in leakage_report) | LEAKAGE_EXCLUSIONS
    available = set(frame.columns)
    calcium_core = [str(feature) for feature in calcium_core if str(feature) in available]
    process_context = [str(feature) for feature in process_context if str(feature) in available]
    all_candidate = [
        str(feature)
        for feature in all_candidate
        if str(feature) in available and not is_leakage_column(str(feature), leakage_columns)
    ]
    return calcium_core, process_context, all_candidate, sorted(leakage_columns)


def time_order_split(frame: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(frame) * train_fraction)
    if split_index <= 0 or split_index >= len(frame):
        raise ValueError("Dataset is too small for an 80/20 time-ordered split.")
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def spearman_abs_score(x: pd.Series, y: pd.Series) -> float | None:
    values = pd.DataFrame(
        {"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}
    ).dropna()
    if len(values) < 30 or values["x"].nunique() <= 1 or values["y"].nunique() <= 1:
        return None
    corr = values["x"].corr(values["y"], method="spearman")
    if corr is None or not np.isfinite(corr):
        return None
    return float(abs(corr))


def select_features_train_only(
    train: pd.DataFrame,
    calcium_core: list[str],
    all_candidate: list[str],
    top_n: int,
) -> dict[str, object]:
    calcium_set = set(calcium_core)
    score_rows: list[dict[str, object]] = []
    for feature in all_candidate:
        target_scores: dict[str, float | None] = {}
        valid_scores: list[float] = []
        for target in ALL_TARGETS:
            score = spearman_abs_score(train[feature], train[target])
            target_scores[target] = score
            if score is not None:
                valid_scores.append(score)
        if not valid_scores:
            continue
        score_rows.append(
            {
                "feature": feature,
                "aggregate_score": float(np.mean(valid_scores)),
                "valid_target_score_count": int(len(valid_scores)),
                "is_calcium_core": feature in calcium_set,
                "target_scores": target_scores,
                "candidate_order": all_candidate.index(feature),
            }
        )

    score_rows = sorted(
        score_rows,
        key=lambda item: (
            -float(item["aggregate_score"]),
            -int(bool(item["is_calcium_core"])),
            int(item["candidate_order"]),
        ),
    )
    global_selected = [str(row["feature"]) for row in score_rows[:top_n]]
    calcium_available = [feature for feature in calcium_core if feature in all_candidate]

    model_features: list[str] = []
    for feature in calcium_available + global_selected:
        if feature not in model_features:
            model_features.append(feature)

    return {
        "method": "train-only mean absolute Spearman score across t90/y_ok/y_low/y_high/y_out_spec",
        "global_top_n_requested": int(top_n),
        "calcium_core": calcium_available,
        "global_selected": global_selected,
        "model_features": model_features,
        "feature_scores": score_rows,
    }


def regression_model(use_hist_gradient_boosting: bool = True) -> object:
    if HAS_HIST_GRADIENT_BOOSTING and use_hist_gradient_boosting:
        estimator = HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.04,
            l2_regularization=0.05,
            random_state=RANDOM_SEED,
        )
    else:
        estimator = GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=2,
            random_state=RANDOM_SEED,
        )
    return make_pipeline(SimpleImputer(strategy="median"), estimator)


def classification_model(
    y_train: pd.Series,
    target: str,
    warnings: list[str],
    use_hist_gradient_boosting: bool = True,
) -> object:
    positives = int(pd.to_numeric(y_train, errors="coerce").sum())
    unique = pd.Series(y_train).dropna().nunique()
    if unique < 2:
        warnings.append(f"{target}: training data has only one class; using DummyClassifier.")
        return make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="prior", random_state=RANDOM_SEED))
    if positives < 10:
        warnings.append(f"{target}: training positives are fewer than 10; metrics may be unstable.")
    if HAS_HIST_GRADIENT_BOOSTING and use_hist_gradient_boosting:
        estimator = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.04,
            l2_regularization=0.05,
            random_state=RANDOM_SEED,
        )
    else:
        estimator = GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=2,
            random_state=RANDOM_SEED,
        )
    return make_pipeline(SimpleImputer(strategy="median"), estimator)


def fit_regression_with_fallback(x_train: pd.DataFrame, y_train: pd.Series, warnings: list[str]) -> object:
    model = regression_model(use_hist_gradient_boosting=True)
    try:
        model.fit(x_train, y_train)
        return model
    except Exception as exc:
        if not HAS_HIST_GRADIENT_BOOSTING:
            raise
        warnings.append(
            f"t90: HistGradientBoostingRegressor failed with {repr(exc)}; falling back to GradientBoostingRegressor."
        )
        model = regression_model(use_hist_gradient_boosting=False)
        model.fit(x_train, y_train)
        return model


def fit_classifier_with_fallback(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    target: str,
    warnings: list[str],
) -> object:
    model = classification_model(y_train, target, warnings, use_hist_gradient_boosting=True)
    try:
        model.fit(x_train, y_train)
        return model
    except Exception as exc:
        if not HAS_HIST_GRADIENT_BOOSTING or isinstance(model[-1], DummyClassifier):
            raise
        warnings.append(
            f"{target}: HistGradientBoostingClassifier failed with {repr(exc)}; falling back to GradientBoostingClassifier."
        )
        model = classification_model(y_train, target, warnings, use_hist_gradient_boosting=False)
        model.fit(x_train, y_train)
        return model


def probability_from_model(model: object, features: pd.DataFrame, target: str, warnings: list[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        if proba.shape[1] == 1:
            classes = getattr(model, "classes_", None)
            only_class = int(classes[0]) if classes is not None and len(classes) else 0
            return np.ones(len(features)) if only_class == 1 else np.zeros(len(features))
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        warnings.append(f"{target}: model lacks predict_proba; using logistic transform of decision_function.")
        scores = model.decision_function(features)
        return 1.0 / (1.0 + np.exp(-scores))
    warnings.append(f"{target}: model lacks probability output; using hard predictions as probability-like scores.")
    return np.asarray(model.predict(features), dtype=float)


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(errors**2)))


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray, train_mean: float) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else math.nan,
        "mean_baseline_mae": float(mean_absolute_error(y_true, np.full(len(y_true), train_mean))),
        "mean_baseline_rmse": rmse(y_true, np.full(len(y_true), train_mean)),
    }


def threshold_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probability >= threshold).astype(int)
    negatives = max(1, int((y_true == 0).sum()))
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_positive_rate": float(((pred == 1) & (y_true == 0)).sum() / negatives),
        "alarm_rate": float(pred.mean()),
        "positive_rate": float(y_true.mean()),
    }


def select_best_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    thresholds = np.unique(np.quantile(probability, np.linspace(0.02, 0.98, 97)))
    thresholds = np.unique(np.concatenate([thresholds, np.linspace(0.02, 0.98, 97), np.asarray([0.5])]))
    best_threshold = 0.5
    best_key = (-1.0, -1.0, 0.0)
    for threshold in thresholds:
        metrics = threshold_metrics(y_true, probability, float(threshold))
        key = (metrics["f1"], metrics["recall"], -metrics["false_positive_rate"])
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def classification_metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float | None]:
    y_array = np.asarray(y_true, dtype=int)
    unique = np.unique(y_array)
    result: dict[str, float | None] = {
        "average_precision": None,
        "roc_auc": None,
        "brier_score": None,
    }
    if len(unique) >= 2:
        result["average_precision"] = float(average_precision_score(y_array, probability))
        result["roc_auc"] = float(roc_auc_score(y_array, probability))
        result["brier_score"] = float(brier_score_loss(y_array, probability))
    result.update(threshold_metrics(y_array, probability, threshold))
    return result


def metric_rows(target: str, task_type: str, split: str, metrics: dict[str, float | None]) -> list[dict[str, object]]:
    return [
        {"target": target, "task_type": task_type, "split": split, "metric": metric, "value": value}
        for metric, value in metrics.items()
    ]


def train_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_features: list[str],
    model_dir: Path,
    warnings: list[str],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, str]]:
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    model_types: dict[str, str] = {}

    x_train = train[model_features]
    x_test = test[model_features]

    reg = fit_regression_with_fallback(x_train, train["t90"], warnings)
    train_pred = reg.predict(x_train)
    test_pred = reg.predict(x_test)
    train_mean = float(train["t90"].mean())
    train_metrics = regression_metrics(train["t90"], train_pred, train_mean)
    test_metrics = regression_metrics(test["t90"], test_pred, train_mean)
    metrics["t90"] = {"task_type": "regression", "train": train_metrics, "test": test_metrics}
    rows.extend(metric_rows("t90", "regression", "train", train_metrics))
    rows.extend(metric_rows("t90", "regression", "test", test_metrics))
    joblib.dump(reg, model_dir / MODEL_FILENAMES["t90"])
    model_types["t90"] = type(reg[-1]).__name__ if hasattr(reg, "__getitem__") else type(reg).__name__

    for target in CLASSIFICATION_TARGETS:
        clf = fit_classifier_with_fallback(x_train, train[target], target, warnings)
        train_probability = probability_from_model(clf, x_train, target, warnings)
        test_probability = probability_from_model(clf, x_test, target, warnings)
        threshold = select_best_f1_threshold(np.asarray(train[target], dtype=int), train_probability)
        train_metrics = classification_metrics(train[target], train_probability, threshold)
        test_metrics = classification_metrics(test[target], test_probability, threshold)
        metrics[target] = {
            "task_type": "classification",
            "threshold_selected_on_train": threshold,
            "train": train_metrics,
            "test": test_metrics,
        }
        rows.extend(metric_rows(target, "classification", "train", train_metrics))
        rows.extend(metric_rows(target, "classification", "test", test_metrics))
        joblib.dump(clf, model_dir / MODEL_FILENAMES[target])
        model_types[target] = type(clf[-1]).__name__ if hasattr(clf, "__getitem__") else type(clf).__name__

    return metrics, rows, model_types


def target_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for target in CLASSIFICATION_TARGETS:
        counts = frame[target].value_counts(dropna=False).sort_index()
        result[target] = {str(key): int(value) for key, value in counts.items()}
    return result


def useful_signal(test_metrics: dict[str, object]) -> bool:
    positive_rate = test_metrics.get("positive_rate")
    ap = test_metrics.get("average_precision")
    auc = test_metrics.get("roc_auc")
    ap_useful = ap is not None and positive_rate is not None and float(ap) >= float(positive_rate) + 0.03
    auc_useful = auc is not None and float(auc) >= 0.58
    return bool(ap_useful or auc_useful)


def decide_next_step(metrics: dict[str, object], train: pd.DataFrame, test: pd.DataFrame) -> str:
    class_metrics = {
        target: metrics[target]["test"]
        for target in CLASSIFICATION_TARGETS
        if target in metrics and isinstance(metrics[target], dict)
    }
    useful = {target: useful_signal(target_metrics) for target, target_metrics in class_metrics.items()}
    low_high_not_degenerate = (
        train["y_low"].nunique() >= 2
        and test["y_low"].nunique() >= 2
        and train["y_high"].nunique() >= 2
        and test["y_high"].nunique() >= 2
    )
    if (useful.get("y_ok") or useful.get("y_out_spec")) and low_high_not_degenerate:
        return "proceed_to_offline_policy_simulation"
    if any(useful.values()):
        return "inspect_model_metrics_before_policy"
    return "insufficient_model_signal_stop"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def format_metric(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        if not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def append_documentation(
    doc_path: Path,
    args: argparse.Namespace,
    dose_report: dict[str, object],
    report: dict[str, object],
    model_types: dict[str, str],
) -> bool:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    dose_primary = dose_report.get("primary_dose_feature")
    dose_diag = {}
    if isinstance(dose_report.get("per_feature_diagnostics"), dict) and dose_primary in dose_report["per_feature_diagnostics"]:
        dose_diag = dose_report["per_feature_diagnostics"][dose_primary]
    dose_rates = dose_report.get("overall_rates", {}) if isinstance(dose_report.get("overall_rates"), dict) else {}
    flags = dose_diag.get("interpretation_flags", {}) if isinstance(dose_diag, dict) else {}
    metrics = report["metrics"]
    lines = [
        "",
        "## 10. 硬脂酸钙剂量响应与控制模型实验",
        "",
        "### 10.1 第二阶段：硬脂酸钙剂量响应分析结果",
        "",
        f"- 输入文件：`{args.input}`、`{args.feature_report}`。",
        f"- 输出文件：`{Path('data/t90_ca_dose_response_bins.csv')}`、`{args.dose_response_report}`。",
        f"- 主剂量特征：`{dose_primary}`。",
        f"- 可用样本数：{dose_diag.get('usable_sample_count', 'NA')}；有效分箱数：{dose_diag.get('effective_bin_count', 'NA')}。",
        (
            "- 总体比例："
            f"ok_rate={format_metric(dose_rates.get('ok_rate'))}，"
            f"low_rate={format_metric(dose_rates.get('low_rate'))}，"
            f"high_rate={format_metric(dose_rates.get('high_rate'))}，"
            f"out_spec_rate={format_metric(dose_rates.get('out_spec_rate'))}。"
        ),
        (
            "- Spearman 相关："
            f"dose-t90={format_metric(dose_diag.get('spearman_corr_dose_t90'))}，"
            f"dose-y_ok={format_metric(dose_diag.get('spearman_corr_dose_y_ok'))}，"
            f"dose-y_low={format_metric(dose_diag.get('spearman_corr_dose_y_low'))}，"
            f"dose-y_high={format_metric(dose_diag.get('spearman_corr_dose_y_high'))}，"
            f"dose-y_out_spec={format_metric(dose_diag.get('spearman_corr_dose_y_out_spec'))}。"
        ),
        f"- 最佳合格率分箱：bin {dose_diag.get('best_ok_rate_bin', 'NA')}，ok_rate={format_metric(dose_diag.get('best_ok_rate'))}。",
        (
            "- 最高剂量分箱表现："
            f"ok_rate={format_metric(dose_diag.get('ok_rate_at_highest_bin'))}，"
            f"high_rate={format_metric(dose_diag.get('high_rate_at_highest_bin'))}。"
        ),
        f"- 解释标记：{json.dumps(as_jsonable(flags), ensure_ascii=False)}。",
        "- 结论：硬脂酸钙信号存在，但关系不是简单的“加得越多越好”；低 T90 与高 T90 风险方向存在权衡，后续必须分别建模。",
        "",
        "### 10.2 第三阶段：控制模型训练",
        "",
        "- 脚本：`scripts/train_t90_ca_control_model.py`。",
        (
            "- 执行命令："
            "`D:\\miniconda3\\envs\\autoGluon\\python.exe .\\scripts\\train_t90_ca_control_model.py "
            "--input .\\data\\t90_ca_feature_dataset.parquet "
            "--feature-report .\\data\\t90_ca_feature_report.json "
            "--dose-response-report .\\data\\t90_ca_dose_response_report.json "
            "--model-dir .\\models\\t90_ca_control "
            "--metrics-output .\\data\\t90_ca_control_metrics.csv "
            "--report .\\data\\t90_ca_control_report.json "
            "--doc .\\docs\\Experimental_Procedure_cn.md "
            "--time-limit 300`。"
        ),
        f"- 输入文件：`{args.input}`、`{args.feature_report}`、`{args.dose_response_report}`。",
        f"- 输出文件：`{args.model_dir}`、`{args.metrics_output}`、`{args.report}`。",
        "- 切分方法：按 `time` 排序后，前 80% 为训练集，后 20% 为测试集；未使用随机切分。",
        f"- 训练集/测试集样本数：{report['train_row_count']} / {report['test_row_count']}。",
        (
            "- 训练时间范围："
            f"{report['train_time_range']['min']} 至 {report['train_time_range']['max']}；"
            "测试时间范围："
            f"{report['test_time_range']['min']} 至 {report['test_time_range']['max']}。"
        ),
        (
            "- 特征选择："
            f"钙剂核心特征 {report['calcium_core_feature_count']} 个，"
            f"全局筛选特征 {report['global_selected_feature_count']} 个，"
            f"最终模型特征 {report['model_feature_count']} 个；筛选仅在训练集上完成。"
        ),
        "- 模型类型：" + "，".join(f"{target}={model_type}" for target, model_type in model_types.items()) + "。",
        "- 测试集指标摘要：",
    ]

    t90_test = metrics["t90"]["test"]
    lines.append(
        f"  - t90：MAE={format_metric(t90_test.get('mae'))}，RMSE={format_metric(t90_test.get('rmse'))}，R2={format_metric(t90_test.get('r2'))}。"
    )
    for target in CLASSIFICATION_TARGETS:
        test_metrics = metrics[target]["test"]
        lines.append(
            f"  - {target}：AP={format_metric(test_metrics.get('average_precision'))}，"
            f"AUC={format_metric(test_metrics.get('roc_auc'))}，"
            f"Brier={format_metric(test_metrics.get('brier_score'))}，"
            f"threshold={format_metric(test_metrics.get('threshold'))}，"
            f"F1={format_metric(test_metrics.get('f1'))}，"
            f"Recall={format_metric(test_metrics.get('recall'))}，"
            f"Precision={format_metric(test_metrics.get('precision'))}。"
        )
    lines.extend(
        [
            f"- recommended_next_step：`{report['recommended_next_step']}`。",
            "- 警告：" + ("；".join(report["warnings"]) if report["warnings"] else "无。"),
            "",
            "### 10.3 当前判断",
            "",
            f"- 当前建议：`{report['recommended_next_step']}`。",
            "- 若 `y_ok` 或 `y_out_spec` 在测试集上相对基准有增益，且 `y_low`/`y_high` 不退化，可以进入离线策略模拟；否则需要先人工检查模型指标。",
            "- 低 T90 样本数量较少，`y_low` 模型可靠性应谨慎看待，不宜单独作为自动控制依据。",
            "- 高剂量区域在第二阶段表现出较高的高 T90 风险，策略模拟必须限制推荐幅度，并分别检查低/高 T90 风险。",
            "- 当前结论来自离线观测数据，不能证明钙剂调整的因果效果；上线试验前仍需工艺工程师审核。",
            "- 后续所有特征选择、阈值选择和校准步骤必须保持训练集内完成，不能使用测试集信息。",
        ]
    )
    with doc_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    return True


def build_report(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_selection: dict[str, object],
    metrics: dict[str, object],
    model_types: dict[str, str],
    warnings: list[str],
) -> dict[str, object]:
    recommended_next_step = decide_next_step(metrics, train, test)
    return {
        "input_path": str(args.input),
        "feature_report_path": str(args.feature_report),
        "dose_response_report_path": str(args.dose_response_report),
        "model_dir": str(args.model_dir),
        "metrics_output_path": str(args.metrics_output),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": int(len(frame)),
        "train_row_count": int(len(train)),
        "test_row_count": int(len(test)),
        "train_time_range": {
            "min": train["time"].min().isoformat(),
            "max": train["time"].max().isoformat(),
        },
        "test_time_range": {
            "min": test["time"].min().isoformat(),
            "max": test["time"].max().isoformat(),
        },
        "target_counts": {
            "all": target_counts(frame),
            "train": target_counts(train),
            "test": target_counts(test),
        },
        "feature_selection": {
            "method": feature_selection["method"],
            "global_top_n_requested": feature_selection["global_top_n_requested"],
            "score_fit_scope": "train_only",
            "score_targets": ALL_TARGETS,
            "top_feature_scores": feature_selection["feature_scores"][:50],
        },
        "model_features": feature_selection["model_features"],
        "model_feature_count": int(len(feature_selection["model_features"])),
        "calcium_core_feature_count": int(len(feature_selection["calcium_core"])),
        "global_selected_feature_count": int(len(feature_selection["global_selected"])),
        "metrics": metrics,
        "model_types": model_types,
        "warnings": warnings,
        "assumptions": [
            "Rows are sorted by time and split 80/20 chronologically.",
            "Feature selection scores are fitted on the training split only.",
            "Calcium core features are always included when present.",
            "Model pipelines may impute with training-fitted medians, but feature missing diagnostics are not imputed.",
            "Thresholds for classification targets are selected from training predictions only.",
            "Low and high T90 risks are modeled separately because dose-response is non-monotonic and tradeoff-prone.",
        ],
        "recommended_next_step": recommended_next_step,
    }


def print_summary(report: dict[str, object], doc_appended: bool) -> None:
    print("T90 calcium-control model training complete.")
    print(f"  train rows: {report['train_row_count']}")
    print(f"  test rows: {report['test_row_count']}")
    print(f"  selected feature count: {report['model_feature_count']}")
    print("  test metrics summary:")
    metrics = report["metrics"]
    t90_test = metrics["t90"]["test"]
    print(
        f"    t90: MAE={format_metric(t90_test.get('mae'))}, RMSE={format_metric(t90_test.get('rmse'))}, R2={format_metric(t90_test.get('r2'))}"
    )
    for target in CLASSIFICATION_TARGETS:
        test_metrics = metrics[target]["test"]
        print(
            f"    {target}: AP={format_metric(test_metrics.get('average_precision'))}, "
            f"AUC={format_metric(test_metrics.get('roc_auc'))}, "
            f"F1={format_metric(test_metrics.get('f1'))}, "
            f"positive_rate={format_metric(test_metrics.get('positive_rate'))}"
        )
    print(f"  recommended next step: {report['recommended_next_step']}")
    print(f"  docs appended: {doc_appended}")


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    frame = load_dataset(args.input)
    feature_report = load_json(args.feature_report)
    dose_report = load_json(args.dose_response_report)
    calcium_core, _process_context, all_candidate, leakage_columns = get_feature_groups(feature_report, frame)

    if not all_candidate:
        raise ValueError("No candidate model features remain after leakage exclusions.")

    train, test = time_order_split(frame)
    feature_selection = select_features_train_only(train, calcium_core, all_candidate, args.global_top_n)
    model_features = feature_selection["model_features"]
    if not model_features:
        raise ValueError("No model features were selected.")

    leakage_used = [
        feature
        for feature in model_features
        if is_leakage_column(feature, set(leakage_columns))
    ]
    if leakage_used:
        raise ValueError(f"Leakage columns were selected as model features: {leakage_used}")

    metrics, metric_rows_output, model_types = train_models(train, test, model_features, args.model_dir, warnings)
    metrics_table = pd.DataFrame(metric_rows_output)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(args.metrics_output, index=False, encoding="utf-8-sig")

    feature_selection_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "score_fit_scope": "train_only",
        "calcium_core": feature_selection["calcium_core"],
        "global_selected": feature_selection["global_selected"],
        "model_features": model_features,
        "feature_scores": feature_selection["feature_scores"],
    }
    write_json(args.model_dir / "feature_selection.json", feature_selection_payload)

    report = build_report(
        args=args,
        frame=frame,
        train=train,
        test=test,
        feature_selection=feature_selection,
        metrics=metrics,
        model_types=model_types,
        warnings=warnings,
    )
    write_json(args.report, report)

    model_metadata = {
        "created_at": report["created_at"],
        "model_files": MODEL_FILENAMES,
        "model_types": model_types,
        "model_feature_count": report["model_feature_count"],
        "recommended_next_step": report["recommended_next_step"],
    }
    write_json(args.model_dir / "model_metadata.json", model_metadata)

    doc_appended = append_documentation(args.doc, args, dose_report, report, model_types)
    print_summary(report, doc_appended)


if __name__ == "__main__":
    main()
