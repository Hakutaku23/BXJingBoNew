# AGENTS.md

## Project Objective

This repository contains experiments for halogenation process data. The practical task is not generic T90 prediction. The real goal is:

> Control T90 qualification by controlling calcium stearate consumption, primarily represented by `硬脂酸钙加注量`.

The qualified T90 interval is:

- qualified: `8.20 <= t90 <= 8.70`
- low out-of-spec: `t90 < 8.20`
- high out-of-spec: `t90 > 8.70`
- nominal center: `8.45`

Do not build only a black-box T90 predictor. The next useful stage is a reproducible calcium-focused experiment pipeline that can support offline control-policy evaluation.

## Current Data Context

The cleaned base dataset is expected at:

```text
data/data_clean.parquet
```

This file is produced from cleaned DCS process data and sparse LIMS T90 labels.

Important columns may include:

```text
time
t90
t90_C
t90_D
t90_E
t90_label_count
卤化工段胶液总量2
反应溴添加量
储罐胶浓在线检测
R510A温度
R511A温度
R512A温度
硬脂酸钙加注量
ESBO加注量
中和碱液添加量
R513温度
R514温度
```

The existing experiment record indicates that instantaneous DCS values are not sufficient. Use lagged and windowed features around process-aligned timestamps.

## Process Lag Priors

Use these residence-time priors unless the repository contains a newer validated configuration:

```text
卤化工段胶液总量2: 174 min
反应溴添加量: 174 min
储罐胶浓在线检测: 174 min
R510A温度: 174 min
R511A温度: 173 min
R512A温度: 166 min
硬脂酸钙加注量: 165 min
ESBO加注量: 165 min
中和碱液添加量: 165 min
R513温度: 165 min
R514温度: 164 min
```

For `硬脂酸钙加注量`, the 165-minute lag is the key control-alignment prior.

## Core Experiment Direction

Implement the next stage in this order:

1. Build calcium-focused lag and window features.
2. Analyze calcium dose-response relationship.
3. Train T90 control-oriented models.
4. Simulate offline calcium setpoint policies.
5. Produce reports for deciding whether the policy is safe enough for plant trial.

The core control variables should include both:

```text
absolute calcium feed = 硬脂酸钙加注量
normalized calcium consumption = 硬脂酸钙加注量 / 卤化工段胶液总量2
```

If the denominator is missing, zero, or invalid, output `NaN`. Do not silently fill it during feature construction.

## Required Scripts

Create or update these scripts:

```text
scripts/build_t90_ca_features.py
scripts/analyze_t90_ca_dose_response.py
scripts/train_t90_ca_control_model.py
scripts/simulate_ca_setpoint_policy.py
```

Keep existing scripts intact unless a compatibility fix is required.

## Expected Outputs

Feature generation:

```text
data/t90_ca_feature_dataset.parquet
data/t90_ca_feature_report.json
```

Dose-response analysis:

```text
data/t90_ca_dose_response_bins.csv
data/t90_ca_dose_response_report.json
```

Model training:

```text
models/t90_ca_control/
data/t90_ca_control_metrics.csv
data/t90_ca_control_report.json
```

Policy simulation:

```text
data/t90_ca_policy_simulation.parquet
data/t90_ca_policy_summary.csv
data/t90_ca_policy_report.json
```

Documentation:

```text
docs/t90_calcium_control_experiment.md
```

## Target Definitions

After feature construction, supervised samples should be rows with non-null `t90`.

Create these targets:

```text
y_ok = 1 if 8.20 <= t90 <= 8.70 else 0
y_low = 1 if t90 < 8.20 else 0
y_high = 1 if t90 > 8.70 else 0
y_out_spec = 1 if t90 < 8.20 or t90 > 8.70 else 0
```

For control use, prefer producing all of:

```text
t90_pred
p_ok
p_low
p_high
p_out_spec
```

## Feature Engineering Requirements

For each LIMS sample time, use only historical DCS values relative to that sample time. Do not use future values.

For calcium, create features around `sample_time - 165 min`.

Required calcium features:

```text
ca_lag_165
ca_win_15_mean
ca_win_30_mean
ca_win_60_mean
ca_win_120_mean
ca_win_60_std
ca_win_60_min
ca_win_60_max
ca_win_60_range
ca_win_60_slope
ca_delta_15
ca_delta_30
ca_delta_60
ca_missing_rate_60
ca_per_rubber_flow_lag_165
ca_per_rubber_flow_win_60_mean
ca_per_rubber_flow_win_60_slope
```

For other process variables, build process-aligned lag/window features using the lag priors above. At minimum, compute mean, std, min, max, range, slope, and missing rate for feasible 15/30/60-minute windows.

If a feature cannot be computed due to missing data, output `NaN` and report missing rates.

## Modeling Requirements

Use time-ordered split only:

```text
first 80% by time: train
last 20% by time: test
```

Never use random train/test split for sparse LIMS labels.

Avoid leakage:

- Fit feature selection on training data only.
- Fit imputers, scalers, encoders, thresholds, and calibration on training data only.
- Do not use future DCS values relative to each LIMS sample time.
- Do not use label or target-derived columns as features.

Exclude these columns from model features:

```text
time
t90
t90_C
t90_D
t90_E
t90_label_count
y_ok
y_low
y_high
y_out_spec
```

Also exclude any prediction, threshold, policy, or target-derived columns.

## Model Choices

Prefer simple and robust models first:

- LightGBM if available
- XGBoost if available
- CatBoost if available
- scikit-learn HistGradientBoosting or GradientBoosting fallback
- AutoGluon only if already available and used in the repo

The pipeline must not fail only because AutoGluon is unavailable. Provide a sklearn fallback.

Fallback baseline:

```text
SimpleImputer(strategy="median")
HistGradientBoostingRegressor or GradientBoostingRegressor
HistGradientBoostingClassifier or GradientBoostingClassifier
```

Save sklearn fallback models with `joblib`.

## Metrics

For classification targets, report:

```text
average_precision
roc_auc
brier_score
precision
recall
f1
false_positive_rate
alarm_rate
threshold
positive_rate
```

Select thresholds using training or validation data only, not test data.

For regression target `t90`, report:

```text
mae
rmse
r2
mean_baseline_mae
mean_baseline_rmse
```

For policy simulation, report:

```text
mean_predicted_p_ok_current
mean_predicted_p_ok_recommended
mean_expected_gain
mean_abs_ca_delta_pct
coverage
ood_rate
recommend_increase_rate
recommend_decrease_rate
recommend_hold_rate
```

## Offline Policy Requirements

Policy simulation must be conservative.

Default candidate calcium setpoints:

```text
current * 0.90
current * 0.95
current
current * 1.05
current * 1.10
```

Also support optional quantile-grid candidates from the training dose distribution.

Safety constraints:

- Recommended calcium value must remain within training P5-P95 by default.
- Default max recommendation change should be ±5%.
- Do not recommend a change when current `p_ok >= 0.75`.
- Do not recommend a change if either `p_low` or `p_high` materially worsens.
- Hold when the sample is out of the training distribution.
- Always allow action `hold`.

Default action labels:

```text
hold
increase_ca_small_step
decrease_ca_small_step
```

## Reports

Every script must write a machine-readable JSON report with:

```text
input path
output path
created_at timestamp
row counts
label counts
train/test time ranges if applicable
feature count
missing-rate summary
metrics
warnings
assumptions
```

Use UTF-8 JSON with `ensure_ascii=False`.

Also print a concise console summary.

## Code Style

Use Python 3.10+.

Required style:

- use `argparse`
- use `pathlib.Path`
- use deterministic random seeds
- use functions with clear names
- avoid hidden global state
- avoid hard-coded absolute paths
- handle Chinese column names safely
- use parquet for large tables
- use CSV for small metric tables
- fail with clear error messages when required columns are missing
- keep Windows path compatibility

Do not commit generated large data files or model artifacts unless explicitly requested.

## Validation

Before finishing, run:

```bash
python -m py_compile scripts/build_t90_ca_features.py
python -m py_compile scripts/analyze_t90_ca_dose_response.py
python -m py_compile scripts/train_t90_ca_control_model.py
python -m py_compile scripts/simulate_ca_setpoint_policy.py
```

If data is available, run the full pipeline with default arguments and confirm expected outputs are produced. If data is unavailable, run compile checks only and document that data-dependent execution was skipped.

## Documentation

If no README exists, do not create a large README unless requested.

Create or update:

```text
docs/t90_calcium_control_experiment.md
```

Include:

- objective
- data source
- feature strategy
- dose-response strategy
- model strategy
- policy simulation strategy
- expected outputs
- run commands
- known limitations
- decision gate before online trial

## Important Limitations

Offline counterfactual policy simulation is not causal proof. Do not present it as deployable automatic control.

Before any plant trial, require:

- stable direction in recent monthly validation
- conservative adjustment size
- acceptable false alarm or intervention rate
- no material transfer of risk between low T90 and high T90
- manual review by process engineers
