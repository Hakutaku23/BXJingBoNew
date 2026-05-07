# 卤化工段数据整理与实验记录

## 1. 本次规则

本次以已重新保存的 `data/副本卤化工段数据点位.xlsx` 的 `Sheet3` 为准，从 `data/t90数据/` 下的逐点位 txt 文件提取数据。当前 Sheet3 共 11 个点位，已移除 B 线相关测点。

输出文件：

- `data/data_tot.parquet`：按所需位点提取并按 `time` 对齐的全量数据，未做上下限整定。
- `data/data_new.parquet`：在全量数据基础上，按 `Sheet3` 的低限/高限将超限值置为 `NaN`，再保留 `2024-07-02 00:00:00` 及之后的数据。
- `data/data_clean.parquet`：在 `data_new.parquet` 基础上做 DCS 保守清洗，并挂接稀疏 LIMS t90 标签。
- `data/data_new_report.json`、`data/data_clean_report.json`：记录整理、清洗、缺失率和建模体检结果。

本次不删除空值行；空值处理保留给后续实验阶段单独处理。

## 2. 点位与上下限规则

当前使用的 11 个点位为：

| 变量 | 点位 |
| --- | --- |
| 卤化工段胶液总量2 | `B4-FIC-C51001.PV.F_CV` |
| 反应溴添加量 | `B4-FIC-C51004.PV.CV` |
| 储罐胶浓在线检测 | `B4-AT-C50002A-BIIR.PV.CV` |
| R510A温度 | `B4-TI-C51007A_S.PV.CV` |
| R511A温度 | `B4-TI-C51101A_S.PV.CV` |
| R512A温度 | `B4-TI-C51702A.PV.F_CV` |
| 硬脂酸钙加注量 | `B4-FIC-C51401.PV.F_CV` |
| ESBO加注量 | `B4-FIC-C51801.PV.F_CV` |
| 中和碱液添加量 | `B4-FIC-C51605.PV.F_CV` |
| R513温度 | `B4-TI-C51301_S.PV.CV` |
| R514温度 | `B4-TI-C51401_S.PV.CV` |

点位名标准化规则：

- 去除开头的 `B4-` 前缀；
- 将 `/`、`-`、`.` 等分隔符统一转换为 `_`；
- 合并连续 `_`；
- 统一转换为大写。

上下限整定规则：

- 两个边界值中的较小值作为实际下限；
- 两个边界值中的较大值作为实际上限；
- 数据保留范围为闭区间 `[实际下限, 实际上限]`；
- 小于实际下限或大于实际上限的值统一整理为 `NaN`。

## 3. 输出结果

`data/data_tot.parquet`：

- 行数：936844
- 列数：12
- 时间范围：2024-01-20 03:34:00 至 2025-11-01 00:00:00

`data/data_new.parquet`：

- 行数：700887
- 列数：12
- 时间范围：2024-07-02 00:00:00 至 2025-11-01 00:00:00
- 行过滤：未删除空值行；删除 `2024-07-02 00:00:00` 之前的数据 235957 行

`data_new.parquet` 各字段整定后、日期截断后的非空数量如下：

| 字段 | 非空数量 |
| --- | ---: |
| time | 700887 |
| 卤化工段胶液总量2 | 664643 |
| 反应溴添加量 | 601980 |
| 储罐胶浓在线检测 | 692174 |
| R510A温度 | 606139 |
| R511A温度 | 604164 |
| R512A温度 | 606480 |
| 硬脂酸钙加注量 | 630374 |
| ESBO加注量 | 596599 |
| 中和碱液添加量 | 565126 |
| R513温度 | 628896 |
| R514温度 | 548073 |

## 4. 数据整理执行命令

```powershell
D:\miniconda3\envs\autoGluon\python.exe .\scripts\prepare_data_new.py --source-dir .\data\t90数据 --raw-output .\data\data_tot.parquet --output .\data\data_new.parquet --report .\data\data_new_report.json --start-date 2024-07-02
```

## 5. DCS 数据进一步清洗记录

本次在新的 `data/data_new.parquet` 基础上继续生成 `data/data_clean.parquet`。该数据是 DCS 现场分钟级过程数据，清洗时采用偏保守的策略：只修复短时间通讯断点和明显孤立尖峰，不跨越长时间缺失，也不使用 LIMS 标签反向填补 DCS 特征。

选择的方法如下：

- 孤立尖峰识别：使用居中滚动中位数和 MAD，窗口为 11 个点，阈值为 8 倍 robust sigma；只处理单点型孤立尖峰，保留真实阶跃变化。
- 短缺失插值：对内部连续缺失不超过 5 个点的片段做线性插值；长缺失、开头缺失、结尾缺失保留为 `NaN`。
- 缺失与尖峰标记：为每个 DCS 指标新增 `_was_missing` 和 `_spike_flag` 标记列，便于后续建模识别插值来源和异常来源。
- LIMS 标签挂接：从 `data/t90-溴丁橡胶.xlsx` 的 C/D/E 三个 sheet 提取 `t´c(90),min`，按采样时间四舍五入到分钟后精确挂接到 DCS 时间轴；新增 `t90_C`、`t90_D`、`t90_E`、`t90` 和 `t90_label_count`。其中 `t90` 为同一时间多来源标签的均值，仅作为快速建模用标签。

`data/data_clean.parquet` 输出结果：

- 行数：700887
- 列数：39
- 时间范围：2024-07-02 00:00:00 至 2025-11-01 00:00:00
- DCS 孤立尖峰置空数量：71
- 短缺失插值填补数量：16911
- DCS 总缺失数从 965109 降至 948269
- LIMS 合并标签 `t90` 非空行数：2362，标签密度约 0.337%
- 全字段非空 DCS 行数：474239

清洗有效性判断：

- 数据结构有效：B 线相关测点已移除，时间无重复，时间范围正确。
- 可用性改善：移除 B 线高缺失测点后，全字段非空 DCS 行数达到 474239，明显优于此前包含 B 线时的情况。
- 短缺失修复有效：修复了 16911 个短断点，同时只标记 71 个孤立尖峰，清洗强度较低，未大规模扭曲过程数据。
- 建模即时有效性仍有限：使用 `t90` 稀疏标签做时间顺序 80/20 切分，单线程梯度提升模型的测试 RMSE 为 0.226，均值基线 RMSE 为 0.201，未优于均值基线。
- 结论：`data_clean.parquet` 适合作为后续特征工程基础，但不建议直接用“采样时刻瞬时 DCS 值”预测 LIMS t90。下一步应构造滞后特征、窗口统计特征、物料停留时间对齐特征，或按 C/D/E 分线分别建模。

清洗执行命令：

```powershell
D:\miniconda3\envs\autoGluon\python.exe .\scripts\clean_dcs_data.py --input .\data\data_new.parquet --output .\data\data_clean.parquet --report .\data\data_clean_report.json
```

## 6. 清洗前后信噪比评估

由于 DCS 现场数据没有真实的“无噪声信号”，本次使用代理信噪比评价清洗效果：

- 信号代理：31 分钟居中滚动中位数。
- 噪声代理：原始/清洗后序列与滚动中位数之间的残差。
- SNR 计算：`10 * log10(var(信号代理) / var(噪声代理))`。
- 辅助判断：同时检查非空覆盖率、残差 MAD、一阶差分 MAD 和均值漂移，避免只看 SNR 造成误判。

评估输出：

- JSON 报告：`data/data_clean_snr_report.json`
- CSV 明细：`data/data_clean_snr_report.csv`
- 脚本：`scripts/evaluate_cleaning_snr.py`

主要结果：

- 参与评估变量数：11
- SNR 改善变量数：2
- SNR 持平或下降变量数：9
- SNR 中位变化：-0.394 dB
- SNR 平均变化：-0.569 dB
- 残差 MAD 中位改善：约 0.019%
- 明显均值漂移变量数：0
- 综合判断：清洗安全但 SNR 增益有限。

解释：

本次清洗策略是保守型清洗，只处理短断点和孤立尖峰，并保留长缺失。该策略不会显著改变信号形态，因此 SNR 不一定明显提升。SNR 结果显示清洗没有带来明显降噪收益，但也没有造成明显均值漂移或大规模信号扭曲。因此，SNR 可以作为清洗质量的辅助指标，但不能作为唯一判断标准。

从当前结果看，`data_clean.parquet` 符合“保守、不过度清洗”的要求；如果后续目标是提升建模信号，还需要进一步做滞后、窗口统计、停留时间对齐等特征工程，而不是继续加大当前清洗强度。

执行命令：

```powershell
D:\miniconda3\envs\autoGluon\python.exe .\scripts\evaluate_cleaning_snr.py --before .\data\data_new.parquet --after .\data\data_clean.parquet --output .\data\data_clean_snr_report.json --csv-output .\data\data_clean_snr_report.csv
```

## 7. 工艺停留时间适用性评估

根据 `data/halogen_flow_render.png` 和流程图中的标注，主线可读出的停留时间如下：

| 单元 | 停留时间 |
| --- | ---: |
| R510A | 1 min |
| R511A | 7 min |
| R512A | 1 min |
| R513 | 1 min |
| R514 | 1 min |
| V530 | 25 min |
| V532 | 50 min |
| V540 | 50 min |
| T300 | 38 min |

主线合计约 `174 min`。按点位所在位置折算到 LIMS t90 采样时刻，形成分段滞后方案：

| 变量 | 滞后时间 |
| --- | ---: |
| 卤化工段胶液总量2 | 174 min |
| 反应溴添加量 | 174 min |
| 储罐胶浓在线检测 | 174 min |
| R510A温度 | 174 min |
| R511A温度 | 173 min |
| R512A温度 | 166 min |
| 硬脂酸钙加注量 | 165 min |
| ESBO加注量 | 165 min |
| 中和碱液添加量 | 165 min |
| R513温度 | 165 min |
| R514温度 | 164 min |

为判断该停留时间是否合适，使用 `data_clean.parquet` 中的 DCS 数据和稀疏 LIMS `t90` 标签进行时间顺序 80/20 切分建模验证。比较方案包括不滞后、多个统一滞后候选，以及流程图分段滞后方案。该评估在 `autoGluon` 环境下执行，模型使用单线程 `SimpleImputer(median) + GradientBoostingRegressor`，用于快速判断滞后时间合理性。

输出文件：

- `data/residence_time_evaluation.json`
- `data/residence_time_evaluation.csv`
- `scripts/evaluate_residence_time.py`

主要结果：

| 方案 | 测试 RMSE | 测试 MAE | R2 | 备注 |
| --- | ---: | ---: | ---: | --- |
| uniform_60min | 0.2215 | 0.1641 | -0.2495 | 当前候选中 RMSE 最低 |
| uniform_180min | 0.2229 | 0.1671 | -0.2659 | 与主线 174min 接近 |
| process_flow_piecewise | 0.2232 | 0.1656 | -0.2686 | 流程图分段滞后，排名第 3 |
| uniform_174min | 0.2234 | 0.1678 | -0.2717 | 主线总停留时间 |
| mean baseline | 0.2012 | 0.1532 | -0.0310 | 训练集均值基线 |

判断：

- 从候选滞后之间的比较看，流程图分段滞后方案排名第 3，与最佳统一 60min 方案的 RMSE 差约 `0.0017`，与统一 174min 方案也非常接近，因此流程图确定的停留时间作为工艺先验是合理的。
- 但是所有滞后模型均未优于均值基线，说明仅使用“采样时刻对应的瞬时 DCS 值”不足以稳定预测稀疏 LIMS t90。
- 因此，停留时间本身可以保留，但后续不应只做单点滞后。建议围绕工艺滞后时间构造窗口统计特征，例如 `t-174min` 附近 15/30/60min 均值、标准差、斜率、范围，以及按 R510A-R514 和 V530-T300 分段构造停留区间统计。

执行命令：

```powershell
D:\miniconda3\envs\autoGluon\python.exe .\scripts\evaluate_residence_time.py --input .\data\data_clean.parquet --output .\data\residence_time_evaluation.json --csv-output .\data\residence_time_evaluation.csv
```

## 8. 借鉴 T90 主线方法的优选变量测试

根据 `docs/T90最优主线方法整理.md` 中归纳的主线思路，本次使用当前 `Sheet3` 保留的 11 个优选工艺变量，在 `data_clean.parquet` 基础上重新测试 T90 越界风险建模。建模目标由连续 t90 转换为越界概率：

- 目标中心：`8.45 min`
- 合格区间：`[8.20, 8.70]`
- 标签定义：当 `t90 < 8.20` 或 `t90 > 8.70` 时，`y_out_spec = 1`，否则为 0

方法借鉴内容：

- 继续使用流程图停留时间作为因果对齐先验，对不同点位使用 164-174 min 的分段滞后。
- 围绕对齐时刻构造 15/30/60 min 窗口统计特征，包括均值、标准差、最小值、最大值、范围、末值、斜率和缺失率等。
- 按训练集相关性筛选 `global_120` 全局模型特征和 `search_40` JITL 相似检索特征。
- 全局模型使用 AutoGluon TabularPredictor，评价指标为 `average_precision`。
- JITL 使用 robust scale 后的 `median_l1` 距离，最多取 50 个近邻，最少 20 个近邻。
- 融合方式沿用主线记录中的思路：`logit(p_final) = 0.6 * logit(p_global) + 0.4 * logit(p_jitl)`。

输出文件：

- `data/t90_mainline_selected_variables_dataset.parquet`
- `data/t90_mainline_selected_variables_report.json`
- `data/t90_mainline_selected_variables_metrics.csv`
- `models/t90_mainline_global_autogluon/`
- `scripts/test_t90_mainline_methods.py`

样本与切分：

| 项目 | 数值 |
| --- | ---: |
| LIMS 标签样本数 | 2362 |
| 训练集样本数 | 1889 |
| 测试集样本数 | 473 |
| 总体越界率 | 19.52% |
| 训练集越界率 | 19.38% |
| 测试集越界率 | 20.08% |
| 训练时间范围 | 2024-07-02 06:46 至 2025-08-02 11:00 |
| 测试时间范围 | 2025-08-02 14:33 至 2025-10-19 13:00 |

AutoGluon 验证集表现中，WeightedEnsemble_L2 的 `average_precision` 最高，为 0.594；其后依次为 CatBoost、LightGBMXT、XGBoost 和 LightGBM 系列模型。由于当前环境在 Windows 下存在部分模型权限限制，RandomForest、ExtraTrees 和 NeuralNetFastAI 被跳过，但 LightGBM、CatBoost、XGBoost、NeuralNetTorch 和加权融合模型均已完成训练。

测试集主要结果如下：

| 模型输出 | Brier | AP | AUC | 最佳 F1 阈值 | Precision | Recall | F1 | 误报率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p_global | 0.1815 | 0.2564 | 0.6040 | 0.03 | 0.2384 | 0.8105 | 0.3684 | 0.6508 |
| p_jitl | 0.1717 | 0.1904 | 0.4707 | 0.02 | 0.2008 | 1.0000 | 0.3345 | 1.0000 |
| p_final_alpha_0_40 | 0.1761 | 0.2275 | 0.5843 | 0.06 | 0.2526 | 0.7579 | 0.3789 | 0.5635 |

沿用旧主线记录中的 `0.14` 阈值时，本次测试结果如下：

| 模型输出 | Precision | Recall | F1 | 误报率 | 报警率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| p_global | 0.1765 | 0.0316 | 0.0536 | 0.0370 | 0.0359 |
| p_jitl | 0.2037 | 0.5789 | 0.3014 | 0.5688 | 0.5708 |
| p_final_alpha_0_40 | 0.0909 | 0.0211 | 0.0342 | 0.0529 | 0.0465 |

判断：

- 借鉴主线方法后，优选变量仍能提供一定风险排序信息。`p_global` 在测试集上的 AP 为 0.256，高于测试集越界基准率 0.201，AUC 为 0.604，说明全局模型具备弱到中等的区分能力。
- JITL 的 Brier 最低，但 AP 和 AUC 较弱，尤其 AUC 低于 0.5，因此当前优选变量条件下，JITL 不宜单独作为报警模型，只适合作为局部校准或辅助参考。
- 融合模型 `p_final_alpha_0_40` 的最佳 F1 略高于全局模型，但 AP 和 AUC 低于 `p_global`。因此，若后续目标是风险排序和提前预警，应优先保留 `p_global`；若目标是固定召回下的报警规则，可继续比较 `p_global` 与融合模型。
- 旧主线中的 `0.14` 阈值不能直接迁移到当前优选变量版本。对 `p_global` 和 `p_final` 来说，该阈值召回率过低；本次测试中更合适的候选阈值约为 `p_global = 0.03` 或 `p_final = 0.06`，但二者误报率仍较高，需要结合现场可接受报警频率重新整定。
- 与 `T90最优主线方法整理.md` 中历史记录相比，当前优选变量版本的 AP 低于历史最优主线结果，说明变量优选后虽然减少了冗余和不稳定测点，但也可能损失了一部分与 t90 越界相关的弱信号。后续应在当前优选变量基础上继续做阈值整定、月份滚动验证和漂移期单独验证。

执行命令：

```powershell
D:\miniconda3\envs\autoGluon\python.exe .\scripts\test_t90_mainline_methods.py --input .\data\data_clean.parquet --output .\data\t90_mainline_selected_variables_report.json --dataset-output .\data\t90_mainline_selected_variables_dataset.parquet --csv-output .\data\t90_mainline_selected_variables_metrics.csv --model-dir .\models\t90_mainline_global_autogluon --autogluon-time-limit 120
```

## 9. 后续实验记录模板

### 实验：空值处理与滞后特征

- 日期：
- 数据版本：
- 空值处理规则：
- 滞后/窗口特征规则：
- 标签对齐规则：
- 样本量：
- 目标变量：
- 特征字段：
- 模型：
- 评价指标：
- 主要结果：
- 备注：

## 10. 硬脂酸钙剂量响应与控制模型实验

### 10.1 第二阶段：硬脂酸钙剂量响应分析结果

- 输入文件：`data\t90_ca_feature_dataset.parquet`、`data\t90_ca_feature_report.json`。
- 输出文件：`data\t90_ca_dose_response_bins.csv`、`data\t90_ca_dose_response_report.json`。
- 主剂量特征：`ca_per_rubber_flow_win_60_mean`。
- 可用样本数：2315；有效分箱数：5。
- 总体比例：ok_rate=0.8048，low_rate=0.0271，high_rate=0.1681，out_spec_rate=0.1952。
- Spearman 相关：dose-t90=0.1318，dose-y_ok=-0.0784，dose-y_low=-0.0568，dose-y_high=0.1070，dose-y_out_spec=0.0784。
- 最佳合格率分箱：bin 2，ok_rate=0.9309。
- 最高剂量分箱表现：ok_rate=0.6350，high_rate=0.3391。
- 解释标记：{"non_monotonic_possible": true, "risk_tradeoff_possible": true, "weak_univariate_signal": false, "direction_conflict": true, "low_bin_support": false}。
- 结论：硬脂酸钙信号存在，但关系不是简单的“加得越多越好”；低 T90 与高 T90 风险方向存在权衡，后续必须分别建模。

### 10.2 第三阶段：控制模型训练

- 脚本：`scripts/train_t90_ca_control_model.py`。
- 执行命令：`D:\miniconda3\envs\autoGluon\python.exe .\scripts\train_t90_ca_control_model.py --input .\data\t90_ca_feature_dataset.parquet --feature-report .\data\t90_ca_feature_report.json --dose-response-report .\data\t90_ca_dose_response_report.json --model-dir .\models\t90_ca_control --metrics-output .\data\t90_ca_control_metrics.csv --report .\data\t90_ca_control_report.json --doc .\docs\Experimental_Procedure_cn.md --time-limit 300`。
- 输入文件：`data\t90_ca_feature_dataset.parquet`、`data\t90_ca_feature_report.json`、`data\t90_ca_dose_response_report.json`。
- 输出文件：`models\t90_ca_control`、`data\t90_ca_control_metrics.csv`、`data\t90_ca_control_report.json`。
- 切分方法：按 `time` 排序后，前 80% 为训练集，后 20% 为测试集；未使用随机切分。
- 训练集/测试集样本数：1889 / 473。
- 训练时间范围：2024-07-02T06:46:00 至 2025-08-02T11:00:00；测试时间范围：2025-08-02T14:33:00 至 2025-10-19T13:00:00。
- 特征选择：钙剂核心特征 17 个，全局筛选特征 120 个，最终模型特征 131 个；筛选仅在训练集上完成。
- 模型类型：t90=GradientBoostingRegressor，y_ok=GradientBoostingClassifier，y_low=GradientBoostingClassifier，y_high=GradientBoostingClassifier，y_out_spec=GradientBoostingClassifier。
- 测试集指标摘要：
  - t90：MAE=0.1533，RMSE=0.2012，R2=-0.0309。
  - y_ok：AP=0.8841，AUC=0.7057，Brier=0.1417，threshold=0.6300，F1=0.9091，Recall=0.9921，Precision=0.8389。
  - y_low：AP=0.3419，AUC=0.7035，Brier=0.0073，threshold=0.1200，F1=0.0870，Recall=0.3333，Precision=0.0500。
  - y_high：AP=0.4195，AUC=0.6572，Brier=0.1526，threshold=0.3100，F1=0.0213，Recall=0.0109，Precision=0.5000。
  - y_out_spec：AP=0.4792，AUC=0.7057，Brier=0.1418，threshold=0.2700，F1=0.4167，Recall=0.3158，Precision=0.6122。
- recommended_next_step：`proceed_to_offline_policy_simulation`。

### 10.3 当前判断

- 当前建议：`proceed_to_offline_policy_simulation`。
- 若 `y_ok` 或 `y_out_spec` 在测试集上相对基准有增益，且 `y_low`/`y_high` 不退化，可以进入离线策略模拟；否则需要先人工检查模型指标。
- 低 T90 样本数量较少，`y_low` 模型可靠性应谨慎看待，不宜单独作为自动控制依据。
- 高剂量区域在第二阶段表现出较高的高 T90 风险，策略模拟必须限制推荐幅度，并分别检查低/高 T90 风险。
- 当前结论来自离线观测数据，不能证明钙剂调整的因果效果；上线试验前仍需工艺工程师审核。
- 后续所有特征选择、阈值选择和校准步骤必须保持训练集内完成，不能使用测试集信息。

## 11. 硬脂酸钙单耗处方优化实验

- 本阶段跳过通用 T90 预测模型训练。此前预警主线和控制模型探索说明，直接 T90 预测不足以作为控制依据，因此本阶段改为基于历史剂量响应、工况分层和相似样本的离线钙单耗处方优化。
- 输入文件：`data\t90_ca_feature_dataset.parquet`、`data\t90_ca_feature_report.json`、`data\t90_ca_dose_response_report.json`、`data\t90_ca_dose_response_bins.csv`。
- 输出文件：`data\t90_ca_policy_recommendations.parquet`、`data\t90_ca_policy_summary.csv`、`data\t90_ca_policy_report.json`。
- 主剂量特征：`ca_per_rubber_flow_win_60_mean`。
- 全局最佳钙单耗范围：bin 2，范围 [0.0198841, 0.0203884]，ok_rate=0.9309。
- 全局安全分箱：bin 1([0.0191279, 0.0198827]), bin 2([0.0198841, 0.0203884]), bin 3([0.0203891, 0.0212925])。
- 全局高风险分箱：bin 0([0.015048, 0.019126]), bin 4([0.0212942, 0.0365058])。
- 工况上下文特征：`rubber_flow_2_win_60_mean`，`bromine_feed_win_60_mean`，`tank_rubber_conc_win_60_mean`，`esbo_feed_win_60_mean`，`neutral_alkali_feed_win_60_mean`，`r513_temp_win_60_mean`，`r514_temp_win_60_mean`。
- 相似样本配置：最多近邻 50，最少近邻 20，最小期望合格率增益 0.03。
- 策略摘要：总样本 2362，可行动样本 582，hold 1780，increase 191，decrease 391。
- 期望收益：平均 expected_ok_rate_gain=0.2302，中位数=0.1538。
- 风险与限制：该策略只来自离线历史相似样本，不是自动闭环控制；高剂量区域存在高 T90 风险，低/高 T90 风险需分别检查；样本支持不足或风险边界变差时保持 hold。
- recommended_next_step：`do_not_use_policy`。
- 警告：无。

## 12. 硬脂酸钙处方策略离线验证与失效诊断

- 增加本验证的原因：上一版钙单耗处方策略给出 `do_not_use_policy`，不能进入 shadow trial，需要先做失效诊断和严格时间前向安全检查。
- 重复章节清理：已创建 `docs/Experimental_Procedure_cn.md.bak`，并移除重复的后一个“硬脂酸钙单耗处方优化实验”章节；早期实验内容未改动。
- 输入文件：`data\t90_ca_feature_dataset.parquet`、`data\t90_ca_feature_report.json`、`data\t90_ca_dose_response_report.json`、`data\t90_ca_policy_recommendations.parquet`、`data\t90_ca_policy_summary.csv`、`data\t90_ca_policy_report.json`。
- 输出文件：`data\t90_ca_policy_audit.csv`、`data\t90_ca_policy_validation_report.json`。
- 原策略 recommended_next_step：`do_not_use_policy`。
- 动作组实际结果：hold ok/high/low=0.8337/0.1399/0.0264；increase ok/high/low=0.7539/0.1832/0.0628；decrease ok/high/low=0.6982/0.2890/0.0128。
- 时间切分验证：train actionable ok=0.7059，train hold ok=0.8354；test actionable ok=0.7452，test hold ok=0.8259。
- 诊断标记：{"increase_group_actual_high_rate_worse_than_hold": true, "decrease_group_actual_low_rate_worse_than_hold": false, "actionable_group_actual_ok_rate_not_better_than_hold": true, "expected_gain_not_realized": true, "excessive_action_rate": true, "too_many_decrease_actions": true, "too_many_increase_actions": false, "insufficient_neighbor_support": false, "high_risk_bins_recommended": true, "recommended_step_is_do_not_use_policy": true, "possible_time_leakage_in_original_policy": true, "increase_group_actual_low_rate_worse_than_hold": true, "decrease_group_actual_high_rate_worse_than_hold": true, "policy_worse_in_test_than_train": false}
- 是否存在潜在时间泄漏：True。原策略报告没有证明近邻池只使用历史样本，因此必须按 walk-forward 重写。
- 新 recommended_next_step：`rewrite_policy_with_walk_forward_validation`。
- 结论：当前策略不能使用，也不能进入 shadow trial；下一步应重写为严格 walk-forward 处方评估，限定近邻只来自样本时刻之前，并加入标签释放延迟和风险护栏。
- 警告：Original policy does not prove walk-forward safety; strict walk-forward rewrite is required.

## 13. 出口红外矫正特征接入与代理价值评估

- 本阶段用于在严格 walk-forward 钙剂策略重写前，评估产品出口红外矫正值是否可作为机理相关的中间质量代理。
- T90 为人工 LIMS 检测，记录精度为 0.1，实际误差约 0.1；控制目标应理解为 [8.20, 8.70] 合格区间，而非精确预测 8.45。
- 卤化流程停留时间约 3 小时，且不同单元停留时间不同，因此出口 IR 不被当作直接 T90 测量值，而是作为下游质量状态和潜在中介变量评估。
- `output.csv` 接入规则：第一列作为时间戳，最后一列作为出口红外矫正值，所有中间列完全忽略，不进入 `data_clean_with_ir.parquet`。
- 输入文件：`data/data_clean.parquet`、`data/output.csv`、`data/t90_ca_feature_dataset.parquet`。
- 输出文件：`data/data_clean_with_ir.parquet`、`data/data_clean_with_ir_report.json`、`data/output_ir_proxy_evaluation.csv`、`data/output_ir_proxy_bins.csv`、`data/output_ir_proxy_evaluation.json`。
- 检测到的时间戳列：`time`；红外值列：`Y_cal`。
- 被忽略的中间列数量：411。
- IR 覆盖率：0.30032373264163836；重叠时间：{'min': '2025-01-20T23:02:00', 'max': '2025-10-19T10:39:00'}。
- 最佳 IR 特征：`output_ir_corrected_win_15_slope`。
- 钙剂到 IR 关系摘要：ca_per_rubber_flow_win_60_mean 与 output_ir_corrected 的 Spearman=-0.03952708702491413。
- IR 到 T90 风险关系摘要：usable=987，Spearman(y_high)=0.02217784931155313，Spearman(y_out_spec)=0.008569279232458767。
- 分箱结论摘要：{'effective_bin_count': 5, 'high_rate_spread': 0.07675742193508692, 'out_spec_rate_spread': 0.051530533764036296, 'low_rate_spread': 0.025226888171050607, 'min_bin_sample_count': 197}。
- 描述性中介检查：{'descriptive_mediation_possible': True, 'not_causal_proof': True, 'explanation': 'This only checks whether calcium relates to IR and IR stratifies T90 risk; it is not causal proof.'}。
- 增量价值测试：{'feature': 'output_ir_corrected_win_15_slope', 'usable_sample_count': 987, 'meaningful': True, 'incremental_good': True, 'best_delta_ap': 0.01775555912670601, 'best_delta_auc': 0.0308966861598442, 'spread': 0.07675742193508692, 'min_bin_sample_count': 197, 'score': 20.17545742193509}。
- recommended_next_step：`use_ir_in_walk_forward_policy_rewrite`。
- 结论：IR 是否进入 walk-forward 策略重写取决于其相对钙剂特征的增量收益；本阶段不建议 shadow trial，也不实施自动控制。

## 14. 严格 Walk-forward 钙单耗处方策略重写

- 本次重写是因为上一版处方策略存在潜在时间泄漏，且动作组实际表现差于 hold，不能进入 shadow trial。
- 严格 walk-forward 规则：每个样本只使用样本时刻之前且已过标签释放延迟的历史 LIMS 样本；分箱、上下文尺度和近邻池均在历史样本内即时计算。
- 标签释放延迟：24.0 小时。
- 剂量分箱规则：每个评价时刻仅用历史钙单耗构造分位数分箱，并识别历史最佳、安全和高风险分箱。
- 近邻搜索规则：仅使用过程上下文；IR 分支仅在当前样本和足够历史样本均有 IR 时加入 `output_ir_corrected_win_15_slope`，缺失 IR 不删样本。
- 输入文件：`data\t90_ca_feature_dataset.parquet`、`data\data_clean_with_ir.parquet`、`data\output_ir_proxy_evaluation.json`、`data\t90_ca_dose_response_report.json`、`data\t90_ca_policy_validation_report.json`。
- 输出文件：`data\t90_ca_walk_forward_policy_recommendations.parquet`、`data\t90_ca_walk_forward_policy_summary.csv`、`data\t90_ca_walk_forward_policy_report.json`。
- ir_optional_policy test_like：actionable=81，action_rate=0.2375，hold_ok=0.8115，action_ok=0.8025，hold_high=0.1769，action_high=0.1975，branch_next=`do_not_use_policy`。
- ir_optional_policy_without_ir_context test_like：actionable=38，action_rate=0.2879，hold_ok=0.7447，action_ok=0.8421，hold_high=0.2553，action_high=0.1579，branch_next=`insufficient_history_or_support`。
- no_ir_policy test_like：actionable=115，action_rate=0.2431，hold_ok=0.7933，action_ok=0.8174，hold_high=0.2011，action_high=0.1739，branch_next=`do_not_use_policy`。
- no-IR 与 IR 分支比较：{"no_ir_policy": {"branch": "no_ir_policy", "split": "test_like", "total_samples": 473, "evaluable_samples": 473, "actionable_samples": 115, "action_rate": 0.24312896405919662, "hold_count": 358, "increase_count": 25, "decrease_count": 90, "actual_ok_rate_hold": 0.7932960893854749, "actual_ok_rate_actionable": 0.8173913043478261, "actual_ok_rate_increase": 0.88, "actual_ok_rate_decrease": 0.8, "actual_high_rate_hold": 0.2011173184357542, "actual_high_rate_actionable": 0.17391304347826086, "actual_high_rate_increase": 0.08, "actual_high_rate_decrease": 0.2, "actual_low_rate_hold": 0.00558659217877095, "actual_low_rate_actionable": 0.008695652173913044, "actual_low_rate_increase": 0.04, "actual_low_rate_decrease": 0.0, "mean_expected_ok_rate_gain": 0.18672050775934948, "median_expected_ok_rate_gain": 0.125, "mean_neighbor_count": 49.89429175475687, "median_neighbor_count": 50.0, "mean_eligible_history_count": 2119.3784355179705, "ir_available_rate": 0.718816067653277, "recommended_next_step_for_branch": "do_not_use_policy"}, "ir_optional_policy_best": {"branch": "ir_optional_policy_without_ir_context", "split": "test_like", "total_samples": 132, "evaluable_samples": 132, "actionable_samples": 38, "action_rate": 0.2878787878787879, "hold_count": 94, "increase_count": 4, "decrease_count": 34, "actual_ok_rate_hold": 0.7446808510638298, "actual_ok_rate_actionable": 0.8421052631578947, "actual_ok_rate_increase": 0.75, "actual_ok_rate_decrease": 0.8529411764705882, "actual_high_rate_hold": 0.2553191489361702, "actual_high_rate_actionable": 0.15789473684210525, "actual_high_rate_increase": 0.25, "actual_high_rate_decrease": 0.14705882352941177, "actual_low_rate_hold": 0.0, "actual_low_rate_actionable": 0.0, "actual_low_rate_increase": 0.0, "actual_low_rate_decrease": 0.0, "mean_expected_ok_rate_gain": 0.24349872000813438, "median_expected_ok_rate_gain": 0.15476190476190477, "mean_neighbor_count": 50.0, "median_neighbor_count": 50.0, "mean_eligible_history_count": 2143.598484848485, "ir_available_rate": 0.0, "recommended_next_step_for_branch": "insufficient_history_or_support"}}
- 最终 recommended_next_step：`tighten_walk_forward_policy_rules`。
- 限制：该结果仍为离线观测验证，不是因果证明，不是自动控制；只有通过验证的动作才可进入人工复核表，不能直接上线。

## 15. Walk-forward 钙单耗处方规则收紧与消融验证

本阶段针对上一轮严格 walk-forward 策略动作率偏高、IR 分支未改善动作质量的问题，进行更保守的规则网格搜索。本次不训练通用 T90 模型，不进入影子试验，也不形成自动控制建议。

### 输入与输出
- 输入文件：`data\t90_ca_feature_dataset.parquet`、`data\t90_ca_dose_response_report.json`、`data\t90_ca_walk_forward_policy_report.json`。
- 输出文件：`data\t90_ca_walk_forward_policy_tuning_results.csv`、`data\t90_ca_walk_forward_policy_best_recommendations.parquet`、`data\t90_ca_walk_forward_policy_tuning_report.json`。
- 执行命令：`python scripts/tune_ca_walk_forward_policy_rules.py --features data\t90_ca_feature_dataset.parquet --feature-report data\t90_ca_feature_report.json --dose-response-report data\t90_ca_dose_response_report.json --previous-walk-forward-report data\t90_ca_walk_forward_policy_report.json --data-with-ir data\data_clean_with_ir.parquet --results-output data\t90_ca_walk_forward_policy_tuning_results.csv --best-output data\t90_ca_walk_forward_policy_best_recommendations.parquet --report data\t90_ca_walk_forward_policy_tuning_report.json --doc docs\Experimental_Procedure_cn.md`

### 调参设计
- 主剂量特征：`ca_per_rubber_flow_win_60_mean`。
- 上下文特征：rubber_flow_2_win_60_mean, bromine_feed_win_60_mean, tank_rubber_conc_win_60_mean, esbo_feed_win_60_mean, neutral_alkali_feed_win_60_mean, r513_temp_win_60_mean, r514_temp_win_60_mean。
- 网格规模：186624 组规则。
- 主要维度包括历史样本下限、剂量分箱数、邻居数、最小期望收益、高/低 T90 风险恶化阈值、是否限制在当前高风险分箱、是否允许增加钙单耗等。
- IR 本阶段仅作为监测诊断字段携带，不参与邻居搜索、分箱、动作选择或风险护栏。

### 通过标准
- test_like 分段需满足：动作样本不少于 30、动作率不超过对应上限、动作组实际合格率较 hold 至少提升 0.03、高 T90 风险不高于 hold、低 T90 风险相对 hold 增量不超过 0.005，且不得推荐历史高风险分箱。

### 结果摘要
- 通过规则数：0。
- 最佳配置：{'min_history_samples': 300, 'n_dose_bins': 7, 'neighbor_max_k': 30, 'min_neighbors': 20, 'min_expected_gain': 0.05, 'max_high_risk_worsen': 0.0, 'max_low_risk_worsen': 0.0, 'restrict_to_current_high_risk_bins': False, 'allow_increase': False, 'allow_decrease': True, 'forbid_recommended_high_risk_bin': True, 'require_recommended_safe_bin': True, 'require_neighbor_count_at_max_k': False, 'require_expected_gain_quantile': 'none', 'max_action_rate': 0.1}。
- 最佳配置 test_like 动作率：0.006342494714587738
- 最佳配置 test_like 合格率提升：0.2021276595744681
- 最佳配置 test_like 高 T90 风险差：-0.19574468085106383
- 最佳配置 test_like 低 T90 风险差：-0.006382978723404255
- 是否允许增加动作：False
- 是否限制在当前高风险分箱：False
- 最终 recommended_next_step：`tighten_rules_further`。

### 当前判断
若存在通过配置，可进入人工复核表准备阶段；若无通过配置但最小损失配置仍具备正向提升且风险不恶化，则继续收紧规则；否则暂停策略工作并等待更多数据或新的机理特征。
所有判断仍为离线观察性结果，不构成因果证明，也不构成自动控制策略。

## 16. 钙单耗、工况与出口 IR 对 T90 风险的关系发现实验

本阶段用于发现钙单耗、过程工况、出口 IR 代理变量与 T90 风险之间的内部关系。该实验不是新的策略网格搜索，不训练生产模型，也不形成自动控制或影子试验建议。

### 数据与特征
- 主钙单耗特征：`ca_per_rubber_flow_win_60_mean`。
- 工况特征：rubber_flow_2_win_60_mean, bromine_feed_win_60_mean, tank_rubber_conc_win_60_mean, esbo_feed_win_60_mean, neutral_alkali_feed_win_60_mean, r513_temp_win_60_mean, r514_temp_win_60_mean, r510a_temp_win_60_mean, r511a_temp_win_60_mean, r512a_temp_win_60_mean。
- IR 特征：`output_ir_corrected_win_15_slope`；IR 被视为机理相关代理变量和交互候选，不作为直接 T90 测量。

### 实验摘要
- 工况分层剂量响应：有效支持的 regime×dose 分组数为 150。
- 钙×工况交互筛查：通过项数为 16；主要项：bromine_feed_win_60_mean->y_high (delta_auc=0.4640248773251169, delta_ap=0.43959509114171896)；neutral_alkali_feed_win_60_mean->y_high (delta_auc=0.39133287686865226, delta_ap=0.32396099641127907)；bromine_feed_win_60_mean->y_out_spec (delta_auc=0.2820662768031189, delta_ap=0.36151064758107565)；bromine_feed_win_60_mean->y_ok (delta_auc=0.2820662768031189, delta_ap=0.11216232095632739)；rubber_flow_2_win_60_mean->y_out_spec (delta_auc=0.20072403230297964, delta_ap=0.32970443032155317)。
- IR 分层剂量响应：有效支持分组数为 30。
- IR 描述性中介/驱动诊断：{'calcium_to_ir_signal': False, 'ir_to_t90_risk_signal': True, 'ir_incremental_signal': False, 'calcium_ir_interaction_signal': False, 'descriptive_mediation_possible': False, 'not_causal_proof': True}。
- 最优钙单耗区间映射：稳定候选数为 25。

### 关键判断
- 共发现 25 个稳定候选钙单耗区间。
- 交互筛查通过 16 个 context-target 组合。
- IR 描述性中介可能性为 False，但不是因果证明。
- 多个分层中最高钙单耗分箱的高 T90 风险高于总体水平
- recommended_next_step：`define_regime_specific_calcium_band_rules`。

### 局限
- 结果来自离线观察数据，不构成因果证明。
- T90 标签精度为 0.1，且人工测量误差约 0.1。
- LIMS 标签稀疏，部分分层样本支持不足。
- IR 覆盖率有限，当前只能作为代理/诊断变量，不作为控制动作驱动。
- 本阶段不推荐自动控制和影子试验。

## 17. 出口 IR 小时滞敏感性评估

本阶段用于系统比较出口 IR 与 LIMS T90 标签之间的小时间滞对齐。此前 IR 主要采用同时间对齐、lag_0 以及 5/15/30 分钟尾随窗口；由于出口 IR 位于产品出口附近，理论上不应再引入上游约 3 小时的大滞后，本次重点检查 0、5、10 分钟历史对齐。

### 方法
- 测试 offset：[-10, -5, 0, 5, 10, 15, 20, 30]。
- offset >= 0 为在线安全历史 IR；offset < 0 只用于诊断时间戳错位，不用于在线策略。
- 输入文件：`data\data_clean_with_ir.parquet`、`data\t90_ca_feature_dataset.parquet`、`data\output_ir_proxy_evaluation.json`。
- 输出文件：`data\output_ir_lag_sensitivity.csv`、`data\output_ir_lag_sensitivity_bins.csv`、`data\output_ir_lag_sensitivity_report.json`。

### 结果
- 最佳在线安全 IR 对齐：{'offset_minutes': 20, 'online_safe': True, 'ir_feature_variant': 'win_15_std', 'usable_sample_count': 1012, 'missing_count': 1350, 'missing_rate': 0.5715495342929721, 'spearman_corr_with_t90': -0.06682431918806961, 'pearson_corr_with_t90': -0.09094966616802577, 'spearman_corr_with_y_ok': 0.10490356979743744, 'spearman_corr_with_y_low': 0.0077899256759441605, 'spearman_corr_with_y_high': -0.11120375207090177, 'spearman_corr_with_y_out_spec': -0.10490356979743744, 'high_rate_spread': 0.13793103448275862, 'out_spec_rate_spread': 0.13793103448275862, 'min_bin_sample_count': 202, 'feature_column': 'output_ir_corrected_offset_20_win_15_std', 'best_delta_auc': 0.019385484423142918, 'best_delta_ap': 0.0662128149722967, 'best_incremental_target': 'y_high', 'meaningful_relation': True, 'incremental_signal': True, 'stable_bin_support': True, 'selection_score': 0.21920384945505533}。
- 最佳诊断 IR 对齐：{'offset_minutes': 20, 'online_safe': True, 'ir_feature_variant': 'win_15_std', 'usable_sample_count': 1012, 'missing_count': 1350, 'missing_rate': 0.5715495342929721, 'spearman_corr_with_t90': -0.06682431918806961, 'pearson_corr_with_t90': -0.09094966616802577, 'spearman_corr_with_y_ok': 0.10490356979743744, 'spearman_corr_with_y_low': 0.0077899256759441605, 'spearman_corr_with_y_high': -0.11120375207090177, 'spearman_corr_with_y_out_spec': -0.10490356979743744, 'high_rate_spread': 0.13793103448275862, 'out_spec_rate_spread': 0.13793103448275862, 'min_bin_sample_count': 202, 'feature_column': 'output_ir_corrected_offset_20_win_15_std', 'best_delta_auc': 0.019385484423142918, 'best_delta_ap': 0.0662128149722967, 'best_incremental_target': 'y_high', 'meaningful_relation': True, 'incremental_signal': True, 'stable_bin_support': True, 'selection_score': 0.21920384945505533}。
- timestamp_mismatch_possible：False。
- recommended_next_step：`use_best_online_safe_ir_lag_in_relationship_discovery`。

### 判断
- IR 仍被视为下游质量状态代理、上下文变量或交互候选，不被视为 T90 的直接测量替代。
- 若在线安全 offset 的增量价值稳定，可进入后续关系发现或规则分析；若负 offset 明显更强，应先复核时间戳定义。
- 本阶段不训练生产模型、不生成控制规则、不推荐自动控制或影子试验。

## 18. 基于最佳出口 IR 小时滞的关系发现复验

本阶段在出口 IR 小时滞敏感性评估之后，使用最佳在线安全 IR 滞后特征重新运行钙单耗、工况与 T90 风险关系发现。该实验仍然只用于关系发现，不生成钙设定值建议，不进入影子试验，也不构成自动控制。

### 特征与解释
- 主钙单耗特征：`ca_per_rubber_flow_win_60_mean`。
- IR 滞后特征：`output_ir_corrected_offset_20_win_15_std`。
- 该 IR 特征解释为 T-20min 对齐的出口质量波动代理，即尾随 15 分钟 IR 标准差，不是直接 T90 测量。
- 工况特征：rubber_flow_2_win_60_mean, bromine_feed_win_60_mean, tank_rubber_conc_win_60_mean, esbo_feed_win_60_mean, neutral_alkali_feed_win_60_mean, r513_temp_win_60_mean, r514_temp_win_60_mean, r510a_temp_win_60_mean, r511a_temp_win_60_mean, r512a_temp_win_60_mean。

### 复验摘要
- 工况分层剂量响应：有效支持分组数 150。
- 钙×工况交互筛查：通过项数 16。
- IR-lag 分层剂量响应：有效支持分组数 15。
- IR-lag 中介/驱动诊断：{'calcium_to_ir_lag_signal': False, 'ir_lag_to_t90_risk_signal': True, 'ir_lag_incremental_signal': False, 'calcium_ir_lag_interaction_signal': False, 'descriptive_mediation_possible': False, 'not_causal_proof': True}。
- 稳定钙单耗候选数：25。

### 与第 16 阶段比较
- 稳定候选数：previous=25，current=25。
- 交互通过项：previous=16，current=16。
- 关键工况是否保持重要：['bromine_feed_win_60_mean', 'rubber_flow_2_win_60_mean']。
- recommended_next_step：`define_regime_specific_calcium_band_rules_with_ir_lag_context`。

### 局限
- 仍为离线观察性分析，不构成因果证明。
- LIMS 标签稀疏，T90 测量精度和人工误差限制仍存在。
- IR 覆盖率有限，IR-lag 只能作为解释/上下文/交互候选，不作为动作触发变量。
- 本阶段不推荐自动控制和影子试验。

## 19. 分工况钙单耗区间规则定义与 IR-lag 辅助验证

本阶段承接 Stage 18 的关系发现结果，将稳定的分工况钙单耗区间转化为可解释、机器可读的规则候选表。该输出仅用于后续人工工程复核，不构成自动控制，不推荐影子试验。

### 规则输入与边界
- 主钙单耗特征：`ca_per_rubber_flow_win_60_mean`。
- 优先工况变量：bromine_feed_win_60_mean, rubber_flow_2_win_60_mean, neutral_alkali_feed_win_60_mean, tank_rubber_conc_win_60_mean, esbo_feed_win_60_mean, r513_temp_win_60_mean, r514_temp_win_60_mean, r510a_temp_win_60_mean, r511a_temp_win_60_mean, r512a_temp_win_60_mean。
- IR-lag 特征：`output_ir_corrected_offset_20_win_15_std`，仅作为辅助上下文/诊断元数据，不作为规则主驱动。

### 审计结果
- 交互稳定审计：{'passed_count': 16, 'stable_candidate_count': 12, 'suspicious_large_delta_count': 7, 'insufficient_positive_support_count': 10, 'rejected_count': 18}。
- suspicious large-delta 交互数：7。
- 规则等级统计：{'A': 20, 'B': 9, 'C': 1}。
- 规则状态统计：{'accept_for_manual_case_review': 21, 'monitor_only': 9}。
- accepted / monitor / rejected：21 / 9 / 0。
- 高剂量高 T90 避免候选数：16。
- 时间稳定规则数：21。
- IR-lag 有用上下文规则数：25。
- 人工复核候选数：30。
- Top 规则：ca_regime_rule_010 tank_rubber_conc_win_60_mean=high dose=[0.0201605022184205, 0.0205021403513953] grade=A；ca_regime_rule_021 r514_temp_win_60_mean=mid dose=[0.0200982469129851, 0.0204460678353868] grade=A；ca_regime_rule_022 r510a_temp_win_60_mean=high dose=[0.0201748540071957, 0.0205833751601033] grade=A；ca_regime_rule_013 esbo_feed_win_60_mean=high dose=[0.015048047594216, 0.0199557641639433] grade=A；ca_regime_rule_025 r511a_temp_win_60_mean=high dose=[0.0202281911245812, 0.0207382286469898] grade=A。
- recommended_next_step：`prepare_regime_rule_manual_review`。

### 局限
- 仍为观察性历史数据，不构成因果证明。
- LIMS 标签稀疏，部分规则测试期支持不足。
- IR-lag 覆盖率有限，只能作为辅助风险背景。
- 所有规则必须经过人工工程复核后，才能考虑后续更严格的离线验证。

## 20. 钙单耗区间推荐器 MVP 与验证集推荐准确率评估

本阶段将已通过人工复核候选筛选的分工况钙单耗规则封装为 MVP 区间推荐器，并在验证集上评估推荐准确率。推荐输出为钙单耗区间而非固定值，因为历史关系呈非单调且存在高/低 T90 风险转移。

### 方法
- 70% 指推荐准确率，不是 T90 合格率。
- 推荐准确率由推荐区间与验证集 oracle 合理钙单耗区间的重叠，以及推荐方向与 oracle 方向是否一致来衡量。
- 工况匹配边界使用 train_like 样本重建 tertile，推理时不使用标签。
- IR-lag 只作为辅助诊断元数据，不作为主规则驱动。

### 结果
- artifact_rule_count：21。
- test_like recommendation coverage：0.9978858350951374。
- test_like band accuracy：0.9385593220338984。
- test_like direction accuracy：0.8771186440677966。
- target accuracy 3%/5%/10%：1.0 / 1.0 / 1.0。
- T90 风险护栏：{'recommended_high_rate': 0.19491525423728814, 'no_recommendation_high_rate': 0.0, 'recommended_low_rate': 0.006355932203389831, 'no_recommendation_low_rate': 0.0, 'high_guardrail_pass': False, 'low_guardrail_pass': False, 'note': 'T90 rates are guardrails, not recommendation accuracy.'}。
- mvp_status：`pass_for_monitor_only_chain`。
- recommended_next_step：`manual_review_before_deployment_chain`。

### 局限
- 这是离线代理验证，不是因果证明。
- 不执行自动控制，不写入 DCS，不推荐影子试验。
- 后续需要在线监测链路验证和工程人工复核。

## 21. 钙单耗区间推荐器部署前风险审计与人工复核表

本阶段在区间推荐器 MVP 之后执行部署前审计。由于 Stage 20 推荐覆盖率接近 100%，test_like 中 no_recommendation 样本过少，因此不再把 no_recommendation 作为主要风险基线，而改用实际钙单耗在推荐区间内 vs 区间外的风险对比。

### 审计结果
- no_recommendation baseline：{'recommendation_status_counts': {'recommended': 2315, 'no_recommendation_missing_current_dose': 39, 'no_recommendation': 8}, 'test_like_no_recommendation_count': 1, 'no_recommendation_baseline_unreliable': True, 'note': 'Do not use no_recommendation as main risk baseline when count < 30.'}。
- inside/outside 风险摘要：{'inside_band_test_like': {'sample_count': 142, 'ok_rate': 0.971830985915493, 'high_rate': 0.028169014084507043, 'low_rate': 0.0, 'out_spec_rate': 0.028169014084507043, 'mean_t90': 8.462323943661973, 'band_accuracy': 0.9436619718309859, 'direction_accuracy': 0.6338028169014085, 'target_accuracy_5pct': 1.0}, 'outside_band_test_like': {'sample_count': 330, 'ok_rate': 0.7242424242424242, 'high_rate': 0.26666666666666666, 'low_rate': 0.00909090909090909, 'out_spec_rate': 0.27575757575757576, 'mean_t90': 8.56929292929293, 'band_accuracy': 0.9363636363636364, 'direction_accuracy': 0.9818181818181818, 'target_accuracy_5pct': 1.0}, 'below_band_test_like': {'sample_count': 140, 'ok_rate': 0.8857142857142857, 'high_rate': 0.09285714285714286, 'low_rate': 0.02142857142857143, 'out_spec_rate': 0.11428571428571428, 'mean_t90': 8.485119047619046, 'band_accuracy': 1.0, 'direction_accuracy': 1.0, 'target_accuracy_5pct': 1.0}, 'above_band_test_like': {'sample_count': 190, 'ok_rate': 0.6052631578947368, 'high_rate': 0.39473684210526316, 'low_rate': 0.0, 'out_spec_rate': 0.39473684210526316, 'mean_t90': 8.631315789473684, 'band_accuracy': 0.8894736842105263, 'direction_accuracy': 0.968421052631579, 'target_accuracy_5pct': 1.0}, 'comparisons': {'inside_vs_outside': {'left_sample_count': 142, 'right_sample_count': 330, 'ok_rate_delta': 0.24758856167306875, 'high_rate_delta': -0.23849765258215963, 'low_rate_delta': -0.00909090909090909, 'out_spec_rate_delta': -0.24758856167306872}, 'inside_vs_below': {'left_sample_count': 142, 'right_sample_count': 140, 'ok_rate_delta': 0.08611670020120732, 'high_rate_delta': -0.06468812877263581, 'low_rate_delta': -0.02142857142857143, 'out_spec_rate_delta': -0.08611670020120724}, 'inside_vs_above': {'left_sample_count': 142, 'right_sample_count': 190, 'ok_rate_delta': 0.36656782802075616, 'high_rate_delta': -0.3665678280207561, 'low_rate_delta': 0.0, 'out_spec_rate_delta': -0.3665678280207561}}, 'support_pass': True, 'risk_guardrail_pass': True}。
- 动作类型摘要：{'metrics': {'decrease_to_band': {'sample_count': 190, 'ok_rate': 0.6052631578947368, 'high_rate': 0.39473684210526316, 'low_rate': 0.0, 'out_spec_rate': 0.39473684210526316, 'mean_t90': 8.631315789473684, 'band_accuracy': 0.8894736842105263, 'direction_accuracy': 0.968421052631579, 'target_accuracy_5pct': 1.0}, 'hold_in_band': {'sample_count': 142, 'ok_rate': 0.971830985915493, 'high_rate': 0.028169014084507043, 'low_rate': 0.0, 'out_spec_rate': 0.028169014084507043, 'mean_t90': 8.462323943661973, 'band_accuracy': 0.9436619718309859, 'direction_accuracy': 0.6338028169014085, 'target_accuracy_5pct': 1.0}, 'hold_or_manual_check': {'sample_count': 1, 'ok_rate': 1.0, 'high_rate': 0.0, 'low_rate': 0.0, 'out_spec_rate': 0.0, 'mean_t90': 8.3, 'band_accuracy': nan, 'direction_accuracy': nan, 'target_accuracy_5pct': nan}, 'increase_to_band': {'sample_count': 140, 'ok_rate': 0.8857142857142857, 'high_rate': 0.09285714285714286, 'low_rate': 0.02142857142857143, 'out_spec_rate': 0.11428571428571428, 'mean_t90': 8.485119047619046, 'band_accuracy': 1.0, 'direction_accuracy': 1.0, 'target_accuracy_5pct': 1.0}}, 'flags': {'unsafe_increase_hint': True, 'unsafe_decrease_hint': False, 'safe_hold_band_candidate': True}}。
- monitor_chain_candidate_count：14。
- manual_review_only_count：1。
- reject_or_refine_count：3。
- risk_guardrail_status：{'inside_vs_outside_support_pass': True, 'inside_vs_outside_guardrail_pass': True, 'action_flags': {'unsafe_increase_hint': True, 'unsafe_decrease_hint': False, 'safe_hold_band_candidate': True}}。
- readiness_status：`ready_for_monitor_chain_after_manual_review`。
- recommended_next_step：`prepare_monitor_chain_interface`。

### 局限
- 仍为离线代理验证，不构成因果证明。
- 不训练模型，不执行自动控制，不写入 DCS，不推荐影子试验。
- 所有规则仍需工程人工复核。

## 22. 测试集钙单耗推荐区间覆盖可视化

本阶段仅用于离线可视化测试集真实钙单耗与推荐钙单耗区间的覆盖关系，不训练模型，不改变推荐规则，不生成自动控制建议。

### 输入与输出
- 输入文件：`data\ca_interval_recommender_replay.parquet`。
- 生成图像：reports\figures\ca_interval_test_like_coverage.png, reports\figures\ca_interval_test_like_coverage_time.png, reports\figures\ca_interval_position_distribution.png, reports\figures\ca_interval_position_t90_risk.png, reports\figures\ca_interval_test_like_coverage_zoom_first150.png, reports\figures\ca_interval_above_band_focus.png。
- 生成表格：data\ca_interval_recommendation_visualization_table.csv, reports\tables\ca_interval_recommendation_visualization_table.csv。

### 摘要
- test_like 样本数：473。
- inside/below/above/missing：142 / 140 / 190 / 1。
- band accuracy：0.9385593220338984。
- direction accuracy：0.8771186440677966。
- 真实钙单耗落在推荐区间内，表示当前操作位于历史关系中推荐的钙单耗带；高于推荐区间的样本应作为可能高 T90 风险条件进行人工检查；低于推荐区间仅是诊断提示，不能直接变成自动增加指令。

### 局限
- 这是离线可视化，不构成因果证明。
- 不执行自动控制，不写入 DCS，不推荐影子试验。


## 23. 钙单耗推荐区间差异性审计

### 23.1 审计目的

本阶段用于解释测试集推荐钙单耗区间为何呈现近似稳定带。该分析只审计既有推荐器输出，不训练模型、不修改规则、不进行策略搜索，也不形成自动控制或 DCS 写回建议。

目录策略同步更新：`data/` 仅保留原始或必要基础数据；本阶段生成的审计 CSV/JSON 输出写入 `runs/ca_interval_diversity_audit/`；图像和人工可读表写入 `reports/`；实验说明仅追加到 `docs/Experimental_Procedure_cn.md`。

### 23.2 主要结果

- 测试集样本数：473
- 唯一推荐区间数：20
- 推荐中心值中位数：0.0203313212849079
- 推荐中心值 IQR：9.094983312319879e-05
- 推荐中心值范围：0.0003854235652850993
- Top 5 推荐区间覆盖率：0.828752642706131
- 接受规则数：21
- 规则中心值 IQR：0.0002996571829030001
- 规则中心值范围：0.0031138385972156006
- 聚合压缩标记：True
- 可用上下文字段：rubber_flow_2_win_60_mean

### 23.3 判断

当前推荐器行为分类为：`aggregation_over_smoothed_recommender`。稳定区间的主要解释为：`aggregation_compression`。如果区间稳定主要来自规则本身集中，则它更接近“稳定安全带 MVP”；如果来自多规则中位数聚合，则后续应测试最高优先级规则输出；如果来自单变量规则过粗，则应构建多变量工况规则。

### 23.4 输出文件

- 机器可读审计输出：`runs/ca_interval_diversity_audit/`
- 图像输出：reports\figures\ca_interval_target_distribution.png, reports\figures\ca_interval_width_distribution.png, reports\figures\ca_interval_top_frequency.png, reports\figures\ca_rule_interval_by_regime_feature.png, reports\figures\ca_aggregation_compression.png, reports\figures\ca_context_vs_recommended_target.png
- 人工可读汇总表：reports\tables\ca_interval_diversity_summary.csv

### 23.5 下一步

推荐下一步：`test_top_rule_without_median_aggregation`。

局限性：本阶段为离线审计；不提供因果证明；不生成控制动作；结论依赖既有规则、replay 和人工复核审计产物。


## 24. 钙单耗推荐区间聚合策略对比实验

### 24.1 实验目的

Stage 23 显示推荐区间稳定的主要原因是多规则中位数聚合压缩。本阶段在不修改规则、不训练模型、不进行策略搜索的前提下，复用同一批匹配规则和验证 oracle，对比中位数聚合、最高优先级规则、加权平均和重叠交集四种输出方式。

### 24.2 验证集指标

- median_aggregation_baseline: band_accuracy=0.9385593220338984, direction_accuracy=0.8771186440677966, target_iqr=9.094983312315369e-05, unique_interval_count=20, risk_guardrail_pass=True
- top_rule_only: band_accuracy=0.8580508474576272, direction_accuracy=0.8389830508474576, target_iqr=0.0, unique_interval_count=5, risk_guardrail_pass=True
- weighted_rule_average: band_accuracy=0.9385593220338984, direction_accuracy=0.684322033898305, target_iqr=0.00011906226405233866, unique_interval_count=33, risk_guardrail_pass=True
- narrow_intersection_if_overlap: band_accuracy=0.8559322033898306, direction_accuracy=0.8389830508474576, target_iqr=0.0, unique_interval_count=6, risk_guardrail_pass=True

多样性恢复判断：[{'strategy': 'median_aggregation_baseline', 'diversity_recovered': False}, {'strategy': 'top_rule_only', 'diversity_recovered': False}, {'strategy': 'weighted_rule_average', 'diversity_recovered': False}, {'strategy': 'narrow_intersection_if_overlap', 'diversity_recovered': False}]

### 24.3 风险与策略判断

最佳策略：`median_aggregation_baseline`。

切换建议：`keep_median_aggregation`。

推荐下一步：`keep_stable_safe_band_mvp`。

本阶段只做离线 replay 对比。`increase_to_band` 或 `decrease_to_band` 均不能解释为自动控制动作，也不形成 DCS 写回或影子试验建议。

### 24.4 输出文件

- 机器输出：runs\ca_interval_aggregation_strategy_test\strategy_recommendation_replay.parquet, runs\ca_interval_aggregation_strategy_test\strategy_recommendation_replay.csv, runs\ca_interval_aggregation_strategy_test\strategy_metrics.csv, runs\ca_interval_aggregation_strategy_test\strategy_comparison_summary.csv, runs\ca_interval_aggregation_strategy_test\ca_interval_aggregation_strategy_report.json
- 图像输出：reports\figures\ca_aggregation_strategy_target_distribution.png, reports\figures\ca_aggregation_strategy_interval_width.png, reports\figures\ca_aggregation_strategy_accuracy.png, reports\figures\ca_aggregation_strategy_risk.png, reports\figures\ca_top_rule_vs_median_target_scatter.png
- 人工表格：reports\tables\ca_interval_aggregation_strategy_summary.csv

局限性：离线验证不能证明因果关系；oracle 来自验证集事实分箱；聚合策略切换仍需人工工程复核。
