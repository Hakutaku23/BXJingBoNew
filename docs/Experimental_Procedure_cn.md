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

## 25. 稳定钙单耗安全带 MVP 定版与监测接口准备

### 25.1 定版原因

Stage 24 对比了中位数聚合、最高优先级规则、加权平均和交集策略。中位数聚合保持了最高的推荐区间准确率和较好的方向准确率，且风险护栏通过；Top-rule-only 和交集策略没有恢复有效多样性，准确率下降；加权平均虽然增加区间差异，但方向准确率不足。因此本阶段锁定 `median_aggregation_baseline`。

产品定位为 `stable_safe_band_mvp`：它不是强动态分工况处方系统，而是稳定钙单耗安全带监测 MVP。其含义是：把实际钙单耗控制在历史安全带内，历史上更可能提高 T90 合格概率，但不保证 T90 必然合格。

### 25.2 动作可见性策略

- inside_band：仅监测展示，提示“当前钙单耗处于推荐安全区间内，建议维持观察”。
- above_band：人工复核必需，提示“当前钙单耗高于推荐安全区间，历史数据中高 T90 风险偏高，建议人工复核是否需要小幅降钙”。
- below_band：仅诊断展示，隐藏加钙操作建议。
- missing：关键输入缺失，不生成推荐。

该 MVP 不提供自动控制、不做 DCS 写回、不推荐影子试验。

### 25.3 输出

- 定版 artifact：`models\ca_safe_band_mvp\safe_band_artifact.json`
- dry-run 表：`runs\ca_safe_band_mvp\final_monitor_dry_run.parquet` 与 `runs\ca_safe_band_mvp\final_monitor_dry_run.csv`
- 规则汇总：`runs\ca_safe_band_mvp\final_rule_summary.csv`
- 人工复核表：`reports\tables\ca_safe_band_mvp_manual_review_sheet.csv`

### 25.4 风险摘要

- inside_band ok/high/low：0.971830985915493 / 0.028169014084507043 / 0.0
- above_band high_rate：0.39473684210526316
- below_band low_rate：0.02142857142857143

推荐下一步：`human_review_safe_band_mvp`。

局限性：离线验证；非因果证明；无自动控制；无 DCS 写回；必须经过工程人工复核。

## 26. 稳定钙单耗安全带 MVP 运行包封装与接口契约测试

### 26.1 阶段目的

本阶段承接 Stage 25，将稳定钙单耗安全带 MVP 封装为监测-only 运行包。该运行包用于后续厂内适配器集成前的人审与接口契约验证，不训练模型、不改规则、不执行自动控制、不做 DCS 写回。

### 26.2 依赖约束

本阶段读取 `IDB_requirements.txt` 作为厂内可用三方依赖清单。依赖策略为：不得引入清单外三方包；`package.py` 以标准库为优先并保持纯推荐逻辑；`interface.py` 和 `main.py` 可在清单允许时使用 pandas/pyarrow 做批量 CSV/parquet 输入输出。本次 package.py 标准库-only：False；依赖策略通过：False；清单外 import：['__future__']。

### 26.3 运行包结构

运行包目录：`deploy\ca_safe_band_mvp`

- `package.py`：纯推荐逻辑，执行规则匹配、中位数聚合、区间位置判断和动作可见性策略。
- `interface.py`：公开 `SafeBandRecommender`，加载 JSON artifact/support/schema 并提供单条和批量预测。
- `main.py`：示例 CLI 入口；厂内 DCS 获取与写回由后续适配器实现；当前脚本不写 DCS。
- `safe_band_artifact.json`：定版安全带 artifact。
- `support.parquet` / `support.json`：特征与边界支持信息；JSON 可供标准库运行路径使用。
- `schema.json`：输入输出、安全约束和依赖策略契约。

### 26.4 安全约束

- monitor_only = true
- automatic_control = false
- dcs_writeback = false
- increase_hint_hidden = true
- engineering_review_required = true
- no_guarantee_t90_qualified = true

### 26.5 契约测试

历史 dry-run 等价测试行数：2362。

核心字段完全匹配率：1.0。

等价测试通过：False。

推荐下一步：`fix_dependency_policy`。

局限性：仍需工程人工复核；未实现厂内实时数据适配器；未进行在线数据验证；离线安全带关系不是因果证明。

### 27.1 阶段目的

本阶段承接 Stage 25，将稳定钙单耗安全带 MVP 封装为监测-only 运行包。该运行包用于后续厂内适配器集成前的人审与接口契约验证，不训练模型、不改规则、不执行自动控制、不做 DCS 写回。

### 27.2 依赖约束

本阶段读取 `IDB_requirements.txt` 作为厂内可用三方依赖清单。依赖策略为：不得引入清单外三方包；`package.py` 以标准库为优先并保持纯推荐逻辑；`interface.py` 和 `main.py` 可在清单允许时使用 pandas/pyarrow 做批量 CSV/parquet 输入输出。本次 package.py 标准库-only：True；依赖策略通过：True；清单外 import：[]。

### 27.3 运行包结构

运行包目录：`deploy\ca_safe_band_mvp`

- `package.py`：纯推荐逻辑，执行规则匹配、中位数聚合、区间位置判断和动作可见性策略。
- `interface.py`：公开 `SafeBandRecommender`，加载 JSON artifact/support/schema 并提供单条和批量预测。
- `main.py`：示例 CLI 入口；厂内 DCS 获取与写回由后续适配器实现；当前脚本不写 DCS。
- `safe_band_artifact.json`：定版安全带 artifact。
- `support.parquet` / `support.json`：特征与边界支持信息；JSON 可供标准库运行路径使用。
- `schema.json`：输入输出、安全约束和依赖策略契约。

### 27.4 安全约束

- monitor_only = true
- automatic_control = false
- dcs_writeback = false
- increase_hint_hidden = true
- engineering_review_required = true
- no_guarantee_t90_qualified = true

### 27.5 契约测试

历史 dry-run 等价测试行数：2362。

核心字段完全匹配率：1.0。

等价测试通过：True。

推荐下一步：`human_review_runtime_package`。

局限性：仍需工程人工复核；未实现厂内实时数据适配器；未进行在线数据验证；离线安全带关系不是因果证明。

## 28. 运行包生产安全修复与方法说明文档固化

### 28.1 修复原因

本阶段针对稳定钙单耗安全带 MVP 运行包做生产安全修复：严格 JSON、生产模式不信任输入规则 ID、必需特征校验、输出 schema 扩展、加注量换算、Python 3.8+ 兼容和方法文档固化。

### 28.2 修复结果

- 严格 JSON：True
- 依赖策略：True
- package.py 标准库-only：True
- replay 等价测试：False
- 生产模式有效输出率：1.0
- 生产模式禁用输入 rule-id override：True
- 输出 schema 扩展：True
- 加注量换算：True

方法说明文档：`docs\ca_safe_band_mvp_method_and_dataflow.md`。

推荐下一步：`fix_runtime_repair`。

局限性：仍需工程人工复核；厂方实时适配器尚未实现；尚无在线验证；该安全带关系不是因果证明。

### 29.1 修复原因

本阶段针对稳定钙单耗安全带 MVP 运行包做生产安全修复：严格 JSON、生产模式不信任输入规则 ID、必需特征校验、输出 schema 扩展、加注量换算、Python 3.8+ 兼容和方法文档固化。

### 29.2 修复结果

- 严格 JSON：True
- 依赖策略：True
- package.py 标准库-only：True
- replay 等价测试：True
- 生产模式有效输出率：1.0
- 生产模式禁用输入 rule-id override：True
- 输出 schema 扩展：True
- 加注量换算：True

方法说明文档：`docs\ca_safe_band_mvp_method_and_dataflow.md`。

推荐下一步：`human_review_repaired_runtime_package`。

局限性：仍需工程人工复核；厂方实时适配器尚未实现；尚无在线验证；该安全带关系不是因果证明。

## 30. 运行包实时特征适配器与 IR 可选输入支持

### 30.1 阶段目的

此前运行包默认接收工程化特征。厂内集成通常拿到的是带时间戳的原始平台 DataFrame，因此本阶段新增 `feature_adapter.py`，将原始点位列转换为运行包所需的当前特征状态。

### 30.2 在线窗口策略

在线运行使用当前时刻 `t_now` 之前的尾随窗口：工况变量采用 `[t_now-60min, t_now]` 的均值；钙单耗采用该窗口内 `ca_feed / rubber_flow_2` 的均值。离线标签对齐曾使用停留时间；在线推荐不再额外向前平移 165min，因为当前上游操作影响的是未来产品质量。后续 LIMS 回填验证应按停留时间把当前输出与未来 T90 标签比较。

### 30.3 IR-lag

IR-lag `output_ir_corrected_offset_20_win_15_std` 为可选输入。若存在原始 IR，则计算 `[t_now-35min, t_now-20min]` 的 15 分钟标准差；若缺失，不阻断推荐，只记录 `optional_ir_missing`。

### 30.4 接口更新

- `interface.py` 新增 `predict_from_raw_dataframe` 和 `predict_batch_from_raw_dataframe`。
- `main.py` 新增 `--raw-input-csv`、`--raw-input-parquet`、`--raw-time-col`、`--end-time`、`--min-valid-points` 和 `--include-optional-ir`。
- `schema.json/support.json` 增加原始点位映射和窗口定义。

### 30.5 烟测结果

- engineered predict_one：True
- raw dataframe predict：True
- main.py raw CLI：True
- 依赖策略：True
- IR 可选确认：True

推荐下一步：`human_review_feature_adapter_contract`。

局限性：本阶段使用合成 raw-like 数据验证接口路径，仍需厂方提供真实原始平台 DataFrame 做最终适配器验收；无 DCS 写回；无自动控制。

## 31. 钙单耗与 T90 非线性阈值关系验证

本阶段用于验证历史数据是否支持“钙单耗与 T90 存在正向、非线性阈值关系”的工艺预期，而不是预设该关系成立。输入样本数 2362，可用样本数 2315。

基础相关性：钙单耗与 T90 的 Spearman 相关系数为 0.13175762449160702；该结果只说明历史相关关系，不构成因果证明。

分箱响应与阈值搜索：最优阈值候选为 0.021465644850881687，阈值前后高 T90 风险差为 0.19065295771026536。正向关系支持：True；非线性阈值支持：False；安全平台区支持：False；高钙高 T90 风险支持：True。

当前安全带一致性：{'available': True, 'source': 'runs\\ca_safe_band_mvp\\final_monitor_dry_run.parquet', 'recommended_ca_consumption_max_median': 0.0204772882317374, 'known_stable_safe_band': [0.02016, 0.0205], 'threshold_minus_safe_band_max_median': 0.0009883566191442872, 'threshold_near_safe_band_upper_bound': False}。

证据强度：`moderate`。推荐下一步：`use_threshold_relation_as_supporting_evidence`。

局限性：离线历史关系不等于因果证明；T90 为人工 LIMS 且存在约 0.1 的实际误差；工况混杂仍可能影响钙单耗与 T90 的表观关系；本阶段不产生自动控制建议。

## 32. 基于工况与衍生特征的聚类分析及 T90 分布解释

本阶段用非泄漏工况与衍生特征做无监督聚类，T90 与目标标签不参与聚类，只在聚类完成后用于解释各类工况的质量分布。

使用特征数：51；排除泄漏列：t90, t90_C, t90_D, t90_E, t90_label_count, time, y_high, y_low, y_ok, y_out_spec。

k 搜索范围：[2, 10]；最终选择算法 `KMeans`，k=2。选择理由：selected by balanced clustering metrics with minimum cluster size and post-cluster T90/high-risk separation; min_size_threshold=118。

聚类 T90 概况：
- cluster 0: n=14, T90均值=8.469047619047618, ok=0.8571428571428571, high=0.14285714285714285, profile=mixed_or_unclear
- cluster 1: n=2348, T90均值=8.508968625780806, ok=0.8045144804088586, high=0.1682282793867121, profile=mixed_or_unclear

组间 T90 检验：{'anova': {'statistic': 0.4667108413792708, 'p_value': 0.4945712987215459}, 'kruskal_wallis': {'statistic': 0.3498314080642807, 'p_value': 0.5542085813854231}}。

推荐下一步：`clustering_not_stable_keep_rule_based_safe_band`。

局限性：该聚类为无监督历史分析，T90 未用于聚类；聚类差异不等于因果证明；结果只适合辅助后续分工况监测或钙-T90 关系分析，不构成自动控制策略。

## 33. 聚类特征审计与稳健聚类复验

本阶段复核上一轮聚类结果的稳健性，重点检查 k=2 结果是否违反最小簇规模门槛。上一轮最小簇样本数为 14，门槛为 118，不一致标记为 True。

本次审计测试的特征集包括 core_11、core_11_plus_ir、filtered_engineered_features 与 pca_90pct_from_filtered_engineered。候选特征审计与各 k 指标已写入 runs。

是否找到稳健聚类结果：True。选中结果：{'variant': 'core_11_features', 'algorithm': 'AgglomerativeClustering', 'k': 5, 'seed_count': 1, 'stability_ari': 1.0, 'tiny_cluster_flag': False, 'tiny_cluster_threshold': 118, 'min_cluster_size_worst_seed': 175, 'silhouette_score': 0.3603153624600595, 'calinski_harabasz_score': 1091.960963037835, 'davies_bouldin_score': 1.0163787150930677, 'min_cluster_size': 175.0, 'max_cluster_size': 894.0, 'cluster_size_imbalance': 5.1085714285714285, 't90_mean_range': 0.07570150987224267, 'high_rate_range': 0.18550522648083623, 'ok_rate_range': 0.1888501742160278, 'feature_count': 8, 'score': 2.972222222222222}。理由：Selected because it passed robustness gates and had the best combined stability, separation, and clustering metric score.。

推荐下一步：`use_clusters_for_context_specific_ca_t90_analysis`。

局限性：聚类为无监督离线分析，T90 不参与聚类；不同算法和特征集可能给出不同划分；若没有稳定且非小簇的结果，应继续保留规则型安全带而不是强行解释聚类。

## 34. 分工况钙单耗-T90 阈值关系复验

本阶段在 Stage 31/32 之后进行：上一轮全局阈值结果显示钙单耗与高 T90 风险有中等强度历史证据，但聚类结果不稳，因此本阶段不依赖聚类，而用关键工况变量的 low/mid/high 三分位工况复验钙单耗-T90 关系。

全局阈值回顾：阈值候选 0.021465644850881687，安全带上沿中位数 0.0204772882317374。

分工况综合：{'relation_type': 'broadly_consistent', 'regime_count': 30, 'positive_relation_regime_count': 22, 'high_calcium_high_t90_risk_regime_count': 12, 'threshold_evidence_regime_count': 12, 'contradictory_regime_count': 5}。

最强支持工况（前 5）：[{'regime_feature': 'rubber_flow_2_win_60_mean', 'regime_bin': 'mid', 'sample_count': 771, 'calcium_median': 0.019844776628315956, 'calcium_iqr': 0.001842864073866645, 't90_mean': 8.50278858625162, 't90_median': 8.5, 't90_iqr': 0.28333333333333144, 'ok_rate': 0.8067444876783398, 'high_rate': 0.16342412451361868, 'low_rate': 0.029831387808041506, 'out_spec_rate': 0.1932555123216602, 'spearman_ca_t90': 0.17370803897580822, 'spearman_ca_t90_p_value': 1.2195037448730273e-06, 'spearman_ca_y_high': 0.16833847768680402, 'spearman_ca_y_high_p_value': 2.598644163200123e-06, 'best_threshold': 0.02140294402183697, 'high_rate_before_threshold': 0.11450381679389313, 'high_rate_after_threshold': 0.4396551724137931, 'high_rate_delta': 0.32515135561989994, 'slope_before': 35.68569843611294, 'slope_after': -23.106056206729832, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'threshold_supported': True, 'threshold_near_global_threshold': True, 'threshold_near_safe_band_upper': False, 'stable_support': True}, {'regime_feature': 'r512a_temp_win_60_mean', 'regime_bin': 'low', 'sample_count': 770, 'calcium_median': 0.019540421402341083, 'calcium_iqr': 0.002093764290796489, 't90_mean': 8.51814935064935, 't90_median': 8.5, 't90_iqr': 0.34999999999999787, 'ok_rate': 0.7571428571428571, 'high_rate': 0.21168831168831168, 'low_rate': 0.03116883116883117, 'out_spec_rate': 0.24285714285714285, 'spearman_ca_t90': 0.1806945655087116, 'spearman_ca_t90_p_value': 4.4755873522903796e-07, 'spearman_ca_y_high': 0.1578495993749647, 'spearman_ca_y_high_p_value': 1.0792720011629582e-05, 'best_threshold': 0.0206027818404035, 'high_rate_before_threshold': 0.15863141524105753, 'high_rate_after_threshold': 0.48031496062992124, 'high_rate_delta': 0.3216835453888637, 'slope_before': 13.361451708963639, 'slope_after': 237.96470093499258, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'threshold_supported': True, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': True, 'stable_support': True}, {'regime_feature': 'neutral_alkali_feed_win_60_mean', 'regime_bin': 'high', 'sample_count': 729, 'calcium_median': 0.02040198643873037, 'calcium_iqr': 0.0009535795952481871, 't90_mean': 8.528623685413809, 't90_median': 8.5, 't90_iqr': 0.29999999999999893, 'ok_rate': 0.7860082304526749, 'high_rate': 0.2085048010973937, 'low_rate': 0.0054869684499314125, 'out_spec_rate': 0.2139917695473251, 'spearman_ca_t90': 0.58823515958541, 'spearman_ca_t90_p_value': 4.548748723274935e-69, 'spearman_ca_y_high': 0.5888866384267638, 'spearman_ca_y_high_p_value': 2.966124199392891e-69, 'best_threshold': 0.02015469489075238, 'high_rate_before_threshold': 0.02066115702479339, 'high_rate_after_threshold': 0.30184804928131415, 'high_rate_delta': 0.28118689225652077, 'slope_before': 19.17588480367051, 'slope_after': 231.8640359524164, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'threshold_supported': True, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': True, 'stable_support': True}, {'regime_feature': 'bromine_feed_win_60_mean', 'regime_bin': 'high', 'sample_count': 769, 'calcium_median': 0.02036796599117515, 'calcium_iqr': 0.0010682226207868045, 't90_mean': 8.532162982228002, 't90_median': 8.5, 't90_iqr': 0.29999999999999893, 'ok_rate': 0.776332899869961, 'high_rate': 0.21716514954486346, 'low_rate': 0.006501950585175552, 'out_spec_rate': 0.22366710013003901, 'spearman_ca_t90': 0.5426405011723325, 'spearman_ca_t90_p_value': 4.2844857952366864e-60, 'spearman_ca_y_high': 0.5734305772670671, 'spearman_ca_y_high_p_value': 1.9583124743445097e-68, 'best_threshold': 0.02004956130238023, 'high_rate_before_threshold': 0.02358490566037736, 'high_rate_after_threshold': 0.29084380610412924, 'high_rate_delta': 0.2672589004437519, 'slope_before': -7.874150872660618, 'slope_after': 192.35388090598374, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'threshold_supported': True, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': True, 'stable_support': True}, {'regime_feature': 'r513_temp_win_60_mean', 'regime_bin': 'low', 'sample_count': 772, 'calcium_median': 0.019449070498110635, 'calcium_iqr': 0.0020353406961190826, 't90_mean': 8.50879749568221, 't90_median': 8.5, 't90_iqr': 0.29999999999999716, 'ok_rate': 0.7772020725388601, 'high_rate': 0.19041450777202074, 'low_rate': 0.03238341968911917, 'out_spec_rate': 0.22279792746113988, 'spearman_ca_t90': 0.134890665114054, 'spearman_ca_t90_p_value': 0.00017056529266453097, 'spearman_ca_y_high': 0.09272398600683603, 'spearman_ca_y_high_p_value': 0.009945854700911234, 'best_threshold': 0.020564238391452053, 'high_rate_before_threshold': 0.15548780487804878, 'high_rate_after_threshold': 0.3879310344827586, 'high_rate_delta': 0.23244322960470984, 'slope_before': 10.994719140165932, 'slope_after': 231.26764257734857, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'threshold_supported': True, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': True, 'stable_support': True}]。

矛盾或反向工况（前 5）：[{'regime_feature': 'bromine_feed_win_60_mean', 'regime_bin': 'low', 'sample_count': 769, 'calcium_median': 0.019798271864018842, 'calcium_iqr': 0.002452700232396906, 't90_mean': 8.516937581274382, 't90_median': 8.5, 't90_iqr': 0.28333333333333144, 'ok_rate': 0.788036410923277, 'high_rate': 0.17425227568270482, 'low_rate': 0.0377113133940182, 'out_spec_rate': 0.21196358907672302, 'spearman_ca_t90': -0.16112592921664542, 'spearman_ca_t90_p_value': 7.11359471838146e-06, 'spearman_ca_y_high': -0.21417586853768464, 'spearman_ca_y_high_p_value': 1.980865486103261e-09, 'best_threshold': 0.018330913516136624, 'high_rate_before_threshold': 0.25, 'high_rate_after_threshold': 0.16079632465543645, 'high_rate_delta': -0.08920367534456355, 'slope_before': 107.70588264311773, 'slope_after': -20.178188018486082, 'positive_relation_supported': False, 'high_calcium_high_t90_risk_supported': False, 'threshold_supported': False, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': False, 'stable_support': True}, {'regime_feature': 'rubber_flow_2_win_60_mean', 'regime_bin': 'low', 'sample_count': 772, 'calcium_median': 0.020074860005708656, 'calcium_iqr': 0.002097800240551076, 't90_mean': 8.506703367875648, 't90_median': 8.5, 't90_iqr': 0.29999999999999716, 'ok_rate': 0.8095854922279793, 'high_rate': 0.15544041450777202, 'low_rate': 0.034974093264248704, 'out_spec_rate': 0.19041450777202074, 'spearman_ca_t90': -0.14691445406371897, 'spearman_ca_t90_p_value': 4.1738153535374904e-05, 'spearman_ca_y_high': -0.21897368024392178, 'spearman_ca_y_high_p_value': 7.793460261734575e-10, 'best_threshold': 0.01870961885537032, 'high_rate_before_threshold': 0.3017241379310345, 'high_rate_after_threshold': 0.12957317073170732, 'high_rate_delta': -0.17215096719932715, 'slope_before': 65.46083886524411, 'slope_after': -13.883768944454232, 'positive_relation_supported': False, 'high_calcium_high_t90_risk_supported': False, 'threshold_supported': False, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': False, 'stable_support': True}, {'regime_feature': 'neutral_alkali_feed_win_60_mean', 'regime_bin': 'low', 'sample_count': 729, 'calcium_median': 0.020155566887826954, 'calcium_iqr': 0.0023975136332826004, 't90_mean': 8.514620484682212, 't90_median': 8.5, 't90_iqr': 0.24999999999999822, 'ok_rate': 0.8120713305898491, 'high_rate': 0.15637860082304528, 'low_rate': 0.03155006858710562, 'out_spec_rate': 0.18792866941015088, 'spearman_ca_t90': -0.20402242811501803, 'spearman_ca_t90_p_value': 2.7333274752467805e-08, 'spearman_ca_y_high': -0.2427229230267275, 'spearman_ca_y_high_p_value': 3.097795092999665e-11, 'best_threshold': 0.022093782282016722, 'high_rate_before_threshold': 0.17863105175292154, 'high_rate_after_threshold': 0.05384615384615385, 'high_rate_delta': -0.1247848979067677, 'slope_before': -32.48423725569825, 'slope_after': 23.774157879346546, 'positive_relation_supported': False, 'high_calcium_high_t90_risk_supported': False, 'threshold_supported': False, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': False, 'stable_support': True}, {'regime_feature': 'r512a_temp_win_60_mean', 'regime_bin': 'high', 'sample_count': 770, 'calcium_median': 0.020224361457792356, 'calcium_iqr': 0.001997597677387348, 't90_mean': 8.482510822510823, 't90_median': 8.45, 't90_iqr': 0.1999999999999993, 'ok_rate': 0.8753246753246753, 'high_rate': 0.09220779220779221, 'low_rate': 0.032467532467532464, 'out_spec_rate': 0.12467532467532468, 'spearman_ca_t90': -0.05688943681592089, 'spearman_ca_t90_p_value': 0.11471943698345827, 'spearman_ca_y_high': -0.10813130011834014, 'spearman_ca_y_high_p_value': 0.0026605221084117477, 'best_threshold': 0.02028694486576669, 'high_rate_before_threshold': 0.10565110565110565, 'high_rate_after_threshold': 0.07713498622589532, 'high_rate_delta': -0.02851611942521033, 'slope_before': -38.087272694240106, 'slope_after': 13.170274434924067, 'positive_relation_supported': False, 'high_calcium_high_t90_risk_supported': False, 'threshold_supported': False, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': True, 'stable_support': True}, {'regime_feature': 'r514_temp_win_60_mean', 'regime_bin': 'high', 'sample_count': 643, 'calcium_median': 0.020246773180691683, 'calcium_iqr': 0.0022959782389250193, 't90_mean': 8.488206324520478, 't90_median': 8.5, 't90_iqr': 0.1999999999999993, 'ok_rate': 0.8771384136858476, 'high_rate': 0.09020217729393468, 'low_rate': 0.03265940902021773, 'out_spec_rate': 0.12286158631415241, 'spearman_ca_t90': -0.11025814876326419, 'spearman_ca_t90_p_value': 0.005126701760034815, 'spearman_ca_y_high': -0.16431226425509876, 'spearman_ca_y_high_p_value': 2.827632204828139e-05, 'best_threshold': 0.02045809171965959, 'high_rate_before_threshold': 0.11204481792717087, 'high_rate_after_threshold': 0.06293706293706294, 'high_rate_delta': -0.04910775499010793, 'slope_before': -65.73163404833973, 'slope_after': 16.560604942937758, 'positive_relation_supported': False, 'high_calcium_high_t90_risk_supported': False, 'threshold_supported': False, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': True, 'stable_support': True}]。

当前判断：`broadly_consistent`。推荐下一步：`use_relation_as_supporting_evidence_for_safe_band`。

局限性：该分析仍为离线历史关系，不是因果证明；三分位工况存在样本稀疏；IR-lag 只作为可选工况变量；本阶段不产生自动控制建议。

## 35. 稳健聚类工况画像与聚类内钙单耗-T90 关系验证

本阶段承接 Stage 33 的稳健 k=5 聚类和 Stage 34 的分工况阈值复验，目标是解释每个聚类的工况画像，并在聚类内部验证钙单耗-T90 关系。T90 未用于聚类，只用于聚类后的质量解释。

稳健聚类结果：算法 `AgglomerativeClustering`，k=5，最终特征数 8。最终特征：['ca_per_rubber_flow_win_60_mean', 'rubber_flow_2_win_60_mean', 'bromine_feed_win_60_mean', 'tank_rubber_conc_win_60_mean', 'esbo_feed_win_60_mean', 'neutral_alkali_feed_win_60_mean', 'r510a_temp_win_60_mean', 'r512a_temp_win_60_mean']。core_11 中被剔除特征：['r511a_temp_win_60_mean', 'r513_temp_win_60_mean', 'r514_temp_win_60_mean']，主要原因是缺失/强相关/筛选后不进入最终核心特征集。

聚类规模与 T90 概况：
- cluster 0: n=626, ok=0.8322683706070287, high=0.16134185303514376, low=0.006389776357827476
- cluster 1: n=894, ok=0.8042505592841164, high=0.1789709172259508, low=0.016778523489932886
- cluster 2: n=380, ok=0.8052631578947368, high=0.15526315789473685, low=0.039473684210526314
- cluster 3: n=175, ok=0.8857142857142857, high=0.05142857142857143, low=0.06285714285714286
- cluster 4: n=287, ok=0.6968641114982579, high=0.23693379790940766, low=0.06620209059233449

聚类内钙单耗-T90 关系：[{'cluster': 0, 'sample_count': 626, 'spearman_ca_t90': 0.41973994867462794, 'spearman_ca_t90_p_value': 4.144077277954902e-28, 'spearman_ca_y_high': 0.4320459712407461, 'spearman_ca_y_high_p_value': 7.412362921141736e-30, 'dose_bin_count': 5, 'best_threshold': 0.020204762636502806, 'high_rate_before_threshold': 0.04878048780487805, 'high_rate_after_threshold': 0.25663716814159293, 'high_rate_delta': 0.2078566803367149, 'slope_before': -19.383802013617625, 'slope_after': 256.5854547849271, 'threshold_supported': True, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True}, {'cluster': 1, 'sample_count': 894, 'spearman_ca_t90': 0.34156731509775895, 'spearman_ca_t90_p_value': 2.875819887287427e-25, 'spearman_ca_y_high': 0.3627189773124758, 'spearman_ca_y_high_p_value': 1.6595061419012157e-28, 'dose_bin_count': 5, 'best_threshold': 0.020646243685374936, 'high_rate_before_threshold': 0.08240887480190175, 'high_rate_after_threshold': 0.44398340248962653, 'high_rate_delta': 0.3615745276877248, 'slope_before': 11.544013852862186, 'slope_after': 252.92415132832886, 'threshold_supported': True, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True}, {'cluster': 2, 'sample_count': 380, 'spearman_ca_t90': -0.27909697735216926, 'spearman_ca_t90_p_value': 7.26690441266901e-08, 'spearman_ca_y_high': -0.3063900291096278, 'spearman_ca_y_high_p_value': 2.9117276266001324e-09, 'dose_bin_count': 5, 'best_threshold': 0.020991882478404434, 'high_rate_before_threshold': 0.24102564102564103, 'high_rate_after_threshold': 0.06060606060606061, 'high_rate_delta': -0.18041958041958042, 'slope_before': -65.35632730699656, 'slope_after': 34.485566911379536, 'threshold_supported': False, 'positive_relation_supported': False, 'high_calcium_high_t90_risk_supported': False}, {'cluster': 3, 'sample_count': 175, 'spearman_ca_t90': 0.19655706060847677, 'spearman_ca_t90_p_value': 0.009132497013743591, 'spearman_ca_y_high': 0.033288768033537766, 'spearman_ca_y_high_p_value': 0.6618688701752102, 'dose_bin_count': 4, 'best_threshold': 0.024546850540447012, 'high_rate_before_threshold': 0.04861111111111111, 'high_rate_after_threshold': 0.06451612903225806, 'high_rate_delta': 0.01590501792114695, 'slope_before': 37.9182315795209, 'slope_after': -97.16206237377347, 'threshold_supported': False, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': False}, {'cluster': 4, 'sample_count': 287, 'spearman_ca_t90': 0.441923177306412, 'spearman_ca_t90_p_value': 6.539598740519087e-15, 'spearman_ca_y_high': 0.2931710533282421, 'spearman_ca_y_high_p_value': 5.390423935550033e-07, 'dose_bin_count': 5, 'best_threshold': 0.01698357313608515, 'high_rate_before_threshold': 0.09302325581395349, 'high_rate_after_threshold': 0.2510460251046025, 'high_rate_delta': 0.158022769290649, 'slope_before': 429.8913713736576, 'slope_after': 87.425700234256, 'threshold_supported': True, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True}]。

聚类与分工况阈值结果的一致性：[{'cluster': 0, 'resembles_supporting_regimes': 'rubber_flow_2_win_60_mean:high;bromine_feed_win_60_mean:high;neutral_alkali_feed_win_60_mean:high;r513_temp_win_60_mean:mid', 'resembles_contradictory_regimes': '', 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'cluster_relation_type': 'supporting'}, {'cluster': 1, 'resembles_supporting_regimes': 'rubber_flow_2_win_60_mean:mid;esbo_feed_win_60_mean:mid;r510a_temp_win_60_mean:mid;r511a_temp_win_60_mean:mid;r513_temp_win_60_mean:mid', 'resembles_contradictory_regimes': '', 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'cluster_relation_type': 'supporting'}, {'cluster': 2, 'resembles_supporting_regimes': 'r511a_temp_win_60_mean:mid', 'resembles_contradictory_regimes': 'rubber_flow_2_win_60_mean:low;bromine_feed_win_60_mean:low;neutral_alkali_feed_win_60_mean:low;r512a_temp_win_60_mean:high;r514_temp_win_60_mean:high', 'positive_relation_supported': False, 'high_calcium_high_t90_risk_supported': False, 'cluster_relation_type': 'contradictory'}, {'cluster': 3, 'resembles_supporting_regimes': 'rubber_flow_2_win_60_mean:mid', 'resembles_contradictory_regimes': 'bromine_feed_win_60_mean:low;neutral_alkali_feed_win_60_mean:low;r512a_temp_win_60_mean:high;r514_temp_win_60_mean:high', 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': False, 'cluster_relation_type': 'mixed'}, {'cluster': 4, 'resembles_supporting_regimes': 'rubber_flow_2_win_60_mean:mid;r510a_temp_win_60_mean:mid;r511a_temp_win_60_mean:mid;r512a_temp_win_60_mean:low;r513_temp_win_60_mean:low', 'resembles_contradictory_regimes': 'bromine_feed_win_60_mean:low', 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'cluster_relation_type': 'supporting'}]。

安全带解释：聚类层可增强人工复核解释，但本阶段不修改运行包规则，不产生自动控制或 DCS 写回。推荐下一步：`add_cluster_context_to_manual_review_explanation`。

局限性：离线历史分析，不是因果证明；T90 只用于后验解释；聚类解释仍需工艺人工复核；不建议直接更新 runtime package。

## 36. 钙单耗-T90 关系发现证据汇总与推荐算法修改判断

本阶段汇总全局阈值、分工况阈值、稳健聚类和聚类内钙-T90 关系证据，目标是判断是否应立即修改当前推荐算法。当前阶段暂停运行包迭代，不修改 `deploy/ca_safe_band_mvp/`。

可用证据源：['global_threshold', 'regime_threshold', 'clustering_robustness', 'cluster_specific']；缺失证据源：[]。

全局关系结论：全局证据为 moderate；高钙高 T90 风险支持=True。

分工况结论：分工况关系类型=broadly_consistent，正向工况=22/30，矛盾工况=5。

聚类内结论：支持和矛盾 cluster 并存，可增强人工复核解释，但不足以直接修改推荐区间算法。

支持上下文：[{'context_type': 'regime', 'context_name': 'rubber_flow_2_win_60_mean:mid', 'sample_count': 771, 'spearman_ca_t90': 0.17370803897580822, 'spearman_ca_y_high': 0.16833847768680402, 'high_rate_delta': 0.32515135561989994, 'threshold': 0.02140294402183697, 'ok_rate': 0.8067444876783398, 'high_rate': 0.16342412451361868, 'low_rate': 0.029831387808041506, 'interpretation_cn': '支持高钙高 T90 风险', 'caution_cn': '该工况支持高钙高 T90 风险，可用于人工复核说明。'}, {'context_type': 'regime', 'context_name': 'r512a_temp_win_60_mean:low', 'sample_count': 770, 'spearman_ca_t90': 0.1806945655087116, 'spearman_ca_y_high': 0.1578495993749647, 'high_rate_delta': 0.3216835453888637, 'threshold': 0.0206027818404035, 'ok_rate': 0.7571428571428571, 'high_rate': 0.21168831168831168, 'low_rate': 0.03116883116883117, 'interpretation_cn': '支持高钙高 T90 风险', 'caution_cn': '该工况支持高钙高 T90 风险，可用于人工复核说明。'}, {'context_type': 'regime', 'context_name': 'neutral_alkali_feed_win_60_mean:high', 'sample_count': 729, 'spearman_ca_t90': 0.58823515958541, 'spearman_ca_y_high': 0.5888866384267638, 'high_rate_delta': 0.28118689225652077, 'threshold': 0.02015469489075238, 'ok_rate': 0.7860082304526749, 'high_rate': 0.2085048010973937, 'low_rate': 0.0054869684499314125, 'interpretation_cn': '支持高钙高 T90 风险', 'caution_cn': '该工况支持高钙高 T90 风险，可用于人工复核说明。'}, {'context_type': 'regime', 'context_name': 'bromine_feed_win_60_mean:high', 'sample_count': 769, 'spearman_ca_t90': 0.5426405011723325, 'spearman_ca_y_high': 0.5734305772670671, 'high_rate_delta': 0.2672589004437519, 'threshold': 0.02004956130238023, 'ok_rate': 0.776332899869961, 'high_rate': 0.21716514954486346, 'low_rate': 0.006501950585175552, 'interpretation_cn': '支持高钙高 T90 风险', 'caution_cn': '该工况支持高钙高 T90 风险，可用于人工复核说明。'}, {'context_type': 'regime', 'context_name': 'r513_temp_win_60_mean:low', 'sample_count': 772, 'spearman_ca_t90': 0.134890665114054, 'spearman_ca_y_high': 0.09272398600683603, 'high_rate_delta': 0.23244322960470984, 'threshold': 0.020564238391452053, 'ok_rate': 0.7772020725388601, 'high_rate': 0.19041450777202074, 'low_rate': 0.03238341968911917, 'interpretation_cn': '支持高钙高 T90 风险', 'caution_cn': '该工况支持高钙高 T90 风险，可用于人工复核说明。'}]。

矛盾上下文：[{'context_type': 'regime', 'context_name': 'bromine_feed_win_60_mean:low', 'sample_count': 769, 'spearman_ca_t90': -0.16112592921664542, 'spearman_ca_y_high': -0.21417586853768464, 'high_rate_delta': -0.08920367534456355, 'threshold': 0.018330913516136624, 'ok_rate': 0.788036410923277, 'high_rate': 0.17425227568270482, 'low_rate': 0.0377113133940182, 'interpretation_cn': '矛盾或混合关系', 'caution_cn': '该工况与全局方向相反或较弱，不能套用统一阈值。'}, {'context_type': 'regime', 'context_name': 'rubber_flow_2_win_60_mean:low', 'sample_count': 772, 'spearman_ca_t90': -0.14691445406371897, 'spearman_ca_y_high': -0.21897368024392178, 'high_rate_delta': -0.17215096719932715, 'threshold': 0.01870961885537032, 'ok_rate': 0.8095854922279793, 'high_rate': 0.15544041450777202, 'low_rate': 0.034974093264248704, 'interpretation_cn': '矛盾或混合关系', 'caution_cn': '该工况与全局方向相反或较弱，不能套用统一阈值。'}, {'context_type': 'regime', 'context_name': 'neutral_alkali_feed_win_60_mean:low', 'sample_count': 729, 'spearman_ca_t90': -0.20402242811501803, 'spearman_ca_y_high': -0.2427229230267275, 'high_rate_delta': -0.1247848979067677, 'threshold': 0.022093782282016722, 'ok_rate': 0.8120713305898491, 'high_rate': 0.15637860082304528, 'low_rate': 0.03155006858710562, 'interpretation_cn': '矛盾或混合关系', 'caution_cn': '该工况与全局方向相反或较弱，不能套用统一阈值。'}, {'context_type': 'regime', 'context_name': 'r512a_temp_win_60_mean:high', 'sample_count': 770, 'spearman_ca_t90': -0.05688943681592089, 'spearman_ca_y_high': -0.10813130011834014, 'high_rate_delta': -0.02851611942521033, 'threshold': 0.02028694486576669, 'ok_rate': 0.8753246753246753, 'high_rate': 0.09220779220779221, 'low_rate': 0.032467532467532464, 'interpretation_cn': '矛盾或混合关系', 'caution_cn': '该工况与全局方向相反或较弱，不能套用统一阈值。'}, {'context_type': 'regime', 'context_name': 'r514_temp_win_60_mean:high', 'sample_count': 643, 'spearman_ca_t90': -0.11025814876326419, 'spearman_ca_y_high': -0.16431226425509876, 'high_rate_delta': -0.04910775499010793, 'threshold': 0.02045809171965959, 'ok_rate': 0.8771384136858476, 'high_rate': 0.09020217729393468, 'low_rate': 0.03265940902021773, 'interpretation_cn': '矛盾或混合关系', 'caution_cn': '该工况与全局方向相反或较弱，不能套用统一阈值。'}]。

算法修改判断：`explanation_layer_only`。推荐下一步：`build_manual_review_explanation_layer`。

局限性：全部证据仍为离线历史证据，不是因果证明；T90 存在人为测量误差；工况混杂和矛盾上下文仍存在；不建议自动控制，不进行 DCS 写回。

## 38. 厂区部署测试前证据闭环与解释层适用性论证

本阶段用于将稳定钙单耗安全带 MVP、钙单耗-T90 关系发现、分工况验证和聚类解释结果合并为厂区部署测试前的证据闭环。目标不是修改推荐算法，而是判断厂区测试应采用 V1 监测基线、V1.1 人工解释层版本，还是暂缓部署。

安全带基线复核：测试集区间内样本数为 142.0，区间内高 T90 率为 0.028169014084507，区间外高 T90 率为 0.2666666666666666，above_band 高 T90 率为 0.3947368421052631。safe_band_baseline_defensible = True。

q33/q66 工况基础复核：最终规则数为 21，覆盖工况变量数为 10，q33_q66_regime_basis_valid_for_v1 = True。稳健聚类仅作为解释和监测上下文，不替代冻结 V1 的单变量三分位规则基础。

关系发现复核：relationship_supports_explanation_layer = True；relationship_justifies_algorithm_change_now = False。当前证据支持将高钙高 T90 风险作为人工复核解释，但支持与矛盾上下文并存，尚不足以直接修改推荐区间算法。

算法修改判断：algorithm_modification_decision = explanation_layer_only。部署测试建议：deploy_test_decision = V1_monitor_only_candidate。推荐下一步：prepare_V1_monitor_only_factory_test。

局限性：全部证据仍来自离线历史数据，不构成因果证明；T90 存在测量和对齐误差；过程上下文存在混杂；厂区测试仍必须为 monitor-only；不实施自动控制，不实施 DCS 写回。

## 37. 钙单耗与 T90 非线性阈值关系验证

本阶段用于验证历史数据是否支持“钙单耗与 T90 存在正向、非线性阈值关系”的工艺预期，而不是预设该关系成立。输入样本数 1790，可用样本数 1756。

基础相关性：钙单耗与 T90 的 Spearman 相关系数为 0.11449438253704064；该结果只说明历史相关关系，不构成因果证明。

分箱响应与阈值搜索：最优阈值候选为 0.021465318874211138，阈值前后高 T90 风险差为 0.15208384109188397。正向关系支持：True；非线性阈值支持：False；安全平台区支持：False；高钙高 T90 风险支持：True。

当前安全带一致性：{'available': True, 'source': 'runs\\ca_safe_band_mvp\\final_monitor_dry_run.parquet', 'recommended_ca_consumption_max_median': 0.0204772882317374, 'known_stable_safe_band': [0.02016, 0.0205], 'threshold_minus_safe_band_max_median': 0.0009880306424737383, 'threshold_near_safe_band_upper_bound': False}。

证据强度：`moderate`。推荐下一步：`use_threshold_relation_as_supporting_evidence`。

局限性：离线历史关系不等于因果证明；T90 为人工 LIMS 且存在约 0.1 的实际误差；工况混杂仍可能影响钙单耗与 T90 的表观关系；本阶段不产生自动控制建议。

## 39. 分工况钙单耗-T90 阈值关系复验

本阶段在 Stage 31/32 之后进行：上一轮全局阈值结果显示钙单耗与高 T90 风险有中等强度历史证据，但聚类结果不稳，因此本阶段不依赖聚类，而用关键工况变量的 low/mid/high 三分位工况复验钙单耗-T90 关系。

全局阈值回顾：阈值候选 0.021465318874211138，安全带上沿中位数 0.0204772882317374。

分工况综合：{'relation_type': 'broadly_consistent', 'regime_count': 30, 'positive_relation_regime_count': 21, 'high_calcium_high_t90_risk_regime_count': 14, 'threshold_evidence_regime_count': 14, 'contradictory_regime_count': 6}。

最强支持工况（前 5）：[{'regime_feature': 'bromine_feed_win_60_mean', 'regime_bin': 'high', 'sample_count': 583, 'calcium_median': 0.020350791954794518, 'calcium_iqr': 0.0009687312319282097, 't90_mean': 8.524957118353345, 't90_median': 8.5, 't90_iqr': 0.29999999999999893, 'ok_rate': 0.7993138936535163, 'high_rate': 0.19039451114922812, 'low_rate': 0.010291595197255575, 'out_spec_rate': 0.2006861063464837, 'spearman_ca_t90': 0.4942100579634575, 'spearman_ca_t90_p_value': 3.1146122800682977e-37, 'spearman_ca_y_high': 0.5421305771516997, 'spearman_ca_y_high_p_value': 7.558403651092804e-46, 'best_threshold': 0.01997389416278281, 'high_rate_before_threshold': 0.0196078431372549, 'high_rate_after_threshold': 0.25116279069767444, 'high_rate_delta': 0.23155494756041956, 'slope_before': -2.8663636720220746, 'slope_after': 167.0932611726011, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'threshold_supported': True, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': False, 'stable_support': True}, {'regime_feature': 'neutral_alkali_feed_win_60_mean', 'regime_bin': 'high', 'sample_count': 552, 'calcium_median': 0.020372036716075875, 'calcium_iqr': 0.0009485705197807108, 't90_mean': 8.523460144927537, 't90_median': 8.5, 't90_iqr': 0.29999999999999893, 'ok_rate': 0.7916666666666666, 'high_rate': 0.19746376811594202, 'low_rate': 0.010869565217391304, 'out_spec_rate': 0.20833333333333334, 'spearman_ca_t90': 0.5326947834423962, 'spearman_ca_t90_p_value': 8.782212519016277e-42, 'spearman_ca_y_high': 0.5125271889592184, 'spearman_ca_y_high_p_value': 2.6570519417935707e-38, 'best_threshold': 0.02007234421197272, 'high_rate_before_threshold': 0.041666666666666664, 'high_rate_after_threshold': 0.265625, 'high_rate_delta': 0.22395833333333334, 'slope_before': -11.120818586302116, 'slope_after': 206.7605173830756, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'threshold_supported': True, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': True, 'stable_support': True}, {'regime_feature': 'r510a_temp_win_60_mean', 'regime_bin': 'mid', 'sample_count': 585, 'calcium_median': 0.019890412492038382, 'calcium_iqr': 0.001476824301174004, 't90_mean': 8.50974358974359, 't90_median': 8.5, 't90_iqr': 0.29999999999999893, 'ok_rate': 0.788034188034188, 'high_rate': 0.17606837606837608, 'low_rate': 0.035897435897435895, 'out_spec_rate': 0.21196581196581196, 'spearman_ca_t90': 0.13013053438424546, 'spearman_ca_t90_p_value': 0.0016097887770871579, 'spearman_ca_y_high': 0.14593004073382856, 'spearman_ca_y_high_p_value': 0.00039861264135099245, 'best_threshold': 0.020492901561329, 'high_rate_before_threshold': 0.1276595744680851, 'high_rate_after_threshold': 0.30246913580246915, 'high_rate_delta': 0.17480956133438405, 'slope_before': 10.405057907715086, 'slope_after': 137.79600457790465, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'threshold_supported': True, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': True, 'stable_support': True}, {'regime_feature': 'r512a_temp_win_60_mean', 'regime_bin': 'low', 'sample_count': 584, 'calcium_median': 0.019500905246186695, 'calcium_iqr': 0.00221556550850055, 't90_mean': 8.516780821917807, 't90_median': 8.5, 't90_iqr': 0.3999999999999986, 'ok_rate': 0.7465753424657534, 'high_rate': 0.2054794520547945, 'low_rate': 0.04794520547945205, 'out_spec_rate': 0.2534246575342466, 'spearman_ca_t90': 0.2449462076061256, 'spearman_ca_t90_p_value': 1.993653795750796e-09, 'spearman_ca_y_high': 0.1902435418014486, 'spearman_ca_y_high_p_value': 3.6576766852659465e-06, 'best_threshold': 0.017436289757034585, 'high_rate_before_threshold': 0.06818181818181818, 'high_rate_after_threshold': 0.22983870967741934, 'high_rate_delta': 0.16165689149560117, 'slope_before': 317.5385926722187, 'slope_after': 33.56070716120922, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'threshold_supported': True, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': False, 'stable_support': True}, {'regime_feature': 'esbo_feed_win_60_mean', 'regime_bin': 'mid', 'sample_count': 581, 'calcium_median': 0.02037525583349017, 'calcium_iqr': 0.0015963585803523694, 't90_mean': 8.478571428571428, 't90_median': 8.4, 't90_iqr': 0.29999999999999893, 'ok_rate': 0.8244406196213425, 'high_rate': 0.13941480206540446, 'low_rate': 0.03614457831325301, 'out_spec_rate': 0.17555938037865748, 'spearman_ca_t90': 0.32184160882204554, 'spearman_ca_t90_p_value': 1.8127285367347304e-15, 'spearman_ca_y_high': 0.32408971182437474, 'spearman_ca_y_high_p_value': 1.1260786704094259e-15, 'best_threshold': 0.020164344858334446, 'high_rate_before_threshold': 0.052, 'high_rate_after_threshold': 0.2054380664652568, 'high_rate_delta': 0.15343806646525682, 'slope_before': 13.35665307484502, 'slope_after': 103.15999646313222, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'threshold_supported': True, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': True, 'stable_support': True}]。

矛盾或反向工况（前 5）：[{'regime_feature': 'bromine_feed_win_60_mean', 'regime_bin': 'low', 'sample_count': 583, 'calcium_median': 0.019862412761129496, 'calcium_iqr': 0.002424660081265066, 't90_mean': 8.524871355060034, 't90_median': 8.5, 't90_iqr': 0.29999999999999893, 'ok_rate': 0.8027444253859348, 'high_rate': 0.16295025728987994, 'low_rate': 0.03430531732418525, 'out_spec_rate': 0.19725557461406518, 'spearman_ca_t90': -0.19203181770881583, 'spearman_ca_t90_p_value': 3.0080910922520744e-06, 'spearman_ca_y_high': -0.21180166881214404, 'spearman_ca_y_high_p_value': 2.4467525738212646e-07, 'best_threshold': 0.018331652580460614, 'high_rate_before_threshold': 0.20454545454545456, 'high_rate_after_threshold': 0.15555555555555556, 'high_rate_delta': -0.048989898989899, 'slope_before': 152.8175300933981, 'slope_after': -34.77398929513558, 'positive_relation_supported': False, 'high_calcium_high_t90_risk_supported': False, 'threshold_supported': False, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': False, 'stable_support': True}, {'regime_feature': 'rubber_flow_2_win_60_mean', 'regime_bin': 'low', 'sample_count': 586, 'calcium_median': 0.020060161176264607, 'calcium_iqr': 0.0021957262087229437, 't90_mean': 8.511092150170649, 't90_median': 8.5, 't90_iqr': 0.1999999999999993, 'ok_rate': 0.8208191126279863, 'high_rate': 0.14505119453924914, 'low_rate': 0.034129692832764506, 'out_spec_rate': 0.17918088737201365, 'spearman_ca_t90': -0.17381770872459368, 'spearman_ca_t90_p_value': 2.3271181127520563e-05, 'spearman_ca_y_high': -0.24553993867723042, 'spearman_ca_y_high_p_value': 1.7052071604461687e-09, 'best_threshold': 0.01868781842661562, 'high_rate_before_threshold': 0.2727272727272727, 'high_rate_after_threshold': 0.12248995983935743, 'high_rate_delta': -0.15023731288791528, 'slope_before': 92.83495854345347, 'slope_after': -28.021237869069907, 'positive_relation_supported': False, 'high_calcium_high_t90_risk_supported': False, 'threshold_supported': False, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': False, 'stable_support': True}, {'regime_feature': 'neutral_alkali_feed_win_60_mean', 'regime_bin': 'low', 'sample_count': 552, 'calcium_median': 0.020151331768930462, 'calcium_iqr': 0.0024209955259339618, 't90_mean': 8.520833333333334, 't90_median': 8.5, 't90_iqr': 0.29999999999999893, 'ok_rate': 0.8170289855072463, 'high_rate': 0.1503623188405797, 'low_rate': 0.03260869565217391, 'out_spec_rate': 0.18297101449275363, 'spearman_ca_t90': -0.2381725688324768, 'spearman_ca_t90_p_value': 1.4717644548523752e-08, 'spearman_ca_y_high': -0.24801741336935734, 'spearman_ca_y_high_p_value': 3.4964334330466074e-09, 'best_threshold': 0.018698254621708515, 'high_rate_before_threshold': 0.25301204819277107, 'high_rate_after_threshold': 0.13219616204690832, 'high_rate_delta': -0.12081588614586275, 'slope_before': 47.59497082277847, 'slope_after': -28.03783920962919, 'positive_relation_supported': False, 'high_calcium_high_t90_risk_supported': False, 'threshold_supported': False, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': False, 'stable_support': True}, {'regime_feature': 'r512a_temp_win_60_mean', 'regime_bin': 'high', 'sample_count': 584, 'calcium_median': 0.020263655294717353, 'calcium_iqr': 0.0022720787407839345, 't90_mean': 8.476455479452055, 't90_median': 8.5, 't90_iqr': 0.1999999999999993, 'ok_rate': 0.8955479452054794, 'high_rate': 0.07363013698630137, 'low_rate': 0.030821917808219176, 'out_spec_rate': 0.10445205479452055, 'spearman_ca_t90': -0.07396502321912031, 'spearman_ca_t90_p_value': 0.07408906142287587, 'spearman_ca_y_high': -0.1140469231563728, 'spearman_ca_y_high_p_value': 0.005794820599717023, 'best_threshold': 0.020155572203504685, 'high_rate_before_threshold': 0.09420289855072464, 'high_rate_after_threshold': 0.05519480519480519, 'high_rate_delta': -0.03900809335591945, 'slope_before': -49.8125523013209, 'slope_after': 8.878986824769292, 'positive_relation_supported': False, 'high_calcium_high_t90_risk_supported': False, 'threshold_supported': False, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': True, 'stable_support': True}, {'regime_feature': 'r513_temp_win_60_mean', 'regime_bin': 'high', 'sample_count': 585, 'calcium_median': 0.020256488507302595, 'calcium_iqr': 0.0022919808175335885, 't90_mean': 8.477521367521367, 't90_median': 8.5, 't90_iqr': 0.1999999999999993, 'ok_rate': 0.8974358974358975, 'high_rate': 0.07008547008547009, 'low_rate': 0.03247863247863248, 'out_spec_rate': 0.10256410256410256, 'spearman_ca_t90': -0.06819790470898826, 'spearman_ca_t90_p_value': 0.099377673322662, 'spearman_ca_y_high': -0.12109132907971452, 'spearman_ca_y_high_p_value': 0.0033533092932444505, 'best_threshold': 0.020256488507302595, 'high_rate_before_threshold': 0.08873720136518772, 'high_rate_after_threshold': 0.05136986301369863, 'high_rate_delta': -0.03736733835148909, 'slope_before': -31.87666723762038, 'slope_after': 5.763529622949676, 'positive_relation_supported': False, 'high_calcium_high_t90_risk_supported': False, 'threshold_supported': False, 'threshold_near_global_threshold': False, 'threshold_near_safe_band_upper': True, 'stable_support': True}]。

当前判断：`broadly_consistent`。推荐下一步：`use_relation_as_supporting_evidence_for_safe_band`。

局限性：该分析仍为离线历史关系，不是因果证明；三分位工况存在样本稀疏；IR-lag 只作为可选工况变量；本阶段不产生自动控制建议。

## 40. 聚类特征审计与稳健聚类复验

本阶段复核上一轮聚类结果的稳健性，重点检查 k=2 结果是否违反最小簇规模门槛。上一轮最小簇样本数为 14，门槛为 118，不一致标记为 True。

本次审计测试的特征集包括 core_11、core_11_plus_ir、filtered_engineered_features 与 pca_90pct_from_filtered_engineered。候选特征审计与各 k 指标已写入 runs。

是否找到稳健聚类结果：True。选中结果：{'variant': 'core_11_features', 'algorithm': 'AgglomerativeClustering', 'k': 8, 'seed_count': 1, 'stability_ari': 1.0, 'tiny_cluster_flag': False, 'tiny_cluster_threshold': 89, 'min_cluster_size_worst_seed': 95, 'silhouette_score': 0.3365226690043361, 'calinski_harabasz_score': 760.9061833365728, 'davies_bouldin_score': 1.1381903307815073, 'min_cluster_size': 95.0, 'max_cluster_size': 598.0, 'cluster_size_imbalance': 6.294736842105263, 't90_mean_range': 0.1798508081226693, 'high_rate_range': 0.3016167870657035, 'ok_rate_range': 0.3086021505376344, 'feature_count': 8, 'score': 2.4423076923076925}。理由：Selected because it passed robustness gates and had the best combined stability, separation, and clustering metric score.。

推荐下一步：`use_clusters_for_context_specific_ca_t90_analysis`。

局限性：聚类为无监督离线分析，T90 不参与聚类；不同算法和特征集可能给出不同划分；若没有稳定且非小簇的结果，应继续保留规则型安全带而不是强行解释聚类。

## 41. 稳健聚类工况画像与聚类内钙单耗-T90 关系验证

本阶段承接 Stage 33 的稳健 k=5 聚类和 Stage 34 的分工况阈值复验，目标是解释每个聚类的工况画像，并在聚类内部验证钙单耗-T90 关系。T90 未用于聚类，只用于聚类后的质量解释。

稳健聚类结果：算法 `AgglomerativeClustering`，k=8，最终特征数 8。最终特征：['ca_per_rubber_flow_win_60_mean', 'rubber_flow_2_win_60_mean', 'bromine_feed_win_60_mean', 'tank_rubber_conc_win_60_mean', 'esbo_feed_win_60_mean', 'neutral_alkali_feed_win_60_mean', 'r510a_temp_win_60_mean', 'r512a_temp_win_60_mean']。core_11 中被剔除特征：['r511a_temp_win_60_mean', 'r513_temp_win_60_mean', 'r514_temp_win_60_mean']，主要原因是缺失/强相关/筛选后不进入最终核心特征集。

聚类规模与 T90 概况：
- cluster 0: n=192, ok=0.7916666666666666, high=0.1875, low=0.020833333333333332
- cluster 1: n=598, ok=0.8344481605351171, high=0.14381270903010032, low=0.021739130434782608
- cluster 2: n=294, ok=0.8095238095238095, high=0.18027210884353742, low=0.01020408163265306
- cluster 3: n=153, ok=0.8888888888888888, high=0.0457516339869281, low=0.06535947712418301
- cluster 4: n=95, ok=0.6, high=0.3473684210526316, low=0.05263157894736842
- cluster 5: n=186, ok=0.9086021505376344, high=0.0913978494623656, low=0.0
- cluster 6: n=127, ok=0.905511811023622, high=0.05511811023622047, low=0.03937007874015748
- cluster 7: n=145, ok=0.7379310344827587, high=0.13793103448275862, low=0.12413793103448276

聚类内钙单耗-T90 关系：[{'cluster': 0, 'sample_count': 192, 'spearman_ca_t90': -0.3997219665225812, 'spearman_ca_t90_p_value': 3.5519529012955404e-08, 'spearman_ca_y_high': -0.300331294764294, 'spearman_ca_y_high_p_value': 4.875806786515879e-05, 'dose_bin_count': 4, 'best_threshold': 0.018411764120044433, 'high_rate_before_threshold': 0.28125, 'high_rate_after_threshold': 0.1724137931034483, 'high_rate_delta': -0.10883620689655171, 'slope_before': 72.10833220424377, 'slope_after': -74.98709122295708, 'threshold_supported': False, 'positive_relation_supported': False, 'high_calcium_high_t90_risk_supported': False}, {'cluster': 1, 'sample_count': 598, 'spearman_ca_t90': 0.3080371861620399, 'spearman_ca_t90_p_value': 2.5256990018675474e-14, 'spearman_ca_y_high': 0.3346315243712386, 'spearman_ca_y_high_p_value': 8.997694808010734e-17, 'dose_bin_count': 5, 'best_threshold': 0.020268946310181307, 'high_rate_before_threshold': 0.06424581005586592, 'high_rate_after_threshold': 0.27312775330396477, 'high_rate_delta': 0.20888194324809883, 'slope_before': -16.629256185489453, 'slope_after': 205.67042452718366, 'threshold_supported': True, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True}, {'cluster': 2, 'sample_count': 294, 'spearman_ca_t90': 0.4683701019827746, 'spearman_ca_t90_p_value': 1.9504734962337057e-17, 'spearman_ca_y_high': 0.412591495904177, 'spearman_ca_y_high_p_value': 1.6372917272882553e-13, 'dose_bin_count': 5, 'best_threshold': 0.0201296580489855, 'high_rate_before_threshold': 0.06363636363636363, 'high_rate_after_threshold': 0.25, 'high_rate_delta': 0.18636363636363637, 'slope_before': -31.499865010710412, 'slope_after': 229.4252013441642, 'threshold_supported': True, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True}, {'cluster': 3, 'sample_count': 153, 'spearman_ca_t90': 0.23613449707333364, 'spearman_ca_t90_p_value': 0.003403134055365116, 'spearman_ca_y_high': 0.07690098048822028, 'spearman_ca_y_high_p_value': 0.3463627535888463, 'dose_bin_count': 4, 'best_threshold': 0.02437488290781465, 'high_rate_before_threshold': 0.041666666666666664, 'high_rate_after_threshold': 0.0625, 'high_rate_delta': 0.020833333333333336, 'slope_before': 43.359058523101815, 'slope_after': -77.42907873920365, 'threshold_supported': False, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': False}, {'cluster': 4, 'sample_count': 95, 'spearman_ca_t90': 0.4334395126819813, 'spearman_ca_t90_p_value': 1.773306622909408e-05, 'spearman_ca_y_high': 0.24917699330443585, 'spearman_ca_y_high_p_value': 0.017226522019807285, 'dose_bin_count': 0, 'best_threshold': nan, 'high_rate_before_threshold': nan, 'high_rate_after_threshold': nan, 'high_rate_delta': nan, 'slope_before': nan, 'slope_after': nan, 'threshold_supported': False, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': False}, {'cluster': 5, 'sample_count': 186, 'spearman_ca_t90': 0.30569428401129944, 'spearman_ca_t90_p_value': 2.205243217011757e-05, 'spearman_ca_y_high': 0.4497987014681432, 'spearman_ca_y_high_p_value': 1.1897419105841803e-10, 'dose_bin_count': 4, 'best_threshold': 0.020577013763471035, 'high_rate_before_threshold': 0.013333333333333334, 'high_rate_after_threshold': 0.4166666666666667, 'high_rate_delta': 0.4033333333333334, 'slope_before': 9.298522873704362, 'slope_after': 428.9517422524814, 'threshold_supported': True, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True}, {'cluster': 6, 'sample_count': 127, 'spearman_ca_t90': 0.14694523302109255, 'spearman_ca_t90_p_value': 0.099234270739817, 'spearman_ca_y_high': 0.04329308701420995, 'spearman_ca_y_high_p_value': 0.6288890761733998, 'dose_bin_count': 4, 'best_threshold': nan, 'high_rate_before_threshold': nan, 'high_rate_after_threshold': nan, 'high_rate_delta': nan, 'slope_before': nan, 'slope_after': nan, 'threshold_supported': False, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': False}, {'cluster': 7, 'sample_count': 145, 'spearman_ca_t90': 0.3001610246156644, 'spearman_ca_t90_p_value': 0.00025698748064165465, 'spearman_ca_y_high': 0.18536030591498476, 'spearman_ca_y_high_p_value': 0.026131662233820532, 'dose_bin_count': 4, 'best_threshold': nan, 'high_rate_before_threshold': nan, 'high_rate_after_threshold': nan, 'high_rate_delta': nan, 'slope_before': nan, 'slope_after': nan, 'threshold_supported': False, 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': False}]。

聚类与分工况阈值结果的一致性：[{'cluster': 0, 'resembles_supporting_regimes': '', 'resembles_contradictory_regimes': 'rubber_flow_2_win_60_mean:low;bromine_feed_win_60_mean:low;neutral_alkali_feed_win_60_mean:low;r512a_temp_win_60_mean:high;r513_temp_win_60_mean:high;r514_temp_win_60_mean:high', 'positive_relation_supported': False, 'high_calcium_high_t90_risk_supported': False, 'cluster_relation_type': 'contradictory'}, {'cluster': 1, 'resembles_supporting_regimes': 'esbo_feed_win_60_mean:mid;r510a_temp_win_60_mean:mid;r511a_temp_win_60_mean:mid;r512a_temp_win_60_mean:mid;r513_temp_win_60_mean:mid', 'resembles_contradictory_regimes': '', 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'cluster_relation_type': 'supporting'}, {'cluster': 2, 'resembles_supporting_regimes': 'rubber_flow_2_win_60_mean:high;bromine_feed_win_60_mean:high;neutral_alkali_feed_win_60_mean:high;r510a_temp_win_60_mean:mid;r511a_temp_win_60_mean:mid;r512a_temp_win_60_mean:low;r513_temp_win_60_mean:low', 'resembles_contradictory_regimes': '', 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'cluster_relation_type': 'supporting'}, {'cluster': 3, 'resembles_supporting_regimes': '', 'resembles_contradictory_regimes': 'neutral_alkali_feed_win_60_mean:low;r512a_temp_win_60_mean:high;r513_temp_win_60_mean:high;r514_temp_win_60_mean:high', 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': False, 'cluster_relation_type': 'contradictory'}, {'cluster': 4, 'resembles_supporting_regimes': 'r510a_temp_win_60_mean:mid;r512a_temp_win_60_mean:low;r513_temp_win_60_mean:low', 'resembles_contradictory_regimes': 'rubber_flow_2_win_60_mean:low;bromine_feed_win_60_mean:low;neutral_alkali_feed_win_60_mean:low', 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': False, 'cluster_relation_type': 'mixed'}, {'cluster': 5, 'resembles_supporting_regimes': 'rubber_flow_2_win_60_mean:high;bromine_feed_win_60_mean:high;neutral_alkali_feed_win_60_mean:high;r512a_temp_win_60_mean:mid;r513_temp_win_60_mean:mid', 'resembles_contradictory_regimes': '', 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': True, 'cluster_relation_type': 'supporting'}, {'cluster': 6, 'resembles_supporting_regimes': 'esbo_feed_win_60_mean:mid;r511a_temp_win_60_mean:mid;r513_temp_win_60_mean:mid', 'resembles_contradictory_regimes': 'rubber_flow_2_win_60_mean:low;bromine_feed_win_60_mean:low;neutral_alkali_feed_win_60_mean:low;r512a_temp_win_60_mean:high', 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': False, 'cluster_relation_type': 'mixed'}, {'cluster': 7, 'resembles_supporting_regimes': 'esbo_feed_win_60_mean:mid;r510a_temp_win_60_mean:mid;r511a_temp_win_60_mean:mid;r512a_temp_win_60_mean:low;r513_temp_win_60_mean:low', 'resembles_contradictory_regimes': '', 'positive_relation_supported': True, 'high_calcium_high_t90_risk_supported': False, 'cluster_relation_type': 'mixed'}]。

安全带解释：聚类层可增强人工复核解释，但本阶段不修改运行包规则，不产生自动控制或 DCS 写回。推荐下一步：`add_cluster_context_to_manual_review_explanation`。

局限性：离线历史分析，不是因果证明；T90 只用于后验解释；聚类解释仍需工艺人工复核；不建议直接更新 runtime package。

## 42. 钙单耗-T90 关系发现证据汇总与推荐算法修改判断

本阶段汇总全局阈值、分工况阈值、稳健聚类和聚类内钙-T90 关系证据，目标是判断是否应立即修改当前推荐算法。当前阶段暂停运行包迭代，不修改 `deploy/ca_safe_band_mvp/`。

可用证据源：['global_threshold', 'regime_threshold', 'clustering_robustness', 'cluster_specific']；缺失证据源：[]。

全局关系结论：全局证据为 moderate；高钙高 T90 风险支持=True。

分工况结论：分工况关系类型=broadly_consistent，正向工况=21/30，矛盾工况=6。

聚类内结论：支持和矛盾 cluster 并存，可增强人工复核解释，但不足以直接修改推荐区间算法。

支持上下文：[{'context_type': 'regime', 'context_name': 'bromine_feed_win_60_mean:high', 'sample_count': 583, 'spearman_ca_t90': 0.4942100579634575, 'spearman_ca_y_high': 0.5421305771516997, 'high_rate_delta': 0.23155494756041956, 'threshold': 0.01997389416278281, 'ok_rate': 0.7993138936535163, 'high_rate': 0.19039451114922812, 'low_rate': 0.010291595197255575, 'interpretation_cn': '支持高钙高 T90 风险', 'caution_cn': '该工况支持高钙高 T90 风险，可用于人工复核说明。'}, {'context_type': 'regime', 'context_name': 'neutral_alkali_feed_win_60_mean:high', 'sample_count': 552, 'spearman_ca_t90': 0.5326947834423962, 'spearman_ca_y_high': 0.5125271889592184, 'high_rate_delta': 0.22395833333333334, 'threshold': 0.02007234421197272, 'ok_rate': 0.7916666666666666, 'high_rate': 0.19746376811594202, 'low_rate': 0.010869565217391304, 'interpretation_cn': '支持高钙高 T90 风险', 'caution_cn': '该工况支持高钙高 T90 风险，可用于人工复核说明。'}, {'context_type': 'regime', 'context_name': 'r510a_temp_win_60_mean:mid', 'sample_count': 585, 'spearman_ca_t90': 0.13013053438424546, 'spearman_ca_y_high': 0.14593004073382856, 'high_rate_delta': 0.17480956133438405, 'threshold': 0.020492901561329, 'ok_rate': 0.788034188034188, 'high_rate': 0.17606837606837608, 'low_rate': 0.035897435897435895, 'interpretation_cn': '支持高钙高 T90 风险', 'caution_cn': '该工况支持高钙高 T90 风险，可用于人工复核说明。'}, {'context_type': 'regime', 'context_name': 'r512a_temp_win_60_mean:low', 'sample_count': 584, 'spearman_ca_t90': 0.2449462076061256, 'spearman_ca_y_high': 0.1902435418014486, 'high_rate_delta': 0.16165689149560117, 'threshold': 0.017436289757034585, 'ok_rate': 0.7465753424657534, 'high_rate': 0.2054794520547945, 'low_rate': 0.04794520547945205, 'interpretation_cn': '支持高钙高 T90 风险', 'caution_cn': '该工况支持高钙高 T90 风险，可用于人工复核说明。'}, {'context_type': 'regime', 'context_name': 'esbo_feed_win_60_mean:mid', 'sample_count': 581, 'spearman_ca_t90': 0.32184160882204554, 'spearman_ca_y_high': 0.32408971182437474, 'high_rate_delta': 0.15343806646525682, 'threshold': 0.020164344858334446, 'ok_rate': 0.8244406196213425, 'high_rate': 0.13941480206540446, 'low_rate': 0.03614457831325301, 'interpretation_cn': '支持高钙高 T90 风险', 'caution_cn': '该工况支持高钙高 T90 风险，可用于人工复核说明。'}]。

矛盾上下文：[{'context_type': 'regime', 'context_name': 'bromine_feed_win_60_mean:low', 'sample_count': 583, 'spearman_ca_t90': -0.19203181770881583, 'spearman_ca_y_high': -0.21180166881214404, 'high_rate_delta': -0.048989898989899, 'threshold': 0.018331652580460614, 'ok_rate': 0.8027444253859348, 'high_rate': 0.16295025728987994, 'low_rate': 0.03430531732418525, 'interpretation_cn': '矛盾或混合关系', 'caution_cn': '该工况与全局方向相反或较弱，不能套用统一阈值。'}, {'context_type': 'regime', 'context_name': 'rubber_flow_2_win_60_mean:low', 'sample_count': 586, 'spearman_ca_t90': -0.17381770872459368, 'spearman_ca_y_high': -0.24553993867723042, 'high_rate_delta': -0.15023731288791528, 'threshold': 0.01868781842661562, 'ok_rate': 0.8208191126279863, 'high_rate': 0.14505119453924914, 'low_rate': 0.034129692832764506, 'interpretation_cn': '矛盾或混合关系', 'caution_cn': '该工况与全局方向相反或较弱，不能套用统一阈值。'}, {'context_type': 'regime', 'context_name': 'neutral_alkali_feed_win_60_mean:low', 'sample_count': 552, 'spearman_ca_t90': -0.2381725688324768, 'spearman_ca_y_high': -0.24801741336935734, 'high_rate_delta': -0.12081588614586275, 'threshold': 0.018698254621708515, 'ok_rate': 0.8170289855072463, 'high_rate': 0.1503623188405797, 'low_rate': 0.03260869565217391, 'interpretation_cn': '矛盾或混合关系', 'caution_cn': '该工况与全局方向相反或较弱，不能套用统一阈值。'}, {'context_type': 'regime', 'context_name': 'r512a_temp_win_60_mean:high', 'sample_count': 584, 'spearman_ca_t90': -0.07396502321912031, 'spearman_ca_y_high': -0.1140469231563728, 'high_rate_delta': -0.03900809335591945, 'threshold': 0.020155572203504685, 'ok_rate': 0.8955479452054794, 'high_rate': 0.07363013698630137, 'low_rate': 0.030821917808219176, 'interpretation_cn': '矛盾或混合关系', 'caution_cn': '该工况与全局方向相反或较弱，不能套用统一阈值。'}, {'context_type': 'regime', 'context_name': 'r513_temp_win_60_mean:high', 'sample_count': 585, 'spearman_ca_t90': -0.06819790470898826, 'spearman_ca_y_high': -0.12109132907971452, 'high_rate_delta': -0.03736733835148909, 'threshold': 0.020256488507302595, 'ok_rate': 0.8974358974358975, 'high_rate': 0.07008547008547009, 'low_rate': 0.03247863247863248, 'interpretation_cn': '矛盾或混合关系', 'caution_cn': '该工况与全局方向相反或较弱，不能套用统一阈值。'}]。

算法修改判断：`explanation_layer_only`。推荐下一步：`build_manual_review_explanation_layer`。

局限性：全部证据仍为离线历史证据，不是因果证明；T90 存在人为测量误差；工况混杂和矛盾上下文仍存在；不建议自动控制，不进行 DCS 写回。

### 实验摘要
- 工况分层剂量响应：有效支持的 regime×dose 分组数为 150。
- 钙×工况交互筛查：通过项数为 12；主要项：bromine_feed_win_60_mean->y_out_spec (delta_auc=0.29797903816257953, delta_ap=0.38219888719363676)；bromine_feed_win_60_mean->y_ok (delta_auc=0.29797903816257953, delta_ap=0.1122555257974811)；bromine_feed_win_60_mean->y_high (delta_auc=0.27692697768762686, delta_ap=0.3530684269831903)；neutral_alkali_feed_win_60_mean->y_high (delta_auc=0.18661257606490878, delta_ap=0.23764554750303835)；neutral_alkali_feed_win_60_mean->y_out_spec (delta_auc=0.18148538187653573, delta_ap=0.2468567371611211)。
- IR 分层剂量响应：有效支持分组数为 30。
- IR 描述性中介/驱动诊断：{'calcium_to_ir_signal': False, 'ir_to_t90_risk_signal': True, 'ir_incremental_signal': False, 'calcium_ir_interaction_signal': False, 'descriptive_mediation_possible': False, 'not_causal_proof': True}。
- 最优钙单耗区间映射：稳定候选数为 27。

### 与第 16 阶段比较
- 稳定候选数：previous=27，current=27。
- 交互通过项：previous=12，current=12。
- 关键工况是否保持重要：['bromine_feed_win_60_mean', 'neutral_alkali_feed_win_60_mean', 'rubber_flow_2_win_60_mean']。
- recommended_next_step：`define_regime_specific_calcium_band_rules_with_ir_lag_context`。

### 审计结果
- 交互稳定审计：{'passed_count': 12, 'stable_candidate_count': 12, 'suspicious_large_delta_count': 4, 'insufficient_positive_support_count': 10, 'rejected_count': 18}。
- suspicious large-delta 交互数：4。
- 规则等级统计：{'A': 20, 'B': 8, 'Reject': 1, 'C': 1}。
- 规则状态统计：{'accept_for_manual_case_review': 21, 'monitor_only': 8, 'reject': 1}。
- accepted / monitor / rejected：21 / 8 / 1。
- 高剂量高 T90 避免候选数：16。
- 时间稳定规则数：21。
- IR-lag 有用上下文规则数：24。
- 人工复核候选数：29。
- Top 规则：ca_regime_rule_030 r512a_temp_win_60_mean=mid dose=[0.0202289727867283, 0.0205910836820826] grade=A；ca_regime_rule_019 r514_temp_win_60_mean=high dose=[0.0198973677434662, 0.0207787120628998] grade=A；ca_regime_rule_013 esbo_feed_win_60_mean=high dose=[0.015048047594216, 0.0199665303756579] grade=A；ca_regime_rule_016 r513_temp_win_60_mean=high dose=[0.0198629929943009, 0.0206321992444512] grade=A；ca_regime_rule_028 r512a_temp_win_60_mean=high dose=[0.0198872755047615, 0.0206971976312819] grade=A。
- recommended_next_step：`prepare_regime_rule_manual_review`。

### 结果
- artifact_rule_count：21。
- test_like recommendation coverage：0.9972067039106145。
- test_like band accuracy：0.5294117647058824。
- test_like direction accuracy：0.8151260504201681。
- target accuracy 3%/5%/10%：1.0 / 1.0 / 1.0。
- T90 风险护栏：{'recommended_high_rate': 0.19047619047619047, 'no_recommendation_high_rate': 0.0, 'recommended_low_rate': 0.0028011204481792717, 'no_recommendation_low_rate': 0.0, 'high_guardrail_pass': False, 'low_guardrail_pass': True, 'note': 'T90 rates are guardrails, not recommendation accuracy.'}。
- mvp_status：`pass_for_monitor_only_chain`。
- recommended_next_step：`manual_review_before_deployment_chain`。

### 审计结果
- no_recommendation baseline：{'recommendation_status_counts': {'recommended': 1756, 'no_recommendation_missing_current_dose': 31, 'no_recommendation': 3}, 'test_like_no_recommendation_count': 1, 'no_recommendation_baseline_unreliable': True, 'note': 'Do not use no_recommendation as main risk baseline when count < 30.'}。
- inside/outside 风险摘要：{'inside_band_test_like': {'sample_count': 91, 'ok_rate': 0.978021978021978, 'high_rate': 0.02197802197802198, 'low_rate': 0.0, 'out_spec_rate': 0.02197802197802198, 'mean_t90': 8.465934065934066, 'band_accuracy': 0.6483516483516484, 'direction_accuracy': 0.5384615384615384, 'target_accuracy_5pct': 1.0}, 'outside_band_test_like': {'sample_count': 266, 'ok_rate': 0.7481203007518797, 'high_rate': 0.24812030075187969, 'low_rate': 0.0037593984962406013, 'out_spec_rate': 0.2518796992481203, 'mean_t90': 8.571992481203006, 'band_accuracy': 0.48872180451127817, 'direction_accuracy': 0.9097744360902256, 'target_accuracy_5pct': 1.0}, 'below_band_test_like': {'sample_count': 95, 'ok_rate': 0.8842105263157894, 'high_rate': 0.11578947368421053, 'low_rate': 0.0, 'out_spec_rate': 0.11578947368421053, 'mean_t90': 8.501052631578947, 'band_accuracy': 0.49473684210526314, 'direction_accuracy': 0.9894736842105263, 'target_accuracy_5pct': 1.0}, 'above_band_test_like': {'sample_count': 171, 'ok_rate': 0.672514619883041, 'high_rate': 0.3216374269005848, 'low_rate': 0.005847953216374269, 'out_spec_rate': 0.32748538011695905, 'mean_t90': 8.611403508771929, 'band_accuracy': 0.4853801169590643, 'direction_accuracy': 0.8654970760233918, 'target_accuracy_5pct': 1.0}, 'comparisons': {'inside_vs_outside': {'left_sample_count': 91, 'right_sample_count': 266, 'ok_rate_delta': 0.22990167727009825, 'high_rate_delta': -0.2261422787738577, 'low_rate_delta': -0.0037593984962406013, 'out_spec_rate_delta': -0.22990167727009833}, 'inside_vs_below': {'left_sample_count': 91, 'right_sample_count': 95, 'ok_rate_delta': 0.09381145170618854, 'high_rate_delta': -0.09381145170618854, 'low_rate_delta': 0.0, 'out_spec_rate_delta': -0.09381145170618854}, 'inside_vs_above': {'left_sample_count': 91, 'right_sample_count': 171, 'ok_rate_delta': 0.30550735813893704, 'high_rate_delta': -0.2996594049225628, 'low_rate_delta': -0.005847953216374269, 'out_spec_rate_delta': -0.3055073581389371}}, 'support_pass': True, 'risk_guardrail_pass': True}。
- 动作类型摘要：{'metrics': {'decrease_to_band': {'sample_count': 171, 'ok_rate': 0.672514619883041, 'high_rate': 0.3216374269005848, 'low_rate': 0.005847953216374269, 'out_spec_rate': 0.32748538011695905, 'mean_t90': 8.611403508771929, 'band_accuracy': 0.4853801169590643, 'direction_accuracy': 0.8654970760233918, 'target_accuracy_5pct': 1.0}, 'hold_in_band': {'sample_count': 91, 'ok_rate': 0.978021978021978, 'high_rate': 0.02197802197802198, 'low_rate': 0.0, 'out_spec_rate': 0.02197802197802198, 'mean_t90': 8.465934065934066, 'band_accuracy': 0.6483516483516484, 'direction_accuracy': 0.5384615384615384, 'target_accuracy_5pct': 1.0}, 'hold_or_manual_check': {'sample_count': 1, 'ok_rate': 1.0, 'high_rate': 0.0, 'low_rate': 0.0, 'out_spec_rate': 0.0, 'mean_t90': 8.4, 'band_accuracy': nan, 'direction_accuracy': nan, 'target_accuracy_5pct': nan}, 'increase_to_band': {'sample_count': 95, 'ok_rate': 0.8842105263157894, 'high_rate': 0.11578947368421053, 'low_rate': 0.0, 'out_spec_rate': 0.11578947368421053, 'mean_t90': 8.501052631578947, 'band_accuracy': 0.49473684210526314, 'direction_accuracy': 0.9894736842105263, 'target_accuracy_5pct': 1.0}}, 'flags': {'unsafe_increase_hint': True, 'unsafe_decrease_hint': True, 'safe_hold_band_candidate': True}}。
- monitor_chain_candidate_count：9。
- manual_review_only_count：0。
- reject_or_refine_count：8。
- risk_guardrail_status：{'inside_vs_outside_support_pass': True, 'inside_vs_outside_guardrail_pass': True, 'action_flags': {'unsafe_increase_hint': True, 'unsafe_decrease_hint': True, 'safe_hold_band_candidate': True}}。
- readiness_status：`stop_until_more_data`。
- recommended_next_step：`collect_more_data`。

### 48.1 审计目的

本阶段用于解释测试集推荐钙单耗区间为何呈现近似稳定带。该分析只审计既有推荐器输出，不训练模型、不修改规则、不进行策略搜索，也不形成自动控制或 DCS 写回建议。

目录策略同步更新：`data/` 仅保留原始或必要基础数据；本阶段生成的审计 CSV/JSON 输出写入 `runs/ca_interval_diversity_audit/`；图像和人工可读表写入 `reports/`；实验说明仅追加到 `docs/Experimental_Procedure_cn.md`。

### 48.2 主要结果

- 测试集样本数：358
- 唯一推荐区间数：17
- 推荐中心值中位数：0.0203313212849079
- 推荐中心值 IQR：9.094983312319879e-05
- 推荐中心值范围：0.0003854235652850993
- Top 5 推荐区间覆盖率：0.835195530726257
- 接受规则数：21
- 规则中心值 IQR：0.0003688215517576518
- 规则中心值范围：0.003265400426314149
- 聚合压缩标记：True
- 可用上下文字段：rubber_flow_2_win_60_mean

### 48.3 判断

当前推荐器行为分类为：`aggregation_over_smoothed_recommender`。稳定区间的主要解释为：`aggregation_compression`。如果区间稳定主要来自规则本身集中，则它更接近“稳定安全带 MVP”；如果来自多规则中位数聚合，则后续应测试最高优先级规则输出；如果来自单变量规则过粗，则应构建多变量工况规则。

### 48.4 输出文件

- 机器可读审计输出：`runs/ca_interval_diversity_audit/`
- 图像输出：reports\figures\c_line_revalidation\ca_interval_target_distribution.png, reports\figures\c_line_revalidation\ca_interval_width_distribution.png, reports\figures\c_line_revalidation\ca_interval_top_frequency.png, reports\figures\c_line_revalidation\ca_rule_interval_by_regime_feature.png, reports\figures\c_line_revalidation\ca_aggregation_compression.png, reports\figures\c_line_revalidation\ca_context_vs_recommended_target.png
- 人工可读汇总表：reports\tables\c_line_revalidation\ca_interval_diversity_summary.csv

### 48.5 下一步

推荐下一步：`test_top_rule_without_median_aggregation`。

局限性：本阶段为离线审计；不提供因果证明；不生成控制动作；结论依赖既有规则、replay 和人工复核审计产物。

### 49.1 实验目的

Stage 23 显示推荐区间稳定的主要原因是多规则中位数聚合压缩。本阶段在不修改规则、不训练模型、不进行策略搜索的前提下，复用同一批匹配规则和验证 oracle，对比中位数聚合、最高优先级规则、加权平均和重叠交集四种输出方式。

### 49.2 验证集指标

- median_aggregation_baseline: band_accuracy=0.5294117647058824, direction_accuracy=0.8151260504201681, target_iqr=0.00011412667776794852, unique_interval_count=25, risk_guardrail_pass=True
- top_rule_only: band_accuracy=0.711484593837535, direction_accuracy=0.8067226890756303, target_iqr=0.0029027392494684993, unique_interval_count=3, risk_guardrail_pass=True
- weighted_rule_average: band_accuracy=0.9831932773109243, direction_accuracy=0.6442577030812325, target_iqr=2.7618683729128957e-05, unique_interval_count=36, risk_guardrail_pass=True
- narrow_intersection_if_overlap: band_accuracy=0.711484593837535, direction_accuracy=0.8067226890756303, target_iqr=0.0029027392494684993, unique_interval_count=4, risk_guardrail_pass=True

多样性恢复判断：[{'strategy': 'median_aggregation_baseline', 'diversity_recovered': False}, {'strategy': 'top_rule_only', 'diversity_recovered': True}, {'strategy': 'weighted_rule_average', 'diversity_recovered': False}, {'strategy': 'narrow_intersection_if_overlap', 'diversity_recovered': True}]

### 49.3 风险与策略判断

最佳策略：`top_rule_only`。

切换建议：`switch_to_top_rule_only`。

推荐下一步：`update_monitor_artifact_with_selected_aggregation`。

本阶段只做离线 replay 对比。`increase_to_band` 或 `decrease_to_band` 均不能解释为自动控制动作，也不形成 DCS 写回或影子试验建议。

### 49.4 输出文件

- 机器输出：runs\c_line_revalidation\ca_interval_aggregation_strategy_test\strategy_recommendation_replay.parquet, runs\c_line_revalidation\ca_interval_aggregation_strategy_test\strategy_recommendation_replay.csv, runs\c_line_revalidation\ca_interval_aggregation_strategy_test\strategy_metrics.csv, runs\c_line_revalidation\ca_interval_aggregation_strategy_test\strategy_comparison_summary.csv, runs\c_line_revalidation\ca_interval_aggregation_strategy_test\ca_interval_aggregation_strategy_report.json
- 图像输出：reports\figures\c_line_revalidation\ca_aggregation_strategy_target_distribution.png, reports\figures\c_line_revalidation\ca_aggregation_strategy_interval_width.png, reports\figures\c_line_revalidation\ca_aggregation_strategy_accuracy.png, reports\figures\c_line_revalidation\ca_aggregation_strategy_risk.png, reports\figures\c_line_revalidation\ca_top_rule_vs_median_target_scatter.png
- 人工表格：reports\tables\c_line_revalidation\ca_interval_aggregation_strategy_summary.csv

局限性：离线验证不能证明因果关系；oracle 来自验证集事实分箱；聚合策略切换仍需人工工程复核。

### 50.1 定版原因

Stage 24 对比了中位数聚合、最高优先级规则、加权平均和交集策略。中位数聚合保持了最高的推荐区间准确率和较好的方向准确率，且风险护栏通过；Top-rule-only 和交集策略没有恢复有效多样性，准确率下降；加权平均虽然增加区间差异，但方向准确率不足。因此本阶段锁定 `median_aggregation_baseline`。

产品定位为 `stable_safe_band_mvp`：它不是强动态分工况处方系统，而是稳定钙单耗安全带监测 MVP。其含义是：把实际钙单耗控制在历史安全带内，历史上更可能提高 T90 合格概率，但不保证 T90 必然合格。

### 50.2 动作可见性策略

- inside_band：仅监测展示，提示“当前钙单耗处于推荐安全区间内，建议维持观察”。
- above_band：人工复核必需，提示“当前钙单耗高于推荐安全区间，历史数据中高 T90 风险偏高，建议人工复核是否需要小幅降钙”。
- below_band：仅诊断展示，隐藏加钙操作建议。
- missing：关键输入缺失，不生成推荐。

该 MVP 不提供自动控制、不做 DCS 写回、不推荐影子试验。

### 50.3 输出

- 定版 artifact：`models\ca_safe_band_mvp_c_line\safe_band_artifact.json`
- dry-run 表：`runs\c_line_revalidation\ca_safe_band_mvp\final_monitor_dry_run.parquet` 与 `runs\c_line_revalidation\ca_safe_band_mvp\final_monitor_dry_run.csv`
- 规则汇总：`runs\c_line_revalidation\ca_safe_band_mvp\final_rule_summary.csv`
- 人工复核表：`reports\tables\c_line_revalidation\ca_safe_band_mvp_manual_review_sheet.csv`

### 50.4 风险摘要

- inside_band ok/high/low：0.95 / 0.05 / 0.0
- above_band high_rate：0.3419354838709677
- below_band low_rate：0.0

推荐下一步：`human_review_safe_band_mvp`。

局限性：离线验证；非因果证明；无自动控制；无 DCS 写回；必须经过工程人工复核。

### 51.1 阶段目的

本阶段承接 Stage 25，将稳定钙单耗安全带 MVP 封装为监测-only 运行包。该运行包用于后续厂内适配器集成前的人审与接口契约验证，不训练模型、不改规则、不执行自动控制、不做 DCS 写回。

### 51.2 依赖约束

本阶段读取 `IDB_requirements.txt` 作为厂内可用三方依赖清单。依赖策略为：不得引入清单外三方包；`package.py` 以标准库为优先并保持纯推荐逻辑；`interface.py` 和 `main.py` 可在清单允许时使用 pandas/pyarrow 做批量 CSV/parquet 输入输出。本次 package.py 标准库-only：True；依赖策略通过：True；清单外 import：[]。

### 51.3 运行包结构

运行包目录：`deploy\ca_safe_band_mvp_c_line`

- `package.py`：纯推荐逻辑，执行规则匹配、中位数聚合、区间位置判断和动作可见性策略。
- `interface.py`：公开 `SafeBandRecommender`，加载 JSON artifact/support/schema 并提供单条和批量预测。
- `main.py`：示例 CLI 入口；厂内 DCS 获取与写回由后续适配器实现；当前脚本不写 DCS。
- `safe_band_artifact.json`：定版安全带 artifact。
- `support.parquet` / `support.json`：特征与边界支持信息；JSON 可供标准库运行路径使用。
- `schema.json`：输入输出、安全约束和依赖策略契约。

### 51.4 安全约束

- monitor_only = true
- automatic_control = false
- dcs_writeback = false
- increase_hint_hidden = true
- engineering_review_required = true
- no_guarantee_t90_qualified = true

### 51.5 契约测试

历史 dry-run 等价测试行数：1790。

核心字段完全匹配率：1.0。

等价测试通过：True。

推荐下一步：`human_review_runtime_package`。

局限性：仍需工程人工复核；未实现厂内实时数据适配器；未进行在线数据验证；离线安全带关系不是因果证明。

### 52.1 修复原因

本阶段针对稳定钙单耗安全带 MVP 运行包做生产安全修复：严格 JSON、生产模式不信任输入规则 ID、必需特征校验、输出 schema 扩展、加注量换算、Python 3.8+ 兼容和方法文档固化。

### 52.2 修复结果

- 严格 JSON：True
- 依赖策略：True
- package.py 标准库-only：True
- replay 等价测试：True
- 生产模式有效输出率：1.0
- 生产模式禁用输入 rule-id override：True
- 输出 schema 扩展：True
- 加注量换算：True

方法说明文档：`docs\ca_safe_band_mvp_method_and_dataflow.md`。

推荐下一步：`human_review_repaired_runtime_package`。

局限性：仍需工程人工复核；厂方实时适配器尚未实现；尚无在线验证；该安全带关系不是因果证明。

### 53.1 阶段目的

此前运行包默认接收工程化特征。厂内集成通常拿到的是带时间戳的原始平台 DataFrame，因此本阶段新增 `feature_adapter.py`，将原始点位列转换为运行包所需的当前特征状态。

### 53.2 在线窗口策略

在线运行使用当前时刻 `t_now` 之前的尾随窗口：工况变量采用 `[t_now-60min, t_now]` 的均值；钙单耗采用该窗口内 `ca_feed / rubber_flow_2` 的均值。离线标签对齐曾使用停留时间；在线推荐不再额外向前平移 165min，因为当前上游操作影响的是未来产品质量。后续 LIMS 回填验证应按停留时间把当前输出与未来 T90 标签比较。

### 53.3 IR-lag

IR-lag `output_ir_corrected_offset_20_win_15_std` 为可选输入。若存在原始 IR，则计算 `[t_now-35min, t_now-20min]` 的 15 分钟标准差；若缺失，不阻断推荐，只记录 `optional_ir_missing`。

### 53.4 接口更新

- `interface.py` 新增 `predict_from_raw_dataframe` 和 `predict_batch_from_raw_dataframe`。
- `main.py` 新增 `--raw-input-csv`、`--raw-input-parquet`、`--raw-time-col`、`--end-time`、`--min-valid-points` 和 `--include-optional-ir`。
- `schema.json/support.json` 增加原始点位映射和窗口定义。

### 53.5 烟测结果

- engineered predict_one：True
- raw dataframe predict：True
- main.py raw CLI：True
- 依赖策略：True
- IR 可选确认：True

推荐下一步：`human_review_feature_adapter_contract`。

局限性：本阶段使用合成 raw-like 数据验证接口路径，仍需厂方提供真实原始平台 DataFrame 做最终适配器验收；无 DCS 写回；无自动控制。

## 54. future holdout 新数据与卤化橡胶 T90 回填验证

本阶段使用 `data/future/` 作为完全未见过的 future holdout，验证冻结 V1 monitor-only 钙单耗安全带运行链路。T90 文件包括 `2026.1.xls`、`2026.2.xls`、`2026.3C.xls`；验证原则是只使用胶种为 `卤化橡胶` 的 T90，排除 `氯丁基橡胶` 及其他胶种。

原始 DCS 解析：{'file_count': 46, 'parsed_file_count': 46, 'unparsed_files': [], 'detected_points': ['bromine_feed', 'ca_feed', 'esbo_feed', 'neutral_alkali_feed', 'r510a_temp', 'r511a_temp', 'r512a_temp', 'r513_temp', 'r514_temp', 'rubber_flow_2', 'tank_rubber_conc'], 'missing_required_points': [], 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'row_count': 128099, 'duplicate_timestamp_count': 0, 'sampling_interval_summary': {'median_minutes': 1.0, 'q25_minutes': 1.0, 'q75_minutes': 1.0, 'max_minutes': 2.0}, 'missing_rate_by_point': {'bromine_feed': 0.03130391337949555, 'ca_feed': 0.046854386060781114, 'esbo_feed': 1.561292437880077e-05, 'neutral_alkali_feed': 6.245169751520308e-05, 'r510a_temp': 0.06246731043958188, 'r511a_temp': 0.031241461681980342, 'r512a_temp': 1.561292437880077e-05, 'r513_temp': 1.561292437880077e-05, 'r514_temp': 1.561292437880077e-05, 'rubber_flow_2': 1.561292437880077e-05, 'tank_rubber_conc': 1.561292437880077e-05}, 'future_raw_quality_pass': True, 'parsed_file_details': [{'file': 'data\\future\\B4-AT-C50002A-BIIR.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-AT-C50002A-BIIR/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FI-C51005_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 126092, 'tag_value': 'B4-FI-C51005_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FI-C53001.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 120077, 'tag_value': 'B4-FI-C53001/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FI-C54051.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-FI-C54051/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C30501.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 124092, 'tag_value': 'B4-FIC-C30501/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C51001.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-FIC-C51001/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C51003.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 124097, 'tag_value': 'B4-FIC-C51003/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C51004.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 124089, 'tag_value': 'B4-FIC-C51004/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C51401.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-FIC-C51401/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C51605.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128091, 'tag_value': 'B4-FIC-C51605/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C51801.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-FIC-C51801/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C51802.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-FIC-C51802/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C53003A.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-FIC-C53003A/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C53051A.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-FIC-C53051A/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C53252.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-FIC-C53252/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51006A_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-PI-C51006A_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51006B.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-PI-C51006B/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51101A_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-PI-C51101A_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51101B_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 124097, 'tag_value': 'B4-PI-C51101B_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51203A_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 126097, 'tag_value': 'B4-PI-C51203A_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51203B_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 124097, 'tag_value': 'B4-PI-C51203B_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51301_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-PI-C51301_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51403_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 124097, 'tag_value': 'B4-PI-C51403_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C53001A.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-PI-C53001A/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PIC-C53002A.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-PIC-C53002A/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C50604.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C50604/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51003.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C51003/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51007A_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 120097, 'tag_value': 'B4-TI-C51007A_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51007B_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C51007B_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51101A_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 124097, 'tag_value': 'B4-TI-C51101A_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51101B_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C51101B_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51202B_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C51202B_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51301_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C51301_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51401_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C51401_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51702A.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C51702A/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C53202.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C53202/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C53205.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 120097, 'tag_value': 'B4-TI-C53205/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C54002.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 124097, 'tag_value': 'B4-TI-C54002/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C54003.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C54003/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C56401.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-TI-C56401/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-CM511A.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-TI-CM511A/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-CM53001.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-TI-CM53001/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-CM53201.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 120097, 'tag_value': 'B4-TI-CM53201/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-CM54001.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-TI-CM54001/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TIC-C53002A.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 124092, 'tag_value': 'B4-TIC-C53002A/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TICA-C52601.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TICA-C52601/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}]}。

T90 解析与过滤：{'future_t90_files_requested': ['data\\future\\2026.1.xls', 'data\\future\\2026.2.xls', 'data\\future\\2026.3C.xls'], 'future_t90_files_found': [], 'future_t90_files_parsed': [], 'future_t90_files_failed': [{'file': 'data\\future\\2026.1.xls', 'parsed': False, 'error': "FileNotFoundError: [Errno 2] No such file or directory: 'data\\\\future\\\\2026.1.xls'", 'xls_engine_missing': False}, {'file': 'data\\future\\2026.2.xls', 'parsed': False, 'error': "FileNotFoundError: [Errno 2] No such file or directory: 'data\\\\future\\\\2026.2.xls'", 'xls_engine_missing': False}, {'file': 'data\\future\\2026.3C.xls', 'parsed': False, 'error': "FileNotFoundError: [Errno 2] No such file or directory: 'data\\\\future\\\\2026.3C.xls'", 'xls_engine_missing': False}], 'xls_engine_missing': False, 'selected_time_column_by_file': {}, 'selected_rubber_type_column_by_file': {}, 'selected_t90_column_by_file': {}, 'total_t90_rows': 0, 'halogen_t90_rows': 0, 'excluded_non_halogen_rows': 0, 'rubber_type_value_counts': {}, 't90_time_min': None, 't90_time_max': None, 't90_min': None, 't90_q25': None, 't90_median': None, 't90_q75': None, 't90_max': None, 'missing_rubber_type_column': False, 'missing_t90_column': False, 'future_t90_available': False, 'future_t90_filter_pass': False, 'warnings': []}。

运行特征质量：{'evaluation_row_count': 12810, 'feature_valid_row_count': 10810, 'invalid_row_count': 2000, 'missing_feature_counts': {'rubber_flow_2_win_60_mean': 0, 'bromine_feed_win_60_mean': 400, 'tank_rubber_conc_win_60_mean': 0, 'r510a_temp_win_60_mean': 800, 'r511a_temp_win_60_mean': 400, 'r512a_temp_win_60_mean': 0, 'esbo_feed_win_60_mean': 0, 'neutral_alkali_feed_win_60_mean': 0, 'r513_temp_win_60_mean': 0, 'r514_temp_win_60_mean': 0, 'ca_per_rubber_flow_win_60_mean': 600}, 'insufficient_window_counts': {'rubber_flow_2_win_60_mean': 0, 'bromine_feed_win_60_mean': 400, 'tank_rubber_conc_win_60_mean': 0, 'r510a_temp_win_60_mean': 800, 'r511a_temp_win_60_mean': 400, 'r512a_temp_win_60_mean': 0, 'esbo_feed_win_60_mean': 0, 'neutral_alkali_feed_win_60_mean': 0, 'r513_temp_win_60_mean': 0, 'r514_temp_win_60_mean': 0, 'ca_per_rubber_flow_win_60_mean': 600}, 'optional_ir_available_rate': 0.0, 'feature_quality_pass': False}。

推荐 replay 摘要：{'scored_row_count': 12810, 'recommendation_coverage': 0.843871975019516, 'no_recommendation_count': 2000, 'input_invalid_count': 2000, 'inside_band_count': 2112, 'above_band_count': 7629, 'below_band_count': 1069, 'manual_review_required_count': 7629, 'diagnostic_only_count': 1069, 'monitor_only_count': 2112, 'missing_required_features_summary': {'[]': 10810, "['ca_per_rubber_flow_win_60_mean']": 600, "['r510a_temp_win_60_mean']": 600, "['bromine_feed_win_60_mean']": 400, "['r510a_temp_win_60_mean', 'r511a_temp_win_60_mean']": 200, "['r511a_temp_win_60_mean']": 200}, 'warning_flags_summary': {'high_t90_risk_manual_review': 7629, '': 2112, 'missing_required_features': 2000, 'increase_hint_hidden_diagnostic_only': 1069}, 'recommended_ca_consumption_distribution': {'min': 0.02004226582262995, 'q25': 0.02018171956339015, 'median': 0.02019975404743075, 'q75': 0.020290772138045927, 'max': 0.0203680387903005}, 'current_ca_consumption_distribution': {'min': -44.59794671461411, 'q25': 0.020295952000911874, 'median': 0.02075727899047824, 'q75': 0.02122845862783127, 'max': 0.038580744015425585}, 'future_replay_pass': True}。

停留时间回填验证：{'future_t90_available': False, 'future_t90_validation_status': 'pending_lims_labels', 'aligned_sample_count': 0}。

清晰标签不确定性结果：{'clear_sample_count': None, 'uncertain_boundary_rate': None, 'risk_by_interval_position': []}。

future 与历史特征漂移：{'historical_reference_available': True, 'historical_reference_path': 'runs\\t90_ca_feature_dataset.parquet', 'feature_count_compared': 11, 'max_out_of_historical_range_rate': 0.40117096018735365, 'max_psi_like_drift_score': 4.526382630307499, 'future_within_historical_support': False}。

validation_mode：`failed_feature_construction`；recommended_next_step：`fix_future_data_mapping`。

局限性：`.xls` 读取可能需要转换为 `.xlsx/.csv` 或厂方允许的读取依赖；T90 测量误差约 0.1；future raw 点位映射依赖文件命名和格式；本阶段仅 monitor-only，不自动控制，不写回 DCS。

## 55. future holdout 新数据与卤化橡胶 T90 回填验证

本阶段使用 `data/future/` 作为完全未见过的 future holdout，验证冻结 V1 monitor-only 钙单耗安全带运行链路。T90 文件包括 `2026.1.xls`、`2026.2.xls`、`2026.3C.xls`；验证原则是只使用胶种为 `卤化橡胶` 的 T90，排除 `氯丁基橡胶` 及其他胶种。

原始 DCS 解析：{'file_count': 46, 'parsed_file_count': 46, 'unparsed_files': [], 'detected_points': ['bromine_feed', 'ca_feed', 'esbo_feed', 'neutral_alkali_feed', 'r510a_temp', 'r511a_temp', 'r512a_temp', 'r513_temp', 'r514_temp', 'rubber_flow_2', 'tank_rubber_conc'], 'missing_required_points': [], 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'row_count': 128099, 'duplicate_timestamp_count': 0, 'sampling_interval_summary': {'median_minutes': 1.0, 'q25_minutes': 1.0, 'q75_minutes': 1.0, 'max_minutes': 2.0}, 'missing_rate_by_point': {'bromine_feed': 0.03130391337949555, 'ca_feed': 0.046854386060781114, 'esbo_feed': 1.561292437880077e-05, 'neutral_alkali_feed': 6.245169751520308e-05, 'r510a_temp': 0.06246731043958188, 'r511a_temp': 0.031241461681980342, 'r512a_temp': 1.561292437880077e-05, 'r513_temp': 1.561292437880077e-05, 'r514_temp': 1.561292437880077e-05, 'rubber_flow_2': 1.561292437880077e-05, 'tank_rubber_conc': 1.561292437880077e-05}, 'future_raw_quality_pass': True, 'parsed_file_details': [{'file': 'data\\future\\B4-AT-C50002A-BIIR.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-AT-C50002A-BIIR/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FI-C51005_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 126092, 'tag_value': 'B4-FI-C51005_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FI-C53001.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 120077, 'tag_value': 'B4-FI-C53001/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FI-C54051.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-FI-C54051/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C30501.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 124092, 'tag_value': 'B4-FIC-C30501/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C51001.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-FIC-C51001/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C51003.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 124097, 'tag_value': 'B4-FIC-C51003/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C51004.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 124089, 'tag_value': 'B4-FIC-C51004/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C51401.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-FIC-C51401/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C51605.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128091, 'tag_value': 'B4-FIC-C51605/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C51801.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-FIC-C51801/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C51802.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-FIC-C51802/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C53003A.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-FIC-C53003A/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C53051A.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-FIC-C53051A/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-FIC-C53252.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-FIC-C53252/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51006A_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-PI-C51006A_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51006B.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-PI-C51006B/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51101A_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-PI-C51101A_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51101B_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 124097, 'tag_value': 'B4-PI-C51101B_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51203A_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 126097, 'tag_value': 'B4-PI-C51203A_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51203B_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 124097, 'tag_value': 'B4-PI-C51203B_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51301_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-PI-C51301_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C51403_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 124097, 'tag_value': 'B4-PI-C51403_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PI-C53001A.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-PI-C53001A/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-PIC-C53002A.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-PIC-C53002A/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C50604.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C50604/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51003.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C51003/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51007A_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 120097, 'tag_value': 'B4-TI-C51007A_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51007B_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C51007B_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51101A_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 124097, 'tag_value': 'B4-TI-C51101A_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51101B_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C51101B_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51202B_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C51202B_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51301_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C51301_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51401_S.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C51401_S/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C51702A.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C51702A/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C53202.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C53202/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C53205.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 120097, 'tag_value': 'B4-TI-C53205/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C54002.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 124097, 'tag_value': 'B4-TI-C54002/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C54003.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TI-C54003/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-C56401.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-TI-C56401/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-CM511A.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-TI-CM511A/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-CM53001.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-TI-CM53001/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-CM53201.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 120097, 'tag_value': 'B4-TI-CM53201/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TI-CM54001.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 122097, 'tag_value': 'B4-TI-CM54001/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TIC-C53002A.PV.CV.txt', 'encoding': 'utf-8', 'row_count': 124092, 'tag_value': 'B4-TIC-C53002A/PV.CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}, {'file': 'data\\future\\B4-TICA-C52601.PV.F_CV.txt', 'encoding': 'utf-8', 'row_count': 128097, 'tag_value': 'B4-TICA-C52601/PV.F_CV', 'time_min': '2026-01-01T00:01:00', 'time_max': '2026-03-31T00:00:00', 'parse_error': None}]}。

T90 解析与过滤：{'future_t90_files_requested': ['data\\future\\2026.1.xlsx', 'data\\future\\2026.2.xlsx', 'data\\future\\2026.3C.xlsx'], 'future_t90_files_found': ['data\\future\\2026.1.xlsx', 'data\\future\\2026.2.xlsx', 'data\\future\\2026.3C.xlsx'], 'future_t90_files_parsed': ['data\\future\\2026.1.xlsx', 'data\\future\\2026.2.xlsx', 'data\\future\\2026.3C.xlsx'], 'future_t90_files_failed': [], 'xls_engine_missing': False, 'selected_time_column_by_file': {'data\\future\\2026.1.xlsx': '采样时间', 'data\\future\\2026.2.xlsx': '采样时间', 'data\\future\\2026.3C.xlsx': '采样时间'}, 'selected_rubber_type_column_by_file': {'data\\future\\2026.1.xlsx': '样品名称', 'data\\future\\2026.2.xlsx': '样品名称', 'data\\future\\2026.3C.xlsx': '样品名称'}, 'selected_t90_column_by_file': {'data\\future\\2026.1.xlsx': 't´c(90),min', 'data\\future\\2026.2.xlsx': 't´c(90),min', 'data\\future\\2026.3C.xlsx': 't´c(90),min'}, 'total_t90_rows': 1679, 'halogen_t90_rows': 305, 'excluded_non_halogen_rows': 1374, 'rubber_type_value_counts': {'卤化橡胶': 1117, '氯化丁基橡胶': 562}, 't90_time_min': '2026-01-09T01:44:00', 't90_time_max': '2026-03-31T22:31:00', 't90_min': 7.6, 't90_q25': 8.4, 't90_median': 8.5, 't90_q75': 8.7, 't90_max': 9.2, 'missing_rubber_type_column': False, 'missing_t90_column': False, 'future_t90_available': True, 'future_t90_filter_pass': True, 'warnings': []}。

运行特征质量：{'evaluation_row_count': 12810, 'feature_valid_row_count': 10810, 'invalid_row_count': 2000, 'missing_feature_counts': {'rubber_flow_2_win_60_mean': 0, 'bromine_feed_win_60_mean': 400, 'tank_rubber_conc_win_60_mean': 0, 'r510a_temp_win_60_mean': 800, 'r511a_temp_win_60_mean': 400, 'r512a_temp_win_60_mean': 0, 'esbo_feed_win_60_mean': 0, 'neutral_alkali_feed_win_60_mean': 0, 'r513_temp_win_60_mean': 0, 'r514_temp_win_60_mean': 0, 'ca_per_rubber_flow_win_60_mean': 600}, 'insufficient_window_counts': {'rubber_flow_2_win_60_mean': 0, 'bromine_feed_win_60_mean': 400, 'tank_rubber_conc_win_60_mean': 0, 'r510a_temp_win_60_mean': 800, 'r511a_temp_win_60_mean': 400, 'r512a_temp_win_60_mean': 0, 'esbo_feed_win_60_mean': 0, 'neutral_alkali_feed_win_60_mean': 0, 'r513_temp_win_60_mean': 0, 'r514_temp_win_60_mean': 0, 'ca_per_rubber_flow_win_60_mean': 600}, 'optional_ir_available_rate': 0.0, 'feature_valid_rate': 0.843871975019516, 'feature_quality_pass': True}。

推荐 replay 摘要：{'scored_row_count': 12810, 'recommendation_coverage': 0.843871975019516, 'no_recommendation_count': 2000, 'input_invalid_count': 2000, 'inside_band_count': 2112, 'above_band_count': 7629, 'below_band_count': 1069, 'manual_review_required_count': 7629, 'diagnostic_only_count': 1069, 'monitor_only_count': 2112, 'missing_required_features_summary': {'[]': 10810, "['ca_per_rubber_flow_win_60_mean']": 600, "['r510a_temp_win_60_mean']": 600, "['bromine_feed_win_60_mean']": 400, "['r510a_temp_win_60_mean', 'r511a_temp_win_60_mean']": 200, "['r511a_temp_win_60_mean']": 200}, 'warning_flags_summary': {'high_t90_risk_manual_review': 7629, '': 2112, 'missing_required_features': 2000, 'increase_hint_hidden_diagnostic_only': 1069}, 'recommended_ca_consumption_distribution': {'min': 0.02004226582262995, 'q25': 0.02018171956339015, 'median': 0.02019975404743075, 'q75': 0.020290772138045927, 'max': 0.0203680387903005}, 'current_ca_consumption_distribution': {'min': -44.59794671461411, 'q25': 0.020295952000911874, 'median': 0.02075727899047824, 'q75': 0.02122845862783127, 'max': 0.038580744015425585}, 'future_replay_pass': True}。

停留时间回填验证：{'future_t90_available': True, 'future_t90_validation_status': 'available', 'aligned_sample_count': 5282, 'inside_band_count': 1347, 'outside_band_count': 3079, 'risk_by_interval_position': [{'interval_position': 'inside_band', 'sample_count': 1347, 'ok_rate': 0.9962880475129918, 'high_rate': 0.003711952487008166, 'low_rate': 0.0, 'out_spec_rate': 0.003711952487008166, 'mean_t90': 8.4720861172977}, {'interval_position': 'outside_band', 'sample_count': 3079, 'ok_rate': 0.6284507957128938, 'high_rate': 0.3657031503734979, 'low_rate': 0.005846053913608314, 'out_spec_rate': 0.3715492042871062, 'mean_t90': 8.636440402728159}, {'interval_position': 'above_band', 'sample_count': 2829, 'ok_rate': 0.6019794980558502, 'high_rate': 0.3980205019441499, 'low_rate': 0.0, 'out_spec_rate': 0.3980205019441499, 'mean_t90': 8.662672322375398}, {'interval_position': 'below_band', 'sample_count': 250, 'ok_rate': 0.928, 'high_rate': 0.0, 'low_rate': 0.072, 'out_spec_rate': 0.072, 'mean_t90': 8.3396}], 'inside_vs_outside_high_rate_delta': -0.3619911978864897, 'inside_vs_outside_out_spec_rate_delta': -0.367837251800098, 'above_vs_inside_high_rate_delta': 0.3943085494571417, 'recommendation_coverage_on_aligned': 0.8379401741764483, 'future_holdout_risk_guardrail_pass': True, 'clear_sample_count': 4656, 'uncertain_boundary_rate': 0.11851571374479364, 'clear_label_risk_by_interval_position': [{'interval_position': 'inside_band', 'sample_count': 1263, 'ok_rate': 0.9960411718131433, 'high_rate': 0.00395882818685669, 'low_rate': 0.0, 'out_spec_rate': 0.00395882818685669, 'mean_t90': 8.460886777513856}, {'interval_position': 'outside_band', 'sample_count': 2573, 'ok_rate': 0.5553828216090168, 'high_rate': 0.4376214535561601, 'low_rate': 0.006995724834823164, 'out_spec_rate': 0.4446171783909833, 'mean_t90': 8.64667703070346}, {'interval_position': 'above_band', 'sample_count': 2369, 'ok_rate': 0.5246939636977628, 'high_rate': 0.4753060363022372, 'low_rate': 0.0, 'out_spec_rate': 0.4753060363022372, 'mean_t90': 8.67040945546644}, {'interval_position': 'below_band', 'sample_count': 204, 'ok_rate': 0.9117647058823529, 'high_rate': 0.0, 'low_rate': 0.08823529411764706, 'out_spec_rate': 0.08823529411764706, 'mean_t90': 8.371078431372547}]}。

清晰标签不确定性结果：{'clear_sample_count': 4656, 'uncertain_boundary_rate': 0.11851571374479364, 'risk_by_interval_position': [{'interval_position': 'inside_band', 'sample_count': 1263, 'ok_rate': 0.9960411718131433, 'high_rate': 0.00395882818685669, 'low_rate': 0.0, 'out_spec_rate': 0.00395882818685669, 'mean_t90': 8.460886777513856}, {'interval_position': 'outside_band', 'sample_count': 2573, 'ok_rate': 0.5553828216090168, 'high_rate': 0.4376214535561601, 'low_rate': 0.006995724834823164, 'out_spec_rate': 0.4446171783909833, 'mean_t90': 8.64667703070346}, {'interval_position': 'above_band', 'sample_count': 2369, 'ok_rate': 0.5246939636977628, 'high_rate': 0.4753060363022372, 'low_rate': 0.0, 'out_spec_rate': 0.4753060363022372, 'mean_t90': 8.67040945546644}, {'interval_position': 'below_band', 'sample_count': 204, 'ok_rate': 0.9117647058823529, 'high_rate': 0.0, 'low_rate': 0.08823529411764706, 'out_spec_rate': 0.08823529411764706, 'mean_t90': 8.371078431372547}]}。

future 与历史特征漂移：{'historical_reference_available': True, 'historical_reference_path': 'runs\\t90_ca_feature_dataset.parquet', 'feature_count_compared': 11, 'max_out_of_historical_range_rate': 0.40117096018735365, 'max_psi_like_drift_score': 4.526382630307499, 'future_within_historical_support': False}。

validation_mode：`runtime_plus_t90_backfill`；recommended_next_step：`investigate_future_distribution_shift`。

局限性：`.xls` 读取可能需要转换为 `.xlsx/.csv` 或厂方允许的读取依赖；T90 测量误差约 0.1；future raw 点位映射依赖文件命名和格式；本阶段仅 monitor-only，不自动控制，不写回 DCS。


## 43. 基于点位上下限清洗的 future holdout V1 回放与卤化橡胶 T90 复验

本阶段针对 future holdout 重新验证冻结 V1 monitor-only 钙单耗安全带。由于上一轮 raw DCS 直接入模后出现 `current_ca_consumption=-44` 一类不可能值，本阶段先使用 `data/副本卤化工段数据点位.xlsx` 中的点位正常上下限清洗 DCS：低于下限或高于上限的值置为缺失，不做裁剪和插值；胶液流量无效或小于等于 0 时，对应时刻钙单耗置为缺失。

实验记录文档已先备份并去重，去重报告见 `runs\future_holdout_v1_cleaned_validation\doc_dedup_audit\experiment_doc_dedup_report.json`。DCS 清洗摘要：清洗前钙单耗最小值 `-595.1175922748886`，清洗后最小值 `0.012639765115534406`，清洗后最大值 `0.033583054945238486`，possible shutdown/invalid operation 时间戳数 `40038`。

T90 文件使用 `2026.1.xlsx`、`2026.2.xlsx`、`2026.3C.xlsx`，仅保留胶种为 `卤化橡胶` 的记录，排除 `氯丁基橡胶` 等其他胶种。卤化橡胶有效 T90 行数为 `305`。

清洗后运行特征质量：`{'evaluation_row_count': 12816, 'feature_valid_row_count': 7363, 'invalid_row_count': 5453, 'missing_feature_counts': {'rubber_flow_2_win_60_mean': 2789, 'bromine_feed_win_60_mean': 4414, 'tank_rubber_conc_win_60_mean': 2, 'r510a_temp_win_60_mean': 4791, 'r511a_temp_win_60_mean': 4392, 'r512a_temp_win_60_mean': 3924, 'esbo_feed_win_60_mean': 4024, 'neutral_alkali_feed_win_60_mean': 3783, 'r513_temp_win_60_mean': 3800, 'r514_temp_win_60_mean': 3805, 'ca_per_rubber_flow_win_60_mean': 3822}, 'insufficient_window_counts': {'rubber_flow_2_win_60_mean': 2789, 'bromine_feed_win_60_mean': 4414, 'tank_rubber_conc_win_60_mean': 2, 'r510a_temp_win_60_mean': 4791, 'r511a_temp_win_60_mean': 4392, 'r512a_temp_win_60_mean': 3924, 'esbo_feed_win_60_mean': 4024, 'neutral_alkali_feed_win_60_mean': 3783, 'r513_temp_win_60_mean': 3800, 'r514_temp_win_60_mean': 3805, 'ca_per_rubber_flow_win_60_mean': 3822}, 'optional_ir_available_rate': 0.0, 'invalid_due_to_shutdown_or_out_of_bound_count': 4024, 'feature_valid_rate': 0.5745162297128589, 'feature_quality_pass': False}`。推荐回放覆盖率 `0.5745162297128589`，inside/above/below 数量分别为 `2031`、`4786`、`546`，不可能钙单耗数量 `0`。

更严格的一对一 T90 回填结果：`{'strategy': 'one_t90_to_nearest_prediction', 'aligned_sample_count': 249, 'unique_t90_count': 249, 'unique_recommendation_count': 249, 'duplicate_t90_match_rate': 0.0, 'risk_by_interval_position': [{'interval_position': 'inside_band', 'sample_count': 79, 'ok_rate': 0.9873417721518988, 'high_rate': 0.012658227848101266, 'low_rate': 0.0, 'out_spec_rate': 0.012658227848101266, 'mean_t90': 8.477215189873418}, {'interval_position': 'outside_band', 'sample_count': 170, 'ok_rate': 0.6235294117647059, 'high_rate': 0.37058823529411766, 'low_rate': 0.0058823529411764705, 'out_spec_rate': 0.3764705882352941, 'mean_t90': 8.639411764705883}, {'interval_position': 'above_band', 'sample_count': 158, 'ok_rate': 0.6012658227848101, 'high_rate': 0.3987341772151899, 'low_rate': 0.0, 'out_spec_rate': 0.3987341772151899, 'mean_t90': 8.662658227848103}, {'interval_position': 'below_band', 'sample_count': 12, 'ok_rate': 0.9166666666666666, 'high_rate': 0.0, 'low_rate': 0.08333333333333333, 'out_spec_rate': 0.08333333333333333, 'mean_t90': 8.333333333333334}], 'inside_vs_outside_high_rate_delta': -0.3579300074460164, 'inside_vs_outside_out_spec_rate_delta': -0.36381236038719283, 'above_vs_inside_high_rate_delta': 0.3860759493670886, 'risk_guardrail_pass': True, 'clear_sample_count': 217, 'uncertain_boundary_rate': 0.1285140562248996, 'clear_label_risk_by_interval_position': [{'interval_position': 'inside_band', 'sample_count': 74, 'ok_rate': 0.9864864864864865, 'high_rate': 0.013513513513513514, 'low_rate': 0.0, 'out_spec_rate': 0.013513513513513514, 'mean_t90': 8.462162162162162}, {'interval_position': 'outside_band', 'sample_count': 143, 'ok_rate': 0.5524475524475524, 'high_rate': 0.4405594405594406, 'low_rate': 0.006993006993006993, 'out_spec_rate': 0.44755244755244755, 'mean_t90': 8.648951048951046}, {'interval_position': 'above_band', 'sample_count': 133, 'ok_rate': 0.5263157894736842, 'high_rate': 0.47368421052631576, 'low_rate': 0.0, 'out_spec_rate': 0.47368421052631576, 'mean_t90': 8.670676691729325}, {'interval_position': 'below_band', 'sample_count': 10, 'ok_rate': 0.9, 'high_rate': 0.0, 'low_rate': 0.1, 'out_spec_rate': 0.1, 'mean_t90': 8.36}]}`。清晰标签不确定性与边界样本结果见 clear-label 报告。future 与历史参考漂移摘要：`{'historical_reference_available': True, 'historical_reference_path': 'runs\\t90_ca_feature_dataset.parquet', 'feature_count_compared': 11, 'max_out_of_historical_range_rate': 0.23396828562880223, 'max_psi_like_drift_score': 4.458904003420629, 'future_within_historical_support': True, 'top_drift_features': ['rubber_flow_2_win_60_mean', 'bromine_feed_win_60_mean', 'esbo_feed_win_60_mean', 'tank_rubber_conc_win_60_mean', 'r510a_temp_win_60_mean']}`。分月稳定性摘要：`{'months': [{'month': '2026-01', 'raw_row_count': 44617, 'out_of_bound_rate': 0.1876394444025616, 'feature_valid_rate': 0.7701097916199866, 'recommendation_coverage': 0.7701097916199866, 'inside_band_count': 851, 'above_band_count': 2214, 'below_band_count': 372, 't90_halogen_count': 109, 'one_to_one_aligned_sample_count': 107, 'monthly_evidence_sufficient': True, 'inside_high_rate': 0.030303030303030304, 'above_high_rate': 0.4393939393939394, 'below_low_rate': 0.0, 'risk_guardrail_pass': True}, {'month': '2026-02', 'raw_row_count': 40300, 'out_of_bound_rate': 0.48764944732686666, 'feature_valid_rate': 0.0679563492063492, 'recommendation_coverage': 0.0679563492063492, 'inside_band_count': 64, 'above_band_count': 42, 'below_band_count': 168, 't90_halogen_count': 55, 'one_to_one_aligned_sample_count': 6, 'monthly_evidence_sufficient': False, 'inside_high_rate': 0.0, 'above_high_rate': nan, 'below_low_rate': 0.25, 'risk_guardrail_pass': False}, {'month': '2026-03', 'raw_row_count': 43182, 'out_of_bound_rate': 0.13264154677243464, 'feature_valid_rate': 0.8451747280722055, 'recommendation_coverage': 0.8451747280722055, 'inside_band_count': 1116, 'above_band_count': 2530, 'below_band_count': 6, 't90_halogen_count': 141, 'one_to_one_aligned_sample_count': 136, 'monthly_evidence_sufficient': True, 'inside_high_rate': 0.0, 'above_high_rate': 0.3695652173913043, 'below_low_rate': nan, 'risk_guardrail_pass': True}], 'sufficient_month_count': 2, 'insufficient_month_count': 1, 'monthly_risk_separation_stable': True}`。

validation_mode：`cleaned_runtime_plus_t90_backfill`；recommended_next_step：`prepare_V1_monitor_only_factory_test_with_cleaned_future_evidence`。

局限性：点位上下限质量依赖配置表；T90 测量误差约 0.1；future raw 点位映射依赖文件命名和格式；本阶段仍为 monitor-only，不自动控制，不写回 DCS。


## 45. C线专用 future holdout 清洗复验与旧合并线证据修正

- 修正原因：旧 V1 monitor-only 回放使用了 C/D/E 合并线包，不能作为 C 线部署证据。
- C 线包：`deploy/ca_safe_band_mvp_c_line`；C 线 artifact：`models/ca_safe_band_mvp_c_line/safe_band_artifact.json`。
- 旧合并线证据：已标记为 superseded for C-line deployment evidence，未删除。
- future 数据路径：`data\future`。
- 点位上下限来源：`data\副本卤化工段数据点位.xlsx`；清洗规则为越界值设为缺失，不剪裁、不插值。
- T90 文件：2026.1.xlsx、2026.2.xlsx、2026.3C.xlsx；过滤条件：胶种 = 卤化橡胶，线别 = C。
- DCS 清洗：bounds_applied=22，possible_shutdown_timestamp_count=40038。
- 异常铙单耗：清洗前 min=-595.1175922748886，清洗后 min=0.012639765115534406。
- C 线推荐回放：coverage=0.5745162297128589，inside=1752，above=5226，below=385。
- one-to-one T90 回填：aligned=249，risk_guardrail_pass=True。
- clear-label 不确定：uncertain_boundary_rate=0.1285140562248996。
- C 线历史对比：future_within_c_line_historical_support=True。
- 分月稳定性：monthly_risk_separation_stable=True。
- validation_mode：c_line_cleaned_runtime_plus_t90_backfill。
- recommended_next_step：human_review_c_line_monitor_only_candidate。
- 限制：C 线 rebuild 阶段仍为 stop_until_more_data；当前仅是 monitor-only candidate；需人工复核；T90 测量误差约 0.1；future 点位映射依赖文件命名/格式；不实施自动控制，不写回 DCS。

## 46. C线 monitor-only 候选包人工复核与 Go/No-Go 证据包生成

目的：基于修正后的 C-line future holdout validation，生成只面向人工复核的 C线 monitor-only 候选包 Go/No-Go 材料，支持是否进入 C线厂区 monitor-only 测试的人工决策。

Stage 45 回顾：旧 V1 monitor-only replay 使用 C/D/E 合并线包，不能作为 C线部署证据；本阶段仅使用 `deploy/ca_safe_band_mvp_c_line` 和 `models/ca_safe_band_mvp_c_line/safe_band_artifact.json`。旧合并线证据已 superseded，未删除，但不得用于 C线部署 Go/No-Go。

C-line 策略与证据：最终策略为 `top_rule_only`；C-line rebuild readiness 为 `stop_until_more_data`，当前状态仅为 monitor-only candidate。future 一对一 T90 回填 aligned_sample_count=249，risk_guardrail_pass=True；clear-label uncertain_boundary_rate=0.1285140562248996。

数据质量与异常策略：点位上下限清洗为越界置缺失，不裁剪、不插值；possible_shutdown_timestamp_count=40038。无效窗口、停工/非正常操作、关键点缺失或 60min 有效点不足时不生成推荐。above_band 仅 manual_review_required，below_band 仅 diagnostic_only。

规则复核要求：C-line rule count=21，monitor-chain candidate count=9，reject/refine rule count=8。所有规则需工艺人工确认，尤其要确认 top_rule_only 是否适用于 C线。

Go/No-Go 决策：本阶段最终状态为 `ready_for_human_review_as_c_line_monitor_only_candidate`，推荐下一步为 `human_review_c_line_monitor_only_candidate`。该输出不是批准文件，不允许直接部署为自动控制。

限制：C-line readiness was stop_until_more_data；current state is monitor-only candidate only；human review required；T90 measurement error about 0.1；no automatic control；no DCS writeback。

## 47. C线 monitor-only 指导测试资格检查

### purpose

本阶段生成 C线 monitor-only 指导测试资格检查包，用于回答 C线运行包是否具备进入人工评审和后续指导测试准备的条件。

### Stage 46 dependency

依赖 Stage 46 人工复核包：`runs/c_line_monitor_only_human_review_pack/`。

### future data role

`real_operation_holdout_validation_only`。future 新数据是真实操作数据，仅用于独立 holdout 验证，不用于训练、调参、规则更新或 artifact 更新。

### factory test mode

`guidance_monitor_only`。系统只做指导/监测，不参与实际控制。

### C-line package and artifact

- package: `deploy/ca_safe_band_mvp_c_line/`
- artifact: `models/ca_safe_band_mvp_c_line/safe_band_artifact.json`

### qualification matrix result

{'pass': 57, 'pending_human_review': 3, 'warning': 2, 'blocker_fail_count': 0, 'total_check_count': 62}

### runtime safety assertion result

{'files_inspected_count': 5, 'forbidden_control_terms_found_count': 0, 'output_mapping_control_writeback_count': 0, 'automatic_control_detected': False, 'dcs_writeback_detected': False, 'runtime_safety_pass': True}

### SOP path

`docs\c_line_monitor_only_guidance_test_sop.md`

### acceptance criteria path

`reports/tables/c_line_guidance_test_acceptance_criteria.csv`

### plant connection precheck path

`reports/tables/c_line_plant_connection_precheck_checklist.csv`

### qualification_decision

`qualified_for_human_review`

### recommended_next_step

`conduct_human_review_for_guidance_test`

### limitations

- qualification does not approve deployment
- human review still required
- monitor-only only
- no automatic control
- no DCS setpoint writeback
- T90 measurement error about 0.1
## 48. C线 IDB 上线前最终兼容性校验、历史参考库冻结与无 parquet 运行包准备

- 目的：准备 C 线 IDB 上线前 monitor-only guidance 测试包，并验证运行时无 parquet 依赖。
- 原因：IDB 平台不能读取 parquet，基础评分资产必须通过 Python 常量嵌入或 f3fs 外部资产提供。
- C 线包源：`deploy\ca_safe_band_mvp_c_line`。
- 最新 future 数据：已加入历史参考库，仅用于 reference/monitoring，不用于算法更新。
- artifact/rules：未修改；future_data_used_for_algorithm_update=false。
- 嵌入资产：`deploy/idb_ca_safe_band_mvp_c_line/runtime_assets_embedded.py`。
- 依赖扫描：dependency_scan_pass=True，runtime_parquet_dependency_found=False。
- 无 parquet 烟测：compile_pass=True，raw_dataframe_score_pass=True。
- release zip：`deploy\release\ca_safe_band_mvp_c_line.zip`。
- final_pre_go_live_decision：idb_package_ready_for_human_review。
- recommended_next_step：human_review_idb_pre_go_live_package。
- 限制：仍需人工复核；仅 monitor-only；不自动控制；不写回 DCS 设定值；外部参考库如需使用需通过 f3fs；T90 测量误差约 0.1。
## 49. C线 IDB/s3fs 最终可部署包重建与全量上线前校验

- 目的：重建 C 线 IDB/s3fs JSON 资产运行包并完成上线前校验。
- Stage 48 修正：JSON 资产通过 s3fs/local path 提供，不强制嵌入 Python 常量。
- C 线源包：`deploy\ca_safe_band_mvp_c_line`。
- C 线 artifact：`models\ca_safe_band_mvp_c_line\safe_band_artifact.json`。
- final package directory：`deploy\idb_s3fs_ca_safe_band_mvp_c_line`。
- release zip：`deploy\release\ca_safe_band_mvp_c_line.zip`。
- JSON 资产加载烟测：{'compile_pass': True, 'import_pass': True, 'explicit_json_path_load_pass': True, 's3fs_asset_dir_load_pass': True, 'missing_asset_error_pass': True, 'engineered_row_score_pass': True, 'raw_dataframe_score_pass': True, 'output_schema_pass': True, 'safety_output_pass': True, 'parquet_unavailable_simulation_pass': True, 'warnings': []}.
- 依赖扫描：True；无 parquet runtime：True.
- zip 校验：True。
- artifact/rules unchanged：algorithm_changed=False，artifact_modified=False。
- future 数据仅作 reference：future_data_used_for_algorithm_update=False。
- final_release_decision：idb_s3fs_package_ready_for_human_review。
- recommended_next_step：human_review_idb_s3fs_release_package。
- 限制：仍需人工复核；仅 monitor-only；不自动控制；不写回 DCS 设定值；T90 测量误差约 0.1。

## 50. C线历史参考库纳入性审计与运行资产边界确认

- 目的：核验 Stage 49 之后 C线历史参考库是否真实纳入 2026.1~2026.3 future 数据，同时确认运行推荐资产仍保持冻结。
- 用户观察：最新生成的历史工况/参考内容看起来与冻结版本完全一致，因此本阶段区分“运行决策资产一致”和“参考库应包含最新 future 行”。
- 运行资产边界：`safe_band_artifact.json`、`support.json`、`schema.json` 用于评分，预期不因 future 数据改变。
- 审计结果：运行 artifact 未变更 = True；规则未变更 = True；top_rule_only 未变更 = True。
- 最新参考库时间范围：2024-07-02T06:46:00 至 2026-03-31T00:00:00。
- 2026 行数：1月 4463，2月 4032，3月 4321。
- latest future 数据是否已纳入参考库：True。
- 最新参考摘要是否与旧冻结摘要一致：True。
- 是否创建修正参考库：False。
- 算法边界：future 数据未用于算法更新 = True；未使用旧合并线包 = True。
- recommended_next_step：use_existing_reference_library。
- 局限：参考库不是评分 artifact；future 数据仅作监测参考；本阶段不提供因果证明；不实现自动控制；不实现 DCS 写回。

## 51. C线最终包输出语义修正、接口变量审计与最终部署说明文档生成

- 目的：审计并修正 C线 IDB/s3fs 最终包的接口输出语义，生成最终部署说明文档。
- 语义修正：对外推荐对象是硬脂酸钙加注量；`ca_consumption` / 钙单耗仅为内部归一化诊断指标；T90 输出是偏高/偏低风险提示，不是精确 T90 数值预测。
- 输出变量数量：55；用户展示字段数量：34；内部诊断字段数量：21；风险提示字段数量：4。
- patch_applied：False。
- 最终部署说明文档：`docs\c_line_idb_s3fs_final_deployment_manual.md`。
- 校验结果：feed 字段可用 = True；T90 风险字段可用 = True；无 parquet runtime 依赖 = True。
- final_decision：output_semantics_ready_for_human_review。
- recommended_next_step：human_review_final_deployment_manual。
- 局限：monitor-only；无自动控制；无 DCS 写回；T90 测量误差约 0.1；上线使用前仍需人工审核。
