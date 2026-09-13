# RCT 异常检测结果摘要

> 数据集：1832 序列 / 122 材料 / 29279 帧；`contamination=0.05`，PCA 20 维（图像降采样 64×64）+ 力统计 4 维 + 序列长度 + 帧间差。

## 检测器总览

| 方法 | 异常数 | 异常率 |
|---|---|---|
| IsolationForest | 92 / 1832 | 5.02% |
| Mahalanobis (chi² 阈值) | 195 / 1832 | 10.64% |

IsolationForest 的异常率与设定的 `contamination=0.05` 一致；Mahalanobis 偏高（10.64%），说明特征分布尾部比 chi² 假设更重，协方差法在该分布上更激进。两者并非互斥——IF 标记的 92 个全部被 Mahalanobis 也标记（IF 是保守子集）。

## 分层异常率对比表（IsolationForest）

| 材料类别 | n | n_anomaly | n_normal | anomaly_ratio |
|---|---:|---:|---:|---:|
| Paper_Cardboard | 613 | 37 | 576 | 0.0604 |
| Plastic_Rubber | 471 | 28 | 443 | 0.0594 |
| Small_Items | 42 | 2 | 40 | 0.0476 |
| Metal | 376 | 17 | 359 | 0.0452 |
| Textiles_Leather | 147 | 5 | 142 | 0.0340 |
| Wood_Bamboo_Cork | 105 | 2 | 103 | 0.0190 |
| Crafts | 63 | 1 | 62 | 0.0159 |
| Unknown | 15 | 0 | 15 | 0.0000 |

**chi² 独立性检验**：`chi2 = 7.666`，`p = 0.363`，`dof = 7`

→ p > 0.05，**各材料类别的异常率无统计学显著差异**。异常分布与材料类别相对独立，未出现某类材料系统性异常的迹象（这正是一个"无标注、无监督"质检所期望的——异常主要是个体离群而非类别效应）。

## 两方法一致性（Cohen's kappa）

| 指标 | 值 |
|---|---|
| Cohen's kappa | 0.615 |
| 整体一致率 | 0.944 |
| 双方均判异常 | 92 |
| 双方均判正常 | 1637 |
| 不一致 | 103 |

kappa = 0.615 属"实质性一致"（Landis-Koch 0.41–0.60 实质性，0.61–0.80 实质性）。103 个不一致全部是"Mahalanobis 判异常、IF 判正常"——即 Mahalanobis 的额外告警，IF 对这部分更保守。两条检测线可互为补充：Mahalanobis 用作高召回预警，IF 用作高精度确认。

## 与参考的对照

参考提到孤立森林在传感器异常检测中无需标注、高准确率、低误报。本次 IF 异常率 5.02% 与设定 contamination 一致，且其标记集合是 Mahalanobis 的子集，符合"低误报"的特征；后续若拿到少量标注，可据此评估真实 precision/recall 并校准 contamination。

## 产物

- `reports/anomaly_results.csv` — 每序列 6 列（sequence_id, material, iso_label, iso_score, mahal_label, mahal_score）
- `reports/anomaly_summary.md` — 本摘要
- `src/tactile_qc/anomaly.py` — 模块实现（+ `tests/test_anomaly.py` 11 测试全过）
