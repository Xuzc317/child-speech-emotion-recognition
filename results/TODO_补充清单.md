# TODO 补充清单

旧实验（exp1~exp5b）中已完成但 B1-B7 尚未覆盖的分析项。按优先级排列。

## 高优先级（论文投稿需要）

| 项目 | 旧实验状态 | B1-B7 状态 | 说明 |
|------|-----------|-----------|------|
| **混淆矩阵** | exp1~exp5b 各 1 张（共 6 张） | 未生成 | B1-B7 共 192 个实验，需选代表性实验生成 CM。建议每个 Phase 选最优配置 |
| **XAI 可视化** | 基于 exp2 样本（1 张三联图） | 未更新 | 可基于 B1/B6 最优模型重新生成 |
| **Layer Fusion 权重图** | 基于 exp2（1 张） | 未更新 | 需从 B1/B4 最优 checkpoint 重新提取 |

## 中优先级（论文补充材料）

| 项目 | 旧实验状态 | B1-B7 状态 | 说明 |
|------|-----------|-----------|------|
| **APC 指标** | 已算（APC_wav=0.718） | 需更新 | 已有 APC 数据，需用 B1-B7 最优模型重算 |
| **FD vs Accuracy 图** | 已生成 | 可用现数据更新 | 数据已就绪（canonical_fd_pairs.json） |

## 低优先级（后续）

| 项目 | 说明 |
|------|------|
| **t-SNE 可视化** | 旧实验有，B1-B7 尚未做 |
| **每类准确率统计** | 旧实验有 attention analysis，B1-B7 未做 |
| **bootstrap 显著性检验** | 旧协议做过，需迁移到 B1-B7 |

## 数据就绪情况

| 数据 | 位置 | 状态 |
|------|------|------|
| 192 实验 JSON | `results/logs/` | ✅ |
| 模型权重 | `checkpoints/autodl/b1~b7/` | ✅ (151/192) |
| FD/SMMD | `results/analysis/canonical_fd_pairs.json` | ✅ |
| Layer weights | `results/analysis/layer_weights.json` | ✅ (旧，待更新) |
| XAI 原始数据 | `results/analysis/xai_raw_data.npz` | ✅ (旧，待更新) |
