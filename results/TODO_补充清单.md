# TODO 补充清单

旧实验中已完成、需在 B1-B7 体系下重新生成的分析项。按执行顺序排列。

## 1. 混淆矩阵 + 每类准确率

- [ ] 选取各 Phase 代表实验，从最优 seed 的 checkpoint 生成混淆矩阵
- [ ] 每类准确率直接从 CM 对角线/行和得出
- 建议选取：B1 E1-02(C-BESD)、E1-05(FAU)、E1-08(IEMOCAP)；B5 E2-01~03；B6 E6-04(MeanPool最优)；B7 E7-03(最佳迁移)
- 脚本：`scripts/plot_confusion_matrix.py`
- 输出到：`results/figures/` + `paper_draft/figures/`

## 2. XAI 可视化 + APC 指标（一起做）

- [ ] 从最优 checkpoint 重新提取注意力权重 + 韵律特征
- [ ] 生成三联图（波形+F0+注意力）并计算 APC_wav / APC_delta
- APC 和 XAI 共用同一次推理，不需单独跑实验
- 建议用 B6 E6-04 (FAU MeanPool 最优) 或 B1 E1-02 (C-BESD 最优)
- 当前 `results/analysis/xai_*` 和 `apc_metrics.json` 来自旧 exp2，需更新
- 脚本：`src/evaluation/xai_visualizer.py`

## 3. Layer Fusion 权重图

- [ ] 从 B1/B4 最优 checkpoint 提取 12 层权重，画 bar chart
- 当前 `results/analysis/layer_weights.json` 来自旧 exp2，需更新
- 脚本：`scripts/plot_layer_weights.py`

## 4. FD vs Accuracy 图

- [ ] 用 `canonical_fd_pairs.json` + B1-B7 各实验 WA 更新散点图
- 数据已就绪，只需重新跑出图脚本
- 脚本：`scripts/plot_fd_vs_accuracy.py`

## 5. t-SNE 可视化 ✅ 已完成

- [x] B1 E1-02/E1-05/E1-08 checkpoint 的 Before/After t-SNE 对比
- 输出：`paper_draft/figures/fig_tsne_*.png` (4 张)
- 方法：raw frozen WavLM mean-pool vs 完整 pipeline penultimate 128-dim

## 6. bootstrap 显著性检验 — 不做

- 每个实验仅 3 seed，n=3 不足以为 bootstrap 提供可靠置信区间
- 当前以 mean±std across 3 seeds 报告即可
- 若 reviewer 要求显著性检验，需补跑 5-10 seed

## 数据就绪情况

| 数据 | 位置 | 状态 |
|------|------|------|
| 192 实验 JSON | `results/logs/` | ✅ |
| 模型权重 | `checkpoints/autodl/b1~b7/` | ✅ (151/192) |
| FD/SMMD | `results/analysis/canonical_fd_pairs.json` | ✅ |
| Layer weights | `results/analysis/layer_weights.json` | ⚠️ 待更新 |
| XAI 原始数据 | `results/analysis/xai_raw_data.npz` | ⚠️ 待更新 |
| t-SNE 图 | `paper_draft/figures/fig_tsne_*.png` | ✅ |
