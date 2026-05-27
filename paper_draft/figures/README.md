# paper_draft/figures/ — 论文配图

投稿论文的所有图片文件。每个图号同时提供 PNG（位图）和 PDF（矢量）两种格式，以及 JSON 元数据（仅混淆矩阵）。

## 正文配图（fig01–fig08）

| 图号 | 文件前缀 | 内容 | 数据源 |
|------|---------|------|--------|
| fig01 | `fig01_main_matrix_wa_uar` | 6 组实验主结果矩阵（WA+UAR 柱状图） | `experiment_results.csv` |
| fig02 | `fig02_fau_prosody_vs_selfattn` | FAU 上两种池化方式对比 | `experiment_results.csv` |
| fig03 | `fig03_fd_vs_accuracy` | FD-Accuracy 统一框架散点图 | `fd_accuracy_table.json` |
| fig04 | `fig04_xai_saliency_triple` | XAI 三联图（波形+F0+注意力权重） | `xai_raw_data.npz` |
| fig05 | `fig05_cbesd_selfattn_vs_prosody` | C-BESD 上两种池化方式对比 | `experiment_results.csv` |
| fig07 | `fig07_confusion_exp1_selfattn` | Exp1 Self-Attn C-BESD 混淆矩阵（N=540） | 模型预测 |
| fig08 | `fig08_confusion_exp2_prosody` | Exp2 Prosody C-BESD 混淆矩阵（N=540） | 模型预测 |

## 附录配图（figA1–figA4）

| 图号 | 文件前缀 | 内容 | N |
|------|---------|------|----|
| figA1 | `figA1_confusion_exp3_iemocap` | Exp3 IEMOCAP 混淆矩阵 | 2371 |
| figA2 | `figA2_confusion_exp4_zero_shot` | Exp4 零样本 C-BESD→FAU 混淆矩阵 | 3389 |
| figA3 | `figA3_confusion_exp5_fau_prosody` | Exp5 FAU Prosody 混淆矩阵 | 3389 |
| figA4 | `figA4_confusion_exp5b_fau_selfattn` | Exp5b FAU Self-Attn 混淆矩阵 | 3389 |

## 缺失的图

| 图号 | 说明 |
|------|------|
| fig00 | 系统架构图 — 需运行 `scripts/generate_architecture_figure.py` |
| fig06 | WavLM 层融合权重图 — 数据在 `results/layer_weights.json`，需运行 `scripts/plot_layer_weights.py` |

## 其他文件

| 文件 | 内容 |
|------|------|
| `FIGURES_MANIFEST.json` | 配图清单（记录了哪些图已生成、哪些缺失） |
| `last/` | 历史版本的图和清单归档 |
