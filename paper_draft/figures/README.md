# paper_draft/figures/ — 论文配图

按类型分目录整理，PNG + PDF 双格式。

## 目录结构

| 目录 | 内容 | 实验 |
|------|------|------|
| `confusion_matrices/` | 混淆矩阵 8组 | E1-02/05/09, E2-01/02/03, E6-04, E7-03 |
| `tsne/` | t-SNE Before/After 4张 | E1-02/05/08 |
| `bubbles/` | 声学气泡图 F0×RMS 8张 | 3数据集 + 合并 |
| `xai/` | XAI 显著性图 3张 | E1-02, E6-04, 三联图 |
| `layer_weights/` | Layer Fusion 权重 1组 | E1-02 |
| `architecture/` | 系统架构图 1组 | — |
| `paper_figures/` | 论文主图 fig01-05 | 全量 |
| `archive/` | 历史版本 (旧协议68文件) | — |

## 元数据

- `FIGURES_MANIFEST.json` — 完整配图清单及数据来源
- 混淆矩阵 JSON 数据: `results/figures/cm_*.json`
- XAI 原始数据: `results/analysis/xai_chart_data.json`
