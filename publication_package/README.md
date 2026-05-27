# publication_package/ — 投稿资产包

投稿所需的数据、图表和表格的集合副本。与 `results/` 保持同步。

## 根目录文件

| 文件 | 内容 |
|------|------|
| `experiment_results.csv` | **主实验矩阵 CSV**（6 行：Exp1–Exp5b），含 WA/UAR/best_epoch/reg_profile |
| `fd_accuracy_table.json` | **FD-Accuracy 表数据源**（fig03）。2026-05-27 已验证，所有 FD 值标注了来源 |
| `distribution_shift.json` | C-BESD vs CREMA-D 的分布偏移（FD=16.33, SMMD=0.412） |
| `layer_weights.json` | WavLM 12 层融合权重（与 `results/layer_weights.json` 一致） |
| `xai_raw_data.npz` | XAI 原始数据（波形、F0、能量、注意力权重），用于重绘 fig04 |

## logs/ — 实验 JSON 副本

与 `results/logs/` 同步的投稿副本。包含 11 个 JSON 文件：
- 6 个主实验（exp1–exp5b）
- 4 个 FAU 多种子（exp5/5b _s123/_s456）
- `apc_metrics.json`（APC=0.718）

> 注意：`experiment_results.csv` 和 `fd_accuracy_table.json` 是投稿独有的汇总裁体，不在 `results/` 中。

## last/ — 历史归档

前一个 Agent 在 2026-05-26 的两个时间点生成的旧版文件快照（`20260526_171321/` 和 `20260526_172433/`）。
