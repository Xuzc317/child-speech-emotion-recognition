# 辅助指标验收状态（A5）

| 指标 | 论文引用 | 本地文件 | 状态 |
|------|----------|----------|------|
| APC_wav | 0.698 | `logs/apc_metrics.json` → 0.6979 | ✅ 一致 |
| APC_δ | −0.147 | −0.1467 | ✅ 一致 |
| FD 表（4 行） | §5.4 | `fd_accuracy_table.json` | ✅ 已整理来源注记 |
| SMMD | 0.41（C-BESD vs FAU） | `distribution_shift.json` → 0.369 | ⚠️ 需在正文统一引用哪次计算 |
| Layer fusion peak | **Layer 8** | `layer_weights.json` → argmax_layer=**8**, entropy≈2.43 | ✅ AutoDL 同步；`fig06_layer_fusion_weights` |

## 备注

- 混淆矩阵仍缺 per-experiment `.pt`（远端 `checkpoints/` 为空；新 run 已改为 `--output_dir checkpoints/<exp_name>`）。

## 下一步（A6 前）

1. 配图可先用：`xai_raw_data.npz`、`experiment_results.csv`、`fd_accuracy_table.json`  
2. 层权重柱状图：暂缓，或先用 v5_622 历史图（需在图注标明协议不同）  
3. 若恢复 checkpoint：运行 `python scripts/export_layer_weights.py`（待添加）
