# results_remote/ — AutoDL 云端备份

通过 `paramiko --pull-all` 从 AutoDL 服务器拉取的原始输出。与 `results/` 的区别：这里是**云端直接产出**的原始副本，`results/` 是经过本地 post-pipeline 合并和验证后的最终版本。

## 根目录文件

| 文件 | 说明 |
|------|------|
| `remote_inventory.txt` | 最后一次 pull 的云端文件清单（含文件大小和时间戳） |
| `distribution_shift.json` | 早期 pull 的分布偏移副本（与 `results/` 一致） |
| `layer_weights.json` | 云端 C2 输出的层权重（与 `results/` 一致） |
| `xai_final.png` | 云端 XAI 可视化图 |
| `xai_raw_data.npz` | 云端 XAI 原始数据 |
| `exp4_zero_shot_fau.json` | ⚠️ 旧版 Exp4（split='all', WA=18.70%），已被 `results/logs/` 中的修正版取代 |
| `exp5_fau_indomain.json` | 早期 pull 副本 |
| `exp2_checkpoint.log` | Exp2 检查点日志片段 |

## results/ — 云端 results 目录的完整镜像

```
results/
├── logs/           # 与 results/logs/ 同步的 12 个 JSON
├── figures/        # 云端生成的混淆矩阵 PNG/PDF/JSON（含原始 exp2_confusion）
├── distribution_shift.json
├── layer_weights.json
├── xai_final.png
└── xai_raw_data.npz
```

## training_logs/ — 云端训练日志

| 文件 | 内容 |
|------|------|
| `matrix_run.log` | Exp3 IEMOCAP 完整训练日志（每 epoch train/val WA） |
| `matrix_cbesd.log` | Exp1/Exp2 C-BESD 训练日志（含完整 epoch 曲线） |
| `exp2_checkpoint.log` | Exp2 检查点日志 |
| `ac_suite_logs/` | **2026-05-27 新拉取**：4 个云端 Shell 脚本完整输出日志 |
| `ac_suite_logs/run_20260526_160746.log` | AC 套件首轮运行：Exp1 46 epoch + Exp2 完整日志 |
| `ac_suite_logs/resume_20260526_185400.log` | Resume: Exp5 s123 + Exp5b s123 训练 epoch 记录 |
| `ac_suite_logs/resume_20260526_223135.log` | Resume: Exp5 s456 + Exp5b s456 + CM 生成 + multiseed 汇总 |

> `matrix_cbesd.log` 是验证 Exp1 92.78% 无数据泄露的关键证据：test_wa (92.78%) ≈ best_val_wa (92.80%)。
