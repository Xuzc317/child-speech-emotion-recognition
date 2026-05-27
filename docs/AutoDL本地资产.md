# AutoDL 产物本地保存情况（审计）

> 审计日期：2026-05-19  
> 结论：**论文可用的数值摘要已对齐本地；原始训练产物（权重、完整日志目录）基本未保留。**

---

## 总览

| 类别 | AutoDL 上应有 | 本地是否保存 | 路径 / 说明 |
|------|---------------|:------------:|-------------|
| **实验结果 JSON（6 组）** | `results/logs/exp*.json` | ✅ 已从远端同步 | `results_remote/` → `merge_remote_logs.py`；含完整 `best_val_wa` |
| **XAI 原始数据** | `xai_raw_data.npz` | ✅ | `publication_package/xai_raw_data.npz` |
| **APC 指标** | `apc_metrics.json` | ✅ | `results/logs/apc_metrics.json`（APC_wav≈0.698） |
| **汇总 CSV** | — | ✅ | `publication_package/experiment_results.csv`（含 Exp5b） |
| **Layer 权重 JSON** | `layer_weights.json` | ✅ | argmax Layer 8（AutoDL 同步） |
| **训练权重 checkpoint** | `checkpoints/exp*/best_model.pt` | ⚠️ | 远端 `checkpoints/` 目录为空（历史 run 共用单文件且未保留）；需按新脚本 `--output_dir` 重训 |
| **训练过程日志** | `matrix_run.log`、`matrix_cbesd.log` | ❌ | 未拉回；`results_remote/` 目录不存在 |
| **远端完整 JSON 备份** | 同上 | ❌ | 曾计划同步到 `results_remote/`（已 gitignore），当前机器上无此目录 |
| **数据集（C-BESD / FAU）** | 远端 `/root/autodl-tmp/...` | ❌ 未通过脚本同步 | 需使用你本机原有数据路径（`dataset.py` 默认 Windows 路径） |
| **epoch 级训练曲线** | 无专门导出 | ❌ | `train.py` 仅写最终 JSON，无 per-epoch history 文件 |

---

## Paramiko 脚本实际下载范围

`scripts/tmp_paramiko_autodl_runner.py` 的 `--auto-resume` / 同步逻辑**仅包含**：

1. `results/logs/*.json` → `results_remote/results/logs/`（本地多已缺失）
2. `publication_package/xai_raw_data.npz`（或 `results/xai_raw_data.npz`）

**不包含**：`checkpoints/`、`matrix_run.log`、原始语料 tar。

---

## 与旧版 `experiments/v5_622/` 的关系

| 内容 | 说明 |
|------|------|
| `experiments/v5_622/*.json`、`.pth`（若曾有） | **上一阶段（6:2:2 协议）** 实验，不是本次 AutoDL Exp1–5b 矩阵 |
| 论文 FD 表中的 6.87、16.33 等 | 部分来自 v5_622 / 旧文档，与当前 `results/distribution_shift.json`（FD=12.33）**不是同一统计设定** |

---

## 若需补全本地资产（阶段 B1）

在 AutoDL **实例与磁盘未销毁** 的前提下，可扩展同步：

```bash
# 需先确认远端仍可 SSH
python scripts/tmp_paramiko_autodl_runner.py --download-only
# 建议后续增加：--sync-checkpoints（待实现）
```

远端预期路径：

- 代码：`/root/autodl-tmp/d-ser/`
- C-BESD：`/root/autodl-tmp/datasets/BESD/BESD/MY`
- FAU：`/root/autodl-tmp/IS2009EmotionChallenge/`

若实例已释放，只能 **本地按相同协议重训** 或接受「仅保留论文矩阵 JSON + XAI npz」。

---

## 当前可继续的工作（无需 AutoDL）

- A5：统一 FD–准确率表注、APC 验收、标注 layer_weights 待 checkpoint  
- A6：PaperViz 配图（用 `publication_package/` 已有数据）  
- A7：补写引言 / 摘要 / 结论  
