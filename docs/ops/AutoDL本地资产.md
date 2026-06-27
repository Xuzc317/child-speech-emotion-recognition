# AutoDL 本地资产清单

> 更新: 2026-06-16

## 已同步到本地的内容

| 资产 | 本地路径 | 大小 |
|------|---------|------|
| 实验 JSON (192 runs) | `results/logs/` | ~1 MB |
| 模型权重 (151个) | `checkpoints/autodl/b1~b7/` | ~55 GB |
| 训练日志 | `results/training_logs/` | ~1 MB |
| 混淆矩阵 (旧) | `results/archive/old_figures/` | ~5 MB |
| XAI 数据 | `results/analysis/` | ~1 MB |

## 云端保留

| 资产 | 路径 | 说明 |
|------|------|------|
| 实验 JSON | `/root/autodl-tmp/d-ser/results/logs/` | 开机后可二次拉取 |
| 模型权重 | `/root/autodl-tmp/d-ser/checkpoints/` | B1-B7 全量，开机后可下载 |
| 数据集 | `/root/autodl-tmp/datasets/` | C-BESD, IEMOCAP, FAU Aibo |

## 同步命令

```bash
python scripts/tmp_paramiko_autodl_runner.py --pull-all
python scripts/download_checkpoints.py
```
