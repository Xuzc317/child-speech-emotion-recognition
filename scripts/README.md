# scripts/ — 脚本工具集

## 实验完整性

| 脚本 | 用途 |
|------|------|
| `verify_all_192.py` | **全量校验**：覆盖 B1-B7 全部 192 runs |
| `verify_experiments.py` | 旧版校验（仅 B1/B3） |
| `verify_experiment_jsons.py` | 12 路径 CANONICAL 验收 |

## 云端管理

| 脚本 | 用途 |
|------|------|
| `tmp_paramiko_autodl_runner.py` | SSH 自动化：pull/push/train/status |
| `download_checkpoints.py` | **下载云端权重**：B5/B6/B7 + B4 补缺 (64 files) |
| `launch_b1.sh` ~ `launch_b7.sh` | 云端批量启动脚本 (B1-B7) |

## 论文出图

| 脚本 | 用途 |
|------|------|
| `generate_paper_figures.py` | fig01–fig05 |
| `generate_architecture_figure.py` | fig00 系统架构图 |
| `plot_confusion_matrix.py` | 混淆矩阵 |
| `plot_layer_weights.py` | fig06 层融合权重 |

## 诊断与分析

| 脚本 | 用途 |
|------|------|
| `compute_canonical_fd.py` | FD/SMMD 计算 |
| `extract_diagnostics.py` | 诊断提取 |
| `eval_zeroshot.py` | 零样本评估 |

## archive/ — 历史脚本

旧版脚本（旧协议 Phase 3-5 训练/分析），60+ 个文件。
