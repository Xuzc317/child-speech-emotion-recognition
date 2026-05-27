# results/ — 权威实验结果

本目录存放 AC 套件 (`ac_suite_2026-05`) 的所有最终实验结果。**投稿论文中引用的任何数字必须来自本目录的 JSON 文件。** 验收命令：`python scripts/verify_experiment_jsons.py`

## 根目录文件

| 文件 | 内容 |
|------|------|
| `canonical_fd_pairs.json` | **2026-05-27 AutoDL GPU 重算的 FD/SMMD**。C-BESD vs IEMOCAP (FD=7.20), C-BESD vs FAU Aibo (FD=8.50) |
| `distribution_shift.json` | C-BESD vs **CREMA-D** 的 FD=16.33 / SMMD=0.412。注意：不是 FAU Aibo！ |
| `layer_weights.json` | WavLM 12 层融合权重。argmax 1-based=9 (0-based=8), entropy=2.48 |
| `xai_final.png` | XAI 可解释性可视化图（韵律+注意力对照） |
| `xai_raw_data.npz` | XAI 原始数据（波形、F0、能量、注意力权重） |
| `xai_sample_visualization.png` | 单样本 XAI 可视化 |

## logs/ — 实验 JSON

| 文件 | 实验 | 含义 |
|------|------|------|
| `exp1_self_attention.json` | Exp1 | C-BESD 域内，Self-Attention 池化，WA=92.78% |
| `exp2_prosody_guided.json` | Exp2 | C-BESD 域内，韵律引导池化，WA=91.30% |
| `exp3_adult_iemocap.json` | Exp3 | IEMOCAP 成人域内，韵律池化，WA=58.67% |
| `exp4_zero_shot_fau.json` | Exp4 | C-BESD→FAU 零样本，WA=19.56% (N=3389 test-only) |
| `exp5_fau_indomain.json` | Exp5 | FAU 域内，韵律池化，WA=66.36% |
| `exp5b_self_attention_fau.json` | Exp5b | FAU 域内，Self-Attention 池化，WA=66.18% |
| `exp5_fau_indomain_s123.json` | C1 | Exp5 种子 123 重复：WA=63.65% |
| `exp5_fau_indomain_s456.json` | C1 | Exp5 种子 456 重复：WA=65.44% |
| `exp5b_self_attention_fau_s123.json` | C1 | Exp5b 种子 123 重复：WA=65.76% |
| `exp5b_self_attention_fau_s456.json` | C1 | Exp5b 种子 456 重复：WA=67.44% |
| `fau_multiseed_summary.json` | C1 | FAU 多种子汇总：Exp5 65.15±1.12%, Exp5b 66.46±0.72% |
| `apc_metrics.json` | C2 | 注意力-韵律相关性：APC_wav=0.718, APC_delta=-0.144 |
| `DATA_FREEZE.json` | — | 数据冻结记录（git commit + UTC 时间戳） |

### JSON 字段说明

每个实验 JSON 包含：
- `test_wa` / `test_uar`：测试集加权准确率 / 非加权平均召回率
- `best_val_wa`：最佳验证集 WA（early stopping 依据）
- `best_epoch`：最佳 epoch 编号
- `reg_profile`：正则配置（`default` 或 `fau`）
- `train_data` / `test_data`：训练/测试语料
- `pooling_type`：池化方式（`self_attention` 或 `prosody_guided`）
- `seed`：随机种子
- `protocol`：实验协议标识 `ac_suite_2026-05`
