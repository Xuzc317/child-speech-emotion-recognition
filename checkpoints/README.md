# checkpoints/ — 模型权重

训练好的 PyTorch 模型权重文件。由于单个 ~363 MB，已被 `.gitignore` 排除，需通过 AutoDL 同步或本地训练获取。

## autodl/ — AC 套件权重（投稿使用）

2026-05-19 在 AutoDL RTX 4090D 上训练。架构：WavLM frozen + 12层 LayerFusion + Pooling + SEMLP (~704K 可训练参数)。

| 子目录 | 对应实验 | 用途 |
|--------|---------|------|
| `exp1_self_attention/` | Exp1 | C-BESD Self-Attn，WA=92.78%，epoch 30 |
| `exp2_prosody_guided/` | Exp2 | C-BESD Prosody，WA=91.30%，epoch 10 |
| `exp3_adult_iemocap/` | Exp3 | IEMOCAP，WA=58.67%，epoch 8 |
| `exp4_zero_shot_fau/` | Exp4 | C-BESD→FAU 零样本，WA=19.56%，epoch 10 |
| `exp5_fau_indomain/` | Exp5 | FAU Prosody，WA=66.36%，epoch 3 |
| `exp5b_self_attention_fau/` | Exp5b | FAU Self-Attn，WA=66.18%，epoch 2 |
| `exp5_fau_indomain_s123/` | C1 | FAU 多种子 s=123 |
| `exp5_fau_indomain_s456/` | C1 | FAU 多种子 s=456 |
| `exp5b_self_attention_fau_s123/` | C1 | FAU-SelfAttn 多种子 s=123 |
| `exp5b_self_attention_fau_s456/` | C1 | FAU-SelfAttn 多种子 s=456 |
| `best_model.pt` | — | 早期共用权重（已被各 exp 子目录取代） |

每个子目录内有一个 `best_model.pt`（~363 MB），可通过 `torch.load(path, weights_only=False)` 加载。

## v5_622/ — 历史 Phase 5 权重

2026-05-03~04 训练，24 个 `.pth` 文件。使用旧协议（预提取特征 + 6:2:2 划分），**不可与 AC 套件数值直接对比**。

| 前缀 | 含义 |
|------|------|
| `A1_baseline` | 基线：WavLM + mean pooling |
| `A2b_adapter` | +Adapter（随机初始化） |
| `A3_prosody_only` | +Prosody Pooling（无 Adapter）← 推荐模型 |
| `B3_final` | Adapter + Prosody Pooling |
| `C1_none` | 无增强（干净训练） |
| `C2_adult` | 成人参数增强 |
| `C3_child` | 儿童约束增强 |
| `C4_extreme` | 极端参数增强 |

每个配置有 `_seed42`、`_seed123`、`_seed456` 三个种子。

## exp2_prosody_guided/ — 本地 C2 诊断用

从 `autodl/exp2_prosody_guided/` 复制，用于本地运行 `run_prosody_diagnostics.py` 计算 APC 和 Layer weights。

## last/ — 归档

旧权重（`best_ser_model.pth`、`exp1/`、`exp5b/`、`smoke/`），已被 `autodl/` 取代。
