# AC 实验套件协议（国际会议可复现）

> **协议 ID**: `ac_suite_2026-05`  
> **固定主种子**: `42`（与 `speaker_splitter` / `get_dataloaders` 一致）  
> **FAU 多种子**: `42, 123, 456`（C1 稳健性）  
> **远程执行**: `bash scripts/autodl_ac_suite.sh`（由 Paramiko `--run-ac-suite` 启动）

---

## 共同设置（所有训练）

| 项 | 值 |
|----|-----|
| SSL | WavLM Base+，冻结 |
| 优化 | AdamW，lr=3e-4，CosineAnnealing，max 100 epochs |
| 早停 | patience=15（验证集 WA） |
| batch_size | 16 |
| 划分 | 说话人独立 hash 划分（seed 写入 `speaker_splitter`） |
| 类别 | angry / happy / neutral / sad |

### 正则配置

| profile | 实验 | weight_decay | label_smoothing | pooling_dropout | grad_clip |
|---------|------|--------------|-----------------|-----------------|-----------|
| `default` | Exp1–4 | 1e-3 | 0.1 | 0.0 | — |
| `fau` | Exp5/5b | 5e-3 | 0.15 | 0.3 | 1.0 |

---

## A：可复现资产（checkpoint + 混淆矩阵）

| ID | 目的 | 训练 | 测试 | Pooling | 输出 |
|----|------|------|------|---------|------|
| **A1-Exp1** | C-BESD 上 Self-Attn 完整权重 | C-BESD | C-BESD | self_attention | `checkpoints/exp1_self_attention/` |
| **A1-Exp2** | Prosody 主模型（若已有则跳过训练） | C-BESD | C-BESD | prosody_guided | `checkpoints/exp2_prosody_guided/` |
| **A1-Exp3** | 成人 IEMOCAP 对照 | IEMOCAP | IEMOCAP | prosody_guided | `checkpoints/exp3_adult_iemocap/` |
| **A1-Exp4** | 跨域零样本（演绎→自发） | C-BESD | FAU-Aibo | prosody_guided | `checkpoints/exp4_zero_shot_fau/` |
| **A1-Exp5** | FAU 域内 Prosody | FAU | FAU | prosody_guided | `checkpoints/exp5_fau_indomain/` |
| **A1-Exp5b** | FAU 域内 Self-Attn | FAU | FAU | self_attention | `checkpoints/exp5b_self_attention_fau/` |
| **A2** | 各实验测试集混淆矩阵（与 JSON 指标同 split、同 seed） | — | — | — | `results/figures/confusion_<exp>.*` |

**验收**：每个 `results/logs/<exp>.json` 含 `seed`、`test_wa`、`test_uar`、`best_val_wa`；混淆矩阵 JSON 中 `confusion_matrix` 行和与测试集样本数一致。

---

## C：稳健性与可解释性

| ID | 目的 | 设置 | 输出 |
|----|------|------|------|
| **C1** | FAU 上报告 mean±std | Exp5/5b × seeds {42,123,456} | `results/logs/exp5*_s*.json`，`fau_multiseed_summary.json` |
| **C2** | APC + Layer8 + XAI 与 Exp2 一致 | Exp2 prosody checkpoint，C-BESD test 单条样本 | `layer_weights.json`，`apc_metrics.json`，`publication_package/xai_raw_data.npz` |

**C2 说明**：APC 在**同一条** C-BESD 测试 utterance 上计算（`get_dataloaders(..., seed=42)` 的首个 batch），与论文「韵律–注意力对齐」叙述一致。

---

## 论文主表数值

主表 6 组结果以 **`scripts/verify_experiment_jsons.py` 中 CANONICAL** 为投稿基准；本套件重训后若数值漂移 >0.5pp，需更新正文或附表并注明 `protocol` 字段。

---

## 本地同步

```bash
python scripts/tmp_paramiko_autodl_runner.py --pull-all
python scripts/merge_remote_logs.py
python scripts/aggregate_fau_multiseed.py
```
