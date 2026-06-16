# results/ — 本地实验结果与分析

本地生成的分析数据和早期结果快照。投稿权威数据参见 `CLAUDE.md` 和 `results_remote/results/logs/`。

## 根目录文件

| 文件 | 内容 |
|------|------|
| `canonical_fd_pairs.json` | FD/SMMD 权威值：C-BESD vs IEMOCAP (FD=7.20), vs FAU (FD=8.50) |
| `distribution_shift.json` | C-BESD vs CREMA-D 的 FD=16.33 / SMMD=0.412 |
| `layer_weights.json` | WavLM 12 层融合权重，argmax L9 (1-based) |
| `xai_final.png` | XAI 可视化 |
| `xai_raw_data.npz` | XAI 原始数据 |
| `AC_SUITE_SUMMARY.json` | AC 套件汇总 |

## logs/ — 早期结果快照

旧版实验 (AC Suite 早期，2026-05-26~27) 的 JSON 结果。

| 文件 | 含义 |
|------|------|
| `exp1_self_attention.json` | C-BESD Self-Attn, WA=92.78% |
| `exp2_prosody_guided.json` | C-BESD Prosody, WA=91.30% |
| `exp3_adult_iemocap.json` | IEMOCAP, WA=58.67% |
| `exp4_zero_shot_fau.json` | Zero-shot C-BESD→FAU, WA=19.56% |
| `exp5_fau_indomain.json` | FAU Prosody, WA=66.36% |
| `exp5b_self_attention_fau.json` | FAU Self-Attn, WA=66.18% |
| `fau_multiseed_summary.json` | FAU 多 seed 汇总 |
| `apc_metrics.json` | APC_wav=0.718 |
| `DATA_FREEZE.json` | 数据冻结记录 |

> ⚠️ 完整的 192 实验 JSON 在 `results_remote/results/logs/` — 使用 `python scripts/verify_all_192.py` 校验。
