# results/ — 实验结果（整合后）

B1-B7 全部 192 个实验的 JSON 结果、分析数据和训练日志。

## 结构

```
results/
├── logs/                  # B1-B7 全量实验 JSON (192 files)
├── analysis/              # FD、XAI、layer weights 等分析数据
├── figures/               # 混淆矩阵（待生成）
├── training_logs/         # 云端训练终端日志
├── archive/               # 旧实验数据归档
├── TODO_补充清单.md        # 待补充分析项
└── README.md
```

## logs/ — 实验 JSON

| Phase | E系列 | 文件数 |
|-------|-------|--------|
| B1 | E1 | 27 |
| B2 | E3 | 18 |
| B3 | E4 | 36 |
| B4 | E5 | 54 |
| B5 | E2 | 9 |
| B6 | E6 | 30 |
| B7 | E7 | 18 |
| **总计** | | **192** |

校验：`python scripts/verify_all_192.py`

## analysis/ — 分析数据

| 文件 | 内容 |
|------|------|
| `canonical_fd_pairs.json` | FD/SMMD 权威值 |
| `layer_weights.json` | WavLM 12 层融合权重 |
| `xai_raw_data.npz` | XAI 原始数据 |
| `xai_final.png` | XAI 可视化 |
| `AC_SUITE_SUMMARY.json` | AC 套件汇总 |

## archive/ — 历史数据

| 目录 | 内容 |
|------|------|
| `old_experiments/` | exp1~exp5b JSON + apc_metrics + fau_multiseed |
| `old_figures/` | 旧实验混淆矩阵 (PNG/PDF/JSON) |
| `old_logs/` | 早期日志归档 |
