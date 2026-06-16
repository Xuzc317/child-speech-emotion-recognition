# results_remote/ — AutoDL 云端完整结果

从 AutoDL 服务器拉取的原始输出。**这是 192 个 B1-B7 实验的权威结果存储位置。**

## results/logs/ — 全量实验 JSON (192 runs)

| Phase | E系列 | 文件前缀 | 数量 |
|-------|-------|---------|------|
| B1 | E1 | `E1-01` ~ `E1-09` × 3 seeds | 27 |
| B2 | E3 | `E3-01` ~ `E3-18` | 18 |
| B3 | E4 | `E4-01` ~ `E4-12` × 3 seeds | 36 |
| B4 | E5 | `E5-01` ~ `E5-09` (含 layer variants) | 54 |
| B5 | E2 | `E2-01` ~ `E2-03` × 3 seeds | 9 |
| B6 | E6 | `E6-01` ~ `E6-10` × 3 seeds | 30 |
| B7 | E7 | `E7-01` ~ `E7-06` × 3 seeds | 18 |

**总计**: 192 JSON 文件，另有 14 个额外文件。

### 校验命令

```bash
python scripts/verify_all_192.py
```

## results/figures/ — 混淆矩阵

云端生成的混淆矩阵 PNG/PDF/JSON。

## training_logs/ — 训练日志

云端训练过程的 Shell 输出日志。

> 同步命令：`python scripts/tmp_paramiko_autodl_runner.py --pull-all`
