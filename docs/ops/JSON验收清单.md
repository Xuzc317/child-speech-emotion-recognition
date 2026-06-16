# JSON 验收清单

> 更新: 2026-06-16

## 验收命令

```bash
# 全量 B1-B7 192 runs 校验
python scripts/verify_all_192.py

# 查看各 Phase 汇总
python scripts/verify_all_192.py | grep -E "COMPLETE|GAPS|GRAND"
```

## 预期结果

| Phase | 文件数 | 预期状态 |
|-------|--------|---------|
| B1 (E1) | 27 | COMPLETE |
| B2 (E3) | 18 | COMPLETE |
| B3 (E4) | 36 | COMPLETE |
| B4 (E5) | 54 | COMPLETE |
| B5 (E2) | 9 | COMPLETE |
| B6 (E6) | 30 | COMPLETE |
| B7 (E7) | 18 | COMPLETE |
| **合计** | **192** | |

## 数据位置

- **权威 JSON**: `results/logs/E*-*.json`
- **分析数据**: `results/analysis/`
- **训练日志**: `results/training_logs/`
