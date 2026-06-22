# P4 WA−UAR Gap Analysis — Definitive Per-Corpus Table

> 独立计算 | 2026-06-22 | 数据源: `results/logs/E*-*.json` (192 files)
> JSON keys 读取: `"test_wa"`, `"test_uar"`, `"test_data"`

## 纳入标准

全部 192 个日志文件。分组依据: `test_data[0]` 字段值映射到语料标签。

## 唯一正确的逐语料 WA/UAR 差距表

| Corpus | N | mean WA | mean UAR | **mean WA−UAR gap** | max gap | min gap | gap std |
|--------|---|---------|----------|--------------------|---------|---------|---------|
| C-BESD | 69 | 81.40% | 81.05% | **0.35pp** | 6.65pp | −0.12pp | 1.23pp |
| FAU Aibo | 69 | 62.87% | 41.72% | **21.15pp** | 30.77pp | −3.72pp | 8.81pp |
| IEMOCAP | 54 | 58.89% | 53.96% | **4.93pp** | 16.54pp | −4.76pp | 3.34pp |
| **GLOBAL** | **192** | — | — | **9.11pp** | 30.77pp | −4.76pp | — |

- mean WA−UAR gap = mean(WA_i − UAR_i)，即先每文件算 gap 再平均
- 等价于 mean(WA) − mean(UAR)（期望的线性性质）
- 所有 std 使用 ddof=1 (sample standard deviation)

## 三个常见数字的精确含义

| 数值 | 含义 | 来源文件 |
|------|------|---------|
| **30.77pp** (≈30.8pp) | 全部 192 文件中 **单个文件** WA−UAR gap 的全局最大值 | `E6-07_s123.json`: WA=67.96%, UAR=37.19% |
| **9.11pp** | 全部 192 文件 WA−UAR gap 的 **算术均值** | 全局统计量 |
| **21.15pp** | 仅 FAU Aibo 测试集 69 文件的 WA−UAR gap 的 **算术均值** | FAU 语料统计量 |

### 三者关系

```
9.11pp = (C-BESD贡献 + FAU贡献 + IEMOCAP贡献) / 192
       = (69×0.35 + 69×21.15 + 54×4.93) / 192
       = 9.11pp
```

9.11pp 受 FAU 的巨大 gap (21.15pp) 主导，但被 C-BESD 的接近零 gap (0.35pp) 稀释。

C-BESD 的 0.35pp 几乎为零（6 类均衡），FAU 的 21.15pp 反映 4 类严重不均衡，
IEMOCAP 的 4.93pp 反映 4 类轻度不均衡。

## 验证方法

独立 Python 脚本，不复用任何前 agent 代码。步骤:
1. `os.listdir(results/logs)` → 192 个 JSON
2. 每个文件 `json.load` → 读 `test_wa`, `test_uar`, `test_data`
3. 按 `test_data[0]` 分组
4. 计算各组的 mean(WA_i − UAR_i)
5. 验证: mean(WA) − mean(UAR) == mean(WA_i − UAR_i)
