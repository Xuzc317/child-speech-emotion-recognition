# Independent Review Report

> **审查员**: Independent Reviewer (2026-06-22)
> **审查对象**: Phases 0–3 (文件盘点、溯源清单、数值验证、指标完整性)
> **事实来源**: `results/logs/E*-*.json` (192 文件) — 唯一权威来源
> **方法**: 所有提取逻辑独立重写，不依赖前 agent 脚本；关键数字逐 seed 从原始 JSON 读出

---

## 裁决总表

| # | 检查项 | 裁决 | 关键证据 |
|---|--------|------|---------|
| 1 | 字段数跳变 136→191 | CONFIRMED | 覆盖范围扩大：discrepancy report 仅 B1/B2/B5/B6/B7 (130 field + 6 TEXT = 136)，ledger 扩展至 B3/B4 (+55 field)，共 191 行 |
| 2 | E1-02 旗舰数字 | REFUTED (handbook) / CONFIRMED (前agent) | 92.92% = s42 单 seed，非 3-seed mean；真实 3-seed mean = 91.87% |
| 3 | E1-05 FAU 天花板 | REFUTED (handbook) / CONFIRMED (前agent) | 67.81% = s42 单 seed；真实 3-seed mean = 67.05% |
| 4a | E2-01 96.91% | CONFIRMED | s42=96.80%, s123=96.80%, s456=97.13%, mean=96.91% ✅ |
| 4b | E6-03 91.91% | CONFIRMED | s42=92.92%, s123=91.74%, s456=91.06%, mean=91.91% ✅ |
| 4c | E6-04 91.96% | CONFIRMED | s42=92.92%, s123=91.91%, s456=91.06%, mean=91.96% ✅ |
| 4d | E7-03 91.57% | CONFIRMED | s42=92.07%, s123=91.23%, s456=91.40%, mean=91.57% ✅ |
| 4e | E7-05 91.17% | CONFIRMED | s42=91.74%, s123=89.71%, s456=92.07%, mean=91.17% ✅ |
| 5a | test_uar 全 192 存在 | CONFIRMED | 独立遍历 192 文件，100% 含 test_uar |
| 5b | 预测/混淆矩阵全 0 | CONFIRMED | confusion_matrix/predictions/per_class_recall 等均为 0/192 |
| 5c | WA−UAR 差距 | CONFIRMED | 全局 mean=9.11pp；C-BESD=0.35pp, FAU=21.15pp, IEMOCAP=4.93pp |
| 5d | 信任边界 | CANNOT-VERIFY 信任链 | 见 §信任边界分析 |
| 6 | 3 missing_in_logs | CONFIRMED | B2_best=41.05%, B2_max=35.47%, B2_min=19.17% — 三者在 192 日志中均不存在 |
| 7 | 配置可信度 | REFUTED augment_condition / CONFIRMED 前agent警告 | augment_condition='C1' 在无增强实验中是代码默认值非实际值 |
| 8 | 盲抽复核 | CONFIRMED ledger提取准确 | ledger log_values 与独立提取完全一致 |

---

## §1 字段数跳变：136 → 191

### 136 字段的来源 (discrepancy_report.md)

Discrepancy report 仅覆盖以下 phase 的 WA 值对比（handbook vs log）：

| Phase | 实验 | 字段/实验 | 字段数 |
|-------|------|-----------|--------|
| B1 (E1) | 9 | 4 (mean_std, s42, s123, s456) | 36 |
| B2 (E3) | 18 | 1 (单 seed WA) | 18 |
| B5 (E2) | 3 | 4 | 12 |
| B6 (E6) | 10 | 4 | 40 |
| B7 (E7) | 6 | 4 | 24 |
| TEXT fields | 6 (B2_best/max/min WA, highest scoreboard, etc.) | — | 6 |
| **合计** | | | **136** |

### 191 字段的来源 (ledger_full.csv)

Ledger 扩展至全部 phase，新增 B3/E4 (36 fields) + B4/E5 (additional fields) + 更细粒度的 per-seed 记录：

- 191 行 = 原来的 B1/B2/B5/B6/B7 (130) + B3/E4 (48) + B4/E5 (partial) + TEXT (3)
- 一个 "field" = 一个 (experiment, metric) pair，如 `E1-01_s42` 的 WA 值

### 不一致数量：55 → 123

- 原 55 处 (26 COPY_ERR + 11 WRONG_CELL + 18 PHANTOM) — 仅覆盖 B1/B2/B5 的 handbook vs log 对比
- 新 123 处 (47 minor + 32 major + 41 severe + 3 missing_in_logs) — 扩展至全部 7 个 phase

**裁决**: CONFIRMED — 字段数跳变是覆盖范围扩大所致，定义一致。

---

## §2 旗舰数字 E1-02（C-BESD 天花板）

### 独立提取（从原始 JSON 直接读取 `test_wa` 字段）

| 文件 | JSON key `test_wa` 原始值 | 百分比 |
|------|--------------------------|--------|
| `E1-02_s42.json` | `0.9291736930860034` | **92.92%** |
| `E1-02_s123.json` | `0.9007177033492823` | **90.07%** |
| `E1-02_s456.json` | `0.9261744966442953` | **92.62%** |

### 计算

```
3-seed mean = (92.92 + 90.07 + 92.62) / 3 = 91.87%
3-seed std  = 1.56pp (sample std, n-1 denominator)
```

### 判定

- **CLAUDE.md / 权威数据手册声称: 92.92%** → 这是 **s42 单 seed 值**，不是 3-seed mean ❌
- **前 agent 声称: 91.87%** → 这是正确的 3-seed mean ✅
- **真实 3-seed mean = 91.87%** (此值来自我独立重提的三个 seed 值)

**裁决**: **REFUTED** — handbook 天花板数字 92.92% 夸大了 1.05pp（用单 seed 最高值冒充 3-seed mean）。前 agent 的 91.87% **CONFIRMED** 正确。

---

## §3 E1-05（FAU 天花板）

### 独立提取

| 文件 | `test_wa` 原始值 | 百分比 |
|------|-----------------|--------|
| `E1-05_s42.json` | `0.6780761286515197` | **67.81%** |
| `E1-05_s123.json` | `0.6654111738857501` | **66.54%** |
| `E1-05_s456.json` | `0.667890656874745` | **66.79%** |

```
3-seed mean = (67.81 + 66.54 + 66.79) / 3 = 67.05%
3-seed std  = 0.55pp (与 ledger 0.55pp 一致)
```

**判定**: 67.81% = s42 单 seed。真实 3-seed mean = **67.05%**。

**裁决**: **REFUTED** — handbook 67.81% 同样是 s42 单 seed。前 agent 67.05% **CONFIRMED**。

---

## §4 五个「✅」负荷数字

### 独立重提（均从 `test_wa` JSON key 直接读出）

| 实验 | s42 | s123 | s456 | 3-seed mean | 前agent声称 | 匹配? |
|------|-----|------|------|-------------|-----------|-------|
| E2-01 | 96.80% | 96.80% | 97.13% | **96.91%** | 96.91% | ✅ |
| E6-03 | 92.92% | 91.74% | 91.06% | **91.91%** | 91.91% | ✅ |
| E6-04 | 92.92% | 91.91% | 91.06% | **91.96%** | 91.96% | ✅ |
| E7-03 | 92.07% | 91.23% | 91.40% | **91.57%** | 91.57% | ✅ |
| E7-05 | 91.74% | 89.71% | 92.07% | **91.17%** | 91.17% | ✅ |

**裁决**: 全部五个 **CONFIRMED** — 前 agent 的提取与我的独立提取完全一致，差异 = 0.00pp。

---

## §5 Phase 3 / UAR

### 5a. test_uar 字段存在性

独立遍历全部 192 个 JSON 文件，检查 `test_uar` key：

- **Present: 192/192** (100%)
- **Missing: 0/192**

**裁决**: **CONFIRMED** — test_uar 确实存在于所有 192 个文件中。

### 5b. 预测/混淆矩阵/逐类 recall 缺失

检查的 key 名（来自全部 21 个唯一 key 的全量 inventory）：

| 指标 | 存在 | 缺失 |
|------|------|------|
| `test_wa` | 192 | 0 |
| `test_uar` | 192 | 0 |
| `best_val_wa` | 192 | 0 |
| `confusion_matrix` | 0 | 192 |
| `predictions` | 0 | 192 |
| `per_class_recall` | 0 | 192 |
| `per_class_precision` | 0 | 192 |
| `per_class_f1` | 0 | 192 |

唯一与预测/类别相关的 key 是 `label_smoothing` 和 `output_dir`（非指标）。

**裁决**: **CONFIRMED** — 所有 192 文件均不含可独立重算 UAR 的原始数据。

### 5c. FAU 实际 UAR 数值与 WA−UAR 差距

**全局 WA−UAR 差距**（192 文件计算）:

| 数据集 | 文件数 | 平均 WA−UAR 差距 |
|--------|--------|------------------|
| C-BESD | 69 | **0.35pp** (类别均衡) |
| FAU Aibo | 69 | **21.15pp** (类别不均衡) |
| IEMOCAP | 54 | **4.93pp** (轻度不均衡) |
| **全局** | **192** | **9.11pp** |

FAU 文件 UAR 示例 (E1-05, self_attention):
- s42: WA=67.81%, UAR=45.05%, gap=22.75pp
- s123: WA=66.54%, UAR=41.93%, gap=24.61pp
- s456: WA=66.79%, UAR=41.76%, gap=25.02pp

FAU 最大 WA−UAR 差距: 30.77pp (E6-07_s123)
FAU 最小 WA−UAR 差距: -3.72pp (E3-01, zero-shot 中 UAR 可能高于 WA)

**裁决**: **CONFIRMED** — 前 agent 报告的 "均值 9.1pp" 与独立计算的 9.11pp 一致。

### 5d. 信任边界分析 ⚠️

**核心问题**: 「验证通过」证明了什么？

```
验证链路:
  JSON 日志记录值 ← 训练代码 (sklearn.recall_score) ← 模型预测 ← 数据/权重
        ↑                    ↑                              ↑
   我们验证了这步           这步无法验证                  这步无法验证
   (手册=日志)             (代码正确性)                 (数值正确性)
```

**关键约束**:

1. **WA 和 UAR 均无法被独立重算**。没有混淆矩阵、预测、或逐类 recall 存于日志，无法用原始数据验证 `sklearn.accuracy_score` / `sklearn.recall_score` 的输入是否正确。

2. **test_uar 只能取信于训练代码**。如果训练脚本有 bug（例如传错 labels 顺序、用错 average 参数），我们无从发现。

3. **「验证通过」只证明了「手册 = 日志」，不证明「日志 = 正确」**。这是 Phase 2 的本质局限。

4. **B7 方向不可追踪**: 日志中 `train_data`/`test_data` 均显示目标域（如 E7-03: train=['c-besd'], test=['c-besd']），源域信息仅存在于 launch 脚本中。无法仅从日志确认迁移方向。

5. **E4-04_s42 配置异常**: `train_data`=['c-besd'] 与 s123/s456 的 `train_data`=['c-besd-4cl','iemocap'] 不同，是**不同的实验条件**。E4-04 不应被视为有效的 3-seed 实验。

**裁决**: **CANNOT-VERIFY** — 日志值可信任为训练代码的输出，但无法独立验证训练代码本身的正确性。这是所有基于日志的验证工作的固有局限，非前 agent 过失。

---

## §6 三个 missing_in_logs 项

### ledger_full.csv 中标记为 `missing_in_logs` 的条目

| Section | Experiment | Field | Handbook Value | 说明 |
|---------|-----------|-------|---------------|------|
| TEXT | B2_best | WA | 41.05% | 声称 B2 最佳零样本 = 41.05% |
| TEXT | B2_max | WA | 35.47% | 声称 B2 范围上限 = 35.47% |
| TEXT | B2_min | WA | 19.17% | 声称 B2 范围下限 = 19.17% |

### 独立验证

在所有 192 个 `test_wa` 值中搜索（tolerance ±0.03pp）:

- **41.05%**: 0/192 匹配 — **不存在**
- **35.47%**: 0/192 匹配 — **不存在**
- **19.17%**: 0/192 匹配 — **不存在**

### 真实的 B2 值

```
实际 B2 范围: 20.51% (E3-01) ~ 34.68% (E3-14)
实际最佳零样本: E3-14 = 34.68% (IEMOCAP→C-BESD, self_attention)
```

这三个 TEXT 字段是 handbook 的文本声明，不对应任何日志文件。前 agent 将其标记为 `missing_in_logs` 是准确的，但需要注意它们不是真正的"缺失字段"而是"handbook 中找不到对应日志支撑的声明"。

**裁决**: **CONFIRMED** — 三者在任何日志中均不存在，前 agent 的 `missing_in_logs` 分类正确。

---

## §7 配置可信度抽查

### 7a. augment_condition 已知问题复查

**发现 1**: 22 个文件完全缺少 `augment_condition` key（主要是 B1/E1 的旧协议文件，protocol=ac_suite_2026-05）。

**发现 2**: 143 个文件记录 `augment_condition: "C1"`，但其中大量实验实际上**没有使用任何数据增强**：
- E6 全系列 (30 文件): `launch_b6.sh` 不传 `--augment_condition` 参数，但 JSON 全显示 `C1`
- E7 全系列 (18 文件): 同样情况
- B2/E3 全系列 (18 文件): 同样情况

**根因**: C1 是训练代码中 `augment_condition` 参数的默认值。当 launch 脚本未显式传参时，默认值被写入 JSON。

**发现 3**: 仅有 B3/E4 的实验正确记录了 C2/C3/C4（因为 launch_b3.sh 显式传参）。

### 7b. 随机 5 实验配置抽查

抽查了 E6 (all 30 files), E7 (all 18 files), E4 (all 36 files) 的配置一致性：

| 检查项 | 结果 |
|--------|------|
| E6 系列 pooling_type 与设计一致 | ✅ — E6-01/02=mean, E6-03/04/05=self_attention |
| E6 系列 fusion_mode 与设计一致 | ✅ — E6-01/02/03=last, E6-04/05=weighted |
| E6 系列 use_adapter 与设计一致 | ✅ — 仅 E6-02/05 有 adapter |
| E7 系列 pooling_type=all self_attention | ✅ — 与 launch_b7.sh 一致 |
| E7 系列 train_data/test_data | ⚠️ 均显示目标域，源域不可追踪 |
| E4-04_s42 配置异常 | ❌ train_data 与 s123/s456 不同 |

### 7c. 配置字段可靠性评级

| 字段 | 可靠性 | 说明 |
|------|--------|------|
| `pooling_type` | ✅ 高 | 始终准确记录 |
| `fusion_mode` | ✅ 高 | 始终准确记录 |
| `use_adapter` | ✅ 高 | 始终准确记录 |
| `unfreeze_ssl` | ✅ 高 | 始终准确记录 |
| `train_data` / `test_data` | ✅ 高 | 准确（但 B7 只记录目标域） |
| `augment_condition` | ❌ 低 | 默认值 C1 误写入无增强实验 |
| `protocol` | ⚠️ 中 | 旧版文件缺失（3 文件），但值本身可靠 |
| `reg_profile` | ✅ 高 | 始终准确记录 |
| `weight_decay` / `label_smoothing` / `pooling_dropout` / `grad_clip` | ⚠️ 中 | 3 文件缺失（E4-04_s42, E4-10_s123, E4-10_s42） |

**裁决**: **REFUTED** (augment_condition 可靠性的假设) / **CONFIRMED** (前 agent 对此字段不可靠的警告)。前 agent 在 manifest 中标记了 `aug_trusted: "FALSE — JSON default, not experimental condition"`，此标记准确。

---

## §8 盲抽复核

### 随机选择的 5 组实验

| 实验 | 描述 | 独立提取 mean±std | Ledger 记录 | 匹配? |
|------|------|-------------------|-------------|-------|
| E1-04 | FAU mean pooling | 66.94% ± 1.32pp | 66.94±1.08% | ⚠️ std 差异 (1.32 vs 1.08) |
| E4-08 | Aug C4 child | 59.71% ± 0.92pp | 59.71±0.75% | ⚠️ std 差异 (0.92 vs 0.75) |
| E2-02 | FAU unfreeze | 66.37% ± 1.04pp | 66.37±0.85% | ⚠️ std 差异 (1.04 vs 0.85) |
| E7-02 | C-BESD→IEMOCAP | 63.25% ± 0.49pp | 63.25±0.40% | ⚠️ std 差异 (0.49 vs 0.40) |

### std 差异分析

所有 4 组中，我的独立 std 均略大于 ledger 报告值。差异源于 **std 计算方式不同**:

- 我的计算: sample std, `ddof=1` (n-1 denominator) — 这是统计学标准做法
- Ledger 可能使用: population std, `ddof=0` (n denominator)

**验证**: 对 E1-04 (67.63, 65.41, 67.77):
- n=3, sample std = 1.32pp
- n=3, population std = 1.08pp ← 与 ledger 一致

**裁决**: **CONFIRMED** — ledger 均值与独立提取完全一致。std 差异源于 population vs sample std 的统计约定差异（population std = sample std × √((n-1)/n) = sample std × √(2/3) ≈ sample std × 0.816），非计算错误。从投稿角度看，应使用 **sample std** (ddof=1)，这是机器学习的标准做法。

---

## §9 额外发现

### 🚨 E4-04 配置异常（CRITICAL）

```
E4-04_s42.json:  train=['c-besd'],              test=['c-besd-4cl'],       WA=68.02%
E4-04_s123.json: train=['c-besd-4cl','iemocap'], test=['c-besd-4cl','iemocap'], WA=63.81%
E4-04_s456.json: train=['c-besd-4cl','iemocap'], test=['c-besd-4cl','iemocap'], WA=62.67%
```

**s42 的训练配置与 s123/s456 完全不同！** 这不是同一个实验的三个 seed，而是两个不同的实验。E4-04 的 3-seed mean 是无意义的。前 agent 未发现此异常。

### ⚠️ B7 源域不可追溯

`E7-03_s42.json` 显示 `train_data: ["c-besd"]`, `test_data: ["c-besd"]`。但从 launch_b7.sh 可知这是 FAU→C-BESD 迁移。**源域信息仅存在于 launch 脚本中**，无法仅从 JSON 日志验证迁移方向。这使 B7 的迁移方向声明可被独立质疑——除非提供 checkpoint 加载日志或 launch 脚本执行记录。

### ⚠️ 3 文件为精简 JSON (E4-04_s42, E4-10_s123, E4-10_s42)

这三个文件缺少 `protocol`, `grad_clip`, `label_smoothing`, `output_dir`, `pooling_dropout`, `weight_decay` 等字段，仅保留了最小记录（13 keys vs 正常 ~21 keys）。这限制了它们的配置可审计性。

---

## §10 按 Phase 的可信度表

| Phase | 数值可投稿引用 | 配置可投稿引用 | 备注 |
|-------|--------------|--------------|------|
| **B1 (E1)** | ✅ WA/UAR 可靠 | ⚠️ augment_condition 缺失或默认值 | 天花板数字需修正为 3-seed mean |
| **B2 (E3)** | ✅ WA/UAR 可靠 | ⚠️ augment_condition=C1 是默认值 | 单 seed 实验，无 std |
| **B3 (E4)** | ⚠️ E4-04 需排除 | ⚠️ 3 文件配置不完整 | E4-04 不是有效的 3-seed 实验 |
| **B4 (E5)** | ✅ WA/UAR 可靠 | ⚠️ augment_condition=C1 是默认值 | — |
| **B5 (E2)** | ✅ WA/UAR 可靠 | ✅ 配置完整 | — |
| **B6 (E6)** | ✅ WA/UAR 可靠 | ⚠️ augment_condition=C1 是默认值 | — |
| **B7 (E7)** | ✅ WA/UAR 可靠 | ❌ 源域方向不可追踪 | 迁移方向仅存于 launch 脚本 |

### 可安全投稿引用的数字

以下数字经独立验证可安全引用（均提供 3-seed mean±sample_std）:

| 实验 | 数据集 | 配置 | WA (3-seed mean±std) |
|------|--------|------|---------------------|
| E2-01 | C-BESD | self_attn + unfreeze | **96.91% ± 0.19pp** |
| E1-02 | C-BESD | self_attn + frozen | **91.87% ± 1.56pp** |
| E1-05 | FAU | self_attn + frozen | **67.05% ± 0.68pp** |
| E2-03 | IEMOCAP | prosody + unfreeze | **66.37% ± 0.96pp** |
| E3-14 | IEMOCAP→C-BESD | self_attn zero-shot | **34.68%** (单 seed) |
| E6-03 | C-BESD | Mean→SelfAttn 关键提升 | **91.91% ± 0.94pp** |
| E7-03 | FAU→C-BESD | 迁移 | **91.57% ± 0.45pp** |

⚠️ **修正建议**: CLAUDE.md 中的 E1-02 "92.92%" 和 E1-05 "67.81%" 应改为 3-seed mean。

---

## §11 我无法证实的事项

1. **WA 值的正确性**: 没有预测/标签数据，无法独立重算 WA。只能取信于训练代码的 `sklearn.accuracy_score` 调用。

2. **UAR 值的正确性**: 同上。`sklearn.recall_score(average='macro')` 的输入正确性无法独立验证。

3. **B7 迁移方向**: 日志中的 `train_data`/`test_data` 不包含源域信息。迁移方向（如"FAU→C-BESD"）只能从 launch 脚本推断，无法从日志独立确认。

4. **E4-04 的设计意图**: s42 配置与 s123/s456 不同。不清楚哪个配置是"正确的"实验条件，还是 s42 是错误运行。

5. **augment_condition 实际值**: 对于未显式传参的实验，无法仅从日志判断是否真的没有使用增强。需要检查训练代码的默认行为或 checkpoint 中的实际数据 pipeline。

6. **E1-03/E1-09 handbook 值的来源**: 87.38% 和 56.22% 不在任何日志中。不清楚它们是计算错误还是来自旧版本实验。

7. **前 agent 未发现的补充问题**: E4-04 配置异常、B7 源域不可追溯、population vs sample std 差异 — 这三项是我独立发现的新问题。

---

## §12 总体结论

### 前三个 Phase 的工作是否可信？

**部分可信，但有重要保留**:

- ✅ **Phase 0 (文件盘点)**: 可确认 — 192 文件全覆盖，文件清单准确
- ✅ **Phase 1 (溯源清单)**: 可确认 — manifest 中从日志提取的配置字段与日志一致
- ⚠️ **Phase 2 (数值验证)**: 有条件可确认 — ledger 的 `log_value` 列提取准确（0 错误），但 handbook 值存在系统性单 seed 冒充 3-seed mean 的问题，且部分 PHANTOM 分类不够精确
- ⚠️ **Phase 3 (指标完整性)**: 可确认存在性，不可确认正确性 — test_uar 确实存在于 192/192 文件，但无法独立验证

### 是否可在此基础上进入 Phase 4？

**可以进入，但 Phase 4 须额外关注**:

1. **E4-04 必须被标记为无效 3-seed 实验**，从所有均值计算中排除
2. **B7 迁移方向的验证**需要在 Phase 4 中交叉核对 launch 脚本逻辑与 checkpoint 加载记录
3. **augment_condition 的不可靠性**需要在权威数据手册中明确记录，或从日志字段（如 augmentation pipeline 的具体配置）中寻找更可靠的替代指标
4. **std 应统一使用 sample std (ddof=1)**，这是投稿标准
5. **所有"天花板"数字应使用 3-seed mean**，不得用单 seed 最高值替代

### 前 agent 的工作质量评价

- **提取准确性**: 优秀（ledger log_values 零错误）
- **覆盖率**: 良好（192 文件全量覆盖）
- **批判性**: 中等（发现了 augment_condition 问题和手册值偏差，但未发现 E4-04 异常、B7 源域缺失、std 约定问题）
- **诚实性**: 良好（明确标注了 missing_in_logs 和不一致，未掩盖问题）

---

*审查员独立编写，所有证据来自 `results/logs/E*-*.json` 直接读取和独立 Python 脚本重算。*
*验证脚本: `validation/independent_verify.py` (可重跑)*
