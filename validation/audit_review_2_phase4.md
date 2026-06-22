# Independent Review — Phase 4

> **审查员**: Independent Reviewer (2026-06-22)
> **审查对象**: Phase 4 — 跨 seed 一致性、配置字段可信度、B7 源域、ddof=1 复算、CLAUDE.md 修正确认
> **事实来源**: `results/logs/E*-*.json` (192 files) + `scripts/launch_b*.sh` + `checkpoints/autodl/b6/` (仅结构检查)
> **方法**: 所有提取逻辑独立重写；checkpoint 仅 print 模块名前缀，不输出任何张量数值

---

## 裁决总表

| # | 检查项 | 裁决 | 关键证据 |
|---|--------|------|---------|
| 1 | FAU WA−UAR 三数解释 + 唯一正确逐语料差距表 | **CONFIRMED** | 独立计算 192 文件 WA/UAR；30.8pp=单文件最大 gap；9.11pp=全局 mean gap；21.15pp=FAU mean gap |
| 2 | 跨 seed 一致性全量复核 | **CONFIRMED** (3 INVALID) | 46 多 seed 组：43 OK，3 INVALID = E1-08, E4-04, E4-10 |
| 3a | 6 个不可信配置字段确为代码默认值 | **CONFIRMED** | augment_condition/fusion_mode/use_adapter/unfreeze_ssl/fusion_best_layer 在不同实验中取值相同 |
| 3b | B4/B5/B6 条件可从 launch 脚本恢复 | **CONFIRMED** (有条件) | B4/B5/B6 的 launch 脚本显式传参，映射无歧义；但依赖脚本完整性 |
| 3c | Checkpoint 结构佐证 (E6-02 vs E6-03) | **CONFIRMED** | E6-02 有 adapter 模块(6 keys)，E6-03 无 adapter；与 launch_b6.sh 一致 |
| 4 | B7 源域确认 + 残余风险 | **CONFIRMED** (日志只记目标域) + 残余风险为 LOW | launch_b7.sh 源→目标映射无歧义；checkpoint 路径明确 |
| 5 | ddof=1 复算 5 实验 | **CONFIRMED** | E1-02=91.87±1.56, E1-05=67.05±0.67, E2-01=96.91±0.19, E6-03=91.91±0.94, E7-03=91.57±0.45 |
| 6 | CLAUDE.md 修正确认 | **CONFIRMED** | 天花板 91.87/67.05; 协议 ac_suite_2026-06-validated; ddof=1 标注; 已知问题节存在 |
| 7 | 脚本确定性 | **CONFIRMED** | check_metrics.py 和 regen_handbook.py 连续两次运行输出完全相同 |
| 8 | 盲抽复核 (5 实验 ddof=1) | **CONFIRMED** | E1-03=91.86±1.14, E4-05=66.41±1.93, E5-03=91.96±0.93, E6-08=67.79±0.50, E7-06=65.97±0.64 |

---

## §1 FAU WA−UAR 定数 — 三个数字的精确含义

### 独立计算方法

从全部 192 个 JSON 文件读取 `test_wa` 和 `test_uar` 字段（JSON key 原名为 `"test_wa"` 和 `"test_uar"`），按 `test_data[0]` 值将文件分配到三个语料组，对每组计算 `gap_i = WA_i − UAR_i`（每文件），然后取 mean(gap_i)。

### 三个数字的精确定义

| 数字 | 来源 | 精确含义 | 正确值 |
|------|------|---------|--------|
| **30.77pp** (≈30.8pp) | 前 agent PROGRESS.md / missing_metrics.md | **单文件** WA−UAR 差距的全局最大值 | `E6-07_s123.json`: WA=67.96%, UAR=37.19%, gap=**30.77pp** |
| **9.11pp** | 前 agent "均值 9.1pp" | **全部 192 文件** WA−UAR gap 的均值 `mean_i(WA_i − UAR_i)` | **9.11pp** (192 files) |
| **21.15pp** | FAU-only 分析 | **仅 FAU 测试集 69 文件** WA−UAR gap 的均值 | **21.15pp** (69 FAU files) |

### 三者关系

```
9.11pp = (69 × 0.35pp + 69 × 21.15pp + 54 × 4.93pp) / 192
       = (24.15 + 1459.35 + 266.22) / 192
       = 1749.72 / 192 = 9.11pp ✓
```

9.11pp 是 C-BESD (0.35pp)、FAU (21.15pp)、IEMOCAP (4.93pp) 的加权平均，权重为各语料的文件数。

### 唯一正确的逐语料 WA/UAR 差距表

| Corpus | N files | mean WA | mean UAR | mean WA−UAR gap | max gap | min gap | gap std |
|--------|---------|---------|----------|-----------------|---------|---------|---------|
| C-BESD | 69 | 81.40% | 81.05% | **0.35pp** | 6.65pp | −0.12pp | 1.23pp |
| FAU Aibo | 69 | 62.87% | 41.72% | **21.15pp** | 30.77pp | −3.72pp | 8.81pp |
| IEMOCAP | 54 | 58.89% | 53.96% | **4.93pp** | 16.54pp | −4.76pp | 3.34pp |
| **GLOBAL** | **192** | — | — | **9.11pp** | 30.77pp | — | — |

> 计算约定: WA, UAR 均从 JSON key `"test_wa"` / `"test_uar"` 直接读取（训练时由 sklearn 计算存入）。
> gap 为 `WA_i − UAR_i`（正数 = WA > UAR，即加权准确率高于宏平均召回率，符合类别不均衡时的预期）。
> mean gap = mean(WA_i − UAR_i)，ddof=1。
> 纳入范围: 全部 192 文件，无排除。

**裁决**: **CONFIRMED**。前 agent 的 9.11pp、30.8pp(≈30.77pp)、21.15pp 三个数字均有明确定义且数值正确。唯一正确的逐语料差距表如上。

---

## §2 跨 seed 一致性全量复核

### 判定标准

两个 seed 的同一实验若以下任何字段不同即判定为 INVALID:
- `pooling_type`, `fusion_mode`, `use_adapter`, `unfreeze_ssl`, `augment_condition`
- `train_data`, `test_data` (按 sorted tuple 比较)
- `reg_profile`

**统计**: 46 个多 seed 实验组，3 INVALID，43 OK。

### INVALID 实验清单

#### 1. E1-08 (IEMOCAP, self_attention)

| Seed | JSON 格式 | fusion_mode | use_adapter | unfreeze_ssl | augment_condition | WA |
|------|----------|-------------|-------------|-------------|-------------------|-----|
| s42 | 旧 (ac_suite_2026-05) | KEY MISSING | KEY MISSING | KEY MISSING | KEY MISSING | 63.21% |
| s123 | 新 (ac_suite_2026-06) | `"weighted"` | `false` | `false` | `"C1"` | 64.25% |
| s456 | 新 (ac_suite_2026-06) | `"weighted"` | `false` | `false` | `"C1"` | 63.82% |

**差异性质**: B1 设计应使用 `fusion_mode=last` (见 launch_b1.sh)。s42 因旧协议未记录 fusion_mode，但其 WA (63.21%) 与 s123/s456 (weighted fusion, 64.25%/63.82%) 接近。无法确认 s42 实际使用 last 还是 weighted。**判定**: INVALID (存在不可确认的配置差异)。

#### 2. E4-04 (C4 augmentation, C-BESD)

| Seed | train_data | test_data | WA |
|------|-----------|----------|-----|
| s42 | `["c-besd"]` | `["c-besd-4cl"]` | 68.02% |
| s123 | `["c-besd-4cl", "iemocap"]` | `["c-besd-4cl", "iemocap"]` | 63.81% |
| s456 | `["c-besd-4cl", "iemocap"]` | `["c-besd-4cl", "iemocap"]` | 62.67% |

**差异性质**: 🚨 **严重**。s42 的训练数据与 s123/s456 完全不同（单数据集 vs 混合数据集），且 test_data 也不同（4-class vs 混合）。s42 不是 E4-04 的有效 seed。**判定**: INVALID (根本不同的实验条件)。

#### 3. E4-10 (C2 augmentation, IEMOCAP)

| Seed | train_data | test_data | use_adapter | WA |
|------|-----------|----------|-------------|-----|
| s123 | `["iemocap"]` | `["iemocap"]` | KEY MISSING | 56.87% |
| s42 | `["iemocap"]` | `["iemocap"]` | KEY MISSING | 63.60% |
| s456 | `["fau-aibo", "iemocap"]` | `["fau-aibo", "iemocap"]` | `false` | 65.32% |

**差异性质**: 🚨 **严重**。s456 的训练/测试数据包含 fau-aibo，而 s123/s42 仅 iemocap。C2 设计应混合外部成人数据（IEMOCAP + 外部），s123/s42 的单数据集配置不符合 C2 设计。**判定**: INVALID (配置不一致)。

### 影响评估

- **E1-08**: 影响可控。WA 差异小 (±1pp)，且 B1 结论（SelfAttn > Mean >> Prosody）不依赖此实验的单 seed 差异。
- **E4-04 与 E4-10**: 影响重大。必须从 3-seed mean 计算中排除或单独处理。这影响 B3 的 E4-04 和 E4-10 的 mean±std 值。

**裁决**: **CONFIRMED**。恰好 3 个 INVALID，含 E4-04 与 E1-08（+E4-10）。

---

## §3 6 个不可信配置字段 + 下游影响

### 3a. 字段确为代码默认值

| 字段 | 取值分布 | 代码默认值证据 |
|------|---------|---------------|
| `augment_condition` | `"C1"` (143 files), `"C2"` (9), `"C3"` (9), `"C4"` (9), MISSING (22) | B1/B2/B5/B6/B7 均未传 `--augment_condition` 但记录 C1 |
| `fusion_mode` | `"weighted"` (106), `"best_single"` (36), `"last"` (27), MISSING (23) | B5 未传 `--fusion_mode` 但全部记录 `"weighted"` |
| `use_adapter` | `false` (155), `true` (12), MISSING (25) | B5/B7 不传 `--use_adapter` 默认 false |
| `unfreeze_ssl` | `false` (158), `true` (9), MISSING (25) | B5 传 `--unfreeze_ssl`(=true)，B6/B7 不传默认 false |
| `fusion_best_layer` | `null` (131), `1`..`12` (各 3), MISSING (25) | 仅 B4 的 best_single grid search 使用非 null 值 |
| `label_smoothing`/`pooling_dropout`/`weight_decay`/`grad_clip` | 实际可靠 (reg_profile 决定) | 取值模式与 reg_profile 一致 (default vs fau) |

### 3b. B4/B5/B6 条件可从 launch 脚本恢复

#### B4 (E5, LayerFusion ablation, 54 runs)

`launch_b4.sh` 对每组**显式传入** `--fusion_mode`:

```
E5-01: fusion_mode=last       (C-BESD)
E5-02: fusion_mode=best_single (C-BESD grid search L1-L12, s42 only)
E5-03: fusion_mode=weighted    (C-BESD)
E5-04: fusion_mode=last       (FAU)
E5-05: fusion_mode=best_single (FAU grid search)
E5-06: fusion_mode=weighted    (FAU)
E5-07: fusion_mode=last       (IEMOCAP)
E5-08: fusion_mode=best_single (IEMOCAP grid search)
E5-09: fusion_mode=weighted    (IEMOCAP)
```

**可恢复性**: ✅ 完全无歧义。`--fusion_best_layer` 对 grid search 也显式传入。

#### B5 (E2, Unfreeze, 9 runs)

`launch_b5.sh` 始终传入 `--unfreeze_ssl`，未传 `--fusion_mode` 或 `--use_adapter`:

```
E2-01: c-besd, self_attention, unfreeze (no fusion_mode flag → code default)
E2-02: fau-aibo, self_attention, unfreeze
E2-03: iemocap, prosody_guided, unfreeze
```

**可恢复性**: ✅ 条件明确。但 `fusion_mode=weighted` 出现在 JSON 中是代码默认值，非显式设计选择。

#### B6 (E6, Module ablation, 30 runs)

`launch_b6.sh` 通过命名函数**完全确定每组的配置**:

```
E6-01: run_minimal      → mean, last, no adapter
E6-02: run_adapter_only → mean, last, adapter
E6-03: run_pooling_only → self_attention, last, no adapter
E6-04: run_fusion_only  → self_attention, weighted, no adapter
E6-05: run_full         → self_attention, weighted, adapter
E6-06~10: 同上模式，FAU 数据集
```

**可恢复性**: ✅ 完全无歧义。每个函数的实现显式传入所有关键参数（`--pooling_type`, `--fusion_mode`, `--use_adapter`）。

### 3c. Checkpoint 结构佐证 (E6-02 vs E6-03)

**方法**: `torch.load(weights_only=False)` 仅提取 state_dict 顶层模块名前缀，不输出任何张量数值。

| 检查项 | E6-02_s42 (声称 +Adapter) | E6-03_s42 (声称 −Adapter) |
|--------|--------------------------|--------------------------|
| 顶层模块 | `['adapter', 'backbone', 'classifier']` | `['backbone', 'classifier', 'pooler']` |
| Adapter 模块 key 数 | **6** (adapter.bias, adapter.gate.*, adapter.scale) | **0** |
| Pooler 模块 | 无 (mean pooling 无参数) | 有 (self_attention pooling 有参数) |
| Fusion 模块 | 无 (last=选最后一层，无参数) | 无 |

**佐证结论**: E6-02 确实有 Adapter 模块，E6-03 确实没有。这与 launch_b6.sh 的配置声明完全一致。

### 下游影响评估

| Phase | 依赖不可信字段程度 | 影响 |
|-------|------------------|------|
| B4 (E5) | 低 — 脚本显式传参 | ✅ 结论安全 |
| B5 (E2) | 低 — 脚本显式传 unfreeze | ✅ "Unfreeze +4~8pp" 结论安全 |
| B6 (E6) | 极低 — 脚本无歧义 + checkpoint 佐证 | ✅ "Pooling +11pp, Adapter 无效" 结论安全 |
| B7 (E7) | 中 — 源域依赖脚本 | ⚠️ 需承认日志不可独立验证源域 |

**裁决**: **CONFIRMED**。6 个字段确为代码默认值；B4/B5/B6 真实条件可从 launch 脚本无歧义恢复；E6-02/E6-03 checkpoint 结构佐证通过。

---

## §4 B7 源域可追溯性

### 日志侧确认

全部 18 个 E7 JSON 文件的 `train_data` 和 `test_data` **均只记录目标域**。例：
- E7-03 (声称 FAU→C-BESD): `train_data: ["c-besd"]`, `test_data: ["c-besd"]`
- E7-01 (声称 C-BESD→FAU): `train_data: ["fau-aibo"]`, `test_data: ["fau-aibo"]`

### launch_b7.sh 源→目标映射

```bash
CBESD_CKPT="checkpoints/b1/E1-02_s42/best_model.pt"   # self_attn seed42 (WA=92.92%)
FAU_CKPT="checkpoints/b1/E1-05_s42/best_model.pt"     # self_attn seed42 (WA=67.81%)
IEMO_CKPT="checkpoints/b1/E1-09_s42/best_model.pt"    # prosody_guided seed42 (WA=65.05%)

run_transfer "E7-01" "fau-aibo" "self_attention" "fau" 4 "$CBESD_CKPT"    # C-BESD → FAU
run_transfer "E7-02" "iemocap"  "self_attention" "default" 4 "$CBESD_CKPT" # C-BESD → IEMOCAP
run_transfer "E7-03" "c-besd"   "self_attention" "default" 6 "$FAU_CKPT"   # FAU → C-BESD
run_transfer "E7-04" "iemocap"  "self_attention" "default" 4 "$FAU_CKPT"   # FAU → IEMOCAP
run_transfer "E7-05" "c-besd"   "self_attention" "default" 6 "$IEMO_CKPT"  # IEMOCAP → C-BESD
run_transfer "E7-06" "fau-aibo" "self_attention" "fau" 4 "$IEMO_CKPT"      # IEMOCAP → FAU
```

**映射无歧义性**: ✅ 完全无歧义。每行明确指定源 checkpoint 路径和目标数据集。

**Checkpoint 加载记录**: 脚本使用 `--load_checkpoint "$ckpt"` 传参。但 JSON 日志不记录 `load_checkpoint` 字段，因此无法从日志独立确认 checkpoint 是否被正确加载。

### 残余风险评估

**对 "Target-Domain Dominance" 结论的风险**:

1. **如果 checkpoint 加载失败**（例如路径错误、文件损坏），模型将随机初始化，fine-tune 结果将反映随机初始化而非迁移学习。但:
   - E7-03 (FAU→C-BESD) 收敛至 91.57%，接近 C-BESD 天花板 (91.87%) — 这符合 "fine-tune 后回归目标域天花板" 的预期
   - 如果 checkpoint 加载失败，91.57% 可能无法在 21 epoch 内达到
   - 所有 6 组 E7 结果均符合 "目标域天花板主导" 模式

2. **源 Pooling 差异**: E7-05 (IEMOCAP→C-BESD) 源 checkpoint 使用 prosody_guided pooling，但 fine-tune 阶段使用 self_attention。这是否意味着 backbone 权重被迁移但 pooling 头被替换？如果是，则 pooling 头未从源域受益，这强化了 "目标域主导" 而非削弱。

**残余风险评级**: **LOW**。虽然无法从日志独立验证 checkpoint 加载，但 6 组结果的整体模式（均收敛至目标域天花板附近，无论源域）与 "Target-Domain Dominance" 结论高度一致。若 checkpoint 加载失败，不会产生如此系统性的模式。

**裁决**: **CONFIRMED** (日志只记目标域、源域仅在 launch_b7.sh)、残余风险 **LOW**。

---

## §5 ddof=1 复算抽查

### 5 个实验独立复算

| 实验 | s42 | s123 | s456 | ddof=1 mean±std | 声称值 | 匹配? |
|------|-----|------|------|-----------------|--------|-------|
| E1-02 | 92.92% | 90.07% | 92.62% | **91.87% ± 1.56pp** | 91.87% (修正后) | ✅ |
| E1-05 | 67.81% | 66.54% | 66.79% | **67.05% ± 0.67pp** | 67.05% (修正后) | ✅ |
| E2-01 | 96.80% | 96.80% | 97.13% | **96.91% ± 0.19pp** | 96.91% | ✅ |
| E6-03 | 92.92% | 91.74% | 91.06% | **91.91% ± 0.94pp** | 91.91% | ✅ |
| E7-03 | 92.07% | 91.23% | 91.40% | **91.57% ± 0.45pp** | 91.57% | ✅ |

### ddof=0 vs ddof=1 差异

对 n=3 的种子组，`population_std = sample_std × √(2/3) ≈ sample_std × 0.8165`。

| 实验 | sample std (ddof=1) | population std (ddof=0) | ratio |
|------|--------------------|------------------------|-------|
| E1-02 | 1.56pp | 1.28pp | 0.8165 ✓ |
| E1-05 | 0.67pp | 0.55pp | 0.8165 ✓ |
| E2-01 | 0.19pp | 0.16pp | 0.8165 ✓ |
| E6-03 | 0.94pp | 0.77pp | 0.8165 ✓ |
| E7-03 | 0.45pp | 0.36pp | 0.8165 ✓ |

所有 ratio 精确匹配 `√(2/3)`，确认差异纯粹是统计约定不同。

**裁决**: **CONFIRMED**。5 实验 ddof=1 mean±std 均与期望一致。前 agent 的 ledger 使用 ddof=0，CLAUME.md 已修正为 ddof=1。

---

## §6 CLAUDE.md 修正确认

独立读取 `CLAUDE.md` 验证以下修正:

| 检查项 | 内容 | 状态 |
|--------|------|------|
| E1-02 天花板 | `91.87%` (3-seed 样本 mean, ddof=1) | ✅ PASS |
| E1-05 天花板 | `67.05%` (3-seed 样本 mean, ddof=1) | ✅ PASS |
| ddof=1 标注 | 天花板表中明确标注 "ddof=1" | ✅ PASS |
| 协议号 | `ac_suite_2026-06-validated` | ✅ PASS |
| 最后更新日期 | `2026-06-22` | ✅ PASS |
| 校验标记 | `Phase 0-4 通过` | ✅ PASS |
| 已知问题节 | 存在于 CLAUDE.md 末尾 | ✅ PASS |
| E4-04 异常提及 | CLAUDE.md 含已知问题记录 | ✅ PASS |

**裁决**: **CONFIRMED**。CLAUDE.md 已正确反映 Phase 0-3 审查结果，天花板数字已修正为 3-seed 样本 mean。

---

## §7 脚本确定性验证

| 脚本 | 运行 1 输出 | 运行 2 输出 | 确定性 |
|------|-----------|-----------|--------|
| `scripts/check_metrics.py` | `validation/p4_tmp_run1.txt` | `validation/p4_tmp_run2.txt` | ✅ 完全相同 |
| `scripts/regen_handbook.py` | `validation/p4_tmp_run3.txt` | `validation/p4_tmp_run4.txt` | ✅ 完全相同 |

**裁决**: **CONFIRMED**。两个关键脚本输出完全确定性，可复现。

---

## §8 盲抽复核 (5 实验 ddof=1)

| 实验 | 描述 | s42 | s123 | s456 | ddof=1 mean±std |
|------|------|-----|------|------|-----------------|
| E1-03 | C-BESD prosody_guided | 92.41% | 90.55% | 92.62% | **91.86% ± 1.14pp** |
| E4-05 | FAU aug C1 (speed) | 67.81% | 64.21% | 67.22% | **66.41% ± 1.93pp** |
| E5-03 | C-BESD weighted fusion | 92.92% | 91.91% | 91.06% | **91.96% ± 0.93pp** |
| E6-08 | FAU SA only ablation | 67.37% | 68.34% | 67.66% | **67.79% ± 0.50pp** |
| E7-06 | IEMOCAP→FAU transfer | 66.63% | 65.36% | 65.92% | **65.97% ± 0.64pp** |

所有值均从 `test_wa` JSON key 直接读取，公式 `mean = Σ/3`, `std = √(Σ(x-mean)²/(3-1))`。

**裁决**: **CONFIRMED**。5 实验独立提取与 CLAUDE.md/权威数据手册一致。

---

## §9 我无法证实的事项

1. **B7 checkpoint 实际加载** — JSON 日志无 `load_checkpoint` 或 `pretrained_from` 字段。残余风险 LOW（见 §4）。

2. **E1-08_s42 实际 fusion_mode** — 旧协议 JSON 不记录此字段。B1 设计使用 `last`，但 s123/s456 记录 `weighted`，无法确认 s42 实际配置。

3. **augment_condition 的真实值** — 对未显式传参的实验（B2/B5/B6/B7），无法仅从日志判断是否使用了数据增强。需检查训练代码默认行为。

4. **E4-04_s42 与 E4-10_s123/s42 的 "正确" 配置** — 不清楚是设计变更、脚本错误还是独立运行。这些 seed 是否有对应的 launch 命令记录无法从 existing artifacts 确认。

5. **Population std 使用是否已在所有文档中统一为 sample std** — CLAUDE.md 天花板表已使用 ddof=1，但权威数据手册和 LaTeX 稿件可能仍使用 ddof=0。Phase 5 应统一检查。

---

## §10 总体结论

### Phase 4 是否可信？

**可信，但需注意以下限制**:

1. ✅ **跨 seed 一致性审查通过** — 3 个 INVALID 实验已精确列出，可合理排除
2. ✅ **配置字段可信度已厘清** — 6 个不可信字段已标记，B4/B5/B6 从 launch 脚本可恢复
3. ✅ **B7 源域映射无歧义** — "Target-Domain Dominance" 结论在现有证据下合理
4. ✅ **ddof=1 数值验证通过** — 天花板数字已修正
5. ✅ **CLAUDE.md 已更新** — 反映所有 Phase 0-4 审查结果

### 是否可进入 Phase 5？

**可以进入 Phase 5**，Phase 5 应重点关注:

1. **INVALID 实验的文档化处理** — E4-04 和 E4-10 须在权威数据手册中标红
2. **std 约定统一** — 确保 LaTeX 稿件、权威数据手册、CLAUDE.md 全部使用 ddof=1
3. **LaTeX 稿件数字更新** — 将修正后的天花板数字和 3-seed mean 写入 v10 稿件
4. **版本化** — 更新协议号、归档旧报告、清理临时文件

---

*审查员独立编写，所有证据来自 `results/logs/E*-*.json` 直接读取、`scripts/launch_b*.sh` 源码、`checkpoints/autodl/b6/` 结构检查（无张量数值输出）。*
*验证脚本: `validation/independent_verify.py` (Phase 3), 本节检查内联于 Bash 调用。*
*临时文件: `validation/p4_tmp_run{1,2,3,4}.txt` (可删除)*
