# Independent Review — Phase 5 (Final Sealing Verification)

> **审查员**: Independent Reviewer (2026-06-22)
> **审查对象**: Phase 5 — 数据版本化封板验证
> **事实来源**: `results/logs/E*-*.json` (192 files) + git history + `scripts/regen_handbook.py` + `scripts/gen_manifest.py`
> **方法**: 所有校验独立重写提取逻辑；重跑前 agent 脚本仅用于验证确定性

---

## 裁决总表

| # | 检查项 | 裁决 | 关键证据 |
|---|--------|------|---------|
| 1 | verify_all.py 一键通过 | **CONFIRMED** | 4/4 PASSED, exit 0; 重建全部校验产物 |
| 2 | regen_handbook.py vs git HEAD 零 diff | **REFUTED** | 91.96% 三实验排名顺序因 tie-breaking 不同，手册≠脚本重生 |
| 3 | FAU UAR 41.72 vs 41.67 精确解释 | **CONFIRMED** — 41.67% 正确 | 0.05pp 源于 test_data vs train_data 分类方法差异；无 INVALID 影响 FAU |
| 4 | 单一真相源 | **REFUTED** (两个差异副本) | `validation/handbook_generated.md` 与权威手册**内容不同**（排名表条目差异） |
| 5 | 手册内容 vs 独立重提 | **CONFIRMED** (11/12 PASS) | 天花板/零样本/UAR/ddof=1 均正确；INVALID 排除**声明不实** |
| 6 | INVALID 真排除 | **REFUTED** | 声称排除但逐语料表实际未排除(ALL=VALID)；manifest 无 INVALID 标记 |
| 7 | Checkpoint 未入库 | **CONFIRMED** | `git ls-files` — 0 个 .pt/.ckpt/.bin 文件 |
| 8 | Git 状态 | **REFUTED** (工作树不干净) | 2 文件 modified 未 commit (PROGRESS.md, provenance_manifest.csv) |
| 9 | LaTeX SAFE-SWAP 审计 | **CONFIRMED** | 14 SAFE-SWAP 正确应用；9 CLAIM-AFFECTED 正确未改动 |
| 10 | 生成器确定性 | **CONFIRMED** | regen_handbook.py / gen_manifest.py 两-run 逐字节一致 |

---

## §1 verify_all.py — 一键可复现

```
$ python scripts/verify_all.py

  [PASS] gen_manifest.py       — manifest 重建成功 (100 rows)
  [PASS] regen_handbook.py     — 手册重建成功 (170 lines)
  [PASS] check_metrics.py      — 指标完整性检查通过 (192 files)
  [PASS] phase4_audit.py       — 可复现性审计通过

  VERIFICATION COMPLETE
  Passed: 4/4  Failed: 0/4
```

退出码: **0**。脚本确实重建了全部校验产物。

**裁决**: **CONFIRMED** — 一键通过。

---

## §2 regen_handbook.py vs git HEAD — 零 diff 验证

### 操作

```bash
git checkout -- "docs/current/权威数据手册.md"   # 还原到 HEAD
python scripts/regen_handbook.py                  # 从日志重生
git diff -- "docs/current/权威数据手册.md"         # 比对
```

### 结果

**出现 diff！** 全局 WA 排名表中，三个同为 91.96±0.93% 的实验排名顺序不同：

| 排名 | 已提交版 (HEAD) | 重生版 (脚本输出) |
|------|----------------|-------------------|
| #2 | E4-01 | **E6-04** |
| #3 | E5-03 | **E4-01** |
| #4 | E6-04 | **E5-03** |

### 根因

三个实验 WA 完全相同 (91.96±0.93%)，`regen_handbook.py` 的排序逻辑在 tie 情况下不稳定（Python dict 遍历顺序或 sorted() 的 stable-sort 语义导致）。这不影响数值正确性，但**破坏了「已提交版 = 日志重生版 逐字节一致」的承诺**。

### 性质

- **严重度**: 低 — 数值无变化，仅排名顺序在 tied values 间漂移
- **可修复性**: 高 — 在排序 key 中加入实验 ID 作为二级排序键即可消除

**裁决**: **REFUTED** — 手册与脚本重生不完全一致。tie-breaking 不确定性导致排名表顺序漂移。

---

## §3 FAU UAR 41.72 vs 41.67 — 精确解释

### 两个数字的计算方法

| 来源 | 分类方式 | FAU 判定 | N | mean UAR |
|------|---------|---------|---|----------|
| Phase 3/4 审查 (41.72%) | `test_data[0]` — 测试语料 | 测试集为 FAU 的文件 | 69 | **41.72%** |
| 权威数据手册 (41.67%) | `train_data[0]` — 训练语料 | 训练集为 FAU 的文件 | 69 | **41.67%** |

### 0.05pp 差异的精确分解

`regen_handbook.py` 第 301-302 行使用 `corpus_name(d.get('train_data',[]))` 进行语料分类。这意味着：

- **train_data[0]='fau-aibo'** → FAU_Aibo 组
- **test_data[0]='fau-aibo'** → Phase 4 的 FAU 组

两个集合的差异来自 **B2 零样本迁移实验**（train ≠ test）：
- E3-01~03: train=C-BESD, test=FAU → train 法归入 C-BESD，test 法归入 FAU
- E3-16~18: train=IEMOCAP, test=FAU → train 法归入 IEMOCAP，test 法归入 FAU

这 6 个实验的 UAR（24.2%, 24.1%, 24.0%, 32.3%, 33.0%, 34.6%）在 test 法下计入 FAU UAR(拉高均值)，在 train 法下不计入 FAU。

### 逐条排除分析

3 个 INVALID 实验 (**E1-08, E4-04, E4-10**) 的 `train_data[0]` 分别为 IEMOCAP / C-BESD / IEMOCAP — **三者均不涉及 FAU_Aibo**。因此：

- **(a) 全部 FAU 文件**: 69 files, mean UAR = 41.6664% → **41.67%**
- **(b) 排除 INVALID**: 69 files, mean UAR = 41.6664% → **41.67%**
- **(a) − (b) = 0.0000pp** — 无任何影响

### 结论

- **41.67% 是唯一正确值**（手册使用方法，train_data 分类）
- **41.72% 是 Phase 3/4 的计算错误**（误用 test_data 分类）
- **0.05pp 完全来自分类方法差异**，与 INVALID 排除**无关**
- **0 个 INVALID 实验影响 FAU UAR 统计**

**裁决**: **CONFIRMED** — 手册 41.67% 正确；Phase 3/4 的 41.72% 是方法学错误。

---

## §4 单一真相源

### 发现的"手册"文件

| 文件 | 位置 | 内容 | 权威性 |
|------|------|------|--------|
| `权威数据手册.md` | `docs/current/` | 中文，AUTO-GENERATED | ✅ **唯一真相源** |
| `handbook_generated.md` | `validation/` | 英文，AUTO-GENERATED | ❌ **第二副本，内容不同** |
| `handbook_snapshot_2026-06-22.md` | `validation/` | 快照 | ❌ 冗余副本 |
| `权威数据手册_v1_公证版.md` | `docs/archive/` | 旧版 | ✅ 已归档，非竞争 |

### handbook_generated.md 与权威手册的差异

两个文件并非同一内容的中英版本。在全局 WA 排名表中：

| 排名 | 权威手册 (中文) | handbook_generated.md (英文) |
|------|---------------|---------------------------|
| #2 | **E4-01** | **E5-03** |
| #3 | E5-03 | E6-04 |
| #4 | E6-04 | E4-01 |

这确认了 **handbook_generated.md 不是权威手册的翻译副本**，而是另一个生成输出（可能来自不同 run 或有不同排序）。

### 风险评估

- `validation/handbook_generated.md` 同样包含 `AUTO-GENERATED` 头，同样声称从日志生成
- 与权威手册内容**不同**（排名表），构成潜在的混淆源
- `validation/handbook_snapshot_2026-06-22.md` 是冗余快照

**裁决**: **REFUTED** — 存在第二个内容不同的 AUTO-GENERATED 手册副本；非单一真相源。

---

## §5 手册内容 vs 独立重提

### 逐项核对

| # | 检查项 | 预期 | 实际 | 结果 |
|---|--------|------|------|------|
| 1 | E1-02 天花板 | 91.87% | ✅ 91.87% (3-seed mean, ddof=1) | CONFIRMED |
| 2 | E1-05 天花板 | 67.05% | ✅ 67.05% (3-seed mean, ddof=1) | CONFIRMED |
| 3 | 零样本最高 | 34.68% (E3-14) | ✅ 34.68% | CONFIRMED |
| 4 | 41.05% 不存在 | 不应出现 | ✅ 仅出现在 PHANTOM 说明中 | CONFIRMED |
| 5 | 35.47% 不存在 | 不应出现 | ✅ 仅出现在 PHANTOM 说明中 | CONFIRMED |
| 6 | 19.17% 不存在 | 不应出现 | ✅ 仅出现在 PHANTOM 说明中 | CONFIRMED |
| 7 | Unfreeze FAU −0.67pp | 存在 | ✅ E2-02 标注 Δ=−0.67pp | CONFIRMED |
| 8 | FAU UAR 41.67% | 逐语料表中 | ✅ WA-UAR by Corpus 表显示 | CONFIRMED |
| 9 | 逐语料 WA/UAR 表 | C-BESD/FAU/IEMOCAP | ✅ 完整表存在 | CONFIRMED |
| 10 | ddof=1 | 多处声明 | ✅ L195, 天花板表标注 | CONFIRMED |
| 11 | 3 INVALID 标注 | 标记排除 | ⚠️ L196 声称排除但未实现（见 §6） | REFUTED |
| 12 | B1 非天花板值 | — | ⚠️ E1-01/03/04/06/07/08/09 仍为旧值 | CANNOT-VERIFY (非 SAFE-SWAP 范围) |

### B1 表中未修正的值

手册 B1 表中以下值仍为旧值（与日志不符），但未在 SAFE-SWAP 账本中列出：

| 实验 | 手册值 | 日志值 | 差异 |
|------|--------|--------|------|
| E1-01 | 79.56±1.14% | 79.23±0.81% | 0.33pp |
| E1-03 | 87.38±0.79% | 91.86±0.93% | 4.48pp |
| E1-04 | 64.40±0.76% | 66.94±1.08% | 2.54pp |
| E1-06 | 65.38±1.23% | 66.77±0.97% | 1.39pp |
| E1-07 | 58.86±1.66% | 61.19±0.62% | 2.33pp |
| E1-08 | 65.16±1.34% | 63.76±0.43% (INVALID) | 1.40pp |
| E1-09 | 56.22±0.66% | 64.38±0.85% | 8.16pp |

这些被视为 CLAIM-AFFECTED（改变叙事），因此未做机械替换。这是合理的保守策略。

**裁决**: **CONFIRMED** (天花板/零样本/UAR/ddof=1 正确)；**REFUTED** (INVALID 排除声明不实)；其余为策略性保留。

---

## §6 INVALID 是否真排除

### 权威手册声称

> L196: `2. **INVALID**: 3 experiments excluded from aggregation (see above)`

### 独立验证

对逐语料 WA-UAR 表进行 ALL vs VALID 对比：

| 语料 | ALL (含 INVALID) | VALID (排除 INVALID) | 手册实际值 | 手册使用? |
|------|-----------------|---------------------|-----------|----------|
| C-BESD | WA=81.00%, UAR=80.68%, N=69 | WA=81.73%, UAR=81.60%, N=66 | WA=81.00%, UAR=80.68%, N=69 | **ALL** ❌ |
| FAU | WA=62.82%, UAR=41.67%, N=69 | WA=62.82%, UAR=41.67%, N=69 | WA=62.82%, UAR=41.67%, N=69 | 相同 ✅ |
| IEMOCAP | WA=59.47%, UAR=54.49%, N=54 | WA=59.05%, UAR=54.08%, N=48 | WA=59.47%, UAR=54.49%, N=54 | **ALL** ❌ |

**手册的逐语料表值 = ALL（未排除 INVALID）**，与 Data Integrity Notes 的声明矛盾。

### provenance_manifest.csv

对 manifest 全文搜索 `INVALID` — **0 次出现**。三个 INVALID 实验在 manifest 中被列为普通实验，含完整的 mean±std 值，无任何标记。

### 排除的 INVALID 文件明细

| 语料 | 排除文件 | 对 WA 的精确影响 |
|------|---------|----------------|
| C-BESD | E4-04_s42/s123/s456 | WA 从 81.00% → 81.73% (+0.73pp)；UAR 从 80.68% → 81.60% (+0.91pp) |
| FAU | (none) | 无影响 |
| IEMOCAP | E1-08_s42/s123/s456, E4-10_s42/s123/s456 | WA 从 59.47% → 59.05% (−0.42pp)；UAR 从 54.49% → 54.08% (−0.41pp) |

**裁决**: **REFUTED** — 声称排除但逐语料表未排除；manifest 无 INVALID 标记。

---

## §7 Checkpoint 未入库

```bash
git ls-files | grep -E '\.(pt|ckpt|bin|pth|safetensors)$'
# Output: (empty — 0 matches)
```

**裁决**: **CONFIRMED** — 无任何 checkpoint/二进制文件在 git 跟踪中。

---

## §8 Git 状态

| 检查项 | 预期 | 实际 | 结果 |
|--------|------|------|------|
| Commit `bb3b2b7` 存在 | ✅ | `bb3b2b7 feat: ac_suite_2026-06-validated` (+tag) | ✅ |
| Tag `ac_suite_2026-06-validated` 存在 | ✅ | 存在，指向 bb3b2b7 | ✅ |
| 工作树干净 (no modified files) | ✅ | **2 files modified** | ❌ |

### 未提交的修改

```
 M validation/PROGRESS.md              — Phase 5 进度更新 + 最终交付清单
 M validation/provenance_manifest.csv  — manifest 更新 (ddof=1?)
```

`PROGRESS.md` 包含 Phase 5 完成状态的更新和最终交付清单（约 +30/-24 行）。
`provenance_manifest.csv` 有约 100 行变更（可能是 ddof=1 重算和 INVALID 标记）。

**裁决**: **REFUTED** — 工作树不干净，2 文件未 commit。封板前的最终修改未纳入 tagged commit。

---

## §9 LaTeX SAFE-SWAP 审计

### 审计方法

对比 `c294b58` (Phase 5 前) vs `bb3b2b7` (HEAD) 的 5 个 .tex 文件 diff。

### SAFE-SWAP 应用核对

| 账本条 | 文件 | 变更 | 实际 diff 中? | 裁决 |
|--------|------|------|-------------|------|
| #1 | v10_0_Abstract | 92.92%→91.87% | ✅ | CONFIRMED |
| #2 | v10_0_Abstract | 67.81%→67.05% | ✅ | CONFIRMED |
| #3 | v10_4_Experiments | 92.92±2.15%→91.87±1.56% | ✅ | CONFIRMED |
| #4 | v10_4_Experiments | 67.81±2.15%→67.05±0.67% | ✅ | CONFIRMED |
| #5 | v10_4_Experiments | 92.92% (B5 frozen ref)→91.87% | ✅ | CONFIRMED |
| #6 | v10_4_Experiments | 67.81% (B5 frozen ref)→67.05% | ✅ | CONFIRMED |
| #7 | v10_5_Analysis | 92.92%→91.87% | ✅ | CONFIRMED |
| #8 | v10_5_Analysis | 67.81%→67.05% | ✅ | CONFIRMED |
| #9 | v10_6_Conclusion | 92.92%→91.87% | ✅ | CONFIRMED |
| #10 | v10_6_Conclusion | 67.81%→67.05% | ✅ | CONFIRMED |
| #11-14 | v10_standalone | 同模式 | ✅ | CONFIRMED |

**全部 14 条 SAFE-SWAP 均已正确应用**，均为机械替换。

### CLAIM-AFFECTED 未改动验证

| 账本条 | 原值 | 是否被改动? | 裁决 |
|--------|------|-----------|------|
| A | 35.47% (best zero-shot) | ❌ 未改动 — 仍引用 old range | CONFIRMED |
| B | E3-10 = 35.47% | ❌ 未改动 | CONFIRMED |
| C | E2-02 = 76.02±0.68% | ❌ 未改动 — LaTeX 仍写 76.02% | CONFIRMED |
| D | +8.2pp FAU unfreeze | ❌ 未改动 | CONFIRMED |
| E | +4.0pp C-BESD unfreeze | ❌ 未改动 | CONFIRMED |
| F | +1.2pp IEMOCAP unfreeze | ❌ 未改动 | CONFIRMED |
| G | 19.17% (range min) | ❌ 未改动 | CONFIRMED |
| H | 41.05% (any ref) | ❌ 未出现在 diff 中 | CONFIRMED |
| I | 35.47% (collapse sentence) | ❌ 未改动 — 仍 "19--35%" | CONFIRMED |

### 边界情况抽查

**Abstract 中的 "19--35%"**: 出现在 `v10_0_Abstract.tex` 和 `v10_standalone.tex` 的 diff 中，但**未被改动**。实际范围应为 20.51%--34.68%。这正确归类为 CLAIM-AFFECTED（改变范围叙事），未做机械替换。✅

**B5 表中的 76.02%**: 出现在 `v10_4_Experiments.tex` diff 中，**未被改动**。实际 E2-02 = 66.37%。正确归类为 CLAIM-AFFECTED。✅

**B1 表中的 87.38%/79.56%/64.40% 等**: 在整个 diff 中**未出现**（未改动）。这些值在账本中未列为 SAFE-SWAP。✅（保守策略）

**裁决**: **CONFIRMED** — 14 SAFE-SWAP 全部正确应用；9 CLAIM-AFFECTED 全部正确保留；无 SAFE-SWAP 误入 CLAIM-AFFECTED 句子的情况。

---

## §10 生成器确定性

| 脚本 | Run 1 | Run 2 | 结果 |
|------|-------|-------|------|
| `scripts/regen_handbook.py` | 输出 A | 输出 B | ✅ **逐字节一致** |
| `scripts/gen_manifest.py` | 输出 C | 输出 D | ✅ **逐字节一致** |

两脚本在同一环境下连续运行两次，输出完全逐字节一致。

**裁决**: **CONFIRMED** — 脚本本身是确定性的（但 §2 显示与已提交版的排序细节可能因 Python/dict 环境而异）。

---

## §11 「我无法证实」清单

1. **手册是否应排除 INVALID 后再计算逐语料统计**: 手册声称排除但实际未排除。不清楚这是预期行为（声明错误）还是实现 bug。

2. **handbook_generated.md 的预期用途**: 不清楚它是权威手册的英文翻译还是独立的生成产物。两个文件内容不同，无法判断哪个是「正确」的排名顺序。

3. **B1 表中未修正的非天花板值**: E1-01/03/04/06/07/08/09 的旧值是否应在 Phase 5 中修正，还是留给叙事阶段处理。账本将它们排除了 SAFE-SWAP。

4. **76.02% (E2-02 unfreeze) 的叙事影响**: LaTeX 仍然声称 FAU unfreeze = 76.02%，但实际日志值为 66.37%。这是最严重的 CLAIM-AFFECTED 错误，必须在叙事阶段修正。但我不确定当前的「保留旧值」策略是否正确 — 76.02% 读者会被严重误导。

---

## §12 总体结论

### 最终裁决: **NOT-SEALED**

以下项目阻止封板：

| # | 阻断项 | 严重度 | 修复建议 |
|---|--------|--------|---------|
| 1 | **工作树不干净** | 🔴 阻断 | commit PROGRESS.md + provenance_manifest.csv，或 stash |
| 2 | **handbook_generated.md 与权威手册内容不同** | 🟡 高 | 删除或同步为权威手册的英文翻译 |
| 3 | **INVALID 排除声明不实** | 🟡 高 | 选择：实际排除并重算，或修正声明文本 |
| 4 | **手册 ≠ regen_handbook.py 输出** (tie-breaking) | 🟢 低 | 在排序 key 中加实验 ID 以消除不确定性 |
| 5 | **manifest 无 INVALID 标记** | 🟡 中 | 在 provenance_manifest.csv 中标记 INVALID 实验 |

### 可安全进入叙事阶段的项

以下已经正确且可封板：

- ✅ 192 日志完整性 + 一键验证通过
- ✅ 天花板数字 (91.87/67.05) 正确
- ✅ FAU UAR 41.67% 正确（0.05pp 差异已精确解释）
- ✅ checkpoint 未入库
- ✅ LaTeX SAFE-SWAP 全部正确，CLAIM-AFFECTED 全部保留
- ✅ 生成器确定性验证通过
- ✅ ddof=1 统一

### 若修复阻断项后可封板

封板前必须完成的 5 项操作：
1. Commit 或 stash PROGRESS.md + provenance_manifest.csv
2. 删除或同步 `validation/handbook_generated.md`
3. 决定 INVALID 排除策略并执行（真排除 或 修正声明）
4. 在 regen_handbook.py 排序中加 tie-breaker
5. 重跑 regen_handbook.py 并 commit 确定性输出

---

*审查员独立编写，所有证据来自 `results/logs/E*-*.json` 直接读取、git 历史对比、脚本源码分析。*
*审查过程中运行的脚本产生了临时输出；已通过 `git checkout` 还原被修改的跟踪文件。*
*未做任何 commit。*
