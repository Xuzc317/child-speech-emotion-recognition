# 复核报告: design_vs_actual_diff.md

> **复核日期**: 2026-06-22
> **复核人**: 独立 reviewer
> **被复核文档**: `validation/design_vs_actual_diff.md` (以下简称 "diff")
> **真值来源**: `docs/current/实验设计方案_v3_含学习笔记.md` (设计), `validation/provenance_manifest.csv` (实际), `scripts/launch_b*.sh` (启动脚本), `src/train.py` (源码)
> **原则**: diff 是待检验声称，非真值。每项独立裁决，引用文件+行号为证据。

---

## 总裁决表

| # | 裁决 | diff 声称 | 实际 |
|---|------|----------|------|
| D1 | **❌ REFUTED** | 受影响实验 = E3-01~03, E3-16~18 (FAU 作为 target) | 受影响实验 = E3-07~12 (FAU 作为 source/training domain)。diff 把方向搞反了 |
| D2 | ✅ CONFIRMED | C-BESD 4类子集，设计§5 L122明确 | 设计原文逐字匹配，执行一致 |
| D3 | ✅ CONFIRMED | fusion_mode default=weighted | 设计 L336 与 train.py:282 均为 weighted |
| D4 | **❌ REFUTED (数学错误)** | "实际 54 runs (27 + 36层grid)" | 27+36=63≠54。正确拆分: 18(多seed) + 36(grid) = 54 |
| D5 | ✅ CONFIRMED | IEMO_CKPT=E1-09_s42 | launch_b7.sh:22 确认，E1-08 INVALID 导致降级使用 |
| D6 | ✅ CONFIRMED | augment_condition default=C1 | 设计 L335 与 train.py:308 均为 C1 |
| D7 | ⚠️ CANNOT-VERIFY | "设计方案_v3 全文未出现 reg_profile 字样" | 字面真 (字符串 "reg_profile" 未出现)，但概念在 §3 有完整定义；表述误导 |

---

## 逐条详细分析

### D1 — reg_profile 影响 B2 实验集 ❌ REFUTED

**diff 声称** (L14, L36-37): 受影响的是 FAU-target 实验 E3-01~03 (C-BESD→FAU) 和 E3-16~18 (IEMOCAP→FAU)，共 6 个。

**裁决依据**:

`reg_profile` 控制 4 个**训练时**超参数：weight_decay, label_smoothing, pooling_dropout, grad_clip (`src/train.py:326-337`)。在 B2 零样本迁移中，模型在源域训练、目标域测试（无微调）。因此 `--reg_profile` 作用于**源域（训练数据）**，与目标域无关。

**设计文档证据**:
- `实验设计方案_v3_含学习笔记.md` §3 E1 表 (L60-70): FAU 作为训练数据时用 `reg=fau` (E1-04~06)，C-BESD/IEMOCAP 用 `reg=default`
- 同理 §6 E4 表: FAU 训练用 `reg=fau`
- 一致的设计规则: **训练数据=FAU → reg=fau**，从未有 "测试数据=FAU → reg=fau" 的规则

**执行证据**:
- `scripts/launch_b2.sh:22-31`: `run_zs()` 不传 `--reg_profile` → 全部 18 个实验使用 code default `'default'`
- B2 实验矩阵 (`scripts/launch_b2.sh:33-61`):
  - E3-01~06: C-BESD 为源域 → reg=default 正确（C-BESD 训练本应 default）
  - **E3-07~12: FAU 为源域 → reg=default 错误（FAU 训练应 fau）**
  - E3-13~18: IEMOCAP 为源域 → reg=default 正确

**实际受影响实验**:
| 实验 | 源域 | 目标域 | 应 reg_profile | 实际 | 影响 |
|------|------|--------|----------------|------|------|
| E3-07~09 | **FAU** | C-BESD | `fau` | `default` | FAU 训练正则化不足 |
| E3-10~12 | **FAU** | IEMOCAP | `fau` | `default` | FAU 训练正则化不足 |

**diff 错误根源**: diff L37 写道 "尽管目标域测试在 FAU 上，模型训练时的正则化却是 default 而非 fau"——把 reg_profile 的作用对象理解为目标域而非源域。这是一个方向性错误。

**裁决**: **REFUTED** — 受影响实验集被写反。正确的受影响集是 E3-07~12（FAU 作为源/训练域），共 6 个实验。

---

### D2 — B2 跨语料 C-BESD 4 类子集 ✅ CONFIRMED

**diff 声称** (§5 L122): "训练时 C-BESD 使用 4 类子集（C_BESD_MAP_4CL，丢弃 disgust/fear），与 FAU/IEMOCAP 标签空间对齐。"

**设计原文** (`实验设计方案_v3_含学习笔记.md:122`):
> 训练时 C-BESD 使用 4 类子集（C_BESD_MAP_4CL，丢弃 disgust/fear），与 FAU/IEMOCAP 标签空间对齐。

逐字匹配。`scripts/launch_b2.sh:40-61` 所有 C-BESD 源域实验传递 `--train_data "c-besd-4cl"`。

**裁决**: **CONFIRMED** — 设计引用准确，执行一致。

---

### D3 — fusion_mode 默认值 ✅ CONFIRMED

**diff 声称**: 设计 L336 `--fusion_mode default=weighted`，train.py:282 `default='weighted'`，一致。

**证据**:
- `实验设计方案_v3_含学习笔记.md:336`: `| \`--fusion_mode\` | weighted | B4 使用 last/best_single/weighted |`
- `src/train.py:282`: `parser.add_argument('--fusion_mode', type=str, default='weighted', ...)`

**裁决**: **CONFIRMED** — 设计与代码默认值一致。

---

### D4 — B4 实验计数 ❌ REFUTED（数学错误）

**diff 声称** (L17): 设计说 "9 组×3 seeds=27 runs"，实际 "54 runs (27 + 36层grid search独立记录)"。

**分析**:

1. 设计文档内部不一致:
   - §7 L238: "9 组 × 3 seeds = 27 runs" — **错误**，3 个实验组是 grid search（非 multi-seed）
   - §10 L320: "E5-01~09 + layer scan \| 27+36 runs" — 27+36=63，同样**过估**（多算了 9 run: 3 grid 实验组被同时计入 multi-seed 和 grid）

2. diff 的数学错误: 27 + 36 = 63 ≠ 54。diff 写 "(27 + 36层grid search独立记录)" 来解释 54，但 27+36 实际等于 63。

3. 正确的 B4 实验拆分:
   - **18 runs**: 6 个 multi-seed 实验 × 3 seeds (E5-01, E5-03, E5-04, E5-06, E5-07, E5-09)
   - **36 runs**: 3 个 grid search 实验 × 12 层 × 1 seed (E5-02, E5-05, E5-08 各 L1-L12)
   - **总计: 18 + 36 = 54** ✓

   （`validation/provenance_manifest.csv` 中 B4 有 42 行 CSV 记录 = 18 行 multi-seed + 36 行 grid 单 seed）

**裁决**: **REFUTED** — diff 的数学不成立 (27+36≠54)。正确表述: "设计 §7 说 27 runs（未计入 grid search 替代 3×3 多 seed），实际 54 runs（18 multi-seed + 36 grid）"。设计自身在 §10 也有 27+36=63 的过估。

---

### D5 — B7 IEMOCAP 源 checkpoint ✅ CONFIRMED

**diff 声称**: 设计说 E1最优=IEMOCAP Self-Attn (E1-08)，实际用 E1-09 prosody_guided。

**证据**:
- 设计 §9 L283: 源模型字段为 "IEMOCAP best"（小写 best，不指定具体 pooling head）
- `scripts/launch_b7.sh:22`: `IEMO_CKPT="checkpoints/b1/E1-09_s42/best_model.pt"` — 确认使用 E1-09
- E1-08 seed=42 INVALID 原因: 旧协议残留，配置字段不可信 (`validation/provenance_manifest.csv`)
- `scripts/launch_b7.sh:31`: 微调阶段硬编码 `--pooling_type self_attention`，源头不参与迁移，无实质影响

**裁决**: **CONFIRMED** — launch 脚本确用 E1-09；影响为零（源池化头不迁移）。

---

### D6 — augment_condition 默认值 ✅ CONFIRMED

**diff 声称**: 设计 L335 `--augment_condition default=C1`，train.py:308 `default='C1'`，一致。

**证据**:
- `实验设计方案_v3_含学习笔记.md:335`: `| \`--augment_condition\` | C1 | B3 使用 C1~C4 |`
- `src/train.py:308`: `parser.add_argument('--augment_condition', type=str, default='C1', ...)`

**裁决**: **CONFIRMED** — 一致。

---

### D7 — reg_profile 在设计文档中的记录 ⚠️ CANNOT-VERIFY（表述误导）

**diff 声称** (L20): "设计方案_v3 全文未出现`reg_profile`字样"。

**字面真值**: 字符串 `reg_profile`（作为 CLI 参数名）确实未在设计文档中出现。CLI 参数速查表 (L330-339) 列出了 7 个参数，未含 `--reg_profile`。

**但明显的误导**: 设计文档对 reg_profile **概念**有完整定义:
- §3 E1 表 (L60-70): `Reg` 列标注每个实验用 `default` 或 `fau`
- §3 "### Reg Profile 定义" (L73-77): 完整对比表 — weight_decay, label_smoothing, pooling_dropout, grad_clip 的两种配置值
- §6 E4 表 (L186-197): 同样有 `Reg` 列
- `src/train.py:326-337` 的 reg_profiles 字典与设计 §3 定义完全一致

**实际脚本 reg_profile 传递情况**:
| 批次 | 是否传 `--reg_profile` | 脚本证据 |
|------|------------------------|---------|
| B1 | ✅ 显式传递 | `launch_b1.sh:34,62-69` (default/fau 依数据集) |
| B2 | ❌ **未传递** | `launch_b2.sh:22-31` run_zs() 不含此参数 |
| B3 | ✅ 显式传递 | `launch_b3.sh:41,52-73` |
| B4 | ✅ 显式传递 | `launch_b4.sh:38,70-85` |
| B5 | ✅ 显式传递 | `launch_b5.sh:36,46-48` |
| B6 | ✅ 显式传递 | `launch_b6.sh:41,129-142` |
| B7 | ✅ 显式传递 | `launch_b7.sh:41` |

**裁决**: **CANNOT-VERIFY** — 声称在字面上为真（字符串 absent），但这是误导性的；设计文档以不同粒度（Reg 列 / Reg Profile 定义节）覆盖了 reg_profile 概念，除 CLI 速查表遗漏外文档完备。建议将 D7 改为 "设计 CLI 速查表遗漏 --reg_profile 条目，但 §3 有 Reg Profile 完整定义"。

---

## 新增发现的分歧（diff 未覆盖）

### D-NEW-1 — B1 batch_size=32 vs 设计"统一 batch_size=16"

| 项目 | 值 |
|------|-----|
| 设计 | L334: `--epochs 100 --batch_size 16 --patience 15` (统一训练配置) |
| 实际 | `scripts/launch_b1.sh:40`: `--batch_size 32` |
| 严重程度 | 📝 低 — B1 完成最早，可能初始剧本不同；数值不影响结论 |
| diff 是否提及 | ❌ 未提及 |

### D-NEW-2 — B5 batch_size=8 vs 设计"统一 batch_size=16"

| 项目 | 值 |
|------|-----|
| 设计 | L334: `--batch_size 16` (统一); L100: "可选 gradient accumulation（batch_size 减半以适配显存）" |
| 实际 | `scripts/launch_b5.sh:38`: `--batch_size 8` |
| 严重程度 | 📝 低 — 设计 L100 已预见到显存原因导致的减半，部分可预期 |
| diff 是否提及 | ❌ 未提及 |

### D-NEW-3 — B7 fine-tuning lr=3e-4 vs 设计建议 1e-4

| 项目 | 值 |
|------|-----|
| 设计 | §9 L290: "微调时可用较小的 lr（如 1e-4）" |
| 实际 | `scripts/launch_b7.sh:30-46`: run_transfer() 未传 `--lr` → 落回 `src/train.py:296` 的 default=3e-4 |
| 严重程度 | ⚠ 低-中等 — 微调 lr 影响迁移效果，但设计用词为 "可用"（非强制） |
| diff 是否提及 | ❌ 未提及 |

---

## 设计引用准确性检核 (Check 2 逐项)

| diff 引用 | 设计位置 | 内容 | 准确性 |
|-----------|---------|------|--------|
| §3 E1表 | L60-70 | Reg 列 default/fau | ✅ 准确 — diff D1 正确引用了 Reg 标注 |
| §5 L122 | L122 | C-BESD 4类子集 | ✅ 准确 — 逐字匹配 |
| §6 E4表 | L186-197 | 同上 Reg 列 | ✅ 准确 |
| §7 L233 | L233 | E5-02 grid search 描述 | ✅ 准确 — 提到了 grid search |
| §7 L238 | L238 | "9 组×3 seeds=27 runs" | ✅ 准确 — 设计确有此写 |
| §9 L283 | L283 | IEMOCAP best | ✅ 准确 — 设计写 "IEMOCAP best" |
| §10 L320 | L320 | "27+36 runs" | ✅ 准确 — 设计确有此写 |
| §10 L330-339 | L330-339 | CLI 参数速查表 7 参数 | ✅ 准确 — 含 augment_condition/fusion_mode |
| §10 L335 | L335 | augment_condition=C1 | ✅ 准确 |
| §10 L336 | L336 | fusion_mode=weighted | ✅ 准确 |
| §1 L26 | L25-26 | data_split_seed=42 固定 | ✅ 准确 |
| §4 L98-99 | L98-99 | 差分 LR backbone 1e-5, head 3e-4 | ✅ 准确 |
| §5 L133 | L133 | B2 每条件 1 seed | ✅ 准确 |

**结论**: diff 对设计文档的所有引用均准确。设计文档本身的计数不一致（§7 27 vs §10 27+36）是设计文档的问题，非 diff 引用错误。

---

## 非 diff 抽查项 (Check 6)

### data_split_seed=42
全部 7 个 launch 脚本均显式传递 `--data_split_seed 42`:
- `launch_b1.sh:36`, `launch_b2.sh:28`, `launch_b3.sh:43`, `launch_b4.sh:40`, `launch_b5.sh:38`, `launch_b6.sh:42,61,80,99,119`, `launch_b7.sh:41`

✅ **确认** — data_split_seed=42 无处不在。

### B5 差分学习率
`scripts/launch_b5.sh:37`: `--unfreeze_ssl --ssl_lr 1e-5 --lr 3e-4`

与设计 §4 L98-99 (`backbone 1e-5, head 3e-4`) 完全一致。

✅ **确认** — B5 差分学习率准确执行。

---

## 总判定

**diff 作为「设计-执行分歧」可信来源的评估: ⚠️ 部分可用，需修正后方可采用。**

**关键缺陷**:
1. **D1 方向性错误 (严重)**: 把受影响实验集写反了（FAU-target vs FAU-source）。reg_profile 作用于训练域（源域），受影响的是 FAU 为**源域**的 E3-07~12，不是 FAU 为**目标域**的 E3-01~03/E3-16~18。论文中若引用此错误将导致设计-执行分歧节的实质性错误。
2. **D4 数学错误 (中等)**: 27+36≠54。虽然结论（54 runs）正确，但解释中的 27+36 加法无法等于 54。

**遗漏**:
- 3 项设计-执行差异未收录 (D-NEW-1~3)，其中 B7 lr 分歧 (3e-4 vs 建议 1e-4) 理论上可能影响迁移结果。

**可用价值**:
- D2, D3, D5, D6 均准确且证据充分
- 非分歧表 (L79-91) 准确
- D7 的建议（CLI 速查表补充 --reg_profile）合理

**建议**: 修正 D1 和 D4 后，diff 可作为设计-执行分歧节的基础材料，并补充 D-NEW-1~3。

---

## 附录: 证据文件索引

| 证据 | 文件 | 关键行 |
|------|------|--------|
| reg_profile 设计定义 | `docs/current/实验设计方案_v3_含学习笔记.md` | L60-77 |
| reg_profile 代码实现 | `src/train.py` | L326-337 |
| reg_profile 规则: FAU 训练→fau | `docs/current/实验设计方案_v3_含学习笔记.md` | L60-70, L73-77 |
| B2 launch 缺少 --reg_profile | `scripts/launch_b2.sh` | L22-31 |
| B2 实验方向 | `scripts/launch_b2.sh` | L33-61 |
| B7 IEMO_CKPT | `scripts/launch_b7.sh` | L22 |
| B4 launch 脚本 (grid + multi-seed) | `scripts/launch_b4.sh` | L48-86 |
| B4 实际实验清单 | `validation/provenance_manifest.csv` | B4 42 行 |
| B1 batch_size=32 | `scripts/launch_b1.sh` | L40 |
| B5 batch_size=8 | `scripts/launch_b5.sh` | L38 |
| B5 差分 LR | `scripts/launch_b5.sh` | L37 |
| B7 lr default=3e-4 | `src/train.py` | L296 |
| B7 设计建议 lr=1e-4 | `docs/current/实验设计方案_v3_含学习笔记.md` | L290 |
| 所有脚本 data_split_seed=42 | launch_b1.sh:36, b2:28, b3:43, b4:40, b5:38, b6 多处, b7:41 | — |
| design §5 L122 (C-BESD 4类) | `docs/current/实验设计方案_v3_含学习笔记.md` | L122 |
| design §3 Reg Profile 定义 | `docs/current/实验设计方案_v3_含学习笔记.md` | L73-77 |
