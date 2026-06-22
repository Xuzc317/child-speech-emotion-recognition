# 设计方案 vs 实际执行 — 分歧记录

> **生成日期**: 2026-06-22
> **设计来源**: `docs/current/实验设计方案_v3_含学习笔记.md`
> **执行来源**: `configs/conditions_of_record.csv` (基于 launch_b1.sh~launch_b7.sh)
> **协议**: `ac_suite_2026-06-validated`

---

## 分歧总览

| # | 类别 | 严重程度 | 设计意图 | 实际执行 | 影响 |
|---|------|---------|---------|---------|------|
| D1 | reg_profile | ⚠ 中等 | FAU 训练应使用 `reg_profile=fau`（B1/B3 均显式传参；设计方案_v3 §3 E1表 & §6 E4表均标注 Reg=fau） | B2 `run_zs()` 未传 `--reg_profile`，全部18个零样本实验落回 code default `'default'` | FAU 为**源域**的 6 个实验(E3-07~12: FAU→C-BESD, FAU→IEMOCAP)用 reg=default 而非 fau。注: reg_profile 作用于训练/源域，非目标域；原 v1 误归为 FAU-target |
| D2 | 类空间 | 📝 信息 | C-BESD 域内 6 类；跨语料强制对齐为 4 类(丢弃 disgust/fear) | `c-besd-4cl` 数据集子集，与设计一致 | 设计明确(§5 L122)，非分歧；论文需说明类空间不统一 |
| D3 | fusion_mode | ✅ 一致 | 设计 L336: `--fusion_mode default=weighted` | train.py:282 `default='weighted'` | 设计与代码默认一致；旧 manifest 误记为 `last`，已修正 |
| D4 | B4 实验计数 | 📝 信息 | 设计 §7 说 9 组×3 seeds=27 runs；§10 说 27+36 runs | 实际 54 runs = 18 (6 multi-seed×3 seeds) + 36 (3 grid search×12 layers) | 设计 §7(27) 与 §10(27+36=63) 自身计数不一致；正确拆分 18+36=54 |
| D5 | B7 IEMOCAP源 | ⚠ 低 | 设计: E1最优=IEMOCAP Self-Attn (E1-08) | 实际: 用 E1-09 prosody_guided (因E1-08被标记 INVALID, s42 使用旧协议) | 源pooling头不影响迁移结果(launch_b7.sh统一用self_attention池化头)，但设计未预见到INVALID情况 |
| D6 | augment_condition默认 | ✅ 一致 | 设计 L335: `--augment_condition` default=C1 | train.py:308 `default='C1'` | 一致；旧 manifest 误记为 `C0_none`(非法值)，已修正 |
| D7 | reg_profile CLI速查遗漏 | 📝 信息 | 设计方案_v3 §10 CLI 速查表(L330-339)缺少 `--reg_profile` 条目 | B1/B3/B4/B5/B6/B7 均显式传 `--reg_profile`；仅 B2 不传 | 设计 CLI 速查表不完整，但 §3 (L73-77) 有 Reg Profile 完整定义(default vs fau) |
| D-NEW-1 | B1 batch_size | 📝 低 | 设计 L334: `--batch_size 16` (统一) | launch_b1.sh:40 `--batch_size 32` | B1 最早实验，可能初始剧本不同；数值不影响结论 |
| D-NEW-2 | B5 batch_size | 📝 低 | 设计 L334: `--batch_size 16` (统一)；L100 预见显存减半 | launch_b5.sh:38 `--batch_size 8` | 设计已预见到显存原因减半，可预期 |
| D-NEW-3 | B7 fine-tune lr | ⚠ 低-中 | 设计 §9 L290: "微调可用较小 lr (如 1e-4)" | launch_b7.sh 未传 --lr → train.py default=3e-4 | 设计用词"可用"(非强制)，但建议值与实际值不一致 |

---

## 逐条详细分析

### D1 — B2 零样本实验 reg_profile 取 default

**设计期望**:
- B1 FAU 实验 (E1-04~06) 显式使用 `reg_profile=fau` (设计方案_v3 §3 E1表)
- B3 FAU 实验 (E4-05~08) 显式使用 `reg_profile=fau` (§6 E4表)
- 所有涉及 FAU 数据集的训练均使用 fau 正则化配置（更强正则化以应对 FAU 的高类别不平衡）

**实际执行**:
- `launch_b2.sh:run_zs()` (L22-31) 不传 `--reg_profile` 参数
- 全部 18 个零样本实验使用 code default `'default'`
- 具体影响：**E3-07~12 (FAU→C-BESD, FAU→IEMOCAP)** 共 6 个实验,
  FAU 作为**源域（训练域）**时应使用 `reg_profile=fau`，实际落回 default
- 注: reg_profile 作用于训练/源域，非测试/目标域；原 v1 误将受影响实验归为 FAU-target (E3-01~03, E3-16~18)，方向性错误

**后果评估**:
- reg_profile 影响 weight_decay、label_smoothing、pooling_dropout、grad_clip
- FAU 源域训练缺少更强的 dropout(0.3 vs 0.0)和 label_smoothing(0.15 vs 0.1)
- 这些实验是零样本(zero-shot)，源域训练后直接测试目标域;
  reg_profile 影响的是源域模型质量，间接影响迁移性能
- 严重程度中等：不影响核心结论(分布偏移→性能下降)，但 FAU 源域训练的模型可能略差于最优

### D2 — B2 跨语料 C-BESD 4 类子集

**设计意图** (§5 L122):
> "训练时 C-BESD 使用 4 类子集（C_BESD_MAP_4CL，丢弃 disgust/fear），与 FAU/IEMOCAP 标签空间对齐。"

**实际执行**: `launch_b2.sh` 传递 `--train_data "c-besd-4cl"`，与设计完全一致。

**论文写作提示**: 域内 B1 用 6 类 C-BESD，跨语料 B2 用 4 类 C-BESD——类空间不同，读者可能困惑，需在论文中解释。

### D3 — fusion_mode 默认值

**设计** L336: `--fusion_mode default=weighted`
**代码** train.py:282: `default='weighted'`
**结论**: 一致。设计方案 L27 也明确 "12层加权求和 (learnable softmax weights)"。
旧版 `conditions_of_record.csv` 错记为 `last`，已由当前版本修正。

### D4 — B4 实验计数

**设计文档内部不一致**:
- §7 L233-238: B4 描述为 9 组实验 (last/best_single/weighted × 3 datasets) × 3 seeds = "27 runs"
  但其中 3 组 (E5-02/05/08) 实际是 layer grid search，每层 1 seed，不应按 3 seeds 计
- §10 L320: "E5-01~09 + layer scan | 27+36 runs" — 将 3 个 grid search 组同时在 multi-seed(27) 和 grid(36) 中重复计数，结果为 63

**实际执行** (`scripts/launch_b4.sh`):
- 6 个 multi-seed 实验 × 3 seeds = **18 runs**: E5-01/03/04/06/07/09
- 3 个 grid search 实验 × 12 layers × 1 seed = **36 runs**: E5-02/05/08
- **总计: 18 + 36 = 54** ✓（`validation/provenance_manifest.csv` 中 B4 有 42 行 CSV 记录）

**严重程度**: 设计文档计数问题，执行无误。18+36=54 为正确拆分。

### D5 — B7 IEMOCAP 源 checkpoint

**设计** §9 L283: IEMOCAP 源模型 = "E1最优" (预期为 E1-08 self_attention)
**实际**: `launch_b7.sh:L22` 使用 `IEMO_CKPT=E1-09_s42 (prosody_guided)`
**原因**: E1-08 的 seed=42 使用了旧协议(aug/fusion/adapter 字段为 None)，被标记为 INVALID。
E1-09 (prosody_guided, WA=65.05%) 成为实际可用的 IEMOCAP 最优 checkpoint。
**影响**: 无。B7 fine-tune 阶段统一使用 `--pooling_type self_attention`，
源 checkpoint 的 prosody_guided 池化头不参与迁移(仅 backbone+LayerFusion 权重搬移)。

### D7 — CLI 速查表遗漏 --reg_profile，但 §3 有完整定义

设计方案_v3 的 "关键 CLI 参数速查" 表 (§10 L330-339) 列出了 7 个参数但遗漏 `--reg_profile`。
然而设计方案 §3 (L73-77) 有 Reg Profile 的完整定义表（weight_decay、label_smoothing、pooling_dropout、grad_clip
的 default vs fau 对比），且 §3 E1 表(L60-70)和 §6 E4 表均用 `Reg` 列标注每个实验的预期值。
B1/B3/B4/B5/B6/B7 的 launch 脚本均显式传递了 `--reg_profile`，仅 B2 因 `run_zs()` 函数简化而遗漏。
建议在 CLI 速查表中补充 `--reg_profile` 条目。

### D-NEW-1 — B1 batch_size=32 vs 设计统一 batch_size=16

**设计** §10 L334: `--epochs 100 --batch_size 16 --patience 15` (统一训练配置)
**实际**: `scripts/launch_b1.sh:40` 传递 `--batch_size 32`
**原因**: B1 完成最早，可能初始实验阶段使用了更大的 batch；数值不影响核心结论
**严重程度**: 📝 低

### D-NEW-2 — B5 batch_size=8 vs 设计统一 batch_size=16

**设计** §10 L334: `--batch_size 16` (统一); §4 L100: "可选 gradient accumulation（batch_size 减半以适配显存）"
**实际**: `scripts/launch_b5.sh:38` 传递 `--batch_size 8`
**原因**: B5 unfreeze 所有 12 层 WavLM，显存需求增加；设计 §4 L100 已预见到显存原因导致的减半
**严重程度**: 📝 低 — 设计已预先说明

### D-NEW-3 — B7 fine-tuning lr=3e-4 vs 设计建议 1e-4

**设计** §9 L290: "微调时可用较小的 lr（如 1e-4）"
**实际**: `scripts/launch_b7.sh:30-46` 的 `run_transfer()` 未传递 `--lr` → 落回 `src/train.py:296` 的 default=3e-4
**原因**: launch 脚本未传递 `--lr`，落回代码默认值 3e-4，与设计建议 1e-4 不一致
**严重程度**: ⚠ 低-中等 — 微调 lr 影响迁移效果；设计用词"可用"(非强制)，`train.py` default=3e-4 与 B1-B6 域内训练使用的一致

---

## Meta 小结: 分歧同源

上述 D1 (reg_profile)、D7 (CLI速查遗漏)、D-NEW-1 (B1 batch_size)、D-NEW-3 (B7 lr) 四处分歧共享同一根源:

> **launch 脚本相对设计欠 specified —— 设计意图中明确的参数（reg_profile=fau、lr=1e-4、batch_size=16）因脚本未传递而落回 `src/train.py` 的代码默认值。**

因此，**实验条件以 launch 脚本实际传递的参数为准，非以设计文档的文字描述为准。** 设计文档描述了"应该是什么"，launch 脚本决定了"实际是什么"——当两者冲突时，launch 脚本是更权威的执行记录。这也是本 diff 文档以脚本为执行来源、`conditions_of_record.csv` 为条件真值的根本原因。

---

## 非分歧项 (确认一致)

| 项目 | 设计 | 执行 | 验证方式 |
|------|------|------|---------|
| Pooling 类型 | E1最优池化用于后续phase | launch脚本一致 | B3/B4/B5/B6/B7 均引用最佳pooling |
| 数据增强 C1-C4 | B3 四种条件 | launch_b3.sh 显式传参 C1/C2/C3/C4 | 与设计§6 E4表一致 |
| B4 融合方式 | last/best_single/weighted | launch_b4.sh 显式传参 | 与设计§7 E5表一致 |
| B6 模块消融设计 | 逐步累加 | launch_b6.sh 分函数(minimal→adapter→pooling→fusion→full) | 与设计§8 E6表一致 |
| B7 迁移方向 | 6方向×3seeds | launch_b7.sh 6组×3 seeds | 与设计§9 E7表一致 |
| 数据划分 | data_split_seed=42 固定 | 所有脚本显式传参 | 与设计§1 L26一致 |
| B5 unfreeze | 差分学习率 backbone 1e-5, head 3e-4 | launch_b5.sh 传 --ssl_lr 1e-5 --lr 3e-4 | 与设计§4 L98-99一致 |
| B2 E3 1 seed | 零样本不需多seed | launch_b2.sh 仅 seed=42 | 设计§5 L133: "每条件 1 seed" |
