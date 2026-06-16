# 百科全书式实验设计方案

> **生成日期**: 2026-06-10  
> **协议 ID**: `ac_suite_2026-06`  
> **定位**: 方案 B — 完整覆盖三数据集 × 多池化 × 消融 × 层融合 × 零样本 × 模型迁移  
> **FAU 分类决策**: 4 分类 — A+E→Angry, P→Happy, N→Neutral, R→Sad

---

## 1. 数据集标准化方案

### 1.1 数据集总览

| 数据集 | 路径 | 样本数 | 原始类别 | 标准分类 | 说话人 | 年龄 | 风格 |
|--------|------|--------|---------|---------|--------|------|------|
| **C-BESD MY** | `D:/大学/论文/.../BESD/BESD/MY/` | 4,180 | 6类 (ANGER/DISGUST/FEAR/HAPPY/NEUTRAL/SAD) | **6分类** (保留全6类) | 75 child IDs | 儿童 | 演绎式 |
| **FAU Aibo** | `D:/大学/数据集IS2009EmotionChallenge/.../wav/` | 6,048 (已标注) | 5类 (A/E/N/P/R) | **4分类** (A+E→Angry, P→Happy, N→Neutral, R→Sad) | 51 speakers | 儿童 10-13y | 自发性 |
| **IEMOCAP** | `D:/大学/论文/.../IEMOCAP/wavs/` | ~8,525 (4类) | 原9类→过滤 | **4分类** (angry/happy/neutral/sad) | 10 speakers | 成人 | 演绎式 |

### 1.2 C-BESD 数据修复

**问题识别**：
- Child ID 仅存在于文件名中（如 `1.EF_12 Angry_1.wav` → child_id=1），CSV 无此列
- 旧版 `speaker_splitter.py` 使用 MD5 hash 划分，但未验证 child_id 提取正确性

**修复方案**：
1. 新增 `src/data/cbesd_parser.py`：从文件名正则提取 `child_id`（`^\d+`）
2. 验证：75 个唯一 child_id，每 child 55-61 样本
3. 更新 `speaker_splitter.py`：明确传入 `child_id` 列而非从文件名猜测
4. 划分协议：**70/15/15** (train/val/test)，按 child_id 严格隔离，seed=42
5. 论文操作描述："C-BESD contains 4,180 utterances from 75 English+Telugu bilingual children (ages unknown), recorded across 6 acted emotion categories. Child identity was extracted from the filename pattern and used for speaker-disjoint partitioning."

### 1.3 FAU Aibo 4 分类决策

**论文操作描述**（写入 Methodology 或 Dataset 节）：

> "FAU Aibo contains 18,216 spontaneous utterances from 51 German children (aged 10-13), originally labeled with 5 emotion categories. We remap the five-class scheme (A=Anger, E=Emphatic, N=Neutral, P=Positive, R=Rest) to a four-class set by merging Anger and Emphatic into a single *Angry* category (2,692 utterances), treating Positive as *Happy* (889), Neutral as *Neutral* (1,200), and Rest as *Sad* (1,267). This mapping aligns the FAU label space with C-BESD's four-class subset and IEMOCAP, enabling cross-corpus comparison on a unified four-emotion taxonomy. Of the 18,216 total recordings, 6,048 carry one of the five original emotion labels; only these labeled utterances are used in our experiments."

**合并逻辑**：
```
A (Anger, 1492) + E (Emphatic, 1200) → Angry (2692)
P (Positive, 889)                    → Happy (889)
N (Neutral, 1200)                    → Neutral (1200)
R (Rest, 1267)                       → Sad (1267)
总计: 6,048 labeled utterances
```

### 1.4 IEMOCAP 标准处理

**论文操作描述**：
> "IEMOCAP provides 9,903 utterances from 10 adult English-speaking actors in dyadic improvisation sessions. We retain the four most frequent emotion categories — angry, happy, neutral, sad — discarding 1,269 utterances with labels outside this set (e.g., excited, frustrated, surprised), following standard practice."

### 1.5 统一预处理流水线

| 步骤 | 参数 | 说明 |
|------|------|------|
| 音频重采样 | 16kHz mono | `audio_processor.py` |
| 峰值归一化 | peak normalization | 消除录音电平差异 |
| 最大时长截断 | 4s (200 frames @ 50Hz) | 防止 CUDA OOM |
| 说话人划分 | SHA256 hash, 70/15/15, seed=42 | 说话人零重叠，assert 验证 |
| 特征提取 | WavLM Base (wavlm-base-sv), frozen, 768-dim | 在线提取（非预存 .npy） |

---

## 2. 完整实验矩阵

### 2.1 实验分类总览

| 大类 | 实验数 | 目的 |
|------|--------|------|
| **E1: 域内池化对比** | 9 | 三数据集 × 3种池化的域内基准 |
| **E2: WavLM 解冻对比** | 3 | 冻结 vs 解冻 WavLM 对儿童/成人语料的影响 |
| **E3: 零样本迁移矩阵** | 18 | 三数据集 3×3 双向零样本（6方向 × 3池化） |
| **E4: 数据增强敏感性** | 12 | 3数据集 × 4增强条件的域内训练 |
| **E5: 层融合消融** | 9 | 单层最优 vs 加权求和 vs 最后层，三数据集 |
| **E6: 组块消融** | 6 | 逐步移除/添加模块，C-BESD + FAU |
| **E7: 模型迁移** | 6 | 最优模型跨数据集微调 |

**总计: 63 组实验**

### 2.2 E1: 域内池化对比 (Domain Pooling Baseline)

**目的**: 确定每个数据集上最优池化方式，作为后续消融的统一基线。

| ID | 数据集 | 分类数 | Pooling | Seed | 期望 WA |
|----|--------|--------|---------|------|---------|
| E1-01 | C-BESD | 6 | Mean Pooling | 42,123,456 | ~78% |
| E1-02 | C-BESD | 6 | Self-Attention | 42,123,456 | ~93% |
| E1-03 | C-BESD | 6 | Prosody Guided | 42,123,456 | ~91% |
| E1-04 | FAU Aibo | 4 | Mean Pooling | 42,123,456 | ~62% |
| E1-05 | FAU Aibo | 4 | Self-Attention | 42,123,456 | ~66% |
| E1-06 | FAU Aibo | 4 | Prosody Guided | 42,123,456 | ~65% |
| E1-07 | IEMOCAP | 4 | Mean Pooling | 42,123,456 | ~54% |
| E1-08 | IEMOCAP | 4 | Self-Attention | 42,123,456 | ~76% |
| E1-09 | IEMOCAP | 4 | Prosody Guided | 42,123,456 | ~66% |

**输出**: 三线表 Table 1 — 三数据集域内池化对比

### 2.3 E2: WavLM 解冻对比

**目的**: 验证"冻结 SSL 主干 + 轻量分类头"范式的必要性。

| ID | 数据集 | Pooling | WavLM | Seed | 备注 |
|----|--------|---------|-------|------|------|
| E2-01 | C-BESD | Self-Attn (E1最优) | **解冻 (full fine-tune)** | 42,123,456 | 参数量 ~95M |
| E2-02 | FAU | Self-Attn (E1最优) | **解冻** | 42,123,456 | 小数据过拟合风险 |
| E2-03 | IEMOCAP | Self-Attn (E1最优) | **解冻** | 42,123,456 | 成人基线 |

### 2.4 E3: 零样本迁移矩阵 (3×3 × 3 Pooling)

**目的**: 测量分布偏移对直接迁移的影响，验证 FD-WA 单调关系。

| ID | 训练→测试 | 分类数 | Pooling | Seed |
|----|----------|--------|---------|------|
| E3-01 | C-BESD→FAU | 4 (C-BESD 4类子集) | Mean | 42 |
| E3-02 | C-BESD→FAU | 4 | Self-Attn | 42 |
| E3-03 | C-BESD→FAU | 4 | Prosody | 42 |
| E3-04 | C-BESD→IEMOCAP | 4 | Mean | 42 |
| E3-05 | C-BESD→IEMOCAP | 4 | Self-Attn | 42 |
| E3-06 | C-BESD→IEMOCAP | 4 | Prosody | 42 |
| E3-07 | FAU→C-BESD | 4 | Mean | 42 |
| E3-08 | FAU→C-BESD | 4 | Self-Attn | 42 |
| E3-09 | FAU→C-BESD | 4 | Prosody | 42 |
| E3-10 | FAU→IEMOCAP | 4 | Mean | 42 |
| E3-11 | FAU→IEMOCAP | 4 | Self-Attn | 42 |
| E3-12 | FAU→IEMOCAP | 4 | Prosody | 42 |
| E3-13 | IEMOCAP→C-BESD | 4 | Mean | 42 |
| E3-14 | IEMOCAP→C-BESD | 4 | Self-Attn | 42 |
| E3-15 | IEMOCAP→C-BESD | 4 | Prosody | 42 |
| E3-16 | IEMOCAP→FAU | 4 | Mean | 42 |
| E3-17 | IEMOCAP→FAU | 4 | Self-Attn | 42 |
| E3-18 | IEMOCAP→FAU | 4 | Prosody | 42 |

**输出**: 三线表 Table 2 — 3×3 零样本迁移矩阵（WA + UAR）

### 2.5 E4: 数据增强敏感性

**目的**: 复现并扩展 FD→WA 单调关系。

**增强条件**:
- C1: 无增强 (clean, FD≈0)
- C2: 成人增强 (IEMOCAP 数据混入训练)
- C3: 儿童同语料增强 (SafeAWGN SNR 10-20dB)
- C4: 极端增强 (成人数据 + AWGN + 变速)

| ID | 数据集 | 增强条件 | Pooling (E1最优) | Seed |
|----|--------|---------|-----------------|------|
| E4-01 | C-BESD | C1 | Best | 42,123,456 |
| E4-02 | C-BESD | C2 | Best | 42,123,456 |
| E4-03 | C-BESD | C3 | Best | 42,123,456 |
| E4-04 | C-BESD | C4 | Best | 42,123,456 |
| E4-05 | FAU | C1 | Best | 42,123,456 |
| E4-06 | FAU | C2 | Best | 42,123,456 |
| E4-07 | FAU | C3 | Best | 42,123,456 |
| E4-08 | FAU | C4 | Best | 42,123,456 |
| E4-09 | IEMOCAP | C1 | Best | 42,123,456 |
| E4-10 | IEMOCAP | C2 | Best | 42,123,456 |
| E4-11 | IEMOCAP | C3 | Best | 42,123,456 |
| E4-12 | IEMOCAP | C4 | Best | 42,123,456 |

### 2.6 E5: 层融合消融

**目的**: 验证 12 层加权融合 vs 单层的最优性。

| ID | 数据集 | 融合方式 | 说明 | Seed |
|----|--------|---------|------|------|
| E5-01 | C-BESD | Last Layer (L12) | 仅最后一层 | 42,123,456 |
| E5-02 | C-BESD | Best Single (grid search) | 网格搜索最优单层 | 42,123,456 |
| E5-03 | C-BESD | **Weighted Sum (12层)** | 当前 baseline | 42,123,456 |
| E5-04 | FAU | Last Layer (L12) | — | 42,123,456 |
| E5-05 | FAU | Best Single | — | 42,123,456 |
| E5-06 | FAU | **Weighted Sum** | — | 42,123,456 |
| E5-07 | IEMOCAP | Last Layer (L12) | — | 42,123,456 |
| E5-08 | IEMOCAP | Best Single | — | 42,123,456 |
| E5-09 | IEMOCAP | **Weighted Sum** | — | 42,123,456 |

### 2.7 E6: 组块消融

**目的**: 量化每个模块（Adapter, Prosody Pooling, Layer Fusion）的独立贡献。

**固定**: C-BESD 6分类 + FAU 4分类，使用 E1 最优池化。

| ID | 数据集 | Adapter | Pooling | LayerFusion | 说明 |
|----|--------|---------|---------|-------------|------|
| E6-01 | C-BESD | ✗ | Mean | ✗ (Last L) | 最简基线 |
| E6-02 | C-BESD | ✓ | Mean | ✗ | +Adapter |
| E6-03 | C-BESD | ✗ | Best | ✗ | +Pooling |
| E6-04 | C-BESD | ✗ | Best | ✓ | +LayerFusion |
| E6-05 | C-BESD | ✓ | Best | ✓ | Full stack |
| E6-06 | FAU | ✓ | Best | ✓ | Full stack (域内基线) |

### 2.8 E7: 模型迁移

**目的**: 验证最优模型在不同数据集上的迁移能力（与零样本不同：允许微调）。

| ID | 源模型 (预训练) | 微调数据 | 测试数据 | Seed |
|----|---------------|---------|---------|------|
| E7-01 | C-BESD best | FAU | FAU | 42,123,456 |
| E7-02 | C-BESD best | IEMOCAP | IEMOCAP | 42,123,456 |
| E7-03 | FAU best | C-BESD | C-BESD | 42,123,456 |
| E7-04 | FAU best | IEMOCAP | IEMOCAP | 42,123,456 |
| E7-05 | IEMOCAP best | C-BESD | C-BESD | 42,123,456 |
| E7-06 | IEMOCAP best | FAU | FAU | 42,123,456 |

---

## 3. 实验执行顺序 (AutoDL 批次)

### 3.1 批次规划

| 批次 | 实验 | GPU 预估 | 依赖 | 输出 |
|------|------|---------|------|------|
| **B1** | E1-01~09 (域内池化) | ~6h | — | Table 1 |
| **B2** | E3-01~18 (零样本) | ~4h | B1 (需最优模型) | Table 2 |
| **B3** | E4-01~12 (增强敏感性) | ~8h | B1 (需最优池化) | Table 3-5 |
| **B4** | E5-01~09 (层融合消融) | ~6h | B1 | Table 6 |
| **B5** | E2-01~03 (WavLM 解冻) | ~12h | B1 | Table 7 |
| **B6** | E6-01~06 (组块消融) | ~4h | B1,B4 | Table 8 |
| **B7** | E7-01~06 (模型迁移) | ~6h | B1(最优模型) | Table 9 |

### 3.2 关键依赖链

```
B1 (域内池化) 
  ├── 确定每个数据集的最优池化方式
  ├── B2 (零样本) ← 需要 B1 最优模型做源模型
  ├── B3 (增强) ← 需要 B1 最优池化配置
  ├── B4 (层融合) ← 可与 B2/B3 并行
  ├── B5 (解冻) ← 可与 B2/B3/B4 并行
  ├── B6 (消融) ← 需要 B4 结果确定层融合配置
  └── B7 (迁移) ← 需要 B1 最优模型 + B2 结果
```

---

## 4. 预期输出物

### 4.1 科研表格 (三线表)

| 表格 | 内容 | 数据来源 |
|------|------|---------|
| Table 1 | 三数据集域内池化对比 (WA + UAR) | E1 |
| Table 2 | 零样本 3×3 迁移矩阵 (WA) | E3 |
| Table 3 | C-BESD 增强敏感性 FD-WA | E4-01~04 |
| Table 4 | FAU 增强敏感性 | E4-05~08 |
| Table 5 | IEMOCAP 增强对照 | E4-09~12 |
| Table 6 | 层融合消融结果 (Last/BestSingle/Weighted) | E5 |
| Table 7 | WavLM 冻结 vs 解冻 | E2 |
| Table 8 | 组块消融 (模块贡献) | E6 |
| Table 9 | 模型迁移 fine-tune 结果 | E7 |

### 4.2 科研图表

| 图表 | 内容 | nature-figure 原型 |
|------|------|-------------------|
| fig01 | 主结果矩阵 (Table 1 可视化) | quantitative grid |
| fig02 | 零样本 3×3 热力图 | quantitative grid |
| fig03 | FD-Accuracy 散点图 (三数据集) | quantitative grid |
| fig04 | XAI 三联图 (韵律+注意力对照) | asymmetric mixed-modality |
| fig05 | 层融合权重分布 | quantitative grid |
| fig06 | 模块消融瀑布图 | quantitative grid |
| fig07-08 | C-BESD 混淆矩阵 (Self-Attn vs Prosody) | quantitative grid |
| fig09 | 模型迁移前后 WA 对比 | quantitative grid |
| fig00 | 系统架构图 | schematic-led composite |

### 4.3 PPT 结构

| 章节 | slide 数 | 内容 |
|------|---------|------|
| 背景与动机 | 3 | 儿童 SER 挑战、分布偏移概念 |
| 数据集 | 3 | 三数据集概况、FAU 4分类决策、划分方案 |
| 方法 | 4 | 架构总览、三层 Pooling、层融合、增强策略 |
| 域内基线 | 3 | Table 1 + fig07/08 |
| 零样本迁移 | 2 | Table 2 热力图 + FD 分析 |
| 消融实验 | 4 | 层融合 + 组块 + 增强敏感性 |
| 模型迁移 | 2 | Table 9 |
| 总结 | 2 | 核心发现 + 局限 + 未来工作 |

---

## 5. Nature-Skills 调用计划

| 阶段 | Skill | 输入 | 输出 |
|------|-------|------|------|
| 实验设计审查 | `nature-reviewer` | 本文档 | 实验逻辑弱点、缺失对照 |
| 文献搜索 | `nature-academic-search` (multi-source-search) | 儿童SER、FD、韵律、跨语料关键词 | BibTeX 参考文献 |
| 论文写作 | `nature-writing` (research, zh-to-en) | 实验 JSON + 本文档 | `paper_draft/*.tex` 全部章节 |
| 文本配引用 | `nature-citation` (CNS及子刊) | 论文 .tex 段落 | claim→citation 映射 + .bib |
| 英文润色 | `nature-polishing` (en, generic) | 定稿 .tex | 润色后终稿 |
| 配图 | `nature-figure` (Python) | 每组实验 JSON | fig00-09 |
| 汇报 | `nature-paper2ppt` | 论文终稿 | 答辩 PPT |

---

*本文档将随实验推进持续更新。所有实验数值以 `results/logs/` 下 JSON 为权威数据源。*
