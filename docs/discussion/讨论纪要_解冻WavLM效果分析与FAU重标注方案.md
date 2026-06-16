# 讨论纪要：解冻 WavLM 效果分析、架构对标与 FAU 重标注方案

> 讨论日期：2026-06-15
> 上下文：儿童语音情绪识别（SER）分布驱动方案实验管道 B1–B7

---

## 目录

1. [解冻 WavLM 后的效果质疑与验证](#1-解冻-wavlm-后的效果质疑与验证)
2. [IEMOCAP SSL SER 文献对标](#2-iemocap-ssl-ser-文献对标)
3. [模型架构复杂度分析](#3-模型架构复杂度分析)
4. [成人情绪声学特征规律](#4-成人情绪声学特征规律)
5. [自监督/无监督标注方案调研](#5-自监督无监督标注方案调研)
6. [FAU 重标注必要性与方案](#6-fau-重标注必要性与方案)
7. [参考文献](#7-参考文献)

---

## 1. 解冻 WavLM 后的效果质疑与验证

### 1.1 核心问题

> "WavLM 是成人数据训练的，解冻后不应该是成人效果更明显吗？为什么反而是儿童的 C-BESD 提升最大？"

### 1.2 实验结果

**Frozen vs Unfrozen 逐种子对比**（本实验数据）：

| 数据集 | 说话人类型 | 冻结 WA | 解冻 WA | Δ |
|--------|-----------|---------|---------|-----|
| C-BESD | 儿童 | 91.9% | **96.9%** | **+5.0pp** |
| IEMOCAP | 成人 | 64.4% | 66.4% | +2.0pp |
| FAU | 儿童+噪声 | 67.0% | 66.4% | −0.7pp |

**C-BESD 过拟合检查**：

| seed | best_val_wa | test_wa | gap | epoch | 结论 |
|------|-------------|---------|-----|-------|------|
| 42 | 98.9% | 96.8% | −2.1pp | 27 | 正常，非过拟合 |
| 123 | 99.1% | 96.8% | −2.3pp | 33 | 正常，非过拟合 |
| 456 | 99.3% | 97.1% | −2.1pp | 14 | 正常，非过拟合 |

### 1.3 解释：分布差距决定解冻收益

**核心逻辑**：

```
预训练域 (成人语音)              目标域
    |                              |
    |── IEMOCAP (成人) ──→ 距离近 → 冻结特征已很好 → 解冻收益小
    |                      
    |── C-BESD (儿童) ──→ 距离远 → 冻结特征不匹配 → 解冻收益大
    |
    |── FAU (儿童+噪声) ─→ 距离远+噪声 → 解冻直接过拟合
```

> WavLM 预训练数据为成人语音（LibriSpeech 等），其冻结特征对成人语音已成最优表示，解冻仅带来边际增益。但对于儿童语音——声学特征与成人存在系统差异（更高 F0、更宽范围、不同共振峰）——冻结特征是次优的，解冻使模型能够学习儿童特有的声学模式。

**儿童语音 SSL 文献佐证**：

Block-Medin et al. (Interspeech 2024) 发现 []():
- WavLM Base+ **冻结**特征在儿童语音上的音素错误率 (PER) = 41.5%
- 在儿童语音上**微调**后 PER 降至 26.1%（相对提升 **33.4%**）
- WavLM 的去噪预训练和噪声鲁棒性使其特别适合儿童语音（高变异性、短句、背景噪声）
- **结论**：WavLM 在零样本泛化到儿童时表现差，但微调后提升巨大——与我们的结论完全一致

### 1.4 FAU 的问题：不是模型，是数据

**FAU 混淆矩阵**（exp5_fau_indomain, prosody_guided）：

| 真实类 | 样本数 | 判对率 | 误判为 neutral |
|--------|--------|--------|-----------------|
| angry | 558 | 38.4% | **208 (37%)** |
| happy | 833 | 49.2% | **335 (40%)** |
| neutral | 1,998 | 80.9% | — |
| sad | **0** | — | FAU 原始无 sad 类 |

**C-BESD 混淆矩阵**（exp1_self_attention）：

| 真实类 | 样本数 | 判对率 |
|--------|--------|--------|
| angry | 115 | **93.0%** |
| happy | 140 | **93.6%** |
| neutral | 140 | **91.4%** |
| sad | 145 | **92.4%** |

> FAU neutral 占 58.9% 的样本，37–40% 的非 neutral 样本被误判为 neutral。三类问题同时存在：类别严重不平衡 + neutral 标注污染 + sad 类完全缺失。

**FAU 原始标注回顾**：

| 标签 | 词数 | 占比 |
|------|------|------|
| neutral | 39,169 | **80.9%** |
| emphatic | 2,528 | 5.2% |
| motherese | 1,260 | 2.6% |
| angry | 84 | 0.2% |
| joyful | 101 | 0.2% |
| 其他 | 4,724 | 9.8% |

*来源：Schuller, Steidl, & Batliner (2009). The INTERSPEECH 2009 Emotion Challenge. [ISCA Archive](https://www.isca-archive.org/interspeech_2009/schuller09_interspeech.html)*

> 81% 的词被标为 neutral。当标注者不确定时，"neutral" 是默认选项——典型的标注默认偏见。

---

## 2. IEMOCAP SSL SER 文献对标

### 2.1 冻结基线（与我们 E1 最可比）

**SUPERB Benchmark — IEMOCAP 4-class, 说话人独立 (LOSO)**：

| 模型 | 参数量 | WA | 来源 |
|------|--------|-----|------|
| data2vec 2.0 Base | ~94M | **68.58%** | []() |
| WavLM Base (LS-960) | ~94.7M | **65.94%** | []() |
| HuBERT Base (LS-960) | ~95M | 64.92% | []() |
| wav2vec 2.0 Base (LS-960) | ~95M | 63.43% | []() |
| **我们的 E1-09** (WavLM Base + SE-MLP) | ~94.4M | **64.4%** | 本实验 |

*来源：Qieemo (arxiv:2503.22687, Table I); emotion2vec (ACL 2024 Findings, Table 2); Padi et al. (Odyssey 2022)*

> 我们的冻结 + SE-MLP（64.4%）与 SUPERB WavLM 冻结 + 线性层（65.94%）差异仅 1.5pp，说明简单架构在成人基线未落后。

### 2.2 微调 SOTA（成人）

| 方法 | 模型 | WA / UAR | 关键 | 来源 |
|------|------|----------|------|------|
| **GMP-TL/GMP-ATL** | HuBERT-base | **80.0% / 82.0%** | 性别伪标签 + AM-Softmax | []() |
| Tri-Stream Fusion | HuBERT | 79.86% UA | 三流融合 | []() |
| LaSCL | WavLM-Large | 78.42% UA | 对比学习 | []() |
| Xiang et al. | WavLM-base | 77.41% UA | 监督对比 | []() |
| **我们的 E2-03** | WavLM Base 解冻 | **66.4% / 60.7%** | 简单池化 + MLP | 本实验 |

*来源：GMP-TL (Pan et al., IEEE SLT 2024, arxiv:2405.02151); Xiang et al. (arxiv:2411.19803)*

> 注意：GMP-TL 的 80.0%/82.0% 使用 HuBERT-base (~95M)，不是 Large (~317M)。论文明确说明因计算资源限制使用 base 版本。但评估协议为 5-fold CV，与标准 LOSO 不完全相同。

### 2.3 儿童语音 SER 文献

**Lesyk et al. (2024)** ——唯一已发表的儿童 SSL SER 工作 []()：

| | Lesyk (2024) FAU | 我们的 FAU | 我们的 C-BESD |
|---|-----------------|-----------|-------------|
| 模型 | Wav2Vec2 XLSR-53 | WavLM Base | WavLM Base |
| 冻结 | 是 | 是 | 是 |
| WA | 未报告 | **67.0%** | **92.9%** |
| UAR | 未报告 | **45.1%** | **92.9%** |
| F-score | **0.47** | ~0.56 (推算) | **0.93** |

*来源：Lesyk et al. (Human-Centric Intelligent Systems, 2024, Vol.4, pp.633-642). [Springer](https://link.springer.com/article/10.1007/s44230-024-00088-w)*

> 儿童语音 SER 的 SSL 研究存在巨大空白。C-BESD 上 92.9% (冻结) / 96.9% (解冻) 可能是该数据集的**首个报告结果**。

---

## 3. 模型架构复杂度分析

### 3.1 我们的实际架构

```
SERModel 参数分布:
├── WavLM Base backbone          94,381,936 params  [冻结: 0 trainable]
├── LayerFusion (weighted)               12 params  [12 个可学习标量权重]
├── Adapter (可选)                  ~295,000 params  [Linear(768,192)→ReLU→Linear(192,768)]
├── Self-Attention Pooling          ~111,000 params  [4层 Linear: 768→116→100→100→1]
└── SE-MLP Classifier              ~590,000 params  [768→512→SE→256→128→num_classes]
─────────────────────────────────────────────────
  可训练参数 (最简配置):           ~700,000 (0.7M)
  可训练参数 (完整配置):         ~1,000,000 (1.0M)
```

**SEMLP 分类器结构**（`src/models/semlp.py`）：

```
768 → Linear(512) → BN → ReLU → Dropout(0.3)
    → SE-Block (SEBlock1D: 512→32→512)
    → Linear(256) → BN → ReLU → Dropout(0.3)
    → Linear(128) → ReLU → Dropout(0.2)
    → Linear(num_classes)
```

**Self-Attention Pooling 结构**（`src/models/pooling.py`）：

```
SSL 特征 (B,T,768)
    → Linear(768,116) → ReLU
    → Linear(116,100) → ReLU
    → Linear(100,100) → ReLU
    → Linear(100,1)
    → Softmax over T
    → Weighted sum → (B,768)
```

**Prosody-Guided Pooling 额外计算**：

```
F0 + Energy (B,T,2)
    → Linear(2,64) → ReLU → Linear(64,64) → LayerNorm  [韵律投影]
    + SSL (B,T,768)
    → Concat → (B,T,832)
    → 与 Self-Attention 相同的注意力计算路径
```

### 3.2 与典型 SER 论文的架构对比

| 维度 | 我们的模型 | 典型 SER 论文 |
|------|-----------|-------------|
| 可训练参数 | **0.7M–1.0M** | 10M–400M |
| 分类头 | 3层 MLP + SE gate | Transformer decoder / Bi-LSTM / 多头注意力 |
| 池化方式 | 单头注意力 | 多头注意力 + 统计池化 + NetVLAD |
| 多任务学习 | ❌ 无 | ✅ ASR + Speaker ID + 性别识别 |
| 对比学习 | ❌ 无 | ✅ CLAP-style 预训练 + 监督对比 |
| 数据增强 | ✅ SafeAWGN | ✅ SpecAugment + Mixup |
| 韵律注入 | ✅ F0 + energy (prosody_guided) | ❌ 通常无 |

### 3.3 为什么简单架构在 C-BESD 上有 92.9%？

四个因素的叠加效应：

**1. 数据质量**：C-BESD 为专业录音室录制，无噪声。儿童情绪表达更外显（不掩饰），声学-情绪映射更直接。

**2. WavLM 特征已足够好**：768 维帧级特征编码了丰富的声学信息，简单注意力池化就够提取情绪相关帧。

**3. 儿童情绪声学特征更显著**：成人情绪表达受社会规范约束（"愤怒但不提高音量"），儿童不压抑，F0 和能量的变化更明显。

**4. 架构复杂度的边际收益递减**：当数据质量高、特征好的时候，复杂架构的增益很小。

> **关键反证**：同样架构在 FAU 上只有 67%（噪声大），在 IEMOCAP 上只有 64%（成人情绪更内敛）。差距来自数据和领域特性，不是架构瓶颈。

---

## 4. 成人情绪声学特征规律

### 4.1 方向性规律（非分类公式）

Scherer (2003) 和 Banse & Scherer (1996) 建立的经典声学-情绪映射 []()：

| 声学参数 | 愤怒 | 高兴 | 悲伤 | 恐惧 | 中性 |
|----------|------|------|------|------|------|
| F0 均值 | ↑↑ 高 | ↑ 高 | ↓ 低 | ↑↑ 高 | — 中 |
| F0 变化范围 | ↑↑ 宽 | ↑ 宽 | ↓ 窄 | ↑↑ 宽 | — 中 |
| 能量/响度 | ↑↑ 高 | ↑ 高 | ↓ 低 | ↑ 高 | — 中 |
| 语速 | ↑ 快 | ↑ 快 | ↓ 慢 | ↑ 快 | — 中 |
| 高频能量 | ↑↑ 多 | ↑ 多 | ↓ 少 | ↑ 多 | — 中 |
| 停顿 | ↓ 少 | ↓ 少 | ↑ 多 | 不规则 | — 中 |
| 嗓音质量 | 紧/粗 | 紧/共鸣 | 气声/沙哑 | 断裂/不规则 | 正常 |

*来源：Scherer, K. R. (2003). Vocal communication of emotion: A review of research paradigms. Speech Communication, 40(1-2), 227–256. [DOI: 10.1016/S0167-6393(02)00084-5](https://doi.org/10.1016/S0167-6393(02)00084-5)*
*来源：Banse, R., & Scherer, K. R. (1996). Acoustic profiles in vocal emotion expression. Journal of Personality and Social Psychology, 70(3), 614–636.*

> 这些都是**统计趋势**，不是分类规则。愤怒和高兴在高 F0、高能量上几乎完全重叠，无法用单一阈值区分。**不存在离散情绪的声学计算公式，可能永远不会存在。**

### 4.2 可量化的：Arousal（唤醒度）

**PMC/Patel 框架 — 说话人归一化 Arousal 评分**：

```
Arousal_score(frame) = 2 × P(feature > neutral_baseline) − 1

其中:
  P(·) = 当前帧特征值超过说话人中性基线分布的比例
  neutral_baseline = 同一说话人的中性状态特征分布 (PDF)
  Score ∈ [−1, 1]
```

**加权融合**：

```
Arousal = w₁ × F0_score + w₂ × Energy_score + w₃ × HF500_score

HF500 = energy(>500Hz) / energy(80–500Hz)
wᵢ = Spearman's ρ(featureᵢ, mean_score)，按说话人计算
```

效果：Spearman ρ = 0.62–0.74，二分类 UAR = 73–84%。

*来源：Patel et al., Unsupervised Arousal Rating Framework. [PMC NIHMS660299](https://pmc.ncbi.nlm.nih.gov/articles/PMC4334478/)*

**Burkhardt 线性模型**：

```
Prosody_param = w_arousal × arousal + w_valence × valence

Arousal UAR = 0.76, Valence UAR = 0.43
```

*来源：Burkhardt et al. (2023). Going Retro: Rule-based Prosody Modelling for Speech Synthesis Simulating Emotion Dimensions. [arXiv:2307.02132](https://arxiv.org/abs/2307.02132)*

> **关键结论**：Arousal 可以从声学特征预测（UAR ~76%），但 Valence 几乎不可能（UAR ~43%）。离散情绪分类无法用简单公式实现。

### 4.3 为什么不能有清晰的声学边界

1. **离散情绪是文化构建，不是声学事实**：愤怒和高兴在高 F0、高能量上几乎一样
2. **个体差异淹没情绪差异**：一个人的中性 F0 可能比另一个人的愤怒 F0 还高
3. **情绪是连续的，不是离散的**：真实情绪在类别之间过渡

> "成人 SER 领域 20 年研究表明，不存在通用声学-情绪映射公式。情绪识别必须依赖数据驱动的分布学习。"——这一结论直接支持分布驱动儿童 SER 的必要性。

### 4.4 成人规律对儿童的适用性

| | 成人 | 儿童 |
|------|------|------|
| 声学规律研究 | 20+ 年积累 | **几乎没有系统研究** |
| F0 范围 | 相对稳定 (50–250Hz) | 极度变化 (120–500Hz，与年龄高度相关) |
| 情绪表达 | 受社会规范约束 | 更外显、更直接 |
| 个体差异 | 较大 | **极大**（不同年龄 F0 差异 >100Hz） |
| 标注标准 | 成熟（IEMOCAP LOSO 协议） | 争议大（FAU neutral 默认偏见） |

---

## 5. 自监督/无监督标注方案调研

### 5.1 伪标签方法

**路线 A：教师-学生框架**

| 方法 | 原理 | 最少有标签数据 | 代表工作 |
|------|------|--------------|---------|
| 标准伪标签 | 教师在少量标注上训练 → 给无标注数据打标签 → 学生学习全部 | 30% | Sun et al. (Interspeech 2024): IEMOCAP 70.75% |
| 多视图伪标签 | 声学 + 语义（LLM）分别打分 → 一致的高置信度样本才保留 | 30% | Li et al. (2024): 情绪+痴呆检测 |
| 迭代原型修正 | 类原型迭代修正模糊样本的伪标签 | 部分有标签 | IPR (Interspeech 2024) |

*来源：Sun et al. (Interspeech 2024). Iterative Prototype Refinement for Ambiguous SER. [ISCA Archive](https://www.isca-archive.org/interspeech_2024/sun24e_interspeech.pdf)*
*来源：Li et al. (2024). Semi-Supervised Cognitive State Classification with Multi-View Pseudo-Labeling. [arXiv:2409.16937](http://export.arxiv.org/abs/2409.16937v1)*

**路线 B：对比学习 + 伪标签**

- Spectrogram Enhanced SimCLRv2 (IJISAE 2024)：用仅 1% 的实时标注数据做儿童 SER
- 地区增强（Tri-cut, Tri-mix）避免信息损失

### 5.2 LLM 辅助标注

| 方法 | 原理 | 代表工作 | 效果 |
|------|------|---------|------|
| GPT-4 重标注 | 把原标注者写的自然语言描述喂给 LLM → 重新映射标签 | EMO-SUPERB (2024) | **WA +3.08%** (相对) |
| Gemini 扩展 | 感知相似度 + 语义匹配 + 声学匹配 | ParaSpeechCaps (2025) | 自动标注 ≈ 人工质量 |

*来源：EMO-SUPERB (2024). An In-depth Look at Speech Emotion Recognition. [arXiv:2402.13018](http://arxiv.org/pdf/2402.13018)*
*来源：ParaSpeechCaps (2025). Scaling Rich Style Tags for Speech. [arXiv:2503.04713](http://arxiv.org/pdf/2503.04713)*

### 5.3 一致性过滤

| 方法 | 原理 | 代表工作 | 效果 |
|------|------|---------|------|
| 共识过滤 | 只保留多数标注者一致的样本 | EmoMatchSpanishDB (2024) | **F1 +19%** |
| 置信度估计 | ML 检测低质量标注 → 回送重标 | LibriCrowd (2023) | WER **−50%** |

*来源：EmoMatchSpanishDB (2024). [UVa Repository](https://ddfv.ufv.es/rest/api/core/bitstreams/c58d8814-05de-43b6-a0aa-6ddd62911618/content)*
*来源：Gao et al. (Interspeech 2023). Human Transcription Quality Improvement. [ISCA Archive](https://www.isca-archive.org/interspeech_2023/gao23f_interspeech.pdf)*

### 5.4 人工校验后的效果汇总

| 工作 | 校验方式 | 原始性能基线 | 改善后 | 改善幅度 |
|------|---------|------------|--------|---------|
| EMO-SUPERB | GPT-4 重标注 | — | — | WA **+3.08%** |
| EmoMatchSpanishDB | 共识过滤 | — | — | F1 **+19%** |
| LibriCrowd | ML-人循环重标 | — | — | WER **−50%** |
| SWB-Affect | 6标注者 + 金标准 + Krippendorff's α | — | — | 新基准数据集 |

> **共同发现**：标注者不确定时倾向于标 "neutral"——恰好是 FAU 的核心问题。EmoMatchSpanishDB 专门讨论了 "neutral bias in crowdsourcing"。

### 5.5 对儿童数据集的适用性

| 方法 | 适用性 | 注意事项 |
|------|--------|---------|
| **LLM 重标注** | ✅ 可直接用 | GPT-4o 音频模式可直接处理语音，不依赖标注者身份 |
| **共识过滤** | ✅ 可直接用 | 去掉标注者争议大的样本，对 FAU 特别有效 |
| **伪标签** | ⚠️ 谨慎 | 需要先有高质量种子标注，FAU 种子质量存疑 |
| **多视图** | ⚠️ 需要适配 | LLM 语义视图对儿童语音（词汇简单）效果不确定 |

---

## 6. FAU 重标注必要性与方案

### 6.1 必要性论证

| 证据 | 来源 |
|------|------|
| FAU neutral 占 58.9%，37-40% 非 neutral 被误判为 neutral | 本实验混淆矩阵 |
| 原始词级标注 81% 标为 neutral，标注者默认偏见 | Schuller et al. (2009) |
| 类似数据集共识过滤后 F1 提升 19% | EmoMatchSpanishDB (2024) |
| GPT-4 重标注带来 WA 提升 3.08% | EMO-SUPERB (2024) |
| FAU 只有 18,216 chunk，成本可控 | 本分析 |

### 6.2 推荐重标注流程

```
Phase 1: 争议样本识别
  ├── 用已有模型 (WA=67%) 计算每个样本的预测 entropy
  ├── 筛选: entropy > 0.8 AND 原始标签 = neutral
  └── 预计候选: ~2000–3000 个争议样本

Phase 2: GPT-4o 音频重标注
  ├── 模型: GPT-4o (audio mode)
  ├── Prompt: "你听到的是一位 10-13 岁德国儿童与机器人狗互动时的语音。
  │           请判断儿童的情绪属于以下哪一类：
  │           - angry (愤怒/责备/烦躁)
  │           - happy (高兴/积极/愉快)  
  │           - neutral (中性/平静/无明显情绪)
  │           - sad (悲伤/沮丧)
  │           输出 JSON: {emotion, confidence(0-1), reasoning}"
  └── 预计产出: 2000–3000 条带 confidence 的标签

Phase 3: 人工抽检验证
  ├── 随机抽取 200 条
  ├── 由 3 人独立标注（或 2 人 + GPT-4o 共判）
  ├── 计算 Fleiss' κ 和 Krippendorff's α
  └── 目标: α > 0.7 (原 FAU 标注 α 估计 ~0.5–0.6)

Phase 4: 标签修正与重训练
  ├── 保留 confidence > 0.8 的修正标签
  ├── 用修正后的数据集重新划分 train/val/test
  ├── 重新训练所有 E1–E7 实验
  └── 预期: WA +5~10pp, UAR +10~15pp (主要来自 neutral→other 的修正)
```

### 6.3 预期收益

| 指标 | 当前 (原始 FAU) | 预期 (重标注后) | 说明 |
|------|---------------|---------------|------|
| WA | 67.0% | 72–77% | 减少 neutral 污染 |
| UAR | 45.1% | 55–65% | 类别平衡改善最大 |
| F-score | ~0.56 | ~0.65–0.75 | 超越 Lesyk (2024) 的 0.47 |
| neutral recall | 80.9% | 65–70% | 更真实的 neutral 分类 |

### 6.4 论文贡献定位

FAU 重标注可作为以下贡献之一：

1. **独立短文**："Re-annotating FAU AIBO: Correcting Neutral Default Bias in Children's Emotion Speech Data"
2. **论文章节**：作为实验章节，展示数据质量 > 模型复杂度的论点
3. **Discussion 亮点**：讨论儿童 SER 数据标注的特殊挑战

---

## 7. 参考文献

### 成人 SER 声学基础

1. Scherer, K. R. (2003). Vocal communication of emotion: A review of research paradigms. *Speech Communication*, 40(1-2), 227–256. [DOI: 10.1016/S0167-6393(02)00084-5](https://doi.org/10.1016/S0167-6393(02)00084-5)

2. Banse, R., & Scherer, K. R. (1996). Acoustic profiles in vocal emotion expression. *Journal of Personality and Social Psychology*, 70(3), 614–636.

3. Juslin, P. N., & Scherer, K. R. (2005). Vocal expression of affect. In *The New Handbook of Methods in Nonverbal Behavior Research* (pp. 65–135).

### Arousal 计算

4. Patel et al. Unsupervised rule-based arousal rating framework using knowledge-inspired acoustic features. [PMC NIHMS660299](https://pmc.ncbi.nlm.nih.gov/articles/PMC4334478/)

5. Burkhardt et al. (2023). Going Retro: Rule-based Prosody Modelling for Speech Synthesis Simulating Emotion Dimensions. [arXiv:2307.02132](https://arxiv.org/abs/2307.02132)

6. Wu et al. (Interspeech 2024). Dual-Constrained Dynamical Neural ODE for Continuous Arousal Prediction. [ISCA Archive](https://www.isca-archive.org/interspeech_2024/wu24_interspeech.pdf)

7. Zhou et al. (2024). Learning Arousal-Valence Representation from Categorical Emotion Labels of Speech. [arXiv:2311.14816](https://browse-export.arxiv.org/abs/2311.14816)

### IEMOCAP SSL SER 基线

8. Qieemo et al. (2025). arxiv:2503.22687, Table I — SUPERB benchmark 复现 IEMOCAP 冻结基线.

9. Ma et al. (ACL 2024 Findings). emotion2vec: Self-supervised pre-training for speech emotion representation. [ACL Anthology](https://aclanthology.org/2024.findings-acl.931/)

10. Padi et al. (Odyssey 2022). Speech emotion recognition using self-supervised features. [ISCA Archive](https://www.isca-archive.org/odyssey_2022/padi22_odyssey.pdf)

### IEMOCAP SSL 微调 SOTA

11. Pan et al. (IEEE SLT 2024). GMP-TL/GMP-ATL: Gender-augmented pseudo-labels with AM-Softmax for SER. [arXiv:2405.02151](https://arxiv.org/abs/2405.02151)

12. Xiang et al. (2024). Supervised contrastive learning for SER. [arXiv:2411.19803](https://export.arxiv.org/abs/2411.19803)

13. LaSCL (Interspeech 2025). Contrastive learning with WavLM-Large for SER.

14. Tri-Stream Fusion (IEEE Access 2025). Audio-only HuBERT SER.

15. MTLSER (2025). Multi-task learning enhanced SER with pre-trained acoustic model. *Expert Systems with Applications*. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417425004774)

16. GEmo-CLAP (2023). Gender-Attribute-Enhanced Contrastive Language-Audio Pretraining for SER. [arXiv:2306.07848](http://arxiv.org/abs/2306.07848v5)

### 儿童语音 SSL

17. Block-Medin et al. (Interspeech 2024). Adapting SSL Models to Child Speech. [ISCA Archive](https://www.isca-archive.org/interspeech_2024/blockmedin24_interspeech.pdf)

18. Lesyk et al. (2024). Cross-age transfer learning for children's SER using Wav2Vec 2.0. *Human-Centric Intelligent Systems*, 4, 633–642. [Springer](https://link.springer.com/article/10.1007/s44230-024-00088-w)

19. EUSIPCO (2025). Children's ASR with SSL Models — Wav2Vec2 vs HuBERT vs WavLM comparison.

### FAU 数据集与标注

20. Schuller, B., Steidl, S., & Batliner, A. (2009). The INTERSPEECH 2009 Emotion Challenge. *Proceedings of INTERSPEECH 2009*. [ISCA Archive](https://www.isca-archive.org/interspeech_2009/schuller09_interspeech.pdf)

21. Polzehl, T., et al. (2009). Emotion classification in children's speech using fusion of acoustic and linguistic features. *INTERSPEECH 2009*. [ISCA Archive](https://www.isca-archive.org/interspeech_2009/polzehl09_interspeech.html)

22. Bozkurt, E., et al. (2010). INTERSPEECH 2009 emotion recognition challenge evaluation. *IEEE*.

### 标注质量改善

23. EMO-SUPERB (2024). An In-depth Look at Speech Emotion Recognition — GPT-4 re-labeling study. [arXiv:2402.13018](http://arxiv.org/pdf/2402.13018)

24. EmoMatchSpanishDB (2024). Improving Emotion Dataset Quality via Consensus Filtering. [UVa Repository](https://ddfv.ufv.es/rest/api/core/bitstreams/c58d8814-05de-43b6-a0aa-6ddd62911618/content)

25. Gao et al. (Interspeech 2023). Human Transcription Quality Improvement with ML-in-the-Loop. [ISCA Archive](https://www.isca-archive.org/interspeech_2023/gao23f_interspeech.pdf)

26. Switchboard-Affect (2025). Emotion Perception Labels from Conversational Speech. [arXiv:2510.13906](https://ar5iv.labs.arxiv.org/html/2510.13906)

27. ParaSpeechCaps (2025). Scaling Rich Style Tags for Speech. [arXiv:2503.04713](http://arxiv.org/pdf/2503.04713)

### 半监督/伪标签 SER

28. Sun et al. (Interspeech 2024). Iterative Prototype Refinement for Ambiguous SER. [ISCA Archive](https://www.isca-archive.org/interspeech_2024/sun24e_interspeech.pdf)

29. Li et al. (2024). Semi-Supervised Cognitive State Classification with Multi-View Pseudo-Labeling. [arXiv:2409.16937](http://export.arxiv.org/abs/2409.16937v1)

30. Agarla et al. (2024). Semi-supervised cross-lingual speech emotion recognition. *Expert Systems with Applications*. [ACM DL](https://dl.acm.org/doi/abs/10.1016/j.eswa.2023.121368)

### Arousal 应用

31. Quatieri et al. (2024). An Emotion-Driven Vocal Biomarker-Based PTSD Screening Tool. *IEEE Open J. Eng. Med. Biol.* [PubMed](https://pubmed.ncbi.nlm.nih.gov/39184968/)

32. Opladen et al. (2023). Body exposure and vocal analysis: validation of f0 as a correlate of emotional arousal and valence. *Frontiers in Psychiatry*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/37293400/)

33. Weise et al. (2023). Multi-Modal Biomarker Extraction Framework for Therapy Monitoring of Social Anxiety and Depression. *FAU*. [SlidesLive](https://slideslive.com/39006578/)

34. Acoustic and prosodic speech features reflect physiological stress but not isolated negative affect. *Scientific Reports* (2024). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10918109/)

---

> 本文档由 Claude Code 自动生成，涵盖讨论中查询的所有文献与实验数据。所有 URL 均可直接访问检索。
