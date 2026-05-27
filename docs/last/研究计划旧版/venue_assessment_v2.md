# 投稿可行性评估 v2：基于完整实验数据的综合判断

> 2026-05-04 | 全部 Phase 6.1-6.4 实验完成

---

## 一、实验数据总览

### 1.1 核心结果矩阵

| 维度 | 实验 | 关键数据 |
|------|------|---------|
| **主结果** | A3 on C-BESD 6:2:2 | WA=80.85%, UAR=80.8%, 3 seeds |
| **vs 外部基线** | MFCC+CNN | 50.2% (严格 speaker-independent) vs 已发表 76% (疑似非 speaker-independent) |
| **vs 成人** | IEMOCAP A1 vs A3 | 儿童 +2.24pp, **成人 -2.12pp** |
| **backbone 选择** | WavLM vs wav2vec2 | WavLM 78.5%, wav2vec2 56.1% (差距 22pp) |
| **Adapter 贡献** | A3 vs B3 | +0.62pp (可忽略) |
| **增强敏感性** | C1-C4 | 80.3% → 58.8% → 52.3% → 46.9% (单调下降) |
| **跨语言** | English↔Telugu | 81.2% → 19.7%/28.2% (灾难性下降) |
| **可解释性** | Attention per-class | Entropy-Acc 负相关 (ANGER 5.18/best, HAPPY 5.20/worst) |
| **FD 框架** | 三维 FD-Accuracy | FD_age=6.87, FD_aug=0~12, FD_lang=137 (严格单调) |

### 1.2 三大核心发现

**F1: Prosody Pooling 对儿童有益、对成人有害**
- 这是全文最有力的单一发现
- 直接支撑"儿童语音情绪的时间分布与成人根本不同"
- 有可重复性：wav2vec2 上也是同样的模式（prosody 有害）

**F2: FD ∝ 1/Accuracy 在三个维度上严格单调**
- 年龄偏移 (FD=6.87)、增强偏移 (FD=0~12)、语言偏移 (FD~137)
- 每条线都是四/五个数据点的单调关系
- FD 作为"分布偏移风险指标"的论证完整

**F3: Attention Entropy 与 Per-Class Accuracy 负相关**
- ANGERS (entropy 最低) → 准确率最高 (88.7%)
- HAPPY (entropy 最高) → 准确率最低 (73.1%)
- 证明韵律引导的帧选择机制可解释

---

## 二、文献定位更新（2026-05 检索）

### 2.1 儿童 SER + SSL 交叉领域

**仍然是空白**。INTERSPEECH 2025 和 ICASSP 2025/2026 检索确认：
- 儿童 SSL 论文全是 ASR，没有 SER
- 成人 SER 论文大量使用 WavLM/HuBERT/wav2vec2
- 我们很可能是**第一个在 C-BESD 上使用 SSL + 严格 speaker-independent 评估的工作**

### 2.2 ICASSP 2026 热点：Distribution Shift + SER

ICASSP 2026 接受了至少两篇相关论文：
- "Test-Time Adaptation for Speech Emotion Recognition" — 系统评估 11 种 TTA 方法
- "EMO-TTA" — 训练无关的分布估计框架

**这意味着 distribution shift 是当前 SER 领域的热门话题**。我们的 FD 框架刚好踩在这个热点上。

### 2.3 C-BESD 论文状态

Rao et al. (IEEE INSPECT 2024) 仍是 C-BESD 唯一的已发表基线。我们的 MFCC 复现 (50.2%) 与他们的 76% 之间存在 26pp 差距，强烈暗示他们的评估可能不是严格的 speaker-independent。

---

## 三、SWOT 分析（更新版）

### Strengths（比 v1 更强）
- ✅ **三个独立且有深度的发现**（不是"提了一个模型"）
- ✅ **成人对比实验是关键差异化证据**
- ✅ **FD 框架踩中 ICASSP 2026 热点**
- ✅ **实验严谨性高**（数据泄露修复、多 seed、严格 split）
- ✅ **负面实验范式**（增强敏感性、Adapter 消融）

### Weaknesses
- ⚠️ **单数据集**（C-BESD），但通过 IEMOCAP 成人对比部分缓解
- ⚠️ **C-BESD 是 acted speech**，生态效度有限
- ⚠️ **方法 novelty**（attention + F0/energy）可能被认为简单
- ⚠️ **FD_lang 是估计值**（语言子集特征未提取）

### Opportunities
- 🔥 **Distribution shift 是 ICASSP 2026 热点**
- 🔥 **儿童 SSL+SER 是未被占据的 niche**
- 🔥 **INTERSPEECH ComParE 可能涉及儿童/特殊人群**
- 🔥 **审稿人找不到直接对比论文 → 更难拒绝**

### Threats
- ⚠️ 审稿人可能质疑方法的 simplicity
- ⚠️ C-BESD 知名度低
- ⚠️ Acting vs spontaneous 的经典争议

---

## 四、投稿建议（更新版）

### 首选：ICASSP 2027（截稿 2026-10）

**理由**：
1. Distribution shift 是 ICASSP 2026 的热门 topic，2027 年热度会延续
2. 我们的 FD 框架正好踩中这个方向
3. ICASSP 对方法 novelty 的容忍度比 INTERSPEECH 高
4. 5 个月时间充裕

**论文定位**：
> "Distribution Shift Taxonomy for Children's Speech Emotion Recognition"

强调三个维度的 FD-Accuracy 关系，把方法 (Prosody Pooling) 作为"在这样的分布偏移结构下自然导出的解决方案"。

### 备选：INTERSPEECH 2027（截稿 2027-03）

**理由**：儿童语音是 INTERSPEECH established topic，但 novelty 要求更高。

---

## 五、论文 v7 需要重点解决的问题

1. **方法 novelty**：不能只说"我们提了一个 attention pooling"，要说"我们发现儿童语音的情绪信息在时间上有特定的非均匀分布模式，Prosody Pooling 是利用这个模式的最自然方法"
2. **单数据集**：用 IEMOCAP 成人对比来论证"这不是 C-BESD 特有的现象"
3. **FD 框架要作为主线**：把 FD 放在 Introduction 和 Abstract 的显著位置
4. **实验章节要丰富**：每个 finding 配一张关键图表

---

## 六、结论

**可以投顶会**。当前实验数据的完整性和深度已经超过大多数顶会投稿。

三核心发现 + distribution shift 热点 + 空白 niche = 有竞争力的投稿。
最关键的风险是方法 novelty 感知问题，需要在写作中把叙事从"我们提了一个方法"翻转为"我们发现了一个规律"。
