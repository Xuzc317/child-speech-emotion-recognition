# 投稿可行性评估：语音顶会 (INTERSPEECH / ICASSP)

> 2026-05-04 | 基于文献调研

---

## 一、核心发现：这是一个真实的研究空白

经过对 INTERSPEECH 2024/2025 和 ICASSP 2025 的系统检索：

| 交叉领域 | 顶会论文数量 | 状态 |
|---------|-----------|------|
| SSL 模型 → 成人 SER | 大量 (10+ per year) | 成熟、竞争激烈 |
| SSL 模型 → 儿童 ASR | 5-8 篇 (IS2024/2025) | 快速增长 |
| SSL 模型 → 儿童 SER | **0 篇主会** | **完全空白** |

C-BESD 论文发在 IEEE INSPECT（印度国内会议），不是顶会。**我们很可能是第一个在 C-BESD 上使用 WavLM + 严格 speaker-independent 评估的工作。**

---

## 二、顶会同类型论文标准

### INTERSPEECH 2024/2025 的 SER 论文典型配置

| 要素 | 顶会标准 | 我们当前 |
|------|---------|---------|
| SSL backbone | WavLM / HuBERT / wav2vec2 | ✅ WavLM |
| 数据集 | ≥2 个公开数据集 | ⚠️ 仅 C-BESD |
| 评价指标 | UAR 为主, WA 为辅助 | ✅ WA+UAR |
| Speaker-independent | 严格标注 split 方法 | ✅ 6:2:2 严格协议 |
| 多 seed | 3 seeds, mean±std | ✅ 3 seeds |
| 对比基线 | 与已发表方法对比 | ⚠️ 仅 1 个外部基线 (76%) |
| 消融实验 | 每个模块有消融 | ✅ A1→A3→B3 |
| 负面实验 | 非必须但加分 | ✅ 增强敏感性 C1-C4 |
| 可解释性 | 越来越被重视 | ❌ 尚未做 |

### 顶会录用论文的共性模式

1. **清晰的问题陈述**："现有方法在 X 场景下失效，因为 Y"
2. **简单有效的方法**：不是堆砌模块，而是一个核心 insight
3. **充分的消融**：证明每个设计选择都有依据
4. **多数据集验证**：IEMOCAP + MSP-Podcast + RAVDESS 等交叉验证
5. **与 SOTA 的显式对比**：引用具体数字，不是"我们的旧基线"

---

## 三、SWOT 分析

### Strengths
- **填补空白**：首个 SSL-based 儿童 SER，审稿人找不到直接对比论文
- **FD 框架**：多维度分布偏移量化是独特的理论贡献，不是单纯"提了个模型"
- **实验严谨**：数据泄露发现+修复、严格 speaker-independent、多 seed
- **诚实叙事**：不夸大 Adapter 作用，负面实验范式在 SER 领域罕见

### Weaknesses
- **单数据集**：C-BESD 是 2024 年新数据集，知名度低
- **acted speech**：C-BESD 是表演型语音，非自然场景，生态效度有限
- **无直接 SOTA 对比**：没有"我们的 80.85% vs SOTA XX%"的句子
- **模型简单**：Prosody Pooling 本质上是一个 attention 机制，reviewer 可能觉得 novelty 不够

### Opportunities
- **INTERSPEECH 2026/2027**：儿童语音是 established topic，SER 是常驻 track
- **ComParE 挑战赛**：INTERSPEECH 的 ComParE 每年换题，儿童 SER 可能成为未来的挑战主题
- **交叉领域热度**：SSL + children 在 ASR 已经热起来了，SER 是自然延伸
- **ICASSP 2027**：domain adaptation / distribution shift 是热门 topic

### Threats
- **审稿人不熟悉儿童 SER**：可能用成人 SER 的标准来评判（不公平但可能发生）
- **C-BESD 知名度低**：审稿人可能质疑数据集的可比性
- **novelty 被质疑**："attention pooling + F0/energy" 可能被认为太简单
- **缺乏自然场景验证**：acting vs spontaneous 是 SER 领域的经典争议

---

## 四、投稿策略建议

### 方案 A：INTERSPEECH 2027 (截稿 ~2027-03)

**优势**：语音领域最对口，儿童语音是 established track
**劣势**：竞争激烈，录用率 ~48%
**论文定位**："首次将 SSL 帧级特征引入儿童 SER + 发现多维度分布偏移规律"

**需要补充的工作**（5 个月）：
1. ✅ 当前实验
2. IEMOCAP 成人对比 (6.1b 进行中)
3. 多 SSL backbone 对比 (6.1c)
4. Attention 可解释性分析 (6.3)
5. 统一 FD-Accuracy 图 (6.4)
6. 至少再一个数据集验证 (FAU-AIBO 或 EmoReact)
7. 论文撰写

### 方案 B：ICASSP 2027 (截稿 ~2026-10)

**优势**：domain shift / distribution 是 ICASSP 热门，5 个月时间充裕
**劣势**：SER 不是 ICASSP 最大 track
**论文定位**："Distribution Shift Taxonomy for Children's SER: A Multi-Dimensional FD Framework"

**需要补充的工作**（同上，但更侧重 FD 理论框架）

### 方案 C：ASRU 2027 (截稿 ~2026-12)

**优势**：语音识别与理解，评估标准相对宽松
**劣势**：影响力略低于 INTERSPEECH/ICASSP
**备选方案**

### 推荐：INTERSPEECH 2027 为主，ICASSP 2027 为备选

理由：
- 儿童语音在 INTERSPEECH 有专门的 regular session
- ComParE 挑战赛可能涉及儿童/特殊人群情感
- 时间充裕（8-10 个月准备）

---

## 五、提升录用概率的关键行动

| 优先级 | 行动 | 预期效果 |
|--------|------|---------|
| ⭐⭐⭐ | 增加第二个数据集 (FAU-AIBO 或 EmoReact) | 消除"单数据集"质疑 |
| ⭐⭐⭐ | 完成可解释性分析 (attention viz + 成人对比) | 提升 scientific depth |
| ⭐⭐⭐ | 完成统一 FD 框架 | 这是我们的独特记忆点 |
| ⭐⭐ | 与其他 SSL backbone 对比 | 证明 WavLM 选择的合理性 |
| ⭐⭐ | Cross-corpus 验证 (C-BESD English → IEMOCAP) | 展示泛化能力 |
| ⭐ | 找合作者/导师 review 初稿 | 外部视角 |

---

## 六、结论

**可以投顶会，但需要补充工作。** 我们的核心优势是 (1) 真实的研究空白 (2) 实验严谨 (3) FD 理论框架独特。最大的短板是单数据集和 novelty 感知问题。

如果接下来 2-3 个月内完成可解释性分析 + FD 框架 + IEMOCAP 对比，到 2026 年底 ICASSP 截稿时，论文的完整性足够支撑一篇顶会投稿。
