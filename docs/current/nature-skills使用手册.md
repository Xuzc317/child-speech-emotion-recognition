# Nature-Skills 完整使用手册

> **适用于**：分布驱动儿童语音情绪识别（SER）论文  
> **安装路径**：`C:/Users/59892/ai-skills/nature-skills/skills/`  
> **Agent 入口**：`C:/Users/59892/.claude/agents/nature-*.md`  
> **协议版本**：`ac_suite_2026-05`  
> **生成日期**：2026-06-10

---

## 目录

1. [技能全景图](#1-技能全景图)
2. [工作流一：重新定义论文论点](#2-工作流一重新定义论文论点)
3. [工作流二：论文初稿（LaTeX/Word）](#3-工作流二论文初稿latexword)
4. [工作流三：真实文献引用](#4-工作流三真实文献引用)
5. [工作流四：论点-引用支撑矩阵](#5-工作流四论点-引用支撑矩阵)
6. [工作流五：科研图表绘制](#6-工作流五科研图表绘制)
7. [完整投稿流水线（端到端）](#7-完整投稿流水线端到端)
8. [通用规则速查](#8-通用规则速查)
9. [当前项目文件映射](#9-当前项目文件映射)

---

## 1. 技能全景图

| 技能 | 版本 | 核心功能 | 你用它做什么 |
|------|------|---------|-------------|
| `nature-reviewer` | v0.1.0 | 模拟 Nature 审稿，输出 3+1 报告 | **投稿前自审 → 找到论点弱项** |
| `nature-writing` | v1.0.0 | 从零起草论文章节 | **根据实验数据重建论文叙述** |
| `nature-polishing` | v6.1.0 | 润色已有文本 + LaTeX 排版修复 | **打磨英文定稿** |
| `nature-academic-search` | v2.0.0 | 多源文献检索 | **搜索支撑论点的真实文献** |
| `nature-citation` | v2.0.0 | 文本自动配引用 + 导出参考文献 | **每句话配真实引用** |
| `nature-figure` | v2.0.0 | 论文配图（Python/R） | **重画所有图** |
| `nature-paper2ppt` | v2.0.0 | 论文转 PPT | **答辩/组会汇报** |

**核心原则（所有 skill 通用）**：不能凭记忆操作，必须从磁盘加载 SKILL.md → manifest.yaml → 对应 fragment/reference 文件。

---

## 2. 工作流一：重新定义论文论点

### 2.1 步骤链

```
你的实验数据 → nature-reviewer（找弱点）→ nature-writing（重建论点）→ nature-reviewer（重新审查）
```

### 2.2 第一步：投稿前审稿（找到论点弱点）

**触发语句**：
```
用 nature-reviewer 对 paper_draft/main.tex 做投稿前 Nature 审稿评估
```

**这个 skill 会输出什么**：

```
Review setup
  - Input scope: 论文全文/部分
  - Assessment boundary: 可评估范围
  - Shared manuscript claim summary: 提取的主论点
  - Visible evidence base: 现有证据基
  - Missing materials affecting confidence: 缺失材料

Reviewer 1 / 2 / 3（三人角度不同）
  - Overall assessment
  - Who would be interested, and why
  - Major strengths
  - Major concerns
  - Technical failings
  - Assessment against 5 个 Nature 标准：
    originality / scientific importance / interdisciplinary readership
    / technical soundness / readability for nonspecialists
  - Recommendation posture

Cross-review synthesis
  - Consensus strengths
  - Consensus technical risks
  - Most important issues to resolve
```

**红线（绝对不能做的事）**：
- 不编造审稿人身份
- 不编造实验、引用、数据
- 不把审稿评估变成 rebuttal 写作
- 不说"这论文肯定能中 Nature"
- 不忽略技术缺陷

**你应该拿到什么**：一份诚实的弱点清单（共识技术风险），这是下面所有工作的基础。

### 2.3 第二步：用 nature-writing 重建论点

拿到上述弱点清单后：

**触发语句**：
```
用 nature-writing，paper_type=research，section=abstract+intro+discussion，
language=zh-to-en，journal=nat-comms

根据下面的实验发现重建论文叙事：

实验发现：
1. WavLM 冻结 backbone + 12层可学习融合 + Prosody Pooling + SEMLP，
   总可训参数 ~704K，C-BESD 儿童演绎式达到 95.10% WA（3-seed）
2. Prosody Pooling 在儿童语音上 +2.24pp，在成人 IEMOCAP 上 -2.12pp，
   方向反转 → 韵律先验的儿童特异性
3. FD（Fréchet Distance）与准确率严格负相关：
   儿童同语料 FD=0 → WA=92.78%；年龄偏移 FD=7.20 → WA=58.67%；
   风格偏移 FD=8.50 → WA=19.56%
4. 成人增强参数严重损害儿童模型（-28pp），成人增强经验不可迁移
```

**nature-writing 的 8 步工作流**（从 static/core/workflow.md 加载）：

1. **Intake**：提取所有 claim、evidence、boundary
2. **Argument chain**：构建 paper argument → `field need → bottleneck → proposed move → evidence → implication → boundary`
3. **Paper-type classification**：research
4. **Skeleton**：按 chapter 规划每节要说什么
5. **Section drafting order**：results → intro → discussion → conclusion → abstract
6. **Per-section draft**：用对应的 section fragment + reference
7. **Claim-evidence audit**：每个 claim 配对 evidence
8. **Style review**：Nature 风格自查

**关键规则**：
- 如果实验数据不支持某个 claim → 标记为 `Assumptions or missing inputs:`，**不编造**
- 缺失的 evidence 或 boundary 必须明确列出
- 写作顺序 ≠ 阅读顺序：先写 results，最后写 abstract

### 2.4 用 writing-strategy.md 的核心思想检验每个论点

| 检查项 | 问题 | 例子 |
|--------|------|------|
| Claim | 这个陈述有明确的主语+谓语吗？ | "分布偏移是主因" → 改成 "FD与WA呈严格单调递减关系" |
| Evidence | 有定量证据吗？ | "韵律有帮助" → 改成 "+2.24pp, 95% CI [1.41, 3.64]" |
| Boundary | 边界条件写了吗？ | "Prosody有帮助" → 改成 "Prosody Pooling 对儿童演绎式有益，对成人可能有害" |

**你的论文的核心 argument chain 应该是**：
```
儿童语音情绪识别缺乏对统计分布偏移的系统理解
→ 现有方法忽略儿童-成人声学分布差异
→ 我们提出分布驱动框架（FD测量 + 韵律先验 + 层融合）
→ 实验证明 FD→WA 严格单调 + 韵律先验儿童特异性 + 成人增强不可迁移
→ 结论：儿童SER需要以分布偏移为核心的工程范式
→ 边界：当前仅在4类、演绎式+自发性、3语料库上验证
```

---

## 3. 工作流二：论文初稿（LaTeX/Word）

### 3.1 分步骤执行（按 writing 内部顺序）

**Step 1 — 先写 Results**：
```
用 nature-writing，section=experiments，paper_type=research

根据权威数据手册 docs/权威数据手册.md 的 6 组实验数据，
起草 4_Experiments_and_Results.tex 的 Results 部分。
每段用 "To test [question], we [action]" 开头。
```

**Step 2 — 写 Introduction**：
```
用 nature-writing，section=intro，journal=nat-comms

按漏斗结构写 Introduction：
1. 儿童语音情绪识别的现实意义
2. 现有方法瓶颈：分布偏移被忽视
3. 前人工作公平综述
4. 能力缺口：分布驱动框架缺失
5. 本文作为直接回应
```

**Step 3 — 写 Discussion**（从发现到意义）：
```
用 nature-writing，section=discussion

Discussion 应：
1. 核心进展：发现FD-WA严格负相关 + 韵律先验儿童特异性
2. 为什么证据支持：多语料库、多种子、多池化交叉验证
3. 如何改变范式：从"更大模型"到"更懂分布"
4. 与先前研究的关系
5. 边界条件
```

**Step 4 — 写 Abstract（最后写）**：
```
用 nature-writing，section=abstract

遵循 6 段式：
1. 领域问题
2. 当前方案不全
3. 本文提出什么
4. 最强定量结果
5. 机制或后果
6. 受限的含义
```

### 3.2 英文润色

```
用 nature-polishing，language=zh-to-en，section=abstract+intro+discussion

对 paper_draft/4_Experiments_and_Results.tex 全文润色，
按 Claim-Evidence-Boundary 原则修复逻辑缺陷后再修语言
```

**润色时应用的优先级**：
1. Paper-type playbook → 架构 + 写作顺序
2. Section-specific job → 每节的修辞目标
3. Journal-specific framing → Nature 子刊的读者期望
4. Language-specific rules → 中文作者常见问题修复
5. Core stance and ethics → 不夸大、不编造

### 3.3 LaTeX 排版修复（独立通道）

如果排版有问题：

```
用 nature-polishing 的 LaTeX layout 通道修复 paper_draft/main.tex：
[描述排版问题：松散页面、浮动体太大、跨页分割等]
```

**规则**：必须 `编译 → 目视渲染 → 修改`，不能只看 `.tex` 源码。使用 `[H]`、`\clearpage`、`placeins` 等工具。

### 3.4 生成 Word 终稿

LaTeX 定稿后，通过 `scripts/generate_docx.py` 生成 Word，再用 polishing 检查一遍。

---

## 4. 工作流三：真实文献引用

### 4.1 搜索文献

**触发语句**：
```
用 nature-academic-search 的 multi-source-search 工作流，
搜索以下领域的论文：
1. children speech emotion recognition WavLM self-supervised learning
2. Fréchet distance distribution shift speech emotion
3. prosody guided attention pooling speech processing
4. distribution-driven speech emotion recognition cross-corpus generalization
```

**支持的源**：PubMed、CrossRef、arXiv、Scopus、ScienceDirect（通过 MCP 工具）

**路由协议**：
```
加载 manifest.yaml → 检测 workflow（multi-source-search）→ 
加载 references/workflows/multi-source-search.md → 
按 T1→T2→T3 源优先级执行 → 报告结果
```

### 4.2 论文文本自动配引用

**触发语句**：
```
用 nature-citation，scope=Nature系列+CNS及子刊，
自动为下述文本段匹配真实支撑文献：

[粘贴需要配引用的段落到这里]
```

**工作流程**（`static/core/workflow.md` 的 7 步）：
1. **Segment**：将文本切分为可引用单元
2. **Parse**：提取每个单元的 claim 类型（支持/借用/对比/复用）
3. **Search**：按 Nature/CNS 范围搜索
4. **Evaluate**：保守评估支撑度（不只看标题匹配）
5. **Export**：输出一种引用管理器格式
6. **Review artifacts**：生成审查报告
7. **Report**：HTML 浏览路径 + 段落到引用的对应表

**范围选项**：
| 用户输入 | --scope 参数 | 实际搜索范围 |
|----------|-------------|------------|
| Nature系列 | nature | Nature 旗舰刊 + 全部子刊 |
| CNS | cns | Nature + Science + Cell 旗舰 |
| CNS及子刊 | cns-sub | N/S/C 旗舰 + 全部子刊 |
| 不限 | all | 全部学术期刊 |

**红线**：
- 不能仅因标题相关就当作支撑
- 不能引用未读摘要的元数据只候选
- 不能不验证摘要就标为支撑
- 不能编造书目字段

---

## 5. 工作流四：论点-引用支撑矩阵

### 5.1 建立结构化对应

将你的论文核心论点与引用文献建立映射：

```
用 nature-citation，scope=CNS及子刊，
为以下论点-证据链逐条配引用：

论点1：儿童语音的声学分布与成人显著不同
  → 需要文献支撑：儿童语音声学特性、儿童SER数据集

论点2：SSL预训练模型（WavLM）可有效编码儿童语音
  → 需要文献支撑：WavLM架构、SSL在语音任务上的表现

论点3：Fréchet Distance可测量语料库间分布偏移
  → 需要文献支撑：FD在语音/域偏移中的应用

论点4：韵律先验对儿童语音情绪识别有特异性增益
  → 需要文献支撑：韵律特征在情绪识别中的作用、年龄差异

论点5：成人增强经验不可迁移至儿童数据
  → 需要文献支撑：域适应失败案例、儿童-成人数据差异
```

**输出格式**（nature-citation 生成的）：

```markdown
| 段落 | Claim类型 | 搜索查询 | 候选文献 | 支撑度 | 引用状态 |
|------|----------|---------|---------|--------|---------|
| P1 | support | children speech acoustic properties developmental | [citation] | HIGH | ready |
| P2 | borrow | WavLM self-supervised speech representation | [citation] | HIGH | ready |
| P3 | borrow | Fréchet distance domain shift measurement | [citation] | MEDIUM | need verify |
```

### 5.2 生成参考文献清单

```
用 nature-academic-search 的 reference-mgmt 工作流，
将上述搜索到的所有参考文献导出为 BibTeX 格式，
整合到 paper_draft/references.bib
```

---

## 6. 工作流五：科研图表绘制

### 6.1 nature-figure 的 5 步路由协议

```
1. 加载 manifest.yaml + static/core/contract.md + static/core/stance.md
2. 【阻断门】用户必须明确选择 Python 或 R — 必须停下来问
3. 加载对应的 backend fragment（python.md 或 r.md）
4. 按 contract → stance → backend 顺序画图
5. 按需加载 references（不默认加载）
```

### 6.2 你必须回答的第一个问题

每当你说"帮我画个图"，我必须问：

> **Python 还是 R？**

你的项目是 Python（matplotlib/seaborn），选 Python。

### 6.3 Figure Contract（画图前必须做的事）

**每一张图都必须先写 contract**：

```text
Core conclusion: [一句话，带动词]
Figure archetype: [quantitative grid / schematic-led composite / image plate+quant / asymmetric mixed]
Target journal/output: [INTERSPEECH/ICASSP]
Backend: Python
Final size: [宽度inch, 高度inch]
Panel map:
  a: [每个面板的独特问题]
  b:
  c:
Evidence hierarchy:
  hero evidence: [最核心的证据]
  validation evidence: [支撑证据]
  controls/robustness: [稳健性检查]
Statistics needed: [误差条、置信区间、显著性检验]
Source data needed: [哪个 JSON/CSV 文件]
Image-integrity notes: [图像处理文档]
Reviewer risk: [审稿人最可能质疑的点]
```

### 6.4 你项目每一张图的 contract 模板

#### fig03 — FD vs Accuracy 关系

```
Core conclusion: "Higher FD produces strictly lower classification accuracy across three
  distribution-shift dimensions (enhancement, age, style)"
Figure archetype: quantitative grid
Target journal: INTERSPEECH 2027
Backend: Python
Final size: 7×3.5 in
Panel map:
  a: scatter/line — FD (x) vs WA (y) with four conditions labeled
  b: bar — self-attention vs prosody pooling at each FD level (optional)
Evidence hierarchy:
  hero evidence: 4 FD-WA pairs forming monotonic descent
  validation evidence: consistent across 3-seed error bars
  controls: C-BESD same-corpus FD=0 baseline
Statistics needed: error bars from 3-seed std
Source data needed: results/canonical_fd_pairs.json + experiment_results.csv
Reviewer risk: n=4 points may look insufficient; add discussion of why these 4 conditions were chosen
```

#### fig07/08 — 混淆矩阵

```
Core conclusion: "Both pooling methods show similar confusion patterns dominated by
  Angry↔Happy (high-arousal) confusions on C-BESD children's speech"
Figure archetype: quantitative grid (symmetric heatmap pair)
Backend: Python
Final size: 7×3.5 in (side by side)
Panel map:
  a: Exp1 Self-Attention CM (N=540)
  b: Exp2 Prosody Guided CM (N=540)
Evidence hierarchy:
  hero: diagonal intensity showing correct classifications
  validation: row-sum = 540 per CM
Statistics needed: per-class recall/precision annotations
Source data: paper_draft/figures/fig07_confusion_exp1_selfattn.json + fig08_*.json
```

#### fig04 — XAI 三联图

```
Core conclusion: "Prosody Pooling shifts attention from pure acoustic correlation
  toward prosody-guided temporal regions, reducing redundant APC by Δ=−0.045"
Figure archetype: asymmetric mixed-modality
Backend: Python
Final size: 7×5 in
Panel map:
  a: waveform + F0 overlay with attention heatmap (hero)
  b: APC scatter (Self-Attn vs Prosody per sample, N=540)
  c: delta-APC histogram (showing negative shift)
Evidence hierarchy:
  hero: panel a shows precise temporal alignment
  validation: panel b+c show population-level effect
Statistics needed: APC_delta mean ± std, paired statistical test
Source data: results/xai_raw_data.npz + results/logs/apc_full_test.json
```

### 6.5 图的原型选择指南

| 论文中的图 | 推荐原型 | 原因 |
|-----------|---------|------|
| fig03 FD-Accuracy | `quantitative grid` | 数值比较为主，FD-WA 单调性 |
| fig01 主结果矩阵 | `quantitative grid` | 多实验 WA/UAR 并列 |
| fig04 XAI 三联图 | `asymmetric mixed-modality` | 波形图 + 散点图 + 直方图混合 |
| fig07/08 CM | `quantitative grid` | 两个热图并排 |
| fig00 架构图 | `schematic-led composite` | 架构总览主导，数据面板为辅 |
| fig06 层权重 | `quantitative grid` | 12 层权重柱状图 |

### 6.6 配色规则（来自 `references/design-theory.md`）

- 一个中性色族 + 一个信号色族 + 一个强调色
- 同一条件/方法在所有面板中保持同一颜色
- Hero panel 用最大的视觉面积
- 面板面积 ≠ 均分 → 重要面板更大

### 6.7 交付前的 QA（来自 `references/qa-contract.md`）

```
- 样本量是否在图例或源数据中可见？
- 误差条和统计检验是否定义？
- 可比较的轴是否对齐？
- 代表性图像是否定量化且可追溯原始文件？
- 图像调整是否全局且记录在案？
- 相同的结论可以用更少的面板得出吗？
```

---

## 7. 完整投稿流水线（端到端）

### 推荐执行顺序

```
Phase 1: 诊断（1-2 天）
  ├── Step 1: nature-reviewer → 审稿评估，找出论点弱项
  └── Step 2: nature-academic-search → 搜索缺失文献

Phase 2: 重建（3-5 天）
  ├── Step 3: nature-writing → 根据实验数据重写 LaTeX 各章节
  ├── Step 4: nature-citation → 每句配真实引用
  └── Step 5: nature-polishing → 英文润色

Phase 3: 图表（2-3 天）
  ├── Step 6: nature-figure → 逐张画图（每张都要先写 contract）
  └── Step 7: nature-polishing LaTeX layout → 排版修复

Phase 4: 审查（1-2 天）
  ├── Step 8: nature-reviewer → 最终投稿前审稿
  └── Step 9: 修复审稿指出的问题

Phase 5: 打包
  └── Step 10: python scripts/build_submission_bundle.py
```

### 每个 Step 的精确触发语句

```bash
# Step 1
"用 nature-reviewer 对 paper_draft/ 全文做 Nature 标准投稿前审稿"

# Step 2
"用 nature-academic-search 的 multi-source-search 工作流，
搜索 [你的搜索词]"

# Step 3（分章节）
"用 nature-writing，section=experiments，language=zh-to-en，
根据 results/logs/ 和 docs/权威数据手册.md 重写实验章节"

# Step 4
"用 nature-citation，scope=CNS及子刊，
为 4_Experiments_and_Results.tex 逐段配真实支撑文献"

# Step 5
"用 nature-polishing，language=en，section=all，
对 paper_draft/ 全部 .tex 文件做 Nature 风格润色"

# Step 6（每张图）
"用 nature-figure，backend=Python，
根据 results/canonical_fd_pairs.json 重绘 fig03 FD-Accuracy"

# Step 7
"用 nature-polishing 的 LaTeX layout 通道修复排版问题"

# Step 8
"用 nature-reviewer 对最终投稿稿做预审"
```

---

## 8. 通用规则速查

### 8.1 所有 skill 必须遵守

| 规则 | 说明 |
|------|------|
| **加载优先** | 每次调用都必须先 Read manifest.yaml，不能凭记忆 |
| **轴值声明** | 检测到的 axis 值必须告诉用户确认 |
| **按需引用** | references 是深度手册，只有触发条件满足时才打开 |
| **标记边界** | 缺信息就标记 `Assumptions or missing inputs:`，不编造 |
| **证据对齐** | 每个 claim 必须有 evidence 和 boundary |

### 8.2 polishing vs writing 的区别

| 维度 | polishing | writing |
|------|----------|---------|
| 输入 | 已有完整文本 | 观点/数据/笔记/中文草稿 |
| 操作 | 改进已有文本 | 从零起草 |
| 核心步骤 | 5 步路由（无 workflow.md） | 5 步路由 + 8 步 draft workflow |
| 典型场景 | "我写好了，帮我润色" | "我有实验数据，帮我写论文" |

### 8.3 你的项目的默认轴值

```
paper_type: research
journal: generic (INTERSPEECH/ICASSP 对应此档)
language: zh-to-en (中文笔记/数据 → 英文论文)
backend: Python (matplotlib/seaborn)
```

### 8.4 关键的写作法则（来自 references）

**Hourglass 结构**：`broad → narrow → broad`
- Introduction：打开领域 → 缩窄到缺口 → 声明本研究
- Discussion：从具体发现 → 拓宽到含义与限制

**Claim-Evidence-Boundary 三元组**：每个科学陈述必须有这三部分

**写作顺序 ≠ 阅读顺序**：
```
写: results → intro + conclusion → title → discussion → methods → abstract
读: title → abstract → intro → results → discussion → methods → conclusion
```

**Introduction 的 4 个问题**：
1. 已知什么？
2. 仍有什么未解决？
3. 本文具体问什么？
4. 如何回答？

**Results 的原则**：`To test [question], we [action].` → 然后报告结果

**Discussion 的 6 步**：核心进展 → 为何证据支持 → 如何改变范式 → 与先前关系 → 什么限制 → 什么未来可做

**严禁词汇（会触发 overclaim）**：
- `prove` → 改成 `demonstrate` / `show`
- `unprecedented` → 删掉或用具体比较
- `best` → 改成 `state-of-the-art on [specific benchmark]`
- unqualified `first` → 改成 `to our knowledge, the first to [specific scope]`

---

## 9. 当前项目文件映射

| 你要改的 | 路径 | 用哪个 skill |
|---------|------|------------|
| 摘要 | `paper_draft/0_Abstract.tex` | writing → polishing |
| 引言 | `paper_draft/1_Introduction.tex` | writing |
| 相关工作 | `paper_draft/2_Related_Work.tex` | writing + citation |
| 方法 | `paper_draft/3_Methodology.tex` | writing |
| 实验与结果 | `paper_draft/4_Experiments_and_Results.tex` | writing + citation |
| 分析与讨论 | `paper_draft/5_Analysis_and_Discussion.tex` | writing |
| 结论 | `paper_draft/6_Conclusion.tex` | writing → polishing |
| 主文档 | `paper_draft/main.tex` | polishing (layout) |
| 参考文献 | `paper_draft/references.bib` | academic-search + citation |
| 图 fig01–06 | `paper_draft/figures/fig0*.png` | figure |
| 混淆矩阵 fig07–08 | `paper_draft/figures/fig07–08_*` | figure |
| 附录 CM figA1–A4 | `paper_draft/figures/figA*` | figure |
| 英文 Word | `paper_draft/Full_Draft.docx` | polishing（LaTeX 定稿后生成） |
| 中文 Word | `paper_draft/Full_Draft_CN.docx` | polishing |
| 实验数据 | `results/logs/exp*.json` | writing（只读输入） |
| 权威数据 | `docs/权威数据手册.md` | writing + reviewer（只读输入） |
| FD 矩阵 | `results/canonical_fd_pairs.json` | figure（数据源） |
| XAI 数据 | `results/xai_raw_data.npz` | figure（数据源） |
| 层权重 | `results/layer_weights.json` | figure（数据源） |

---

> *本文档基于 nature-skills 的 SKILL.md + manifest.yaml + references 文件撰写，所有引用内容均来自磁盘加载的实际文件，非凭记忆生成。*
