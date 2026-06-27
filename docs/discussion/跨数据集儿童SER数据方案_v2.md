# 跨数据集儿童语音情绪识别 —— 数据构造方案 v2

> 创建: 2026-06-05 | 更新: 2026-06-05 (v2) | 核心变化: 不再信任原始标签，用声学特征重新定义情绪类别

---

## 0. 方案核心思想

### 为什么这个方向值得做

当前可用的儿童语音情绪数据虽然分散在不同数据集、不同语言、不同采集条件下，但它们的共同点远多于差异：
- **FAU Aibo**：德语自发型，51 人，规模最大但类别失衡——neutral 占 59%，sad 缺失
- **C-BESD**：英语+泰卢固语双语表演型，4 类均衡，每类 ~700——是目前唯一 4 类齐全的儿童情绪数据集。我们使用的是其 MY（数据集标识，非语言缩写）子集
- **ChildMandarin**：中文自发型，397 人（3-5 岁，22 省），40,913 条——规模远超 FAU 和 C-BESD 之和，平均时长 3.52s 与 4s 管线匹配

#### ChildMandarin 说话人多样性分析

397 名 3-5 岁儿童、22 省覆盖，声纹差异不可忽视：

| 声纹变异来源 | 对情绪特征的影响 | 严重程度 |
|-------------|----------------|---------|
| **年龄差异（3 vs 5 岁）** | F0 基线差 50-80 Hz——3 岁 neutral 的 F0 可能和 5 岁 angry 的 F0 接近 | 🔴 需处理 |
| **性别差异** | 男女童 F0 差异在幼儿期已存在 | 🟡 需关注 |
| **方言/口音** | 22 省→声调系统差异→F0 轮廓模式不同。粤语区升调和东北升调可能对应不同情绪 | 🟡 需在 Phase 2 验证 |
| **录音环境** | 不同手机、不同房间→SNR 差异→RMS 基准漂移 | 🟡 需在 RMS 归一化时考虑 |

**处理方案**：

1. **Per-speaker 归一化（核心策略）**：对每个说话人，将其所有样本的 F0 和 RMS 减去该说话人所有样本的中位数，再做情绪分类。这样 3 岁孩子的高 F0 基线不会被误判为 happy/angry——判断依据从"绝对 F0 值"变为"相对该说话人日常语调的偏离程度"

2. **Phase 2 验证时按年龄分组**：3 岁组、4 岁组、5 岁组分别统计每类情绪的声学区间，确认"angry F0 − neutral F0"这个**差值**（而非绝对值）在不同年龄段是否稳定。如果差值稳定 → 可跨年龄统一标注；如果不稳定 → 按年龄分组标注

3. **标注时的说话人分离**：Phase 4 对 ChildMandarin 自动标注时，每条样本的置信度计算应基于该说话人自身的声学 baseline，而非全数据集的全局 baseline

**关于"397 人会不会太混乱"的评估**：

397 人的声纹多样性既是风险也是优势：
- **风险**：如果跨说话人 F0/RMS 变异大于跨情绪变异，自动标注的精度下降
- **优势**：跨越 397 人学出来的情绪特征，比在 51 人（FAU）上学到的更泛化——模型不会把"某个说话人的声音特征"误认为"某类情绪的特征"
- **已有的保护机制**：Per-speaker 归一化 + Phase 5 人工抽检——如果发现标注质量因说话人多样性而显著下降，可以在 Phase 5 中得到数据证据，及时调整标注策略

三者的互补性很强：**FAU 提供自发性 angry/happy/neutral 的声学基准，C-BESD 提供 4 类均衡的标签模板，ChildMandarin 提供大规模中文自发样本。** 如果它们的声学特征能够对齐（或至少在可控差异内），合并后的数据集将是目前最大的儿童语音情绪训练资源。

### 从"合并数据"到"先验证再合并"

### 完整流程（5 个 Phase）

```
Phase 1: 文献调研 → 建立 4 类情绪的理论声学参考区间
    ↓
Phase 2: FAU + C-BESD 实测 → 提取声学特征 + 验证原标签质量
    ↓
Phase 3: 决策 → 信任原标签 or 声学重聚类 or 分级处理
    ↓
Phase 4: ChildMandarin 自动标注 → 声学指纹打分 + 置信度过滤
    ↓
Phase 5: 人工抽检 → 每种情绪 20-30 条 → 达标/不达标/无法识别
```

---

## Phase 1: 文献调研 —— 4 类情绪的声学理论区间

### 目标

为 angry / happy / neutral / sad 各建立一组**有文献依据、可计算、可跨数据集验证**的声学特征参考区间。

### 特征集（6 维）

| 特征 | 计算方式 | 物理含义 | 关键文献 |
|------|---------|---------|---------|
| F0 mean | 整条语音 F0 均值 (Hz) | 整体音高水平 | Juslin & Laukka (2003); Dmitrieva et al. (2008) |
| F0 std | 整条语音 F0 标准差 | 音高变异性 | Hubbard (1991); Schröder (2001) |
| F0 range | P95 − P5 (Hz) | 音高动态范围 | Dmitrieva et al. (2008) |
| RMS mean | 整条语音 RMS 均值 | 整体响度 | Schröder (2001) |
| HNR | 谐波噪声比 (dB) | 声带振动稳定性 | Scherer (1986) |
| Duration | 有效语音时长 (s) | 语速/持续长度 | Juslin & Laukka (2003) |

### 每类情绪的理论参考区间

#### Angry

| 特征 | 儿童预期区间 | 文献依据 |
|------|------------|---------|
| F0 mean | **200–500 Hz**（高于 neutral 30–60%） | Hubbard (1991): infant anger F0=523Hz 为所有情绪最高；Juslin & Laukka (2003): anger has highest mean F0 |
| F0 std | **高**（所有情绪中 F0 变异性最大） | Juslin & Laukka (2003); Schröder (2001) |
| F0 range | **宽**（P95−P5 > 200 Hz） | Dmitrieva et al. (2008): anger has widest F0 range |
| RMS mean | **高**（高于 neutral 50–200%） | Schröder (2001): anger peak RMS = 2.3× neutral |
| HNR | **中低**（不规则的声带振动） | Scherer (1986): anger has lower HNR due to vocal tension |
| Duration | **短**（< 2.5s） | Juslin & Laukka (2003): anger speech is rapid |

#### Happy

| 特征 | 儿童预期区间 | 文献依据 |
|------|------------|---------|
| F0 mean | **180–450 Hz**（与 angry 相近或略低） | Juslin & Laukka (2003); Schröder (2001) |
| F0 std | **中高**（通常低于 angry） | Schröder (2001): happy has elevated F0 variability but less extreme than anger |
| F0 range | **中宽** | Juslin & Laukka (2003) |
| RMS mean | **中高** | Schröder (2001): happy has elevated energy |
| HNR | **中** | Scherer (1986) |
| Duration | **中**（2–4s） | |

#### Neutral

| 特征 | 儿童预期区间 | 文献依据 |
|------|------------|---------|
| F0 mean | **200–350 Hz**（取决于年龄） | Dmitrieva et al. (2008): child neutral F0 baseline |
| F0 std | **低**（平稳语调） | Juslin & Laukka (2003): neutral has lowest F0 variability |
| F0 range | **窄**（P95−P5 < 150 Hz） | |
| RMS mean | **中等**（不极端） | Schröder (2001) |
| HNR | **中高**（稳定声带振动） | Scherer (1986) |
| Duration | **中等**（2–4s） | |

#### Sad（最关键也最难）

| 特征 | 儿童预期区间 | 文献依据 |
|------|------------|---------|
| F0 mean | **150–250 Hz**（低于 neutral 20–40%） | Juslin & Laukka (2003): sad has the lowest mean F0 |
| F0 std | **低**（低于 neutral 的变异性） | Hubbard (1991): sad has reduced F0 variability |
| F0 range | **窄**（P95−P5 < 100 Hz） | |
| F0 contour | **下降趋势**（语调以下降轮廓为特征） | Juslin & Laukka (2003): sad speech has falling F0 contour |
| RMS mean | **低**（低于 neutral 30–50%） | Schröder (2001): sad has attenuated energy |
| HNR | **较高**（放松的声带 → 稳定振动） | Scherer (1986): sad has higher HNR than angry |
| Duration | **较长**（> 3s，语速慢） | Juslin & Laukka (2003): sad speech has slower tempo |

**Sad 的特殊说明**：FAU 中没有 sad 样本（测试集为 0），C-BESD 的 696 条 sad 为表演型。我们依赖文献理论区间 + C-BESD 实测（前提是 C-BESD sad 的实测分布落在文献理论区间内——这需要通过 Phase 2 验证）。如果验证通过，说明即使是表演型 sad，其声学特征也符合 sad 的通用声学规律（演员按 sad 的声学模板表演），可以作为参考源。

---

## Phase 2: FAU + C-BESD 实测验证

### 目标

对每个数据集的每条样本提取 6 维声学特征，回答三个问题：

1. **同情绪跨数据集一致性**：FAU angry 和 C-BESD angry 的声学分布是同一个东西吗？
2. **同数据集跨情绪可区分性**：FAU 内部能区分 angry 和 neutral 吗？
3. **与文献参考区间的一致性**：实测分布是否落在 Phase 1 的理论预期区间内？

### 验证指标

| 指标 | 计算方式 | 阈值 | 通过标准 |
|------|---------|------|---------|
| 分布重叠率 | $\frac{|FAU[emo] \cap CBESD[emo]|}{|FAU[emo] \cup CBESD[emo]|}$ | > 50% | 双源一致 |
| 类间 FD | $\text{FD}(emo_i, emo_j)$ 在同一数据集内 | FD(angry, neutral) > FD(angry, happy) | 语言有效的情绪区分 |
| 文献一致率 | 实测分布的 95% CI 落在理论区间内的比例 | > 60% | 与文献一致 |

### 输出：每类情绪的审核评级

```
审核结果 = {
    angry: {
        fau_distribution: {f0_mean_range: (250, 400), rms_mean_range: (...), ...},
        cbesd_distribution: {f0_mean_range: (350, 600), rms_mean_range: (...), ...},
        cross_overlap: 0.35,           # < 50% → 不通过
        fau_vs_literature: 0.72,       # 通过
        cbesd_vs_literature: 0.55,     # 通过
        fau_separability: 0.85,        # FAU 内 angry vs neutral 可区分
        grade: "B",                    # 双源不一致但各自与文献一致
        action: "优先使用 FAU 区间（自发性）；对 ChildMandarin 标注时保守",
    },
    sad: {
        fau_distribution: null,        # FAU 无 sad
        cbesd_distribution: {f0_mean_range: (160, 240), ...},
        cbesd_vs_literature: 0.68,     # 通过 → C-BESD sad 可信
        grade: "C",                    # 仅表演型参考
        action: "使用 C-BESD sad 区间，标注 ChildMandarin sad 时启用 FAU neutral 回收",
    },
    ...
}
```

---

## Phase 3: 决策 —— 信任 or 重聚类 or 降级

### 分级处理策略

| 审核等级 | 定义 | 处理方式 |
|---------|------|---------|
| **A 级** | 双源一致 + 文献支撑（如 happy） | 信任原标签，合并 FAU+C-BESD 的分布区间作为声学指纹 |
| **B 级** | 单源可用 + 文献支撑（如 angry，双源不一致） | 优先信任自发性来源（FAU），标注时保守 |
| **C 级** | 仅表演型参考 + 文献支撑（如 sad） | 信任但降低置信度阈值 |
| **D 级** | 审核不通过（实测与文献严重偏离） | **不信任原标签 → 对 ChildMandarin 该情绪不标注** |

### Phase 3 的可能结果与对应路径

Phase 2 跑完数据后，有三种可能的情况——每种都有对应的下一步：

| 审核结果 | 可能性 | 含义 | 处理 |
|---------|--------|------|------|
| **通过**（重叠率 > 60%） | 🟢 乐观预期 | FAU 和 C-BESD 对同一种情绪的定义一致，声学特征区间可合并 | 直接合并，推进 Phase 4 自动标注 |
| **部分通过**（重叠率 30-60%） | 🟡 中性预期 | 部分情绪（大概率 happy/neutral）一致，部分（如 angry）有差异——差异可能来自表演型 vs 自发型的声学表现不同，而非标签错误 | 保留交集区间，交集外的样本标记为 ambiguous，不参与训练 |
| **不通过**（重叠率 < 30%） | 🔴 保守预期 | 特定情绪在跨数据集上声学表现差异显著 | 对该情绪使用自发性来源（FAU）的区间作为主参考，分析偏差原因，在论文中讨论而非回避 |


### FAU neutral 的 " 回收"
是对FAU的中性进行重新标注，我怀疑里面包含了大量SAD和分辨失败的Happy和Angry。具体如：FAU neutral 样本 → 声学特征提取 → sad、Happy、Angry 指纹匹配
    ├── 匹配度 > 阈值 → 重新标注，并移入 对应的训练池
    ├── 匹配度在 neutral-其他情绪 中间 → "ambiguous"，不参与训练
    └── 匹配度低 → 保留为 neutral

**如果回收成功**（预计回收 500-1,000 条），FAU 自身的 sad 样本和其他情绪样本可能足以支持训练，不再依赖外部数据。**如果回收量不足**，则使用 C-BESD sad（前提是 Phase 2 验证其声学特征在 sad 理论区间内）。
---

## Phase 4: ChildMandarin 自动标注

### 标注算法

对 ChildMandarin 每条样本：

```
1. 提取 6 维声学特征
2. 对每种情绪 emo_i，计算匹配分数：
   score(emo_i) = Σ w_j × match(feature_j, emo_i_reference_range)
   其中 w_j 为各特征的权重，从 FAU 训练数据中通过逻辑回归学习
3. 如果 max(score) > threshold 且 max(score) − second_max(score) > margin：
   → 标注为 argmax(score)，置信度 = max(score)
   否则：
   → 标记为 "uncertain"
```

### 标注输出

每条 ChildMandarin 样本输出：
- `predicted_label`: angry / happy / neutral / sad / uncertain
- `confidence`: 0-1 的匹配分数
- `reference_grade`: A/B/C（参考来源的审核等级）
- `recommend_training`: True/False（是否建议用于训练）

### 训练池准入标准

| 条件 | 是否入训练池 |
|------|------------|
| confidence > 0.8 且 reference_grade = A | ✅ 入 |
| confidence > 0.7 且 reference_grade = A/B | ✅ 入 |
| confidence > 0.7 且 reference_grade = C | ⚠️ 入，但训练时给更高 dropout |
| confidence < 0.7 或 reference_grade = D 或 label = "uncertain" | ❌ 不入 |

---

## Phase 5: 人工抽检

### 抽检方案

从 ChildMandarin 自动标注结果中，每种情绪随机抽取 20–30 条（共 80–120 条），由人工听音频并判断：

| 判定 | 标准 |
|------|------|
| **达标** | 标签正确，该段语音确实表达了对应情绪 |
| **不达标** | 标签错误，该段语音表达的是另一种情绪 |
| **无法识别** | 音频质量差（噪音、不完整）或情绪表达模糊（无法判断） |

### 验收标准

| 每种情绪的准确率 | 后续行动 |
|----------------|---------|
| > 80% | 该情绪的自动标注方案通过，入训练池 |
| 60–80% | 通过但需谨慎，降低入池 confidence 阈值 |
| < 60% | 该情绪的自动标注方案不可靠，需要重做 Phase 1-3 或放弃 ChildMandarin 的该情绪 |

---

## 执行步骤

| 步骤 | 内容 | 预计耗时 | 依赖 |
|------|------|---------|------|
| 1 | Phase 1: 整理文献声学参考区间（本文档已完成初步框架） | — | — |
| 2 | Phase 2: 编写并运行 FAU + C-BESD 声学特征提取脚本 | ~2h | 云端服务器 |
| 3 | Phase 2: 生成审核报告（同情绪跨数据集重叠率、类间 FD、文献一致性） | ~1h | 步骤 2 |
| 4 | Phase 3: 根据审核结果做决策（信任/重聚类/降级） | — | 人工判断 |
| 5 | Phase 3: 运行 FAU neutral sad 回收脚本 | ~1h | 步骤 2-4 |
| 6 | Phase 4: ChildMandarin 自动标注（32,658 条） | ~2h | 步骤 2-5 |
| 7 | Phase 5: 人工抽检 80–120 条 | — | 人工 |
| 8 | 根据抽检结果调整标注参数 | — | 步骤 7 |
| 9 | 构造最终训练集 + 跑对比实验 | ~3h | 步骤 8 |

---

## 与 v1 的关键区别

| | v1 | v2 |
|---|---|---|
| **核心假设** | FAU/C-BESD/新数据集的情绪标签一致 | 需要先验证一致性，不一致就重定义 |
| **对原始标签的态度** | 信任 | 怀疑 → 用声学特征验证 |
| **Sad 的来源** | 外部数据集 or C-BESD fallback | 优先从 FAU neutral 回收，外部数据作为补充 |
| **ChildMandarin 的标注** | 伪标签（依赖模型） | 声学指纹匹配（依赖特征+文献，不依赖模型 bias） |
| **论文贡献** | 数据处理方案 | 若验证通过，跨数据集声学一致性方法可成为通用方案 |

---

*文档版本: v2 | 创建: 2026-06-05 | 更新: 2026-06-05*
