# child-ser-corpus-builder

> 跨数据集儿童语音情绪识别 —— 数据集构造工程
>
> 关联主项目：[分布驱动儿童SER](https://github.com/xuzichao/child-ser-distribution-driven)

---

## 0. 项目定位

### 0.1 为什么需要这个仓库？

儿童语音情绪识别（SER）面临一个根本性瓶颈：**没有大规模、多类型、标签可靠且类别均衡的儿童情绪数据集**。

现有数据集各有限制：

| 数据集 | 优势 | 限制 |
|--------|------|------|
| C-BESD | 唯一4类齐全的儿童数据集 | 表演式，样本量小（~2,780），域内泛化已近上限（~93%） |
| FAU Aibo | 最大规模儿童自发型（18,216条） | 类别严重失衡（neutral 59%），sad 缺失，标签粗放 |
| ChildMandarin | 规模极大（40,913条），397人跨方言 | **无情绪标签** |

本仓库的目标：**不信任原始标签，用声学特征重新定义情绪类别，构造一个跨数据集、跨语言、跨风格的儿童情绪识别训练集。**

### 0.2 与主项目的关系

```
┌─────────────────────────────────────┐
│  本仓库（child-ser-corpus-builder）  │
│  - 声学指纹提取与分析                 │
│  - 标签审核与重建                     │
│  - ChildMandarin 自动标注             │
│  - 最终训练集构造                     │
│  产出: unified_trainset.h5            │
└──────────────┬──────────────────────┘
               │ 训练集
               ▼
┌─────────────────────────────────────┐
│  主项目（分布驱动儿童SER）             │
│  - 模型训练与评估                     │
│  - FD 分布偏移分析                    │
│  - 可解释性分析                       │
│  - 论文撰写                           │
└─────────────────────────────────────┘
```

### 0.3 核心方法论

```
不信任原始标签
      ↓
用声学特征重新定义情绪类别（acoustic fingerprint）
      ↓
跨数据集审核（文献 + 互验证 + 聚类分析）
      ↓
分级信任 → 自动标注 → 人工抽检
      ↓
构造统一训练集
```

---

## 1. 数据集资产清单

### 1.1 已有儿童数据集

| 属性 | C-BESD | FAU Aibo | ChildMandarin |
|------|--------|----------|---------------|
| **语言** | 马来语 + 英语 + 泰卢固语 | 德语 | 中文 |
| **风格** | 表演式（acted） | 自发型（spontaneous） | 自发型（spontaneous） |
| **样本数** | ~2,780 | 18,216 | 40,913 |
| **说话人数** | 237（儿童） | 51（儿童 10-13岁） | 397（儿童 3-5岁） |
| **情绪标签** | ✅ 4类均衡 | ⚠️ 5类失衡 | ❌ 无标签 |
| **平均时长** | ~2.5s | ~2.0s | ~3.52s |
| **录制环境** | 受控 | 自然交互 | 自然交互 |
| **标签可信度** | 中高（表演式，情绪明确） | 中低（自发型标注粗放） | N/A |

### 1.2 情绪标签分布

**C-BESD**（4类均衡）:

| 情绪 | 样本数 | 占比 |
|------|--------|------|
| angry | ~695 | ~25% |
| happy | ~702 | ~25% |
| neutral | ~691 | ~25% |
| sad | ~696 | ~25% |

**FAU Aibo**（5类失衡）:

| 情绪 | 样本数 | 占比 |
|------|--------|------|
| angry | ~2,548 | ~14% |
| happy | ~2,168 | ~12% |
| neutral | ~10,796 | **~59%** |
| emphatic | ~1,459 | ~8% |
| rest | ~1,245 | ~7% |
| **sad** | **0** | **缺失** |

> 注：FAU 原始 11 类标签已按 AC 协议映射为 5 类。emphatic 和 rest 在 4 类统一映射中被丢弃。

**ChildMandarin**：40,913 条，**全部无情绪标签**。

### 1.3 角色分配

| 数据集 | 角色 | 入训练池 |
|--------|------|---------|
| **C-BESD** | 声学指纹·表演型参考 | ✅ 是 |
| **FAU Aibo** | 声学指纹·自发型参考 | ✅ 是（审核+neutral回收后） |
| **ChildMandarin** | 标注目标 | ✅ 是（自动标注+人工抽检后） |
| **IEMOCAP** | 负对照（验证儿童/成人声学边界） | ❌ 否 |

> **IEMOCAP 排除理由**：FD(C-BESD, IEMOCAP) = 7.20 是系统性的年龄偏移（儿童 F0 基线比成人高 200-300Hz），不是"处理"能消除的。IEMOCAP 仅用于验证：我们的儿童声学参考区间确实排除了成人分布。

---

## 2. 六阶段执行方案

```
Phase 0: 文献声学参考区间建立
    ↓
Phase 1: C-BESD + FAU + IEMOCAP 声学特征实测
    ↓
Phase 2: 三向审核（跨数据集 × 跨情绪 × 文献一致性）
    ↓
Phase 3: 决策 + FAU neutral 回收
    ↓
Phase 4: ChildMandarin 自动标注
    ↓
Phase 5: 人工抽检验收
    ↓
Phase 6: 构造最终训练集 + 对比实验设计
```

---

### Phase 0: 文献声学参考区间建立

#### 目标
从文献中建立四类情绪（angry, happy, neutral, sad）在 6 个声学维度上的理论参考区间。

#### 6 维声学特征

| # | 特征 | 符号 | 与情绪的已知关系 | 物理含义 |
|---|------|------|----------------|---------|
| 1 | F0 均值 | μ_F0 | angry↑, sad↓ | 基频高低反映 arousal |
| 2 | F0 标准差 | σ_F0 | angry↑, neutral↓ | 基频变异性反映情绪波动 |
| 3 | RMS 能量均值 | μ_E | angry/happy↑, sad↓ | 能量 = arousal 的直接代理 |
| 4 | RMS 能量标准差 | σ_E | 与 arousal 变异性相关 | 能量波动反映情绪不稳定度 |
| 5 | 语速 | SR | angry/happy↑, sad↓ | 音节/秒，反映心理运动速度 |
| 6 | HNR 均值 | μ_HNR | angry↓（喉音紧张）, neutral↑ | 谐波噪声比反映音质 |

#### 技术要点
- **F0 提取参数**：`librosa.pyin(fmin=60, fmax=1200)` —— 儿童 F0 可达 500-600Hz，默认 `fmax=400Hz` 会严重低估
- **Per-speaker z-score 归一化**：397 人声纹差异不能靠全局归一化解决
- **文献来源**：儿童语音声学文献 + 成人情绪声学文献（标注适用性差异）

#### 产出
- `references/acoustic_reference_ranges.json` — 四情绪 × 六维度的理论区间
- `references/literature_notes.md` — 文献调研笔记

---

### Phase 1: 声学特征实测

#### 目标
对 C-BESD、FAU Aibo、IEMOCAP 的所有样本提取 6 维声学特征，为审核提供实测数据。

#### 脚本
`scripts/extract_acoustic_features.py`

```
输入: 数据集根目录 + 标签文件
处理:
  1. 逐 WAV 加载 (16kHz mono)
  2. 提取 6 维特征（F0/pyin, RMS, speaking rate, HNR）
  3. Per-speaker z-score 归一化
  4. 保存为 (N, 6) numpy array + 元数据 CSV
输出: data/features/{dataset}_features.npy + {dataset}_metadata.csv
```

#### 元数据字段
`filename, speaker_id, original_label, f0_mean, f0_std, energy_mean, energy_std, speaking_rate, hnr_mean`

#### IEMOCAP 提取
与 C-BESD/FAU 完全相同的提取管线，标签映射到 4 类统一标签。仅用于 Phase 2 的对比审核，不进入训练池。

#### 产出
- `data/features/cbesd_features.npy` + `cbesd_metadata.csv`
- `data/features/fau_aibo_features.npy` + `fau_aibo_metadata.csv`
- `data/features/iemocap_features.npy` + `iemocap_metadata.csv`

---

### Phase 2: 三向审核

#### 审核维度

| 维度 | 指标 | 公式 | 阈值 |
|------|------|------|------|
| **跨数据集一致性** | 分布重叠率 OVL | ∫min(p_CBESD(x), p_FAU(x)) dx | >0.60 = 一致 |
| **跨情绪可区分性** | Cohen's d | |μ_emo − μ_other| / σ_pooled | >0.80 = 可区分 |
| **文献一致性** | 文献区间匹配率 | 实测 [P25, P75] 在文献区间内的比例 | >0.50 = 一致 |

#### 审核矩阵（4 情绪 × 3 维度）

```
         angry   happy   neutral   sad
C↔F OVL   ?       ?        ?        ?
情绪 d'   ?       ?        ?        ?
文献匹配   ?       ?        ?        ?
─────────────────────────────────────
审核等级   ?       ?        ?        ?
```

#### 预期发现（假设）

| 情绪 | 预期审核结果 | 理由 |
|------|------------|------|
| **happy** | 🟢 A级（三向一致） | 高 arousal + 正 valence 声学特征鲜明，表演/自发差异小 |
| **angry** | 🟡 B级（自发/表演有差异） | 表演型 angry 可能过度夸张 F0 和能量 |
| **neutral** | 🟡 B级（跨数据集可能不一致） | 自发型 neutral 更"平淡"；FAU neutral 可能混入其他情绪 |
| **sad** | 🟠 C级（仅 C-BESD 参考，FAU 无 sad） | 单向参考，无法跨数据集验证 |

#### 产出
- `reports/audit_matrix.json` — 12 格审核结果
- `reports/audit_report.md` — 审核分析与建议
- `figures/` — 各情绪的分布对比图、小提琴图、OVL 可视化

---

### Phase 3: 决策 + FAU neutral 回收

#### 3a: FAU neutral 内部聚类分析（前置验证）

**核心问题**：你怀疑 FAU neutral 里藏了 sad/angry/happy——但这需要先验证。

```
步骤:
  1. 提取 FAU neutral 所有样本的 6 维特征
  2. GMM (n_components=2~4) 或 KDE 聚类
  3. 如果多模态 → neutral 内部存在子群，回收有依据 → 进入 3b
  4. 如果单峰 → 儿童自发语音 neutral 确实占主导 → 跳过回收
```

**为什么必须做这一步**：如果 neutral 内部是单峰的，儿童在非诱导场景下确实大多平淡说话，强行回收只会引入噪声。

#### 3b: 声学指纹匹配回收

```
FAU neutral 样本 → 6 维特征
  ├── 匹配 sad 指纹（来自 C-BESD sad） → 匹配度 > θ → relabel as sad
  ├── 匹配 angry/happy 指纹 → 匹配度 > θ → relabel
  ├── 匹配度在模糊区间 → "ambiguous"，不入训练
  └── 匹配度低 → 保留为 neutral
```

**回收数量预估**：500-1,000 条（基于 FAU neutral ~10,796 的 5-10%）

#### 3c: 分级信任策略

| 等级 | 条件 | 处理方式 | 入池 |
|------|------|---------|------|
| **A** | OVL>60% + d'>0.8 + 文献一致 | 信任原标签，合并 FAU+C-BESD 分布区间为声学指纹 | ✅ |
| **B** | OVL 30-60% 或 d' 0.5-0.8 | 优先信任自发型来源（FAU），标注时保守 | ✅ |
| **C** | 仅表演型参考 + 文献支撑 | 信任但降低置信度阈值 | ⚠️ |
| **D** | 三项全不通过 | **不对 ChildMandarin 该情绪做自动标注** | ❌ |

#### 产出
- `reports/fau_neutral_clustering.md` — 聚类分析结果
- `reports/neutral_recovery_report.json` — 回收明细
- `data/features/fau_aibo_relabeled_metadata.csv` — 修正后的 FAU 标签

---

### Phase 4: ChildMandarin 自动标注

#### 声学指纹参考

每类情绪 A/B/C 级的参考区间，从 Phase 3 审核通过的样本中提取（per 特征 的 [P25, P75] 或 [μ−σ, μ+σ]）。

#### 标注算法

```
对每条 ChildMandarin 样本：
  1. 提取 6 维声学特征
  2. Per-speaker z-score 归一化（关键：397 人声纹差异）
  3. 计算与每种情绪的匹配分数：
     score(emo_i) = Σ w_j × match(feature_j, emo_i_ref_range)
     其中:
       - match(f) = 1 if f ∈ [P25, P75] of emo_i, else 0
       - w_j 两级权重设定:
           a) 先验权重: 从文献直接设定（如 F0 mean 对 angry 权重大）
           b) 数据驱动权重: 从 Phase 3 的 A 级样本中逻辑回归学习
           c) 两者不一致时 → 人工审查
  4. 决策逻辑：
     IF max(score) > T_conf
        AND max(score) − second_max(score) > T_margin
        AND reference_grade(argmax) ≠ D
       → 标注为 argmax(score)
     ELSE
       → "uncertain"
```

#### 超参数

| 参数 | 建议默认值 | 说明 |
|------|-----------|------|
| T_conf | 0.65 | 置信度阈值 |
| T_margin | 0.15 | 最高分与次高分的最小差距 |
| 特征权重 w_j | Phase 3 学习 | 防止循环论证：权重从 A 级样本学，不用全部 FAU |

#### 标注输出

每条 ChildMandarin 样本输出：

| 字段 | 类型 | 说明 |
|------|------|------|
| `predicted_label` | str | angry / happy / neutral / sad / uncertain |
| `confidence` | float [0,1] | 匹配分数 |
| `reference_grade` | str | A / B / C（参考来源的审核等级） |
| `recommend_training` | bool | confidence > 0.7 且 grade ∈ {A,B} → True |

#### 产出
- `data/annotations/childmandarin_labels.csv` — 40,913 条标注结果
- `data/annotations/childmandarin_training_pool.csv` — 建议入池子集

---

### Phase 5: 人工抽检验收

#### 抽检方案

从 ChildMandarin 自动标注结果中，每种情绪随机抽取 20-30 条（共 80-120 条）。

#### 判定标准

| 判定 | 定义 |
|------|------|
| **达标** | 标签正确，该段语音确实表达了对应情绪 |
| **不达标** | 标签错误，该段语音表达的是另一种情绪 |
| **无法识别** | 音频质量差（噪音、不完整）或情绪表达模糊 |

#### 验收标准

| 每种情绪的准确率 | 后续行动 |
|----------------|---------|
| > 80% | ✅ 该情绪自动标注方案通过，入训练池 |
| 60-80% | ⚠️ 通过但需谨慎，降低 confidence 阈值 |
| < 60% | ❌ 该情绪方案不可靠，重做 Phase 1-3 或放弃 CM 的该情绪 |

#### 产出
- `reports/manual_spot_check.csv` — 抽检明细
- `reports/spot_check_summary.md` — 抽检总结与行动建议

---

### Phase 6: 构造最终训练集 + 对比实验设计

#### 6.1 训练集构造

```
最终训练集 = {
    C-BESD:  原标签，全部入池 (N ≈ 2,780, A级信任)
    FAU:     审核后标签 + neutral回收后标签，A/B级入池 (N ≈ 8,000-12,000)
    ChildMandarin: 自动标注, confidence>0.7,  A/B级入池 (N 待 Phase 4-5 确定)
}
→ 统一 4 类标签 {angry, happy, neutral, sad}
→ Speaker-independent hash split (70/15/15)
→ 输出格式: .h5 或 .npy，兼容主项目 data_loader
```

#### 6.2 训练池准入标准

| 条件 | 入池 |
|------|------|
| C-BESD（原标签） | ✅ 全部 |
| FAU（A 级情绪 + 原标签） | ✅ 全部 |
| FAU（B 级情绪 + 原标签） | ✅ 全部 |
| FAU（neutral 回收后重标） | ✅ confidence > 阈值 |
| FAU（ambiguous） | ❌ 不入 |
| ChildMandarin（confidence > 0.8 且 grade = A） | ✅ |
| ChildMandarin（confidence > 0.7 且 grade = A/B） | ✅ |
| ChildMandarin（confidence > 0.7 且 grade = C） | ⚠️ 入但训练时加更高 dropout |
| ChildMandarin（confidence < 0.7 或 grade = D 或 uncertain） | ❌ |

#### 6.3 产出格式

```
output/
├── unified_trainset/
│   ├── train_wavs.npy          # (N_train, T, 1) waveform
│   ├── train_labels.npy        # (N_train,) int 0-3
│   ├── train_speakers.npy      # (N_train,) hashed speaker_id
│   ├── train_sources.npy       # (N_train,) 0=C-BESD, 1=FAU, 2=CM
│   ├── val_wavs.npy
│   ├── val_labels.npy
│   ├── val_speakers.npy
│   ├── val_sources.npy
│   ├── test_wavs.npy
│   ├── test_labels.npy
│   ├── test_speakers.npy
│   ├── test_sources.npy
│   └── dataset_manifest.json   # 统计信息
```

---

## 3. 后续：对比实验设计（待讨论）

> ⚠️ 本节为初步框架，具体实验方案在 Phase 0-5 完成后详细讨论。

### 3.1 实验矩阵框架

| 实验 | 训练集 | 测试集 | 验证目标 |
|------|--------|--------|---------|
| E1 | C-BESD only | C-BESD test | 原性能上限基准 |
| E2 | FAU only | FAU test | 自发型域内基准 |
| E3 | C-BESD + FAU | C-BESD test | 自发型→表演型迁移增益 |
| E4 | C-BESD + FAU | FAU test | 混合→自发型泛化 |
| E5 | All (C-BESD+FAU+CM) | C-BESD test | 最大数据量的上限 |
| E6 | All (C-BESD+FAU+CM) | FAU test | 最大数据→自发型泛化 |
| E7 | All (C-BESD+FAU+CM) | CM held-out | ChildMandarin 标注质量间接验证 |

### 3.2 评估指标

| 指标 | 用途 | 参考来源 |
|------|------|---------|
| **WA** (Weighted Accuracy) | 主指标，总体准确率 | AC Suite 标准 |
| **UAR** (Unweighted Average Recall) | 辅指标，类别不平衡稳健 | AC Suite 标准 |
| **t-SNE / UMAP** | 特征空间可视化：按情绪着色 + 按数据源着色 | shi25d, hu25c |
| **FD (Fréchet Distance)** | 分布偏移量化：训练集 vs 测试集 | 主项目 Part 1 核心指标 |
| **混淆矩阵** | 每类情绪的分类模式分析 | 主论文图表 |
| **APC** (Average Pooling Contribution) | 注意力可解释性 | 主项目 Part 1 指标 |

### 3.3 关键假设

1. **数据量假设**：C-BESD + FAU + CM 合并训练 > 任一单数据集训练
2. **FD 假设**：合并后的训练集与 C-BESD test 的 FD 应该 **小于** FAU→C-BESD 的零样本 FD（即混合训练缩小了分布偏移）
3. **可复现假设**：Part 1 的四条规律在新数据集上仍然成立
4. **跨语言假设**：中文自发型 (CM) + 德语自发型 (FAU) 的声学特征在 per-speaker 归一化后具有跨语言一致性

### 3.4 参考论文

| 论文 | 方法 | 借鉴点 |
|------|------|--------|
| **shi25d** | Speaker-Aware Multi-Task | t-SNE 验证说话人-情绪解耦 |
| **hu25c** | LaSCL 对比学习 + Mixed Augmentation | t-SNE 验证特征判别力，LOSO 协议 |
| **zhang25m** | Whisper+LoRA+GPT-4o 儿童 fluency | VTLN 儿童化成人语音（可作为反向参考：证明儿童→成人不可逆） |

---

## 4. 仓库结构

```
child-ser-corpus-builder/
├── README.md                           # 本文档
├── references/                         # 文献调研
│   ├── literature_notes.md             #   文献笔记
│   └── acoustic_reference_ranges.json  #   理论声学参考区间
├── scripts/                            # 执行脚本
│   ├── extract_acoustic_features.py    #   Phase 1: 声学特征提取
│   ├── audit_cross_dataset.py          #   Phase 2: 三向审核
│   ├── cluster_fau_neutral.py          #   Phase 3a: FAU neutral 聚类
│   ├── recover_fau_neutral.py          #   Phase 3b: neutral 回收
│   ├── annotate_childmandarin.py       #   Phase 4: CM 自动标注
│   └── spot_check_sampler.py           #   Phase 5: 抽检采样
├── data/                               # 中间数据（gitignore 原始音频 + 大 npy）
│   ├── features/                       #   声学特征 npy + metadata csv
│   ├── annotations/                    #   CM 标注结果 csv
│   └── unified_trainset/              #   最终训练集（Phase 6 产出）
├── reports/                            # 审核与抽检报告
│   ├── audit_report.md                 #   Phase 2 审核报告
│   ├── audit_matrix.json               #   12 格审核矩阵
│   ├── fau_neutral_clustering.md       #   Phase 3a 聚类分析
│   ├── neutral_recovery_report.json    #   Phase 3b 回收明细
│   └── spot_check_summary.md           #   Phase 5 抽检总结
├── figures/                            # 可视化
└── .gitignore
```

---

## 5. 执行状态

| 步骤 | 状态 | 备注 |
|------|------|------|
| Phase 0: 文献参考区间 | ⬜ 待执行 | 需文献调研 + 建立 JSON |
| Phase 1: 声学特征实测 | ⬜ 待执行 | 依赖 Phase 0 |
| Phase 2: 三向审核 | ⬜ 待执行 | 依赖 Phase 1 |
| Phase 3: 决策 + neutral 回收 | ⬜ 待执行 | 依赖 Phase 2 |
| Phase 4: CM 自动标注 | ⬜ 待执行 | 依赖 Phase 2-3 |
| Phase 5: 人工抽检 | ⬜ 待执行 | 依赖 Phase 4 |
| Phase 6: 构造训练集 | ⬜ 待执行 | 依赖 Phase 5 |
| 对比实验设计 | ⬜ 待讨论 | Phase 0-5 完成后细化 |

---

## 6. 实验协议

### 6.1 Speaker 划分

使用 MD5 hash 确定性划分，70/15/15 (train/val/test)，zero-leakage assert（同一说话人不跨集合）。

ChildMandarin 的 397 人覆盖 22 省，划分时需确保 train/val/test 的地域分布大致均匀（避免某省全部落入 test）。

### 6.2 Per-speaker 归一化

所有声学特征在进入匹配/分类前，**必须先做 per-speaker z-score 归一化**。这是处理 397 人声纹差异的核心操作。

```
对说话人 s 的所有样本：
  feature_normalized = (feature − μ_s) / σ_s
其中 μ_s 和 σ_s 从该说话人的所有样本计算
```

### 6.3 数据泄露防范

| 风险 | 防范 |
|------|------|
| Phase 3 权重学习时偷看 CM 分布 | 权重仅从 C-BESD + FAU 学习，CM 完全 hold-out |
| Per-speaker 归一化跨集合泄露 | 每个 speaker 的 μ/σ 仅从该 speaker 的 train 样本计算（对 CM 亦然） |
| Phase 5 抽检信息回流到 Phase 4 | 抽检结果仅用于评估，不用于调参（除非准确率 < 60% 需要重做） |

---

*最后更新: 2026-06-06*
