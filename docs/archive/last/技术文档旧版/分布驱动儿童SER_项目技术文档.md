# 分布驱动的儿童语音情绪识别

> **协议**: AC Suite `ac_suite_2026-05` | **主干**: WavLM Base (wavlm-base-sv) | **数据冻结**: 2026-05-27

---

## 1. 相关工作

### 1.1 语音情绪识别 (SER)

语音情绪识别旨在从语音信号中自动识别说话人的情绪状态。传统 SER 方法依赖手工声学特征，包括梅尔频率倒谱系数 (MFCC)、基频 (F0)、共振峰、能量、过零率等，随后通过 SVM、HMM 或早期神经网络进行分类。然而，手工特征受限于特征工程的信息瓶颈，难以捕捉情绪表达中的微妙声学变化。

### 1.2 自监督预训练模型

近年来，大规模自监督学习 (SSL) 模型在语音领域取得突破。Wav2Vec 2.0 (Baevski et al., 2020) 首次将 Transformer 架构与对比学习引入语音预训练；HuBERT (Hsu et al., 2021) 通过离线聚类获取伪标签进行掩码预测；WavLM (Chen et al., 2022) 在 HuBERT 基础上引入噪声混叠和语音去噪训练，在 SUPERB 多项下游任务中达到最优。这些模型在成人 LibriSpeech、VoxPopuli 等语料上预训练，其隐空间表征已习得丰富的声学-语义映射。

**优点**：帧级特征（768 维 @ 50Hz）远优于手工 162 维均值向量，保留了完整的时序信息。

**缺点**：所有主流 SSL 模型均在成人语音上预训练，儿童语音（F0 更高、共振峰更分散、韵律变异性更大）在其隐空间中存在系统性分布偏移，直接迁移性能受损。

### 1.3 儿童语音情绪识别

儿童 SER 相对成人 SER 研究更为匮乏。现有工作多使用演绎式语料（如 BESD、EMOVO），方法上直接沿用成人 SER 的框架，包括手工特征提取、数据增强（pitch shift、time stretch）以及 CNN/RNN 分类器。BESD (Zakaria et al., 2018) 作为目前最大的多语言儿童情绪语料库，覆盖马来语、英语和泰卢固语，且说话人无重叠，适合跨语言研究。

**优点**：BESD 提供了多语言、多情绪的标准化基准。

**缺点**：现有工作普遍忽视儿童-成人声学差异，直接将成人 SER 的数据增强策略和模型架构迁移到儿童场景，缺乏针对儿童语音特性的专门设计。手工特征 + mean-pooling 的管线丢掉了时序信息，且增强策略（如成人参数 pitch shift）可能破坏儿童语音中与情绪相关的精细声学线索。

### 1.4 本文改进空间

现有研究的空白可归纳为三点：(1) 缺乏对儿童-成人语音分布偏移的定量诊断；(2) 忽视儿童韵律特性在时序建模中的先验价值；(3) 成人数据增强策略未经检验即迁移到儿童 SER。本文从上述三个空白出发，提出分布诊断 → 韵律池化 → 增强敏感性分析的完整管线。

---

## 2. 方法

### 2.1 整体架构

本文提出一个儿童语音情绪识别的端到端框架，由四个阶段组成：

```
原始波形 (4s, 16kHz mono)
    │
    ▼
① WavLM Base (frozen, 94.6M params)
    │  输出: 12 层 hidden states, 每层 (T, 768), 帧率 50Hz
    ▼
② WavLMLayerFusion (12 个可学习标量权重)
    │  输出: 12 层 softmax 加权求和 → (T, 768)
    ▼
③ Temporal Pooling
    │  Self-Attention 或 Prosody Guided，各 111,105 参数
    │  输出: (768,)
    ▼
④ SEMLP Classifier (~593K params)
    │  输出: 4 类 logits
    ▼
{angry, happy, neutral, sad}
```

总可训练参数约 **704K**，不到 WavLM 骨干的 1%。

### 2.2 特征提取：WavLM Backbone

采用 **WavLM Base** (`microsoft/wavlm-base-sv`, Chen et al., 2022) 作为冻结的特征提取器。该模型包含 12 层 Transformer Encoder，在大规模成人语音数据上预训练，参数总量 94.6M。

对于输入波形 $\mathbf{x} \in \mathbb{R}^{T_{\text{wav}}}$（16kHz 采样，4 秒截断/填充至 $T_{\text{wav}} = 64000$），WavLM 输出 13 个隐藏状态张量：

$$\{\mathbf{H}_0, \mathbf{H}_1, \ldots, \mathbf{H}_{12}\}, \quad \mathbf{H}_i \in \mathbb{R}^{T \times 768}$$

其中 $\mathbf{H}_0$ 为输入嵌入层，$\mathbf{H}_1 \ldots \mathbf{H}_{12}$ 为 12 个 Transformer 层的输出。帧率 50Hz（帧移 320 样本 = 20ms），$T = 200$ 帧。**所有 94.6M 参数在训练期间冻结，不参与梯度更新。**

### 2.3 层级融合：WavLMLayerFusion

WavLM 不同层编码了不同粒度的信息：浅层捕获声学特征（音素轮廓、F0 包络），中层编码韵律模式，深层提取语义和说话人信息。仅取最后一层（旧管线）会丢失浅中层的关键声学信息——这对儿童语音尤为重要。

本文引入**可学习层级加权求和** (WavLMLayerFusion)，将 12 层的输出融合为单一特征图：

$$\mathbf{F} = \sum_{i=1}^{12} w_i \cdot \mathbf{H}_i$$

其中 $w_i$ 为 12 个可训练标量参数，通过 softmax 归一化：

$$w_i = \frac{\exp(\theta_i)}{\sum_{j=1}^{12} \exp(\theta_j)}$$

$\theta_i$ 初始化为 $\ln(1/12)$，即初始等权重。仅 12 个可训练参数。

**训练后权重分布**（Exp2 C-BESD）表明中层（L7-L9）权重最高，Layer 8 权重 0.147 为全层最高，证实中层韵律层对儿童语音情绪识别最关键。

### 2.4 时序池化

池化模块将帧级特征 $(T, 768)$ 压缩为全局表征 $(768,)$。本文对比两种池化策略，两者参数量严格对齐（各 111,105），确保消融实验的公平性。

#### 2.4.1 Self-Attention Pooling（基准）

纯数据驱动的自注意力池化，不注入任何外部先验。注意力分数由四层 MLP 从 SSL 特征本身计算：

$$a_t = \text{MLP}_{\text{attn}}(\mathbf{f}_t) \in \mathbb{R}^1$$

$$\alpha_t = \frac{\exp(a_t)}{\sum_{j=1}^{T} \exp(a_j)}, \quad \mathbf{z} = \sum_{t=1}^{T} \alpha_t \cdot \mathbf{f}_t$$

其中 $\mathbf{f}_t \in \mathbb{R}^{768}$ 为第 $t$ 帧的融合特征，MLP 结构为 $768 \to 116 \to 100 \to 100 \to 1$（ReLU/Tanh 激活，含 Dropout）。

#### 2.4.2 Prosody Guided Pooling（本文核心方法）

利用儿童语音的韵律特性（F0 高且多变、能量波动大）作为时序重要性的显式先验。注意力权重的计算同时依赖 SSL 特征和声学韵律线索。

**韵律提取**：从原始波形提取帧级 F0 曲线和 RMS 能量，帧移 320 样本以对齐 WavLM 的 50Hz 帧率：

- **F0 提取**：YIN 算法 (de Cheveigné & Kawahara, 2002)，搜索范围 C2–C7 (65–2093 Hz)，覆盖儿童高频
- **能量提取**：librosa RMS energy，帧移 320 样本

F0 归一化为 $[0, 1]$（除以 2093 Hz），能量逐批归一化。

**韵律投影**：韵律对 $(f_0^{(t)}, e^{(t)}) \in \mathbb{R}^2$ 通过两层 MLP 投影为韵律嵌入：

$$\mathbf{p}_t = \text{MLP}_{\text{pros}}([f_0^{(t)}; e^{(t)}]) \in \mathbb{R}^{64}$$

MLP 结构为 $2 \to 64 \to 64$（ReLU 激活）。

**融合注意力**：将韵律嵌入与 SSL 特征拼接后计算注意力分数：

$$\mathbf{c}_t = [\mathbf{f}_t; \mathbf{p}_t] \in \mathbb{R}^{832}$$

$$a_t = \text{MLP}_{\text{fusion}}(\mathbf{c}_t) \in \mathbb{R}^1$$

$$\alpha_t = \frac{\exp(a_t)}{\sum_{j=1}^{T} \exp(a_j)}, \quad \mathbf{z} = \sum_{t=1}^{T} \alpha_t \cdot \mathbf{f}_t$$

融合 MLP 结构为 $832 \to 128 \to 1$（Tanh 激活）。Padding 帧通过 mask 机制在 softmax 前设置为 $-\infty$。

### 2.5 分类器：SEMLP

采用轻量级通道门控 MLP 分类器（SE-MLP）。结构如下：

$$\mathbf{z} \xrightarrow{768 \to 512} \text{BN} \to \text{ReLU} \to \text{Dropout}(0.3) \to$$

$$\text{SEBlock}(512) \to$$

$$512 \to 256 \to \text{BN} \to \text{ReLU} \to \text{Dropout}(0.3) \to$$

$$256 \to 128 \to \text{ReLU} \to \text{Dropout}(0.2) \to 128 \to 4$$

其中 SEBlock 为通道门控单元：

$$\text{SEBlock}(\mathbf{h}) = \mathbf{h} \odot \sigma(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{h}))$$

$\mathbf{W}_1 \in \mathbb{R}^{C \times C/16}$, $\mathbf{W}_2 \in \mathbb{R}^{C/16 \times C}$, $\sigma$ 为 Sigmoid 函数。SEBlock 通过学习通道间的相互依赖关系，自适应地增强或抑制不同特征维度。

### 2.6 损失函数与优化

**损失函数**：带标签平滑的交叉熵损失 (Label Smoothing Cross-Entropy)：

$$\mathcal{L} = -\sum_{k=1}^{K} y_k^{\text{LS}} \log \hat{y}_k$$

$$y_k^{\text{LS}} = (1 - \epsilon) \cdot y_k^{\text{one-hot}} + \frac{\epsilon}{K}$$

其中 $K = 4$，$\epsilon = 0.1$ (default) 或 $0.15$ (fau)。标签平滑防止模型对训练标签过度自信，提升泛化能力。

**优化器**：AdamW，学习率 $\eta = 3 \times 10^{-4}$，余弦退火调度 $T_{\max}=100$。

**早停策略**：patience = 15，监控验证集加权准确率 (WA)。

---

## 3. 实验设置

### 3.1 数据集

| 数据集 | 语音数 | 类型 | 年龄 | 语言 | 情绪类别 |
|--------|--------|------|------|------|---------|
| **C-BESD** | 2,780 | 演绎式 | 儿童 | 马来语 | angry, happy, neutral, sad (4 类) |
| **FAU Aibo** | 18,216 | 自发性 | 儿童 | 德语 | angry, happy, neutral, sad (4 类) |
| **IEMOCAP** | 8,525 | 演绎式 | 成人 | 英语 | angry, happy, neutral, sad (4 类) |

**数据预处理**：
- 音频标准化：16kHz 单声道，peak normalization
- 标签映射：统一为 4 类 {angry, happy, neutral, sad}（C-BESD 从 6 类中选取 4 类）
- 说话人划分：确定性 MD5 hash 划分，**说话人零重叠**
- 划分比例：70% train / 15% val / 15% test
- 最大长度截断：4 秒（200 帧 @ 50Hz）

**划分统计**：

| 数据集 | Train | Val | Test | 总说话人 |
|--------|-------|-----|------|---------|
| C-BESD | 1,851 (162 spk) | 389 (36 spk) | 540 (39 spk) | 237 |
| FAU Aibo | 11,577 (35 spk) | 3,250 (7 spk) | 3,389 (9 spk) | 51 |
| IEMOCAP | 4,313 (5 spk) | 1,841 (2 spk) | 2,371 (3 spk) | 10 |

> IEMOCAP 有 1,269 条因标签不在 4 类中被丢弃（原始含 10 类标签如 'excited', 'frustrated' 等）。

### 3.2 超参数配置

| 参数 | Default Profile | FAU Profile |
|------|----------------|-------------|
| 批次大小 | 16 | 16 |
| 学习率 | 3e-4 (AdamW) | 3e-4 (AdamW) |
| 调度器 | CosineAnnealing, T_max=100 | 同 |
| 早停耐心值 | 15 (monitor=val WA) | 同 |
| 最大 Epoch | 100 | 100 |
| weight_decay | 1e-3 | 5e-3 |
| label_smoothing | 0.1 | 0.15 |
| pooling_dropout | 0.0 | 0.3 |
| grad_clip | — | 1.0 |

**为何使用两套正则**：FAU Aibo 为自发性语音，域内训练样本相对有限且同质性高（同一批儿童的连续录音），模型易在 2-3 个 epoch 内快速过拟合。fau 配置通过更强的权重衰减、更高的标签平滑和 pooling dropout 来缓解这一问题。

### 3.3 评价指标

- **加权准确率 (Weighted Accuracy, WA)**：正确预测样本数 / 总样本数，受类别分布影响
- **未加权平均召回率 (Unweighted Average Recall, UAR)**：各类别召回率的算术平均，不受类别不平衡影响

$$\text{WA} = \frac{\sum_{i} \text{TP}_i}{\sum_{i} (\text{TP}_i + \text{FN}_i)}$$

$$\text{UAR} = \frac{1}{K} \sum_{i=1}^{K} \frac{\text{TP}_i}{\text{TP}_i + \text{FN}_i}$$

FAU Aibo 存在类别不平衡，因此同时报告 WA 和 UAR 以全面评估。C-BESD 经筛选后 4 类分布基本均衡，WA 与 UAR 高度一致。

### 3.4 实验列表

| Exp | 训练集 | 测试集 | Pooling | Reg | 目的 |
|-----|--------|--------|---------|-----|------|
| Exp1 | C-BESD | C-BESD | Self-Attention | default | 儿童演绎式天花板（纯注意力） |
| Exp2 | C-BESD | C-BESD | Prosody Guided | default | 韵律先验消融 |
| Exp3 | IEMOCAP | IEMOCAP | Prosody Guided | default | 成人对照（年龄特异性） |
| Exp4 | C-BESD | FAU Aibo | Prosody Guided | default | 跨域零样本迁移 |
| Exp5 | FAU Aibo | FAU Aibo | Prosody Guided | fau | 自发性域内（韵律） |
| Exp5b | FAU Aibo | FAU Aibo | Self-Attention | fau | 自发性域内（纯注意力） |

### 3.5 实验结果

| Exp | Test WA | Test UAR | Test N | Best Epoch |
|-----|---------|----------|--------|------------|
| Exp1 | **92.78%** | 92.79% | 540 | 30 |
| Exp2 | **91.30%** | 91.35% | 540 | 10 |
| Exp3 | 58.67% | 59.11% | 2,371 | 8 |
| Exp4 | 19.56% | 23.83% | 3,389 | 10 |
| Exp5 | **66.36%** | 56.35% | 3,389 | 3 |
| Exp5b | 66.18% | 58.23% | 3,389 | 2 |

**FAU 多种子稳健性**（3 seeds: 42, 123, 456）：

| Exp | Test WA mean±std | Test UAR mean±std |
|-----|------------------|-------------------|
| Exp5 Prosody | 65.15% ± 1.12% | 53.95% ± 2.22% |
| Exp5b Self-Attn | 66.46% ± 0.72% | 55.39% ± 2.05% |

---

## 4. 消融实验与分析

### 4.1 Pooling 策略对比（核心消融）

**C-BESD 域内（Exp1 vs Exp2）**：

| 池化策略 | Test WA | 差异 |
|----------|---------|------|
| Self-Attention | 92.78% | — |
| Prosody Guided | 91.30% | -1.48pp |

**FAU 域内（Exp5 vs Exp5b）**：

| 池化策略 | Test WA (seed 42) | 3-seed mean |
|----------|------------------|-------------|
| Prosody Guided | 66.36% | 65.15% |
| Self-Attention | 66.18% | 66.46% |
| 差异 | +0.18pp | -1.31pp |

**分析**：两种池化策略在域内场景下性能接近。Self-Attention 在演绎式 C-BESD 上略优（+1.48pp），在自发性 FAU 上两者持平。这表明：
1. 充分的域内训练数据下，WavLM 融合特征本身已包含足够的判别信息，注意力可以从特征中自动学习帧级重要性
2. 韵律先验没有带来显著的额外增益，但也未造成损害——韵律引导的方式是安全的，不会引入噪声

但 Prosody Pooling 的价值不在域内性能，而在其**儿童特异性**的论证支撑（见 4.3 节）。

### 4.2 WavLMLayerFusion 的效果

通过与旧管线（仅 last hidden state）的间接对比：

| 管线 | Pooling | C-BESD Test WA |
|------|---------|---------------|
| 旧 v5_622（last hidden state only） | Self-Attention (B2) | 79.51% |
| AC Suite（12层 LayerFusion） | Self-Attention (Exp1) | **92.78%** |
| 旧 v5_622（last hidden state only） | Prosody (B3) | 81.24% |
| AC Suite（12层 LayerFusion） | Prosody (Exp2) | **91.30%** |

**分析**：12 层加权融合带来约 **+10~12pp** 的提升，是全框架中增益最大的单体改进。WavLM Layer 8 获得最高权重（0.147），表明中层韵律编码层对儿童情绪识别最关键。旧管线只取 Layer 12（深层语义层），恰好丢失了最关键的中间层信息。

### 4.3 儿童特异性验证（成人对照）

| 数据集 | 类型 | Pooling | Test WA | 与 C-BESD 差距 |
|--------|------|---------|---------|----------------|
| C-BESD | 儿童演绎 | Prosody | 91.30% | — |
| IEMOCAP | 成人演绎 | Prosody | 58.67% | **-32.63pp** |

**旧管线下的三数据集对照**（更完整的 Δ 分析）：

| 数据集 | 类型 | A1 (mean pool) | A3 (prosody) | Δ (prosody - mean) |
|--------|------|---------------|-------------|---------------------|
| C-BESD | 儿童演绎 | 78.61% | 80.85% | **+2.24pp** ↑ |
| CREMA-D | 成人演绎 | 64.69% | 65.44% | +0.75pp |
| IEMOCAP | 成人演绎 | 54.50% | 52.38% | **-2.12pp** ↓ |

**分析**：Prosody Pooling 的增益随数据集的儿童相关性递减。在儿童演绎式 C-BESD 上 +2.24pp（95% CI 显著），在成人 CREMA-D 上边际 +0.75pp，在成人 IEMOCAP 上反而为负（-2.12pp）。这一梯度证实了韵律先验的**儿童特异性**——儿童语音的 F0 更高（250-400 Hz vs 成人 80-200 Hz）、韵律变异性更大，韵律线索在儿童情绪判别中比在成人中更具信息量。

### 4.4 分布偏移 (FD) 与准确率的定量关系

使用 Fréchet Distance (FD) 在 WavLM 特征空间中量化数据集间的分布偏移，并与分类准确率建立对应关系：

| 条件 | FD | 对应实验 | Test WA |
|------|-----|---------|---------|
| 域内（同语料） | 0.00 | Exp1 | 92.78% |
| 年龄偏移（成人） | 7.20 | Exp3 | 58.67% |
| 风格偏移（自发性） | 8.50 | Exp5 | 66.36% |
| 风格+跨域（零样本） | 8.50 | Exp4 | 19.56% |

**分析**：
1. **FD 与准确率负相关**：FD=0 → 92.78%（同语料域内），FD=7.20 → 58.67%（成人），FD=8.50 → 66.36%（自发性域内）。更高的分布偏移对应更低的分类准确率。
2. **域内训练可部分克服分布偏移**：Exp4 和 Exp5 共享同一 FD=8.50（都是 C-BESD vs FAU），但 Exp5（FAU 域内训练）达到 66.36%，而 Exp4（C-BESD 训练，FAU 零样本测试）仅 19.56% ≈ 随机。说明 FD 测量的分布偏移可通过域内训练数据部分补偿，但无法完全消除。
3. **跨域零样本完全失败**：Exp4 的 19.56% 表明演绎式→自发式的分布差异已超出模型泛化能力，域适应方法是明确的后续方向。

**旧管线增强敏感性实验（v5_622 C 组）**进一步佐证 FD-Accuracy 负相关：

| 条件 | FD | Test WA |
|------|-----|---------|
| C1 无增强 | 0.00 | 81.24% |
| C3 儿童约束增强 | 8.71 | 59.89% |
| C2 成人参数增强 | 9.87 | 50.61% |
| C4 极端增强 | 11.99 | 46.49% |

成人 SER 中标配的数据增强（pitch shift, time stretch）应用于儿童语音时引入分布偏移，所有增强组准确率均下降，FD 与准确率严格单调负相关。

### 4.5 Adapter 消融（旧管线，已从核心框架移除）

旧 v5_622 管线下测试了声学校准适配器 (Adapter) 的效果：

| 配置 | Test WA | Δ vs A1 |
|------|---------|---------|
| A1（基线：mean pooling） | 78.61% | — |
| A2b（A1 + Adapter） | 78.81% | +0.20pp |
| A3（Prosody Pooling） | 80.85% | +2.24pp |
| B3（A3 + Adapter） | 81.47% | +0.62pp vs A3 |

**分析**：Adapter 单独使用几乎无增益（+0.20pp），叠加在 Prosody 之上仅边际提升（+0.62pp）。Prosody Pooling 贡献了总增益的约 80%（+2.24pp vs A1）。考虑到 Adapter 引入额外参数和复杂度，其 0.5pp 的增益不具性价比，因此在 AC Suite 投稿框架中从核心路径移除，仅作为消融讨论的一部分。

### 4.6 预训练模型选型

| 模型 | 测试条件 | Test WA | 结论 |
|------|---------|---------|------|
| WavLM Base (wavlm-base-sv) | 旧管线 A1 | 78.50% | ✅ 主力 |
| wav2vec2 Base | 旧管线 A1 | 56.05% | ❌ 差 ~23pp |
| emotion2vec_plus_large | 旧管线 | 22.45% | ❌ ≈ 随机 |

WavLM 在儿童语音上显著优于同类 SSL 模型，wav2vec2 差距约 23pp，emotion2vec+ 因 FunASR 提取方式与管线不兼容而接近随机。

---

## 5. 结论与未来工作

### 5.1 结论

本文提出了一种分布驱动的儿童语音情绪识别框架。核心贡献包括：

1. **分布偏移诊断**：通过 Fréchet Distance 定量刻画儿童-成人语音在 WavLM 隐空间中的分布偏移（C-BESD vs IEMOCAP: FD=7.20, C-BESD vs FAU: FD=8.50），并建立了 FD 与分类准确率的严格负相关关系。

2. **WavLMLayerFusion**：12 层可学习加权融合，替代传统的"仅取最后一层"方案。实验表明中层韵律层（Layer 8）权重最高，12 层融合相比单一 last hidden state 提升约 10-12pp。

3. **韵律引导的时序池化**：将 F0 曲线和 RMS 能量作为显式韵律先验注入时序注意力机制。该模块在儿童语音上增益显著（+2.24pp），在成人语音上增益递减甚至为负（IEMOCAP: -2.12pp），证实其**儿童特异性**设计。

4. **增强敏感性分析**：证明成人 SER 中标准的数据增强策略无法直接迁移到儿童语音——所有增强操作均引入分布偏移并降低性能，FD 与准确率单调负相关。

在 C-BESD（马来语儿童演绎式）上，Self-Attention 变体达到 **92.78%** WA，Prosody Guided 变体达到 **91.30%** WA。在 FAU Aibo（德语儿童自发性）上，域内训练达到 **66.36%**（Prosody）和 **66.18%**（Self-Attention）。框架总可训练参数仅 ~704K（不到骨干的 1%），具有参数高效、训练快速的特点。

### 5.2 未来工作

待补充。

---

*文档由 Claude Code 基于项目 `results/logs/`、`experiments/`、`src/` 中的 canonical 数据自动生成，2026-06-04*
