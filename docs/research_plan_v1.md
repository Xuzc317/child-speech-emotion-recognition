# 研究路线规划：可解释性 + 分布偏移理论

> v1 | 2026-05-04 | 目标：语音顶会 (INTERSPEECH / ICASSP / ASRU)
> 分支：`research/interpretability-fd`
> 核心假设：儿童语音的情绪信息在时间维度上非均匀分布，且这种分布模式对多维度偏移（年龄/增强/语言）敏感

---

## 一、当前基线

| 指标 | 数值 |
|------|------|
| 模型 | A3 = WavLM frozen + Prosody Pooling + DrseNet |
| 6:2:2 Test | 80.85% ± 0.60% |
| vs 旧基线 (162-dim) | +45pp |
| 核心增益来源 | Prosody Pooling: +2.24pp (A1→A3) |
| 可训练参数 | ~600K |

**已知问题**：
- Padding mask 未使用（~40% 帧为零填充，参与 attention softmax）
- FD 计算在 `compute_adapter_init.py` 中有 bug（仅均值差，缺协方差项）
- 零填充帧参与 pooling 计算

---

## 二、两条研究路线

### 路径 A：可解释性深度挖掘

**核心问题**：Prosody Pooling 为什么有效？它学到了什么？

**假说**：
- H1：不同情绪类别的 attention 权重在时间轴上呈现不同的分布模式（ANGER 集中在高能量帧，SAD 集中在低 F0 帧）
- H2：儿童语音的情绪韵律模式与成人存在系统性差异（儿童对同一情绪使用不同的时间-韵律策略）
- H3：跨语言（英语 vs 泰卢固语）的韵律-情绪映射模式差异可解释跨语言迁移失败

#### A.1 Attention 权重可视化（1-2 天）

```
输入：A3 模型 + 测试集样本
输出：
  ├── per-class attention heatmap（6 类情绪 × 200 帧）
  ├── attention weight vs F0 contour 叠加图
  └── 高注意力帧的统计特征（F0 mean/std, energy mean/std）
```

**方法**：
1. 对测试集中每类情绪随机采样 50 个样本
2. 提取 `TemporalImportancePooling` 中 softmax 前的 attention logits
3. 按情绪类别聚合，绘制平均 attention 分布曲线
4. 与 F0 contour 和 energy contour 叠加对比
5. 统计高权重帧 (top 20%) 的声学特征

**预期发现**：
- ANGER: attention 集中在 utterance 中后段，与高能量 burst 对齐
- SAD: attention 偏向 utterance 中段，F0 低谷区域
- NEUTRAL: attention 相对均匀分布
- 这将直接支撑论文核心论点："儿童语音的情绪信息在时间维度上非均匀分布"

#### A.2 跨情绪 attention 差异性量化（1-2 天）

**指标设计**：
- Attention entropy：衡量 attention 分布的集中度
- Attention skewness：衡量 attention 偏向前/后段的程度
- Attention-F0 correlation：attention 权重与 F0 的逐帧相关系数

**论证逻辑**：
```
高 entropy (+ low corr) → 情绪信息均匀 → mean pooling 足够 → NEUTRAL
低 entropy (+ high corr) → 情绪信息集中 → Prosody Pooling 关键 → ANGER/SAD
```

如果 per-class 指标与 per-class accuracy 改善呈正相关，则为 Prosody Pooling 提供了强有力的可解释性证据。

#### A.3 成人对比实验（2-3 天）

**实验**：在 IEMOCAP 成人数据上分别跑 A1 (mean pool) 和 A3 (prosody pool)

**预期**：成人的 Prosody Pooling 增益远小于儿童 (<1pp vs 2.24pp)

**论证**：
```
儿童 A1→A3: +2.24pp  ← 儿童更需要韵律引导
成人 A1→A3: +X.XXpp  ← 成人语音的情感信息在时间上更均匀？
```

如果差值显著，则证明"儿童语音的情绪韵律模式与成人存在根本性差异"——这比"我们提了一个更好的 pooling 方法"有更高的科研贡献度。

#### A.4 跨语言 attention 模式对比（2-3 天，可选）

**实验**：分别在英文和泰卢固语测试集上提取 attention 分布

**分析**：同一情绪类别的 attention 模式在两种语言间是否一致？

**预期**：语言间的 attention 模式差异 > 语言内差异，解释为什么跨语言迁移失效

---

### 路径 C：统一分布偏移理论框架

**核心问题**：能否用统一的数学框架（FD）定量预测多维度分布偏移下的分类退化？

**假说**：
- H4：FD 可统一量化三种偏移维度（年龄/增强/语言），且 FD 值与分类退化程度呈严格单调关系
- H5：不同偏移维度的 FD-Accuracy 曲线具有不同的斜率，语言的斜率 > 增强 > 年龄

#### C.1 修正 FD 计算 + 补全数据（1 天）

**Bug 修复**：`compute_adapter_init.py` 第 132-134 行的 FD 计算仅用了均值差平方范数：
```
当前（错误）：fd = diff_mean @ diff_mean
正确：     fd = diff @ diff + trace(sigma1 + sigma2 - 2 * sqrtm(sigma1 @ sigma2))
```

正确的实现在 `constrained_aug.py` 中已有。需要重新计算并更新论文第三章中的 FD=0.37 数值。

**补全数据**：计算以下维度的 FD
| 偏移维度 | 分布 A | 分布 B | 预期 FD |
|---------|--------|--------|---------|
| 年龄 (FD_age) | IEMOCAP (成人) | BESD MY (儿童) | ~X.XX |
| 增强 (FD_aug) | C1 (clean) | C2/C3/C4 | 0 / 8.71 / 9.87 / 11.99 |
| 语言 (FD_lang) | BESD ENGLISH | BESD TELUGU | ~??? |

#### C.2 统一 FD-Accuracy 图（1-2 天）

**目标**：将三条 FD-Accuracy 曲线放在同一张图里

```
Y 轴：Test Accuracy
X 轴：FD (log scale)

图中三组数据：
  年龄偏移线：A1 (FD=0?) → A3 (FD=0.37?)
  增强偏移线：C1(0, 80%) → C3(8.71, 59%) → C2(9.87, 52%) → C4(11.99, 47%)
  语言偏移线：MY-mixed(0, 81%) → X2(??, 28%) → X1(??, 20%)
```

**论证链**：
1. FD ∝ 1/Accuracy 在三个维度上均成立
2. 语言偏移的 FD 远大于增强偏移，解释了为什么跨语言迁移最难
3. 年龄偏移的 FD 最小，解释了为什么 A3 在同语言内表现好
4. FD 可以作为儿童 SER 的"分布偏移风险指标"

**这是全文最有冲击力的图表**。

#### C.3 FD 作为预测工具（可选，1 周）

**实验**：给定一个新的偏移来源（如录音设备、噪声环境），能否通过 FD 预测其造成的性能下降？

---

## 三、实施计划

| 阶段 | 内容 | 预计时间 | 优先级 |
|------|------|---------|--------|
| Phase 6.1 | 修复 padding mask + 跑实验 | 1 天 | ⭐⭐⭐ |
| Phase 6.2 | 修正 FD 计算 + 补全 FD 数据 | 1 天 | ⭐⭐⭐ |
| Phase 6.3 | Attention 可视化 + per-class 分析 | 2 天 | ⭐⭐⭐ |
| Phase 6.4 | 成人对比实验 (IEMOCAP) | 2 天 | ⭐⭐ |
| Phase 6.5 | 统一 FD-Accuracy 图 | 1 天 | ⭐⭐⭐ |
| Phase 6.6 | 跨语言 attention 对比 | 2 天 | ⭐ |
| Phase 6.7 | 论文 v7 撰写 | 1 周 | ⭐⭐⭐ |

---

## 四、目标会议

| 会议 | 截稿 | 特点 |
|------|------|------|
| INTERSPEECH 2026 | ~2026-03 (已过) | 语音顶会 |
| ASRU 2026 | ~2026-06 | 语音识别与理解 |
| ICASSP 2027 | ~2026-10 | 信号处理顶会 |
| SLT 2026 | ~2026-07 | 口语语言技术 |

**建议首选 INTERSPEECH 2027 或 ICASSP 2027**，时间充裕。

---

## 五、预期贡献（论文叙事升级）

```
当前（v6）：
  "我们提了一个韵律引导的池化方法，在儿童 SER 上 +2.24pp"

目标（v7）：
  "我们发现儿童语音的情绪信息在时间维度上呈现非均匀分布，
   且这种分布模式对不同维度的偏移（年龄/增强/语言）敏感。
   我们提出了 (1) 韵律引导池化来利用这种非均匀性，
   (2) 统一的 FD 框架来量化多维度偏移的影响。
   三个维度的 FD-Accuracy 单调关系构成了儿童 SER 的
   分布偏移理论框架。"
```

---

## 六、待办

- [ ] 修复 padding mask（pooling.py + train_ssl.py）
- [ ] 修正 FD 计算（compute_adapter_init.py）
- [ ] 计算 FD_age, FD_lang
- [ ] Attention 可视化脚本
- [ ] 成人对比实验（IEMOCAP A1 vs A3）
- [ ] 统一 FD-Accuracy 图
- [ ] 论文 v7

---

> 本文档是**活文档**，随着探索推进持续更新。每次显著进展后更新本文件并 commit。
