# 研究路线规划 v2：基于文献调研的重新定位

> 2026-05-04 | 目标：语音顶会 (INTERSPEECH 2027 / ICASSP 2027)
> 分支：`research/interpretability-fd`

---

## 一、文献调研：儿童 SER 的真实研究格局

### 1.1 现状：一个真实的 Research Gap

儿童语音情绪识别 + SSL 预训练模型（WavLM/HuBERT）的交叉领域，在 2024-2025 年几乎**无人涉足**：

| 研究方向 | 状态 |
|---------|------|
| SSL 模型 → 成人 SER | 成熟（WavLM/HuBERT 已成标配） |
| SSL 模型 → 儿童 ASR | 有系统 benchmark（INTERSPEECH 2024） |
| 儿童 SER | 停滞在手工特征 + CNN 阶段 |
| **SSL 模型 → 儿童 SER** | **基本空白** |

### 1.2 可对标的外部基准

| 来源 | 数据集 | 方法 | 结果 |
|------|--------|------|------|
| Rao et al. (IEEE INSPECT 2024) | **C-BESD** | MFCC+ZCR+Pitch → CNN | **76%** |
| Lesyk et al. (HCIS 2024) | FAU-AIBO | Wav2Vec adult→child transfer | F-score 0.47 |
| Preethi et al. (IJISAE 2024) | Zenodo kids | SimCLRV2 semi-supervised | 无公开数值 |
| InterSpeech 2024 (EmoBox) | 多语料 benchmark | 含儿童语音评估 | — |

**关键发现**：
- C-BESD 论文是 2024 年 12 月刚发表的，只有一组 MFCC+CNN 基线（76%）
- 我们的 A3 模型（WavLM + Prosody Pooling + SEMLP）**80.85% 已经超过该基线约 5pp**
- 没有任何已发表工作使用 WavLM + 韵律引导池化做儿童 SER
- 没有任何工作系统量化儿童 SER 的多维度分布偏移

### 1.3 成人 SER 的 SOTA 参考（非直接对标）

| 模型 | 数据集 | 结果 | 来源 |
|------|--------|------|------|
| WavLM + fine-tune | IEMOCAP | ~75-78% WA | ICASSP 2024 |
| HuBERT + LoRA | IEMOCAP | ~76% WA | InterSpeech 2024 |
| Whisper + adapter | MELD | ~67% WF1 | ICASSP 2025 |

> 成人 SER 的数值不可直接与儿童 SER 比较（数据集、情绪类别、说话人分布都不同），但它们提供了方法论的参考坐标。

---

## 二、我们的真实位置

### 不再用的叙事（自说自话）
```
❌ "比我们自己的旧基线提升了 45pp（35% → 80.85%）"
   → 旧基线是项目内部的 162-dim 手工特征，不是公开基准
   → 审稿人不关心你的内部迭代历史

❌ "数据泄露修复后从 89% 降到 81%"
   → 这是工程修正，不是科研贡献
```

### 应该用的叙事（对外对标）
```
✅ "在 C-BESD 上达到 80.85%，超过已发表基线（76%）约 5pp"
   → 有外部参照系，可验证

✅ "首次将 SSL 帧级特征 + 韵律引导池化应用于儿童 SER"
   → WavLM for children's SER is underexplored

✅ "提出多维度分布偏移框架（FD-age / FD-aug / FD-lang），
    定量解释儿童 SER 中的三类性能退化"
   → 这是我们的独特贡献
```

---

## 三、修正后的研究路线

### Phase 6.1：建立外部基准（1-2 天）

**目标**：让论文有可引用的外部比较对象

1. **复现 C-BESD 基线**：用 MFCC + CNN 在我们的 6:2:2 split 上跑，验证 76% 是否可复现
2. **成人 SER 参照**：在 IEMOCAP 上用同样的 A3 架构跑一次，作为"成人 vs 儿童"的对比数据
3. **其他 SSL backbone 对比**：WavLM vs HuBERT vs wav2vec2 在 C-BESD 上的对比（论证为什么选 WavLM）

### Phase 6.2：修复技术债（1 天）

- Padding mask 支持（消除 40% 零填充帧干扰）
- 修正 FD 计算 bug（`compute_adapter_init.py`）
- 补全 FD_age（成人 vs 儿童）、FD_lang（英语 vs 泰卢固语）

### Phase 6.3：可解释性分析（2-3 天）

原路径 A：
- Per-class attention 可视化
- Attention-F0 correlation 量化
- 成人 vs 儿童 attention 模式对比（核心假设：儿童更需要韵律引导）

### Phase 6.4：统一 FD 框架（1-2 天）

原路径 C：
- 三维 FD-Accuracy 图（年龄 / 增强 / 语言）
- FD 作为"分布偏移风险指标"的理论框架
- 论证：语言偏移 > 增强偏移 > 年龄偏移

### Phase 6.5：论文 v7（1 周）

按顶会格式重写，核心叙事：
```
1. Introduction
   - 儿童 SER 被忽视，SSL 模型未进入该领域
   - 仅有的 C-BESD 基线是 76%（手工特征+CNN）

2. Related Work
   - 成人 SER 中的 SSL 模型
   - 儿童 ASR 中的 SSL 模型（但 SER 仍是空白）
   - 分布偏移理论（domain shift, FD）

3. Method
   - Prosody-guided Temporal Importance Pooling（核心）
   - SEMLP classifier（一句话）
   
4. Experiments (这是论文的核心差异化章节)
   4.1 C-BESD 主结果：A3 = 80.85% vs 基线 76%
   4.2 消融：A1(mean) → A3(prosody) → B3(+adapter)
   4.3 成人对比：IEMOCAP 上的 Prosody Pooling 增益远小于儿童
   4.4 增强敏感性（负面实验）：FD-Aug vs Accuracy
   4.5 跨语言迁移：FD-Lang 最大，解释为何跨语言几乎失效
   4.6 统一 FD-Accuracy 图（三个维度的单调关系）

5. Conclusion
   - 首次在儿童 SER 中引入 SSL + 韵律池化
   - 提出 FD 框架量化多维度分布偏移
   - 儿童 SER 仍远未解决（跨语言 20%），呼吁更多关注
```

---

## 四、目标会议再评估

| 会议 | 截稿 | 匹配度 |
|------|------|--------|
| **INTERSPEECH 2027** | ~2027-03 | ⭐⭐⭐ 语音领域顶会，儿童语音是 established topic |
| **ICASSP 2027** | ~2026-10 | ⭐⭐⭐ 信号处理顶会，SER 是常驻 track |
| **ASRU 2027** | ~2026-12 | ⭐⭐ 更偏 ASR，SER 不是主 topic |
| **SLT 2026** | ~2026-07 | ⭐⭐ 时间太紧 |

**推荐**：ICASSP 2027（10 月截稿，5 个月时间充裕）或 INTERSPEECH 2027（3 月截稿）。

---

## 补充：SER 领域标准评价指标

SER 顶会论文**以 UAR (Unweighted Average Recall) 为主指标**，WA (Weighted Accuracy) 为辅助。
INTERSPEECH ComParE 挑战赛从 2009 年起只用 UAR 排名。

```python
from sklearn.metrics import recall_score, accuracy_score
uar = recall_score(y_true, y_pred, average='macro')   # 每类等权
wa  = accuracy_score(y_true, y_pred)                   # 即常规 accuracy
```

C-BESD 类别均衡（6 类每类 ~417），WA ≈ UAR，但论文中必须两个都报告。
表格规范写法：`WA (%) / UAR (%)`，差异用 "percentage points" 而非 "pp"。

## 五、待办清单

- [ ] Phase 6.1a：复现 C-BESD 基线（MFCC+CNN on 6:2:2）
- [ ] Phase 6.1b：IEMOCAP 上跑 A1 vs A3 对比
- [ ] Phase 6.1c：多 SSL backbone 对比（WavLM vs HuBERT vs wav2vec2）
- [ ] Phase 6.2a：修复 padding mask
- [ ] Phase 6.2b：修正 FD 计算 + 补全 FD_age, FD_lang
- [ ] Phase 6.3a：Attention 可视化（per-class + per-sample）
- [ ] Phase 6.3b：成人 vs 儿童 attention 模式对比
- [ ] Phase 6.4：统一 FD-Accuracy 三线图
- [ ] Phase 6.5：论文 v7 撰写

---

## 六、关键风险

| 风险 | 缓解措施 |
|------|---------|
| IEMOCAP 的 Prosody Pooling 增益也很大（削弱"儿童特异性"论点） | 可以转而论证"韵律池化对高 F0 变异群体普遍有效"，仍成立 |
| C-BESD 基线 76% 复现不一致 | 用我们自己的 split 重新跑 76% 基线，标注方法论差异 |
| 跨语言 FD 计算因数据量不足不显著 | 提前检查泰卢固语 vs 英语的样本量和特征分布 |
| Padding mask 修复后准确率不升反降 | 分析原因（可能零填充帧提供了"空白上下文"），如实报告 |

---

> **v2 核心变化**：从"跟自己比"转为"跟外部基准比"；从"内部迭代叙事"转为"填补研究空白叙事"。
> 本文档为活文档，随着实验推进持续更新。
