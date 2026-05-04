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

## 三、修正后的研究路线（对应顶会投稿要求）

> 对标 INTERSPEECH/ICASSP 顶会标准。每个 Phase 标注了对应解决的顶会审稿人关注点。

### Phase 6.1：建立外部基准 ⇢ 解决"单数据集""无可比 SOTA"问题

#### 6.1a MFCC+CNN 基线 ✅
- **结果**: Test WA=50.2%, UAR=50.3% (严格 6:2:2)
- **发现**: C-BESD 已发表 76% 可能非 speaker-independent（26pp 差距）
- **论文价值**: 我们的 A3 比严格基线高 30pp，比已发表高 5pp

#### 6.1b IEMOCAP 成人 A1 vs A3 — 🔄 云端执行中
- **目的**: 验证"儿童更需要韵律引导"的核心论点
- **硬件**: 云端 RTX 4090D (24GB)，本地 RTX 3060 太慢
- **预计耗时**: 云端 ~30min

#### 6.1c 多 SSL backbone 对比
- WavLM vs HuBERT vs wav2vec2 on C-BESD
- **目的**: (1) 论证 WavLM 选择合理性 (2) 提供额外外部基准点
- **硬件**: 云端 RTX 4090D

### Phase 6.2：修复技术债 ⇢ 消除已知 limitation

- Padding mask 支持
- 修正 FD 计算 bug（`compute_adapter_init.py`）
- 补全 FD_age、FD_lang

### Phase 6.3：可解释性分析 ⇢ 提升 scientific depth

- Per-class attention 可视化
- Attention-F0 correlation 量化
- 儿童 vs 成人 attention 模式对比

### Phase 6.4：统一 FD 框架 ⇢ 论文核心记忆点

- 三维 FD-Accuracy 图（年龄 / 增强 / 语言）
- FD 作为"分布偏移风险指标"

### Phase 6.5：论文 v7

```
实验章节（顶会标准）:
4.1 数据集与实验设置
4.2 C-BESD 主结果（WA+UAR, mean±std）
    - A3 = 80.85% / 80.8% (WA/UAR)
    - vs MFCC 严格基线 50.2% (+30pp)
    - vs 已发表基线 76% (+5pp)
4.3 消融与模块贡献
    - A1→A3→B3 贡献链
    - Adapter 仅 +0.5pp, Prosody +2.24pp
4.4 成人对比 (IEMOCAP)
4.5 多 backbone 对比 (WavLM/HuBERT/wav2vec2)
4.6 增强敏感性分析 (C1-C4)
4.7 跨语言迁移 (X1/X2)
4.8 统一 FD-Accuracy 框架
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

---

## 执行日志

### Phase 6.1a：复现 C-BESD 基线 (MFCC+CNN) — ✅ 完成

- **结果**: Test WA=50.18%, UAR=50.29% (严格 6:2:2 speaker-independent)
- **vs 已发表**: Rao et al. (IEEE INSPECT 2024) 报告 76%，但未说明 split 方法
- **关键发现**: 26pp 差距可能来自 split 方法差异（非 speaker-independent），类似旧 DrseCNN 论文的 86%→35% 泄露问题
- **论文价值**: 我们的 A3 (80.85%) 相比**严格基线**高出 30pp，论证更有力
- **数据保存**: `data/mfcc_baseline/` (特征), `experiments/v5_622/mfcc_baseline_result.json` (结果)

### Phase 6.1b：IEMOCAP A1 vs A3 — 🔄 云端运行中

- **脚本**: `scripts/run_iemocap_contrast.py`
- **数据**: IEMOCAP 9903 样本已上传至 `/root/autodl-tmp/IEMOCAP/wavs/`
- **硬件**: 云端 RTX 4090D (24GB)，~8 it/s WavLM 提取
- **预计耗时**: ~25-30 分钟

### Phase 6.1c：多 SSL backbone 对比 — 待开始
