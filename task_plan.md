# 任务计划：论文与PPT数据同步更新

## 目标
将权威数据手册 (`docs/权威数据手册.md`) 中的真实实验数据同步到 `paper_draft/*.tex` 和 `paper_draft/儿童SER_项目汇报_v3.pptx`，修复已知错误，标记仍缺失的实验。

## 当前阶段
Phase 1 — 论文 .tex 文件更新（进行中）

## 阶段

### Phase 1: 修复已知错误
- [ ] **1.1** `0_Abstract.tex`: 修复韵律在成人语音上的效应方向（原文说 "-~2pp"，实际是 SA >> PG，差+9.73pp，韵律有害）
- [ ] **1.2** `0_Abstract.tex`: 更新 C-BESD Δ(SA-PG) 为 -0.13pp (n.s.)，非原文的 +2pp
- [ ] **1.3** `4_Experiments_and_Results.tex`: E1-08 IEMOCAP Self-Attn 模拟值 76.0±8.0 → 真实 75.96%±7.94%
- [ ] **1.4** `paper_draft/儿童SER_项目汇报_v3.pptx`: 修复 Slide 5 "Exp4" 标签错误（Exp4 是零样本FAU，不是 IEMOCAP SA）

### Phase 2: 更新 E1 表格 (In-Domain Pooling Baseline)
- [ ] **2.1** 替换 C-BESD Self-Attn: 93.0±1.5 → 94.97%±1.66% (3-seed)
- [ ] **2.2** 替换 C-BESD Prosody: 91.0±2.5 → 95.10%±2.78% (3-seed)
- [ ] **2.3** 替换 IEMOCAP Prosody: 66.0±7.5 → 66.23%±7.56% (2-seed)
- [ ] **2.4** 替换 FAU Self-Attn: 66.5±0.8 → 66.46%±0.72% (3-seed)
- [ ] **2.5** 替换 FAU Prosody: 65.0±1.2 → 65.15%±1.12% (3-seed)
- [ ] **2.6** 标记 E1-01 (C-BESD Mean)、E1-04 (FAU Mean)、E1-07 (IEMOCAP Mean) 仍缺失 [待B1实验]
- [ ] **2.7** IEMOCAP Self-Attn 标记为 2-seed [仅2种子有效]

### Phase 3: 更新 E3 零样本矩阵
- [ ] **3.1** 将模拟值替换为权威数据手册 §1.3 的 6 方向完整矩阵
- [ ] **3.2** 移除 FD 列的旧模拟值，标注 FD 来源为 canonical_fd_pairs.json
- [ ] **3.3** 更新 E3 key observation 中的 Spearman ρ 描述

### Phase 4: 更新 E4、E5、E6、E7
- [ ] **4.1** E4 数据增强表：标注当前真实数据来自 v5 6:2:2 协议，非 AC Suite 70/15/15
- [ ] **4.2** E5 层融合消融：标记所有值仍 [待B1实验验证]
- [ ] **4.3** E6 模块消融：用 v5 Run 2 真实数据更新 C-BESD 部分
- [ ] **4.4** E7 模型迁移：标记全部 [待B1实验验证]

### Phase 5: 更新 Analysis & Discussion
- [ ] **5.1** Finding 1: FD 值更新（7.20, 8.50, 16.48 → 16.33）
- [ ] **5.2** Finding 2: Prosody Gap 表用真实多种子数据重写
- [ ] **5.3** Finding 3: 成人数据增强有害性用 v5 C2 真实数据
- [ ] **5.4** Finding 4: Layer 权重更新（argmax=9, entropy=2.484）
- [ ] **5.5** APC 小节：更新 APC_wav=0.7406, APC_delta=-0.0450（540样本全量）
- [ ] **5.6** Limitations：标注 IEMOCAP 仅2种子有效

### Phase 6: PPT 更新
- [ ] **6.1** 修复 Slide 5 Exp4 标签错误
- [ ] **6.2** 更新 Slide 5 为正确的 IEMOCAP 双池化对比（SA 75.96% vs PG 66.23%, 2-seed）
- [ ] **6.3** 如果 B1 实验完成，同步 E1 全量数据到 PPT

### Phase 7: 全局标记清理
- [ ] **7.1** grep 所有 .tex 文件中的 [待实验验证]，确认每个标记的状态
- [ ] **7.2** 已更新的移除 [待实验验证]，仍缺失的保留并标注具体依赖

## 关键决策
1. E1-01/04/07 Mean pooling 基线在 B1 实验完成前保留模拟值
2. E4 数据增强使用 v5 真实数据但标注协议差异
3. E5 层融合消融全部保留 [待B1实验验证]（AC Suite 无对应实验）
4. IEMOCAP SA 标注为 2-seed（s456 val=0 不可用）
