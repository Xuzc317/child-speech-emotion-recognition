# AI 项目理解提示词

复制下方提示词给 AI，让它系统遍历项目文件，完全理解最新最权威的信息。

---

```
你是一个学术论文协作助手。请按以下步骤系统阅读本项目，每步完成后向我简要汇报你理解了什么。

## 项目背景

这是一个儿童语音情绪识别 (SER) 研究项目。核心思路：从儿童语音的统计分布出发，构建分布偏移诊断框架 (FD-WA)，验证"分布偏移→性能下降"的因果关系。

三数据集：C-BESD (儿童演绎)、FAU Aibo (儿童自然)、IEMOCAP (成人对照)。
主干模型：WavLM Base → 12层 LayerFusion → Pooling → SEMLP 分类器。
实验矩阵：B1-B7 共 192 个实验，全部完成。

## 阅读步骤

### 第一步：全局理解

阅读以下文件，理解项目架构和实验结果全貌：

1. `CLAUDE.md` — 项目架构、192实验矩阵总览、各 Phase 关键数值和结论、目录结构
2. `README.md` — 项目概览、核心结果速览

### 第二步：实验设计

阅读以下文件，理解为什么要这样设计实验、每个 Phase 解决了什么问题：

3. `docs/current/实验设计方案_v3_含学习笔记.md` — 实验设计逻辑、动机、假设
4. `docs/current/权威数据手册.md` — B1-B7 全量 192 runs 每实验数值表格和关键结论

### 第三步：方法细节

阅读以下文件，理解每个模块的技术实现：

5. `src/README.md` — 代码架构总览
6. `docs/current/模块1_数据管道.md` — 数据预处理、说话人划分、标签映射
7. `docs/current/模块2_WavLM主干网络.md` — SSL backbone 选择和加载
8. `docs/current/模块3_注意力池化.md` — Self-Attention / Prosody Guided / Mean Pooling
9. `docs/current/模块5_可解释性可视化.md` — XAI：Attention-Prosody Correlation
10. `docs/current/模块6_分布偏移诊断.md` — FD (Fréchet Distance) + SMMD 双指标

### 第四步：论文主体

阅读当前论文稿，理解叙事结构和论证逻辑：

11. `paper_draft/current/main.tex` — LaTeX 入口
12. `paper_draft/current/0_Abstract.tex` 到 `6_Conclusion.tex` — 各章节内容
13. `paper_draft/current/references.bib` — 参考文献

### 第五步：实验数据和插图

遍历以下目录，确认有哪些数据和图可用：

14. `results/logs/` — 192 个 JSON 文件（E1-01_s42.json 到 E7-06_s456.json），每个含 test_wa、test_uar、best_epoch。抽样读取 3-5 个不同 Phase 的文件理解 JSON 结构
15. `results/analysis/` — FD/SMMD (canonical_fd_pairs.json)、XAI 原始数据 (xai_raw_data.npz)、层权重 (layer_weights.json)
16. `paper_draft/figures/` — 论文配图（fig01-fig08, figA1-figA4）和 t-SNE 图（fig_tsne_*.png）。读取 FIGURES_MANIFEST.json 了解配图清单

### 第六步：补充实验计划

17. `docs/current/补充实验方案_v1.md` — 待完成的混淆矩阵、XAI 更新、层权重图计划
18. `results/figures/` — 补充实验的输出（混淆矩阵等，如果已生成）

### 第七步：代码深度理解

抽样阅读核心代码文件，理解实现细节：

19. `src/models/pooling.py` — 三种 Pooling 的具体实现
20. `src/models/layer_fusion.py` — 12 层可学习加权求和
21. `src/models/semlp.py` — 分类器结构
22. `src/data/speaker_splitter.py` — 说话人独立划分的逻辑
23. `src/evaluation/distribution_metrics.py` — FD/SMMD 计算
24. `src/train.py` — 训练主流程和 SERModel 类

## 重要警告

以下是旧数据/不完整数据，不要作为项目理解的依据：

- `_legacy/` 目录下全部内容 — 已废弃的旧协议数据、无关项目
- `experiments/` 目录 — Phase 3-5 旧协议实验，不可与当前数据对比
- `results/archive/` — 旧实验 exp1-exp5b 的混淆矩阵和 JSON，来自旧协议
- `docs/archive/` — 已归档的历史文档
- 任何以 exp1/exp2/exp3/exp4/exp5/exp5b 命名的文件 — 早期 AC Suite 的 6 个实验，已被 B1-B7 覆盖
- `results_remote/` — 旧同步目录，数据已整合到 results/
- CLAUDE.md 中 B6 和 B7 标注为"FAU Aibo"有误，实际实验在 C-BESD (6类) 上运行；以权威数据手册和 JSON 文件中的 num_classes 为准

## 输出要求

完成全部七步阅读后，请用以下格式向我汇报：

1. **项目一句话总结**：这个项目做了什么、核心发现是什么
2. **架构总结**：模型长什么样、数据怎么流
3. **实验全景**：B1-B7 各 Phase 目的 + 最关键的一个数值
4. **可用的图和表**：论文已有哪些配图、还缺什么
5. **可以改进的地方**：你注意到的任何不一致、缺失或有疑问的地方
```
