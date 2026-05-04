# 参考文献目录

> 论文 v7 参考文献，已下载 PDF 到本地

---

## 核心对标文献

### [1] WavLM 基础模型
- **标题**: WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing
- **作者**: Chen, S. et al. (Microsoft Research)
- **发表**: IEEE SPL, 2022
- **文件**: `chen2022_wavlm.pdf` (929 KB)
- **下载**: https://arxiv.org/abs/2110.13900

### [2] Frozen SSL + 注意力池化 on IEMOCAP（最可比工作）
- **标题**: Speech-Based Emotion Recognition with Self-Supervised Models Using Attentive Channel-Wise Correlations and Label Smoothing
- **作者**: Kakouros, S. et al.
- **发表**: arXiv:2211.01756, 2022
- **文件**: `kakouros2022_attentive.pdf` (391 KB)
- **下载**: https://arxiv.org/abs/2211.01756
- **关键数据**: Frozen WavLM + mean pool on IEMOCAP 4-class = 69.44%; + attention = 75.60%

### [3] Frozen SSL 嵌入对比 on CREMA-D
- **标题**: A Comparative Study of Pre-trained Speech and Audio Embeddings for Speech Emotion Recognition
- **作者**: Phukan, O. C. et al.
- **发表**: arXiv:2304.11472, 2023
- **文件**: `phukan2023_comparative.pdf` (2,854 KB)
- **下载**: https://arxiv.org/abs/2304.11472
- **关键数据**: 8 种 frozen PTM 嵌入在 CREMA-D 6-class 上的线性探测对比

### [4] 儿童 ASR 基准（SSL 模型对比）
- **标题**: Benchmarking Children's ASR with Supervised and Self-supervised Speech Foundation Models
- **作者**: Fan, R. et al.
- **发表**: INTERSPEECH, 2024
- **文件**: `fan2024_children_asr.pdf` (242 KB)
- **下载**: https://arxiv.org/abs/2406.10507
- **关键数据**: WavLM/HuBERT/wav2vec2 在儿童语音上的系统对比

### [5] SER 测试时自适应（Distribution Shift 热点）
- **标题**: Test-Time Adaptation for Speech Emotion Recognition
- **作者**: Dong, J., Jia, H., Dang, T.
- **发表**: ICASSP, 2026
- **文件**: `dong2026_tta_ser.pdf` (365 KB)
- **下载**: https://arxiv.org/abs/2601.16240
- **关键数据**: 11 种 TTA 方法的系统评估，domain shift 方向

## 待获取文献

### [6] C-BESD 数据集原始论文
- **标题**: Children's Speech Emotion Recognition Using Feature Fusion-Based Deep Neural Networks
- **作者**: Rao, D. V. et al.
- **发表**: IEEE INSPECT, 2024
- **DOI**: 10.1109/INSPECT63485.2024.10896094
- **获取方式**: IEEE Xplore（需机构权限或购买）

### [7] 跨年龄情绪迁移
- **标题**: Empathetic Deep Learning: Transferring Adult Speech Emotion Models to Children With Gender-Specific Adaptations
- **作者**: Lesyk, E. et al.
- **发表**: Human-Centric Intelligent Systems, 2024
- **文件**: `lesyk2024_empathetic.pdf` (19.5 MB)
- **下载**: https://link.springer.com/article/10.1007/s42452-024-05890-6

### [8] Paralinguistic Representations for SER (INTERSPEECH 2024)
- **标题**: Are Paralinguistic Representations all that is needed for Speech Emotion Recognition?
- **作者**: Phukan, O. C. et al.
- **发表**: INTERSPEECH, 2024
- **文件**: `phukan2024_paralinguistic.pdf` (1.3 MB)
- **下载**: https://arxiv.org/abs/2402.01579

---

## 文献使用说明

| 文献 | 用于论文何处 |
|------|------------|
| [1] WavLM | 方法章节 — 骨干网络选择依据 |
| [2] Kakouros | 实验章节 — 对标 frozen SSL + attention on IEMOCAP |
| [3] Phukan | 实验章节 — 对标 frozen SSL on CREMA-D |
| [4] Fan | 相关工作 — 儿童 SSL 研究现状 |
| [5] Dong | 相关工作/讨论 — Distribution shift 研究趋势 |
| [6] Rao | 实验章节 — C-BESD 已发表基线 (76%) |
| [7] Lesyk | 相关工作 — 成人→儿童情绪迁移 |
