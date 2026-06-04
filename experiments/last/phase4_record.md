# Phase 4 补充实验记录

日期: 2026-04-28
云端: RTX 4090D 24GB, SSH connect.cqa1.seetacloud.com:12112

---

## 实验一：跨语言迁移（English ↔ Telugu）

### 目的
验证语言偏移对儿童SER的影响，论证"语言是分布偏移的极端维度"。

### 方法
- 数据集: C-BESD ENGLISH (2,095 WAVs) 和 TELUGU (2,085 WAVs)
- 说话人零重叠: ENGLISH 139人 vs TELUGU 420人
- 配置: WavLM + Adapter (random init) + Prosody Pooling + DrseCNN (B3)
- 3 seeds: 42, 123, 456

### 结果

| 实验 | Val Acc (%) | Test Acc (%) |
|------|------------|-------------|
| X1 English→Telugu | 86.86 ± 0.60 | 19.68 ± 0.45 |
| X2 Telugu→English | 89.03 ± 0.35 | 28.15 ± 0.38 |

对比: MY混合训练 B3 Test = 81.24% ± 1.45%

### 结论
- 语言偏移导致分类几乎完全失效（X1仅比随机高3pp）
- Val-Test 鸿沟（86%→20%）= 典型分布外泛化失败
- 现有模块无法弥合语言差异
- 已写入论文 v3 5.5节

---

## 实验二：emotion2vec+ 大模型对比（效果差，未写入论文）

### 目的
对比 WavLM vs emotion2vec+ 系列模型在儿童SER上的表现。

### 模型
- emotion2vec_plus_base (~300M, 768-dim) — 下载完成但磁盘满未跑完训练
- emotion2vec_plus_large (~600M, 1024-dim) — 完整跑完

### 方法
- 数据集: MY (泰卢固语+英语混合, 4,179 WAVs)
- 配置: B3 (Adapter + Prosody Pooling + DrseCNN)
- 注意: adapter_init.npz 是 768-dim，1024-dim 模型使用随机初始化

### 结果

| 模型 | Val Acc (%) | Test Acc (%) |
|------|------------|-------------|
| emotion2vec_plus_large (1024-dim) | 23.42 ± 0.76 | 22.45 ± 0.92 |

参考: WavLM B3 Test = 81.24%

### 可能原因分析
1. FunASR 特征提取可能不正确 — 大量 decoder key missing 警告
2. emotion2vec 预训练数据主要是中文，对泰卢固语/英语儿童语音迁移差
3. 1024-dim 特征可能需要不同的 Adapter/Pooling 超参
4. 冻结参数164M vs WavLM 94M，但特征质量反而更差
5. Phase 1 中 emotion2vec_base 为 ~66%，plus_large 反而 22% 不合逻辑

### 待排查
- FunASR 的 extract_features(mode='default') 对 plus 模型的输出是否正确
- 是否需要不同的帧率/层选择
- 模型文件约 1.81GB，下载正常无损坏

---

## 脚本清单

| 文件 | 用途 |
|------|------|
| scripts/run_cross_language.py | 跨语言迁移实验（特征提取+训练） |
| scripts/run_model_comparison.py | emotion2vec+ 对比实验 |
| scripts/run_e2v_large.py | emotion2vec_plus_large 独立脚本 |
| experiments/cross_language_results.json | 跨语言实验原始结果 |
| experiments/model_comparison_results.json | emotion2vec+ 实验原始结果 |

## 代码改动

| 文件 | 改动 |
|------|------|
| src/models/ssl_backbone.py | _load_model() 新增 emotion2vec_plus_base/large 分支；_load_emotion2vec_funasr() 参数化 model_id+hidden_size；Emotion2vecFunASRWrapper 支持 hidden_size；_load_wavlm() 加入 local_files_only+本地快照路径回退 |
| scripts/extract_ssl_features.py | --model choices 新增 emotion2vec_plus_base/large |
| src/training/train_ssl.py | 新增 --feature_dim 参数，DistributionCalibratedSER 支持可变维度 |
