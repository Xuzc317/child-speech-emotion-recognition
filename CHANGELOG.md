# 变更日志 (Changelog)

记录所有修改的历程，便于回顾。

---

## [2026-04-25] 修复数据泄露问题

### 背景
发现预处理和训练管道存在严重的数据泄露问题：
- `get_features()` 为每个 WAV 文件生成 3 个增强版本（原始/加噪/拉伸变调）
- 所有 3 个版本全部存入 `data_all.npy`
- `random_split` 在样本级别划分训练/测试集
- 导致同一 WAV 的增强版本有 63% 概率同时出现在训练集和测试集中

### 修改计划
1. 修复预处理管道：先划分原始 WAV 文件为 train/test，再对训练集做增强
2. 修复训练脚本：适配新的数据处理方式
3. 修复评估脚本：确保测试集不受污染
4. 重新生成无泄露的特征文件

---

### 变更记录

| 时间 | 文件 | 变更 | 原因 |
|------|------|------|------|
| 2026-04-25 19:00 | CHANGELOG.md | 新建 | 开始记录所有修改历程 |
| 2026-04-25 18:50 | data_all.npy, label_all.npy | 重新生成 | 上次运行崩溃导致 label 全为0，已从 BESD 原始 WAV 重新提取 |
| 2026-04-25 19:10 | src/data/preprocess.py | 新建 | 正确的预处理管道：先按WAV文件级别做分层划分(stratified split, seed=42)，再对训练集做3x增强，测试集仅保留原始特征。输出 train_data.npy / train_labels.npy / test_data.npy / test_labels.npy |
| 2026-04-25 19:10 | src/training/train.py | 修改 | 移除 random_split 逻辑，改为直接加载 preprocess.py 生成的 train/test 分离数据；参数从 --data_path/--label_path/--train_ratio 改为 --train_data/--train_label/--test_data/--test_label |
| 2026-04-25 19:10 | evaluate_model.py | 修改 | 移除 random_split 和特征提取回退逻辑，改为直接加载 preprocess.py 生成的无污染测试数据 test_data.npy / test_labels.npy |

---

## [2026-04-25] 重构：说话人无关划分 + 切换至 MY 数据集

### 背景
论文声称采用说话人无关（speaker-independent）70/30 划分，但实际代码仅为文件级别划分，存在数据泄露。同时，ENGLISH 仅是 BESD 的子集，MY = ENGLISH + TELUGU 才是完整数据集。

### 修改内容
1. 数据集从 `ENGLISH/` 切换至 `MY/`（4,180 个 WAV 文件，239 名唯一说话人）
2. 实现真正的说话人无关划分：从文件名提取说话人 ID，按说话人级别做 70/30 划分（seed=42）
3. 说话人 ID 提取：解析文件名中情感词之前的说话人前缀（如 `1.EF_12 Angry_1.wav` → `1.EF_12`）
4. 训练集：每 WAV 提取 3 个增强版本（原始+加噪+拉伸变调）
5. 测试集：仅提取原始特征（无增强），确保说话人不重叠

### 变更记录

| 时间 | 文件 | 变更 | 原因 |
|------|------|------|------|
| 2026-04-25 | src/data/preprocess.py | 重写 | 切换至 MY 数据集，按说话人级别划分（239 名→167 训练/72 测试），修复文件名解析边界情况（anger/disguist 变体、缺失点号等） |
| 2026-04-25 | regenerate_features.py | 修改 | WAV_DIR 从 ENGLISH 切换至 MY |
| 2026-04-25 | ROADMAP.md | 修改 | 更新说话人无关划分任务状态及数据统计（239 说话人，4,180 WAV） |
| 2026-04-25 | CHANGELOG.md | 修改 | 追加本次变更日志 |
| 2026-04-25 | src/data/preprocess.py | 添加 normalize_speaker_id() | 修正说话人 ID 变体：连字符归一化（`1.EF-12` → `1.EF_12`）和缺失点号修复（`3EF_9` → `3.EF_9`），确保同一说话人不被拆分到训练/测试集两侧 |
| 2026-04-25 | ROADMAP.md | 修改 | 更新训练/测试数据处理策略：双方均做 3x 处理（应称为"数据处理"而非"数据增强"）；更新统计数字至 237 说话人、4,179 WAV |
| 2026-04-25 | train_data.npy 等 | 重新生成 | 重新运行 preprocess.py 生成正确的说话人无关划分数据（237 说话人，165/72，0 重叠） |

---

## [2026-04-26] 代码审查修复

### 背景
进行系统性代码审查，发现 8 个问题：维度注释错误、import 位置不当、checkpoint 命名混乱、CNNModel 硬编码维度、说话人划分缺少分层、评估脚本重复定义、缺少依赖包。

### 变更记录

| 时间 | 文件 | 变更 | 原因 |
|------|------|------|------|
| 2026-04-26 | src/data/dataset.py | 修正 extract_features 维度注释：94 → 162；移除 EnglishDataset 中未使用的 MFCC transform | 实际维度为 1+12+20+1+128=162（librosa 默认 n_mfcc=20, n_mels=128），旧注释误写为 1+12+40+1+40=94 |
| 2026-04-26 | src/data/preprocess.py | import re 移至文件顶部 | 避免每次调用 normalize_speaker_id() 时重复 import |
| 2026-04-26 | src/data/preprocess.py | split_speakers() 改为 profile-based 分层划分 | 按说话人的情感类别组合分组后分别 70/30 划分，比全局随机划分更均衡 |
| 2026-04-26 | src/models/models.py | CNNModel 最后 MaxPool1d(5,2) 保留，Linear 输入改为动态计算（dummy forward 推导 flat_dim） | 原 AdaptiveAvgPool1d 方案与 ONNX 不兼容；新方案用标准 MaxPool1d + 动态维度推导，兼容任意输入维度且 ONNX 可导出 |
| 2026-04-26 | CLAUDE.md | 新建 | 记录 ONNX 兼容性约束（禁止 AdaptiveAvgPool1d）、特征维度说明、数据划分架构、旧 checkpoint 警告 |
| 2026-04-26 | checkpoints/ | 全部 .pth/.csv 移至 checkpoints/legacy/ | 旧模型均基于泄露数据训练，评估结果无效；添加 README 说明 |
| 2026-04-26 | evaluate_model.py | 移除重复 NpyDataset/collate_fn，改为 from src.data.dataset import；CKPT_PATH 改为 None + 防护提示 | 消除代码重复；避免误用泄露数据训练的旧 checkpoint |
| 2026-04-26 | requirements.txt | 添加 parselmouth, scipy | analyze_besd_final.py 依赖 parselmouth，test_signal.py 依赖 scipy |
| 2026-04-26 | ROADMAP.md | 更新划分策略说明和维度信息 | 同步最新实现细节 |
| 2026-04-26 | train_data.npy 等 | 重新生成 | 基于 profile-based 分层划分重新提取特征（165 说话人训练 / 72 测试，每类分布更均衡） |
