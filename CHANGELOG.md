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
