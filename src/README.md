# src/ — 源代码

AC 套件（`ac_suite_2026-05`）的核心训练和推理代码。

## 根目录文件

| 文件 | 用途 |
|------|------|
| `train.py` | **主训练入口**。在线 WavLM + LayerFusion + Pooling + SEMLP 端到端训练。支持 `--pooling_type`、`--reg_profile`、`--train_data`/`--test_data` 等参数 |
| `extract_diagnostics.py` | **诊断提取脚本**（C2）。从 checkpoint 提取：A) Layer weights, B) XAI saliency + APC, C) Distribution shift（注意：C 写死 C-BESD vs CREMA-D） |

## data/ — 数据加载

| 文件 | 用途 |
|------|------|
| `audio_processor.py` | 标准化音频预处理：16kHz 重采样、mono 转换、peak normalization |
| `label_mapper.py` | 统一 4 类标签映射（angry/happy/neutral/sad），从 BESD 6 类中选 4 类 |
| `speaker_splitter.py` | **确定性 MD5 hash 说话人独立划分**（70/15/15），含运行时 zero-leakage assert |
| `dataset.py` | `UnifiedSERDataset`：统一跨语料库数据集类 |
| `data_loader.py` | `get_dataloaders()` 和 `get_cross_corpus_dataloaders()` 构建器。2026-05-27 新增 `test_split` 参数 |
| `dataset_ssl.py` | [legacy] SSL 特征数据集类（预提取特征模式） |
| `preprocess.py` | [legacy] BESD-only speaker 划分逻辑 |
| `prepare_extra_datasets.py` | 额外数据集准备（MyST/KidsTALC 等） |
| `statistics.py` | 儿童语音统计分析（F0/energy 分布） |

## models/ — 模型组件

| 文件 | 用途 |
|------|------|
| `ssl_backbone.py` | **WavLM/emotion2vec 封装**。通过 HuggingFace 或 FunASR+ModelScope 加载，支持 `return_all_layers` 输出 12 层隐藏状态 |
| `layer_fusion.py` | **WavLMLayerFusion**：12 层可学习加权求和（softmax 归一化），是 AC 套件相比 v5 的关键提升 |
| `pooling.py` | **时序池化模块**：`TemporalImportancePooling`（韵律引导）和 `SelfAttentionPooling`（纯数据驱动）。两者参数数量严格一致（111,105） |
| `semlp.py` | **SEMLP 分类器**：Squeeze-Excitation MLP 分类头（~593K 参数） |
| `adapter.py` | [legacy] AcousticCalibrationAdapter：分布校准适配器（v5 发现贡献仅 ~0.5pp，AC 套件已移除） |

## evaluation/ — 评估与诊断

| 文件 | 用途 |
|------|------|
| `distribution_metrics.py` | **DistributionShiftProbe**：FD（Fréchet Distance）+ SMMD 双指标分布偏移测量 |
| `xai_visualizer.py` | **AttentionProsodyExplainer**：APC 计算 + 注意力-韵律联合可视化 |

## training/ — 训练

| 文件 | 用途 |
|------|------|
| `train_ssl.py` | [legacy] 预提取特征模式的训练脚本（Phase 1–5，已被 `train.py` 取代） |

## augmentation/ — 数据增强

| 文件 | 用途 |
|------|------|
| `safe_augmentation.py` | SafeAWGN：加性高斯白噪声增强（SNR 10-20dB） |
| `constrained_aug.py` | [legacy] 分布约束增强（Phase 2.3，已废弃） |

## utils/ — 工具

| 文件 | 用途 |
|------|------|
| `experiment_logger.py` | 实验日志系统（config + metrics + results + registry） |
