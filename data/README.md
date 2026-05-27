# data/ — 预处理特征与中间数据

存放 Phase 1–3 期间预提取的 SSL 特征、增强实验数据和 Adapter 初始化参数。**这些文件在 AC 套件（在线训练）中不再使用**，但保留用于历史参考和可能的本地诊断。

## 预提取 SSL 特征（Phase 1）

| 文件 | 形状 | 内容 |
|------|------|------|
| `train_wavlm_feats.npy` | (2912, 200, 768) | WavLM 训练集帧级特征 |
| `test_wavlm_feats.npy` | (1267, 200, 768) | WavLM 测试集帧级特征 |
| `val_wavlm_feats.npy` | — | WavLM 验证集帧级特征 |
| `train_wavlm_labels.npy` | — | 对应训练集标签 |
| `test_wavlm_labels.npy` | — | 对应测试集标签 |
| `val_wavlm_labels.npy` | — | 对应验证集标签 |
| `train_wavlm_prosody.npy` | — | 训练集韵律特征（F0+能量） |
| `test_wavlm_prosody.npy` | — | 测试集韵律特征 |
| `val_wavlm_prosody.npy` | — | 验证集韵律特征 |
| `train_e2v_feats.npy` / `test_e2v_feats.npy` | — | emotion2vec 特征（对比实验，已弃用） |
| `train_e2v_labels.npy` / `test_e2v_labels.npy` | — | emotion2vec 标签 |

## 增强实验特征（Phase 2.3 负面实验）

按成人/儿童/极端三种参数对音频做 pitch shift + time stretch 后提取的 WavLM 特征：

| 前缀 | 增强参数 | 含义 |
|------|---------|------|
| `train_C2_adult_*` / `test_C2_adult_*` | pitch ±6st, stretch 0.7-1.3 | 成人 SER 标准增强参数 |
| `train_C3_child_*` / `test_C3_child_*` | pitch ±3st, stretch 0.85-1.15 | 儿童约束增强参数 |
| `train_C4_extreme_*` / `test_C4_extreme_*` | pitch ±12st, stretch 0.5-1.5 | 极端增强参数 |

文件名后缀 `_wavlm_feats.npy` 为特征，`_labels.npy` 为标签。

## 其他

| 文件 | 内容 |
|------|------|
| `adapter_init.npz` | AcousticCalibrationAdapter 统计先验初始化参数（儿童 vs 成人 per-dim mean/std） |
| `mfcc_baseline/` | MFCC 基线特征（用于对比 SSL 提升幅度） |

## last/

`fau_aibo.tar.gz`：FAU Aibo 原始数据压缩包（已解压到本地数据集路径）。
