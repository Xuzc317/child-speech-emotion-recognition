# 模型权重文件
## Model Checkpoints

## 说明

此目录包含项目的**最佳模型权重文件**（精选自原始40+个权重文件）。这些权重文件**不随GitHub仓库上传**（已加入`.gitignore`），仅供本地实验和论文复现使用。

## 当前模型权重清单

### 已保留的最佳模型

| 模型 | 文件名 | 大小 | 验证准确率 | 状态 |
|------|--------|------|-----------|------|
| **DrseCNN (最佳记录)** | `best_DrseCNN_model_0.8682.pth` | ~45.5 MB | 0.8682 | 随机种子未固定 |
| BiLSTM (最佳) | `best_BiLSTMModel_model_0.7226.pth` | ~12.1 MB | 0.7226 | - |
| SigWavNet (最佳) | `best_sigwavnet_0.7318_model.pth` | ~16.9 MB | 0.7318 | - |
| CNN (最佳) | `best_cnn_model_all_0.7214.pth` | ~5.8 MB | 0.7214 | - |
| BiLSTM (次佳) | `best_BiLSTM_0.5415_model.pth` | ~2.6 MB | 0.5415 | - |
| Transformer | `best_transformer_0.6948_model.pth` | ~2.0 MB | 0.6948 | - |
| Dense | `best_dense_model_0.6079.pth` | ~5.8 MB | 0.6079 | - |
| LSTM | `best_lstm_model_0.4414.pth` | ~8.7 MB | 0.4414 | - |

### 性能说明

**重要**: 早期实验记录的最佳准确率约0.8682，但由于当时未固定随机种子，该结果可能无法完全稳定复现。当前仓库以可复现的**0.8599准确率**和**0.8506 Macro F1**作为主要报告结果。

## 为什么不包含权重文件？

### 1. 文件大小限制
- GitHub免费账户推荐仓库大小 < 1GB
- 权重文件总计约1.5 GB，超过推荐限制
- 大文件影响克隆速度和存储

### 2. Git LFS成本
- 使用Git LFS需要额外成本
- 免费额度有限 (每月1GB带宽)
- 不适合公开开源项目

### 3. 学术诚信考虑
- 数据集有使用限制，连带模型权重可能受限
- 避免潜在的数据集分发问题

### 4. 实际需求
- 研究代码和文档是主要展示内容
- 权重文件可以重新训练生成
- 重点展示方法和实验，而非具体权重

## 如何获取或生成权重文件？

### 方案A: 自行训练生成
```bash
# 1. 获取C-BESD数据集
# 2. 提取特征 (见data/README.md)
# 3. 运行训练
python src/training/train.py

# 4. 权重文件将保存在:
#    best_model.pth (默认名称)
```

### 方案B: 使用模拟数据测试
```python
# 创建模拟数据验证模型架构
import torch
from src.models.models import DrseCNN

# 初始化模型
model = DrseCNN(num_classes=6)

# 创建模拟数据
dummy_input = torch.randn(32, 1, 100, 94)  # (batch, channel, time, features)
output = model(dummy_input)
print(f"模型输出形状: {output.shape}")  # 应为 (32, 6)
```

### 方案C: 联系作者获取
如需原始训练好的权重文件，可以：
1. 通过GitHub Issues联系项目作者
2. 提供合法的C-BESD数据集使用证明
3. 说明使用目的和研究计划

## 模型权重文件结构

### PyTorch权重文件格式
```python
# 文件内容结构
{
    'state_dict': {
        'layer1.weight': tensor(...),
        'layer1.bias': tensor(...),
        # ... 所有模型参数
    },
    'epoch': 337,
    'accuracy': 0.8599,
    'args': {
        'learning_rate': 0.0005,
        'batch_size': 128,
        # ... 训练参数
    }
}
```

### 加载和使用
```python
import torch
from src.models.models import DrseCNN

# 1. 初始化模型
model = DrseCNN(num_classes=6)

# 2. 加载权重 (如果有文件)
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

# 3. 使用模型
model.eval()
with torch.no_grad():
    predictions = model(input_data)
```

## 权重文件管理建议

### 个人使用
1. **本地存储**: 将权重文件保存在本地，不提交到Git
2. **版本管理**: 为不同实验保存不同权重文件
3. **备份**: 重要权重文件定期备份

### 团队协作
1. **共享存储**: 使用云存储 (Google Drive, OneDrive) 共享大文件
2. **文档记录**: 记录每个权重文件的训练配置和性能
3. **标准命名**: 使用统一命名规范，如`模型_准确率_日期.pth`

### 长期维护
1. **定期清理**: 删除中间结果，保留最佳模型
2. **模型压缩**: 使用模型量化、剪枝等技术减小文件大小
3. **格式转换**: 考虑转换为ONNX等通用格式

### GitHub存储建议
如果需要将权重文件包含在GitHub仓库中：
1. **Git LFS**: 使用Git Large File Storage管理大文件
2. **外部存储**: 存储在云存储服务（如Google Drive、OneDrive）并提供下载链接
3. **版本控制**: 仅保留关键检查点，避免存储所有中间结果

## 相关文件

- `src/models/models.py` - 模型定义，可用于重新训练
- `src/training/train.py` - 训练脚本，可生成新权重
- `docs/model-structure.md` - 模型架构详细说明
- `docs/reproducibility-notes.md` - 复现训练步骤

## 注意事项

1. **版权说明**: 使用C-BESD数据集训练的模型权重受数据集使用协议约束
2. **技术依赖**: 权重文件与特定模型架构和代码版本绑定
3. **性能保证**: 自行训练的模型性能可能因随机性和环境差异而不同
4. **使用限制**: 模型仅供研究和学习使用，不用于商业或医疗诊断

---

**总结**: 本仓库专注于展示研究方法和代码实现，模型权重文件可通过重新训练获得。这种设计符合学术开源的最佳实践，强调方法的可复现性而非具体权重。