# 评估模块
## Evaluation Module

## 说明

评估模块包含模型性能评估和错误分析的代码。当前版本中，评估功能已集成在训练脚本中。

## 功能概述

### 已实现功能
1. **基本评估指标**
   - 准确率 (Accuracy)
   - Macro F1分数
   - Recall (召回率)
   - 混淆矩阵计算

2. **集成在训练脚本中**
   - `src/training/train.py` 中的 `evaluate()` 函数
   - 训练过程中的验证评估
   - 最佳模型保存基于验证准确率

### 计划功能
1. **详细错误分析**
   - 类别级性能指标
   - 困难样本分析
   - 错误模式可视化

2. **额外评估指标**
   - Precision (精确率)
   - ROC-AUC曲线
   - 校准曲线

3. **模型比较工具**
   - 统计显著性检验
   - 计算效率对比
   - 内存使用分析

## 使用示例

### 基本评估
```python
from src.training.train import evaluate

# 使用训练脚本中的评估函数
val_loss, val_acc = evaluate(model, criterion, val_loader)
print(f"验证损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
```

### 扩展评估 (示例代码)
```python
import torch
from sklearn.metrics import classification_report, confusion_matrix

def detailed_evaluation(model, dataloader, device='cuda'):
    """详细评估函数"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算详细指标
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)
    
    return report, cm
```

## 相关文件

- `src/training/train.py` - 包含基础评估函数
- `experiments/results/validation_metrics.csv` - 评估结果记录
- `docs/experiment-results.md` - 评估结果分析

---

**注**: 当前版本专注于核心训练流程，评估功能相对基础。可根据需要扩展更详细的评估模块。