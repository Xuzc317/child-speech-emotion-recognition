# confusion_matrix_6class.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import englishdata as data
from nets import CNNModel,DrseCNN,CNNBiLSTMModel
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD']

def generate_confusion_matrix(model_path):
    # 加载数据和模型
    dataset = data.npyDataset('data_all.npy', 'label_all.npy')
    model = CNNModel(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 获取预测结果
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for inputs, labels in DataLoader(dataset, batch_size=128, collate_fn=data.collate_fn):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            true_labels.extend(labels.numpy())
            pred_labels.extend(preds.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 可视化设置
    plt.figure(figsize=(12,10))
    sns.set(font_scale=1.1)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    annot_kws={'size':12},
                    cbar_kws={'label': '样本数量'})
    
    # 图表标注
    ax.set_title('儿童语音情绪识别混淆矩阵 (6类)', fontsize=14, pad=20)
    ax.set_xlabel('预测标签', fontsize=12)
    ax.set_ylabel('真实标签', fontsize=12)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    
    # 调整标签显示
    plt.xticks(rotation=45, ha='left')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存和显示
    plt.savefig('confusion_matrix_6class.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(true_labels, pred_labels, target_names=class_names))

if __name__ == "__main__":
    generate_confusion_matrix("best_cnn_model_all_0.7214.pth")