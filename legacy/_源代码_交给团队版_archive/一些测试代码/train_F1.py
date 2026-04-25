# # import englishdata as data
# # from nets import CNNModel,DrseCNN,CNNBiLSTMModel  # 根据你的模型类型修改这里
# # import torch
# # from torch.utils.data import DataLoader
# # import numpy as np
# # from sklearn.metrics import f1_score
# # from tqdm import tqdm

# # # 配置参数
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model_path = r'C:\Users\59892\Desktop\毕业\code\bestmodel\best_cnn_model_all_0.7214.pth'  # 你的模型文件路径
# # batch_size = 128
# # num_classes = 6  # 类别数量

# # # 数据路径配置（需要与训练时相同）
# # wav_dir = r'C:\Users\59892\Desktop\毕业\code\data_all.npy'
# # label_file = r'C:\Users\59892\Desktop\毕业\code\label_all.npy'

# # def load_model(model_class):
# #     """加载预训练模型"""
# #     model = model_class(num_classes=num_classes).to(device)
# #     model.load_state_dict(torch.load(model_path, map_location=device))
# #     model.eval()
# #     return model

# # def get_testloader():
# #     """获取测试数据加载器（使用完整数据集）"""
# #     dataset = data.npyDataset(wav_dir, label_file)
# #     return DataLoader(
# #         dataset,
# #         batch_size=batch_size,
# #         shuffle=False,
# #         collate_fn=data.collate_fn
# #     )

# # def evaluate_f1(model, test_loader):
# #     """计算F1-score"""
# #     all_preds = []
# #     all_labels = []
    
# #     with torch.no_grad():
# #         for inputs, labels in tqdm(test_loader, desc='Evaluating'):
# #             inputs = inputs.to(device)
# #             labels = labels.to(device)
            
# #             outputs = model(inputs)
# #             _, preds = torch.max(outputs, 1)
            
# #             all_preds.extend(preds.cpu().numpy())
# #             all_labels.extend(labels.cpu().numpy())
    
# #     return f1_score(all_labels, all_preds, average='macro')  # 使用macro平均

# # if __name__ == "__main__":
# #     # 1. 加载模型（修改模型类为你的实际模型）
# #     model = load_model(CNNModel)  # 例如：DrseCNN/CNNBiLSTMModel等
    
# #     # 2. 准备数据
# #     test_loader = get_testloader()
    
# #     # 3. 计算指标
# #     f1 = evaluate_f1(model, test_loader)
    
# #     # 4. 输出结果
# #     print(f"\n{'='*40}")
# #     print(f"Model loaded from: {model_path}")
# #     print(f"Test F1-score (macro): {f1:.4f}")
# #     print(f"{'='*40}")

# import torch
# from torch.utils.data import DataLoader
# import englishdata as data
# from nets import DrseCNN, CNNBiLSTMModel, CNNModel  # 确保与训练时使用的模型一致
# import numpy as np
# from sklearn.metrics import classification_report, f1_score

# # 配置设置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# batch_size = 128
# model_path = "best_cnn_model_all_0.7214.pth"  # 替换为你的模型路径
# wav_dir = r'C:\Users\59892\Desktop\毕业\code\data_all.npy'
# label_file = r'C:\Users\59892\Desktop\毕业\code\label_all.npy'

# def get_test_dataloader(wav_dir, label_file):
#     """获取完整测试集"""
#     dataset = data.npyDataset(wav_dir, label_file)
#     test_loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=data.collate_fn
#     )
#     return test_loader

# def evaluate_model():
#     # 加载数据
#     test_loader = get_test_dataloader(wav_dir, label_file)
    
#     # 初始化模型（必须与训练时结构完全一致）
#     model = CNNModel(num_classes=6).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
    
#     # 收集预测结果
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             preds = torch.argmax(outputs, dim=1)
            
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
    
#     # 计算指标
#     accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
#     macro_f1 = f1_score(all_labels, all_preds, average='macro')
#     report = classification_report(all_labels, all_preds, digits=4)
    
#     # 打印结果
#     print(f"【验证集准确率】: {accuracy:.4f}")
#     print(f"【宏平均F1分数】: {macro_f1:.4f}")
#     print("\n【分类报告】:")
#     print(report)

# if __name__ == "__main__":
#     evaluate_model()

import torch
from torch.utils.data import DataLoader, random_split
import englishdata as data
from nets import DrseCNN  # 必须与训练时模型结构完全一致
import numpy as np
from sklearn.metrics import classification_report, f1_score

# 配置参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
model_path = "best_HeavyCNN_model_0.8682.pth"  # 替换为你的模型路径
wav_dir = r'C:\Users\59892\Desktop\毕业\code\data_all.npy'
label_file = r'C:\Users\59892\Desktop\毕业\code\label_all.npy'
scale = 0.7  # 必须与train_2.py中的划分比例一致

def get_val_dataloader(wav_dir, label_file, scale=0.7):
    """严格复现train_2.py的验证集划分逻辑"""
    dataset = data.npyDataset(wav_dir, label_file)
    # 使用相同的随机种子保证划分一致
    # generator = torch.Generator().manual_seed(42)  # 固定随机种子
    _, val_dataset = random_split(
        dataset, 
        [scale, 1-scale], 
        # generator=generator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data.collate_fn
    )
    return val_loader

def evaluate_val_set():
    # 加载模型（结构与训练时完全一致）
    model = DrseCNN(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到评估模式
    
    # 获取验证集
    val_loader = get_val_dataloader(wav_dir, label_file, scale)
    
    # 收集预测和标签
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, digits=4)
    
    # 打印结果
    print(f"【验证集准确率】: {accuracy:.4f} (应与train_2.py的val_acc一致)")
    print(f"【宏平均F1分数】: {macro_f1:.4f}")
    print("\n【分类报告】:")
    print(report)

if __name__ == "__main__":
    evaluate_val_set()