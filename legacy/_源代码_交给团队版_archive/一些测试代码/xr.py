# """
# xr.py - SE/Residual Ablation Study
# Usage: 
# 1. Place data in ./data/{train,val,test}_features.npy and _labels.npy
# 2. Install dependencies: 
#    pip install torch numpy scikit-learn tqdm
# 3. Run: python xr.py
# """
# # 设计消融实验的模型变体：

# # 基线模型（BaseCNN）：没有SE模块和残差连接的普通CNN。
# # 仅残差结构（ResCNN）：加入残差连接，但无SE模块。
# # 仅SE模块（SECNN）：加入SE模块，但无残差连接。
# # 完整模型（DrseCNN）：同时包含SE模块和残差结构。

# """
# xr.py - SE/Residual Ablation Study for Speech Emotion Recognition
# Usage:
# 1. Ensure data files exist: 
#    - ./data/train_features.npy
#    - ./data/train_labels.npy
#    - ./data/test_features.npy  
#    - ./data/test_labels.npy
# 2. Install requirements:
#    pip install torch numpy scikit-learn tqdm matplotlib
# 3. Run: python xr.py
# """

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from nets import CNNModel,DrseCNN,CNNBiLSTMModel
from nets_sp import BaseCNN,ResCNN,SECNN
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import os



# Configuration
CFG = {
    "input_dim": 94,        # MFCC+ZCR+Chroma等特征维度
    "num_classes": 6,      # 6类情绪
    "epochs": 15,
    "batch_size": 128,
    "lr": 5e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}



# ##########################
# 数据加载与训练工具
# ##########################
# def get_dataloader(wav_dir, label_file, scale=0.7):
#     dataset = data.npyDataset(wav_dir, label_file)
    
#     total_size = len(dataset)
#     train_size = int(total_size * scale)
#     test_size = total_size - train_size
    
#     generator = torch.Generator()
#     train_dataset, test_dataset = random_split(
#         dataset, 
#         [train_size, test_size], 
#         generator=generator
#     )
    
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=data.collate_fn
#     )
    
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=data.collate_fn
#     )
    
#     return train_dataloader, test_dataloader




def get_dataloaders(data_dir='./data', batch_size=128):
    """更健壮的数据加载函数"""
    try:
        # 使用完整路径
        train_path = os.path.join(data_dir, 'train_features.npy')
        test_path = os.path.join(data_dir, 'test_features.npy')
        
        # 检查文件是否存在
        if not all(os.path.exists(p) for p in [train_path, test_path]):
            raise FileNotFoundError("数据文件缺失，请先运行 englishdata.py")
            
        # 加载数据
        train_data = np.load(train_path)
        train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
        test_data = np.load(test_path)
        test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))
        
        # 验证形状
        assert train_data.ndim == 3, f"训练数据应为3维，实际为{train_data.shape}"
        assert test_data.shape[1:] == train_data.shape[1:], "测试/训练数据形状不匹配"
        
        # 创建DataLoader
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(train_data), torch.LongTensor(train_labels)),
            batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(test_data), torch.LongTensor(test_labels)),
            batch_size=batch_size, shuffle=False
        )
        
        return train_loader, test_loader
        
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise

# 全局导出
train_loader, test_loader = get_dataloaders()

def train(model, criterion, optimizer, dataloader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    


    for inputs, labels in tqdm(dataloader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss/len(dataloader), correct/total

def evaluate(model, criterion, dataloader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc='Evaluating'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
   
    return running_loss/len(dataloader), correct/total

def load_data():
    """加载预处理数据"""
    def load_npy(path):
        data = np.load(path)
        # 自动处理2D或3D输入
        if len(data.shape) == 2:  # 如果是(B,94)的原始特征
            data = np.expand_dims(data, 1)  # 变为(B,1,94)
        return torch.FloatTensor(data)  # 不再需要permute

    try:
        train_X = load_npy("./data/train_features.npy")
        train_y = torch.LongTensor(np.load("./data/train_labels.npy"))
        test_X = load_npy("./data/test_features.npy") 
        test_y = torch.LongTensor(np.load("./data/test_labels.npy"))
        
        print(f"数据形状 - 训练特征: {train_X.shape}, 标签: {train_y.shape}")
    except FileNotFoundError:
        print("错误：未找到数据文件，请先运行 englishdata.py")
        exit(1)
    
    return (
        DataLoader(TensorDataset(train_X, train_y), 
                 batch_size=CFG["batch_size"], shuffle=True),
        DataLoader(TensorDataset(test_X, test_y),
                 batch_size=CFG["batch_size"])
    )

def train_model(model, train_loader, test_loader):
    """训练与评估流程"""
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"],weight_decay = 5e-5)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    
    for epoch in range(CFG["epochs"]):
        # 训练阶段
        model.train()
        train_loss, correct, total ,train_acc= 0, 0, 0, 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(CFG["device"]), labels.to(CFG["device"])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_acc = correct/total
        # 验证阶段
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(CFG["device"]), labels.to(CFG["device"])
                # inputs = torch.unsqueese(inputs)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                test_acc = test_correct/test_total
        
        # 保存最佳模型
        test_acc = test_correct / test_total
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss/len(train_loader):.4f} Train Acc: {train_acc/len(train_loader):.4f} |"
              f"Test Loss: {test_acc:.4f} Test Acc: {test_acc:.4f}")
        

    # 最终评估
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(CFG["device"])
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return {"accuracy": acc, "f1_score": f1}

# ##########################
# 主程序
# ##########################


if __name__ == "__main__":
    # 加载数据
    train_loader, test_loader = load_data()
    
    # 消融实验模型列表
    models = {
        "BaseCNN": BaseCNN(input_dim=94, num_classes=6).to(CFG["device"]),
        "ResCNN": ResCNN(num_classes=6).to(CFG["device"]),
        "SECNN": SECNN(input_dim=94, num_classes=6).to(CFG["device"]),
        "DrseCNN": DrseCNN().to(CFG["device"])
        # "CNNModelCNN": CNNModel().to(CFG["device"]),
        # "CNNBiLSEM": CNNBiLSTMModel().to(CFG["device"])
    }
    
    # 运行实验

    results = {}
    for name, model in models.items():
        print(f"\n===== Training {name} =====")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        results[name] = train_model(model, train_loader, test_loader)
    
    # 打印结果对比
    print("\n===== Ablation Study Results =====")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(10, 5))
    x = range(len(results))
    plt.bar(x, [m['accuracy'] for m in results.values()], width=0.2, label='Accuracy', color='#cbd5e8')
    plt.bar([i + 0.4 for i in x], [m['f1_score'] for m in results.values()], width=0.2, label='F1-Score', color='#f4cae4')
    plt.xticks([i + 0.2 for i in x], results.keys())
    plt.ylim(0.5, 0.9)
    plt.title("Component Ablation Study")
    plt.legend()
    plt.savefig("ablation_results.png")
    plt.show()


