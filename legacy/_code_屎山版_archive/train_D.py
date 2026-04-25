#BiGRU_LSTM Training Script

# ------------------------------------------------------------
# Learning Rate: 0.00025000
# Train Loss: 0.5913 | Acc: 78.54%
# Val Loss: 1.4402 | Acc: 55.79%
# ------------------------------------------------------------
# Training complete in 1.94 minutes
# Best validation accuracy: 56.73%
# Final model saved as final_HybridBiGRU_LSTM.pth

import englishdata as data
from nets import CNNModel, DrseNet, CNNBiLSTMModel, BiGRU_LSTM
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from sklearn.metrics import f1_score, recall_score

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置与train_2.py一致的训练参数和随机种子
lr = 2e-4
weight_decay = 1e-6
epochs = 400
batch_size = 128
save_path = "best_BiGRU_model.pth"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2024)

def get_dataloader(wav_dir, label_file, scale=0.7):
    dataset = data.npyDataset(wav_dir, label_file)
    
    total_size = len(dataset)
    train_size = int(total_size * scale)
    test_size = total_size - train_size
    
    generator = torch.Generator()
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size], 
        generator=generator
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data.collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data.collate_fn
    )
    
    return train_dataloader, test_dataloader

def train(model, criterion, optimizer, dataloader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for inputs, labels in tqdm(dataloader, desc='Training'):
        # 打印输入形状进行调试
        # print(f"原始输入形状: {inputs.shape}")
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 重塑输入形状为 (batch_size, seq_len, input_size)
        if isinstance(model, BiGRU_LSTM):
            # 确保输入是2D: (batch_size, features)
            # 如果是1D特征，应只有batch_size和features两个维度
            if inputs.dim() == 2:
                # 如果只有批量大小和特征两个维度
                # 添加时间步维度: (batch_size, 1, features)
                inputs = inputs.unsqueeze(1)
            elif inputs.dim() == 3 and inputs.size(1) == 1:
                # 如果已经是 (batch_size, 1, features)，不需要额外操作
                pass
            elif inputs.dim() == 4:
                # 如果是4D输入（可能是图像数据），需要调整为2D特征向量
                # 假设形状是 (batch_size, 1, 1, features)
                if inputs.size(2) == 1 and inputs.size(3) == 1:
                    # 压缩第二维和第三维
                    inputs = inputs.squeeze(2).squeeze(1)
                else:
                    # 如果不确定维度，取最后一个维度作为特征
                    inputs = inputs.view(inputs.size(0), -1).unsqueeze(1)
            else:
                # 其他情况尝试重塑为 (batch_size, 1, features)
                inputs = inputs.view(inputs.size(0), 1, -1)
                
        # print(f"模型输入形状: {inputs.shape}")
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    return running_loss/len(dataloader), correct/total, f1, recall

def evaluate(model, criterion, dataloader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 重塑输入形状为 (batch_size, seq_len, input_size)
            if isinstance(model, BiGRU_LSTM):
                if inputs.dim() == 2:
                    inputs = inputs.unsqueeze(1)
                elif inputs.dim() == 4:
                    # 如果是4D输入，处理为2D特征向量
                    if inputs.size(2) == 1 and inputs.size(3) == 1:
                        inputs = inputs.squeeze(2).squeeze(1)
                    else:
                        inputs = inputs.view(inputs.size(0), -1).unsqueeze(1)
                elif inputs.dim() != 3:
                    inputs = inputs.view(inputs.size(0), 1, -1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    return running_loss/len(dataloader), correct/total, f1, recall

def plot_training_curves(train_accs, val_accs, train_losses, val_losses, model_name):
    plt.figure(figsize=(15, 5))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_curves_{model_name}.png')
    plt.close()

def main():
    # 数据路径（保持你的原始配置）
    wav_dir = r'C:\Users\59892\Desktop\毕业\code\data_all.npy'
    label_file = r'C:\Users\59892\Desktop\毕业\code\label_all.npy'
    
    # 获取数据加载器
    train_loader, test_loader = get_dataloader(wav_dir, label_file)
    
    # 查看输入数据的实际维度
    sample_input, _ = next(iter(train_loader))
    actual_input_size = sample_input.shape[-1]
    # print(f"输入数据的特征维度: {actual_input_size}")
    
    # 初始化模型 - 使用HybridBiGRU_LSTM
    # 将输入尺寸调整为实际数据维度
    model = BiGRU_LSTM(
        input_size=actual_input_size,  # 使用实际特征维度而非预设值
        gru_hidden_size=128,
        lstm_hidden_size1=256,
        lstm_hidden_size2=128,
        dropout_rate=0.4,  # 可调整的dropout率
        num_classes=6
    ).to(device)
    
    # 打印模型信息
    model_name = "BiGRU_LSTM"
    print(f"Model architecture: {model_name}")
    print(f"输入尺寸: {actual_input_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化配置
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # 训练循环
    best_acc = 0.0
    best_f1 = 0.0
    best_recall = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_f1s, val_recalls = [], []
    training_start_time = time.time()
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        train_loss, train_acc, train_f1, train_recall = train(model, criterion, optimizer, train_loader)
        val_loss, val_acc, val_f1, val_recall = evaluate(model, criterion, test_loader)
        
        # 学习率调度
        scheduler.step(val_acc)
        
        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_recalls.append(val_recall)
        
        epoch_duration = time.time() - epoch_start_time
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            best_recall = val_recall
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at epoch {epoch+1} with acc {val_acc:.2%}")
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{epochs}] ({epoch_duration:.2f}s)")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")
        print(f"Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f}")
        print("-"*60)
    
    total_training_time = time.time() - training_start_time
    print(f"Training complete in {total_training_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_acc:.2%}")
    
    # 绘制训练曲线
    plot_training_curves(train_accs, val_accs, train_losses, val_losses, model_name)
    
    # 保存最终模型
    torch.save(model.state_dict(), f"final_{model_name}.pth")
    print(f"Final model saved as final_{model_name}.pth")
    
    # 保存验证准确率曲线和总结
    import csv
    with open('BiGRU_LSTM_valacc.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'val_acc'])
        for i, acc in enumerate(val_accs, 1):
            writer.writerow([i, acc])
    with open('BiGRU_LSTM_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['best_val_acc', 'best_f1', 'best_recall', 'seed'])
        writer.writerow([best_acc, best_f1, best_recall, 2024])
    print(f'最优 acc={best_acc:.4f}, f1={best_f1:.4f}, recall={best_recall:.4f}, seed=2024')

if __name__ == "__main__":
    main()