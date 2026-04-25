#VISION_03:MFCC--CNN+Transformer

import englishdata as data
from nets import CNNModel,DenseModel,LSTMModel,HeavyCNN,HybridCTN
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
# ==== ADD START ==== 新增模块导入
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from utils.early_stopping import EarlyStopping
import numpy as np
# ==== ADD END ====

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 优化后的超参数
lr = 5e-4
weight_decay = 5e-5
epochs = 100
batch_size = 64
save_path = "best_cnn_model.pth"


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

def train(model, criterion, optimizer, dataloader, scaler, scheduler=None):  # ==== 修改参数 ====
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # ==== ADD START ==== 混合精度训练
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler:  # OneCycleLR需要每个step更新
            scheduler.step()
        # ==== ADD END ====
        
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
        
        # ==== ADD START ==== 验证时混合精度
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        # ==== ADD END ====
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
   
    return running_loss/len(dataloader), correct/total

def main():
    # 数据路径（保持你的原始配置）
    wav_dir = r'C:\Users\59892\Desktop\毕业\code\data_all.npy'
    label_file = r'C:\Users\59892\Desktop\毕业\code\label_all.npy'
    
    # 获取数据加载器
    train_loader, test_loader = get_dataloader(wav_dir, label_file)
    
    # ==== ADD START ==== 类别权重（根据实际数据调整）
    class_weights = torch.tensor([1.0, 0.8, 1.2, 1.0, 0.9, 1.1]).to(device)
    # ==== ADD END ====
    
    # 初始化模型
    model = HybridCTN(num_classes=6, input_length=163).to(device)  # ==== 需要指定input_length ====
    
    # ==== ADD START ==== SWA模型初始化
    swa_model = AveragedModel(model)
    swa_start = int(epochs * 0.75)  # 最后25%的epoch使用SWA
    # ==== ADD END ====
    
    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化配置
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,        # ==== 添加类别权重 ====
        label_smoothing=0.2          # ==== 添加标签平滑 ====
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # ==== ADD START ==== 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )
    scaler = GradScaler()
    early_stopper = EarlyStopping(patience=10, verbose=True)
    # ==== ADD END ====
    
    # 训练循环
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        train_loss, train_acc = train(model, criterion, optimizer, train_loader, scaler, scheduler)
        val_loss, val_acc = evaluate(model, criterion, test_loader)
        
        # ==== ADD START ==== SWA更新
        if epoch >= swa_start:
            swa_model.update_parameters(model)
        # ==== ADD END ====
        
        # ==== ADD START ==== 早停机制
        early_stopper(val_acc, model)
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break
        # ==== ADD END ====
        
        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at epoch {epoch+1} with acc {val_acc:.2%}")
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")
        print("-"*60)
    
    # ==== ADD START ==== SWA最终更新
    torch.optim.swa_utils.update_bn(train_loader, swa_model)
    swa_save_path = "swa_model.pth"
    torch.save(swa_model.state_dict(), swa_save_path)
    print(f"SWA model saved at {swa_save_path}")
    # ==== ADD END ====
    
    print(f"Best validation accuracy: {best_acc:.2%}")

if __name__ == "__main__":
    main()
