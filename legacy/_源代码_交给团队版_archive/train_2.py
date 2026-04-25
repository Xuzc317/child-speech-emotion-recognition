import englishdata as data
from nets import CNNModel,DrseCNN,CNNBiLSTMModel
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 5e-4
weight_decay = 5e-5
epochs = 128
batch_size = 128
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

def plot_training_curves(train_accs, val_accs, train_losses, val_losses):
    plt.figure(figsize=(15, 5))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    # plt.plot(val_f1s, label='Val F1', linestyle='--')
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves_CNN.png')  # 保存图片
    plt.close()

def main():
    # 数据路径（保持你的原始配置）
    wav_dir = r'C:\Users\59892\Desktop\毕业\code\data_all.npy'
    label_file = r'C:\Users\59892\Desktop\毕业\code\label_all.npy'
    
    # 获取数据加载器
    train_loader, test_loader = get_dataloader(wav_dir, label_file)
    
    # 初始化模型
    # model = CNNModel(num_classes=6).to(device)
    # model =LSTMModel(num_classes=6).to(device)
    # model =HybridCTN(num_classes=6).to(device)   
    # model =DenseModel(num_classes=6).to(device)
    model =DrseCNN(num_classes=6).to(device)
    # model =CNNBiLSTMModel(num_classes=6).to(device)

    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化配置
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 训练循环
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        train_loss, train_acc = train(model, criterion, optimizer, train_loader)
        val_loss, val_acc = evaluate(model, criterion, test_loader)
        
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
    print(best_acc)

    # 绘制训练曲线
    plot_training_curves(train_accs, val_accs, train_losses, val_losses)
    print(f"Best validation accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()