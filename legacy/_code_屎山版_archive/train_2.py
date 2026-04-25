import englishdata as data
from nets import CNNModel,DrseNet,CNNBiLSTMModel,LSTMModel,DenseModel,BiLSTM,AdaptedSigWavNet,SimpleRNNNet # 导入所有需要的模型
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score
import numpy as np
import csv
import random

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练参数
lr = 5e-4
weight_decay = 5e-5
epochs = 128
batch_size = 128
save_path = "best_DrseNet_model_NotData.pth"

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
    all_preds = []
    all_labels = []
    with torch.no_grad(): # 评估模式下禁用梯度计算
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
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
    wav_dir = r'C:\Users\59892\Desktop\毕业\code\data.npy'
    label_file = r'C:\Users\59892\Desktop\毕业\code\label.npy'
    
    # 获取数据加载器
    train_loader, test_loader = get_dataloader(wav_dir, label_file)
    
    # 初始化模型
    # model = CNNModel(num_classes=6).to(device)
    # model =LSTMModel(num_classes=6).to(device)
    # model =HybridCTN(num_classes=6).to(device)   
    # model =DenseModel(num_classes=6).to(device)
    model =DrseNet(num_classes=6).to(device)
    # model =CNNBiLSTMModel(num_classes=6).to(device)

    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化配置
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 训练循环
    best_acc = 0.0
    best_f1 = 0.0
    best_recall = 0.0
    best_epoch = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_f1s, val_recalls = [], []
    
    for epoch in range(epochs):
        train_loss, train_acc = train(model, criterion, optimizer, train_loader)
        val_loss, val_acc, val_f1, val_recall = evaluate(model, criterion, test_loader)
        
        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_recalls.append(val_recall)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            best_recall = val_recall
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at epoch {epoch+1} with acc {val_acc:.2%}")
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.4f} | Recall: {val_recall:.4f}")
        print("-"*60)
    
    print(f"--- Final Results ---")
    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print(f"Best F1-score: {best_f1:.4f}")
    print(f"Best Recall: {best_recall:.4f}")

    # 保存验证指标到CSV
    csv_file_name = 'validation_metrics.csv'
    with open(csv_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'val_acc', 'val_f1', 'val_recall'])
        for i, acc in enumerate(val_accs, 1):
            writer.writerow([i, acc, val_f1s[i-1], val_recalls[i-1]])
    print(f'Validation metrics saved to {csv_file_name}')

    # 绘制训练曲线
    plot_training_curves(train_accs, val_accs, train_losses, val_losses)


if __name__ == "__main__":
    main()
# def plot_training_curves(train_accs, val_accs, train_losses, val_losses, val_f1s=None, model_name=""):
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_accs, label='Train Acc')
#     plt.plot(val_accs, label='Val Acc')
#     if val_f1s is not None:
#         plt.plot(val_f1s, label='Val F1', linestyle='--')
#     plt.title(f'{model_name} Training Metrics')
#     plt.xlabel('Epoch')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(val_losses, label='Val Loss')
#     plt.title(f'{model_name} Training Losses')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'training_curves_{model_name}.png') # 根据模型名称保存
#     plt.close()

# def run_all_ablation_experiments():
#     model_configs = [
#         ("DrseNet_Full", DrseNet),
#         ("DrseNet_NoSE", NoSE_DrseNet),
#         ("DrseNet_NoRes", NoRes_DrseNet),
#         ("DrseNet_PlainCNN", PlainCNN_DrseNet),
#         ("DrseNet_SimplifiedStage", SimplifiedStage_DrseNet),
#     ]
#     wav_dir = r'C:\Users\59892\Desktop\毕业\code\data_all.npy'
#     label_file = r'C:\Users\59892\Desktop\毕业\code\label_all.npy'
#     train_loader, test_loader = get_dataloader(wav_dir, label_file)
#     summary_rows = [["model_name", "best_val_acc", "best_f1", "best_recall", "epoch"]]
#     for model_name, model_cls in model_configs:
#         print(f"\n===== Running experiment for: {model_name} =====")
#         model = model_cls(num_classes=6).to(device)
#         criterion = torch.nn.CrossEntropyLoss()
#         optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#         best_acc = 0.0
#         best_f1 = 0.0
#         best_recall = 0.0
#         best_epoch = 0
#         train_losses, val_losses = [], []
#         train_accs, val_accs = [], []
#         val_f1s, val_recalls = [], []
#         for epoch in range(epochs):
#             train_loss, train_acc = train(model, criterion, optimizer, train_loader)
#             val_loss, val_acc, val_f1, val_recall = evaluate(model, criterion, test_loader)
#             train_losses.append(train_loss)
#             train_accs.append(train_acc)
#             val_losses.append(val_loss)
#             val_accs.append(val_acc)
#             val_f1s.append(val_f1)
#             val_recalls.append(val_recall)
#             if val_acc > best_acc:
#                 best_acc = val_acc
#                 best_f1 = val_f1
#                 best_recall = val_recall
#                 best_epoch = epoch + 1
#                 torch.save(model.state_dict(), f"{save_path.replace('.pth', '')}_{model_name}.pth")
#                 print(f"New best model saved for {model_name} at epoch {epoch+1} with acc {val_acc:.2%}")
#             print(f"Epoch [{epoch+1}/{epochs}] for {model_name}")
#             print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
#             print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.4f} | Recall: {val_recall:.4f}")
#             print("-"*60)
#         print(f"--- {model_name} Final Results ---")
#         print(f"Best validation accuracy: {best_acc:.4f}")
#         print(f"Best F1-score: {best_f1:.4f}")
#         print(f"Best Recall: {best_recall:.4f}")
#         plot_training_curves(train_accs, val_accs, train_losses, val_losses, val_f1s, model_name)
#         csv_file_name = f'{model_name}_valacc.csv'
#         with open(csv_file_name, 'w', newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(['epoch', 'val_acc', 'val_f1', 'val_recall'])
#             for i, acc in enumerate(val_accs, 1):
#                 writer.writerow([i, acc, val_f1s[i-1], val_recalls[i-1]])
#         summary_rows.append([model_name, best_acc, best_f1, best_recall, best_epoch])
#     # 汇总所有实验结果
#     with open('ablation_summary.csv', 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerows(summary_rows)
#     print("\n所有消融实验已完成，汇总结果已保存到 ablation_summary.csv")

#     # Model 5: Simple RNN
#     # 自动推断输入特征维度
#     sample_batch = next(iter(train_loader))[0]
#     if sample_batch.dim() == 2:
#         input_size = sample_batch.shape[1]
#     elif sample_batch.dim() == 3:
#         input_size = sample_batch.shape[2]
#     else:
#         raise ValueError('未知的数据 shape，无法自动推断 input_size')
#     model_name = "SimpleRNNNet"
#     # model = SimpleRNNNet(input_size=input_size, num_classes=6).to(device)
#     model = DrseNet(num_classes=6).to(device)
#     state_dict = torch.load("best_Drsecnn_model_0.8599.pth", map_location=torch.device('cuda'))
#     model.load_state_dict(state_dict)
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     best_acc = 0.0
#     best_f1 = 0.0
#     best_recall = 0.0
#     best_epoch = 0
#     train_losses, val_losses = [], []
#     train_accs, val_accs = [], []
#     val_f1s, val_recalls = [], []
#     for epoch in range(epochs):
#         # train_loss, train_acc = train(model, criterion, optimizer, train_loader)
#         val_loss, val_acc, val_f1, val_recall = evaluate(model, criterion, test_loader)
#         train_losses.append(train_loss)
#         train_accs.append(train_acc)
#         val_losses.append(val_loss)
#         val_accs.append(val_acc)
#         val_f1s.append(val_f1)
#         val_recalls.append(val_recall)
#         if val_acc > best_acc:
#             best_acc = val_acc
#             best_f1 = val_f1
#             best_recall = val_recall
#             best_epoch = epoch + 1
#             torch.save(model.state_dict(), f"{save_path.replace('.pth', '')}_{model_name}.pth")
#             print(f"New best model saved for {model_name} at epoch {epoch+1} with acc {val_acc:.2%}")
#         print(f"Epoch [{epoch+1}/{epochs}] for {model_name}")
#         print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
#         print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.4f} | Recall: {val_recall:.4f}")
#         print("-"*60)
#     print(f"--- {model_name} Final Results ---")
#     print(f"Best validation accuracy: {best_acc:.4f}")
#     print(f"Best F1-score: {best_f1:.4f}")
#     print(f"Best Recall: {best_recall:.4f}")
#     plot_training_curves(train_accs, val_accs, train_losses, val_losses, val_f1s, model_name)
#     csv_file_name = f'{model_name}_valacc.csv'
#     with open(csv_file_name, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['epoch', 'val_acc', 'val_f1', 'val_recall'])
#         for i, acc in enumerate(val_accs, 1):
#             writer.writerow([i, acc, val_f1s[i-1], val_recalls[i-1]])
#     summary_rows.append([model_name, best_acc, best_f1, best_recall, best_epoch])
#     # 汇总所有实验结果
#     with open('ablation_summary.csv', 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerows(summary_rows)
#     print("\n所有消融实验已完成，汇总结果已保存到 ablation_summary.csv")

# def main():
#     wav_dir = r'C:\Users\59892\Desktop\毕业\code\data_all.npy'
#     label_file = r'C:\Users\59892\Desktop\毕业\code\label_all.npy'
#     train_loader, test_loader = get_dataloader(wav_dir, label_file)
#     model_name = "DrseNet_Full"
#     model = DrseNet(num_classes=6).to(device)
#     sdict = torch.load("best_Drsecnn_model_0.8599.pth", map_location=torch.device('cuda'))
#     model.load_state_dict(sdict)
#     print(f"Model architecture:\n{model}")
#     print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     best_acc = 0.0
#     best_f1 = 0.0
#     best_recall = 0.0
#     best_epoch = 0
#     train_losses, val_losses = [], []
#     train_accs, val_accs = [], []
#     val_f1s, val_recalls = [], []
#     for epoch in range(epochs):
#         # train_loss, train_acc = train(model, criterion, optimizer, train_loader)
#         val_loss, val_acc, val_f1, val_recall = evaluate(model, criterion, test_loader)
#         # train_losses.append(train_loss)
#         # train_accs.append(train_acc)
#         val_losses.append(val_loss)
#         val_accs.append(val_acc)
#         val_f1s.append(val_f1)
#         val_recalls.append(val_recall)
#         if val_acc > best_acc:
#             best_acc = val_acc
#             best_f1 = val_f1
#             best_recall = val_recall
#             best_epoch = epoch + 1
#             torch.save(model.state_dict(), f"{save_path.replace('.pth', '')}_{model_name}.pth")
#             print(f"New best model saved for {model_name} at epoch {epoch+1} with acc {val_acc:.2%}")
#         print(f"Epoch [{epoch+1}/{epochs}] for {model_name}")
#         # print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
#         print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.4f} | Recall: {val_recall:.4f}")
#         print("-"*60)
#     print(f"--- {model_name} Final Results ---")
#     print(f"Best validation accuracy: {best_acc:.4f}")
#     print(f"Best F1-score: {best_f1:.4f}")
#     print(f"Best Recall: {best_recall:.4f}")
#     plot_training_curves(train_accs, val_accs, train_losses, val_losses, val_f1s, model_name)
#     csv_file_name = f'{model_name}_valacc.csv'
#     with open(csv_file_name, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['epoch', 'val_acc', 'val_f1', 'val_recall'])
#         for i, acc in enumerate(val_accs, 1):
#             writer.writerow([i, acc, val_f1s[i-1], val_recalls[i-1]])
#     print(f'Validation metrics saved to {csv_file_name}')

# if __name__ == "__main__":
#     main()