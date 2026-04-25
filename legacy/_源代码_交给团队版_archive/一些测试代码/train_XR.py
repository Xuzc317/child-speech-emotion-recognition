import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nets import CNNModel,DrseCNN,CNNBiLSTMModel
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AblationDataset(Dataset):
    def __init__(self, data_path, label_path, feature_type='MFCC'):
        self.datas = np.load(data_path)
        self.labels = np.load(label_path)
        self.feature_type = feature_type
        self.mask = {
            'MFCC': slice(13, 53),
            'ZCR': slice(0, 1),
            'Mel': slice(53, 93)
        }[feature_type]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        feature = np.zeros(94)
        feature[self.mask] = self.datas[idx][self.mask]
        return (
            torch.tensor(feature, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)  # 关键修改
        )

def evaluate_model(feature_type):
    dataset = AblationDataset('data_all.npy', 'label_all.npy', feature_type)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = HeavyCNN(num_classes=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练（简化为10个epoch）
    model.train()
    for epoch in range(64):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))  # 调整输入维度 (batch,1,94)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # 评估
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in DataLoader(dataset, batch_size=128):
            inputs = inputs.to(device)
            outputs = model(inputs.unsqueeze(1))
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1

if __name__ == "__main__":
    features = ['MFCC', 'ZCR', 'Mel']
    print("| Feature | Accuracy | F1-Score |")
    print("|---------|----------|----------|")
    for feat in features:
        acc, f1 = evaluate_model(feat)
        print(f"| {feat} | {acc:.4f} | {f1:.4f} |")