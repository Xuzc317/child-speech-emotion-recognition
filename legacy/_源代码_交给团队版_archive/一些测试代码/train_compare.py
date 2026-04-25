# train_compare.py
import torch, os
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import englishdata as data
from nets import CNNModel, LSTMModel, TransformerModel, CRNNModel, TDNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {
    'CNN': CNNModel(num_classes=6),
    'CRNN': CRNNModel(num_classes=6),
    'Transformer': TransformerModel(feat_dim=163, num_classes=6),
    'TDNN': TDNNModel(num_classes=6),
}
lr, weight_decay = 5e-4, 5e-5
epochs, batch_size = 64, 256

def get_dataloaders():
    ds = data.npyDataset('./data_all.npy','./label_all.npy')
    tr, te = random_split(ds, [int(0.7*len(ds)), len(ds)-int(0.7*len(ds))])
    return DataLoader(tr, batch_size, True, collate_fn=data.collate_fn), \
           DataLoader(te, batch_size, False, collate_fn=data.collate_fn)

def train_one(model, train_loader, criterion, optimizer):
    model.train()
    total, correct, loss_sum = 0,0,0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        loss_sum += loss.item()
        pred = out.argmax(1)
        correct += (pred==y).sum().item(); total+=y.size(0)
    return loss_sum/len(train_loader), correct/total

def eval_one(model, val_loader, criterion):
    model.eval()
    total, correct, loss_sum = 0,0,0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item()
            correct += (out.argmax(1)==y).sum().item()
            total+=y.size(0)
    return loss_sum/len(val_loader), correct/total

def main():
    train_loader, test_loader = get_dataloaders()
    results = {}
    for name, model in models.items():
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_acc = 0
        for epoch in range(epochs):
            _, train_acc = train_one(model, train_loader, criterion, optimizer)
            _, val_acc   = eval_one(model, test_loader, criterion)
            best_acc = max(best_acc, val_acc)
        results[name] = best_acc
        print(f"{name} best val acc: {best_acc:.4f}")
    # 打印对比
    print("=== All Model Comparison ===")
    for k,v in results.items():
        print(f"{k}: {v*100:.2f}%")

if __name__=='__main__':
    main()
