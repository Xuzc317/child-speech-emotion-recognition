#VISION_01:wav2+Transformer

import data
from nets import TransformerModel, LSTMModel,RNNModel
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch import optim
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import Counter

torch.set_default_device('cuda')

lr = 1e-4
weight_decay = 0.001
epochs = 10

def get_dataloader(wav_dir, label_file, scale=0.8, batch_size=16):
    dataset = data.EmotionDataset(wav_dir, label_file)

    # 计算训练集和测试集的大小
    total_size = len(dataset)
    train_size = int(total_size * scale)
    test_size = total_size - train_size

    # 使用 random_split 划分数据集
    generator = torch.Generator(device='cuda')
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    # 创建训练集和测试集的 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data.collate_fn, generator=torch.Generator(device='cuda'))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data.collate_fn, generator=torch.Generator(device='cuda'))

    return train_dataloader, test_dataloader


def train(model, criterion, optimizer, dataloader):
    model.train()

    running_loss = 0.
    total_correct = 0.
    
    for src, labels, mask in tqdm(dataloader, desc='Training', leave=False):
        optimizer.zero_grad()

        outputs = model(src, src_mask=mask)
        # outputs = model(src)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predictions = torch.max(outputs, -1)
        total_correct += (predictions == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = total_correct / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc


def evaluate(model, dataloader):
    model.eval() # set the model to evaluation mode
    
    total_correct = 0
    with torch.no_grad():
        for src, labels, mask in tqdm(dataloader, desc='Evaluating', leave=False):
            outputs = model(src, src_mask=mask)
            _, predictions = torch.max(outputs, -1)
            total_correct += (predictions == labels).sum().item()


    epoch_acc = total_correct / len(dataloader.dataset)
    

    return epoch_acc

def main():
    wav_dir = r'D:\大学\毕设\IS2009EmotionChallenge\IS2009EmotionChallenge\wav'
    # label_file = r'D:\大学\毕设\IS2009EmotionChallenge\IS2009EmotionChallenge\labels\word_labels_5cl_corpus_processed_noN.txt'
    label_file = r'D:\大学\毕设\IS2009EmotionChallenge\IS2009EmotionChallenge\labels\word_labels_5cl_corpus_out.txt'

    train_dataloader, test_dataloader = get_dataloader(wav_dir, label_file, scale=0.7, batch_size=6)

    # 初始化模型
    model = TransformerModel(768, output_dim=5).to("cuda")
    # model = LSTMModel(768).to("cuda")
    # model = RNNModel(768, output_dim=5).to("cuda")
    # model = CNNModel(768, output_dim=5).to('cuda')
    # 加载参数
    # model.load_state_dict(torch.load("model.pth"))
    # model.eval()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss() # 使用类别权重

    best_acc = 0.
    val_acc = 0.

    for epoch in range(epochs):
        train_loss, train_acc = train(model, criterion, optimizer, train_dataloader)
        val_acc  = evaluate(model, test_dataloader)#+ val_precision, val_recall, val_f1 获取所有返回值

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Acc: {:.4f}'.format(
            epoch+1, epochs, train_loss, train_acc, val_acc))
        

        if val_acc > best_acc:
            best_acc = val_acc
            # save the checkpoint
            torch.save(model.state_dict(), 'best_model_{val_acc}_{epoch}.pth')

main()