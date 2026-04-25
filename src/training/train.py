import argparse
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import npyDataset, collate_fn
from src.models.models import CNNModel, DrseCNN, CNNBiLSTMModel


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default hyperparameters (can be overridden by command-line arguments)
DEFAULT_LR = 5e-4
DEFAULT_WEIGHT_DECAY = 5e-5
DEFAULT_EPOCHS = 128
DEFAULT_BATCH_SIZE = 128
DEFAULT_SAVE_PATH = "best_model.pth"

def get_dataloader(wav_dir, label_file, batch_size, scale=0.7):
    dataset = npyDataset(wav_dir, label_file)

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
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
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
    parser = argparse.ArgumentParser(description='Train emotion recognition model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the feature data file (data_all.npy)')
    parser.add_argument('--label_path', type=str, required=True,
                        help='Path to the label data file (label_all.npy)')
    parser.add_argument('--model', type=str, default='DrseCNN',
                        choices=['CNNModel', 'DrseCNN', 'CNNBiLSTMModel'],
                        help='Model architecture to use')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help=f'Learning rate (default: {DEFAULT_LR})')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY,
                        help=f'Weight decay (default: {DEFAULT_WEIGHT_DECAY})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Number of epochs (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--save_path', type=str, default=DEFAULT_SAVE_PATH,
                        help=f'Path to save the best model (default: {DEFAULT_SAVE_PATH})')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train dataset ratio (default: 0.7)')

    args = parser.parse_args()

    # Check if data files exist
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        print("Please prepare the dataset and generated feature files first.")
        print("Refer to src/data/dataset.py for feature extraction instructions.")
        return

    if not os.path.exists(args.label_path):
        print(f"Error: Label file not found at {args.label_path}")
        print("Please prepare the dataset and generated feature files first.")
        print("Refer to src/data/dataset.py for feature extraction instructions.")
        return

    print(f"Using data: {args.data_path}")
    print(f"Using labels: {args.label_path}")
    print(f"Training model: {args.model}")
    print(f"Hyperparameters: lr={args.lr}, wd={args.weight_decay}, "
          f"epochs={args.epochs}, batch_size={args.batch_size}")

    # Get data loaders
    train_loader, test_loader = get_dataloader(
        args.data_path, args.label_path, args.batch_size, scale=args.train_ratio
    )

    # Initialize model
    if args.model == 'CNNModel':
        model = CNNModel(num_classes=6).to(device)
    elif args.model == 'DrseCNN':
        model = DrseCNN(num_classes=6).to(device)
    elif args.model == 'CNNBiLSTMModel':
        model = CNNBiLSTMModel(num_classes=6).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimization
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, criterion, optimizer, train_loader)
        val_loss, val_acc = evaluate(model, criterion, test_loader)

        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"New best model saved at epoch {epoch+1} with acc {val_acc:.2%}")

        # Print progress
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")
        print("-"*60)

    print(f"Final best validation accuracy: {best_acc:.2%}")

    # Plot training curves
    plot_training_curves(train_accs, val_accs, train_losses, val_losses)
    print(f"Training curves saved to training_curves.png")
    print(f"Best model saved to {args.save_path}")


if __name__ == "__main__":
    main()