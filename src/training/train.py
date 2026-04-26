import argparse
import json
import math
import os
import random
import signal
import sys
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import npyDataset, collate_fn
from src.data.augmentation import mixup_features, feature_noise
from src.utils.tracker import ExperimentTracker
from src.models.models import (
    CNNModel, DrseCNN, DrseNNet_new, CNNBiLSTMModel,
    CRNNModel, Transformer, TDNNModel,
    DenseModel, LSTMModel, OptimizedBiLSTM
)

MODEL_REGISTRY = {
    'CNNModel': CNNModel,
    'DrseCNN': DrseCNN,
    'DrseNNet_new': DrseNNet_new,
    'CNNBiLSTMModel': CNNBiLSTMModel,
    'CRNNModel': CRNNModel,
    'Transformer': Transformer,
    'TDNNModel': TDNNModel,
    'DenseModel': DenseModel,
    'LSTMModel': LSTMModel,
    'OptimizedBiLSTM': OptimizedBiLSTM,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── defaults ──────────────────────────────────────────────────────────
DEFAULT_LR          = 5e-4
DEFAULT_WD          = 5e-5
DEFAULT_EPOCHS      = 150
DEFAULT_BATCH_SIZE  = 128
DEFAULT_PATIENCE    = 30
DEFAULT_SEED        = 42
DEFAULT_WARMUP      = 5
DEFAULT_MIXUP_ALPHA = 0.2
DEFAULT_LABEL_SMOOTH = 0.1
DEFAULT_NOISE_STD   = 0.005
MAX_GRAD_NORM       = 1.0
EMA_DECAY           = 0.999

_g_interrupted = False


def _sig_handler(signum, frame):
    global _g_interrupted
    _g_interrupted = True
    print("\n[Interrupted] Saving emergency checkpoint after current epoch...")


signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)


# ── seed ───────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── data loaders ───────────────────────────────────────────────────────
def get_dataloaders(train_data_path, train_label_path,
                    test_data_path, test_label_path,
                    batch_size, num_workers=4):
    train_dataset = npyDataset(train_data_path, train_label_path)
    test_dataset  = npyDataset(test_data_path, test_label_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ── EMA ────────────────────────────────────────────────────────────────
class EMAModel:
    """Exponential Moving Average of model weights for stable eval."""
    def __init__(self, model, decay=EMA_DECAY):
        self.module = model
        self.decay = decay
        self.shadow = {}
        self._register()

    def _register(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Call after eval to restore training weights."""
        pass  # shadow is already a copy; no-op if we don't modify training weights


# ── LR scheduler ──────────────────────────────────────────────────────
def create_scheduler(optimizer, warmup_epochs, total_epochs, scheduler_type="cosine"):
    """Warmup + Cosine Annealing (or ReduceLROnPlateau)."""
    if scheduler_type == "cosine":
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return 0.1 + 0.9 * epoch / max(warmup_epochs, 1)
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return 0.001 + 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.999
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


# ── train / eval ──────────────────────────────────────────────────────
def train_one_epoch(model, criterion, optimizer, dataloader,
                    scaler=None, mixup_alpha=0.0, noise_std=0.0,
                    ema=None, epoch=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        # Feature noise
        if noise_std > 0:
            inputs = feature_noise(inputs, noise_std)

        # Mixup
        labels_a, labels_b, lam = labels, None, 1.0
        if mixup_alpha > 0:
            inputs, labels_a, labels_b, lam = mixup_features(inputs, labels, mixup_alpha)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = lam * criterion(outputs, labels_a)
                if labels_b is not None:
                    loss = loss + (1.0 - lam) * criterion(outputs, labels_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = lam * criterion(outputs, labels_a)
            if labels_b is not None:
                loss = loss + (1.0 - lam) * criterion(outputs, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        if ema is not None:
            ema.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), correct / total


def evaluate(model, criterion, dataloader, ema=None):
    model.eval()

    # Use EMA shadow weights for evaluation
    if ema is not None:
        ema_backup = {n: p.data.clone()
                      for n, p in model.named_parameters() if p.requires_grad}
        ema.apply_shadow()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Restore training weights
    if ema is not None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(ema_backup[n])

    return running_loss / len(dataloader), correct / total


# ── plotting ──────────────────────────────────────────────────────────
def plot_training_curves(train_accs, val_accs, train_losses, val_losses, model_name):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_curves_{model_name}.png')
    plt.close()


# ── main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Train emotion recognition model')
    # Data
    parser.add_argument('--train_data', default='train_data.npy')
    parser.add_argument('--train_label', default='train_labels.npy')
    parser.add_argument('--test_data', default='test_data.npy')
    parser.add_argument('--test_label', default='test_labels.npy')
    # Model
    parser.add_argument('--model', default='DrseCNN',
                        choices=list(MODEL_REGISTRY.keys()))
    # Training
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WD)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    # LR
    parser.add_argument('--lr_scheduler', default='cosine',
                        choices=['cosine', 'plateau'])
    parser.add_argument('--warmup_epochs', type=int, default=DEFAULT_WARMUP)
    # Augmentation
    parser.add_argument('--mixup_alpha', type=float, default=DEFAULT_MIXUP_ALPHA)
    parser.add_argument('--noise_std', type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument('--label_smoothing', type=float, default=DEFAULT_LABEL_SMOOTH)
    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--no_ema', action='store_true')
    parser.add_argument('--save_path', default='best_model.pth')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    # WandB
    parser.add_argument('--wandb_entity', default=None)
    parser.add_argument('--wandb_project', default='child-speech-emotion')
    parser.add_argument('--run_name', default=None)
    # Modes
    parser.add_argument('--dry_run', action='store_true',
                        help='Print config and exit without training')
    parser.add_argument('--validate_only', type=str, default=None,
                        help='Load checkpoint and evaluate only, then exit')

    args = parser.parse_args()

    set_seed(args.seed)

    # Print resolved config
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model}  |  LR: {args.lr}  |  WD: {args.weight_decay}")
    print(f"Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  Patience: {args.patience}")
    print(f"LR schedule: {args.lr_scheduler}  |  Warmup: {args.warmup_epochs} epochs")
    print(f"Mixup a: {args.mixup_alpha}  |  Noise s: {args.noise_std}  |  Label smooth: {args.label_smoothing}")
    print(f"EMA: {not args.no_ema}  |  Seed: {args.seed}  |  Workers: {args.num_workers}")
    if device.type == 'cuda':
        try:
            mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"GPU: {torch.cuda.get_device_name(0)}  |  Memory: {mem_gb:.0f} GB")
        except Exception:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    if args.dry_run:
        m = MODEL_REGISTRY[args.model](num_classes=6)
        ver = getattr(m, 'VERSION', 'N/A')
        print(f"[Dry run] Model {args.model}  Version {ver}  Params {sum(p.numel() for p in m.parameters()):,}")
        return

    # Data file checks
    for path, name in [(args.train_data, 'train_data'),
                        (args.train_label, 'train_label'),
                        (args.test_data, 'test_data'),
                        (args.test_label, 'test_label')]:
        if not os.path.exists(path):
            print(f"Error: {name} file not found at {path}")
            print("Run src/data/preprocess.py first.")
            return

    # Dataloaders
    train_loader, test_loader = get_dataloaders(
        args.train_data, args.train_label,
        args.test_data, args.test_label,
        args.batch_size, args.num_workers
    )

    # Model
    model = MODEL_REGISTRY[args.model](num_classes=6).to(device)
    model_version = getattr(model, 'VERSION', 'v0')
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model version: {model_version}")

    # Loss
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # Scheduler
    scheduler = create_scheduler(optimizer, args.warmup_epochs,
                                 args.epochs, args.lr_scheduler)

    # AMP
    use_amp = (device.type == 'cuda') and not args.no_amp
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")

    # EMA
    ema = None if args.no_ema else EMAModel(model)

    # Tracker
    tracker = ExperimentTracker(
        model=model, project=args.wandb_project, entity=args.wandb_entity,
        config=vars(args), run_name=args.run_name,
        log_dir="experiments/logs"
    )

    # Save path — include model version to avoid overwriting across versions
    save_path = args.save_path
    if save_path == 'best_model.pth':
        save_path = f'best_{args.model}_{model_version}.pth'
    config_path = save_path.replace('.pth', '_config.json')
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                exist_ok=True)

    # Validate-only mode
    if args.validate_only:
        print(f"\n[Validate-only] Loading checkpoint: {args.validate_only}")
        ckpt = torch.load(args.validate_only, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict)
        val_loss, val_acc = evaluate(model, criterion, test_loader)
        print(f"Validation — Loss: {val_loss:.4f}  Acc: {val_acc:.2%}")
        return

    # Resume
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"Resumed at epoch {start_epoch+1}, best_acc={best_acc:.2%}")

    # Training loop
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_loader,
            scaler=scaler, mixup_alpha=args.mixup_alpha,
            noise_std=args.noise_std, ema=ema, epoch=epoch
        )
        val_loss, val_acc = evaluate(model, criterion, test_loader, ema=ema)

        epoch_time = time.time() - t0

        # Cosine scheduler steps every epoch; plateau scheduler needs val_acc
        if args.lr_scheduler == "plateau":
            scheduler.step(val_acc)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Record
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Per-class accuracy for tracking
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                preds = model(inputs).argmax(1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        per_class = {}
        cls_names = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD']
        for i in range(6):
            mask = [l == i for l in all_labels]
            if any(mask):
                correct_i = sum(1 for p, l in zip(all_preds, all_labels) if p == l and l == i)
                per_class[f'val/acc_{cls_names[i]}'] = correct_i / sum(mask)

        # Log
        tracker.log({
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'train/lr': current_lr,
            'epoch_time_s': epoch_time,
            **per_class,
        }, step=epoch + 1)

        # Best checkpoint — save only the best, with full reproducibility info
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0

            # Collect all hyperparameters for reproducibility
            hp = vars(args).copy()
            hp['model_version'] = model_version
            hp['num_params'] = sum(p.numel() for p in model.parameters())

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'model_name': args.model,
                'model_version': model_version,
                'hyperparameters': hp,
            }, save_path)

            # Save human-readable config alongside checkpoint
            with open(config_path, 'w') as f:
                json.dump(hp, f, indent=2, ensure_ascii=False)

            print(f"[Best] epoch {epoch+1}  acc {val_acc:.2%}  → {save_path}")
        else:
            patience_counter += 1

        # Console
        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"LR {current_lr:.2e}  {epoch_time:.0f}s")
        print(f"  Train — Loss {train_loss:.4f}  Acc {train_acc:.2%}")
        print(f"  Val   — Loss {val_loss:.4f}  Acc {val_acc:.2%}")
        print("-" * 60)

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if _g_interrupted:
            print("Interrupted — saving emergency checkpoint...")
            emergency_path = f"checkpoints/emergency_{args.model}_{model_version}_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'model_name': args.model,
                'model_version': model_version,
            }, emergency_path)
            print(f"Emergency checkpoint saved: {emergency_path}")
            break

    # Finish
    tracker.finish()
    print(f"\nBest validation accuracy: {best_acc:.2%}")
    print(f"Model saved to: {save_path}")
    print(f"Config saved to: {config_path}")
    plot_training_curves(train_accs, val_accs, train_losses, val_losses, args.model)
    print(f"Training curves saved to training_curves_{args.model}.png")


if __name__ == "__main__":
    main()
