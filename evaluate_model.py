"""Evaluate model checkpoint on BESD dataset — clean (no data leakage).

Usage:
    # One-shot evaluation
    python evaluate_model.py --ckpt checkpoints/best_DrseCNN.pth --model DrseCNN

    # Watch mode: re-evaluate every 60s during training
    python evaluate_model.py --ckpt checkpoints/best_DrseCNN.pth --model DrseCNN --watch

IMPORTANT: Old checkpoints in checkpoints/legacy/ were trained on leaked data.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)

from src.data.dataset import npyDataset, collate_fn
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

CLASS_NAMES = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD']

# ── helpers ───────────────────────────────────────────────────────────
def load_checkpoint(path, device):
    """Load checkpoint, handling both old (plain state_dict) and new formats."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    extra = {}
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        extra['epoch'] = ckpt.get('epoch', None)
        extra['best_acc'] = ckpt.get('best_acc', None)
        extra['model_version'] = ckpt.get('model_version', None)
        info_parts = []
        if extra['epoch'] is not None:
            info_parts.append(f"epoch {extra['epoch']}")
        if extra['best_acc'] is not None:
            info_parts.append(f"best_acc={extra['best_acc']:.2%}")
        if extra['model_version'] is not None:
            info_parts.append(f"version={extra['model_version']}")
        print(f"  Checkpoint: {', '.join(info_parts)}")
    else:
        state_dict = ckpt
    return state_dict, extra


def evaluate_model(model, dataloader, device, num_classes=6):
    """Run full evaluation: accuracy, per-class metrics, confusion matrix."""
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    total = len(all_labels)
    correct = int((all_preds == all_labels).sum())
    acc = correct / total

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None,
        labels=list(range(num_classes)), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    return {
        'accuracy': acc,
        'correct': correct,
        'total': total,
        'per_class': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        },
        'macro': {'precision': macro_p, 'recall': macro_r, 'f1': macro_f1},
        'confusion_matrix': cm,
    }


def print_report(results, model_name, ckpt_path):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 70)
    print(f"{model_name} Evaluation ({ckpt_path}) — CLEAN, no data leakage")
    print("=" * 70)

    print(f"\nSamples: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")

    print(f"\n--- Macro ---")
    print(f"Precision: {results['macro']['precision']:.4f}")
    print(f"Recall:    {results['macro']['recall']:.4f}")
    print(f"F1:        {results['macro']['f1']:.4f}")

    print(f"\n--- Per-class ---")
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 45)
    pc = results['per_class']
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<12} {pc['precision'][i]:<10.4f} {pc['recall'][i]:<10.4f} {pc['f1'][i]:<10.4f}")

    print(f"\n--- Confusion Matrix ---")
    cm = results['confusion_matrix']
    header = "        " + " ".join(f"{n[:6]:>7}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"{name:<8}" + " ".join(f"{cm[i][j]:>7d}" for j in range(len(CLASS_NAMES)))
        print(row)

    print("-" * 70)
    return results['accuracy']


# ── main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Evaluate emotion recognition model')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint (.pth)')
    parser.add_argument('--model', type=str, default='DrseCNN',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--test_data', default='test_data.npy')
    parser.add_argument('--test_label', default='test_labels.npy')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--watch', action='store_true',
                        help='Watch mode: re-evaluate every 60s (for monitoring active training)')
    parser.add_argument('--watch_interval', type=int, default=60,
                        help='Seconds between watch evaluations')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data checks
    if not os.path.exists(args.test_data) or not os.path.exists(args.test_label):
        print("ERROR: Test data not found. Run: python src/data/preprocess.py")
        sys.exit(1)

    test_dataset = npyDataset(args.test_data, args.test_label)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)
    print(f"Test set: {len(test_dataset)} samples")

    if args.watch:
        print(f"\nWatch mode: checking '{args.ckpt}' every {args.watch_interval}s...")
        print("Press Ctrl+C to stop.\n")

        last_mtime = 0
        best_watch_acc = 0.0

        try:
            while True:
                try:
                    current_mtime = os.path.getmtime(args.ckpt)
                except OSError:
                    print(f"[{time.strftime('%H:%M:%S')}] Waiting for checkpoint...")
                    time.sleep(args.watch_interval)
                    continue

                if current_mtime > last_mtime:
                    last_mtime = current_mtime
                    model = MODEL_REGISTRY[args.model](num_classes=6).to(device)
                    try:
                        state_dict, _ = load_checkpoint(args.ckpt, device)
                        model.load_state_dict(state_dict)
                    except Exception as e:
                        print(f"[{time.strftime('%H:%M:%S')}] Failed to load: {e}")
                        time.sleep(args.watch_interval)
                        continue

                    results = evaluate_model(model, test_loader, device)
                    acc = print_report(results, args.model, args.ckpt)
                    if acc > best_watch_acc:
                        best_watch_acc = acc
                        print(f"  >>> New best: {acc:.2%}")

                time.sleep(args.watch_interval)

        except KeyboardInterrupt:
            print(f"\nWatch stopped. Best observed accuracy: {best_watch_acc:.2%}")

    else:
        # One-shot evaluation
        if not os.path.exists(args.ckpt):
            print(f"ERROR: Checkpoint not found: {args.ckpt}")
            print("Old checkpoints in checkpoints/legacy/ have data leakage — do NOT use them.")
            sys.exit(1)

        model = MODEL_REGISTRY[args.model](num_classes=6).to(device)
        state_dict, _ = load_checkpoint(args.ckpt, device)
        model.load_state_dict(state_dict)

        results = evaluate_model(model, test_loader, device)
        print_report(results, args.model, args.ckpt)


if __name__ == "__main__":
    main()
