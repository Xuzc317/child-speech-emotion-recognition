"""Evaluate DrseCNN checkpoint on BESD dataset — clean (no data leakage).

Uses pre-split test data from preprocess.py. Test set contains only original
(non-augmented) features from WAV files NOT seen during training.
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)

# ---- Config ----
CKPT_PATH = "checkpoints/best_Drsecnn_model_0.8599.pth"
TEST_DATA = "test_data.npy"
TEST_LABEL = "test_labels.npy"
BATCH_SIZE = 128
NUM_CLASSES = 6
CLASS_NAMES = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ---- Step 1: Load pre-split test data ----
if not os.path.exists(TEST_DATA) or not os.path.exists(TEST_LABEL):
    print("ERROR: Clean test data not found.")
    print("Please run: python src/data/preprocess.py")
    print("Then re-train the model with: python src/training/train.py")
    sys.exit(1)

class NpyDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.int64)


def collate_fn(batch):
    inputs = torch.stack([item[0] for item in batch])
    inputs = inputs.view(inputs.shape[0], 1, -1)
    labels = torch.stack([item[1] for item in batch])
    return inputs, labels


test_dataset = NpyDataset(TEST_DATA, TEST_LABEL)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
print(f"Test set: {len(test_dataset)} samples (original features, no augmentation)")
print(f"Label distribution: {np.bincount(np.load(TEST_LABEL))}")

# ---- Step 2: Load model ----
from src.models.models import DrseCNN

model = DrseCNN(num_classes=NUM_CLASSES).to(DEVICE)
state_dict = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded: {CKPT_PATH}")

# ---- Step 3: Evaluate ----
all_preds = []
all_labels = []
total_correct = 0
total_samples = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = total_correct / total_samples
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
)
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average='macro', zero_division=0
)
weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average='weighted', zero_division=0
)
cm = confusion_matrix(all_labels, all_preds)

# ---- Step 4: Report ----
print("\n" + "=" * 70)
print("DrseCNN Model Evaluation (best_Drsecnn_model_0.8599.pth)")
print("CLEAN evaluation — no data leakage")
print("=" * 70)

print(f"\nTotal samples: {total_samples}")
print(f"Correct predictions: {total_correct}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\n--- Macro Average ---")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")

print(f"\n--- Weighted Average ---")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall:    {weighted_recall:.4f}")
print(f"Weighted F1:        {weighted_f1:.4f}")

print(f"\n--- Per-class ---")
print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<8}")
print("-" * 55)
for i, name in enumerate(CLASS_NAMES):
    count = np.sum(all_labels == i)
    print(f"{name:<12} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {count:<8}")

print(f"\n--- Confusion Matrix ---")
header = "        " + " ".join(f"{n[:6]:>7}" for n in CLASS_NAMES)
print(header)
for i, name in enumerate(CLASS_NAMES):
    row = f"{name:<8}" + " ".join(f"{cm[i][j]:>7d}" for j in range(NUM_CLASSES))
    print(row)
