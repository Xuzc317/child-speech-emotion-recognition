"""
Generate confusion matrix from best model checkpoint (A3 Full Fine-tune, seed 456).
"""
import os, sys, glob as _glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = r'D:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER'
sys.path.insert(0, os.path.join(ROOT, 'src'))

from data.dataset_ssl import SSLFeatureDataset, collate_fn_ssl_features
from models.adapter import AcousticCalibrationAdapter
from models.pooling import TemporalImportancePooling
from models.semlp import SEMLP

# ── Chinese font for matplotlib ──
_font_candidates = ['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simsun.ttc','C:/Windows/Fonts/simhei.ttf']
_font_candidates += _glob.glob('C:/Windows/Fonts/*.ttf') + _glob.glob('C:/Windows/Fonts/*.ttc')
_zh_name = 'sans-serif'
for _fp in _font_candidates:
    if os.path.exists(_fp):
        from matplotlib.font_manager import FontProperties as _FP
        _zh_name = _FP(fname=_fp).get_name()
        break
plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':[_zh_name,'DejaVu Sans'],
    'axes.unicode_minus':False,'font.size':12,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Model (same as A3) ──
class CalibratedSER(nn.Module):
    def __init__(self, dim=768, num_classes=6):
        super().__init__()
        self.adapter = AcousticCalibrationAdapter(dim=dim)
        self.pooling = TemporalImportancePooling(ssl_dim=dim)
        self.classifier = SEMLP(input_dim=dim, num_classes=num_classes)

    def forward(self, x, f0=None, energy=None):
        x = self.adapter(x)
        if f0 is not None and energy is not None:
            x = self.pooling(x, f0, energy)
        else:
            x = x.mean(dim=1)
        return self.classifier(x)

# ── Load data ──
print("Loading test data...")
test_dataset = SSLFeatureDataset(
    os.path.join(ROOT, 'data', 'test_wavlm_feats.npy'),
    os.path.join(ROOT, 'data', 'test_wavlm_labels.npy'),
    os.path.join(ROOT, 'data', 'test_wavlm_prosody.npy'),
)
print(f"  Test samples: {len(test_dataset)}")
print(f"  Feats shape: {test_dataset.datas.shape}")
print(f"  Labels shape: {test_dataset.labels.shape}")
if test_dataset.prosody is not None:
    print(f"  Prosody shape: {test_dataset.prosody.shape}")

test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False,
    collate_fn=collate_fn_ssl_features, num_workers=0, pin_memory=True,
)

# ── Load model ──
print("Loading model checkpoint...")
model = CalibratedSER(dim=768, num_classes=6).to(device)
ckpt = torch.load(os.path.join(ROOT, 'checkpoints', 'best_ser_model.pth'),
                  map_location=device, weights_only=True)
model.load_state_dict(ckpt, strict=False)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"  Model params: {total_params:,}")

# ── Inference ──
print("Running inference on test set...")
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        prosody_batch = batch[3].to(device) if len(batch) >= 4 else None
        if prosody_batch is not None:
            f0 = prosody_batch[:, :, 0:1]
            energy = prosody_batch[:, :, 1:2]
        else:
            f0 = energy = None
        outputs = model(inputs, f0=f0, energy=energy)
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

acc = (all_preds == all_labels).mean()
print(f"  Test accuracy: {acc:.4f} ({acc*100:.2f}%)")

# ── Confusion matrix ──
class_names = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD']
n_classes = len(class_names)
cm = np.zeros((n_classes, n_classes), dtype=int)
for t, p in zip(all_labels, all_preds):
    cm[t, p] += 1

print("\nConfusion Matrix (rows=true, cols=pred):")
print("           " + "  ".join(f"{n:>7}" for n in class_names))
for i, name in enumerate(class_names):
    print(f"{name:>8}  " + "  ".join(f"{cm[i,j]:>7}" for j in range(n_classes)))

# Per-class metrics
print("\nPer-class metrics:")
for i, name in enumerate(class_names):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  {name:>8}: Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  Support={cm[i,:].sum()}")

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Raw counts ---
ax = axes[0]
im = ax.imshow(cm, cmap='Blues')
for i in range(n_classes):
    for j in range(n_classes):
        text = ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                       fontsize=14,
                       color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
ax.set_xticklabels(class_names, fontsize=12)
ax.set_yticklabels(class_names, fontsize=12)
ax.set_xlabel('Predicted', fontsize=14)
ax.set_ylabel('True', fontsize=14)
ax.set_title(f'Confusion Matrix (Counts)\nTest Acc = {acc*100:.2f}%', fontsize=16, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.78)

# --- Normalized (recall) ---
ax = axes[1]
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
for i in range(n_classes):
    for j in range(n_classes):
        val = cm_norm[i, j]
        text = ax.text(j, i, f'{val:.2f}' if val > 0 else '0',
                       ha='center', va='center', fontsize=13,
                       color='white' if val > 0.5 else 'black')
ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
ax.set_xticklabels(class_names, fontsize=12)
ax.set_yticklabels(class_names, fontsize=12)
ax.set_xlabel('Predicted', fontsize=14)
ax.set_ylabel('True', fontsize=14)
ax.set_title(f'Confusion Matrix (Normalized by Row)\nModel: A3 Full Fine-tune | WavLM + Adapter + Prosody Pooling', fontsize=15, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.78)

plt.tight_layout()
out_path = os.path.join(ROOT, 'assets', 'figures', 'confusion_matrix_A3.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out_path}")

# ── Per-class bar chart ──
fig2, ax2 = plt.subplots(figsize=(10, 5))
x = np.arange(n_classes)
width = 0.25
precisions, recalls, f1s = [], [], []
for i in range(n_classes):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
    recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    f1s.append(2 * precisions[-1] * recalls[-1] / (precisions[-1] + recalls[-1])
               if (precisions[-1] + recalls[-1]) > 0 else 0)

ax2.bar(x - width, [p*100 for p in precisions], width, color='#2196F3', label='Precision', edgecolor='white')
ax2.bar(x, [r*100 for r in recalls], width, color='#4CAF50', label='Recall', edgecolor='white')
ax2.bar(x + width, [f*100 for f in f1s], width, color='#FF9800', label='F1-score', edgecolor='white')
ax2.set_xticks(x); ax2.set_xticklabels(class_names, fontsize=13)
ax2.set_ylabel('Percentage (%)', fontsize=14)
ax2.set_title(f'Per-Class Metrics\nModel: A3 Full Fine-tune | Overall Test Acc = {acc*100:.2f}%', fontsize=16, fontweight='bold')
ax2.legend(fontsize=12)
ax2.set_ylim(0, 105)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
out_path2 = os.path.join(ROOT, 'assets', 'figures', 'per_class_metrics_A3.png')
fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: {out_path2}")
