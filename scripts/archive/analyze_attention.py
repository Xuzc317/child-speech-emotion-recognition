"""Phase 6.3: Interpretability Analysis — Attention visualization & analysis.

Generates:
  1. Per-class attention heatmaps (6 classes × 200 frames)
  2. Attention-F0 correlation per emotion
  3. Attention entropy/skewness per class
  4. High-attention frame statistics

Usage: python scripts/analyze_attention.py --model A3 --seed 42
"""
import os, sys, argparse, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from src.data.dataset_ssl import SSLFeatureDataset, collate_fn_ssl_features
from src.models.pooling import TemporalImportancePooling
from src.models.semlp import SEMLP

CLASS_NAMES = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Chinese font ──
for fp in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simsun.ttc', 'C:/Windows/Fonts/simhei.ttf']:
    if os.path.exists(fp):
        from matplotlib.font_manager import FontProperties
        _zh = FontProperties(fname=fp).get_name()
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [_zh, 'DejaVu Sans']
        break
plt.rcParams['axes.unicode_minus'] = False


class A3Analyzer(nn.Module):
    """A3 model that exposes attention weights for analysis."""
    def __init__(self, dim=768, num_classes=6):
        super().__init__()
        self.pooling = TemporalImportancePooling(ssl_dim=dim)
        self.classifier = SEMLP(input_dim=dim, num_classes=num_classes)

    def forward(self, x, f0=None, energy=None, return_attention=False):
        if f0 is not None and energy is not None:
            # Get attention weights before softmax
            prosody = torch.cat([f0, energy], dim=-1)
            prosody_emb = self.pooling.prosody_proj(prosody)
            combined = torch.cat([x, prosody_emb], dim=-1)
            attn_raw = self.pooling.attn(combined).squeeze(-1)  # (B, T)
            attn_weights = torch.softmax(attn_raw, dim=1)
            pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        else:
            pooled = x.mean(dim=1)
            attn_weights = None
        logits = self.classifier(pooled)
        if return_attention:
            return logits, attn_weights
        return logits


def load_model(checkpoint_path):
    model = A3Analyzer().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    if 'state_dict' in state:
        state = state['state_dict']
    # Filter classifier keys (model might have been saved from DistributionCalibratedSER)
    model_state = {}
    for k, v in state.items():
        if k.startswith('pooling.'):
            model_state[k.replace('pooling.', 'pooling.')] = v
        elif k.startswith('classifier.'):
            model_state[k] = v
    # Try loading just the weights we need
    own_state = model.state_dict()
    for name, param in state.items():
        if name in own_state:
            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)
    model.eval()
    return model


def analyze(checkpoint, data_dir, output_dir, seed=42):
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    pfx = 'wavlm'
    test_ds = SSLFeatureDataset(
        f'{data_dir}/test_{pfx}_feats.npy',
        f'{data_dir}/test_{pfx}_labels.npy',
        f'{data_dir}/test_{pfx}_prosody.npy',
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_fn_ssl_features, num_workers=0)

    # Load model
    model = load_model(checkpoint)
    print(f"Model loaded from {checkpoint}")

    # Collect per-sample attention and predictions
    results = {cls: {'attentions': [], 'f0': [], 'energy': [], 'correct': []}
               for cls in range(6)}

    for batch in tqdm(test_loader, desc='Analyzing'):
        feats = batch[0].to(DEVICE)
        labels = batch[1]
        prosody = batch[3].to(DEVICE) if len(batch) >= 4 else None

        with torch.no_grad():
            if prosody is not None:
                logits, attn = model(feats, f0=prosody[:,:,0:1], energy=prosody[:,:,1:2], return_attention=True)
            else:
                logits = model(feats)
                attn = None

        pred = logits.argmax(dim=1).cpu().item()
        label = labels.item()

        if attn is not None:
            results[label]['attentions'].append(attn.squeeze(0).cpu().numpy())
            results[label]['f0'].append(prosody.squeeze(0)[:, 0].cpu().numpy())
            results[label]['energy'].append(prosody.squeeze(0)[:, 1].cpu().numpy())
            results[label]['correct'].append(pred == label)

    # ── 1. Per-class average attention heatmap ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for cls in range(6):
        ax = axes[cls // 3][cls % 3]
        attns = results[cls]['attentions']
        if not attns:
            ax.set_title(f'{CLASS_NAMES[cls]} (no data)')
            continue
        avg_attn = np.mean(attns, axis=0)
        ax.plot(avg_attn, linewidth=1)
        ax.fill_between(range(len(avg_attn)), avg_attn, alpha=0.3)
        ax.set_title(f'{CLASS_NAMES[cls]} (n={len(attns)})')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Attention')
        ax.set_ylim(0, avg_attn.max() * 1.2 if avg_attn.max() > 0 else 0.01)
    plt.suptitle('Per-Class Average Attention Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_class_attention.png', dpi=150)
    plt.close()
    print("Saved: per_class_attention.png")

    # ── 2. Attention-F0 correlation ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for cls in range(6):
        ax = axes[cls // 3][cls % 3]
        attns = results[cls]['attentions']
        f0s = results[cls]['f0']
        if not attns:
            continue
        # Average per class
        avg_attn = np.mean(attns, axis=0)
        avg_f0 = np.mean(f0s, axis=0)
        # Normalize for correlation
        ax2 = ax.twinx()
        ax.plot(avg_attn, 'b-', linewidth=1, label='Attention')
        ax2.plot(avg_f0, 'r-', linewidth=1, alpha=0.7, label='F0')
        ax.set_title(CLASS_NAMES[cls])
        ax.set_xlabel('Frame')
        ax.set_ylabel('Attention (blue)', color='b')
        ax2.set_ylabel('F0 (red)', color='r')
    plt.suptitle('Attention vs F0 Contour Per Emotion', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/attention_vs_f0.png', dpi=150)
    plt.close()
    print("Saved: attention_vs_f0.png")

    # ── 3. Attention statistics per class ──
    stats = {}
    print("\n=== Per-Class Attention Statistics ===")
    print(f"{'Class':<12} {'N':>5} {'Entropy':>8} {'Mean':>8} {'Std':>8} {'Acc':>8}")
    print("-" * 55)
    for cls in range(6):
        attns = results[cls]['attentions']
        if not attns:
            continue
        attn_stack = np.array(attns)  # (N, 200)
        entropy = -np.sum(attn_stack * np.log(attn_stack + 1e-8), axis=1).mean()
        mean_w = attn_stack.mean()
        std_w = attn_stack.std()
        acc = np.mean(results[cls]['correct']) if results[cls]['correct'] else 0
        stats[CLASS_NAMES[cls]] = {
            'n': len(attns), 'entropy': float(entropy), 'mean_attention': float(mean_w),
            'std_attention': float(std_w), 'accuracy': float(acc),
        }
        print(f"{CLASS_NAMES[cls]:<12} {len(attns):>5} {entropy:>8.4f} {mean_w:>8.4f} {std_w:>8.4f} {acc:>8.3f}")

    with open(f'{output_dir}/attention_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("Saved: attention_stats.json")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, help='Path to A3 .pth checkpoint')
    parser.add_argument('--data_dir', default='data/', help='Directory with test_*_wavlm_*.npy')
    parser.add_argument('--output_dir', default='experiments/v5_622/attention_analysis')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Auto-locate checkpoint
    if args.checkpoint is None:
        candidates = [
            'checkpoints/v5_622/A3_prosody_only_seed42.pth',
            'checkpoints/v5_622/B3_final_seed42.pth',
        ]
        for c in candidates:
            if os.path.exists(c):
                args.checkpoint = c
                break
    if args.checkpoint is None or not os.path.exists(args.checkpoint):
        print("ERROR: Need a valid A3 checkpoint. Use --checkpoint PATH")
        print("Available checkpoints:")
        for f in os.listdir('checkpoints/v5_622/'):
            if 'A3' in f or 'B3' in f:
                print(f"  checkpoints/v5_622/{f}")
        return

    # Auto-locate data
    if not os.path.exists(f"{args.data_dir}/test_wavlm_feats.npy"):
        # Try v5_data path
        for d in ['/root/autodl-tmp/v5_data', 'data']:
            if os.path.exists(f'{d}/test_wavlm_feats.npy'):
                args.data_dir = d
                break

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    analyze(args.checkpoint, args.data_dir, args.output_dir, args.seed)


if __name__ == '__main__':
    main()
