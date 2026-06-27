"""t-SNE visualization: Before vs After distribution-driven pipeline.

For each dataset (C-BESD, FAU Aibo, IEMOCAP), two panels:
  BEFORE:  Raw frozen WavLM last-layer 768-dim → mean-pool (no pipeline modules)
  AFTER:   WavLM → LayerFusion → Adapter → Pooling → SEMLP penultimate 128-dim

Reference: Neumann & Vu (2019, ICASSP) — t-SNE of penultimate classifier layer.

Usage:
    cd /root/autodl-tmp/d-ser
    export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
    export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
    export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav
    python scripts/plot_tsne.py
"""

import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_dataloaders
from src.train import SERModel
from src.models import SSLBackbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ──────────────────────────────────────────────────

CHECKPOINTS = {
    'C-BESD': {
        'ckpt': 'checkpoints/b1/E1-02_s42/best_model.pt',
        'dataset': ['c-besd'],
        'num_classes': 6,
    },
    'FAU Aibo': {
        'ckpt': 'checkpoints/b1/E1-05_s42/best_model.pt',
        'dataset': ['fau-aibo'],
        'num_classes': 4,
    },
    'IEMOCAP': {
        'ckpt': 'checkpoints/b1/E1-08_s42/best_model.pt',
        'dataset': ['iemocap'],
        'num_classes': 4,
    },
}

OUTPUT_DIR = 'paper_draft/figures'
PERPLEXITY = 30
MAX_SAMPLES = 2000


# ── Feature Extractors ──────────────────────────────────────

def _extract_with_collector(dataloader, collector_fn, max_samples=MAX_SAMPLES):
    """Generic feature collector with per-class stratified sampling.

    Ensures all classes are represented by sampling max_samples//n_classes
    per class, then filling remaining quota proportionally.
    """
    from collections import defaultdict
    buckets = defaultdict(list)

    for waveforms, labels, lengths, _, _ in dataloader:
        feats = collector_fn(waveforms, labels, lengths)
        for feat, lbl in zip(feats, labels.numpy()):
            buckets[int(lbl)].append(feat)
        # Stop when we have enough per class
        total = sum(len(v) for v in buckets.values())
        if total >= max_samples * 2:  # oversample then subsample
            break

    # Stratified sampling: at least 10 per class, then proportional fill
    n_classes = len(buckets)
    per_class_min = max(10, max_samples // (n_classes * 2))
    sampled_feats, sampled_labels = [], []

    for cls_idx in sorted(buckets.keys()):
        arr = np.array(buckets[cls_idx])
        n_take = min(len(arr), max_samples // n_classes)
        idxs = np.random.choice(len(arr), size=n_take, replace=False)
        sampled_feats.append(arr[idxs])
        sampled_labels.extend([cls_idx] * n_take)

    feats = np.vstack(sampled_feats)[:max_samples]
    labs = np.array(sampled_labels)[:max_samples]

    # Shuffle
    perm = np.random.permutation(len(labs))
    return feats[perm], labs[perm]


def extract_raw_wavlm(dataloader, max_samples=MAX_SAMPLES):
    """BEFORE: Raw frozen WavLM last-layer 768-dim → mean-pool."""
    backbone = SSLBackbone(model_name='wavlm', frozen=True, device=device)
    backbone.eval()

    def collector(waveforms, labels, lengths):
        wf = waveforms.to(device)
        ln = lengths.to(device)
        with torch.no_grad():
            hidden = backbone(wf, return_all_layers=False)
            B, T, D = hidden.shape
            mask = torch.arange(T, device=device).unsqueeze(0) < ln.unsqueeze(1)
            hidden_m = hidden * mask.unsqueeze(-1).float()
            pooled = hidden_m.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        return pooled.cpu().numpy()

    return _extract_with_collector(dataloader, collector, max_samples)


def extract_pipeline_penultimate(model, dataloader, max_samples=MAX_SAMPLES):
    """AFTER: Full pipeline → SEMLP penultimate 128-dim."""
    model.eval()

    def collector(waveforms, labels, lengths):
        wf = waveforms.to(device)
        ln = lengths.to(device)
        with torch.no_grad():
            _, feats = model(wf, lengths=ln, return_features='penultimate')
        return feats.cpu().numpy()

    return _extract_with_collector(dataloader, collector, max_samples)


# ── Plotting ─────────────────────────────────────────────────

def plot_tsne_panel(ax, features_2d, labels, title, label_names):
    """Plot a single t-SNE panel."""
    cmap = plt.cm.tab10
    for i, name in enumerate(label_names):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=[cmap(i)], label=name, alpha=0.5, s=8, edgecolors='none'
        )
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(markerscale=2, fontsize=7, loc='lower right', framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])


def run_tsne(feats, name):
    """Reduce features to 2D via t-SNE."""
    print(f'  Running t-SNE for {name} ({feats.shape[0]} samples, {feats.shape[1]} dim)...')
    tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=42,
                 n_iter=1000, verbose=0)
    return tsne.fit_transform(feats)


# ── Main ────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3 datasets x 2 conditions = 6 panels in 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(14, 20))
    fig.suptitle(
        't-SNE: Before vs After Distribution-Driven Pipeline\n'
        '(Self-Attention Pooling, seed=42, perplexity=30)',
        fontsize=14, fontweight='bold', y=1.01
    )

    row_labels = ['C-BESD\n(Children Acted)', 'FAU Aibo\n(Children Spontaneous)', 'IEMOCAP\n(Adults Acted)']
    col_labels = ['Before: Raw Frozen WavLM\n(768-dim, mean-pool)', 'After: Distribution-Driven Pipeline\n(WavLM→Fusion→Pooling→SEMLP, 128-dim)']

    for row_idx, (name, cfg) in enumerate(CHECKPOINTS.items()):
        print(f'\n{"="*60}')
        print(f'[{name}]')
        print(f'{"="*60}')

        # DataLoader
        dls = get_dataloaders(cfg['dataset'], batch_size=64, num_workers=4,
                              seed=42, splits=['test'])

        if cfg['num_classes'] == 6:
            label_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
        else:
            label_names = ['angry', 'happy', 'neutral', 'sad']

        # --- BEFORE: Raw WavLM ---
        print('  [BEFORE] Extracting raw WavLM features...')
        feats_before, labels_before = extract_raw_wavlm(dls['test'], MAX_SAMPLES)
        print(f'    {feats_before.shape[0]} samples, {len(set(labels_before))} classes')
        feats_before_2d = run_tsne(feats_before, f'{name} BEFORE')
        plot_tsne_panel(axes[row_idx, 0], feats_before_2d, labels_before,
                        f'{name}\nBefore: Raw WavLM mean-pool', label_names)

        # --- AFTER: Full pipeline ---
        print(f'  [AFTER] Loading checkpoint: {cfg["ckpt"]}')
        ckpt = torch.load(cfg['ckpt'], map_location=device)
        model_config = ckpt.get('config', {
            'pooling_type': 'self_attention', 'num_classes': cfg['num_classes'],
            'ssl_model': 'wavlm', 'pooling_dropout': 0.0,
            'fusion_mode': 'weighted', 'fusion_best_layer': 8,
            'use_adapter': False, 'unfreeze_ssl': False,
        })
        for k in ['fusion_mode', 'fusion_best_layer', 'use_adapter', 'unfreeze_ssl']:
            if k not in model_config:
                model_config[k] = False if k in ('use_adapter', 'unfreeze_ssl') else (
                    'weighted' if k == 'fusion_mode' else 8)

        model = SERModel(model_config).to(device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

        print('  [AFTER] Extracting pipeline penultimate features...')
        feats_after, labels_after = extract_pipeline_penultimate(model, dls['test'], MAX_SAMPLES)
        print(f'    {feats_after.shape[0]} samples, {len(set(labels_after))} classes')
        feats_after_2d = run_tsne(feats_after, f'{name} AFTER')
        plot_tsne_panel(axes[row_idx, 1], feats_after_2d, labels_after,
                        f'{name}\nAfter: Full Pipeline penultimate', label_names)

    # Add column labels
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].set_title(label, fontsize=11, fontweight='bold', pad=20)

    plt.tight_layout()
    combined_path = os.path.join(OUTPUT_DIR, 'fig_tsne_before_after.png')
    fig.savefig(combined_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSaved combined: {combined_path}')

    # Also save individual per-dataset before/after pairs (3 figures, 2 panels each)
    for row_idx, (name, cfg) in enumerate(CHECKPOINTS.items()):
        dls = get_dataloaders(cfg['dataset'], batch_size=64, num_workers=4,
                              seed=42, splits=['test'])
        if cfg['num_classes'] == 6:
            label_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
        else:
            label_names = ['angry', 'happy', 'neutral', 'sad']

        # BEFORE
        feats_b, labels_b = extract_raw_wavlm(dls['test'], MAX_SAMPLES)
        feats_b_2d = run_tsne(feats_b, f'{name} BEFORE')

        # AFTER (need to reload model)
        ckpt = torch.load(cfg['ckpt'], map_location=device)
        model_config = ckpt.get('config', {
            'pooling_type': 'self_attention', 'num_classes': cfg['num_classes'],
            'ssl_model': 'wavlm', 'pooling_dropout': 0.0,
            'fusion_mode': 'weighted', 'fusion_best_layer': 8,
            'use_adapter': False, 'unfreeze_ssl': False,
        })
        for k in ['fusion_mode', 'fusion_best_layer', 'use_adapter', 'unfreeze_ssl']:
            if k not in model_config:
                model_config[k] = False if k in ('use_adapter', 'unfreeze_ssl') else (
                    'weighted' if k == 'fusion_mode' else 8)
        model = SERModel(model_config).to(device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        feats_a, labels_a = extract_pipeline_penultimate(model, dls['test'], MAX_SAMPLES)
        feats_a_2d = run_tsne(feats_a, f'{name} AFTER')

        fig_pair, axes_pair = plt.subplots(1, 2, figsize=(14, 6.5))
        plot_tsne_panel(axes_pair[0], feats_b_2d, labels_b,
                        f'Before: Raw Frozen WavLM (768-dim, mean-pool)', label_names)
        plot_tsne_panel(axes_pair[1], feats_a_2d, labels_a,
                        f'After: Full Pipeline (128-dim penultimate)', label_names)
        fig_pair.suptitle(f'{name} — t-SNE: Before vs After Distribution-Driven Pipeline',
                          fontsize=13, fontweight='bold')
        plt.tight_layout()
        safe_name = name.replace(' ', '_')
        pair_path = os.path.join(OUTPUT_DIR, f'fig_tsne_{safe_name}_before_after.png')
        fig_pair.savefig(pair_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig_pair)
        print(f'Saved pair: {pair_path}')

    print('\nAll done!')


if __name__ == '__main__':
    main()
