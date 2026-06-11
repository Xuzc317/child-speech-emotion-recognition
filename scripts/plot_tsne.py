"""t-SNE visualization of pooled features from E1 best self_attention models.

Three datasets side-by-side: C-BESD (6-class), FAU Aibo (4-class), IEMOCAP (4-class).
Uses the same pooling method (self_attention) for fair cross-dataset comparison.

Usage:
    python scripts/plot_tsne.py
    # Or on AutoDL:
    cd /root/autodl-tmp/d-ser && python scripts/plot_tsne.py
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ──────────────────────────────────────────────────

CHECKPOINTS = {
    'C-BESD': {
        'ckpt': 'checkpoints/b1/E1-02_s42/best_model.pt',
        'dataset': ['c-besd'],
        'num_classes': 6,
        'reg_profile': 'default',
    },
    'FAU Aibo': {
        'ckpt': 'checkpoints/b1/E1-05_s42/best_model.pt',
        'dataset': ['fau-aibo'],
        'num_classes': 4,
        'reg_profile': 'fau',
    },
    'IEMOCAP': {
        'ckpt': 'checkpoints/b1/E1-08_s42/best_model.pt',
        'dataset': ['iemocap'],
        'num_classes': 4,
        'reg_profile': 'default',
    },
}

OUTPUT_DIR = 'paper_draft/figures'
PERPLEXITY = 30
MAX_SAMPLES = 2000  # per dataset — t-SNE is O(n^2)


# ── Helpers ─────────────────────────────────────────────────

def extract_features(model, dataloader, max_samples=MAX_SAMPLES):
    """Run inference and collect pooled features + labels."""
    model.eval()
    all_features, all_labels = [], []
    count = 0

    with torch.no_grad():
        for waveforms, labels, lengths, _, _ in dataloader:
            waveforms = waveforms.to(device)
            lengths = lengths.to(device)
            _, pooled = model(waveforms, lengths=lengths, return_features=True)
            all_features.append(pooled.cpu().numpy())
            all_labels.extend(labels.numpy())
            count += len(labels)
            if count >= max_samples:
                break

    feats = np.vstack(all_features)[:max_samples]
    labs = np.array(all_labels)[:max_samples]
    return feats, labs


def plot_tsne_panel(ax, features_2d, labels, title, label_names):
    """Plot a single t-SNE panel."""
    n_classes = len(label_names)
    cmap = plt.cm.tab10
    for i, name in enumerate(label_names):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=[cmap(i)], label=name, alpha=0.5, s=8, edgecolors='none'
        )
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(markerscale=2, fontsize=8, loc='lower right', framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])


# ── Main ────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
    fig.suptitle(
        't-SNE of Pooled Features (Self-Attention Pooling, seed=42)',
        fontsize=15, fontweight='bold', y=1.01
    )

    for ax, (name, cfg) in zip(axes, CHECKPOINTS.items()):
        print(f'[{name}] Loading checkpoint: {cfg["ckpt"]}')
        ckpt = torch.load(cfg['ckpt'], map_location=device)

        # Build model from saved config (fallback to defaults for legacy checkpoints)
        model_config = ckpt.get('config', {
            'pooling_type': 'self_attention',
            'num_classes': cfg['num_classes'],
            'ssl_model': 'wavlm',
            'pooling_dropout': 0.0,
            'fusion_mode': 'weighted',
            'fusion_best_layer': 8,
            'use_adapter': False,
            'unfreeze_ssl': False,
        })
        # Ensure keys exist for legacy checkpoints
        for k in ['fusion_mode', 'fusion_best_layer', 'use_adapter', 'unfreeze_ssl']:
            if k not in model_config:
                model_config[k] = False if k in ('use_adapter', 'unfreeze_ssl') else (
                    'weighted' if k == 'fusion_mode' else 8)

        model = SERModel(model_config).to(device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

        # DataLoader: test split only
        dls = get_dataloaders(
            cfg['dataset'], batch_size=64, num_workers=4,
            seed=42, splits=['test'],
        )

        print(f'  Extracting features (max {MAX_SAMPLES})...')
        feats, labels = extract_features(model, dls['test'], MAX_SAMPLES)
        print(f'  Collected {len(feats)} samples, {len(set(labels))} classes')

        # t-SNE
        print(f'  Running t-SNE (perplexity={PERPLEXITY})...')
        tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=42,
                     n_iter=1000, verbose=0)
        feats_2d = tsne.fit_transform(feats)

        # Labels
        if cfg['num_classes'] == 6:
            label_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
        else:
            label_names = ['angry', 'happy', 'neutral', 'sad']

        plot_tsne_panel(ax, feats_2d, labels, name, label_names)
        print(f'  Done.')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig_tsne_e1_pooled_features.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'\nSaved: {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
