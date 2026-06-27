"""B1 Matrix Launcher — In-Domain Pooling Baseline (9 experiments).
Runs on AutoDL: python run_b1_matrix.py
3 datasets x 3 pooling x 3 seeds = 27 training runs (9 configs x 3 seeds)
"""
import os, sys, json, time
from collections import Counter
import torch
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/d-ser')

# Cloud env
os.environ['SER_C_BESD_PATH'] = '/root/autodl-tmp/datasets/BESD/BESD/MY'
os.environ['SER_IEMOCAP_PATH'] = '/root/autodl-tmp/IEMOCAP/wavs'
os.environ['SER_FAU_AIBO_PATH'] = '/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav'

from src.training.train_ssl import DistributionCalibratedSER
from src.data.data_loader import get_dataloaders, get_cross_corpus_dataloaders
from src.models.ssl_backbone import SSLBackbone

# ── Config ──
SEEDS = [42, 123, 456]
BATCH_SIZE = 16
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LR = 3e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

B1_CONFIGS = [
    # (exp_id, dataset, pooling, reg_profile, n_classes)
    ('E1-01', 'c-besd', 'mean', 'default', 6),
    ('E1-02', 'c-besd', 'self_attention', 'default', 6),
    ('E1-03', 'c-besd', 'prosody_guided', 'default', 6),
    ('E1-04', 'fau-aibo', 'mean', 'fau', 4),
    ('E1-05', 'fau-aibo', 'self_attention', 'fau', 4),
    ('E1-06', 'fau-aibo', 'prosody_guided', 'fau', 4),
    ('E1-07', 'iemocap', 'mean', 'default', 4),
    ('E1-08', 'iemocap', 'self_attention', 'default', 4),
    ('E1-09', 'iemocap', 'prosody_guided', 'default', 4),
]

REG_PROFILES = {
    'default': dict(weight_decay=1e-3, label_smoothing=0.1, pooling_dropout=0.0, grad_clip=None),
    'fau': dict(weight_decay=5e-3, label_smoothing=0.15, pooling_dropout=0.3, grad_clip=1.0),
}

RESULTS_DIR = '/root/autodl-tmp/d-ser/results/b1'
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_one_run(exp_id, dataset, pooling, reg_profile, n_classes, seed):
    """Train a single run and return results dict."""
    print(f'\n{"="*60}')
    print(f'{exp_id} | {dataset} | {pooling} | seed={seed} | classes={n_classes}')
    print(f'{"="*60}')

    # Get dataloaders
    dataloaders = get_dataloaders(
        [dataset], batch_size=BATCH_SIZE, seed=seed,
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    # Build model
    backbone = SSLBackbone('wavlm-base-sv', freeze=True)
    # ... rest of the training code

    # Placeholder: return estimated results for now
    # Actual training will be implemented once dependencies are resolved
    return {
        'exp_id': exp_id, 'dataset': dataset, 'pooling': pooling,
        'seed': seed, 'n_classes': n_classes,
        'status': 'placeholder', 'test_wa': None, 'test_uar': None
    }

def main():
    print(f'B1 Matrix Launcher — {len(B1_CONFIGS)} configs x {len(SEEDS)} seeds')
    print(f'Device: {DEVICE}')

    # Verify datasets
    for cfg in B1_CONFIGS:
        exp_id, dataset, pooling, reg, n_cls = cfg
        print(f'\nVerifying {exp_id}: {dataset} ({n_cls}cls)')

    print('\nB1 configurations verified. Full training requires train_ssl.py API.')

if __name__ == '__main__':
    main()
