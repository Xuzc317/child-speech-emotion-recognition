"""Cloud experiment runner for v5 final protocol.

Runs all needed experiments under the new split (outer 8:2 + inner val).

Main ablation (3 configs × 3 seeds = 9 runs):
  A1:  WavLM + mean pooling + SEMLP
  A2b: WavLM + Adapter(random init) + mean pooling + SEMLP
  B3:  WavLM + Adapter(random init) + Prosody Pooling + SEMLP  [FINAL MODEL]

Augmentation sensitivity (4 configs × 3 seeds = 12 runs):
  C1: B3 + no augmentation (clean train)
  C2: B3 + adult augmentation (pitch ±6, stretch 0.7-1.3)
  C3: B3 + child-constrained augmentation (pitch ±3, stretch 0.85-1.15)
  C4: B3 + extreme augmentation (pitch ±12, stretch 0.5-1.5)

Total: 7 configs × 3 seeds = 21 runs
Estimated time on RTX 4090D: ~2-3 hours

Usage:
  python scripts/run_experiments_v5.py --data_dir data/ --output_dir logs/v5/
"""

import sys, os, json, time, argparse, numpy as np, torch

# Allow importing from project root regardless of launch directory
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from src.data.dataset_ssl import SSLFeatureDataset, collate_fn_ssl_features
from src.training.train_ssl import DistributionCalibratedSER
from src.utils.experiment_logger import ExperimentLogger
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

# ── Config ──────────────────────────────────────────────

SEEDS = [42, 123, 456]
EPOCHS = 80
PATIENCE = 15
BATCH_SIZE = 128
LR = 3e-4
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.1

# Augmentation config label → feature file suffix
AUG_SUFFIX = {
    'C1_none':    'C1_none',
    'C2_adult':   'C2_adult',
    'C3_child':   'C3_child',
    'C4_extreme': 'C4_extreme',
}


# ── Helpers ─────────────────────────────────────────────

def build_dataloader(feat_path, label_path, prosody_path=None, bs=BATCH_SIZE, shuffle=False):
    ds = SSLFeatureDataset(feat_path, label_path, prosody_path)
    return DataLoader(ds, bs, shuffle=shuffle, collate_fn=collate_fn_ssl_features,
                      num_workers=4, pin_memory=True)


def evaluate(model, loader, use_prosody, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            pb = batch[3].to(device) if len(batch) >= 4 else None
            if use_prosody and pb is not None:
                outputs = model(inputs, f0=pb[:, :, 0:1], energy=pb[:, :, 1:2])
            else:
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_one_seed(model, train_loader, val_loader, use_prosody, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val, patience_ct = 0.0, 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            pb = batch[3].to(device) if len(batch) >= 4 else None
            optimizer.zero_grad()
            if use_prosody and pb is not None:
                outputs = model(inputs, f0=pb[:, :, 0:1], energy=pb[:, :, 1:2])
            else:
                outputs = model(inputs)
            criterion(outputs, labels).backward()
            optimizer.step()

        val_acc = evaluate(model, val_loader, use_prosody, device)
        scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            patience_ct = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ct += 1
        if patience_ct >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return best_val


def run_config(name, train_feat, train_label, val_feat, val_label, test_feat, test_label,
               train_prosody, val_prosody, test_prosody,
               use_adapter, use_prosody, adapter_init, device):
    """Train and evaluate one configuration across 3 seeds."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()

    results = {}
    for seed in SEEDS:
        # Rebuild dataloaders each seed for fresh shuffling
        train_loader = build_dataloader(train_feat, train_label, train_prosody, bs=BATCH_SIZE, shuffle=True)
        val_loader   = build_dataloader(val_feat,   val_label,   val_prosody,   bs=BATCH_SIZE, shuffle=False)
        test_loader  = build_dataloader(test_feat,  test_label,  test_prosody,  bs=BATCH_SIZE, shuffle=False)

        config = {'use_backbone': False, 'ssl_model': 'wavlm',
                  'frozen_backbone': True, 'use_adapter': use_adapter,
                  'use_prosody': use_prosody}
        model = DistributionCalibratedSER(config, adapter_init=adapter_init).to(device)

        val_acc = train_one_seed(model, train_loader, val_loader, use_prosody, device, seed)
        test_acc = evaluate(model, test_loader, use_prosody, device)

        results[f'seed_{seed}'] = {'val': float(val_acc), 'test': float(test_acc)}
        print(f"  seed={seed}: val={val_acc:.4f} test={test_acc:.4f}")

    vv = [r['val'] for r in results.values()]
    tv = [r['test'] for r in results.values()]
    elapsed = time.time() - t0
    print(f"  Mean: val={np.mean(vv):.4f}±{np.std(vv):.4f}  test={np.mean(tv):.4f}±{np.std(tv):.4f}  ({elapsed:.0f}s)")
    return results


# ── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='v5 Cloud Experiment Runner')
    parser.add_argument('--data_dir', default='data/', help='Directory with pre-extracted .npy files')
    parser.add_argument('--output_dir', default='logs/v5/', help='Output directory for results')
    parser.add_argument('--device', default='cuda', help='Device (cuda / cpu)')
    parser.add_argument('--skip_aug', action='store_true', help='Skip augmentation experiments (C1-C4)')
    parser.add_argument('--prefix', default='wavlm', help='Feature file prefix (wavlm / e2v)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f'Device: {torch.cuda.get_device_name(0)}')
    else:
        print('Device: CPU')

    os.makedirs(args.output_dir, exist_ok=True)

    data = args.data_dir
    pfx = args.prefix

    # ── Verify required files exist ──
    required = [
        f'{data}train_{pfx}_feats.npy', f'{data}train_{pfx}_labels.npy',
        f'{data}val_{pfx}_feats.npy',   f'{data}val_{pfx}_labels.npy',
        f'{data}test_{pfx}_feats.npy',  f'{data}test_{pfx}_labels.npy',
        f'{data}train_{pfx}_prosody.npy',
        f'{data}val_{pfx}_prosody.npy',
        f'{data}test_{pfx}_prosody.npy',
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print('ERROR: Missing required files:')
        for f in missing:
            print(f'  {f}')
        print('\nRun the following on cloud first:')
        print(f'  python scripts/extract_ssl_features.py --model wavlm --prefix {pfx} --device cuda --output_dir {data}')
        print(f'  python scripts/extract_prosody_features.py --prefix {pfx} --output_dir {data}')
        print(f'  (adapter_init.npz is optional — only needed for stat-prior A2 experiment)')
        sys.exit(1)

    # Paths for clean features
    train_feat = f'{data}train_{pfx}_feats.npy'
    train_label = f'{data}train_{pfx}_labels.npy'
    val_feat = f'{data}val_{pfx}_feats.npy'
    val_label = f'{data}val_{pfx}_labels.npy'
    test_feat = f'{data}test_{pfx}_feats.npy'
    test_label = f'{data}test_{pfx}_labels.npy'
    train_prosody = f'{data}train_{pfx}_prosody.npy'
    val_prosody = f'{data}val_{pfx}_prosody.npy'
    test_prosody = f'{data}test_{pfx}_prosody.npy'

    # Load adapter_init (optional — only needed for stat-prior A2; random-init A2b/B3 don't use it)
    adapter_init_path = f'{data}adapter_init.npz'
    if os.path.exists(adapter_init_path):
        adapter_init = dict(np.load(adapter_init_path))
        print('Loaded adapter_init.npz (for stat-prior experiments)')
    else:
        adapter_init = None
        print('adapter_init.npz NOT found — stat-prior experiments will be skipped, random-init OK')

    # Print dataset info
    for split, fp, lp in [('Train', train_feat, train_label), ('Val', val_feat, val_label), ('Test', test_feat, test_label)]:
        feats = np.load(fp)
        labels = np.load(lp)
        unique_cls, counts = np.unique(labels, return_counts=True)
        cls_str = ', '.join(f'c{c}={n}' for c, n in zip(unique_cls, counts))
        print(f'{split}: {len(labels)} samples, shape={feats.shape}, classes=[{cls_str}]')

    all_results = {'meta': {'protocol': 'outer 8:2 + inner val',
                             'split_function': 'split_speakers_7_3_with_inner_val',
                             'seeds': SEEDS, 'batch_size': BATCH_SIZE,
                             'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
                             'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
                   'experiments': {}}

    # ═══════════════════════════════════════════════════════
    # Part 1: Main Ablation (A1, A2b, B3)
    # ═══════════════════════════════════════════════════════

    print('\n' + '='*70)
    print('  PART 1: MAIN ABLATION STUDY')
    print('='*70)

    # A1: Baseline — no adapter, mean pooling
    all_results['experiments']['A1_baseline'] = run_config(
        'A1_baseline', train_feat, train_label, val_feat, val_label, test_feat, test_label,
        None, None, None,
        use_adapter=False, use_prosody=False, adapter_init=None, device=device)

    # A2b: Adapter random init + mean pooling
    all_results['experiments']['A2b_adapter'] = run_config(
        'A2b_adapter', train_feat, train_label, val_feat, val_label, test_feat, test_label,
        None, None, None,
        use_adapter=True, use_prosody=False, adapter_init=None, device=device)

    # B3: Adapter random init + Prosody Pooling (FINAL MODEL)
    all_results['experiments']['B3_final'] = run_config(
        'B3_final', train_feat, train_label, val_feat, val_label, test_feat, test_label,
        train_prosody, val_prosody, test_prosody,
        use_adapter=True, use_prosody=True, adapter_init=None, device=device)

    # ═══════════════════════════════════════════════════════
    # Part 2: Augmentation Sensitivity (C1-C4)
    # ═══════════════════════════════════════════════════════

    if not args.skip_aug:
        print('\n' + '='*70)
        print('  PART 2: AUGMENTATION SENSITIVITY ANALYSIS')
        print('='*70)

        # Check augmented features exist
        aug_configs = ['C1_none', 'C2_adult', 'C3_child', 'C4_extreme']
        aug_missing = []
        for cfg in aug_configs:
            fp = f'{data}train_{cfg}_{pfx}_feats.npy'
            if not os.path.exists(fp):
                aug_missing.append(fp)
        if aug_missing:
            print('WARNING: Augmented features not found:')
            for f in aug_missing:
                print(f'  {f}')
            print('Skipping augmentation experiments. Run:')
            print('  python scripts/extract_augmented_features.py --device cuda')
        else:
            for cfg in aug_configs:
                aug_train_feat = f'{data}train_{cfg}_{pfx}_feats.npy'
                aug_train_label = f'{data}train_{cfg}_{pfx}_labels.npy'
                # Augmentation experiments use clean val/test (fair eval)
                all_results['experiments'][cfg] = run_config(
                    cfg, aug_train_feat, aug_train_label,
                    val_feat, val_label, test_feat, test_label,
                    train_prosody, val_prosody, test_prosody,
                    use_adapter=True, use_prosody=True, adapter_init=None, device=device)

    # ═══════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════

    result_path = os.path.join(args.output_dir, 'v5_results.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Print final table ──
    print('\n' + '='*80)
    print('  FINAL RESULTS TABLE (v5 protocol: outer 8:2 + inner val)')
    print('='*80)
    print(f'{"Experiment":<20} {"Val Mean":>10} {"Val Std":>8} {"Test Mean":>10} {"Test Std":>8}')
    print('-'*58)
    for name, data in all_results['experiments'].items():
        vv = [d['val'] for d in data.values()]
        tv = [d['test'] for d in data.values()]
        print(f'{name:<20} {np.mean(vv):>9.4f}±{np.std(vv):<7.4f} {np.mean(tv):>9.4f}±{np.std(tv):<7.4f}')

    print(f'\nResults saved to: {result_path}')
    print(f'Done at {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
