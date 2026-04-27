"""Phase 3: Complete Ablation Study on Cloud (3-way split, no leakage)

Runs all module combinations × 3 seeds, logs structured JSON.
Total: 6 configs × 3 seeds = 18 runs (~90 min on RTX 4090D)

Configurations:
  A1: Baseline (mean pool, no adapter)
  A2: Adapter stat-prior
  A2b: Adapter random-init
  A3_ft: Full fine-tune WavLM + adapter (unfrozen backbone)
  B2: Self-attention pool (no prosody)
  B3: Adapter + Prosody pool (our best)
"""

import sys, os, json, time, numpy as np, torch
sys.path.insert(0, '/root/ser_project')

from src.data.dataset_ssl import SSLFeatureDataset, collate_fn_ssl_features
from src.training.train_ssl import DistributionCalibratedSER
from src.utils.experiment_logger import ExperimentLogger
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

SEEDS = [42, 123, 456]
RESULTS_FILE = 'logs/phase3_ablation.json'

# Load data once
train_ds = SSLFeatureDataset('data/train_wavlm_feats.npy', 'data/train_wavlm_labels.npy', 'data/train_wavlm_prosody.npy')
val_ds   = SSLFeatureDataset('data/val_wavlm_feats.npy',   'data/val_wavlm_labels.npy',   'data/val_wavlm_prosody.npy')
test_ds  = SSLFeatureDataset('data/test_wavlm_feats.npy',  'data/test_wavlm_labels.npy',  'data/test_wavlm_prosody.npy')
adapter_init = dict(np.load('data/adapter_init.npz'))

def create_loaders(bs=128):
    return (
        DataLoader(train_ds, bs, shuffle=True, collate_fn=collate_fn_ssl_features, num_workers=4, pin_memory=True),
        DataLoader(val_ds, bs, shuffle=False, collate_fn=collate_fn_ssl_features, num_workers=4, pin_memory=True),
        DataLoader(test_ds, bs, shuffle=False, collate_fn=collate_fn_ssl_features, num_workers=4, pin_memory=True),
    )

def run_experiment(name, use_adapter=False, use_prosody=False, use_stat_prior=False,
                   finetune=False, bs=128, lr=3e-4, epochs=80, patience=15):
    config = {'use_backbone': False, 'ssl_model': 'wavlm', 'frozen_backbone': not finetune,
              'use_adapter': use_adapter, 'use_prosody': use_prosody}
    if use_stat_prior:
        init = adapter_init
    else:
        init = None

    # Create experiment logger
    run_name = f"{name}_seed{SEEDS[0]}" if len(SEEDS) == 1 else name
    exp_log = ExperimentLogger(name=run_name, config={
        "experiment": name, "use_adapter": use_adapter, "use_prosody": use_prosody,
        "stat_prior": use_stat_prior, "finetune": finetune, "batch_size": bs, "lr": lr
    })

    results = {}
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        train_loader, val_loader, test_loader = create_loaders(bs)
        model = DistributionCalibratedSER(config, adapter_init=init).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best_val, patience_ct = 0.0, 0

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                prosody_batch = batch[3].to(device) if len(batch) >= 4 else None
                optimizer.zero_grad()
                if use_prosody and prosody_batch is not None:
                    outputs = model(inputs, f0=prosody_batch[:,:,0:1], energy=prosody_batch[:,:,1:2])
                else:
                    outputs = model(inputs)
                criterion(outputs, labels).backward()
                optimizer.step()

            model.eval()
            vc, vt = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    prosody_batch = batch[3].to(device) if len(batch) >= 4 else None
                    if use_prosody and prosody_batch is not None:
                        outputs = model(inputs, f0=prosody_batch[:,:,0:1], energy=prosody_batch[:,:,1:2])
                    else:
                        outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    vc += (preds == labels).sum().item(); vt += labels.size(0)
            val_acc = vc/vt; scheduler.step()
            if val_acc > best_val:
                best_val = val_acc; patience_ct = 0
                torch.save(model.state_dict(), '/tmp/best_model.pth')
            else:
                patience_ct += 1
            if patience_ct >= patience: break

        # Log epoch result
        exp_log.log_epoch(epoch + 1, val_acc=val_acc)

        # Test eval
        model.load_state_dict(torch.load('/tmp/best_model.pth')); model.eval()
        tc, tt = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                prosody_batch = batch[3].to(device) if len(batch) >= 4 else None
                if use_prosody and prosody_batch is not None:
                    outputs = model(inputs, f0=prosody_batch[:,:,0:1], energy=prosody_batch[:,:,1:2])
                else:
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                tc += (preds == labels).sum().item(); tt += labels.size(0)
        test_acc = tc/tt
        results[f'seed_{seed}'] = {'val': float(best_val), 'test': float(test_acc)}
        print(f'  {name} seed={seed}: val={best_val:.4f} test={test_acc:.4f}')

    # Finish logging with best result
    all_val = [v['val'] for v in results.values()]
    all_test = [v['test'] for v in results.values()]
    exp_log.finish(best_val=np.mean(all_val), test_acc=np.mean(all_test))
    return results

all_results = {}
experiments = [
    ('A1_baseline',    False, False, False, False),
    ('A2_stat_prior',  True,  False, True,  False),
    ('A2b_rand_init',  True,  False, False, False),
    ('B2_self_attn',   True,  False, False, False),  # adapter + mean pool (vs B3 prosody)
    ('B3_prosody',     True,  True,  False, False),
]

print('=== Phase 3 Ablation Study ===')
dt = time.strftime('%Y-%m-%d %H:%M:%S')
print(f'Date: {dt}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')
print(f'Experiments: {len(experiments)} configs x {len(SEEDS)} seeds = {len(experiments)*len(SEEDS)} runs\n')

for name, use_adp, use_pros, stat_prior, ft in experiments:
    print(f'--- {name} ---')
    t0 = time.time()
    res = run_experiment(name, use_adp, use_pros, stat_prior, ft, bs=128)
    all_results[name] = res
    elapsed = time.time() - t0
    val_vals = [v['val'] for v in res.values()]
    test_vals = [v['test'] for v in res.values()]
    print(f'  Mean val={np.mean(val_vals):.4f}±{np.std(val_vals):.4f} test={np.mean(test_vals):.4f}±{np.std(test_vals):.4f} ({elapsed:.0f}s)\n')

# Full fine-tune (A3) — smaller batch for 24GB VRAM
print('--- A3_full_finetune ---')
t0 = time.time()
res = run_experiment('A3_full_finetune', True, True, False, True, bs=16, lr=1e-4, epochs=50, patience=10)
all_results['A3_full_finetune'] = res
val_vals = [v['val'] for v in res.values()]
test_vals = [v['test'] for v in res.values()]
print(f'  Mean val={np.mean(val_vals):.4f}±{np.std(val_vals):.4f} test={np.mean(test_vals):.4f}±{np.std(test_vals):.4f} ({time.time()-t0:.0f}s)\n')

# Save results
with open(RESULTS_FILE, 'w') as f:
    json.dump(all_results, f, indent=2)

# Summary table
print('\n' + '='*80)
print('FINAL ABLATION TABLE')
print('='*80)
print('{:<20} {:>10} {:>8} {:>10} {:>8}'.format('Config', 'Val Mean', 'Val Std', 'Test Mean', 'Test Std'))
print('-'*58)
for name, data in all_results.items():
    vv = [d['val'] for d in data.values()]
    tv = [d['test'] for d in data.values()]
    print(f'{name:<20} {np.mean(vv):>9.4f}±{np.std(vv):<7.4f} {np.mean(tv):>9.4f}±{np.std(tv):<7.4f}')

print(f'\nSaved to {RESULTS_FILE}')
dt2 = time.strftime('%Y-%m-%d %H:%M:%S')
print(f'Done at {dt2}')
