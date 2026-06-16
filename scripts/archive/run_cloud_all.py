"""Cloud experiment runner — fixed data leakage (3-way split, val early-stop)"""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, '/root/ser_project')

from src.data.dataset_ssl import SSLFeatureDataset, collate_fn_ssl_features
from src.training.train_ssl import DistributionCalibratedSER
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
results = {}

for seed in [42, 123, 456]:
    torch.manual_seed(seed); np.random.seed(seed)
    print(f'\n=== Seed {seed} ===')
    train_ds = SSLFeatureDataset('data/train_wavlm_feats.npy', 'data/train_wavlm_labels.npy', 'data/train_wavlm_prosody.npy')
    val_ds   = SSLFeatureDataset('data/val_wavlm_feats.npy',   'data/val_wavlm_labels.npy',   'data/val_wavlm_prosody.npy')
    test_ds  = SSLFeatureDataset('data/test_wavlm_feats.npy',  'data/test_wavlm_labels.npy',  'data/test_wavlm_prosody.npy')

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn_ssl_features, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, collate_fn=collate_fn_ssl_features, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, collate_fn=collate_fn_ssl_features, num_workers=4, pin_memory=True)

    adapter_init = dict(np.load('data/adapter_init.npz'))
    config = {'use_backbone': False, 'ssl_model': 'wavlm', 'frozen_backbone': True, 'use_adapter': True, 'use_prosody': True}
    model = DistributionCalibratedSER(config, adapter_init=adapter_init).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    best_val_acc, patience_counter = 0.0, 0

    for epoch in range(80):
        model.train()
        for batch in train_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            prosody_batch = batch[3].to(device) if len(batch) >= 4 else None
            optimizer.zero_grad()
            if prosody_batch is not None:
                outputs = model(inputs, f0=prosody_batch[:,:,0:1], energy=prosody_batch[:,:,1:2])
            else:
                outputs = model(inputs)
            criterion(outputs, labels).backward()
            optimizer.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                prosody_batch = batch[3].to(device) if len(batch) >= 4 else None
                if prosody_batch is not None:
                    outputs = model(inputs, f0=prosody_batch[:,:,0:1], energy=prosody_batch[:,:,1:2])
                else:
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total; scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc; patience_counter = 0
            torch.save(model.state_dict(), '/tmp/best_model.pth')
        else:
            patience_counter += 1

        if epoch % 5 == 0 or val_acc > best_val_acc - 0.001:
            print(f'Epoch {epoch+1}: Val={val_acc:.4f} best={best_val_acc:.4f}')
        if patience_counter >= 15:
            print(f'Early stop epoch {epoch+1}'); break

    model.load_state_dict(torch.load('/tmp/best_model.pth')); model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            prosody_batch = batch[3].to(device) if len(batch) >= 4 else None
            if prosody_batch is not None:
                outputs = model(inputs, f0=prosody_batch[:,:,0:1], energy=prosody_batch[:,:,1:2])
            else:
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    results[f'seed_{seed}'] = {'val': float(best_val_acc), 'test': float(test_acc)}
    print(f'Seed {seed}: val={best_val_acc:.4f} test={test_acc:.4f}')

with open('logs/cloud_fixed_results.json', 'w') as f:
    json.dump(results, f, indent=2)
for k in results:
    v = results[k]['val']
    t = results[k]['test']
    print(f'{k}: val={v:.4f} test={t:.4f}')
dt = time.strftime('%Y-%m-%d %H:%M:%S')
print(f'Done at {dt}')
