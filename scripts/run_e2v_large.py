"""Run emotion2vec_plus_large B3 experiment standalone"""
import os, sys, time, json
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
sys.path.insert(0, '/root/ser_project')
from src.data.preprocess import collect_wav_files, split_speakers_3way
from src.models.ssl_backbone import SSLBackbone, preprocess_wav
from src.training.train_ssl import DistributionCalibratedSER, SSLFeatureDataset, collate_fn_ssl_features

DEVICE = torch.device('cuda')
DATA_BASE = '/root/autodl-tmp/child-speech-emotion-recognition/数据集/BESD/BESD/MY'
OUT_DIR = '/root/ser_project/data'
PREFIX = 'e2v_large'
HIDDEN_SIZE = 1024
MAX_FRAMES = 200

print(f'=== emotion2vec_plus_large @ {time.strftime("%H:%M:%S")} ===')
entries = collect_wav_files(DATA_BASE)
print(f'Total WAVs: {len(entries)}')
train, val, test, _, _, _ = split_speakers_3way(entries)
print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')

# Extract features
backbone = SSLBackbone(model_name='emotion2vec_plus_large', frozen=True, device=DEVICE)
for split_name, split_entries in [('train', train), ('val', val), ('test', test)]:
    feats_path = os.path.join(OUT_DIR, f'{split_name}_{PREFIX}_feats.npy')
    if os.path.exists(feats_path):
        print(f'{split_name} features exist, skip')
        continue
    feats_list, labels_list = [], []
    t0 = time.time()
    for idx, (path, label, sid) in enumerate(split_entries):
        try:
            waveform, sr, _ = preprocess_wav(path, target_sr=16000, duration=4.0)
            waveform = waveform.to(DEVICE)
            with torch.no_grad():
                feat = backbone(waveform)
            feat = feat.squeeze(0).cpu().numpy()
            if feat.shape[0] > MAX_FRAMES:
                feat = feat[:MAX_FRAMES]
            else:
                pad = np.zeros((MAX_FRAMES - feat.shape[0], feat.shape[1]), dtype=np.float32)
                feat = np.concatenate([feat, pad], axis=0)
            feats_list.append(feat)
            labels_list.append(label)
        except Exception as e:
            pass
        if (idx + 1) % 200 == 0:
            print(f'  {split_name} [{idx+1}/{len(split_entries)}] {time.time()-t0:.0f}s')
    feats_arr = np.stack(feats_list, axis=0).astype(np.float32)
    labels_arr = np.array(labels_list, dtype=np.int64)
    np.save(feats_path, feats_arr)
    np.save(os.path.join(OUT_DIR, f'{split_name}_{PREFIX}_labels.npy'), labels_arr)
    print(f'  Saved {split_name}: {feats_arr.shape}')
del backbone

# Extract prosody
import librosa
print('Extracting prosody...')
for split_name, split_entries in [('train', train), ('val', val), ('test', test)]:
    pros_path = os.path.join(OUT_DIR, f'{split_name}_{PREFIX}_prosody.npy')
    if os.path.exists(pros_path):
        print(f'{split_name} prosody exist, skip')
        continue
    prosody_list = []
    t0 = time.time()
    for idx, (path, label, sid) in enumerate(split_entries):
        try:
            y, sr = librosa.load(path, sr=16000)
            f0 = librosa.yin(y, sr=sr, fmin=65, fmax=2093, hop_length=320)
            f0 = np.nan_to_num(f0, nan=0.0)
            energy = librosa.feature.rms(y=y, hop_length=320).squeeze(0)
            min_len = min(len(f0), len(energy))
            f0, energy = f0[:min_len], energy[:min_len]
            if len(f0) > MAX_FRAMES:
                f0, energy = f0[:MAX_FRAMES], energy[:MAX_FRAMES]
            else:
                f0 = np.pad(f0, (0, MAX_FRAMES - len(f0)))
                energy = np.pad(energy, (0, MAX_FRAMES - len(energy)))
            prosody_list.append(np.stack([f0, energy], axis=-1))
        except:
            prosody_list.append(np.zeros((MAX_FRAMES, 2), dtype=np.float32))
        if (idx + 1) % 200 == 0:
            print(f'  {split_name} [{idx+1}/{len(split_entries)}] {time.time()-t0:.0f}s')
    prosody_arr = np.stack(prosody_list, axis=0).astype(np.float32)
    np.save(pros_path, prosody_arr)
    print(f'  Saved {split_name}: {prosody_arr.shape}')

# Training
print('=== Training (B3 config, 1024-dim) ===')
adapter_init_path = os.path.join(OUT_DIR, 'adapter_init.npz')
# Note: adapter_init is 768-dim, not 1024 — use random init for 1024-dim model
adapter_init = None  # Will use random init
print('Using random init adapter (adapter_init.npz is 768-dim, model is 1024-dim)')

vals, tests = [], []
for seed in [42, 123, 456]:
    print(f'\n--- Seed {seed} ---')
    torch.manual_seed(seed)
    np.random.seed(seed)
    td = SSLFeatureDataset(
        os.path.join(OUT_DIR, f'train_{PREFIX}_feats.npy'),
        os.path.join(OUT_DIR, f'train_{PREFIX}_labels.npy'),
        os.path.join(OUT_DIR, f'train_{PREFIX}_prosody.npy'))
    vd = SSLFeatureDataset(
        os.path.join(OUT_DIR, f'val_{PREFIX}_feats.npy'),
        os.path.join(OUT_DIR, f'val_{PREFIX}_labels.npy'),
        os.path.join(OUT_DIR, f'val_{PREFIX}_prosody.npy'))
    ted = SSLFeatureDataset(
        os.path.join(OUT_DIR, f'test_{PREFIX}_feats.npy'),
        os.path.join(OUT_DIR, f'test_{PREFIX}_labels.npy'),
        os.path.join(OUT_DIR, f'test_{PREFIX}_prosody.npy'))
    tl = DataLoader(td, batch_size=128, shuffle=True, collate_fn=collate_fn_ssl_features, num_workers=0)
    vl = DataLoader(vd, batch_size=128, shuffle=False, collate_fn=collate_fn_ssl_features, num_workers=0)
    tel = DataLoader(ted, batch_size=128, shuffle=False, collate_fn=collate_fn_ssl_features, num_workers=0)

    config = {'use_backbone': False, 'feature_dim': HIDDEN_SIZE, 'num_classes': 6,
              'use_adapter': True, 'use_prosody': True, 'frozen_backbone': True,
              'ssl_model': 'emotion2vec_plus_large'}
    model = DistributionCalibratedSER(config, adapter_init).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val, best_test, patience, best_ep = 0, 0, 0, 0

    for epoch in range(80):
        model.train()
        for feats, labels, masks, prosody in tl:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            f0 = prosody[..., 0:1].to(DEVICE) if prosody is not None else None
            energy = prosody[..., 1:2].to(DEVICE) if prosody is not None else None
            optimizer.zero_grad()
            loss = criterion(model(feats, f0=f0, energy=energy), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for feats, labels, masks, prosody in vl:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                f0 = prosody[..., 0:1].to(DEVICE) if prosody is not None else None
                energy = prosody[..., 1:2].to(DEVICE) if prosody is not None else None
                logits = model(feats, f0=f0, energy=energy)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        if val_acc > best_val:
            best_val = val_acc; patience = 0; best_ep = epoch + 1
            correct, total = 0, 0
            with torch.no_grad():
                for feats, labels, masks, prosody in tel:
                    feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                    f0 = prosody[..., 0:1].to(DEVICE) if prosody is not None else None
                    energy = prosody[..., 1:2].to(DEVICE) if prosody is not None else None
                    logits = model(feats, f0=f0, energy=energy)
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            best_test = correct / total
        else:
            patience += 1
        if epoch % 5 == 0 or patience >= 15:
            print(f'  Epoch {epoch+1}: val={val_acc:.4f} best_val={best_val:.4f} test={best_test:.4f}')
        if patience >= 15:
            print(f'  Early stop')
            break
    vals.append(best_val); tests.append(best_test)
    print(f'  Seed {seed}: best_val={best_val:.4f} best_test={best_test:.4f} @epoch={best_ep}')

results = {
    'emotion2vec_plus_large': {
        'hidden_size': HIDDEN_SIZE,
        'val_mean': float(np.mean(vals)), 'val_std': float(np.std(vals)),
        'test_mean': float(np.mean(tests)), 'test_std': float(np.std(tests)),
        'val_seeds': [float(v) for v in vals], 'test_seeds': [float(t) for t in tests],
    }
}
print(f'\n=== FINAL ===')
print(f'e2v_plus_large(1024dim): Val={np.mean(vals):.4f}+/-{np.std(vals):.4f} Test={np.mean(tests):.4f}+/-{np.std(tests):.4f}')
with open(os.path.join(OUT_DIR, 'model_comparison_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print('Done!')
