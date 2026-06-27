"""Phase 4: Model Comparison — emotion2vec_plus_base / emotion2vec_plus_large vs WavLM"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, '/root/ser_project')
from src.data.preprocess import collect_wav_files, split_speakers_3way
from src.models.ssl_backbone import SSLBackbone, preprocess_wav
from src.training.train_ssl import DistributionCalibratedSER, SSLFeatureDataset, collate_fn_ssl_features

DATA_BASE = '/root/autodl-tmp/child-speech-emotion-recognition/数据集/BESD/BESD'
OUT_DIR = '/root/ser_project/data'
DEVICE = torch.device('cuda')
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_CONFIGS = {
    'emotion2vec_plus_base': {'hidden_size': 768, 'model_id': 'iic/emotion2vec_plus_base'},
    'emotion2vec_plus_large': {'hidden_size': 1024, 'model_id': 'iic/emotion2vec_plus_large'},
}

def extract_features(wav_dir, model_name, prefix, device, hidden_size):
    print(f'\n=== Extracting {prefix} features (model={model_name}, dim={hidden_size}) ===')
    entries = collect_wav_files(wav_dir)
    print(f'  Total WAVs: {len(entries)}')
    train, val, test, t_sids, v_sids, te_sids = split_speakers_3way(entries)
    print(f'  Train: {len(train)} ({len(t_sids)} speakers), Val: {len(val)}, Test: {len(test)}')

    backbone = SSLBackbone(model_name=model_name, frozen=True, device=device)
    max_frames = 200

    for split_name, split_entries in [('train', train), ('val', val), ('test', test)]:
        feats_list, labels_list = [], []
        t0 = time.time()
        for idx, (path, label, sid) in enumerate(split_entries):
            try:
                waveform, sr, _ = preprocess_wav(path, target_sr=16000, duration=4.0)
                waveform = waveform.to(device)
                with torch.no_grad():
                    feat = backbone(waveform)
                feat = feat.squeeze(0).cpu().numpy()
                if feat.shape[0] > max_frames:
                    feat = feat[:max_frames]
                else:
                    pad = np.zeros((max_frames - feat.shape[0], feat.shape[1]), dtype=np.float32)
                    feat = np.concatenate([feat, pad], axis=0)
                feats_list.append(feat)
                labels_list.append(label)
            except Exception as e:
                pass
            if (idx + 1) % 200 == 0:
                print(f'  {split_name} [{idx+1}/{len(split_entries)}] {time.time()-t0:.0f}s')

        feats_arr = np.stack(feats_list, axis=0).astype(np.float32)
        labels_arr = np.array(labels_list, dtype=np.int64)
        np.save(os.path.join(OUT_DIR, f'{split_name}_{prefix}_feats.npy'), feats_arr)
        np.save(os.path.join(OUT_DIR, f'{split_name}_{prefix}_labels.npy'), labels_arr)
        print(f'  Saved {split_name}_{prefix}_feats.npy: {feats_arr.shape}')
    del backbone

def extract_prosody_for_data(wav_dir, prefix, device):
    print(f'\n=== Extracting {prefix} prosody ===')
    import librosa
    entries = collect_wav_files(wav_dir)
    train, val, test, _, _, _ = split_speakers_3way(entries)
    max_frames = 200

    for split_name, split_entries in [('train', train), ('val', val), ('test', test)]:
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
                if len(f0) > max_frames:
                    f0, energy = f0[:max_frames], energy[:max_frames]
                else:
                    f0 = np.pad(f0, (0, max_frames - len(f0)))
                    energy = np.pad(energy, (0, max_frames - len(energy)))
                prosody_list.append(np.stack([f0, energy], axis=-1))
            except:
                prosody_list.append(np.zeros((max_frames, 2), dtype=np.float32))
            if (idx + 1) % 200 == 0:
                print(f'  {split_name} [{idx+1}/{len(split_entries)}] {time.time()-t0:.0f}s')

        prosody_arr = np.stack(prosody_list, axis=0).astype(np.float32)
        np.save(os.path.join(OUT_DIR, f'{split_name}_{prefix}_prosody.npy'), prosody_arr)
        print(f'  Saved {split_name}_{prefix}_prosody.npy: {prosody_arr.shape}')

def run_training(exp_name, prefix, hidden_size, seed):
    print(f'\n=== {exp_name} seed={seed} (dim={hidden_size}) ===')

    adapter_init_path = os.path.join(OUT_DIR, 'adapter_init.npz')
    adapter_init = dict(np.load(adapter_init_path)) if os.path.exists(adapter_init_path) else None

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = SSLFeatureDataset(
        os.path.join(OUT_DIR, f'train_{prefix}_feats.npy'),
        os.path.join(OUT_DIR, f'train_{prefix}_labels.npy'),
        os.path.join(OUT_DIR, f'train_{prefix}_prosody.npy'))
    val_dataset = SSLFeatureDataset(
        os.path.join(OUT_DIR, f'val_{prefix}_feats.npy'),
        os.path.join(OUT_DIR, f'val_{prefix}_labels.npy'),
        os.path.join(OUT_DIR, f'val_{prefix}_prosody.npy'))
    test_dataset = SSLFeatureDataset(
        os.path.join(OUT_DIR, f'test_{prefix}_feats.npy'),
        os.path.join(OUT_DIR, f'test_{prefix}_labels.npy'),
        os.path.join(OUT_DIR, f'test_{prefix}_prosody.npy'))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn_ssl_features, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn_ssl_features, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn_ssl_features, num_workers=0)

    config = {'use_backbone': False, 'feature_dim': hidden_size, 'num_classes': 6,
              'use_adapter': True, 'use_prosody': True, 'frozen_backbone': True, 'ssl_model': 'wavlm'}

    model = DistributionCalibratedSER(config, adapter_init).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc, best_test_acc = 0, 0
    patience_counter, best_epoch = 0, 0

    for epoch in range(80):
        model.train()
        for feats, labels, masks, prosody in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            f0 = prosody[..., 0:1].to(DEVICE) if prosody is not None else None
            energy = prosody[..., 1:2].to(DEVICE) if prosody is not None else None
            optimizer.zero_grad()
            logits = model(feats, f0=f0, energy=energy)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for feats, labels, masks, prosody in val_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                f0 = prosody[..., 0:1].to(DEVICE) if prosody is not None else None
                energy = prosody[..., 1:2].to(DEVICE) if prosody is not None else None
                logits = model(feats, f0=f0, energy=energy)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_epoch = epoch + 1
            correct, total = 0, 0
            with torch.no_grad():
                for feats, labels, masks, prosody in test_loader:
                    feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                    f0 = prosody[..., 0:1].to(DEVICE) if prosody is not None else None
                    energy = prosody[..., 1:2].to(DEVICE) if prosody is not None else None
                    logits = model(feats, f0=f0, energy=energy)
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            best_test_acc = correct / total
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter >= 15:
            print(f'  Epoch {epoch+1}: val={val_acc:.4f} best_val={best_val_acc:.4f} test={best_test_acc:.4f}')

        if patience_counter >= 15:
            print(f'  Early stop')
            break

    print(f'  {exp_name} seed={seed}: best_val={best_val_acc:.4f} best_test={best_test_acc:.4f} @epoch={best_epoch}')
    return best_val_acc, best_test_acc


if __name__ == '__main__':
    print('=' * 60)
    print('Phase 4: Model Comparison Experiments')
    print(f'Device: {DEVICE}')
    print(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    # MY data directory (Telugu + English mixed)
    my_wav_dir = os.path.join(DATA_BASE, 'MY')

    results = {}
    for model_name, cfg in MODEL_CONFIGS.items():
        prefix = model_name.replace('emotion2vec', 'e2v')
        hidden_size = cfg['hidden_size']

        # Extract features
        feats_path = os.path.join(OUT_DIR, f'train_{prefix}_feats.npy')
        if not os.path.exists(feats_path):
            extract_features(my_wav_dir, model_name, prefix, DEVICE, hidden_size)
        else:
            print(f'{prefix} features already exist, skipping extraction')

        # Extract prosody (reuse if from same wav_dir)
        prosody_path = os.path.join(OUT_DIR, f'train_{prefix}_prosody.npy')
        if not os.path.exists(prosody_path):
            extract_prosody_for_data(my_wav_dir, prefix, DEVICE)
        else:
            print(f'{prefix} prosody already exist, skipping')

        # Run training
        vals, tests = [], []
        for seed in [42, 123, 456]:
            val, test = run_training(f'MC_{model_name}', prefix, hidden_size, seed)
            vals.append(val)
            tests.append(test)

        results[model_name] = {
            'hidden_size': hidden_size,
            'val_mean': float(np.mean(vals)), 'val_std': float(np.std(vals)),
            'test_mean': float(np.mean(tests)), 'test_std': float(np.std(tests)),
            'val_seeds': [float(v) for v in vals],
            'test_seeds': [float(t) for t in tests],
        }
        print(f'\n{model_name}({hidden_size}dim): Val={np.mean(vals):.4f}+/-{np.std(vals):.4f} Test={np.mean(tests):.4f}+/-{np.std(tests):.4f}')

    with open(os.path.join(OUT_DIR, 'model_comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {OUT_DIR}/model_comparison_results.json')
    print('Done!')
