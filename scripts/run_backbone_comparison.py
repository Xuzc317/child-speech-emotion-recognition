"""Phase 6.1c: Multi-SSL backbone comparison on C-BESD.

Compares WavLM, HuBERT, wav2vec2, emotion2vec on C-BESD 6:2:2.
Each backbone: extract features → train A1 (mean pool) + A3 (prosody pool).
"""
import os, sys, argparse, time, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from src.data.preprocess import collect_wav_files, split_speakers_7_3_with_inner_val
from src.models.ssl_backbone import SSLBackbone
from src.models.pooling import TemporalImportancePooling, extract_prosody
from src.models.semlp import SEMLP

MAX_FRAMES = 200
SR = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BACKBONES = {
    'wavlm': 'microsoft/wavlm-base-sv',
    'hubert': 'facebook/hubert-base-ls960',
    'wav2vec2': 'facebook/wav2vec2-base-960h',
    'emotion2vec': 'iic/emotion2vec_base',
}

class PreExtDataset(Dataset):
    def __init__(self, feats, labels, prosody=None):
        self.feats = torch.from_numpy(feats)
        self.labels = torch.from_numpy(labels).long()
        self.prosody = torch.from_numpy(prosody) if prosody is not None else None
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        if self.prosody is not None:
            return self.feats[i], self.labels[i], self.prosody[i]
        return self.feats[i], self.labels[i]

def collate(batch):
    feats = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    if len(batch[0]) == 3:
        prosody = torch.stack([b[2] for b in batch])
        return feats, labels, prosody
    return feats, labels

class A1Model(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.cls = SEMLP(input_dim=dim)
    def forward(self, x):
        return self.cls(x.mean(dim=1))

class A3Model(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.pool = TemporalImportancePooling(ssl_dim=dim)
        self.cls = SEMLP(input_dim=dim)
    def forward(self, x, f0=None, energy=None):
        if f0 is not None and energy is not None:
            x = self.pool(x, f0, energy)
        else:
            x = x.mean(dim=1)
        return self.cls(x)

def evaluate(model, loader, use_prosody, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            feats, labels = batch[0].to(device), batch[1].to(device)
            if use_prosody and len(batch) >= 3:
                pb = batch[2].to(device)
                out = model(feats, f0=pb[:,:,0:1], energy=pb[:,:,1:2])
            else:
                out = model(feats)
            all_preds.append(out.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    wa = (all_preds == all_labels).float().mean().item()
    uar = np.mean([(all_preds[all_labels==c]==c).float().mean().item()
                   for c in range(6) if (all_labels==c).sum()>0])
    return wa, uar

def train_model(model, train_loader, val_loader, use_prosody, device, args):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_uar, patience_ct = 0.0, 0
    best_state, best_wa = None, 0.0
    for ep in range(args.epochs):
        model.train()
        for batch in train_loader:
            feats, labels = batch[0].to(device), batch[1].to(device)
            opt.zero_grad()
            if use_prosody and len(batch) >= 3:
                pb = batch[2].to(device)
                out = model(feats, f0=pb[:,:,0:1], energy=pb[:,:,1:2])
            else:
                out = model(feats)
            criterion(out, labels).backward()
            opt.step()
        sched.step()
        wa, uar = evaluate(model, val_loader, use_prosody, device)
        if uar > best_uar:
            best_uar, best_wa = uar, wa
            patience_ct = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ct += 1
        if patience_ct >= args.patience:
            break
    model.load_state_dict(best_state)
    return best_wa, best_uar

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--backbones', nargs='+', default=['wavlm','hubert','wav2vec2'])
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    global DEVICE; DEVICE = device

    # Load data split once
    wav_dir = os.path.join(_project_root, '..', '提交到团队', '数据集', 'BESD', 'BESD', 'MY')
    cloud_dir = '/root/autodl-tmp/child-speech-emotion-recognition/数据集/BESD/BESD/MY'
    if not os.path.isdir(wav_dir):
        wav_dir = cloud_dir  # cloud fallback
    entries = collect_wav_files(wav_dir)
    train_e, val_e, test_e, _, _, _, _ = split_speakers_7_3_with_inner_val(entries, seed=42)
    print(f"Split: train={len(train_e)} val={len(val_e)} test={len(test_e)}")

    # Extract prosody once (same for all backbones — from raw waveform)
    print("Extracting prosody features (shared across backbones)...")
    import librosa
    def extract_prosody_batch(entries_list):
        f0_list, e_list = [], []
        for path, _, _ in tqdm(entries_list):
            y, _ = librosa.load(path, sr=SR)
            if len(y) < SR*4:
                y = np.pad(y, (0, SR*4 - len(y)))
            else:
                y = y[:SR*4]
            f0, energy = extract_prosody(y, sr=SR, hop_length=320)
            f0_pad = np.zeros(MAX_FRAMES, dtype=np.float32)
            energy_pad = np.zeros(MAX_FRAMES, dtype=np.float32)
            fl = min(len(f0), MAX_FRAMES)
            el = min(len(energy), MAX_FRAMES)
            f0_pad[:fl] = f0[:fl]; energy_pad[:el] = energy[:el]
            f0_list.append(f0_pad); e_list.append(energy_pad)
        return np.stack([np.array(f0_list), np.array(e_list)], axis=-1)

    train_p = extract_prosody_batch(train_e)
    val_p = extract_prosody_batch(val_e)
    test_p = extract_prosody_batch(test_e)

    results = {}
    dim_map = {'wavlm': 768, 'hubert': 768, 'wav2vec2': 768, 'emotion2vec': 768}

    for name in args.backbones:
        print(f"\n{'='*60}\nBackbone: {name}\n{'='*60}")
        dim = dim_map.get(name, 768)

        # Load backbone
        try:
            backbone = SSLBackbone(model_name=name, frozen=True, device=device)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # Extract SSL features
        def extract_ssl(entries_list, desc):
            feats = []
            for path, _, _ in tqdm(entries_list, desc=desc):
                import librosa as _lb
                y, _ = _lb.load(path, sr=SR)
                if len(y) < SR*4: y = np.pad(y, (0, SR*4-len(y)))
                else: y = y[:SR*4]
                wav = torch.from_numpy(y).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    f = backbone(wav).squeeze(0).cpu().numpy()
                if f.shape[0] > MAX_FRAMES: f = f[:MAX_FRAMES]
                else:
                    pad = np.zeros((MAX_FRAMES-f.shape[0], f.shape[1]), dtype=np.float32)
                    f = np.concatenate([f, pad], axis=0)
                feats.append(f)
            return np.stack(feats)

        print("Extracting SSL features...")
        t0 = time.time()
        train_f = extract_ssl(train_e, f'{name} train')
        val_f = extract_ssl(val_e, f'{name} val')
        test_f = extract_ssl(test_e, f'{name} test')
        print(f"  Extracted in {time.time()-t0:.0f}s")

        # Train A1 and A3
        for model_name, use_prosody in [('A1_mean', False), ('A3_prosody', True)]:
            print(f"  Training {model_name}...")
            ds_kw = {'feats': train_f, 'labels': np.array([e[1] for e in train_e])}
            val_kw = {'feats': val_f, 'labels': np.array([e[1] for e in val_e])}
            test_kw = {'feats': test_f, 'labels': np.array([e[1] for e in test_e])}
            if use_prosody:
                ds_kw['prosody'] = train_p; val_kw['prosody'] = val_p; test_kw['prosody'] = test_p

            train_ldr = DataLoader(PreExtDataset(**ds_kw), args.bs, shuffle=True, collate_fn=collate, num_workers=4)
            val_ldr = DataLoader(PreExtDataset(**val_kw), args.bs, shuffle=False, collate_fn=collate, num_workers=4)
            test_ldr = DataLoader(PreExtDataset(**test_kw), args.bs, shuffle=False, collate_fn=collate, num_workers=4)

            model = (A3Model if use_prosody else A1Model)(dim=dim).to(device)
            best_wa, best_uar = train_model(model, train_ldr, val_ldr, use_prosody, device, args)
            test_wa, test_uar = evaluate(model, test_ldr, use_prosody, device)
            key = f'{name}_{model_name}'
            results[key] = {'best_val_wa': float(best_wa), 'best_val_uar': float(best_uar),
                           'test_wa': float(test_wa), 'test_uar': float(test_uar)}
            print(f"    Val: WA={best_wa:.4f} UAR={best_uar:.4f} | Test: WA={test_wa:.4f} UAR={test_uar:.4f}")

    # Summary
    print("\n" + "="*70)
    print("BACKBONE COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Backbone':<15} {'A1 WA':>8} {'A1 UAR':>8} {'A3 WA':>8} {'A3 UAR':>8} {'ΔWA':>8}")
    print("-"*55)
    for name in args.backbones:
        a1 = results.get(f'{name}_A1_mean', {})
        a3 = results.get(f'{name}_A3_prosody', {})
        if a1 and a3:
            dwa = a3['test_wa'] - a1['test_wa']
            print(f"{name:<15} {a1['test_wa']:>8.4f} {a1['test_uar']:>8.4f} {a3['test_wa']:>8.4f} {a3['test_uar']:>8.4f} {dwa:>+8.4f}")

    os.makedirs('experiments/v5_622', exist_ok=True)
    with open('experiments/v5_622/backbone_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved: experiments/v5_622/backbone_comparison.json")

if __name__ == '__main__':
    main()
