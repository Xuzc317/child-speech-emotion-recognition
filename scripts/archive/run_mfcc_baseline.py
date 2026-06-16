"""Phase 6.1a: Reproduce C-BESD baseline — MFCC+CNN on our 6:2:2 split.

C-BESD paper (Rao et al., IEEE INSPECT 2024): MFCC + ZCR + Harmonic Ratio + Pitch → CNN → 76%.
We reproduce this on our 6:2:2 speaker-independent split for fair comparison.
"""
import os, sys, argparse, time, json
import numpy as np
import librosa
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from src.data.preprocess import collect_wav_files, split_speakers_7_3_with_inner_val

FEATURE_DIM = 40 + 1 + 1 + 1  # MFCC(40) + ZCR(1) + harmonic_ratio(1) + pitch(1) = 43
MAX_FRAMES = 200
SR = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_mfcc_features(wav_path):
    """Extract MFCC + ZCR + Harmonic Ratio + Pitch per frame."""
    y, sr = librosa.load(wav_path, sr=SR)
    target_len = SR * 4
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=320, n_fft=800).T  # (T, 40)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=320).T  # (T, 1)
    # Harmonic ratio via spectral flatness inversion
    spectral = np.abs(librosa.stft(y, hop_length=320, n_fft=800))
    flatness = librosa.feature.spectral_flatness(S=spectral**2).T  # (T, 1) — proxy for harmonic ratio
    # Pitch via YIN (efficient)
    f0 = librosa.yin(y, fmin=65, fmax=2093, sr=sr, hop_length=320)
    f0 = np.nan_to_num(f0, nan=0.0).reshape(-1, 1)  # (T, 1)

    # Align all features to same length
    min_len = min(mfcc.shape[0], zcr.shape[0], flatness.shape[0], f0.shape[0])
    feats = np.concatenate([mfcc[:min_len], zcr[:min_len], flatness[:min_len], f0[:min_len]], axis=1)

    # Pad to MAX_FRAMES
    if feats.shape[0] > MAX_FRAMES:
        feats = feats[:MAX_FRAMES]
    else:
        pad = np.zeros((MAX_FRAMES - feats.shape[0], feats.shape[1]), dtype=np.float32)
        feats = np.concatenate([feats, pad], axis=0)
    return feats.astype(np.float32)  # (200, 43)


class MFCCDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label, _ = self.entries[idx]
        feats = torch.from_numpy(extract_mfcc_features(path))
        return feats, torch.tensor(label, dtype=torch.int64)


def collate_mfcc(batch):
    feats = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return feats, labels


class MFCC_CNN(nn.Module):
    """Simple CNN matching the C-BESD paper: Conv1D over time → classifier."""
    def __init__(self, input_dim=43, num_classes=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T, 43) -> (B, 43, T)
        x = x.transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        return self.cls(x)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(DEVICE)
            outputs = model(feats)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    wa = (all_preds == all_labels).float().mean().item()
    # UAR: per-class recall average
    uar = 0.0
    for cls in range(6):
        mask = all_labels == cls
        if mask.sum() > 0:
            uar += (all_preds[mask] == cls).float().mean().item()
    uar /= 6
    return wa, uar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    global DEVICE
    DEVICE = device

    # ── Data split ──
    print("Loading WAV files and splitting (6:2:2 speaker-independent)...")
    wav_dir = os.path.join(_project_root, '..', '提交到团队', '数据集', 'BESD', 'BESD', 'MY')
    if not os.path.isdir(wav_dir):
        # Fallback: search common paths
        for candidate in [
            os.path.join(_project_root, '数据集', 'BESD', 'BESD', 'MY'),
            '/root/autodl-tmp/v5_data/',  # cloud
        ]:
            if os.path.isdir(candidate):
                wav_dir = candidate
                break
    entries = collect_wav_files(wav_dir)
    train_e, val_e, test_e, _, _, _, stats = split_speakers_7_3_with_inner_val(entries, seed=args.seed)
    print(f"Train: {len(train_e)}, Val: {len(val_e)}, Test: {len(test_e)}")

    # ── Datasets (extract on first epoch, cache in memory) ──
    print("\nExtracting MFCC features on-the-fly...")
    # For efficiency, pre-extract to memory
    def pre_extract(entries, name):
        feats_list, labels_list = [], []
        for path, label, _ in tqdm(entries, desc=name):
            feats_list.append(extract_mfcc_features(path))
            labels_list.append(label)
        return np.stack(feats_list), np.array(labels_list)

    train_feats, train_labels = pre_extract(train_e, "Train MFCC")
    val_feats, val_labels = pre_extract(val_e, "Val MFCC")
    test_feats, test_labels = pre_extract(test_e, "Test MFCC")
    input_dim = train_feats.shape[-1]
    print(f"Feature shape: {train_feats.shape}, dim={input_dim}")

    # Save features for reproducibility
    os.makedirs('data/mfcc_baseline', exist_ok=True)
    np.save('data/mfcc_baseline/train_mfcc_feats.npy', train_feats)
    np.save('data/mfcc_baseline/train_mfcc_labels.npy', train_labels)
    np.save('data/mfcc_baseline/val_mfcc_feats.npy', val_feats)
    np.save('data/mfcc_baseline/val_mfcc_labels.npy', val_labels)
    np.save('data/mfcc_baseline/test_mfcc_feats.npy', test_feats)
    np.save('data/mfcc_baseline/test_mfcc_labels.npy', test_labels)

    # ── Dataloaders ──
    # Use simple torch Dataset from pre-extracted arrays
    class PreExtractedDataset(Dataset):
        def __init__(self, feats, labels):
            self.feats = torch.from_numpy(feats)
            self.labels = torch.from_numpy(labels).long()
        def __len__(self): return len(self.labels)
        def __getitem__(self, i): return self.feats[i], self.labels[i]

    train_ds = PreExtractedDataset(train_feats, train_labels)
    val_ds = PreExtractedDataset(val_feats, val_labels)
    test_ds = PreExtractedDataset(test_feats, test_labels)

    train_loader = DataLoader(train_ds, args.bs, shuffle=True, collate_fn=collate_mfcc, num_workers=0)
    val_loader = DataLoader(val_ds, args.bs, shuffle=False, collate_fn=collate_mfcc, num_workers=0)
    test_loader = DataLoader(test_ds, args.bs, shuffle=False, collate_fn=collate_mfcc, num_workers=0)

    # ── Model ──
    model = MFCC_CNN(input_dim=input_dim).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_uar, patience_ct = 0.0, 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_wa, val_uar = evaluate(model, val_loader)
        if val_uar > best_val_uar:
            best_val_uar = val_uar
            patience_ct = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_val_wa = val_wa
        else:
            patience_ct += 1
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: Val WA={val_wa:.4f} UAR={val_uar:.4f}  best_uar={best_val_uar:.4f}")
        if patience_ct >= args.patience:
            print(f"Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    test_wa, test_uar = evaluate(model, test_loader)
    print(f"\n=== C-BESD MFCC+CNN Baseline Results (seed={args.seed}) ===")
    print(f"  Best Val: WA={best_val_wa:.4f} UAR={best_val_uar:.4f}")
    print(f"  Test:     WA={test_wa:.4f} UAR={test_uar:.4f}")
    print(f"  Published baseline [Rao 2024]: WA=76.0%")

    # Save result
    result = {
        'experiment': 'C-BESD MFCC+CNN baseline (Phase 6.1a)',
        'split': '6:2:2 speaker-independent (seed=42)',
        'seed': args.seed,
        'best_val_wa': float(best_val_wa),
        'best_val_uar': float(best_val_uar),
        'test_wa': float(test_wa),
        'test_uar': float(test_uar),
        'published_baseline_wa': 0.76,
    }
    with open('experiments/v5_622/mfcc_baseline_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("Saved: experiments/v5_622/mfcc_baseline_result.json")


if __name__ == '__main__':
    main()
