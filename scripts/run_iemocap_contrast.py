"""Phase 6.1b: Adult SER contrast — A1 vs A3 on IEMOCAP.

Hypothesis: Prosody Pooling benefit is LARGER for children (high F0 variability)
than for adults (lower F0 variability). This would support the "children need
prosody guidance more" narrative.

If the benefit is similar on adults, the narrative shifts to "prosody pooling
is universally beneficial for high-F0-variance populations" — still valid.
"""
import os, sys, argparse, time, json, glob
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from src.models.ssl_backbone import SSLBackbone
from src.models.pooling import TemporalImportancePooling, extract_prosody
from src.models.semlp import SEMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 200
SR = 16000
CLASS_NAMES = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD']

# IEMOCAP emotion mapping (simplified 6-class merge to match C-BESD)
IEMOCAP_MAP = {
    'ang': 'ANGER', 'hap': 'HAPPY', 'sad': 'SAD',
    'neu': 'NEUTRAL', 'fea': 'FEAR', 'dis': 'DISGUST',
    'exc': 'HAPPY',  # excitement -> happy
    'fru': 'ANGER',  # frustration -> anger
    'sur': None,     # surprise — skip (no C-BESD match)
    'oth': None,     # other — skip
    'xxx': None,     # unknown — skip
}


def load_iemocap(data_dir):
    """Collect IEMOCAP WAV files with emotion labels."""
    entries = []
    wav_dir = os.path.join(data_dir, 'wavs')
    if not os.path.isdir(wav_dir):
        # Try alternate layout
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.wav'):
                    entries.append(os.path.join(root, f))
        # Need emotion labels — try to parse from filename or session dir
        # IEMOCAP standard: sessionX/wav/Ses0XF_improXX_FXXX.wav, labels in txt files
        raise RuntimeError(
            f"IEMOCAP wavs/ not found at {wav_dir}. "
            "Expected structure: <data_dir>/wavs/<class>/*.wav "
            "(as pre-organized by scripts/extract_ssl_features.py logic)"
        )

    for cls_name in sorted(os.listdir(wav_dir)):
        cls_dir = os.path.join(wav_dir, cls_name)
        if os.path.isdir(cls_dir):
            mapped = IEMOCAP_MAP.get(cls_name)
            if mapped is None:
                continue
            label_idx = CLASS_NAMES.index(mapped)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.wav'):
                    entries.append((os.path.join(cls_dir, fname), label_idx))
    return entries


def speaker_split_iemocap(entries, val_ratio=0.15, seed=42):
    """Simple speaker-independent split for IEMOCAP.
    IEMOCAP has 10 speakers (5 sessions × 2 speakers each).
    We use 8 for trainval, 2 for test.
    """
    # Extract speaker from filename: Ses01F_impro01_F000.wav -> speaker = F/M from session
    import re
    speaker_set = set()
    speaker_files = {}
    for path, label in entries:
        fname = os.path.basename(path)
        # IEMOCAP organized by class dir, try to extract speaker
        parts = fname.split('_')
        if len(parts) >= 1:
            sid = parts[0][:5]  # e.g. Ses01
            speaker_set.add(sid)
            speaker_files.setdefault(sid, []).append((path, label))

    sids = sorted(speaker_set)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(sids))
    n_test = max(1, int(len(sids) * 0.2))
    test_sids = set(sids[i] for i in perm[:n_test])
    trainval_sids = set(sids[i] for i in perm[n_test:])

    # Inner val split
    trainval_list = list(trainval_sids)
    perm2 = rng.permutation(len(trainval_list))
    n_val = max(1, int(len(trainval_list) * val_ratio))
    val_sids = set(trainval_list[i] for i in perm2[:n_val])
    train_sids = set(trainval_list[i] for i in perm2[n_val:])

    def gather(sids):
        result = []
        for sid in sids:
            result.extend(speaker_files.get(sid, []))
        return result

    return gather(train_sids), gather(val_sids), gather(test_sids)


class IEMOCAPDataset(Dataset):
    def __init__(self, entries, backbone, use_prosody=False):
        self.entries = entries
        self.backbone = backbone
        self.use_prosody = use_prosody

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        # Preprocess + extract SSL features
        data, sr = librosa.load(path, sr=SR)
        target_len = SR * 4
        if len(data) < target_len:
            data = np.pad(data, (0, target_len - len(data)))
        else:
            data = data[:target_len]

        waveform = torch.from_numpy(data).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feats = self.backbone(waveform).squeeze(0).cpu()  # (T, 768)

        # Pad to MAX_FRAMES
        if feats.shape[0] > MAX_FRAMES:
            feats = feats[:MAX_FRAMES]
        else:
            pad = torch.zeros(MAX_FRAMES - feats.shape[0], feats.shape[1])
            feats = torch.cat([feats, pad], dim=0)

        result = [feats, torch.tensor(label, dtype=torch.int64)]

        if self.use_prosody:
            # Extract prosody from waveform
            f0, energy = extract_prosody(data, sr=SR, hop_length=320)
            # Pad to MAX_FRAMES
            f0_pad = np.zeros(MAX_FRAMES, dtype=np.float32)
            energy_pad = np.zeros(MAX_FRAMES, dtype=np.float32)
            f0_len = min(len(f0), MAX_FRAMES)
            energy_len = min(len(energy), MAX_FRAMES)
            f0_pad[:f0_len] = f0[:f0_len]
            energy_pad[:energy_len] = energy[:energy_len]
            prosody = np.stack([f0_pad, energy_pad], axis=1)  # (200, 2)
            result.append(torch.from_numpy(prosody))

        return tuple(result)


def collate_iemocap(batch):
    feats = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    has_prosody = len(batch[0]) == 3
    result = [feats, labels]
    if has_prosody:
        prosody = torch.stack([b[2] for b in batch])
        result.append(prosody)
    return tuple(result)


class A1_Model(nn.Module):
    """WavLM + mean pool + SEMLP (no adapter, no prosody)."""
    def __init__(self):
        super().__init__()
        self.classifier = SEMLP(input_dim=768, num_classes=6)

    def forward(self, x):
        return self.classifier(x.mean(dim=1))


class A3_Model(nn.Module):
    """WavLM + Prosody Pooling + SEMLP."""
    def __init__(self):
        super().__init__()
        self.pooling = TemporalImportancePooling(ssl_dim=768)
        self.classifier = SEMLP(input_dim=768, num_classes=6)

    def forward(self, x, f0=None, energy=None):
        if f0 is not None and energy is not None:
            x = self.pooling(x, f0, energy)
        else:
            x = x.mean(dim=1)
        return self.classifier(x)


def evaluate_iemocap(model, loader, use_prosody):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            feats, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            if use_prosody and len(batch) >= 3:
                pb = batch[2].to(DEVICE)
                outputs = model(feats, f0=pb[:, :, 0:1], energy=pb[:, :, 1:2])
            else:
                outputs = model(feats)
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    wa = (all_preds == all_labels).float().mean().item()
    uar = np.mean([
        (all_preds[all_labels == c] == c).float().mean().item()
        for c in range(6) if (all_labels == c).sum() > 0
    ])
    return wa, uar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iemocap_dir', default=None)
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

    # Locate IEMOCAP
    if args.iemocap_dir:
        iemocap_dir = args.iemocap_dir
    else:
        iemocap_dir = os.path.join(_project_root, '..', '提交到团队', '数据集', 'IEMOCAP')
    if not os.path.isdir(iemocap_dir):
        print(f"IEMOCAP not found at {iemocap_dir}")
        print("Skipping Phase 6.1b — need IEMOCAP data.")
        return

    import librosa  # lazy import
    print("Loading WavLM backbone (frozen)...")
    backbone = SSLBackbone(model_name='wavlm', frozen=True, device=device)

    print("Loading IEMOCAP entries...")
    entries = load_iemocap(iemocap_dir)
    print(f"Total: {len(entries)} samples")
    train_e, val_e, test_e = speaker_split_iemocap(entries)
    print(f"Train: {len(train_e)}, Val: {len(val_e)}, Test: {len(test_e)}")

    # Pre-extract features for faster training
    def pre_extract(entries_list):
        ds = IEMOCAPDataset(entries_list, backbone, use_prosody=True)
        feats_list, labels_list, prosody_list = [], [], []
        for i in tqdm(range(len(ds)), desc='Extract'):
            items = ds[i]
            feats_list.append(items[0])
            labels_list.append(items[1])
            if len(items) >= 3:
                prosody_list.append(items[2])
        return (torch.stack(feats_list), torch.stack(labels_list),
                torch.stack(prosody_list) if prosody_list else None)

    print("Pre-extracting features...")
    train_f, train_l, train_p = pre_extract(train_e)
    val_f, val_l, val_p = pre_extract(val_e)
    test_f, test_l, test_p = pre_extract(test_e)

    class IEMOCAPPreExt(Dataset):
        def __init__(self, feats, labels, prosody=None):
            self.feats = feats
            self.labels = labels
            self.prosody = prosody
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            if self.prosody is not None:
                return self.feats[i], self.labels[i], self.prosody[i]
            return self.feats[i], self.labels[i]

    results = {}
    for name, use_prosody in [('A1_mean_pool', False), ('A3_prosody', True)]:
        print(f"\n=== IEMOCAP {name} ===")
        ds_kwargs = {'feats': train_f, 'labels': train_l}
        val_kwargs = {'feats': val_f, 'labels': val_l}
        test_kwargs = {'feats': test_f, 'labels': test_l}
        if use_prosody:
            ds_kwargs['prosody'] = train_p
            val_kwargs['prosody'] = val_p
            test_kwargs['prosody'] = test_p

        train_ds = IEMOCAPPreExt(**ds_kwargs)
        val_ds = IEMOCAPPreExt(**val_kwargs)
        test_ds = IEMOCAPPreExt(**test_kwargs)

        train_loader = DataLoader(train_ds, args.bs, shuffle=True, collate_fn=collate_iemocap, num_workers=0)
        val_loader = DataLoader(val_ds, args.bs, shuffle=False, collate_fn=collate_iemocap, num_workers=0)
        test_loader = DataLoader(test_ds, args.bs, shuffle=False, collate_fn=collate_iemocap, num_workers=0)

        model = (A3_Model if use_prosody else A1_Model)().to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_uar, patience_ct = 0.0, 0
        best_state, best_val_wa = None, 0.0

        for epoch in range(args.epochs):
            model.train()
            for batch in train_loader:
                feats = batch[0].to(device)
                labels = batch[1].to(device)
                optimizer.zero_grad()
                if use_prosody and len(batch) >= 3:
                    pb = batch[2].to(device)
                    outputs = model(feats, f0=pb[:, :, 0:1], energy=pb[:, :, 1:2])
                else:
                    outputs = model(feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

            val_wa, val_uar = evaluate_iemocap(model, val_loader, use_prosody)
            if val_uar > best_val_uar:
                best_val_uar = val_uar
                best_val_wa = val_wa
                patience_ct = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ct += 1
            if patience_ct >= args.patience:
                break

        model.load_state_dict(best_state)
        test_wa, test_uar = evaluate_iemocap(model, test_loader, use_prosody)
        print(f"  Best Val: WA={best_val_wa:.4f} UAR={best_val_uar:.4f}")
        print(f"  Test:     WA={test_wa:.4f} UAR={test_uar:.4f}")
        results[name] = {
            'best_val_wa': float(best_val_wa), 'best_val_uar': float(best_val_uar),
            'test_wa': float(test_wa), 'test_uar': float(test_uar),
        }

    # Compare
    delta_wa = results['A3_prosody']['test_wa'] - results['A1_mean_pool']['test_wa']
    delta_uar = results['A3_prosody']['test_uar'] - results['A1_mean_pool']['test_uar']
    print(f"\n=== Prosody Pooling gain on IEMOCAP (adults) ===")
    print(f"  ΔWA = {delta_wa:+.4f}, ΔUAR = {delta_uar:+.4f}")
    print(f"  (Compare: C-BESD children A1→A3 ΔWA = +2.24pp)")

    result = {
        'A1': results['A1_mean_pool'],
        'A3': results['A3_prosody'],
        'delta_wa': delta_wa,
        'delta_uar': delta_uar,
        'children_delta_wa_reference': 0.0224,
    }
    os.makedirs('experiments/v5_622', exist_ok=True)
    with open('experiments/v5_622/iemocap_contrast_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("Saved: experiments/v5_622/iemocap_contrast_result.json")


if __name__ == '__main__':
    main()
