"""Phase 6.1d: CREMA-D adult contrast — A1 vs A3 on CREMA-D.

Strengthens the cross-age narrative with a second adult dataset.
"""
import os, sys, argparse, json, time, re
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
from src.models.ssl_backbone import SSLBackbone
from src.models.pooling import TemporalImportancePooling, extract_prosody
from src.models.semlp import SEMLP

MAX_FRAMES = 200
SR = 16000
CLASS_NAMES = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD']
EMO_MAP = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_filename(fname):
    """Parse '1001_DFA_ANG_XX.wav' → (speaker_id, emotion_idx)."""
    parts = os.path.splitext(fname)[0].split('_')
    if len(parts) >= 3:
        emo_code = parts[2]
        if emo_code in EMO_MAP:
            return parts[0], EMO_MAP[emo_code]
    return None, None


def collect_cremad(wav_dir):
    """Collect CREMA-D WAV files grouped by speaker and emotion."""
    entries = []
    for fname in sorted(os.listdir(wav_dir)):
        if not fname.endswith('.wav'):
            continue
        sid, label = parse_filename(fname)
        if sid is None:
            continue
        entries.append((os.path.join(wav_dir, fname), label, sid))
    return entries


def speaker_split(entries, test_ratio=0.2, val_ratio=0.15, seed=42):
    """Speaker-independent split."""
    speaker_map = {}
    for path, label, sid in entries:
        speaker_map.setdefault(sid, []).append((path, label))

    sids = sorted(speaker_map.keys())
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(sids))
    n_test = max(1, int(len(sids) * test_ratio))
    test_sids = set(sids[i] for i in perm[:n_test])
    trainval_sids = set(sids[i] for i in perm[n_test:])

    tv_list = list(trainval_sids)
    perm2 = rng.permutation(len(tv_list))
    n_val = max(1, int(len(tv_list) * val_ratio))
    val_sids = set(tv_list[i] for i in perm2[:n_val])
    train_sids = set(tv_list[i] for i in perm2[n_val:])

    def gather(sids):
        result = []
        for sid in sids:
            result.extend(speaker_map[sid])
        return result
    return gather(train_sids), gather(val_sids), gather(test_sids)


def pre_extract(entries_list, backbone, use_prosody=True):
    """Extract SSL features + optional prosody."""
    import soundfile as sf
    feats_list, labels_list, prosody_list = [], [], []
    for path, label in tqdm(entries_list, desc='Extract'):
        y, sr_in = sf.read(path)
        if sr_in != SR:
            import librosa
            y = librosa.resample(y.astype(np.float64), orig_sr=sr_in, target_sr=SR).astype(np.float32)
        target = SR * 4
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]
        wav = torch.from_numpy(y).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f = backbone(wav).squeeze(0).cpu()
        if f.shape[0] > MAX_FRAMES:
            f = f[:MAX_FRAMES]
        else:
            pad = torch.zeros(MAX_FRAMES - f.shape[0], f.shape[1])
            f = torch.cat([f, pad], dim=0)
        feats_list.append(f)
        labels_list.append(label)

        if use_prosody:
            f0, energy = extract_prosody(y, sr=SR, hop_length=320)
            f0_p = np.zeros(MAX_FRAMES, dtype=np.float32)
            en_p = np.zeros(MAX_FRAMES, dtype=np.float32)
            fl = min(len(f0), MAX_FRAMES)
            el = min(len(energy), MAX_FRAMES)
            f0_p[:fl] = f0[:fl]
            en_p[:el] = energy[:el]
            prosody_list.append(np.stack([f0_p, en_p], axis=1))

    result = [torch.stack(feats_list), torch.tensor(labels_list, dtype=torch.long)]
    if use_prosody:
        result.append(torch.tensor(np.stack(prosody_list), dtype=torch.float32))
    return tuple(result)


class PreExtDS(Dataset):
    def __init__(self, feats, labels, prosody=None):
        self.feats = feats
        self.labels = labels
        self.prosody = prosody
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        if self.prosody is not None:
            return self.feats[i], self.labels[i], self.prosody[i]
        return self.feats[i], self.labels[i]


def collate(batch):
    feats = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    if len(batch[0]) == 3:
        pro = torch.stack([b[2] for b in batch])
        return feats, labels, pro
    return feats, labels


class A1Model(nn.Module):
    def __init__(self): super().__init__(); self.cls = SEMLP()
    def forward(self, x): return self.cls(x.mean(dim=1))

class A3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = TemporalImportancePooling()
        self.cls = SEMLP()
    def forward(self, x, f0=None, e=None):
        if f0 is not None and e is not None:
            x = self.pool(x, f0, e)
        else:
            x = x.mean(dim=1)
        return self.cls(x)


def evaluate(model, loader, use_prosody):
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for batch in loader:
            feats, labels = batch[0].to(DEVICE), batch[1]
            if use_prosody and len(batch) >= 3:
                pb = batch[2].to(DEVICE)
                out = model(feats, f0=pb[:,:,0:1], e=pb[:,:,1:2])
            else:
                out = model(feats)
            all_p.append(out.argmax(dim=1).cpu())
            all_l.append(labels)
    all_p = torch.cat(all_p); all_l = torch.cat(all_l)
    wa = (all_p == all_l).float().mean().item()
    uar = np.mean([(all_p[all_l==c]==c).float().mean().item() for c in range(6) if (all_l==c).sum()>0])
    return wa, uar


def train_model(model, train_ldr, val_ldr, use_prosody, args):
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_uar, patience, best_state, best_wa = 0, 0, None, 0
    for ep in range(args.epochs):
        model.train()
        for batch in train_ldr:
            feats, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad()
            if use_prosody and len(batch) >= 3:
                pb = batch[2].to(DEVICE)
                out = model(feats, f0=pb[:,:,0:1], e=pb[:,:,1:2])
            else:
                out = model(feats)
            crit(out, labels).backward()
            opt.step()
        sched.step()
        wa, uar = evaluate(model, val_ldr, use_prosody)
        if uar > best_uar:
            best_uar, best_wa = uar, wa; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= args.patience: break
    model.load_state_dict(best_state)
    return best_wa, best_uar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cremad_dir', default='/d/大学/crema_temp/AudioWAV')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()

    global DEVICE; DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    entries = collect_cremad(args.cremad_dir)
    print(f"Total samples: {len(entries)}")
    from collections import Counter
    dist = Counter(e[1] for e in entries)
    print(f"Class dist: {{{', '.join(f'{CLASS_NAMES[k]}={v}' for k,v in sorted(dist.items()))}}}")

    train_e, val_e, test_e = speaker_split(entries)
    print(f"Split: train={len(train_e)} val={len(val_e)} test={len(test_e)}")

    print("Loading WavLM...")
    backbone = SSLBackbone(model_name='wavlm', frozen=True, device=DEVICE)

    print("Extracting features...")
    t0 = time.time()
    train_f, train_l, train_p = pre_extract(train_e, backbone)
    val_f, val_l, val_p = pre_extract(val_e, backbone)
    test_f, test_l, test_p = pre_extract(test_e, backbone)
    print(f"Extraction done in {time.time()-t0:.0f}s")

    results = {}
    for name, use_prosody in [('A1_mean', False), ('A3_prosody', True)]:
        print(f"\n=== CREMA-D {name} ===")
        td_kw = {'feats': train_f, 'labels': train_l}
        vd_kw = {'feats': val_f, 'labels': val_l}
        sd_kw = {'feats': test_f, 'labels': test_l}
        if use_prosody:
            td_kw['prosody'] = train_p; vd_kw['prosody'] = val_p; sd_kw['prosody'] = test_p

        train_ldr = DataLoader(PreExtDS(**td_kw), args.bs, shuffle=True, collate_fn=collate, num_workers=0)
        val_ldr = DataLoader(PreExtDS(**vd_kw), args.bs, shuffle=False, collate_fn=collate, num_workers=0)
        test_ldr = DataLoader(PreExtDS(**sd_kw), args.bs, shuffle=False, collate_fn=collate, num_workers=0)

        model = (A3Model if use_prosody else A1Model)().to(DEVICE)
        best_wa, best_uar = train_model(model, train_ldr, val_ldr, use_prosody, args)
        test_wa, test_uar = evaluate(model, test_ldr, use_prosody)
        results[name] = {'best_val_wa': float(best_wa), 'best_val_uar': float(best_uar),
                         'test_wa': float(test_wa), 'test_uar': float(test_uar)}
        print(f"  Val: WA={best_wa:.4f} UAR={best_uar:.4f} | Test: WA={test_wa:.4f} UAR={test_uar:.4f}")

    dwa = results['A3_prosody']['test_wa'] - results['A1_mean']['test_wa']
    duar = results['A3_prosody']['test_uar'] - results['A1_mean']['test_uar']
    print(f"\n=== CREMA-D Prosody Pooling gain ===")
    print(f"  ΔWA={dwa:+.4f} ΔUAR={duar:+.4f}")
    print(f"  (Compare: children +2.24pp, IEMOCAP adults -2.12pp)")

    os.makedirs('experiments/v5_622', exist_ok=True)
    with open('experiments/v5_622/cremad_contrast_result.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: experiments/v5_622/cremad_contrast_result.json")


if __name__ == '__main__':
    main()
