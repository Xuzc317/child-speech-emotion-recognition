"""Supplementary experiments for paper v7 revision.

1. IEMOCAP 4-class benchmark (A1 + A3, to compare with Kakouros 2022)
2. C-BESD A2 stat-prior vs random-init Adapter comparison
3. Acoustic statistics (F0/energy mean/std/range) for all three datasets

Logs results to both stdout and a summary JSON.
"""
import os, sys, argparse, time, json
import numpy as np
from collections import Counter

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data.dataset_ssl import SSLFeatureDataset, collate_fn_ssl_features
from src.models.ssl_backbone import SSLBackbone
from src.models.pooling import TemporalImportancePooling, extract_prosody
from src.models.semlp import SEMLP
from src.models.adapter import AcousticCalibrationAdapter
from src.data.preprocess import collect_wav_files, split_speakers_7_3_with_inner_val

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD']

# IEMOCAP 4-class subset
IEMOCAP_4CLASS = ['ANGER', 'HAPPY', 'NEUTRAL', 'SAD']

class DistributionCalibratedSER(nn.Module):
    def __init__(self, config, adapter_init=None):
        super().__init__()
        dim = 768
        self.use_backbone = config.get('use_backbone', False)
        self.use_adapter = config.get('use_adapter', False)
        self.use_prosody = config.get('use_prosody', False)
        if self.use_adapter:
            self.adapter = AcousticCalibrationAdapter(dim=dim, init_scale=adapter_init.get('scale') if adapter_init else None, init_bias=adapter_init.get('bias') if adapter_init else None)
        if self.use_prosody:
            self.pooling = TemporalImportancePooling(ssl_dim=dim)
        self.classifier = SEMLP(input_dim=dim, num_classes=config.get('num_classes', 6))

    def forward(self, x, f0=None, energy=None, mask=None):
        if self.use_adapter:
            x = self.adapter(x)
        if self.use_prosody and f0 is not None and energy is not None:
            pooled = self.pooling(x, f0, energy, mask=mask)
        elif mask is not None:
            m = mask.to(dtype=x.dtype).unsqueeze(-1)
            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return self.classifier(pooled)


def evaluate(model, loader, use_prosody, device):
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for batch in loader:
            feats, labels = batch[0].to(device), batch[1].to(device)
            mask = batch[2].to(device) if len(batch) >= 3 else None
            prosody = batch[3].to(device) if len(batch) >= 4 else None
            if use_prosody and prosody is not None:
                out = model(feats, f0=prosody[:,:,0:1], energy=prosody[:,:,1:2], mask=mask)
            else:
                out = model(feats, mask=mask)
            all_p.append(out.argmax(dim=1).cpu())
            all_l.append(labels.cpu())
    all_p = torch.cat(all_p); all_l = torch.cat(all_l)
    wa = (all_p == all_l).float().mean().item()
    uar = np.mean([(all_p[all_l==c]==c).float().mean().item() for c in range(all_p.max().item()+1)])
    return wa, uar


def train_one_run(model, train_ldr, val_ldr, use_prosody, device, args):
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_uar, patience, best_state, best_wa = 0, 0, None, 0
    for ep in range(args.epochs):
        model.train()
        for batch in train_ldr:
            feats, labels = batch[0].to(device), batch[1].to(device)
            mask = batch[2].to(device) if len(batch) >= 3 else None
            prosody = batch[3].to(device) if len(batch) >= 4 else None
            opt.zero_grad()
            if use_prosody and prosody is not None:
                out = model(feats, f0=prosody[:,:,0:1], energy=prosody[:,:,1:2], mask=mask)
            else:
                out = model(feats, mask=mask)
            crit(out, labels).backward()
            opt.step()
        sched.step()
        wa, uar = evaluate(model, val_ldr, use_prosody, device)
        if uar > best_uar:
            best_uar, best_wa = uar, wa; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= args.patience: break
    model.load_state_dict(best_state)
    return best_wa, best_uar


def compute_acoustic_stats(wav_entries):
    """Compute F0, energy, duration statistics from WAV files."""
    import librosa
    f0_means, f0_maxs, energy_means = [], [], []
    for path, _, _ in tqdm(wav_entries, desc='Acoustic stats'):
        y, sr = librosa.load(path, sr=16000)
        dur = len(y) / sr
        f0 = librosa.yin(y, fmin=65, fmax=2093, sr=sr, hop_length=320)
        f0_v = f0[f0 > 0]
        energy = librosa.feature.rms(y=y, hop_length=320).squeeze()
        if len(f0_v) > 0:
            f0_means.append(f0_v.mean())
            f0_maxs.append(f0_v.max())
        energy_means.append(energy.mean())
    return {
        'n': len(wav_entries),
        'f0_mean': float(np.mean(f0_means)), 'f0_std': float(np.std(f0_means)),
        'f0_max': float(np.max(f0_maxs)),
        'energy_mean': float(np.mean(energy_means)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--data_dir', default='/root/autodl-tmp/v5_data/')
    parser.add_argument('--iemocap_dir', default='/root/autodl-tmp/IEMOCAP/wavs/')
    parser.add_argument('--output_dir', default='experiments/v5_622/')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    # ===== 1. C-BESD A2 stat-prior vs random-init =====
    print("\n" + "="*60)
    print("1. C-BESD A2 STAT-PRIOR vs RANDOM-INIT ADAPTER")
    print("="*60)

    data_dir = args.data_dir
    for init_mode in ['random', 'stat_prior']:
        adapter_init = None
        if init_mode == 'stat_prior':
            init_paths = [f'{data_dir}/adapter_init.npz', '/root/ser_project/data/adapter_init.npz']
            init_path = None
            for p in init_paths:
                if os.path.exists(p):
                    init_path = p
                    break
            if init_path:
                adapter_init = dict(np.load(init_path))
                print(f"  Loaded stat prior from {init_path}")
            else:
                print("  Stat prior not found, skipping")
                continue

        seed_results = {'val': [], 'test': []}
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            config = {'use_backbone': False, 'use_adapter': True, 'use_prosody': False, 'num_classes': 6}
            model = DistributionCalibratedSER(config, adapter_init=adapter_init).to(device)

            train_ds = SSLFeatureDataset(f'{data_dir}/train_wavlm_feats.npy', f'{data_dir}/train_wavlm_labels.npy')
            val_ds = SSLFeatureDataset(f'{data_dir}/val_wavlm_feats.npy', f'{data_dir}/val_wavlm_labels.npy')
            test_ds = SSLFeatureDataset(f'{data_dir}/test_wavlm_feats.npy', f'{data_dir}/test_wavlm_labels.npy')
            train_ldr = DataLoader(train_ds, args.bs, shuffle=True, collate_fn=collate_fn_ssl_features, num_workers=4)
            val_ldr = DataLoader(val_ds, args.bs, shuffle=False, collate_fn=collate_fn_ssl_features, num_workers=4)
            test_ldr = DataLoader(test_ds, args.bs, shuffle=False, collate_fn=collate_fn_ssl_features, num_workers=4)

            best_wa, best_uar = train_one_run(model, train_ldr, val_ldr, False, device, args)
            test_wa, test_uar = evaluate(model, test_ldr, False, device)
            print(f"  A2_{init_mode} seed={seed}: val_wa={best_wa:.4f} val_uar={best_uar:.4f} test_wa={test_wa:.4f} test_uar={test_uar:.4f}")
            seed_results['val'].append(best_wa)
            seed_results['test'].append(test_wa)

        results[f'A2_{init_mode}'] = {
            'val_mean': float(np.mean(seed_results['val'])), 'val_std': float(np.std(seed_results['val'])),
            'test_mean': float(np.mean(seed_results['test'])), 'test_std': float(np.std(seed_results['test'])),
        }
        print(f"  A2_{init_mode} mean: val={results[f'A2_{init_mode}']['val_mean']:.4f} test={results[f'A2_{init_mode}']['test_mean']:.4f}")

    # ===== 2. IEMOCAP 4-class =====
    print("\n" + "="*60)
    print("2. IEMOCAP 4-CLASS BENCHMARK")
    print("="*60)

    iemocap_dir = args.iemocap_dir
    if os.path.isdir(iemocap_dir):
        # Collect IEMOCAP entries, filter to 4 classes
        all_entries = []
        for cls_name in sorted(os.listdir(iemocap_dir)):
            cls_dir = os.path.join(iemocap_dir, cls_name)
            if not os.path.isdir(cls_dir) or cls_name not in IEMOCAP_4CLASS:
                continue
            label_idx = IEMOCAP_4CLASS.index(cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.wav'):
                    all_entries.append((os.path.join(cls_dir, fname), label_idx, fname[:5]))

        print(f"  IEMOCAP 4-class samples: {len(all_entries)}")
        for cls_name in IEMOCAP_4CLASS:
            print(f"    {cls_name}: {sum(1 for e in all_entries if e[1]==IEMOCAP_4CLASS.index(cls_name))}")

        # Speaker split
        import librosa
        speaker_map = {}
        for path, label, sid in all_entries:
            speaker_map.setdefault(sid, []).append((path, label))
        sids = sorted(speaker_map.keys())
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(sids))
        n_test = max(1, int(len(sids)*0.2))
        n_val = max(1, int((len(sids)-n_test)*0.15))
        test_sids = set(sids[i] for i in perm[:n_test])
        val_sids = set(sids[i] for i in perm[n_test:n_test+n_val])
        train_sids = set(sids[i] for i in perm[n_test+n_val:])

        def gather(sids): result = []; [result.extend(speaker_map[s]) for s in sids]; return result
        train_e, val_e, test_e = gather(train_sids), gather(val_sids), gather(test_sids)
        print(f"  Split: train={len(train_e)} val={len(val_e)} test={len(test_e)}")

        # Extract features
        backbone = SSLBackbone(model_name='wavlm', frozen=True, device=device)
        def extract(entries_list, use_prosody=True):
            feats, labels, prosody = [], [], []
            for path, label in tqdm(entries_list, desc='Extract'):
                y, _ = librosa.load(path, sr=16000)
                target = 16000 * 4
                if len(y) < target: y = np.pad(y, (0, target-len(y)))
                else: y = y[:target]
                wav = torch.from_numpy(y).float().unsqueeze(0).to(device)
                with torch.no_grad(): f = backbone(wav).squeeze(0).cpu()
                if f.shape[0] > 200: f = f[:200]
                else: f = torch.cat([f, torch.zeros(200-f.shape[0], f.shape[1])], dim=0)
                feats.append(f); labels.append(label)
                if use_prosody:
                    f0, en = extract_prosody(y, sr=16000, hop_length=320)
                    fp = np.zeros(200, dtype=np.float32); ep = np.zeros(200, dtype=np.float32)
                    fl, el = min(len(f0),200), min(len(en),200)
                    fp[:fl]=f0[:fl]; ep[:el]=en[:el]
                    prosody.append(np.stack([fp, ep], axis=1))
            r = [torch.stack(feats), torch.tensor(labels, dtype=torch.long)]
            if use_prosody: r.append(torch.tensor(np.stack(prosody), dtype=torch.float32))
            return tuple(r)

        train_d = extract(train_e); val_d = extract(val_e); test_d = extract(test_e)

        class PreDS(Dataset):
            def __init__(self, feats, labels, prosody=None):
                self.f, self.l, self.p = feats, labels, prosody
            def __len__(self): return len(self.l)
            def __getitem__(self, i):
                if self.p is not None: return self.f[i], self.l[i], self.p[i]
                return self.f[i], self.l[i]

        def coll(batch):
            f = torch.stack([b[0] for b in batch]); l = torch.stack([b[1] for b in batch])
            if len(batch[0])==3:
                p = torch.stack([b[2] for b in batch])
                mask = torch.ones(len(batch), f.shape[1], dtype=torch.bool)
                return f, l, mask, p
            return f, l, torch.ones(len(batch), f.shape[1], dtype=torch.bool)

        for name, use_prosody, ncls in [('A1_mean', False, 4), ('A3_prosody', True, 4)]:
            print(f"\n  IEMOCAP 4-class {name}:")
            dsk = {'feats': train_d[0], 'labels': train_d[1]}; vk = {'feats': val_d[0], 'labels': val_d[1]}; sk = {'feats': test_d[0], 'labels': test_d[1]}
            if use_prosody: dsk['prosody']=train_d[2]; vk['prosody']=val_d[2]; sk['prosody']=test_d[2]
            tdl = DataLoader(PreDS(**dsk), args.bs, shuffle=True, collate_fn=coll, num_workers=4)
            vdl = DataLoader(PreDS(**vk), args.bs, shuffle=False, collate_fn=coll, num_workers=4)
            sdl = DataLoader(PreDS(**sk), args.bs, shuffle=False, collate_fn=coll, num_workers=4)

            config = {'use_backbone': False, 'use_adapter': False, 'use_prosody': use_prosody, 'num_classes': ncls}
            model = DistributionCalibratedSER(config).to(device)
            bwa, buar = train_one_run(model, tdl, vdl, use_prosody, device, args)
            twa, tuar = evaluate(model, sdl, use_prosody, device)
            print(f"    Val: WA={bwa:.4f} UAR={buar:.4f} | Test: WA={twa:.4f} UAR={tuar:.4f}")
            results[f'iemocap_4class_{name}'] = {'val_wa': float(bwa), 'val_uar': float(buar), 'test_wa': float(twa), 'test_uar': float(tuar)}

        # Acoustic stats
        print("\n  Computing IEMOCAP acoustic stats...")
        ie_stats = compute_acoustic_stats(all_entries)
        results['iemocap_acoustic'] = ie_stats
        print(f"  F0 mean={ie_stats['f0_mean']:.1f}, max={ie_stats['f0_max']:.1f}, energy={ie_stats['energy_mean']:.4f}")
    else:
        print(f"  IEMOCAP not found at {iemocap_dir}")

    # ===== Save =====
    out_path = f'{args.output_dir}/supplementary_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
