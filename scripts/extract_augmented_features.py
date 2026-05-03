"""Sub-Phase 2.3: 预增强 + SSL 特征提取（离线模式）

为四组增强实验生成预提取特征：
  C1: 无增强（= 现有的 train_wavlm_feats.npy）
  C2: 成人参数（pitch ±6, stretch 0.7-1.3）
  C3: 儿童约束（pitch ±3, stretch 0.85-1.15）
  C4: 极端参数（pitch ±12, stretch 0.5-1.5）

用法:
  python scripts/extract_augmented_features.py --device cuda
"""

import os, sys, argparse, time, numpy as np, torch, librosa
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocess import collect_wav_files, split_speakers_7_3_with_inner_val
from src.models.ssl_backbone import SSLBackbone, preprocess_wav

# Augmentation configs
CONFIGS = {
    'C1_none':     {'pitch_range': (0, 0),    'stretch_range': (1.0, 1.0),   'noise_snr': None},
    'C2_adult':    {'pitch_range': (-6, 6),   'stretch_range': (0.7, 1.3),   'noise_snr': None},
    'C3_child':    {'pitch_range': (-3, 3),   'stretch_range': (0.85, 1.15), 'noise_snr': None},
    'C4_extreme':  {'pitch_range': (-12, 12), 'stretch_range': (0.5, 1.5),   'noise_snr': None},
}


def apply_aug(waveform, sr, config, rng):
    """Apply pitch shift and time stretch."""
    if config['pitch_range'][1] > 0:
        n_steps = rng.uniform(*config['pitch_range'])
        waveform = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)
    if config['stretch_range'][1] > 1.0:
        rate = rng.uniform(*config['stretch_range'])
        stretched = librosa.effects.time_stretch(waveform, rate=rate)
        # Align length
        if len(stretched) > len(waveform):
            stretched = stretched[:len(waveform)]
        else:
            stretched = np.pad(stretched, (0, len(waveform) - len(stretched)))
        waveform = stretched
    return waveform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', default='wavlm')
    parser.add_argument('--output_dir', default='data/')
    parser.add_argument('--configs', nargs='+', default=['C1_none', 'C2_adult', 'C3_child', 'C4_extreme'])
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--wav_dir', default=None, help='BESD MY dataset directory')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # Collect WAVs with speaker split
    wav_dir = args.wav_dir
    if not wav_dir:
        wav_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               '..', '提交到团队', '数据集', 'BESD', 'BESD', 'MY')
    entries = collect_wav_files(wav_dir)
    train_entries, val_entries, test_entries, _, _, _, _ = split_speakers_7_3_with_inner_val(entries)

    # Load backbone once
    print(f"Loading {args.model} backbone (frozen)...")
    backbone = SSLBackbone(model_name=args.model, frozen=True, device=device)

    max_frames = 200
    target_sr = 16000
    target_len = target_sr * 4

    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        print(f"\n{'='*60}")
        print(f"Config: {cfg_name} — pitch={cfg['pitch_range']}, stretch={cfg['stretch_range']}")
        print(f"{'='*60}")

        # Only extract TRAIN features. Val/test always use clean features
        # (augmentation only affects training data; val/test must stay clean for fair eval).
        for split_name, entries in [('train', train_entries)]:
            it = entries[:args.max_samples] if args.max_samples else entries
            feats_list, labels_list, skipped = [], [], 0
            t0 = time.time()

            for idx, (path, label, _) in enumerate(tqdm(it, desc=f'{cfg_name} {split_name}')):
                try:
                    data, sr = librosa.load(path, sr=target_sr)
                    if len(data) < target_len:
                        data = np.pad(data, (0, target_len - len(data)))
                    else:
                        data = data[:target_len]

                    # Apply augmentation
                    data_aug = apply_aug(data, sr, cfg, rng)

                    # SSL feature extraction
                    waveform = torch.from_numpy(data_aug).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        feats = backbone(waveform).squeeze(0).cpu().numpy()  # (T, 768)

                    # Pad to max_frames
                    if feats.shape[0] > max_frames:
                        feats = feats[:max_frames]
                    else:
                        pad = np.zeros((max_frames - feats.shape[0], feats.shape[1]), dtype=np.float32)
                        feats = np.concatenate([feats, pad], axis=0)

                    feats_list.append(feats)
                    labels_list.append(label)
                except Exception as e:
                    skipped += 1

                if (idx + 1) % 500 == 0:
                    print(f"  [{idx+1}/{len(it)}] {time.time()-t0:.0f}s")

            feats_arr = np.stack(feats_list, axis=0).astype(np.float32)
            labels_arr = np.array(labels_list, dtype=np.int64)

            feat_path = os.path.join(args.output_dir, f'{split_name}_{cfg_name}_wavlm_feats.npy')
            label_path = os.path.join(args.output_dir, f'{split_name}_{cfg_name}_labels.npy')
            np.save(feat_path, feats_arr)
            np.save(label_path, labels_arr)
            print(f"  Saved {feats_arr.shape} to {feat_path} ({skipped} skipped)")

    print("\nDone! All augmented features extracted.")


if __name__ == '__main__':
    main()
