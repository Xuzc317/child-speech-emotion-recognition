"""Phase 1：预提取 SSL 帧级特征

为 BESD MY 数据集的每个 WAV 提取 emotion2vec/WavLM 的帧级特征，
保存到 .npy 文件供训练时使用。

用法：
  python scripts/extract_ssl_features.py \
      --wav_dir /path/to/BESD/MY \
      --output_dir data/ \
      --model emotion2vec \
      --device cuda
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocess import collect_wav_files, split_speakers_3way, CLASS_NAMES
from src.models.ssl_backbone import SSLBackbone, preprocess_wav


def extract_ssl_feature(path, backbone, device, target_sr=16000):
    """对单个 WAV 提取 SSL 帧级特征。"""
    waveform, sr, _ = preprocess_wav(path, target_sr=target_sr, duration=4.0)
    waveform = waveform.to(device)
    with torch.no_grad():
        feats = backbone(waveform)  # (1, T, 768)
    return feats.squeeze(0).cpu().numpy()  # (T, 768)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', default=None, help='BESD MY dataset directory')
    parser.add_argument('--output_dir', default='data/', help='Output directory')
    parser.add_argument('--model', default='emotion2vec', choices=['emotion2vec', 'wavlm'])
    parser.add_argument('--prefix', default='', help='Filename prefix (e.g. wavlm_ or e2v_)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit samples for testing')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 收集 WAV 文件
    print("Collecting WAV files...")
    wav_dir = args.wav_dir
    if not wav_dir:
        # Fallback: search known locations relative to this script
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(script_dir, "..", "提交到团队", "数据集", "BESD", "BESD", "MY"),
            os.path.join(script_dir, "..", "..", "提交到团队", "数据集", "BESD", "BESD", "MY"),
            os.path.join(script_dir, "..", "..", "..", "提交到团队", "数据集", "BESD", "BESD", "MY"),
        ]
        for cand in candidates:
            if os.path.isdir(cand):
                wav_dir = cand
                break
        if not wav_dir:
            print("ERROR: --wav_dir not specified and fallback not found.")
            print("Checked:", candidates)
            sys.exit(1)
    entries = collect_wav_files(wav_dir)

    # 2. Speaker-independent 60/20/20 三路划分
    print("Splitting speakers (60/20/20)...")
    train_entries, val_entries, test_entries, train_sids, val_sids, test_sids = split_speakers_3way(entries)
    print(f"  Train: {len(train_entries)} WAVs from {len(train_sids)} speakers")
    print(f"  Val:   {len(val_entries)} WAVs from {len(val_sids)} speakers")
    print(f"  Test:  {len(test_entries)} WAVs from {len(test_sids)} speakers")

    if args.max_samples:
        train_entries = train_entries[:args.max_samples]
        val_entries = val_entries[:args.max_samples]
        test_entries = test_entries[:args.max_samples]

    # 3. 加载 SSL backbone（frozen）
    print(f"Loading {args.model} backbone (frozen)...")
    backbone = SSLBackbone(model_name=args.model, frozen=True, device=device)

    # 4. 提取特征
    def extract_and_save(entries, split_name, tag):
        features = []
        labels = []
        skipped = 0
        max_frames = 200  # 4s @ 50Hz
        print(f"Extracting {split_name} features...")
        t0 = time.time()
        for idx, (path, label, sid) in enumerate(entries):
            try:
                feat = extract_ssl_feature(path, backbone, device)
                # Pad or truncate to max_frames
                if feat.shape[0] > max_frames:
                    feat = feat[:max_frames]
                else:
                    pad = np.zeros((max_frames - feat.shape[0], feat.shape[1]), dtype=np.float32)
                    feat = np.concatenate([feat, pad], axis=0)
                features.append(feat)
                labels.append(label)
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"  SKIP: {os.path.basename(path)} — {e}")
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [{idx+1}/{len(entries)}] {elapsed:.0f}s elapsed")

        if skipped:
            print(f"  Skipped {skipped} files")

        # Stack into rectangular array (N, T, 768)
        feats_arr = np.stack(features, axis=0).astype(np.float32)
        labels_arr = np.array(labels, dtype=np.int64)
        np.save(os.path.join(args.output_dir, f'{split_name}_{tag}_feats.npy'), feats_arr)
        np.save(os.path.join(args.output_dir, f'{split_name}_{tag}_labels.npy'), labels_arr)
        print(f"  Saved {len(features)} samples to {split_name}_{tag}_feats.npy (shape: {feats_arr.shape})")

    tag = args.prefix if args.prefix else 'ssl'
    extract_and_save(train_entries, 'train', tag)
    extract_and_save(val_entries,  'val',   tag)
    extract_and_save(test_entries, 'test',  tag)

    print("Done!")


if __name__ == "__main__":
    main()
