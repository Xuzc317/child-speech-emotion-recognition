"""Sub-Phase 2.2: 预提取韵律特征 (F0 + energy)，与 SSL 帧对齐保存。

用法:
  python scripts/extract_prosody_features.py \
      --wav_dir /path/to/BESD/MY \
      --output_dir data/ \
      --prefix wavlm
"""

import os, sys, argparse, time
import numpy as np
import librosa
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocess import collect_wav_files, split_speakers_7_3_with_inner_val, CLASS_NAMES
from src.models.pooling import extract_prosody


def extract_and_align(wav_path, target_sr=16000, target_frames=200, hop_length=320):
    """提取 F0 + energy 并对齐到 target_frames 长度。"""
    data, sr = librosa.load(wav_path, sr=target_sr)
    target_len = target_sr * 4  # 4 seconds
    if len(data) < target_len:
        data = np.pad(data, (0, target_len - len(data)))
    else:
        data = data[:target_len]

    f0, energy = extract_prosody(data, sr=sr, hop_length=hop_length)

    # Pad/truncate to target_frames
    f0 = np.pad(f0[:target_frames], (0, max(0, target_frames - len(f0))))
    energy = np.pad(energy[:target_frames], (0, max(0, target_frames - len(energy))))
    f0 = f0[:target_frames]
    energy = energy[:target_frames]

    # Stack as (T, 2)
    return np.stack([f0, energy], axis=1).astype(np.float32)  # (200, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', default=None)
    parser.add_argument('--output_dir', default='data/')
    parser.add_argument('--prefix', default='wavlm')
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    wav_dir = args.wav_dir
    if not wav_dir:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wav_dir = os.path.join(script_dir, '..', '提交到团队', '数据集', 'BESD', 'BESD', 'MY')

    entries = collect_wav_files(wav_dir)
    train_entries, val_entries, test_entries, _, _, _, _ = split_speakers_7_3_with_inner_val(entries)

    def process(entries, split_name):
        feats, skipped = [], 0
        it = entries[:args.max_samples] if args.max_samples else entries
        t0 = time.time()
        for idx, (path, label, sid) in enumerate(tqdm(it, desc=f'{split_name} prosody')):
            try:
                feat = extract_and_align(path)
                feats.append(feat)
            except Exception as e:
                feats.append(np.zeros((200, 2), dtype=np.float32))
                skipped += 1
            if (idx + 1) % 1000 == 0:
                print(f"  [{idx+1}/{len(it)}] {time.time()-t0:.0f}s")

        arr = np.stack(feats, axis=0)  # (N, 200, 2)
        out = os.path.join(args.output_dir, f'{split_name}_{args.prefix}_prosody.npy')
        np.save(out, arr)
        print(f"  Saved {arr.shape} to {out} ({skipped} skipped)")

    process(train_entries, 'train')
    process(val_entries,  'val')
    process(test_entries, 'test')
    print("Done!")


if __name__ == '__main__':
    main()
