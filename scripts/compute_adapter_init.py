"""Phase 2.1: 计算 Adapter 统计先验初始化参数

在儿童 (BESD MY) 和成人 (IEMOCAP) 数据集上分别跑 frozen WavLM，
收集帧级 SSL 特征，计算 per-dimension mean/std 差异，
用差值初始化 AcousticCalibrationAdapter 的 scale/bias。

用法:
  python scripts/compute_adapter_init.py \
      --child_wav_dir /path/to/BESD/MY \
      --adult_wav_dir /path/to/IEMOCAP/wavs \
      --output_dir data/ \
      --device cuda
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ssl_backbone import SSLBackbone, preprocess_wav
from src.data.preprocess import collect_wav_files, split_speakers_7_3_with_inner_val


def collect_frame_features(wav_paths, backbone, device, max_files=None):
    """对所有 WAV 提取 SSL 帧级特征，收集所有帧的 (N_frames_total, 768) 矩阵。"""
    all_frames = []
    paths = wav_paths[:max_files] if max_files else wav_paths
    for path in tqdm(paths, desc='Extracting SSL features'):
        try:
            waveform, sr, _ = preprocess_wav(path, target_sr=16000, duration=4.0)
            waveform = waveform.to(device)
            with torch.no_grad():
                feats = backbone(waveform)  # (1, T, 768)
            feats = feats.squeeze(0).cpu().numpy()  # (T, 768)
            all_frames.append(feats)
        except Exception as e:
            pass
    if not all_frames:
        raise RuntimeError("No features extracted!")
    return np.concatenate(all_frames, axis=0)  # (total_frames, 768)


def compute_stats(frame_feats):
    """计算 per-dimension mean 和 std。"""
    mean = frame_feats.mean(axis=0)  # (768,)
    std = frame_feats.std(axis=0)    # (768,)
    std = np.clip(std, 1e-5, None)   # 防止除零
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child_wav_dir', default=None)
    parser.add_argument('--adult_wav_dir', default=None)
    parser.add_argument('--output_dir', default='data/')
    parser.add_argument('--model', default='wavlm')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_child', type=int, default=2000, help='Limit child samples')
    parser.add_argument('--max_adult', type=int, default=2000, help='Limit adult samples')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Default paths
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_base = os.path.join(script_dir, '..', '提交到团队', '数据集')
    child_dir = args.child_wav_dir or os.path.join(data_base, 'BESD', 'BESD', 'MY')
    adult_dir = args.adult_wav_dir or os.path.join(data_base, 'IEMOCAP', 'wavs')

    # Load backbone
    print(f"Loading {args.model} backbone (frozen)...")
    backbone = SSLBackbone(model_name=args.model, frozen=True, device=device)

    # Collect child features — CRITICAL: ONLY use TRAIN speakers for adapter_init.
    # Using val or test speakers here would leak their distribution statistics
    # into the adapter's initial parameters (scale/bias), violating the
    # hold-out principle. This is enforced by the split function.
    print(f"\n=== Child speech (TRAIN-ONLY for adapter init): {child_dir} ===")
    all_entries = collect_wav_files(child_dir)
    train_entries, _, _, _, _, _, _ = split_speakers_7_3_with_inner_val(all_entries)
    child_paths = [e[0] for e in train_entries]  # ONLY train speaker paths
    print(f"  {len(child_paths)} train-only WAVs (val+test excluded, from {len(all_entries)} total)")
    child_feats = collect_frame_features(child_paths, backbone, device, args.max_child)
    child_mean, child_std = compute_stats(child_feats)
    print(f"  Frames: {child_feats.shape[0]}, Mean norm: {np.linalg.norm(child_mean):.2f}")

    # Collect adult features (IEMOCAP has no speaker split needed — all adult)
    print(f"\n=== Adult speech: {adult_dir} ===")
    adult_paths = []
    for cls_name in sorted(os.listdir(adult_dir)):
        cls_dir = os.path.join(adult_dir, cls_name)
        if os.path.isdir(cls_dir):
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.wav'):
                    adult_paths.append(os.path.join(cls_dir, fname))
    print(f"  Found {len(adult_paths)} WAVs")
    adult_feats = collect_frame_features(adult_paths, backbone, device, args.max_adult)
    adult_mean, adult_std = compute_stats(adult_feats)
    print(f"  Frames: {adult_feats.shape[0]}, Mean norm: {np.linalg.norm(adult_mean):.2f}")

    # Compute adapter initialization
    # 目标：使 child feature 经 x*scale + bias 后的分布对齐 adult 分布
    # E[x*scale + bias] = μ_adult  ⇒  bias = μ_adult - μ_child * scale
    # Var[x*scale + bias] ≈ Var[adult]  ⇒  scale = σ_adult / σ_child
    scale_init = adult_std / child_std
    bias_init = adult_mean - child_mean * scale_init

    print(f"\n=== Adapter Init Stats ===")
    print(f"  scale: min={scale_init.min():.4f}, max={scale_init.max():.4f}, mean={scale_init.mean():.4f}")
    print(f"  bias:  min={bias_init.min():.4f}, max={bias_init.max():.4f}, mean={bias_init.mean():.4f}")
    print(f"  scale deviation from 1.0: {np.abs(scale_init - 1.0).mean():.4f}")
    print(f"  bias deviation from 0.0: {np.abs(bias_init).mean():.4f}")

    # Save
    out = {
        'scale': scale_init.astype(np.float32),
        'bias': bias_init.astype(np.float32),
        'child_mean': child_mean.astype(np.float32),
        'child_std': child_std.astype(np.float32),
        'adult_mean': adult_mean.astype(np.float32),
        'adult_std': adult_std.astype(np.float32),
    }
    save_path = os.path.join(args.output_dir, 'adapter_init.npz')
    np.savez(save_path, **out)
    print(f"\nSaved adapter init to {save_path}")

    # Also compute Frechet Distance between child and adult SSL distributions
    diff_mean = adult_mean - child_mean
    fd = float(diff_mean @ diff_mean)
    print(f"Distribution gap (||mean_diff||^2): {fd:.4f}")
    print("Done!")


if __name__ == '__main__':
    main()
