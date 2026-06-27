"""Sub-Phase 2.3: 测量增强引起的分布偏移（Frechet Distance）。

对每组增强实验，计算增强后 SSL 特征相对于原始特征的 Frechet Distance。
验证 FD 越大 → 增强越偏离儿童分布 → 准确率越低。

用法:
  python scripts/measure_distribution_shift.py
"""

import os, sys, argparse, numpy as np
from scipy import linalg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_fd(feats_1, feats_2, max_samples=5000):
    """Frechet Distance between two feature sets."""
    # Pool across time: take mean per utterance
    mu1 = feats_1[:max_samples].mean(axis=1).mean(axis=0) if feats_1.ndim == 3 else feats_1[:max_samples].mean(axis=0)
    mu2 = feats_2[:max_samples].mean(axis=1).mean(axis=0) if feats_2.ndim == 3 else feats_2[:max_samples].mean(axis=0)

    # Flatten: (N, 768) for FD computation
    f1 = feats_1[:max_samples].reshape(-1, feats_1.shape[-1]) if feats_1.ndim == 3 else feats_1[:max_samples]
    f2 = feats_2[:max_samples].reshape(-1, feats_2.shape[-1]) if feats_2.ndim == 3 else feats_2[:max_samples]

    mu1 = f1.mean(axis=0)
    sigma1 = np.cov(f1, rowvar=False)
    mu2 = f2.mean(axis=0)
    sigma2 = np.cov(f2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/')
    args = parser.parse_args()

    # Load original features (C1 = no augmentation)
    orig_train = np.load(os.path.join(args.data_dir, 'train_wavlm_feats.npy'))
    orig_test = np.load(os.path.join(args.data_dir, 'test_wavlm_feats.npy'))

    configs = {
        'C2_adult':   'pitch ±6, stretch 0.7-1.3',
        'C3_child':   'pitch ±3, stretch 0.85-1.15',
        'C4_extreme': 'pitch ±12, stretch 0.5-1.5',
    }

    print(f"{'Config':<12} {'Description':<35} {'FD (train)':<12} {'FD (test)':<12}")
    print("-" * 72)

    results = {}
    for cfg_name, desc in configs.items():
        aug_path = os.path.join(args.data_dir, f'train_{cfg_name}_wavlm_feats.npy')
        if not os.path.exists(aug_path):
            print(f"  {cfg_name:<12} SKIP - file not found: {aug_path}")
            continue
        aug_train = np.load(aug_path)
        aug_test = np.load(os.path.join(args.data_dir, f'test_{cfg_name}_wavlm_feats.npy'))

        fd_train = compute_fd(orig_train, aug_train)
        fd_test = compute_fd(orig_test, aug_test)
        results[cfg_name] = (fd_train, fd_test)
        print(f"  {cfg_name:<12} {desc:<35} {fd_train:<12.4f} {fd_test:<12.4f}")

    print("\n=== Interpretation ===")
    for cfg, (fd_t, fd_v) in sorted(results.items(), key=lambda x: x[1][0]):
        print(f"  {cfg}: FD_train={fd_t:.4f}, FD_test={fd_v:.4f}")
    print("  Expected: C4 > C2 > C3 (FD increases with parameter extremity)")
    print("  If FD ∝ 1/Accuracy: C4 should have lowest accuracy, C3 highest")


if __name__ == '__main__':
    main()
