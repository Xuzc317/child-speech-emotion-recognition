"""Compute real FD_lang from English vs Telugu SSL features."""
import sys, os, json, time
import numpy as np
from scipy import linalg
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.models.ssl_backbone import SSLBackbone
from src.data.preprocess import collect_wav_files

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 200
MAX_SAMPLES = 500  # Sample size for covariance computation

def extract_features(wav_dir, backbone):
    """Extract WavLM features from a directory."""
    entries = collect_wav_files(wav_dir)
    feats_list = []
    for path, label, _ in entries[:MAX_SAMPLES]:
        import librosa
        y, _ = librosa.load(path, sr=16000)
        target = 16000 * 4
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]
        wav = torch.from_numpy(y).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f = backbone(wav).squeeze(0).cpu().numpy()
        if f.shape[0] > MAX_FRAMES:
            f = f[:MAX_FRAMES]
        else:
            pad = np.zeros((MAX_FRAMES - f.shape[0], f.shape[1]), dtype=np.float32)
            f = np.concatenate([f, pad], axis=0)
        feats_list.append(f)
    return np.stack(feats_list)  # (N, 200, 768)

def compute_fd(feats1, feats2):
    """Full Frechet Distance."""
    # Flatten temporal dimension
    f1 = feats1.reshape(-1, feats1.shape[-1])
    f2 = feats2.reshape(-1, feats2.shape[-1])

    mu1 = f1.mean(axis=0)
    mu2 = f2.mean(axis=0)
    sigma1 = np.cov(f1, rowvar=False)
    sigma2 = np.cov(f2, rowvar=False)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fd, mu1, mu2, sigma1, sigma2

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--en_dir', default='/root/autodl-tmp/child-speech-emotion-recognition/数据集/BESD/BESD/ENGLISH')
    parser.add_argument('--te_dir', default='/root/autodl-tmp/child-speech-emotion-recognition/数据集/BESD/BESD/TELUGU')
    parser.add_argument('--output', default='experiments/v5_622/fd_lang_result.json')
    args = parser.parse_args()

    print(f"Loading WavLM backbone...")
    backbone = SSLBackbone(model_name='wavlm', frozen=True, device=DEVICE)

    print(f"Extracting ENGLISH features (max {MAX_SAMPLES} samples)...")
    t0 = time.time()
    en_feats = extract_features(args.en_dir, backbone)
    print(f"  Shape: {en_feats.shape}, time={time.time()-t0:.0f}s")

    print(f"Extracting TELUGU features (max {MAX_SAMPLES} samples)...")
    t0 = time.time()
    te_feats = extract_features(args.te_dir, backbone)
    print(f"  Shape: {te_feats.shape}, time={time.time()-t0:.0f}s")

    print("Computing FD...")
    fd, mu1, mu2, s1, s2 = compute_fd(en_feats, te_feats)
    print(f"FD_lang (EN vs TE) = {fd:.2f}")

    result = {
        'fd_lang': float(fd),
        'en_samples': int(en_feats.shape[0]),
        'te_samples': int(te_feats.shape[0]),
        'feature_dim': int(en_feats.shape[-1]),
        'note': 'Computed from real WavLM features, not estimated'
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {args.output}")

if __name__ == '__main__':
    main()
