import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

base = "D:/大学/论文/儿童语音情绪识别/提交到团队/数据集/BESD/BESD/MY"

# Use files that actually exist
test_files = [
    f"{base}/ANGER/1.EF_12 Angry_1.wav",
    f"{base}/HAPPY/1.EF_7 happy_1.wav",
    f"{base}/NEUTRAL/1.EF_7 neutral_1.wav",
    f"{base}/SAD/1.EF_7 sad_1.wav",
]

for fp in test_files:
    try:
        y, sr = librosa.load(fp, sr=16000)
        dur = len(y) / sr
        hop = 512

        print(f"\n=== {fp.split('/')[-1]} === dur={dur:.2f}s")

        # Method 1: Onset detection (default)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop)
        print(f"Onset detect: {len(onsets)} syll ({len(onsets)/dur:.2f}/s)")

        # Method 2: Onset strength peaks
        o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        o_s = gaussian_filter1d(o_env, sigma=1.5)
        for h_factor in [0.5, 1.0, 2.0]:
            th = np.mean(o_s) + h_factor * np.std(o_s)
            peaks_o, _ = find_peaks(o_s, height=th, distance=int(0.08*sr/hop))
            print(f"  Onset peaks (mean+{h_factor}*std): {len(peaks_o)} ({len(peaks_o)/dur:.2f}/s)")

        # Method 3: RMS envelope peaks
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        for sigma in [1, 2, 3]:
            rms_s = gaussian_filter1d(rms, sigma=sigma)
            rms_n = rms_s / (np.max(rms_s) + 1e-10)
            for h in [0.05, 0.08, 0.12]:
                p, _ = find_peaks(rms_n, height=h, distance=int(0.12*sr/hop))
                rate = len(p) / dur
                print(f"  RMS(sigma={sigma},h={h}): {len(p)} ({rate:.2f}/s)")

        # Method 4: Spectral centroid peaks
        spec_c = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        spec_s = gaussian_filter1d(spec_c, sigma=2)
        spec_n = (spec_s - np.min(spec_s)) / (np.max(spec_s) - np.min(spec_s) + 1e-10)
        p_sc, _ = find_peaks(spec_n, height=0.2, distance=int(0.12*sr/hop))
        print(f"  Spectral centroid: {len(p_sc)} ({len(p_sc)/dur:.2f}/s)")

        print(f"  Target (~3.5/s): {dur*3.5:.0f} syll")

    except Exception as e:
        print(f"Error: {e}")
