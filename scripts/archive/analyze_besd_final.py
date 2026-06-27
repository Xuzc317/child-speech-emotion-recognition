# -*- coding: utf-8 -*-
"""
C-BESD Dataset Statistical Analysis - Final Version
Computes: F0, Duration, Silence, SNR, Syllable Rate
"""
import os, time, warnings
import numpy as np
import pandas as pd
import librosa
import parselmouth
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import sys
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

BASE = "D:/大学/论文/儿童语音情绪识别/提交到团队/数据集/BESD/BESD/MY"
EMOTIONS = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD']

def extract_f0(y, sr):
    try:
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
        f0 = pitch.selected_array['frequency']
        f0 = f0[f0 > 0]
        if len(f0) > 10:
            return float(np.mean(f0)), float(np.std(f0))
    except:
        pass
    return np.nan, np.nan

def silence_stats(y, sr, top_db=25, min_len=0.03):
    non_silent = librosa.effects.split(y, top_db=top_db, hop_length=512, frame_length=2048)
    if len(non_silent) == 0:
        return 0.0, 0.0
    sil = []
    prev = 0
    for s, e in non_silent:
        if s > prev:
            sl = (s - prev) / sr
            if sl >= min_len:
                sil.append(sl)
        prev = e
    if prev < len(y):
        sl = (len(y) - prev) / sr
        if sl >= min_len:
            sil.append(sl)
    if sil:
        return float(np.mean(sil)), float(np.sum(sil))
    return 0.0, 0.0

def compute_snr_segmental(y, sr):
    frame_length = int(0.032 * sr)
    hop_length = int(0.016 * sr)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    frame_energies = np.sum(frames ** 2, axis=0)
    if len(frame_energies) < 5:
        return np.nan
    sorted_e = np.sort(frame_energies)
    noise_idx = max(1, int(len(sorted_e) * 0.20))
    noise_floor = np.mean(sorted_e[:noise_idx])
    if noise_floor < 1e-15:
        return np.nan
    frame_snr = 10 * np.log10(frame_energies / noise_floor)
    frame_snr = np.clip(frame_snr, -10, 35)
    return float(np.mean(frame_snr))

def estimate_syllables(y, sr):
    dur = len(y) / sr
    if dur < 0.2:
        return np.nan, 0
    hop = 512

    # Onset strength with careful parameter tuning
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)
    # Light smoothing
    o_smooth = gaussian_filter1d(o_env, sigma=2.0)
    o_norm = o_smooth / (np.max(o_smooth) + 1e-10)

    # Adaptive threshold
    threshold = max(0.06 * np.max(o_norm), np.mean(o_norm) + 0.2 * np.std(o_norm))
    min_dist = int(0.08 * sr / hop)  # 80ms minimum between syllables

    peaks, _ = find_peaks(o_norm, height=threshold, distance=min_dist)
    n_syll = len(peaks)

    # Also get raw onset count
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units='time')
    n_onsets = len(onsets)

    # Use the larger of the two estimates (they capture different aspects)
    estimated = max(n_syll, int(n_onsets / 1.6))
    estimated = min(estimated, int(dur * 10))
    estimated = max(estimated, 1)

    rate = estimated / dur
    return rate, estimated

# Collect files
all_files = []
for emo in EMOTIONS:
    d = os.path.join(BASE, emo)
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        if f.endswith('.wav'):
            all_files.append((emo, f))

print(f"Total files: {len(all_files)}")
print("-" * 60)
start = time.time()

results = []
for idx, (emo, fname) in enumerate(all_files):
    fp = os.path.join(BASE, emo, fname)
    try:
        y, sr = librosa.load(fp, sr=16000, mono=True)
        dur = len(y) / sr
        if dur < 0.1:
            continue

        f0_m, f0_s = extract_f0(y, sr)
        sil_m, sil_t = silence_stats(y, sr)
        snr_seg = compute_snr_segmental(y, sr)
        syll_rate, n_syll = estimate_syllables(y, sr)

        results.append({
            'filename': fname, 'emotion': emo,
            'duration': round(dur, 3),
            'f0_mean': round(f0_m, 2) if not np.isnan(f0_m) else np.nan,
            'f0_std': round(f0_s, 2) if not np.isnan(f0_s) else np.nan,
            'avg_silence_len': round(sil_m, 4),
            'total_silence': round(sil_t, 4),
            'snr': round(snr_seg, 2) if not np.isnan(snr_seg) else np.nan,
            'syllable_rate': round(syll_rate, 3) if not np.isnan(syll_rate) else np.nan,
            'n_syllables': n_syll
        })
    except Exception as e:
        print(f"  Error [{emo}/{fname}]: {e}")

    if (idx + 1) % 500 == 0:
        print(f"  [{idx+1}/{len(all_files)}] {time.time()-start:.0f}s")

print(f"\nDone: {len(results)} files in {time.time()-start:.0f}s")

df = pd.DataFrame(results)
col_f0 = df['f0_mean'].dropna()
col_sil = df[df['avg_silence_len'] > 0]['avg_silence_len'].dropna()
col_snr = df['snr'].dropna()
col_sr = df['syllable_rate'].dropna()

print("\n" + "=" * 80)
print("C-BESD DATASET STATISTICAL ANALYSIS RESULTS")
print("=" * 80)

print(f"\n{'Metric':28s} {'Mean':>10s} {'Std':>10s} {'Median':>10s}")
print("-" * 58)
metrics = [
    ('基频F0 (Hz)', col_f0.mean(), col_f0.std(), col_f0.median()),
    ('时长 (s)', df['duration'].mean(), df['duration'].std(), df['duration'].median()),
    ('静音长度 (s)', col_sil.mean(), col_sil.std(), col_sil.median()),
    ('信噪比 (dB)', col_snr.mean(), col_snr.std(), col_snr.median()),
    ('音节速率 (syll/s)', col_sr.mean(), col_sr.std(), col_sr.median()),
]
for name, mv, sv, medv in metrics:
    print(f"{name:28s} {mv:>8.2f}  {sv:>8.2f}  {medv:>8.2f}" )

print(f"\n{'='*80}")
print("FORMATTED TABLE FOR PAPER")
print(f"{'='*80}")
print(f"{'指标':20s} {'儿童平均值':>12s} {'儿童标准差':>12s}")
print("-" * 44)
for name, mv, sv, _ in metrics:
    print(f"{name:20s} {mv:>8.2f}    +/-{sv:<.2f}")

print(f"\n{'='*80}")
print("ADDITIONAL DETAILS")
print(f"{'='*80}")
print(f"Files analyzed:          {len(df)}")
print(f"F0 valid files:          {len(col_f0)}")
print(f"Avg within-file F0 std:  {df['f0_std'].dropna().mean():.2f} Hz")
print(f"Duration range:          {df['duration'].min():.2f} - {df['duration'].max():.2f} s")
print(f"Avg syllables/utt:       {df['n_syllables'].mean():.1f} +/- {df['n_syllables'].std():.1f}")
print(f"Syllable rate CV:        {col_sr.std()/col_sr.mean()*100:.1f}%")

# Save to CSV
cols_out = ['filename', 'emotion', 'duration', 'f0_mean', 'f0_std',
            'avg_silence_len', 'total_silence', 'snr', 'syllable_rate', 'n_syllables']
df_out = df[cols_out]
csv_path = "D:/大学/论文/儿童语音情绪识别/提交到团队/数据集/BESD/BESD/MY/c_besd_statistics.csv"
df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\nCSV saved: {csv_path}")
