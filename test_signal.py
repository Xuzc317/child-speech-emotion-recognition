import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, butter, sosfilt

base = "D:/大学/论文/儿童语音情绪识别/提交到团队/数据集/BESD/BESD/MY"

y, sr = librosa.load(f"{base}/ANGER/1.EF_12 Angry_1.wav", sr=16000)
dur = len(y) / sr
hop = 512

print(f"Duration: {dur:.2f}s")
print(f"Samples: {len(y)}")
print(f"RMS: mean={np.std(y):.5f}, max={np.max(np.abs(y)):.5f}")
print(f"DC offset: {np.mean(y):.6f}")

# Check if signal is clipped
print(f"Clipped samples: {np.sum(np.abs(y) >= 1.0)}/{len(y)}")

# Spectrogram analysis
S = np.abs(librosa.stft(y))
freqs = librosa.fft_frequencies(sr=sr)
S_db = librosa.amplitude_to_db(S, ref=np.max)

# Energy distribution across frequency bands
bands = {
    'low (0-300Hz)': (0, 300),
    'speech (300-3000Hz)': (300, 3000),
    'high (3000-8000Hz)': (3000, 8000)
}
for name, (fmin, fmax) in bands.items():
    mask = (freqs >= fmin) & (freqs < fmax)
    energy = np.sum(S[mask]**2)
    total = np.sum(S**2)
    print(f"Energy in {name}: {energy/total*100:.1f}%")

# Try band-limited RMS for syllable detection
def bandpass_filter(y, sr, low=300, high=3000, order=4):
    sos = butter(order, [low, high], btype='band', fs=sr, output='sos')
    return sosfilt(sos, y)

y_bp = bandpass_filter(y, sr)

# Compare full-band vs band-limited RMS peaks
print("\n--- Syllable detection comparison ---")
for y_in, label in [(y, 'full-band'), (y_bp, 'band-limited (300-3000Hz)')]:
    rms = librosa.feature.rms(y=y_in, hop_length=hop)[0]
    for sigma in [1, 2]:
        rms_s = gaussian_filter1d(rms, sigma=sigma)
        rms_n = rms_s / (np.max(rms_s) + 1e-10)
        for h in [0.05, 0.08]:
            p, _ = find_peaks(rms_n, height=h, distance=int(0.10*sr/hop))
            print(f"  {label}, sigma={sigma}, h={h}: {len(p)} peaks ({len(p)/dur:.2f}/s)")

# Also try: compute onset_strength and use Otsu thresholding
# Or: use multi-band energy peaks
print("\n--- Mel-band energy peaks ---")
mel_S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, hop_length=hop)
# Sum mel bands that correspond to speech
speech_bands = mel_S[8:32]  # ~300-3000Hz in mel scale
speech_energy = np.sum(speech_bands, axis=0)
se_n = speech_energy / (np.max(speech_energy) + 1e-10)
se_s = gaussian_filter1d(se_n, sigma=2)
p, _ = find_peaks(se_s, height=0.08, distance=int(0.10*sr/hop))
print(f"  Mel-band energy: {len(p)} peaks ({len(p)/dur:.2f}/s)")
