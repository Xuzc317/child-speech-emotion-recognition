#!/usr/bin/env python3
"""plot_bubbles_v2.py — Acoustic bubble charts per dataset (test set only).

Reference layout: Tsangko et al. (2026) Fig.1
  Rows = acoustic concepts (F0, RMS energy)
  Columns = emotion classes (angry, happy, neutral, sad)
  Bubble area = proportion of the emotion's samples in that bin.

Outputs to /root/autodl-tmp/acoustic_analysis/
"""
import os, sys, re, hashlib, warnings
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import soundfile as sf
import librosa

warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────
FMIN, FMAX = 65, 2093
HOP_LENGTH = 320
SR = 16000
SPLIT_SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

EMOTIONS = ['angry', 'happy', 'neutral', 'sad']
BIN_LABELS_SHORT = ['VL', 'L', 'M', 'H', 'VH']
BIN_LABELS_FULL = ['Very Low', 'Low', 'Mid', 'High', 'Very High']
N_BINS = 5

OUTPUT_DIR = '/root/autodl-tmp/acoustic_analysis'

# ── Dataset paths ──────────────────────────────────────────────
C_BESD_PATH = os.environ.get('SER_C_BESD_PATH',
    '/root/autodl-tmp/datasets/BESD/BESD/MY')
IEMOCAP_PATH = os.environ.get('SER_IEMOCAP_PATH',
    '/root/autodl-tmp/IEMOCAP/wavs')
FAU_AIBO_PATH = os.environ.get('SER_FAU_AIBO_PATH',
    '/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav')

FAU_LABEL_FILE_CANDIDATES = [
    os.path.join(os.path.dirname(FAU_AIBO_PATH), 'labels',
                 'IS2009EmotionChallenge', 'chunk_labels_5cl_corpus.txt'),
    os.path.join(os.path.dirname(os.path.dirname(FAU_AIBO_PATH)),
                 'labels', 'IS2009EmotionChallenge', 'chunk_labels_5cl_corpus.txt'),
]

# ── Label mappers (inline from label_mapper.py) ─────────────────
IEMOCAP_MAP = {
    'ang': 'angry', 'anger': 'angry', 'angry': 'angry',
    'frustrated': 'angry',
    'hap': 'happy', 'happy': 'happy', 'exc': 'happy', 'excited': 'happy',
    'neu': 'neutral', 'neutral': 'neutral',
    'sad': 'sad',
}
IEMOCAP_DISCARD = {'fea', 'fear', 'dis', 'disgust', 'sur', 'surprise', 'xxx', 'oth'}

FAU_AIBO_MAP = {
    'a': 'angry', 'e': 'angry', 'p': 'happy', 'n': 'neutral', 'r': 'sad',
}

C_BESD_MAP_4CL = {
    'angry': 'angry', 'anger': 'angry',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
    'disgust': None, 'disguist': None, 'fear': None,
}

# IEMOCAP emotion folders to scan
IEMOCAP_FOLDERS = {'ANGER', 'HAPPY', 'NEUTRAL', 'SAD'}

# C-BESD emotion folders
C_BESD_FOLDERS = {
    'angry': 'ANGER', 'disgust': 'DISGUST', 'fear': 'FEAR',
    'happy': 'HAPPY', 'neutral': 'NEUTRAL', 'sad': 'SAD',
}


# ═══════════════════════════════════════════════════════════════
# Speaker splitting (replicates speaker_splitter.py logic)
# ═══════════════════════════════════════════════════════════════

def _hash_speaker(speaker_id: str, seed: int = 42) -> float:
    h = hashlib.md5(f"{speaker_id}:{seed}".encode()).hexdigest()
    return int(h, 16) / (16 ** len(h))


def is_test_speaker(speaker_id: str) -> bool:
    """True if this speaker belongs to the test split (hash >= 0.85)."""
    h = _hash_speaker(speaker_id, SPLIT_SEED)
    return h >= (TRAIN_RATIO + VAL_RATIO)


# ═══════════════════════════════════════════════════════════════
# Speaker ID extraction
# ═══════════════════════════════════════════════════════════════

def extract_speaker_cbesd(filename: str) -> str:
    """'1.EF_12 Angry_1.wav' → 'C01'"""
    m = re.match(r'^(\d+)', filename)
    if m:
        return f'C{m.group(1).zfill(2)}'
    basename = os.path.splitext(filename)[0].lower()
    return f'C{basename[:2]}'


def extract_speaker_iemocap(filename: str) -> str:
    """'Ses01F_impro01_F006_frustrated.wav' → 'Ses01F'"""
    return filename[:6]


def extract_speaker_fau(filename: str) -> str:
    """'Mont_01_000_00.wav' → 'Mont_01'"""
    parts = filename.split('_')
    return f"{parts[0]}_{parts[1]}"


def iemocap_gender(speaker_id: str) -> str:
    """'Ses01F' → 'F', 'Ses05M' → 'M'"""
    return speaker_id[-1].upper()


# ═══════════════════════════════════════════════════════════════
# Data collection per dataset (test set only)
# ═══════════════════════════════════════════════════════════════

def collect_cbesd(root: str) -> List[Tuple[str, str, str]]:
    """Return [(wav_path, unified_label, speaker_id), ...] for test speakers, 4-class only."""
    entries = []
    for emo_label, folder in C_BESD_FOLDERS.items():
        # 4-class filter: skip disgust/fear
        mapped = C_BESD_MAP_4CL.get(emo_label)
        if mapped is None:
            continue
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith('.wav') or 'copy' in fname.lower():
                continue
            sid = extract_speaker_cbesd(fname)
            if not is_test_speaker(sid):
                continue
            entries.append((os.path.join(folder_path, fname), mapped, sid))
    return entries


def collect_iemocap(root: str) -> List[Tuple[str, str, str]]:
    """Return [(wav_path, unified_label, speaker_id), ...] for test speakers, 4-class only."""
    entries = []
    for folder in IEMOCAP_FOLDERS:
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith('.wav'):
                continue
            # Extract raw label from filename
            stem = os.path.splitext(fname)[0]
            parts = stem.split('_')
            raw_label = parts[-1] if len(parts) >= 4 else parts[-1]
            raw_label = raw_label.strip().lower()

            # Map to unified label
            if raw_label in IEMOCAP_DISCARD:
                continue
            unified = IEMOCAP_MAP.get(raw_label)
            if unified is None:
                continue

            sid = extract_speaker_iemocap(fname)
            if not is_test_speaker(sid):
                continue
            entries.append((os.path.join(folder_path, fname), unified, sid))
    return entries


def collect_fau_aibo(root: str) -> List[Tuple[str, str, str]]:
    """Return [(wav_path, unified_label, speaker_id), ...] for test speakers."""
    # Find label file
    label_file = None
    for cand in FAU_LABEL_FILE_CANDIDATES:
        if os.path.isfile(cand):
            label_file = cand
            break
    if label_file is None:
        print(f"  ERROR: FAU label file not found. Tried: {FAU_LABEL_FILE_CANDIDATES}")
        return []

    # Parse label file
    file_to_label: Dict[str, str] = {}
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                file_to_label[parts[0] + '.wav'] = parts[1].strip().lower()

    entries = []
    for fname in os.listdir(root):
        if not fname.endswith('.wav'):
            continue
        raw_label = file_to_label.get(fname)
        if raw_label is None:
            continue
        unified = FAU_AIBO_MAP.get(raw_label)
        if unified is None:
            continue
        sid = extract_speaker_fau(fname)
        if not is_test_speaker(sid):
            continue
        entries.append((os.path.join(root, fname), unified, sid))
    return entries


# ═══════════════════════════════════════════════════════════════
# Acoustic feature extraction
# ═══════════════════════════════════════════════════════════════

def extract_features(wav_path: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract mean F0 (voiced frames) and mean RMS energy.

    Returns (mean_f0, mean_rms) or (None, None) on failure.
    """
    try:
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        if len(y) < HOP_LENGTH:
            return None, None

        # F0 via YIN
        f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=SR, hop_length=HOP_LENGTH)
        voiced = f0[f0 > 0]
        mean_f0 = float(np.mean(voiced)) if len(voiced) > 0 else None

        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        mean_rms = float(np.mean(rms))

        return mean_f0, mean_rms
    except Exception as e:
        print(f"    WARN: {os.path.basename(wav_path)}: {e}", file=sys.stderr)
        return None, None


# ═══════════════════════════════════════════════════════════════
# Discretization
# ═══════════════════════════════════════════════════════════════

def compute_quantile_bins(values: np.ndarray) -> np.ndarray:
    """Compute 5 boundaries (q0.2, q0.4, q0.6, q0.8) from all values."""
    return np.quantile(values, [0.2, 0.4, 0.6, 0.8])


def assign_bin(value: float, boundaries: np.ndarray) -> int:
    """Map value to bin index 0-4 based on quantile boundaries."""
    for i, b in enumerate(boundaries):
        if value < b:
            return i
    return len(boundaries)  # >= q0.8 → last bin (index 4)


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

COLORS_5 = ['#a6cee3', '#6fb3d9', '#3794bf', '#1f78b4', '#08519c']


def build_freq_matrix(
    features: Dict[str, List[Tuple[float, float]]],  # emotion → [(f0, rms)]
    f0_boundaries: np.ndarray,
    rms_boundaries: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build 2 × 4 × 5 frequency matrices for F0 and RMS.

    Returns (f0_matrix[emotion_idx, bin_idx], rms_matrix[emotion_idx, bin_idx])
    where values are proportions within each emotion.
    """
    f0_matrix = np.zeros((4, 5))
    rms_matrix = np.zeros((4, 5))

    for ei, emo in enumerate(EMOTIONS):
        samples = features.get(emo, [])
        total = len(samples)
        if total == 0:
            continue
        for f0_val, rms_val in samples:
            if f0_val is not None:
                b = assign_bin(f0_val, f0_boundaries)
                f0_matrix[ei, b] += 1
            if rms_val is not None:
                b = assign_bin(rms_val, rms_boundaries)
                rms_matrix[ei, b] += 1
        # Normalize to proportions
        if f0_matrix[ei].sum() > 0:
            f0_matrix[ei] /= f0_matrix[ei].sum()
        if rms_matrix[ei].sum() > 0:
            rms_matrix[ei] /= rms_matrix[ei].sum()

    return f0_matrix, rms_matrix


def build_freq_matrix_gendered(
    features: Dict[str, List[Tuple[float, float, str]]],  # + gender
    f0_boundaries_f: np.ndarray,
    f0_boundaries_m: np.ndarray,
    rms_boundaries: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """For IEMOCAP: gender-specific F0 boundaries, shared RMS boundaries."""
    f0_matrix = np.zeros((4, 5))
    rms_matrix = np.zeros((4, 5))

    for ei, emo in enumerate(EMOTIONS):
        samples = features.get(emo, [])
        total = len(samples)
        if total == 0:
            continue
        for f0_val, rms_val, gender in samples:
            if f0_val is not None:
                boundaries = f0_boundaries_f if gender == 'F' else f0_boundaries_m
                b = assign_bin(f0_val, boundaries)
                f0_matrix[ei, b] += 1
            if rms_val is not None:
                b = assign_bin(rms_val, rms_boundaries)
                rms_matrix[ei, b] += 1
        if f0_matrix[ei].sum() > 0:
            f0_matrix[ei] /= f0_matrix[ei].sum()
        if rms_matrix[ei].sum() > 0:
            rms_matrix[ei] /= rms_matrix[ei].sum()

    return f0_matrix, rms_matrix


def plot_dataset_figure(
    f0_matrix: np.ndarray,
    rms_matrix: np.ndarray,
    dataset_label: str,
    n_samples: int,
    output_path: str,
    suptitle: str = "Acoustic Feature Distributions by Emotion Class (Test Set)",
):
    """Plot a single dataset figure with Tsangko et al. layout.

    2 rows (F0, RMS) × 4 columns (angry, happy, neutral, sad).
    Each cell: 5 bubbles for VL/L/M/H/VH, area ∝ proportion.
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), facecolor='white')
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.01)

    row_names = ['F0 (Hz)', 'RMS Energy']
    col_names = ['Angry', 'Happy', 'Neutral', 'Sad']

    matrices = [f0_matrix, rms_matrix]

    max_area = max(
        max(f0_matrix.max(), rms_matrix.max()),
        0.01  # floor to avoid zero-division
    )

    for row in range(2):
        for col in range(4):
            ax = axes[row, col]
            proportions = matrices[row][col]

            # Scaled bubble sizes
            sizes = proportions * 800 / max_area if max_area > 0 else np.zeros(5)
            sizes = np.maximum(sizes, 5)  # min visible size

            x_positions = np.arange(5) + np.random.uniform(-0.15, 0.15, 5)

            ax.scatter(
                x_positions, np.zeros(5) + np.random.uniform(-0.08, 0.08, 5),
                s=sizes, c=COLORS_5, alpha=0.85, edgecolors='grey',
                linewidths=0.5, zorder=3,
            )

            # Formatting
            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xticks(range(5))
            ax.set_xticklabels(BIN_LABELS_FULL, fontsize=7, rotation=30)
            ax.tick_params(left=False, labelleft=False)
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(False)

            # Column titles (emotion) — top row only
            if row == 0:
                ax.set_title(col_names[col], fontsize=11, fontweight='bold', pad=8)

            # Row labels — leftmost column only
            if col == 0:
                ax.set_ylabel(row_names[row], fontsize=10, fontweight='bold')

            # Annotate proportions on bubbles
            for i, p in enumerate(proportions):
                if p > 0.02:
                    ax.annotate(
                        f'{p:.1%}', (x_positions[i], 0),
                        textcoords="offset points", xytext=(0, 6),
                        ha='center', fontsize=6, color='black', alpha=0.8,
                    )

    # Dataset title
    fig.text(0.5, 0.01, f'{dataset_label} (n={n_samples})',
             ha='center', fontsize=12, fontweight='bold')

    # Legend for bin colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor='grey', linewidth=0.5,
                             label=BIN_LABELS_FULL[i]) for i, c in enumerate(COLORS_5)]
    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, -0.04),
               ncol=5, fontsize=8, frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def combine_vertically(input_paths: List[str], output_path: str):
    """Stack 3 PNGs vertically into one combined figure."""
    from PIL import Image

    images = [Image.open(p) for p in input_paths]
    widths = [im.width for im in images]
    max_width = max(widths)
    total_height = sum(im.height for im in images)

    combined = Image.new('RGB', (max_width, total_height), 'white')
    y_offset = 0
    for im in images:
        x_offset = (max_width - im.width) // 2
        combined.paste(im, (x_offset, y_offset))
        y_offset += im.height

    combined.save(output_path)
    print(f"  Saved combined: {output_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def process_dataset(name: str, collect_fn, root: str, gendered: bool = False):
    """Collect test data, extract features, compute bins, return (f0_matrix, rms_matrix, n)."""
    print(f"\n{'='*60}")
    print(f"Processing {name} (gendered={gendered})")
    print(f"  Root: {root}")
    print(f"  Collecting test-set files...")

    entries = collect_fn(root)
    print(f"  Collected {len(entries)} test-set files")

    if len(entries) == 0:
        print(f"  SKIP: No test samples for {name}")
        return None, None, None, 0

    # Count per emotion
    emo_counts = defaultdict(int)
    for _, emo, _ in entries:
        emo_counts[emo] += 1
    print(f"  Per emotion: {dict(emo_counts)}")
    n_samples = sum(emo_counts.values())

    # Extract features
    print(f"  Extracting acoustic features...")
    if gendered:
        # IEMOCAP: store (f0, rms, gender)
        features: Dict[str, List[Tuple]] = {e: [] for e in EMOTIONS}
    else:
        features: Dict[str, List[Tuple]] = {e: [] for e in EMOTIONS}

    missing_f0, missing_rms = 0, 0
    for i, (wav_path, emo, sid) in enumerate(entries):
        if (i + 1) % 500 == 0:
            print(f"    ... {i+1}/{len(entries)}")
        f0_val, rms_val = extract_features(wav_path)
        if f0_val is None:
            missing_f0 += 1
        if rms_val is None:
            missing_rms += 1
        if gendered:
            gender = iemocap_gender(sid)
            features[emo].append((f0_val, rms_val, gender))
        else:
            features[emo].append((f0_val, rms_val))

    print(f"  Feature extraction done. Missing F0: {missing_f0}, Missing RMS: {missing_rms}")

    # Compute bin boundaries
    if gendered:
        # Gender-specific F0 bins
        f0_f_vals = []
        f0_m_vals = []
        rms_vals = []
        for emo in EMOTIONS:
            for f0_val, rms_val, gender in features[emo]:
                if f0_val is not None:
                    if gender == 'F':
                        f0_f_vals.append(f0_val)
                    else:
                        f0_m_vals.append(f0_val)
                if rms_val is not None:
                    rms_vals.append(rms_val)

        f0_f_arr = np.array(f0_f_vals)
        f0_m_arr = np.array(f0_m_vals)
        rms_arr = np.array(rms_vals)

        f0_bounds_f = compute_quantile_bins(f0_f_arr) if len(f0_f_arr) > 0 else np.zeros(4)
        f0_bounds_m = compute_quantile_bins(f0_m_arr) if len(f0_m_arr) > 0 else np.zeros(4)
        rms_bounds = compute_quantile_bins(rms_arr) if len(rms_arr) > 0 else np.zeros(4)

        print(f"  F0 boundaries (F): {np.round(f0_bounds_f, 1)}")
        print(f"  F0 boundaries (M): {np.round(f0_bounds_m, 1)}")
        print(f"  RMS boundaries:    {np.round(rms_bounds, 4)}")

        f0_matrix, rms_matrix = build_freq_matrix_gendered(
            features, f0_bounds_f, f0_bounds_m, rms_bounds,
        )
    else:
        f0_vals = []
        rms_vals = []
        for emo in EMOTIONS:
            for f0_val, rms_val in features[emo]:
                if f0_val is not None:
                    f0_vals.append(f0_val)
                if rms_val is not None:
                    rms_vals.append(rms_val)

        f0_arr = np.array(f0_vals)
        rms_arr = np.array(rms_vals)
        f0_bounds = compute_quantile_bins(f0_arr) if len(f0_arr) > 0 else np.zeros(4)
        rms_bounds = compute_quantile_bins(rms_arr) if len(rms_arr) > 0 else np.zeros(4)

        print(f"  F0 boundaries:  {np.round(f0_bounds, 1)}")
        print(f"  RMS boundaries: {np.round(rms_bounds, 4)}")

        f0_matrix, rms_matrix = build_freq_matrix(features, f0_bounds, rms_bounds)

    return f0_matrix, rms_matrix, name, n_samples


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}

    # 1. C-BESD (4-class only, non-gendered)
    f0_m, rms_m, name, n = process_dataset(
        'C-BESD', collect_cbesd, C_BESD_PATH, gendered=False,
    )
    if f0_m is not None:
        results['cbesd'] = (f0_m, rms_m, n)

    # 2. FAU Aibo (non-gendered)
    f0_m, rms_m, name, n = process_dataset(
        'FAU Aibo', collect_fau_aibo, FAU_AIBO_PATH, gendered=False,
    )
    if f0_m is not None:
        results['fauaibo'] = (f0_m, rms_m, n)

    # 3. IEMOCAP (gendered F0 bins)
    f0_m, rms_m, name, n = process_dataset(
        'IEMOCAP', collect_iemocap, IEMOCAP_PATH, gendered=True,
    )
    if f0_m is not None:
        results['iemocap'] = (f0_m, rms_m, n)

    # ── Plotting ──
    print(f"\n{'='*60}")
    print("Generating figures...")

    suptitle = "Acoustic Feature Distributions by Emotion Class (Test Set)"

    output_paths = []

    if 'cbesd' in results:
        f0_m, rms_m, n = results['cbesd']
        path = os.path.join(OUTPUT_DIR, 'fig_bubbles_v2_cbesd.png')
        plot_dataset_figure(f0_m, rms_m, 'C-BESD', n, path, suptitle)
        output_paths.append(path)

    if 'fauaibo' in results:
        f0_m, rms_m, n = results['fauaibo']
        path = os.path.join(OUTPUT_DIR, 'fig_bubbles_v2_fauaibo.png')
        plot_dataset_figure(f0_m, rms_m, 'FAU Aibo', n, path, suptitle)
        output_paths.append(path)

    if 'iemocap' in results:
        f0_m, rms_m, n = results['iemocap']
        path = os.path.join(OUTPUT_DIR, 'fig_bubbles_v2_iemocap.png')
        plot_dataset_figure(f0_m, rms_m, 'IEMOCAP', n, path, suptitle)
        output_paths.append(path)

    # Combined
    if len(output_paths) >= 2:
        combined_path = os.path.join(OUTPUT_DIR, 'fig_bubbles_v2_combined.png')
        combine_vertically(output_paths, combined_path)

    print(f"\nDone! Output directory: {OUTPUT_DIR}")
    for p in os.listdir(OUTPUT_DIR):
        if p.endswith('.png'):
            fpath = os.path.join(OUTPUT_DIR, p)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {p} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
