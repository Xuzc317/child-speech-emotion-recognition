#!/usr/bin/env python3
"""plot_bubbles_v3.py — Acoustic bubble charts per dataset (test set only), v3.

Changes from v2:
  - C-BESD uses full 6 emotions (angry, disgust, fear, happy, neutral, sad)
  - FAU Aibo and IEMOCAP keep 4 emotions
  - Fixed color per emotion (not per bin level)
  - Layout: 2 rows (F0 / RMS energy) x N columns (emotions)
  - Each cell: 5 bubbles for 5 discrete bins, area proportional to proportion
  - No +80 offset on bubble sizing — pure proportion
  - Percentage annotations on bubbles >= 10%
  - Small jitter to avoid overlap
  - Color-emotion legend per figure
  - Clean white background, no grid lines
  - figsize (18, 8) for C-BESD (6 columns), (14, 7) for FAU/IEMOCAP (4 columns)
  - dpi=200

Outputs to /root/autodl-tmp/acoustic_analysis/
"""
import os, sys, re, hashlib, warnings
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import soundfile as sf
import librosa

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
FMIN, FMAX = 65, 2093
HOP_LENGTH = 320
SR = 16000
SPLIT_SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

BIN_LABELS_SHORT = ['VL', 'L', 'M', 'H', 'VH']
BIN_LABELS_FULL = ['Very Low', 'Low', 'Mid', 'High', 'Very High']
N_BINS = 5

OUTPUT_DIR = '/root/autodl-tmp/acoustic_analysis'

# ── Emotion lists per dataset ──
CBESD_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
FAU_EMOTIONS  = ['angry', 'happy', 'neutral', 'sad']
IEMOCAP_EMOTIONS = ['angry', 'happy', 'neutral', 'sad']

# ── Fixed emotion colors (per specification) ──
EMOTION_COLORS: Dict[str, Tuple[float, float, float]] = {
    'angry':   (255/255, 184/255, 122/255),  # warm orange
    'happy':   (139/255, 202/255, 139/255),  # soft green
    'neutral': (232/255, 136/255, 136/255),  # soft red
    'sad':     (196/255, 171/255, 218/255),  # purple-grey
    'disgust': (180/255, 180/255, 210/255),  # purple-grey variant
    'fear':    (160/255, 200/255, 180/255),  # grey-green variant
}

# ── Dataset paths (from env vars) ──
C_BESD_PATH = os.environ.get('SER_C_BESD_PATH',
    '/root/autodl-tmp/datasets/BESD/BESD/MY')
IEMOCAP_PATH = os.environ.get('SER_IEMOCAP_PATH',
    '/root/autodl-tmp/IEMOCAP/wavs')
FAU_AIBO_PATH = os.environ.get('SER_FAU_AIBO_PATH',
    '/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav')

# FAU label file candidates
FAU_LABEL_FILE_CANDIDATES = [
    os.path.join(os.path.dirname(FAU_AIBO_PATH), 'labels',
                 'IS2009EmotionChallenge', 'chunk_labels_5cl_corpus.txt'),
    os.path.join(os.path.dirname(os.path.dirname(FAU_AIBO_PATH)),
                 'labels', 'IS2009EmotionChallenge', 'chunk_labels_5cl_corpus.txt'),
]

# ── Label mappers (inline from label_mapper.py) ──
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

# C-BESD emotion folders
C_BESD_FOLDERS = {
    'angry': 'ANGER', 'disgust': 'DISGUST', 'fear': 'FEAR',
    'happy': 'HAPPY', 'neutral': 'NEUTRAL', 'sad': 'SAD',
}

# IEMOCAP emotion folders
IEMOCAP_FOLDERS = {'ANGER', 'HAPPY', 'NEUTRAL', 'SAD'}


# ═══════════════════════════════════════════════════════════════════
# Speaker splitting (replicates speaker_splitter.py logic)
# ═══════════════════════════════════════════════════════════════════

def _hash_speaker(speaker_id: str, seed: int = 42) -> float:
    h = hashlib.md5(f"{speaker_id}:{seed}".encode()).hexdigest()
    return int(h, 16) / (16 ** len(h))


def is_test_speaker(speaker_id: str) -> bool:
    h = _hash_speaker(speaker_id, SPLIT_SEED)
    return h >= (TRAIN_RATIO + VAL_RATIO)


# ═══════════════════════════════════════════════════════════════════
# Speaker ID extraction
# ═══════════════════════════════════════════════════════════════════

def extract_speaker_cbesd(filename: str) -> str:
    m = re.match(r'^(\d+)', filename)
    if m:
        return f'C{m.group(1).zfill(2)}'
    basename = os.path.splitext(filename)[0].lower()
    return f'C{basename[:2]}'


def extract_speaker_iemocap(filename: str) -> str:
    return filename[:6]


def extract_speaker_fau(filename: str) -> str:
    parts = filename.split('_')
    return f"{parts[0]}_{parts[1]}"


def iemocap_gender(speaker_id: str) -> str:
    return speaker_id[-1].upper()


# ═══════════════════════════════════════════════════════════════════
# Data collection per dataset (test set only)
# ═══════════════════════════════════════════════════════════════════

def collect_cbesd(root: str) -> List[Tuple[str, str, str]]:
    """Return [(wav_path, unified_label, speaker_id), ...] for test speakers, 6-class."""
    entries = []
    for emo_label, folder in C_BESD_FOLDERS.items():
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith('.wav') or 'copy' in fname.lower():
                continue
            sid = extract_speaker_cbesd(fname)
            if not is_test_speaker(sid):
                continue
            entries.append((os.path.join(folder_path, fname), emo_label, sid))
    return entries


def collect_iemocap(root: str) -> List[Tuple[str, str, str]]:
    """Return [(wav_path, unified_label, speaker_id), ...] for test speakers, 4-class."""
    entries = []
    for folder in IEMOCAP_FOLDERS:
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith('.wav'):
                continue
            stem = os.path.splitext(fname)[0]
            parts = stem.split('_')
            raw_label = parts[-1] if len(parts) >= 4 else parts[-1]
            raw_label = raw_label.strip().lower()

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
    label_file = None
    for cand in FAU_LABEL_FILE_CANDIDATES:
        if os.path.isfile(cand):
            label_file = cand
            break
    if label_file is None:
        print(f"  ERROR: FAU label file not found. Tried: {FAU_LABEL_FILE_CANDIDATES}")
        return []

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


# ═══════════════════════════════════════════════════════════════════
# Acoustic feature extraction
# ═══════════════════════════════════════════════════════════════════

def extract_features(wav_path: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract mean F0 (voiced frames) and mean RMS energy.

    Returns (mean_f0, mean_rms) or (None, None) on failure.
    """
    try:
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        if len(y) < HOP_LENGTH:
            return None, None

        f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=SR, hop_length=HOP_LENGTH)
        voiced = f0[f0 > 0]
        mean_f0 = float(np.mean(voiced)) if len(voiced) > 0 else None

        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        mean_rms = float(np.mean(rms))

        return mean_f0, mean_rms
    except Exception as e:
        print(f"    WARN: {os.path.basename(wav_path)}: {e}", file=sys.stderr)
        return None, None


# ═══════════════════════════════════════════════════════════════════
# Discretization
# ═══════════════════════════════════════════════════════════════════

def compute_quantile_bins(values: np.ndarray) -> np.ndarray:
    """Compute 5 boundaries (q0.2, q0.4, q0.6, q0.8) from all values."""
    return np.quantile(values, [0.2, 0.4, 0.6, 0.8])


def assign_bin(value: float, boundaries: np.ndarray) -> int:
    """Map value to bin index 0-4 based on quantile boundaries."""
    for i, b in enumerate(boundaries):
        if value < b:
            return i
    return len(boundaries)


# ═══════════════════════════════════════════════════════════════════
# Frequency matrix building
# ═══════════════════════════════════════════════════════════════════

def build_freq_matrix(
    features: Dict[str, List[Tuple[float, float]]],
    f0_boundaries: np.ndarray,
    rms_boundaries: np.ndarray,
    emotions: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (n_emotions x 5) frequency matrices for F0 and RMS.

    Returns (f0_matrix, rms_matrix) where values are proportions within each emotion.
    """
    n = len(emotions)
    f0_matrix = np.zeros((n, N_BINS))
    rms_matrix = np.zeros((n, N_BINS))

    for ei, emo in enumerate(emotions):
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
        if f0_matrix[ei].sum() > 0:
            f0_matrix[ei] /= f0_matrix[ei].sum()
        if rms_matrix[ei].sum() > 0:
            rms_matrix[ei] /= rms_matrix[ei].sum()

    return f0_matrix, rms_matrix


def build_freq_matrix_gendered(
    features: Dict[str, List[Tuple[float, float, str]]],
    f0_boundaries_f: np.ndarray,
    f0_boundaries_m: np.ndarray,
    rms_boundaries: np.ndarray,
    emotions: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """For IEMOCAP: gender-specific F0 boundaries, shared RMS boundaries."""
    n = len(emotions)
    f0_matrix = np.zeros((n, N_BINS))
    rms_matrix = np.zeros((n, N_BINS))

    for ei, emo in enumerate(emotions):
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


# ═══════════════════════════════════════════════════════════════════
# Plotting (v3 layout)
# ═══════════════════════════════════════════════════════════════════

def plot_dataset_figure_v3(
    f0_matrix: np.ndarray,
    rms_matrix: np.ndarray,
    emotions: List[str],
    dataset_label: str,
    n_samples: int,
    output_path: str,
):
    """Plot a single dataset figure with v3 layout.

    2 rows (F0, RMS) x N columns (emotions).
    Each cell: 5 bubbles for VL/L/M/H/VH, area proportional to proportion.
    Bubbles colored by emotion (same color per column).
    """
    n_cols = len(emotions)
    is_cbesd = (n_cols == 6)
    figsize = (18, 8) if is_cbesd else (14, 7)

    fig, axes = plt.subplots(2, n_cols, figsize=figsize, facecolor='white')

    # Ensure axes is always 2D
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    row_names = ['F0 (Hz)', 'RMS Energy']
    matrices = [f0_matrix, rms_matrix]

    # Global max for consistent bubble scaling across all cells
    global_max = max(f0_matrix.max(), rms_matrix.max(), 1e-6)
    MAX_BUBBLE = 1400

    for row in range(2):
        matrix = matrices[row]
        for col in range(n_cols):
            ax = axes[row, col]
            emo = emotions[col]
            proportions = matrix[col]  # shape (5,) — one proportion per bin
            color = EMOTION_COLORS[emo]

            # Deterministic jitter per cell
            rng = np.random.RandomState(42 + row * 10 + col)
            x_jitter = rng.uniform(-0.10, 0.10, N_BINS)
            y_jitter = rng.uniform(-0.06, 0.06, N_BINS)

            # Draw bubbles
            for bi in range(N_BINS):
                prop = proportions[bi]
                if prop < 0.001:
                    continue
                size = MAX_BUBBLE * (prop / global_max)
                size = max(size, 20)  # minimum visible size

                ax.scatter(
                    x_jitter[bi], bi + y_jitter[bi],
                    s=size, c=[color], alpha=0.85,
                    edgecolors='white', linewidths=0.8, zorder=3,
                )

                # Percentage annotation (>= 10%)
                if prop >= 0.10:
                    ax.annotate(
                        f'{prop:.0%}',
                        (x_jitter[bi], bi + y_jitter[bi]),
                        ha='center', va='center',
                        fontsize=6, fontweight='bold',
                        color='black', alpha=0.85,
                    )

            # ── Axis formatting ──
            ax.set_xlim(-0.45, 0.45)
            ax.set_ylim(-0.5, N_BINS - 0.5)
            ax.set_xticks([])

            # Y-axis: bin labels on first column only
            if col == 0:
                ax.set_yticks(range(N_BINS))
                ax.set_yticklabels(BIN_LABELS_SHORT, fontsize=7)
                ax.tick_params(left=True, labelleft=True, labelsize=7)
            else:
                ax.set_yticks([])
                ax.tick_params(left=False, labelleft=False)

            # Column titles (emotion) — top row only
            if row == 0:
                emo_display = 'Angry' if emo == 'angry' else emo.capitalize()
                ax.set_title(emo_display, fontsize=12, fontweight='bold', pad=8)

            # Row labels — leftmost column only
            if col == 0:
                ax.set_ylabel(row_names[row], fontsize=11, fontweight='bold', labelpad=8)

            # Clean styling
            ax.set_facecolor('white')
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#e0e0e0')
                spine.set_linewidth(0.5)

    # ── Figure title ──
    class_label = '6-class' if is_cbesd else '4-class'
    fig.suptitle(
        f'{dataset_label} ({class_label}, n={n_samples})',
        fontsize=15, fontweight='bold', y=1.01,
    )

    # ── Color-emotion legend ──
    legend_elements = [
        Patch(facecolor=EMOTION_COLORS[e], edgecolor='white', linewidth=0.5,
              label=e.capitalize())
        for e in emotions
    ]
    fig.legend(
        handles=legend_elements, loc='lower center',
        ncol=n_cols, fontsize=10, frameon=False,
        bbox_to_anchor=(0.5, -0.03), handlelength=1.5, handleheight=1.5,
    )

    plt.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.10,
                        wspace=0.12, hspace=0.20)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def combine_vertically(input_paths: List[str], output_path: str):
    """Stack PNGs vertically into one combined figure."""
    from PIL import Image

    images = [Image.open(p) for p in input_paths if os.path.isfile(p)]
    if not images:
        print("  No images to combine")
        return

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


# ═══════════════════════════════════════════════════════════════════
# Dataset processing pipeline
# ═══════════════════════════════════════════════════════════════════

def process_dataset(name: str, collect_fn, root: str, emotions: List[str],
                    gendered: bool = False):
    """Collect test data, extract features, compute bins, return matrices."""
    print(f"\n{'='*60}")
    print(f"Processing {name} ({len(emotions)}-class, gendered={gendered})")
    print(f"  Root: {root}")
    print(f"  Collecting test-set files...")

    entries = collect_fn(root)
    print(f"  Collected {len(entries)} test-set files")

    if len(entries) == 0:
        print(f"  SKIP: No test samples for {name}")
        return None, None, 0

    # Count per emotion
    emo_counts = defaultdict(int)
    for _, emo, _ in entries:
        emo_counts[emo] += 1
    print(f"  Per emotion: {dict(emo_counts)}")
    n_samples = sum(emo_counts.values())

    # Extract features
    print(f"  Extracting acoustic features...")
    if gendered:
        features: Dict = {e: [] for e in emotions}
    else:
        features: Dict = {e: [] for e in emotions}

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
        # Gender-specific F0 bins (IEMOCAP)
        f0_f_vals = []
        f0_m_vals = []
        rms_vals = []
        for emo in emotions:
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
            features, f0_bounds_f, f0_bounds_m, rms_bounds, emotions,
        )
    else:
        f0_vals = []
        rms_vals = []
        for emo in emotions:
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

        f0_matrix, rms_matrix = build_freq_matrix(features, f0_bounds, rms_bounds, emotions)

    return f0_matrix, rms_matrix, n_samples


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}

    # 1. C-BESD (6-class, non-gendered)
    f0_m, rms_m, n = process_dataset(
        'C-BESD', collect_cbesd, C_BESD_PATH, CBESD_EMOTIONS, gendered=False,
    )
    if f0_m is not None:
        results['cbesd'] = (f0_m, rms_m, n, CBESD_EMOTIONS)

    # 2. FAU Aibo (4-class, non-gendered)
    f0_m, rms_m, n = process_dataset(
        'FAU Aibo', collect_fau_aibo, FAU_AIBO_PATH, FAU_EMOTIONS, gendered=False,
    )
    if f0_m is not None:
        results['fauaibo'] = (f0_m, rms_m, n, FAU_EMOTIONS)

    # 3. IEMOCAP (4-class, gendered F0 bins)
    f0_m, rms_m, n = process_dataset(
        'IEMOCAP', collect_iemocap, IEMOCAP_PATH, IEMOCAP_EMOTIONS, gendered=True,
    )
    if f0_m is not None:
        results['iemocap'] = (f0_m, rms_m, n, IEMOCAP_EMOTIONS)

    # ── Plotting ──
    print(f"\n{'='*60}")
    print("Generating v3 figures...")

    name_map = {'cbesd': 'C-BESD', 'fauaibo': 'FAU Aibo', 'iemocap': 'IEMOCAP'}
    output_paths = []

    for key in ['cbesd', 'fauaibo', 'iemocap']:
        if key in results:
            f0_m, rms_m, n, emotions = results[key]
            path = os.path.join(OUTPUT_DIR, f'fig_bubbles_v3_{key}.png')
            plot_dataset_figure_v3(
                f0_m, rms_m, emotions, name_map[key], n, path,
            )
            output_paths.append(path)

    # Combined
    if len(output_paths) >= 2:
        combined_path = os.path.join(OUTPUT_DIR, 'fig_bubbles_v3_combined.png')
        combine_vertically(output_paths, combined_path)

    print(f"\nDone! Output directory: {OUTPUT_DIR}")
    for p in sorted(os.listdir(OUTPUT_DIR)):
        if p.endswith('.png'):
            fpath = os.path.join(OUTPUT_DIR, p)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {p} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
