"""Clean preprocessing: split WAV files FIRST, then augment only training set.

This fixes the data leakage bug where augmented versions of the same WAV
could appear in both train and test sets.

Order: WAV files -> stratified split by file -> augment train only -> save
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import get_features, extract_features

# ---- Config ----
WAV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "数据集", "BESD", "BESD", "ENGLISH")
OUT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_RATIO = 0.7
SEED = 42
CLASS_NAMES = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD']

TRAIN_DATA = os.path.join(OUT_DIR, "train_data.npy")
TRAIN_LABEL = os.path.join(OUT_DIR, "train_labels.npy")
TEST_DATA = os.path.join(OUT_DIR, "test_data.npy")
TEST_LABEL = os.path.join(OUT_DIR, "test_labels.npy")


def collect_wav_files():
    """Collect WAV file paths grouped by class. Returns dict[class_name] = [paths]"""
    files_by_class = {}
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(WAV_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"  WARNING: directory not found: {cls_dir}")
            files_by_class[cls] = []
            continue
        wavs = sorted([os.path.join(cls_dir, f)
                       for f in os.listdir(cls_dir) if f.endswith('.wav')])
        files_by_class[cls] = wavs
        print(f"  {cls}: {len(wavs)} files")
    return files_by_class


def stratified_split(files_by_class):
    """Split WAV file paths per class (stratified). Returns train_paths, test_paths."""
    rng = np.random.RandomState(SEED)
    train_paths = []
    test_paths = []
    for cls in CLASS_NAMES:
        paths = files_by_class.get(cls, [])
        n = len(paths)
        n_train = int(n * TRAIN_RATIO)
        perm = rng.permutation(n)
        for idx in perm[:n_train]:
            train_paths.append((paths[idx], CLASS_NAMES.index(cls)))
        for idx in perm[n_train:]:
            test_paths.append((paths[idx], CLASS_NAMES.index(cls)))
    return train_paths, test_paths


def extract_train_features(train_paths):
    """For training WAVs: extract all 3 augmentations (original, noise, stretch+pitch)."""
    features = []
    labels = []
    for path, label in train_paths:
        try:
            feat = get_features(path)  # shape: (3, 162)
            for k in range(3):
                features.append(feat[k])
                labels.append(label)
        except Exception as e:
            print(f"  SKIP train {path}: {e}")
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def extract_test_features(test_paths):
    """For test WAVs: extract ONLY the original feature (no augmentation)."""
    features = []
    labels = []
    for path, label in test_paths:
        try:
            import librosa
            data, sr = librosa.load(path, duration=2.5, offset=0.6)
            feat = extract_features(data, sr)  # shape: (162,) — original only
            features.append(feat)
            labels.append(label)
        except Exception as e:
            print(f"  SKIP test {path}: {e}")
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def main():
    print("=" * 60)
    print("Clean Preprocessing: Split-then-Augment")
    print(f"Seed: {SEED}, Train ratio: {TRAIN_RATIO}")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    # Step 1: Collect WAV files by class
    print("\n[1/4] Collecting WAV files...")
    files_by_class = collect_wav_files()
    total = sum(len(v) for v in files_by_class.values())
    print(f"  Total: {total} original WAV files")

    # Step 2: Stratified split at FILE level (no leakage)
    print("\n[2/4] Splitting at file level (stratified per class)...")
    train_paths, test_paths = stratified_split(files_by_class)
    print(f"  Train WAVs: {len(train_paths)}")
    print(f"  Test WAVs:  {len(test_paths)}")

    # Verify no overlap
    train_set = set(p for p, _ in train_paths)
    test_set = set(p for p, _ in test_paths)
    assert train_set.isdisjoint(test_set), "FATAL: train/test overlap detected!"
    print("  No overlap between train and test WAV files.")

    # Step 3: Extract features
    print("\n[3/4] Extracting features...")
    print("  Training set (3x augmentation per WAV)...")
    X_train, y_train = extract_train_features(train_paths)
    print(f"    X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"    Label distribution: {np.bincount(y_train)}")

    print("  Test set (original features only, no augmentation)...")
    X_test, y_test = extract_test_features(test_paths)
    print(f"    X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"    Label distribution: {np.bincount(y_test)}")

    # Step 4: Save
    print("\n[4/4] Saving files...")
    np.save(TRAIN_DATA, X_train)
    np.save(TRAIN_LABEL, y_train)
    np.save(TEST_DATA, X_test)
    np.save(TEST_LABEL, y_test)
    print(f"  {TRAIN_DATA}")
    print(f"  {TRAIN_LABEL}")
    print(f"  {TEST_DATA}")
    print(f"  {TEST_LABEL}")

    print("\n" + "=" * 60)
    print("Preprocessing complete. No data leakage.")
    print(f"Train: {len(X_train)} samples from {len(train_paths)} WAVs (3x augmented)")
    print(f"Test:  {len(X_test)} samples from {len(test_paths)} WAVs (original only)")
    print("=" * 60)


if __name__ == "__main__":
    main()
