"""Regenerate data_all.npy and label_all.npy from BESD WAV files"""
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import get_features

WAV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "数据集", "BESD", "BESD", "MY")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_all.npy")
LABEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_all.npy")

CLASS_NAMES = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD']
CLASS_DICT = {name: i for i, name in enumerate(CLASS_NAMES)}

print(f"WAV directory: {WAV_DIR}")
print(f"Output data: {DATA_PATH}")
print(f"Output labels: {LABEL_PATH}")

all_features = []
all_labels = []

for class_name in CLASS_NAMES:
    class_dir = os.path.join(WAV_DIR, class_name)
    if not os.path.isdir(class_dir):
        print(f"  WARNING: directory not found: {class_dir}")
        continue
    wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
    print(f"Processing {class_name}: {len(wav_files)} wav files")
    for wav_file in wav_files:
        wav_path = os.path.join(class_dir, wav_file)
        try:
            feat = get_features(wav_path)  # shape: (3, 162)
            for k in range(3):
                all_features.append(feat[k])
                all_labels.append(CLASS_DICT[class_name])
        except Exception as e:
            print(f"  SKIP {wav_file}: {e}")

all_features = np.array(all_features, dtype=np.float32)
all_labels = np.array(all_labels, dtype=np.int64)

print(f"\nSaving features: {all_features.shape} (dtype={all_features.dtype})")
np.save(DATA_PATH, all_features)
print(f"Saving labels: {all_labels.shape} (dtype={all_labels.dtype})")
np.save(LABEL_PATH, all_labels)

# Verify
print(f"\nVerification:")
print(f"  Label distribution: {np.bincount(all_labels)}")
print(f"  Unique labels: {np.unique(all_labels)}")
print("Done!")
