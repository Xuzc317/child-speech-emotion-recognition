import os, sys, hashlib, warnings, re
from collections import defaultdict
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

DATASET_ROOTS = {
    "c-besd": os.environ.get("SER_C_BESD_PATH", "/root/autodl-tmp/datasets/BESD/BESD/MY"),
    "fau-aibo": os.environ.get("SER_FAU_AIBO_PATH", "/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav"),
    "iemocap": os.environ.get("SER_IEMOCAP_PATH", "/root/autodl-tmp/IEMOCAP/wavs"),
}
FAU_LABEL_FILE = "/root/autodl-tmp/IS2009EmotionChallenge/labels/IS2009EmotionChallenge/chunk_labels_5cl_corpus.txt"
OUTPUT_DIR = "/root/autodl-tmp/acoustic_analysis"

print("Script start")
print("Roots:", DATASET_ROOTS)
