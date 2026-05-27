"""Compute canonical FD/SMMD for C-BESD vs IEMOCAP and C-BESD vs FAU Aibo.

Uses the AC suite protocol: online WavLM features, 70/15/15 hash split, seed=42.
Updates fd_accuracy_table.json with verified FD values.

Run on AutoDL GPU:
  cd /root/autodl-tmp/d-ser
  export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
  export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
  export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/wav
  python scripts/compute_canonical_fd.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import SSLBackbone
from src.evaluation import DistributionShiftProbe
from src.data import get_dataloaders


def compute_fd_for_pair(probe, ref_name, tgt_name, max_samples=500):
    """Compute FD and SMMD between two corpora."""
    print(f"\n{'='*60}")
    print(f"Computing FD: {ref_name} vs {tgt_name}")
    print(f"{'='*60}")

    dls_ref = get_dataloaders([ref_name], batch_size=16, seed=42)
    dls_tgt = get_dataloaders([tgt_name], batch_size=16, seed=42)

    print(f"  {ref_name} train samples: {len(dls_ref['train'].dataset)}")
    print(f"  {tgt_name} train samples: {len(dls_tgt['train'].dataset)}")

    print(f"  Extracting {ref_name} features (max {max_samples})...")
    feats_ref = probe.extract_features(dls_ref["train"], max_samples=max_samples)
    print(f"    shape: {feats_ref.shape}")

    print(f"  Extracting {tgt_name} features (max {max_samples})...")
    feats_tgt = probe.extract_features(dls_tgt["train"], max_samples=max_samples)
    print(f"    shape: {feats_tgt.shape}")

    result = probe.evaluate(feats_ref, feats_tgt)
    print(f"  FD={result['fd']:.4f}, SMMD={result['smmd']:.4f}")
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading WavLM backbone (frozen)...")
    backbone = SSLBackbone(model_name="wavlm", frozen=True, device=device)
    probe = DistributionShiftProbe(backbone=backbone, pooling="mean", device=device)

    pairs = [
        ("c-besd", "iemocap"),
        ("c-besd", "fau-aibo"),
    ]

    results = {}
    for ref, tgt in pairs:
        r = compute_fd_for_pair(probe, ref, tgt, max_samples=500)
        results[f"{ref}_vs_{tgt}"] = {
            "ref_corpus": ref,
            "tgt_corpus": tgt,
            "fd": r["fd"],
            "smmd": r["smmd"],
            "protocol": "ac_suite_2026-05",
            "method": "DistributionShiftProbe (online WavLM features, mean pooling, max_samples=500, seed=42)",
        }

    # Save results
    out_path = ROOT / "results" / "canonical_fd_pairs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # Print summary
    print("\n=== CANONICAL FD PAIRS (AC Suite Protocol) ===")
    for key, r in results.items():
        print(f"  {key}: FD={r['fd']:.4f}, SMMD={r['smmd']:.4f}")

    print("\nNext: update publication_package/fd_accuracy_table.json with these values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
