#!/usr/bin/env python3
"""Chunk B (corrected): regenerate genuine LayerFusion layer-weight JSONs.

WHY THIS IS NO LONGER "E5-01":
  Inspecting the checkpoints showed E5-01 (and E5-04) use fusion_mode='last' --
  a single best layer, with NO learnable WavLMLayerFusion.layer_weights to
  extract (the original exit-3 failure was therefore correct, not a bug).
  The runs that actually carry trained weighted-fusion logits are E5-03
  (fusion_mode='weighted') and E1-02 (the default self_attn / frozen / C-BESD
  model, whose layer_fusion.layer_weights are byte-identical to E5-03 -- they
  are effectively the same trained model, val_wa=0.95896 in both).

WHAT WAS BROKEN:
  results/analysis/layer_weights_{E5-03,E1-02}_C-BESD.json stored the RAW signed
  logits and computed entropy directly on them -> log(negative) -> entropy=NaN.
  src/models/layer_fusion.py applies F.softmax(layer_weights) in forward, and
  get_layer_weights() returns the softmaxed weights, so the faithful mixing
  distribution is softmax(logits). argmax is unchanged (softmax is monotonic).

WHAT THIS DOES:
  For each weighted-fusion run, re-read layer_fusion.layer_weights from the
  checkpoint, compute softmax_weights + entropy (nats) on the softmax dist,
  back up the old JSON (no-clobber .bak), and rewrite it preserving the existing
  keys (layer_weights, argmax_layer, entropy) plus additive softmax_weights /
  entropy_base / weight_key / fusion_mode / source_checkpoint. Prints a short
  summary only -- no full-tensor / full-state-dict dump.
"""
import json
import math
import os
import shutil
import sys

import torch

RUNS = [
    ("E5-03", [
        "checkpoints/autodl/b4/E5-03_s42/best_model.pt",
        "checkpoints/b4/E5-03_s42/best_model.pt",
    ]),
    ("E1-02", [
        "checkpoints/autodl/b1/E1-02_s42/best_model.pt",
        "checkpoints/b1/E1-02_s42/best_model.pt",
    ]),
]
OUT_TMPL = "results/analysis/layer_weights_{exp}_C-BESD.json"
EPS = 1e-12


def pick(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without the weights_only kwarg
        return torch.load(path, map_location="cpu")


def iter_tensors(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_tensors(v, f"{prefix}{k}.")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from iter_tensors(v, f"{prefix}{i}.")
    elif torch.is_tensor(obj):
        yield (prefix[:-1] if prefix else "<root>"), obj


def get_fusion_mode(obj):
    if not isinstance(obj, dict):
        return None
    for slot in ("args", "config"):
        a = obj.get(slot)
        if a is None:
            continue
        if isinstance(a, dict) and "fusion_mode" in a:
            return a["fusion_mode"]
        if getattr(a, "fusion_mode", None) is not None:
            return getattr(a, "fusion_mode")
    return None


def find_fusion_weights(obj):
    """Return (key, 1-D float tensor) for the learnable layer-fusion weights.

    Prefers the exact key '...layer_fusion.layer_weights'. Excludes WavLM's
    internal 'gru_rel_pos_const' (shape (1,12,1,1) -> squeezes to 12 -- a false
    positive). Falls back to any 12/13-length fusion/weight 1-D tensor.
    """
    best = None  # (rank, key, tensor); lower rank = better
    for k, v in iter_tensors(obj):
        kl = k.lower()
        if "gru_rel_pos" in kl:  # WavLM internal const, NOT layer fusion
            continue
        t = v.squeeze()
        if t.dim() != 1 or int(t.numel()) not in (12, 13):
            continue
        if kl.endswith("layer_fusion.layer_weights"):
            rank = 0
        elif "layer_fusion" in kl:
            rank = 1
        elif "fusion" in kl and "weight" in kl:
            rank = 2
        elif "layer_weight" in kl:
            rank = 3
        else:
            continue
        if best is None or rank < best[0]:
            best = (rank, k, t.detach().float())
    if best is None:
        return None, None
    return best[1], best[2]


def main():
    any_ok = False
    for exp, paths in RUNS:
        out = OUT_TMPL.format(exp=exp)
        ck = pick(paths)
        print(f"== {exp}")
        if ck is None:
            print(f"   SKIP: checkpoint not found: {paths}")
            continue
        obj = safe_load(ck)
        fmode = get_fusion_mode(obj)
        key, raw = find_fusion_weights(obj)
        if key is None:
            print(f"   SKIP: no learnable layer_fusion weights "
                  f"(fusion_mode={fmode}); nothing to extract")
            continue

        logits = raw.tolist()
        probs = torch.softmax(raw, dim=0).tolist()
        argmax_layer = max(range(len(probs)), key=lambda i: probs[i])
        entropy = float(-sum(p * math.log(p + EPS) for p in probs))

        if os.path.isfile(out) and not os.path.isfile(out + ".bak"):
            shutil.copyfile(out, out + ".bak")
            print(f"   backup: {out} -> {out}.bak")

        rec = {
            "layer_weights": logits,           # raw signed logits (schema unchanged)
            "argmax_layer": int(argmax_layer),
            "entropy": entropy,                # FIX: was NaN (entropy on raw logits)
            "softmax_weights": probs,          # additive: model's actual mixing weights
            "entropy_base": "nat",
            "weight_key": key,
            "fusion_mode": fmode,
            "source_checkpoint": ck,
        }
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        any_ok = True

        top3 = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
        print(f"   key={key}  fusion_mode={fmode}")
        print(f"   argmax_layer={argmax_layer} (L{argmax_layer + 1})  "
              f"entropy={entropy:.4f} nat (max ln12={math.log(12):.4f}; was NaN)")
        print("   top3 by softmax: "
              + ", ".join(f"L{i + 1}={probs[i]:.3f}" for i in top3))
        print(f"   wrote {out}")

    sys.exit(0 if any_ok else 3)


if __name__ == "__main__":
    main()
