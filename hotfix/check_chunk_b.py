#!/usr/bin/env python3
"""Chunk B: is E5-03 layer-weights an exact duplicate of E1-02? Prints a short verdict.
Does NOT dump file contents."""
import json
import math
import os
import sys

F_E503 = "results/analysis/layer_weights_E5-03_C-BESD.json"
F_E102 = "results/analysis/layer_weights_E1-02_C-BESD.json"


def load(p):
    if not os.path.isfile(p):
        print("ERROR: missing file:", p)
        sys.exit(2)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get(d, key):
    """Find `key` at top level or nested one or more levels down."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            if isinstance(v, dict):
                r = get(v, key)
                if r is not None:
                    return r
    return None


def main():
    a = load(F_E503)
    b = load(F_E102)

    am_a, am_b = get(a, "argmax_layer"), get(b, "argmax_layer")
    en_a, en_b = get(a, "entropy"), get(b, "entropy")
    print(f"E5-03: argmax_layer={am_a} entropy={en_a}")
    print(f"E1-02: argmax_layer={am_b} entropy={en_b}")

    full_equal = (a == b)
    fields_equal = (
        am_a is not None and am_a == am_b
        and get(a, "layer_weights") == get(b, "layer_weights")
    )

    nan_a = isinstance(en_a, float) and math.isnan(en_a)
    nan_b = isinstance(en_b, float) and math.isnan(en_b)

    if full_equal or fields_equal:
        # Identical is EXPECTED: E5-03 (weighted-fusion baseline) and E1-02
        # (default self_attn/frozen/C-BESD model) are the same trained model
        # -- identical layer_fusion.layer_weights and identical val_wa=0.95896.
        # So this is a legitimate duplicate, not a copy error. The real defect
        # is entropy=NaN (entropy computed on raw signed logits, not softmax).
        print("INFO: E5-03 == E1-02 (same trained model; identical fusion logits).")
        if nan_a or nan_b:
            print("DEFECT: entropy=NaN -> run `python hotfix/extract_e5_01.py` to fix.")
            sys.exit(1)
        print("OK: entropy is valid; duplication is expected, not an error.")
        sys.exit(0)

    print("OK: E5-03 differs from E1-02.")


if __name__ == "__main__":
    main()
