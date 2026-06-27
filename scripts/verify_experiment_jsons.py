"""Verify experiment JSON logs against the approved AutoDL final paper matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "logs"

# Approved matrix (test metrics as fractions, WA/UAR in percent for display)
CANONICAL = {
    "exp1_self_attention.json": {
        "exp_name": "exp1_self_attention",
        "pooling_type": "self_attention",
        "train_data": ["c-besd"],
        "test_data": ["c-besd"],
        "test_wa": 0.9278,
        "test_uar": 0.9279,
        "best_epoch": 30,
        "reg_profile": "default",
    },
    "exp2_prosody_guided.json": {
        "exp_name": "exp2_prosody_guided",
        "pooling_type": "prosody_guided",
        "train_data": ["c-besd"],
        "test_data": ["c-besd"],
        "test_wa": 0.9130,
        "test_uar": 0.9135,
        "best_epoch": 10,
        "reg_profile": "default",
    },
    "exp3_adult_iemocap.json": {
        "exp_name": "exp3_adult_iemocap",
        "pooling_type": "prosody_guided",
        "train_data": ["iemocap"],
        "test_data": ["iemocap"],
        "test_wa": 0.5867,
        "test_uar": 0.5911,
        "best_epoch": 8,
        "reg_profile": "default",
    },
    "exp4_zero_shot_fau.json": {
        "exp_name": "exp4_zero_shot_fau",
        "pooling_type": "prosody_guided",
        "train_data": ["c-besd"],
        "test_data": ["fau-aibo"],
        "test_wa": 0.1956,   # 2026-05-27: corrected from split='all' (0.1870) to split='test' N=3389
        "test_uar": 0.2383,
        "best_epoch": 10,
        "reg_profile": "default",
    },
    "exp5_fau_indomain.json": {
        "exp_name": "exp5_fau_indomain",
        "pooling_type": "prosody_guided",
        "train_data": ["fau-aibo"],
        "test_data": ["fau-aibo"],
        "test_wa": 0.6636,
        "test_uar": 0.5635,
        "best_epoch": 3,
        "reg_profile": "fau",
    },
    "exp5b_self_attention_fau.json": {
        "exp_name": "exp5b_self_attention_fau",
        "pooling_type": "self_attention",
        "train_data": ["fau-aibo"],
        "test_data": ["fau-aibo"],
        "test_wa": 0.6618,
        "test_uar": 0.5823,
        "best_epoch": 2,
        "reg_profile": "fau",
    },
}

REG_HYPER = {
    "default": {
        "weight_decay": 1e-3,
        "label_smoothing": 0.1,
        "pooling_dropout": 0.0,
        "grad_clip": None,
    },
    "fau": {
        "weight_decay": 5e-3,
        "label_smoothing": 0.15,
        "pooling_dropout": 0.3,
        "grad_clip": 1.0,
    },
}

TOL_WA = 0.0005  # 0.05 pp on fraction scale


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _check(name: str, path: Path) -> list[str]:
    errors: list[str] = []
    expected = CANONICAL[name]
    try:
        data = _load(path)
    except FileNotFoundError:
        return [f"MISSING: {path}"]
    for key in ("exp_name", "pooling_type", "best_epoch", "reg_profile"):
        if data.get(key) != expected[key]:
            errors.append(f"{key}: got {data.get(key)!r}, want {expected[key]!r}")
    for key in ("test_wa", "test_uar"):
        got = data.get(key)
        want = expected[key]
        if got is None:
            errors.append(f"{key}: missing")
        elif abs(float(got) - want) > TOL_WA:
            errors.append(f"{key}: got {float(got):.4f}, want {want:.4f}")
    reg = expected["reg_profile"]
    hyp = REG_HYPER[reg]
    for hk, hv in hyp.items():
        got = data.get(hk)
        if hk == "grad_clip":
            if hv is None and got not in (None, "null"):
                if got is not None:
                    errors.append(f"grad_clip: want None, got {got}")
            elif hv is not None and float(got) != float(hv):
                errors.append(f"grad_clip: got {got}, want {hv}")
        elif got is not None and abs(float(got) - float(hv)) > 1e-9:
            errors.append(f"{hk}: got {got}, want {hv}")
    return errors


def main() -> int:
    roots = [LOGS, ROOT / "publication_package" / "logs"]
    failed = False
    for name in CANONICAL:
        print(f"\n=== {name} ===")
        for root in roots:
            path = root / name
            errs = _check(name, path)
            label = path.relative_to(ROOT)
            if errs:
                failed = True
                print(f"  FAIL {label}")
                for e in errs:
                    print(f"    - {e}")
            else:
                d = _load(path)
                wa = float(d["test_wa"]) * 100
                print(f"  PASS {label}  WA={wa:.2f}%  epoch={d['best_epoch']}  reg={d.get('reg_profile')}")
    if failed:
        print("\nSome checks FAILED. Run: python scripts/sync_autodl_canonical_logs.py")
        return 1
    print("\nAll checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
