"""Aggregate FAU multi-seed JSON logs (C1) into mean±std summary."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "logs"
OUT = ROOT / "results" / "logs" / "fau_multiseed_summary.json"

PATTERNS = {
    "exp5_fau_indomain": re.compile(r"^exp5_fau_indomain(_s\d+)?\.json$"),
    "exp5b_self_attention_fau": re.compile(r"^exp5b_self_attention_fau(_s\d+)?\.json$"),
}
SEEDS = (42, 123, 456)


def _load_group(base: str) -> list[dict]:
    rows = []
    for seed in SEEDS:
        for name in (f"{base}_s{seed}.json", f"{base}.json" if seed == 42 else None):
            if name is None:
                continue
            path = LOGS / name
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("seed", 42) == seed or name.endswith(f"_s{seed}.json"):
                rows.append(data)
                break
    return rows


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n_seeds": 0, "error": "no logs found"}
    wa = np.array([r["test_wa"] for r in rows], dtype=float)
    uar = np.array([r["test_uar"] for r in rows], dtype=float)
    return {
        "n_seeds": len(rows),
        "seeds": [int(r.get("seed", -1)) for r in rows],
        "test_wa_mean": float(wa.mean()),
        "test_wa_std": float(wa.std(ddof=0)) if len(wa) > 1 else 0.0,
        "test_uar_mean": float(uar.mean()),
        "test_uar_std": float(uar.std(ddof=0)) if len(uar) > 1 else 0.0,
        "runs": [
            {
                "exp_name": r["exp_name"],
                "seed": r.get("seed"),
                "test_wa": r["test_wa"],
                "test_uar": r["test_uar"],
                "best_epoch": r.get("best_epoch"),
            }
            for r in rows
        ],
    }


def main() -> None:
    summary = {"protocol": "ac_suite_2026-05", "seeds_target": list(SEEDS)}
    for base in PATTERNS:
        rows = _load_group(base)
        summary[base] = _summarize(rows)
        s = summary[base]
        if s.get("n_seeds"):
            print(
                f"{base}: WA={s['test_wa_mean']*100:.2f}±{s['test_wa_std']*100:.2f}% "
                f"UAR={s['test_uar_mean']*100:.2f}±{s['test_uar_std']*100:.2f}% (n={s['n_seeds']})"
            )
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
