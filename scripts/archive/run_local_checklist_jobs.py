"""Run local checklist jobs (no training): verify, C2, confusion audit, DATA_FREEZE.

Usage:
  python scripts/run_local_checklist_jobs.py
  python scripts/run_local_checklist_jobs.py --skip-c2   # no GPU / no C-BESD
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CM_EXPECT = {
    "confusion_exp1_self_attention.json": 540,
    "confusion_exp2_prosody_guided.json": 540,
    "confusion_exp3_adult_iemocap.json": 2371,
    "confusion_exp4_zero_shot_fau.json": 3389,
    "confusion_exp5_fau_indomain.json": 3389,
    "confusion_exp5b_self_attention_fau.json": 3389,
}


def _run(cmd: list[str], desc: str) -> bool:
    print(f"\n>>> {desc}")
    r = subprocess.run(cmd, cwd=ROOT)
    ok = r.returncode == 0
    print(f"    {'OK' if ok else 'FAIL'} (exit {r.returncode})")
    return ok


def _fig_dir() -> Path:
    for p in (
        ROOT / "results_remote" / "results" / "figures",
        ROOT / "results" / "figures",
        ROOT / "paper_draft" / "figures",
    ):
        if (p / "confusion_exp1_self_attention.json").exists():
            return p
        if any(p.glob("confusion_exp*.json")):
            return p
    return ROOT / "results_remote" / "results" / "figures"


def audit_confusion() -> list[str]:
    issues = []
    fig = _fig_dir()
    print(f"\n>>> Confusion audit ({fig})")
    for name, expect in CM_EXPECT.items():
        path = fig / name
        if not path.exists():
            issues.append(f"missing {name}")
            print(f"  MISSING {name}")
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        cm = np.array(d.get("confusion_matrix", d.get("matrix", [])))
        total = int(cm.sum()) if cm.size else -1
        ok = total == expect
        print(f"  {'PASS' if ok else 'FAIL'} {name}: sum={total} (expect {expect})")
        if not ok:
            issues.append(f"{name}: sum={total} != {expect}")
    return issues


def ensure_exp2_ckpt() -> Path | None:
    dst = ROOT / "checkpoints" / "exp2_prosody_guided" / "best_model.pt"
    src = ROOT / "checkpoints" / "autodl" / "exp2_prosody_guided" / "best_model.pt"
    if dst.exists():
        return dst
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            import shutil

            shutil.copy2(src, dst)
            print(f"[ckpt] copied {src.name} -> {dst}")
        return dst
    print("[ckpt] exp2 checkpoint not found (skip C2)")
    return None


def update_data_freeze() -> None:
    logs = ROOT / "results" / "logs"
    rec = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "ac_suite_2026-05",
        "source": "run_local_checklist_jobs.py",
    }
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        rec["git_commit"] = r.stdout.strip()
    except Exception:
        rec["git_commit"] = None
    for name in [
        "exp1_self_attention.json",
        "exp2_prosody_guided.json",
        "fau_multiseed_summary.json",
    ]:
        p = logs / name
        if p.exists():
            rec[name] = json.loads(p.read_text(encoding="utf-8"))
    out = logs / "DATA_FREEZE.json"
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[freeze] updated {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-c2", action="store_true")
    args = parser.parse_args()

    ok_verify = _run(
        [sys.executable, "scripts/verify_experiment_jsons.py"],
        "Phase 1.1 verify JSONs",
    )
    _run(
        [sys.executable, "scripts/aggregate_fau_multiseed.py"],
        "Phase 1 multiseed aggregate",
    )

    cm_issues = audit_confusion()

    ok_c2 = True
    if not args.skip_c2:
        ckpt = ensure_exp2_ckpt()
        if ckpt:
            ok_c2 = _run(
                [
                    sys.executable,
                    "scripts/run_prosody_diagnostics.py",
                    "--checkpoint",
                    str(ckpt),
                ],
                "Phase 1.4 C2 diagnostics (local, not training)",
            )
        else:
            ok_c2 = False
            print(">>> Phase 1.4 C2 skipped (no checkpoint)")

    update_data_freeze()

    print("\n=== LOCAL CHECKLIST JOBS SUMMARY ===")
    print(f"  verify: {'PASS' if ok_verify else 'FAIL'}")
    print(f"  C2:     {'PASS/skipped' if ok_c2 or args.skip_c2 else 'FAIL'}")
    print(f"  CM:     {len(cm_issues)} issue(s)")
    for i in cm_issues:
        print(f"    - {i}")

    if not ok_verify or cm_issues or (not ok_c2 and not args.skip_c2):
        sys.exit(1)


if __name__ == "__main__":
    main()
