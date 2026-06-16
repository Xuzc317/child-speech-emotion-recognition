"""Merge results_remote JSON (authoritative AutoDL) into results/logs and publication_package."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE_LOGS = ROOT / "results_remote" / "results" / "logs"
LOCAL_LOGS = ROOT / "results" / "logs"
PUB_LOGS = ROOT / "publication_package" / "logs"
PUB_ROOT = ROOT / "publication_package"

CORE = [
    "exp1_self_attention.json",
    "exp2_prosody_guided.json",
    "exp3_adult_iemocap.json",
    "exp4_zero_shot_fau.json",
    "exp5_fau_indomain.json",
    "exp5b_self_attention_fau.json",
]
MULTISEED = [
    "exp5_fau_indomain_s123.json",
    "exp5_fau_indomain_s456.json",
    "exp5b_self_attention_fau_s123.json",
    "exp5b_self_attention_fau_s456.json",
]
AUX_LOGS = ("apc_metrics.json", "fau_multiseed_summary.json")


def main() -> None:
    if not REMOTE_LOGS.exists():
        print(f"Missing {REMOTE_LOGS}; run: python scripts/tmp_paramiko_autodl_runner.py --auto-resume")
        return

    LOCAL_LOGS.mkdir(parents=True, exist_ok=True)
    PUB_LOGS.mkdir(parents=True, exist_ok=True)

    for name in CORE:
        src = REMOTE_LOGS / name
        if not src.exists():
            print(f"SKIP missing remote: {name}")
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        data["source"] = "autodl_remote_sync"
        data["matrix_version"] = "autodl_final_2026-05-19"
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        (LOCAL_LOGS / name).write_text(text, encoding="utf-8")
        (PUB_LOGS / name).write_text(text, encoding="utf-8")
        wa = data["test_wa"] * 100
        print(f"  merged {name}: WA={wa:.2f}% best_val={data.get('best_val_wa', 0)*100:.2f}%")

    for name in MULTISEED:
        src = REMOTE_LOGS / name
        if not src.exists():
            print(f"SKIP missing remote: {name}")
            continue
        text = src.read_text(encoding="utf-8")
        (LOCAL_LOGS / name).write_text(text, encoding="utf-8")
        (PUB_LOGS / name).write_text(text, encoding="utf-8")
        d = json.loads(text)
        print(f"  merged {name}: WA={d['test_wa']*100:.2f}% seed={d.get('seed')}")

    for aux in AUX_LOGS:
        src = REMOTE_LOGS / aux
        if src.exists():
            shutil.copy2(src, LOCAL_LOGS / aux)
            if aux != "fau_multiseed_summary.json":
                shutil.copy2(src, PUB_LOGS / aux)
            print(f"  merged {aux}")

    for aux in ("layer_weights.json", "distribution_shift.json"):
        for src in (ROOT / "results_remote" / "results" / aux, ROOT / "results_remote" / aux):
            if src.exists():
                shutil.copy2(src, ROOT / "results" / aux.replace("layer_weights", "layer_weights"))
                if aux == "layer_weights.json":
                    shutil.copy2(src, ROOT / "results" / "layer_weights.json")
                    shutil.copy2(src, PUB_ROOT / "layer_weights.json")
                else:
                    shutil.copy2(src, PUB_ROOT / aux)
                print(f"  merged {aux}")
                break

    # Refresh experiment_results.csv from merged logs
    import csv
    rows = []
    for name in CORE:
        p = LOCAL_LOGS / name
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "experiment": d["exp_name"],
            "train_data": d["train_data"][0],
            "test_data": d["test_data"][0],
            "pooling_type": d["pooling_type"],
            "reg_profile": d.get("reg_profile", ""),
            "test_wa": d["test_wa"],
            "test_uar": d["test_uar"],
            "best_val_wa": d.get("best_val_wa"),
            "best_epoch": d["best_epoch"],
            "matrix_version": "autodl_remote_sync",
        })
    csv_path = PUB_ROOT / "experiment_results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  updated {csv_path}")

    print("\nRun: python scripts/verify_experiment_jsons.py")


if __name__ == "__main__":
    main()
