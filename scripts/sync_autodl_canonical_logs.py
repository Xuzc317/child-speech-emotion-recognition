"""Write approved AutoDL final matrix into results/logs and publication_package."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from verify_experiment_jsons import CANONICAL, REG_HYPER

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "logs"
PUB_LOGS = ROOT / "publication_package" / "logs"
CSV_PATH = ROOT / "publication_package" / "experiment_results.csv"
MANIFEST = ROOT / "results" / "logs" / "MANIFEST_autodl_final.json"

MATRIX_VERSION = "autodl_final_2026-05-19"
SOURCE_NOTE = (
    "Restored from user-approved AutoDL final matrix (paper_draft). "
    "Original results_remote/ sync was not retained locally. "
    "best_val_wa omitted unless recovered from remote re-sync (task B1)."
)

SPEECH_TYPE = {
    "exp1_self_attention.json": "acted",
    "exp2_prosody_guided.json": "acted",
    "exp3_adult_iemocap.json": "spontaneous",
    "exp4_zero_shot_fau.json": "spontaneous",
    "exp5_fau_indomain.json": "spontaneous",
    "exp5b_self_attention_fau.json": "spontaneous",
}

TRAIN_DISPLAY = {
    "c-besd": "C-BESD",
    "iemocap": "IEMOCAP",
    "fau-aibo": "FAU-Aibo",
}


def _build_record(name: str, spec: dict) -> dict:
    reg = spec["reg_profile"]
    hyp = REG_HYPER[reg]
    rec = {
        **spec,
        "best_val_wa": None,
        "weight_decay": hyp["weight_decay"],
        "label_smoothing": hyp["label_smoothing"],
        "pooling_dropout": hyp["pooling_dropout"],
        "grad_clip": hyp["grad_clip"],
        "matrix_version": MATRIX_VERSION,
        "source": SOURCE_NOTE,
    }
    return rec


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _update_csv() -> None:
    rows = []
    for name, spec in CANONICAL.items():
        train = spec["train_data"][0]
        test = spec["test_data"][0]
        rows.append({
            "experiment": spec["exp_name"],
            "train_data": TRAIN_DISPLAY.get(train, train),
            "test_data": TRAIN_DISPLAY.get(test, test),
            "pooling_type": spec["pooling_type"],
            "speech_type": SPEECH_TYPE[name],
            "reg_profile": spec["reg_profile"],
            "test_wa": spec["test_wa"],
            "test_uar": spec["test_uar"],
            "best_val_wa": "",
            "best_epoch": spec["best_epoch"],
            "matrix_version": MATRIX_VERSION,
        })
    fieldnames = list(rows[0].keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    written = []
    for name, spec in CANONICAL.items():
        record = _build_record(name, spec)
        for root in (LOGS, PUB_LOGS):
            _write_json(root / name, record)
        written.append(name)

    PUB_LOGS.mkdir(parents=True, exist_ok=True)
    aux_pairs = [
        (ROOT / "results" / "logs" / "apc_metrics.json", PUB_LOGS / "apc_metrics.json"),
        (ROOT / "results" / "distribution_shift.json", ROOT / "publication_package" / "distribution_shift.json"),
        (ROOT / "results" / "layer_weights.json", ROOT / "publication_package" / "layer_weights.json"),
    ]
    for src, dst in aux_pairs:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    manifest = {
        "matrix_version": MATRIX_VERSION,
        "note": SOURCE_NOTE,
        "files": written,
        "paper_percent_metrics": {
            name: {
                "WA": round(spec["test_wa"] * 100, 2),
                "UAR": round(spec["test_uar"] * 100, 2),
                "best_epoch": spec["best_epoch"],
            }
            for name, spec in CANONICAL.items()
        },
    }
    _write_json(MANIFEST, manifest)
    _update_csv()
    print(f"Wrote {len(written)} experiment JSONs -> {LOGS} and {PUB_LOGS}")
    print(f"Updated {CSV_PATH}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
