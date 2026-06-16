"""Upload C2 fix and run aggregate + prosody diagnostics on AutoDL."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import tmp_paramiko_autodl_runner as autodl  # noqa: E402

REMOTE = autodl.REMOTE_PROJECT_ROOT
FINISH = f"""
set -e
cd "{REMOTE}"
export PYTHONPATH="${{PWD}}:${{PYTHONPATH:-}}"
export SER_C_BESD_PATH="/root/autodl-tmp/datasets/BESD/BESD/MY"
export SER_IEMOCAP_PATH="/root/autodl-tmp/IEMOCAP/wavs"
export SER_FAU_AIBO_PATH="/root/autodl-tmp/IS2009EmotionChallenge/wav"
PY="/root/miniconda3/bin/python"
echo "=== finish C2: multiseed aggregate ==="
"$PY" scripts/aggregate_fau_multiseed.py
echo "=== finish C2: prosody diagnostics ==="
"$PY" scripts/run_prosody_diagnostics.py --checkpoint checkpoints/exp2_prosody_guided/best_model.pt --seed 42
cp -f results/layer_weights.json publication_package/layer_weights.json 2>/dev/null || true
L=$(ls -t ac_suite_logs/resume_*.log 2>/dev/null | head -1)
echo "=== AC SUITE RESUME DONE $(date -Iseconds) ===" | tee -a "$L"
ls -la results/logs/fau_multiseed_summary.json results/logs/apc_metrics.json publication_package/xai_raw_data.npz 2>/dev/null || true
"""


def main() -> None:
    local_root = ROOT
    print("[connect] finishing C2 on AutoDL...")
    client = autodl._connect()
    sftp = client.open_sftp()
    try:
        autodl._upload_tree(
            sftp,
            local_root=local_root,
            remote_root=REMOTE,
            include_dirs=["src"],
            include_files=[
                "scripts/run_prosody_diagnostics.py",
                "scripts/aggregate_fau_multiseed.py",
            ],
        )
        code, out, err = autodl._exec(client, FINISH)
        print(out)
        if err:
            print(err, file=sys.stderr)
        if code != 0:
            raise SystemExit(code)
        print("[done] remote C2 + multiseed summary complete")
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    main()
