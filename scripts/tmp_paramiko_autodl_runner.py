"""Temporary Paramiko runner for AutoDL experiments.

This script follows the user-specified SSH override:
  - Connects using paramiko.SSHClient + AutoAddPolicy
  - Uses exec_command() with nohup to launch remote jobs
  - Uses open_sftp() to upload code and download results
"""

from __future__ import annotations

import argparse
import os
import posixpath
from pathlib import Path
from stat import S_ISDIR
import textwrap

import paramiko


REMOTE_HOST = "connect.cqa1.seetacloud.com"
REMOTE_PORT = 12112
REMOTE_USER = "root"
REMOTE_PASS = "9HmcVfCXUFVD"

REMOTE_PROJECT_ROOT = "/root/autodl-tmp/d-ser"
REMOTE_FAU_EXTRACT_ROOT = "/root/autodl-tmp/IS2009EmotionChallenge"


def _connect() -> paramiko.SSHClient:
    # Exact connection logic requested by the user.
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        REMOTE_HOST,
        port=REMOTE_PORT,
        username=REMOTE_USER,
        password=REMOTE_PASS,
    )
    return client


def _exec(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command)
    code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="ignore")
    err = stderr.read().decode("utf-8", errors="ignore")
    return code, out, err


def _ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = [p for p in remote_dir.split("/") if p]
    path = ""
    for part in parts:
        path = f"{path}/{part}"
        try:
            sftp.stat(path)
        except OSError:
            sftp.mkdir(path)


def _upload_tree(
    sftp: paramiko.SFTPClient,
    local_root: Path,
    remote_root: str,
    include_dirs: list[str],
    include_files: list[str],
) -> None:
    _ensure_remote_dir(sftp, remote_root)

    for rel_file in include_files:
        src = local_root / rel_file
        if not src.exists():
            continue
        dst = posixpath.join(remote_root, rel_file.replace("\\", "/"))
        _ensure_remote_dir(sftp, posixpath.dirname(dst))
        sftp.put(str(src), dst)

    for rel_dir in include_dirs:
        src_dir = local_root / rel_dir
        if not src_dir.exists():
            continue
        for p in src_dir.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(local_root).as_posix()
            dst = posixpath.join(remote_root, rel)
            _ensure_remote_dir(sftp, posixpath.dirname(dst))
            sftp.put(str(p), dst)


def _download_tree(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        remote_path = posixpath.join(remote_dir, entry.filename)
        local_path = local_dir / entry.filename
        if S_ISDIR(entry.st_mode):
            _download_tree(sftp, remote_path, local_path)
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_path, str(local_path))


def _write_remote_runner(client: paramiko.SSHClient) -> None:
    runner = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        cd "{REMOTE_PROJECT_ROOT}"
        mkdir -p results/logs checkpoints
        export PYTHONPATH="{REMOTE_PROJECT_ROOT}:$PYTHONPATH"

        python -m src.train --train_data c-besd --pooling_type self_attention --exp_name exp1_self_attention
        python -m src.train --train_data c-besd --pooling_type prosody_guided --exp_name exp2_prosody_guided
        python -m src.train --train_data iemocap --pooling_type prosody_guided --exp_name exp3_adult_iemocap
        python -m src.train --train_data c-besd --test_data fau-aibo --pooling_type prosody_guided --exp_name exp4_zero_shot_fau

        python -m src.train --train_data fau-aibo --pooling_type prosody_guided --exp_name exp5_fau_indomain --reg_profile fau
        python -m src.train --train_data fau-aibo --pooling_type self_attention --exp_name exp5b_self_attention_fau --reg_profile fau

        if [ -d "/root/autodl-tmp/datasets/myst" ]; then
          python -m src.train --train_data myst --pooling_type prosody_guided --exp_name exp_myst_prosody --reg_profile fau || true
          python -m src.train --train_data myst --pooling_type self_attention --exp_name exp_myst_self_attention --reg_profile fau || true
        fi
        if [ -d "/root/autodl-tmp/datasets/kidstalc" ]; then
          python -m src.train --train_data kidstalc --pooling_type prosody_guided --exp_name exp_kidstalc_prosody --reg_profile fau || true
          python -m src.train --train_data kidstalc --pooling_type self_attention --exp_name exp_kidstalc_self_attention --reg_profile fau || true
        fi
        """
    )
    runner_escaped = runner.replace("\\", "\\\\").replace('"', '\\"')
    cmd = (
        f'python - <<\'PY\'\n'
        f'from pathlib import Path\n'
        f'content = "{runner_escaped}"\n'
        f'Path("{REMOTE_PROJECT_ROOT}/run_matrix.sh").write_text(content, encoding="utf-8")\n'
        f'PY'
    )
    code, out, err = _exec(client, cmd)
    if code != 0:
        raise RuntimeError(f"Failed writing remote runner: {err or out}")
    _exec(client, f"chmod +x {REMOTE_PROJECT_ROOT}/run_matrix.sh")


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporary AutoDL Paramiko runner")
    parser.add_argument("--local-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--start-only", action="store_true")
    args = parser.parse_args()

    local_root = Path(args.local_root).resolve()
    client = _connect()
    sftp = client.open_sftp()

    try:
        if not args.download_only:
            _upload_tree(
                sftp,
                local_root=local_root,
                remote_root=REMOTE_PROJECT_ROOT,
                include_dirs=["src", "scripts", "docs"],
                include_files=["requirements.txt", "run_experiments.sh", ".gitignore"],
            )

            fau_tar = local_root / "fau_aibo.tar.gz"
            if fau_tar.exists():
                remote_tar = f"{REMOTE_FAU_EXTRACT_ROOT}/fau_aibo.tar.gz"
                _ensure_remote_dir(sftp, REMOTE_FAU_EXTRACT_ROOT)
                sftp.put(str(fau_tar), remote_tar)
                _exec(
                    client,
                    f'mkdir -p "{REMOTE_FAU_EXTRACT_ROOT}" '
                    f'&& tar -xzf "{remote_tar}" -C "{REMOTE_FAU_EXTRACT_ROOT}"',
                )

            _write_remote_runner(client)

        if not args.download_only:
            start_cmd = (
                f"cd {REMOTE_PROJECT_ROOT} && "
                "nohup bash run_matrix.sh > matrix_run.log 2>&1 & echo $!"
            )
            code, out, err = _exec(client, start_cmd)
            if code != 0:
                raise RuntimeError(f"Failed to start remote matrix: {err or out}")
            print(f"Remote matrix started with PID: {out.strip()}")

        if not args.start_only:
            code, _, _ = _exec(client, f"test -d {REMOTE_PROJECT_ROOT}/results/logs")
            if code == 0:
                _download_tree(
                    sftp,
                    f"{REMOTE_PROJECT_ROOT}/results/logs",
                    local_root / "results_remote" / "results" / "logs",
                )
            # Critical XAI bundle sync.
            for candidate in [
                f"{REMOTE_PROJECT_ROOT}/publication_package/xai_raw_data.npz",
                f"{REMOTE_PROJECT_ROOT}/results/xai_raw_data.npz",
            ]:
                try:
                    sftp.stat(candidate)
                    target = local_root / "publication_package" / "xai_raw_data.npz"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(candidate, str(target))
                    break
                except OSError:
                    continue
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    main()
