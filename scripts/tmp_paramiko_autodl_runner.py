"""Temporary Paramiko runner for AutoDL experiments.

This script follows the user-specified SSH override:
  - Connects using paramiko.SSHClient + AutoAddPolicy
  - Uses exec_command() with nohup to launch remote jobs
  - Uses open_sftp() to upload code and download results
"""

from __future__ import annotations

import argparse
import posixpath
from pathlib import Path
from stat import S_ISDIR

import paramiko


REMOTE_HOST = "connect.cqa1.seetacloud.com"
REMOTE_PORT = 12112
REMOTE_USER = "root"
REMOTE_PASS = "9HmcVfCXUFVD"

REMOTE_PROJECT_ROOT = "/root/autodl-tmp/d-ser"
REMOTE_FAU_EXTRACT_ROOT = "/root/autodl-tmp/IS2009EmotionChallenge"
REMOTE_C_BESD_ROOT = "/root/autodl-tmp/datasets/BESD/BESD/MY"
CORE_EXPERIMENT_LOGS = {
    "exp1_self_attention.json",
    "exp2_prosody_guided.json",
    "exp3_adult_iemocap.json",
    "exp4_zero_shot_fau.json",
    "exp5_fau_indomain.json",
    "exp5b_self_attention_fau.json",
}


def _connect() -> paramiko.SSHClient:
    # Exact connection logic requested by the user.
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        REMOTE_HOST,
        port=REMOTE_PORT,
        username=REMOTE_USER,
        password=REMOTE_PASS,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
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

    uploaded = 0
    for rel_file in include_files:
        src = local_root / rel_file
        if not src.exists():
            continue
        dst = posixpath.join(remote_root, rel_file.replace("\\", "/"))
        _ensure_remote_dir(sftp, posixpath.dirname(dst))
        sftp.put(str(src), dst)
        uploaded += 1
        print(f"[upload:file] {rel_file}")

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
            uploaded += 1
            if uploaded % 25 == 0:
                print(f"[upload] {uploaded} files synced...")
    print(f"[upload] done, total files: {uploaded}")


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


def _upload_external_dir(
    sftp: paramiko.SFTPClient,
    local_dir: Path,
    remote_dir: str,
) -> None:
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")
    _ensure_remote_dir(sftp, remote_dir)
    uploaded = 0
    for p in local_dir.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(local_dir).as_posix()
        dst = posixpath.join(remote_dir, rel)
        _ensure_remote_dir(sftp, posixpath.dirname(dst))
        sftp.put(str(p), dst)
        uploaded += 1
        if uploaded % 250 == 0:
            print(f"[dataset-upload] {uploaded} files -> {remote_dir}")
    print(f"[dataset-upload] completed {uploaded} files -> {remote_dir}")


def _list_remote_logs(client: paramiko.SSHClient) -> set[str]:
    cmd = f"cd {REMOTE_PROJECT_ROOT} && ls -1 results/logs 2>/dev/null || true"
    _, out, _ = _exec(client, cmd)
    names = {line.strip() for line in out.splitlines() if line.strip()}
    return names


def _has_running_training(client: paramiko.SSHClient) -> bool:
    cmd = (
        "ps -ef | grep -E 'src.train|run_matrix.sh|matrix_cbesd' | "
        "grep -v grep >/dev/null && echo RUNNING || echo IDLE"
    )
    _, out, _ = _exec(client, cmd)
    return "RUNNING" in out


def _write_remote_runner(client: paramiko.SSHClient, sftp: paramiko.SFTPClient) -> None:
    runner = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euxo pipefail",
            f'cd "{REMOTE_PROJECT_ROOT}"',
            "mkdir -p results/logs checkpoints",
            f'export PYTHONPATH="{REMOTE_PROJECT_ROOT}:${{PYTHONPATH:-}}"',
            'PYTHON_BIN="$(command -v python || true)"',
            'if [ -z "$PYTHON_BIN" ]; then PYTHON_BIN="$(command -v python3 || true)"; fi',
            'if [ -z "$PYTHON_BIN" ] && [ -x "/root/miniconda3/bin/python" ]; then PYTHON_BIN="/root/miniconda3/bin/python"; fi',
            'if [ -z "$PYTHON_BIN" ] && [ -x "/opt/conda/bin/python" ]; then PYTHON_BIN="/opt/conda/bin/python"; fi',
            'if [ -z "$PYTHON_BIN" ]; then',
            '  echo "No python interpreter found on remote host." >&2',
            "  exit 1",
            "fi",
            "",
            'pick_existing_dir() {',
            '  for p in "$@"; do',
            '    if [ -d "$p" ]; then',
            '      echo "$p"',
            '      return 0',
            '    fi',
            '  done',
            '  return 1',
            '}',
            "",
            'SER_C_BESD_PATH="${SER_C_BESD_PATH:-$(pick_existing_dir /root/autodl-tmp/BESD/BESD/MY /root/autodl-tmp/datasets/BESD/BESD/MY /root/autodl-tmp/datasets/c-besd/MY || true)}"',
            'SER_IEMOCAP_PATH="${SER_IEMOCAP_PATH:-$(pick_existing_dir /root/autodl-tmp/IEMOCAP/wavs /root/autodl-tmp/datasets/IEMOCAP/wavs || true)}"',
            'SER_FAU_AIBO_PATH="${SER_FAU_AIBO_PATH:-$(pick_existing_dir /root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/IS2009EmotionChallenge/wav /root/autodl-tmp/IS2009EmotionChallenge/wav || true)}"',
            'export SER_C_BESD_PATH SER_IEMOCAP_PATH SER_FAU_AIBO_PATH',
            'echo "[paths] SER_C_BESD_PATH=${SER_C_BESD_PATH:-<missing>}"',
            'echo "[paths] SER_IEMOCAP_PATH=${SER_IEMOCAP_PATH:-<missing>}"',
            'echo "[paths] SER_FAU_AIBO_PATH=${SER_FAU_AIBO_PATH:-<missing>}"',
            "",
            'if [ -n "${SER_C_BESD_PATH:-}" ]; then',
            '  "$PYTHON_BIN" -m src.train --train_data c-besd --pooling_type self_attention --exp_name exp1_self_attention',
            '  "$PYTHON_BIN" -m src.train --train_data c-besd --pooling_type prosody_guided --exp_name exp2_prosody_guided',
            "else",
            '  echo "[skip] Exp1/Exp2 skipped: C-BESD path unavailable."',
            "fi",
            "",
            'if [ -n "${SER_IEMOCAP_PATH:-}" ]; then',
            '  "$PYTHON_BIN" -m src.train --train_data iemocap --pooling_type prosody_guided --exp_name exp3_adult_iemocap',
            "else",
            '  echo "[skip] Exp3 skipped: IEMOCAP path unavailable."',
            "fi",
            "",
            'if [ -n "${SER_C_BESD_PATH:-}" ] && [ -n "${SER_FAU_AIBO_PATH:-}" ]; then',
            '  "$PYTHON_BIN" -m src.train --train_data c-besd --test_data fau-aibo --pooling_type prosody_guided --exp_name exp4_zero_shot_fau',
            "else",
            '  echo "[skip] Exp4 skipped: C-BESD or FAU path unavailable."',
            "fi",
            "",
            'if [ -n "${SER_FAU_AIBO_PATH:-}" ]; then',
            '  "$PYTHON_BIN" -m src.train --train_data fau-aibo --pooling_type prosody_guided --exp_name exp5_fau_indomain --reg_profile fau',
            '  "$PYTHON_BIN" -m src.train --train_data fau-aibo --pooling_type self_attention --exp_name exp5b_self_attention_fau --reg_profile fau',
            "else",
            '  echo "[skip] Exp5/Exp5b skipped: FAU path unavailable."',
            "fi",
            "",
            'if [ -d "/root/autodl-tmp/datasets/myst" ]; then',
            '  "$PYTHON_BIN" -m src.train --train_data myst --pooling_type prosody_guided --exp_name exp_myst_prosody --reg_profile fau || true',
            '  "$PYTHON_BIN" -m src.train --train_data myst --pooling_type self_attention --exp_name exp_myst_self_attention --reg_profile fau || true',
            "fi",
            'if [ -d "/root/autodl-tmp/datasets/kidstalc" ]; then',
            '  "$PYTHON_BIN" -m src.train --train_data kidstalc --pooling_type prosody_guided --exp_name exp_kidstalc_prosody --reg_profile fau || true',
            '  "$PYTHON_BIN" -m src.train --train_data kidstalc --pooling_type self_attention --exp_name exp_kidstalc_self_attention --reg_profile fau || true',
            "fi",
            "",
        ]
    )
    remote_script = f"{REMOTE_PROJECT_ROOT}/run_matrix.sh"
    with sftp.file(remote_script, "w") as f:
        f.write(runner)
    code, out, err = _exec(client, f"chmod +x {remote_script}")
    if code != 0:
        raise RuntimeError(f"Failed chmod on remote runner: {err or out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporary AutoDL Paramiko runner")
    parser.add_argument("--local-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--start-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--log-lines", type=int, default=40)
    parser.add_argument("--debug-once", action="store_true")
    parser.add_argument("--sync-cbesd", action="store_true")
    parser.add_argument("--cbesd-local-path", default=None)
    parser.add_argument("--run-missing-cbesd", action="store_true")
    parser.add_argument("--auto-resume", action="store_true")
    args = parser.parse_args()

    local_root = Path(args.local_root).resolve()
    print("[connect] opening SSH connection...")
    client = _connect()
    print("[connect] connected.")
    sftp = client.open_sftp()

    try:
        if args.status_only:
            status_cmd = (
                f"cd {REMOTE_PROJECT_ROOT} && "
                "echo '====PROCESS====' && "
                "(ps -ef | grep -E 'run_matrix.sh|matrix_cbesd|src.train' | grep -v grep || true) && "
                "echo '====FILES====' && "
                "(ls -l run_matrix.sh matrix_run.log matrix_cbesd.log 2>/dev/null || true) && "
                "echo '====SCRIPT_HEAD====' && "
                "(sed -n '1,40p' run_matrix.sh 2>/dev/null || true) && "
                "echo '====TAIL====' && "
                f"(tail -n {args.log_lines} matrix_run.log || true) && "
                "echo '====TAIL_CBESD====' && "
                f"(tail -n {args.log_lines} matrix_cbesd.log || true) && "
                "echo '====LOG_FILES====' && "
                "(ls -1 results/logs 2>/dev/null || true)"
            )
            _, out, err = _exec(client, status_cmd)
            print(out)
            if err:
                print(err)
            return

        if args.sync_cbesd:
            default_cbesd = local_root.parent / "提交到团队" / "数据集" / "BESD" / "BESD" / "MY"
            local_cbesd = Path(args.cbesd_local_path).resolve() if args.cbesd_local_path else default_cbesd
            print(f"[dataset-upload] local C-BESD path: {local_cbesd}")
            _upload_external_dir(sftp, local_cbesd, REMOTE_C_BESD_ROOT)
            code, out, err = _exec(client, f"find {REMOTE_C_BESD_ROOT} -type f | wc -l")
            print(f"[dataset-upload] remote C-BESD file count: {(out or '').strip()}")
            if code != 0 and err:
                print(err)
            return

        if args.run_missing_cbesd:
            missing_cmd = (
                f'cd "{REMOTE_PROJECT_ROOT}" && '
                'export PYTHONPATH="/root/autodl-tmp/d-ser:${PYTHONPATH:-}" && '
                'export SER_C_BESD_PATH="/root/autodl-tmp/datasets/BESD/BESD/MY"; '
                'export SER_FAU_AIBO_PATH="/root/autodl-tmp/IS2009EmotionChallenge/wav"; '
                'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; '
                'nohup bash -lc \''
                'PYTHON_BIN=\"/root/miniconda3/bin/python\"; '
                'if [ ! -x \"$PYTHON_BIN\" ]; then PYTHON_BIN=\"$(command -v python3 || command -v python)\"; fi; '
                '\"$PYTHON_BIN\" -m src.train --train_data c-besd --pooling_type self_attention --exp_name exp1_self_attention && '
                '\"$PYTHON_BIN\" -m src.train --train_data c-besd --pooling_type prosody_guided --exp_name exp2_prosody_guided && '
                '\"$PYTHON_BIN\" -m src.train --train_data c-besd --test_data fau-aibo --pooling_type prosody_guided --exp_name exp4_zero_shot_fau'
                '\' > matrix_cbesd.log 2>&1 < /dev/null & echo $!'
            )
            code, out, err = _exec(client, missing_cmd)
            if code != 0:
                raise RuntimeError(f"Failed to launch missing C-BESD experiments: {err or out}")
            pid = out.strip().splitlines()[-1].strip() if out.strip() else ""
            print(f"[run-missing-cbesd] launched with PID: {pid}")
            return

        if args.auto_resume:
            logs_now = _list_remote_logs(client)
            missing = sorted(CORE_EXPERIMENT_LOGS - logs_now)
            running = _has_running_training(client)
            print(f"[auto] running={running}")
            print(f"[auto] logs_present={sorted(logs_now)}")
            print(f"[auto] logs_missing={missing}")

            if missing and not running:
                # Ensure missing C-BESD/zero-shot block is relaunched when idle.
                relaunch_cmd = (
                    f'cd "{REMOTE_PROJECT_ROOT}" && '
                    'export PYTHONPATH="/root/autodl-tmp/d-ser:${PYTHONPATH:-}" && '
                    'export SER_C_BESD_PATH="/root/autodl-tmp/datasets/BESD/BESD/MY"; '
                    'export SER_IEMOCAP_PATH="/root/autodl-tmp/IEMOCAP/wavs"; '
                    'export SER_FAU_AIBO_PATH="/root/autodl-tmp/IS2009EmotionChallenge/wav"; '
                    'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; '
                    'nohup bash -lc \''
                    'PYTHON_BIN=\"/root/miniconda3/bin/python\"; '
                    'if [ ! -x \"$PYTHON_BIN\" ]; then PYTHON_BIN=\"$(command -v python3 || command -v python)\"; fi; '
                    'if [ ! -f results/logs/exp1_self_attention.json ]; then \"$PYTHON_BIN\" -m src.train --train_data c-besd --pooling_type self_attention --exp_name exp1_self_attention; fi; '
                    'if [ ! -f results/logs/exp2_prosody_guided.json ]; then \"$PYTHON_BIN\" -m src.train --train_data c-besd --pooling_type prosody_guided --exp_name exp2_prosody_guided; fi; '
                    'if [ ! -f results/logs/exp3_adult_iemocap.json ]; then \"$PYTHON_BIN\" -m src.train --train_data iemocap --pooling_type prosody_guided --exp_name exp3_adult_iemocap; fi; '
                    'if [ ! -f results/logs/exp4_zero_shot_fau.json ]; then \"$PYTHON_BIN\" -m src.train --train_data c-besd --test_data fau-aibo --pooling_type prosody_guided --exp_name exp4_zero_shot_fau; fi; '
                    'if [ ! -f results/logs/exp5_fau_indomain.json ]; then \"$PYTHON_BIN\" -m src.train --train_data fau-aibo --pooling_type prosody_guided --exp_name exp5_fau_indomain --reg_profile fau; fi; '
                    'if [ ! -f results/logs/exp5b_self_attention_fau.json ]; then \"$PYTHON_BIN\" -m src.train --train_data fau-aibo --pooling_type self_attention --exp_name exp5b_self_attention_fau --reg_profile fau; fi; '
                    '\' > matrix_autoresume.log 2>&1 < /dev/null & echo $!'
                )
                code, out, err = _exec(client, relaunch_cmd)
                if code != 0:
                    raise RuntimeError(f"Failed to relaunch auto-resume job: {err or out}")
                pid = out.strip().splitlines()[-1].strip() if out.strip() else ""
                print(f"[auto] relaunched missing experiments, PID={pid}")

            # Always sync current results to local.
            code, _, _ = _exec(client, f"test -d {REMOTE_PROJECT_ROOT}/results/logs")
            if code == 0:
                _download_tree(
                    sftp,
                    f"{REMOTE_PROJECT_ROOT}/results/logs",
                    local_root / "results_remote" / "results" / "logs",
                )
            for candidate in [
                f"{REMOTE_PROJECT_ROOT}/publication_package/xai_raw_data.npz",
                f"{REMOTE_PROJECT_ROOT}/results/xai_raw_data.npz",
            ]:
                try:
                    sftp.stat(candidate)
                    target = local_root / "publication_package" / "xai_raw_data.npz"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(candidate, str(target))
                    print(f"[auto] synced xai_raw_data.npz from: {candidate}")
                    break
                except OSError:
                    continue
            return

        if args.debug_once:
            debug_cmd = (
                f"cd {REMOTE_PROJECT_ROOT} && "
                "set -x && bash run_matrix.sh"
            )
            code, out, err = _exec(client, debug_cmd)
            print(f"[debug] exit_code={code}")
            if out:
                print(out)
            if err:
                print(err)
            return

        if not args.download_only:
            print("[sync] uploading local code subset...")
            _upload_tree(
                sftp,
                local_root=local_root,
                remote_root=REMOTE_PROJECT_ROOT,
                include_dirs=["src"],
                include_files=[
                    "requirements.txt",
                    ".gitignore",
                    "scripts/tmp_paramiko_autodl_runner.py",
                    "docs/dataset_instructions.md",
                ],
            )

            fau_ready_cmd = (
                f'test -d "{REMOTE_FAU_EXTRACT_ROOT}/IS2009EmotionChallenge" '
                f'-o -d "{REMOTE_FAU_EXTRACT_ROOT}/wav"'
            )
            fau_ready, _, _ = _exec(client, fau_ready_cmd)
            fau_tar = local_root / "fau_aibo.tar.gz"
            if fau_ready == 0:
                print("[fau] remote FAU dataset already present; skipping tar extraction.")
            elif fau_tar.exists():
                print("[fau] uploading and extracting fau_aibo.tar.gz ...")
                remote_tar = f"{REMOTE_FAU_EXTRACT_ROOT}/fau_aibo.tar.gz"
                _ensure_remote_dir(sftp, REMOTE_FAU_EXTRACT_ROOT)
                sftp.put(str(fau_tar), remote_tar)
                code, out, err = _exec(
                    client,
                    f'mkdir -p "{REMOTE_FAU_EXTRACT_ROOT}" '
                    f'&& tar -xzf "{remote_tar}" -C "{REMOTE_FAU_EXTRACT_ROOT}"',
                )
                if code != 0:
                    raise RuntimeError(f"FAU extraction failed: {err or out}")
                print("[fau] extraction done.")
            else:
                print("[fau] local fau_aibo.tar.gz not found; assuming remote data is already prepared.")

            print("[remote] writing run_matrix.sh ...")
            _write_remote_runner(client, sftp)

        if not args.download_only:
            start_cmd = (
                f"cd {REMOTE_PROJECT_ROOT} && "
                "nohup bash run_matrix.sh > matrix_run.log 2>&1 < /dev/null & echo $!"
            )
            code, out, err = _exec(client, start_cmd)
            if code != 0:
                raise RuntimeError(f"Failed to start remote matrix: {err or out}")
            pid = out.strip().splitlines()[-1].strip() if out.strip() else ""
            print(f"Remote matrix started with PID: {pid}")
            check_cmd = (
                f"sleep 2; "
                f"if ps -p {pid} >/dev/null 2>&1; then "
                f"echo '[start-check] process alive'; "
                f"else echo '[start-check] process exited'; tail -n 80 {REMOTE_PROJECT_ROOT}/matrix_run.log || true; fi"
            )
            _, check_out, check_err = _exec(client, check_cmd)
            if check_out:
                print(check_out)
            if check_err:
                print(check_err)

        if not args.start_only:
            print("[download] syncing remote logs and XAI bundle...")
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
