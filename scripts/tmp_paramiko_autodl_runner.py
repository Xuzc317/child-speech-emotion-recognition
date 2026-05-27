"""Temporary Paramiko runner for AutoDL experiments.

This script follows the user-specified SSH override:
  - Connects using paramiko.SSHClient + AutoAddPolicy
  - Uses exec_command() with nohup to launch remote jobs
  - Uses open_sftp() to upload code and download results
"""

from __future__ import annotations

import argparse
import posixpath
import time
from pathlib import Path
from stat import S_ISDIR

import paramiko


REMOTE_HOST = "connect.cqa1.seetacloud.com"
REMOTE_PORT = 14393  # cloned instance 2026-05-26
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


def _download_file_if_exists(
    sftp: paramiko.SFTPClient, remote_path: str, local_path: Path
) -> bool:
    try:
        sftp.stat(remote_path)
    except OSError:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_path, str(local_path))
    print(f"[pull] file {remote_path} -> {local_path}")
    return True


def _download_dir_if_exists(
    sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path
) -> bool:
    try:
        sftp.listdir_attr(remote_dir)
    except OSError:
        return False
    _download_tree(sftp, remote_dir, local_dir)
    print(f"[pull] dir  {remote_dir} -> {local_dir}")
    return True


def _remote_inventory(client: paramiko.SSHClient) -> str:
    inv_cmd = (
        f'cd "{REMOTE_PROJECT_ROOT}" && '
        "echo '=== results/logs ===' && ls -la results/logs 2>/dev/null || true && "
        "echo '=== checkpoints ===' && ls -laR checkpoints 2>/dev/null | head -80 && "
        "echo '=== training logs ===' && ls -la matrix*.log exp2_checkpoint.log 2>/dev/null || true && "
        "echo '=== results/figures ===' && ls -la results/figures 2>/dev/null || true && "
        "echo '=== results aux ===' && ls -la results/layer_weights.json results/distribution_shift.json 2>/dev/null || true && "
        "echo '=== publication_package ===' && ls -la publication_package 2>/dev/null | head -30 && "
        "echo '=== optional datasets ===' && "
        "for d in /root/autodl-tmp/datasets/myst /root/autodl-tmp/datasets/kidstalc; do "
        '  [ -d \"$d\" ] && echo \"present: $d\" || echo \"missing: $d\"; '
        "done"
    )
    _, out, err = _exec(client, inv_cmd)
    return (out or "") + (err or "")


def _pull_all_from_remote(
    client: paramiko.SSHClient, sftp: paramiko.SFTPClient, local_root: Path
) -> None:
    """Download logs, checkpoints, training logs, figures, and aux artifacts."""
    remote_base = REMOTE_PROJECT_ROOT
    stash = local_root / "results_remote"
    stash.mkdir(parents=True, exist_ok=True)

    inventory = _remote_inventory(client)
    inv_path = stash / "remote_inventory.txt"
    inv_path.write_text(inventory, encoding="utf-8")
    print(f"[pull] inventory -> {inv_path}")
    print(inventory[:2500])
    if len(inventory) > 2500:
        print(f"... ({len(inventory)} chars total, see {inv_path})")

    _download_dir_if_exists(
        sftp, f"{remote_base}/results/logs", stash / "results" / "logs"
    )

    for aux in ("layer_weights.json", "distribution_shift.json"):
        _download_file_if_exists(
            sftp,
            f"{remote_base}/results/{aux}",
            stash / "results" / aux,
        )

    _download_dir_if_exists(
        sftp, f"{remote_base}/results/figures", stash / "results" / "figures"
    )

    _download_dir_if_exists(
        sftp, f"{remote_base}/checkpoints", local_root / "checkpoints" / "autodl"
    )

    train_logs = stash / "training_logs"
    train_logs.mkdir(parents=True, exist_ok=True)
    for name in (
        "matrix_run.log",
        "matrix_cbesd.log",
        "matrix_autoresume.log",
        "exp2_checkpoint.log",
    ):
        _download_file_if_exists(
            sftp, f"{remote_base}/{name}", train_logs / name
        )

    pub_remote = f"{remote_base}/publication_package"
    pub_local = local_root / "publication_package"
    try:
        for entry in sftp.listdir_attr(pub_remote):
            if entry.st_mode and S_ISDIR(entry.st_mode):
                continue
            _download_file_if_exists(
                sftp,
                posixpath.join(pub_remote, entry.filename),
                pub_local / entry.filename,
            )
    except OSError:
        print(f"[pull] skip missing: {pub_remote}")

    for candidate in (
        f"{remote_base}/publication_package/xai_raw_data.npz",
        f"{remote_base}/results/xai_raw_data.npz",
    ):
        target = pub_local / "xai_raw_data.npz"
        if _download_file_if_exists(sftp, candidate, target):
            break

    fig_src = stash / "results" / "figures"
    if fig_src.exists():
        fig_dst = local_root / "paper_draft" / "figures"
        fig_dst.mkdir(parents=True, exist_ok=True)
        for p in fig_src.glob("*"):
            if p.is_file():
                dst = fig_dst / p.name
                if not dst.exists() or p.stat().st_mtime > dst.stat().st_mtime:
                    dst.write_bytes(p.read_bytes())
                    print(f"[pull] figure copy -> {dst.name}")


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


def _remote_python_setup() -> str:
    """Shell snippet: cd project, PYTHONPATH, PYTHON_BIN, dataset paths."""
    return (
        f'cd "{REMOTE_PROJECT_ROOT}" && '
        f'export PYTHONPATH="{REMOTE_PROJECT_ROOT}:${{PYTHONPATH:-}}" && '
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && '
        'PYTHON_BIN="$(command -v python || true)"; '
        'if [ -z "$PYTHON_BIN" ]; then PYTHON_BIN="$(command -v python3 || true)"; fi; '
        'if [ -z "$PYTHON_BIN" ] && [ -x "/root/miniconda3/bin/python" ]; then PYTHON_BIN="/root/miniconda3/bin/python"; fi; '
        'if [ -z "$PYTHON_BIN" ] && [ -x "/opt/conda/bin/python" ]; then PYTHON_BIN="/opt/conda/bin/python"; fi; '
        'export SER_C_BESD_PATH="${SER_C_BESD_PATH:-$( '
        '  for p in /root/autodl-tmp/datasets/BESD/BESD/MY /root/autodl-tmp/BESD/BESD/MY; do '
        '    [ -d "$p" ] && echo "$p" && break; '
        '  done)}"; '
        'export SER_FAU_AIBO_PATH="${SER_FAU_AIBO_PATH:-$( '
        '  for p in /root/autodl-tmp/IS2009EmotionChallenge/wav '
        '/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/IS2009EmotionChallenge/wav; do '
        '    [ -d "$p" ] && echo "$p" && break; '
        '  done)}"; '
    )


def _remote_path_exists(sftp: paramiko.SFTPClient, client: paramiko.SSHClient, path: str) -> bool:
    try:
        sftp.stat(path)
        return True
    except OSError:
        code, _, _ = _exec(client, f"test -e {path}")
        return code == 0


def _run_exp2_checkpoint(
    client: paramiko.SSHClient,
    sftp: paramiko.SFTPClient,
    local_root: Path,
    *,
    train: bool = True,
) -> None:
    """Train Exp2 with per-exp output_dir, plot confusion matrix on remote, sync locally."""
    ckpt_remote = f"{REMOTE_PROJECT_ROOT}/checkpoints/exp2_prosody_guided/best_model.pt"
    log_remote = f"{REMOTE_PROJECT_ROOT}/exp2_checkpoint.log"
    fig_remote = f"{REMOTE_PROJECT_ROOT}/results/figures/exp2_confusion"

    if train:
        train_cmd = (
            f'cd "{REMOTE_PROJECT_ROOT}" && '
            f'export PYTHONPATH="{REMOTE_PROJECT_ROOT}:${{PYTHONPATH:-}}" && '
            'export SER_C_BESD_PATH="/root/autodl-tmp/datasets/BESD/BESD/MY"; '
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; '
            'mkdir -p checkpoints/exp2_prosody_guided; '
            'nohup bash -lc \''
            'PYTHON_BIN=\"/root/miniconda3/bin/python\"; '
            'if [ ! -x \"$PYTHON_BIN\" ]; then PYTHON_BIN=\"$(command -v python3 || command -v python)\"; fi; '
            '\"$PYTHON_BIN\" -m src.train --train_data c-besd --pooling_type prosody_guided '
            '--exp_name exp2_prosody_guided --output_dir checkpoints/exp2_prosody_guided'
            f'\' > {log_remote} 2>&1 < /dev/null & echo $!'
        )
        code, out, err = _exec(client, train_cmd)
        if code != 0:
            raise RuntimeError(f"Failed to launch Exp2: {err or out}")
        pid = out.strip().splitlines()[-1].strip() if out.strip() else "?"
        print(f"[exp2] training launched PID={pid}, log={log_remote}")

        deadline = time.time() + 3600
        polls = 0
        while time.time() < deadline:
            time.sleep(30)
            polls += 1
            _, tail, _ = _exec(client, f"tail -n 8 {log_remote} 2>/dev/null || true")
            ckpt_ok = _remote_path_exists(sftp, client, ckpt_remote)
            _, result_line, _ = _exec(
                client, f"grep 'RESULT:.*exp2_prosody_guided' {log_remote} | tail -n 1"
            )
            has_result = "RESULT:" in result_line and "exp2_prosody_guided" in result_line
            running = _has_running_training(client)
            print(
                f"[exp2] poll #{polls}: ckpt={'yes' if ckpt_ok else 'no'} "
                f"running={running} result={'yes' if has_result else 'no'}"
            )
            if tail.strip():
                print(tail.strip().splitlines()[-1][:160])
            if has_result and not running:
                if not ckpt_ok:
                    for _ in range(6):
                        time.sleep(5)
                        ckpt_ok = _remote_path_exists(sftp, client, ckpt_remote)
                        if ckpt_ok:
                            break
                if ckpt_ok:
                    print(f"[exp2] training finished: {result_line.strip()[:200]}")
                    break
                raise RuntimeError(f"RESULT logged but checkpoint missing: {ckpt_remote}")
            if not running and polls >= 2 and not ckpt_ok and not has_result:
                _, full_tail, _ = _exec(client, f"tail -n 40 {log_remote}")
                raise RuntimeError(f"Exp2 exited without checkpoint:\n{full_tail}")
        else:
            raise RuntimeError("Exp2 training timed out after 3600s")
    else:
        if not _remote_path_exists(sftp, client, ckpt_remote):
            raise RuntimeError(f"Checkpoint not found for post-only: {ckpt_remote}")
        print("[exp2] post-only: using existing checkpoint")

    plot_cmd = (
        f'cd "{REMOTE_PROJECT_ROOT}" && '
        f'export PYTHONPATH="{REMOTE_PROJECT_ROOT}:${{PYTHONPATH:-}}" && '
        'export SER_C_BESD_PATH="/root/autodl-tmp/datasets/BESD/BESD/MY"; '
        'PYTHON_BIN="/root/miniconda3/bin/python"; '
        'if [ ! -x "$PYTHON_BIN" ]; then PYTHON_BIN="$(command -v python3 || command -v python)"; fi; '
        'mkdir -p results/figures; '
        f'"$PYTHON_BIN" scripts/plot_confusion_matrix.py '
        f'--checkpoint checkpoints/exp2_prosody_guided/best_model.pt '
        f'--train_data c-besd --pooling_type prosody_guided '
        f'--output {fig_remote}'
    )
    print("[exp2] plotting confusion matrix on remote...")
    code, out, err = _exec(client, plot_cmd)
    print(out or err)
    if code != 0:
        raise RuntimeError(f"Remote confusion plot failed: {err or out}")

    local_ckpt = local_root / "checkpoints" / "autodl" / "exp2_prosody_guided"
    local_ckpt.parent.mkdir(parents=True, exist_ok=True)
    _download_tree(sftp, f"{REMOTE_PROJECT_ROOT}/checkpoints/exp2_prosody_guided", local_ckpt)
    print(f"[exp2] checkpoint -> {local_ckpt}")

    local_fig = local_root / "paper_draft" / "figures"
    local_fig.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "json"):
        try:
            sftp.get(f"{fig_remote}.{ext}", str(local_fig / f"fig07_confusion_exp2_prosody.{ext}"))
            print(f"[exp2] figure -> fig07_confusion_exp2_prosody.{ext}")
        except OSError:
            pass

    try:
        sftp.get(log_remote, str(local_root / "results_remote" / "exp2_checkpoint.log"))
    except OSError:
        pass


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
            '  "$PYTHON_BIN" -m src.train --train_data c-besd --pooling_type self_attention --exp_name exp1_self_attention --output_dir checkpoints/exp1_self_attention',
            '  "$PYTHON_BIN" -m src.train --train_data c-besd --pooling_type prosody_guided --exp_name exp2_prosody_guided --output_dir checkpoints/exp2_prosody_guided',
            "else",
            '  echo "[skip] Exp1/Exp2 skipped: C-BESD path unavailable."',
            "fi",
            "",
            'if [ -n "${SER_IEMOCAP_PATH:-}" ]; then',
            '  "$PYTHON_BIN" -m src.train --train_data iemocap --pooling_type prosody_guided --exp_name exp3_adult_iemocap --output_dir checkpoints/exp3_adult_iemocap',
            "else",
            '  echo "[skip] Exp3 skipped: IEMOCAP path unavailable."',
            "fi",
            "",
            'if [ -n "${SER_C_BESD_PATH:-}" ] && [ -n "${SER_FAU_AIBO_PATH:-}" ]; then',
            '  "$PYTHON_BIN" -m src.train --train_data c-besd --test_data fau-aibo --pooling_type prosody_guided --exp_name exp4_zero_shot_fau --output_dir checkpoints/exp4_zero_shot_fau',
            "else",
            '  echo "[skip] Exp4 skipped: C-BESD or FAU path unavailable."',
            "fi",
            "",
            'if [ -n "${SER_FAU_AIBO_PATH:-}" ]; then',
            '  "$PYTHON_BIN" -m src.train --train_data fau-aibo --pooling_type prosody_guided --exp_name exp5_fau_indomain --reg_profile fau --output_dir checkpoints/exp5_fau_indomain',
            '  "$PYTHON_BIN" -m src.train --train_data fau-aibo --pooling_type self_attention --exp_name exp5b_self_attention_fau --reg_profile fau --output_dir checkpoints/exp5b_self_attention_fau',
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
    parser.add_argument(
        "--sync-checkpoints",
        action="store_true",
        help="Download remote checkpoints/ to local checkpoints/autodl/",
    )
    parser.add_argument("--probe", action="store_true", help="Test SSH connectivity and exit")
    parser.add_argument(
        "--run-exp2-checkpoint",
        action="store_true",
        help="Re-run Exp2 with output_dir, plot confusion matrix, sync checkpoint locally",
    )
    parser.add_argument(
        "--exp2-post-only",
        action="store_true",
        help="Skip Exp2 training; plot confusion matrix and sync existing checkpoint",
    )
    parser.add_argument(
        "--pull-all",
        action="store_true",
        help="Download logs, checkpoints, training logs, figures, XAI (no upload/train)",
    )
    parser.add_argument(
        "--run-ac-suite",
        action="store_true",
        help="Upload code + launch scripts/autodl_ac_suite.sh on AutoDL (nohup)",
    )
    parser.add_argument(
        "--ac-suite-status",
        action="store_true",
        help="Tail ac_suite log and list checkpoints (no train)",
    )
    parser.add_argument(
        "--run-ac-suite-resume",
        action="store_true",
        help="Resume CM + multiseed + C2 after A1 training (fix CM bug)",
    )
    args = parser.parse_args()

    local_root = Path(args.local_root).resolve()
    print("[connect] opening SSH connection...")
    client = _connect()
    print("[connect] connected.")
    sftp = client.open_sftp()

    try:
        if args.run_exp2_checkpoint or args.exp2_post_only:
            print("[sync] uploading src + plot script...")
            _upload_tree(
                sftp,
                local_root=local_root,
                remote_root=REMOTE_PROJECT_ROOT,
                include_dirs=["src"],
                include_files=["scripts/plot_confusion_matrix.py"],
            )
            _run_exp2_checkpoint(
                client, sftp, local_root, train=not args.exp2_post_only
            )
            return

        if args.probe:
            code, out, err = _exec(client, f"echo OK && ls -la {REMOTE_PROJECT_ROOT} 2>/dev/null | head -5")
            print("probe:", "REACHABLE" if code == 0 else "FAILED")
            print(out or err)
            return

        if args.pull_all:
            _pull_all_from_remote(client, sftp, local_root)
            return

        if args.ac_suite_status:
            status_cmd = (
                f'cd "{REMOTE_PROJECT_ROOT}" && '
                "echo '====PROCESS====' && "
                "(ps -ef | grep -E 'autodl_ac_suite|src.train' | grep -v grep || true) && "
                "echo '====SUITE_LOG====' && "
                "(ls -lt ac_suite_logs/*.log 2>/dev/null | head -3 || true) && "
                "(L=$(ls -t ac_suite_logs/*.log 2>/dev/null | head -1); "
                '[ -n "$L" ] && tail -n 25 "$L" || echo "no suite log") && '
                "echo '====CHECKPOINTS====' && "
                "(ls -la checkpoints/*/best_model.pt 2>/dev/null || true)"
            )
            _, out, err = _exec(client, status_cmd)
            print(out or err)
            return

        if args.run_ac_suite:
            print("[ac-suite] uploading code...")
            _upload_tree(
                sftp,
                local_root=local_root,
                remote_root=REMOTE_PROJECT_ROOT,
                include_dirs=["src"],
                include_files=[
                    "scripts/autodl_ac_suite.sh",
                    "scripts/plot_confusion_matrix.py",
                    "scripts/run_prosody_diagnostics.py",
                    "scripts/aggregate_fau_multiseed.py",
                    "requirements.txt",
                ],
            )
            remote_sh = f"{REMOTE_PROJECT_ROOT}/scripts/autodl_ac_suite.sh"
            _exec(client, f"chmod +x {remote_sh}")
            launch = (
                f'cd "{REMOTE_PROJECT_ROOT}" && '
                f'export PYTHONPATH="{REMOTE_PROJECT_ROOT}:${{PYTHONPATH:-}}" && '
                'export SER_C_BESD_PATH="/root/autodl-tmp/datasets/BESD/BESD/MY"; '
                'export SER_IEMOCAP_PATH="/root/autodl-tmp/IEMOCAP/wavs"; '
                'export SER_FAU_AIBO_PATH="/root/autodl-tmp/IS2009EmotionChallenge/wav"; '
                'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; '
                'nohup bash -lc \''
                'PYTHON_BIN="/root/miniconda3/bin/python"; '
                'if [ ! -x "$PYTHON_BIN" ]; then PYTHON_BIN="$(command -v python3 || command -v python)"; fi; '
                'export PYTHON_BIN; '
                'bash scripts/autodl_ac_suite.sh'
                '\' > ac_suite_launcher.log 2>&1 < /dev/null & echo $!'
            )
            code, out, err = _exec(client, launch)
            if code != 0:
                raise RuntimeError(f"AC suite launch failed: {err or out}")
            pid = out.strip().splitlines()[-1].strip() if out.strip() else "?"
            print(f"[ac-suite] launched PID={pid}")
            print("[ac-suite] monitor: python scripts/tmp_paramiko_autodl_runner.py --ac-suite-status")
            print("[ac-suite] when done: python scripts/tmp_paramiko_autodl_runner.py --pull-all")
            return

        if args.run_ac_suite_resume:
            print("[ac-suite-resume] uploading fixed scripts...")
            _upload_tree(
                sftp,
                local_root=local_root,
                remote_root=REMOTE_PROJECT_ROOT,
                include_dirs=["src"],
                include_files=[
                    "scripts/autodl_ac_suite_resume.sh",
                    "scripts/plot_confusion_matrix.py",
                    "scripts/run_prosody_diagnostics.py",
                    "scripts/aggregate_fau_multiseed.py",
                ],
            )
            remote_sh = f"{REMOTE_PROJECT_ROOT}/scripts/autodl_ac_suite_resume.sh"
            _exec(client, f"chmod +x {remote_sh}")
            launch = (
                f'cd "{REMOTE_PROJECT_ROOT}" && '
                'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; '
                'nohup bash -lc \''
                'PYTHON_BIN="/root/miniconda3/bin/python"; '
                'if [ ! -x "$PYTHON_BIN" ]; then PYTHON_BIN="$(command -v python3 || command -v python)"; fi; '
                'export PYTHON_BIN; '
                'bash scripts/autodl_ac_suite_resume.sh'
                '\' > ac_suite_resume_launcher.log 2>&1 < /dev/null & echo $!'
            )
            code, out, err = _exec(client, launch)
            if code != 0:
                raise RuntimeError(f"AC suite resume failed: {err or out}")
            pid = out.strip().splitlines()[-1].strip() if out.strip() else "?"
            print(f"[ac-suite-resume] launched PID={pid}")
            return

        if args.sync_checkpoints:
            remote_ckpt = f"{REMOTE_PROJECT_ROOT}/checkpoints"
            local_ckpt = local_root / "checkpoints" / "autodl"
            code, _, _ = _exec(client, f"test -d {remote_ckpt}")
            if code != 0:
                print(f"[sync-checkpoints] remote dir missing: {remote_ckpt}")
            else:
                print(f"[sync-checkpoints] downloading {remote_ckpt} -> {local_ckpt}")
                _download_tree(sftp, remote_ckpt, local_ckpt)
                print("[sync-checkpoints] done.")
            return

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
            code_ckpt, _, _ = _exec(client, f"test -d {REMOTE_PROJECT_ROOT}/checkpoints")
            if code_ckpt == 0:
                _download_tree(
                    sftp,
                    f"{REMOTE_PROJECT_ROOT}/checkpoints",
                    local_root / "checkpoints" / "autodl",
                )
                print("[auto] synced checkpoints/ -> checkpoints/autodl/")
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
