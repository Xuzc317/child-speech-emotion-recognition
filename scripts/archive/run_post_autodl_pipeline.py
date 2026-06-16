"""Poll AutoDL until AC resume completes, then run post_autodl checklist Phases 0-5.

Usage:
  python scripts/run_post_autodl_pipeline.py
  python scripts/run_post_autodl_pipeline.py --skip-poll   # cloud already done
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import tmp_paramiko_autodl_runner as autodl  # noqa: E402

REMOTE = autodl.REMOTE_PROJECT_ROOT
POLL_SEC = 120
MAX_WAIT_SEC = 8 * 3600
SSH_RETRIES = 5


def _connect_retry():
    last_err = None
    for i in range(SSH_RETRIES):
        try:
            return autodl._connect()
        except Exception as e:
            last_err = e
            wait = min(60 * (i + 1), 300)
            print(f"[poll] SSH failed ({e}); retry in {wait}s ({i + 1}/{SSH_RETRIES})")
            time.sleep(wait)
    raise last_err


def _run(cmd: list[str], desc: str) -> None:
    print(f"\n>>> {desc}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"Failed: {' '.join(cmd)} (exit {r.returncode})")


CM_MAP = {
    "confusion_exp1_self_attention": "fig07_confusion_exp1_selfattn",
    "confusion_exp2_prosody_guided": "fig08_confusion_exp2_prosody",
    "confusion_exp3_adult_iemocap": "figA1_confusion_exp3_iemocap",
    "confusion_exp4_zero_shot_fau": "figA2_confusion_exp4_zero_shot",
    "confusion_exp5_fau_indomain": "figA3_confusion_exp5_fau_prosody",
    "confusion_exp5b_self_attention_fau": "figA4_confusion_exp5b_fau_selfattn",
}

ARCHIVE_DIRS = [
    ROOT / "results" / "logs",
    ROOT / "publication_package",
    ROOT / "paper_draft" / "figures",
    ROOT / "paper_draft",
]


def _remote_done(client) -> bool:
    cmd = (
        f'cd "{REMOTE}" && '
        "(ps -ef | grep 'src.train\\|autodl_ac_suite' | grep -v grep >/dev/null && echo RUNNING || echo IDLE) && "
        "(grep -q 'AC SUITE RESUME DONE' ac_suite_logs/resume_*.log 2>/dev/null && echo DONE || echo PENDING)"
    )
    _, out, _ = autodl._exec(client, cmd)
    lines = out.strip().splitlines()
    idle = any("IDLE" in x for x in lines)
    done = any("DONE" in x for x in lines)
    return idle and done


def poll_until_done() -> None:
    print("[poll] waiting for AutoDL AC resume...")
    deadline = time.time() + MAX_WAIT_SEC
    while time.time() < deadline:
        try:
            client = _connect_retry()
        except Exception as e:
            print(f"[poll] cannot reach AutoDL: {e}")
            time.sleep(POLL_SEC)
            continue
        try:
            if _remote_done(client):
                print("[poll] remote suite finished.")
                return
            _, out, _ = autodl._exec(
                client,
                f"tail -n 3 $(ls -t {REMOTE}/ac_suite_logs/resume_*.log 2>/dev/null | head -1) 2>/dev/null",
            )
            print(f"[poll] still running... {out.strip()[-120:] if out.strip() else ''}")
        finally:
            client.close()
        time.sleep(POLL_SEC)
    raise TimeoutError("AutoDL suite did not finish within MAX_WAIT_SEC")


def archive_to_last() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for base in ARCHIVE_DIRS:
        last = base / "last" / ts
        if not base.exists():
            continue
        moved = 0
        for p in list(base.iterdir()):
            if p.name == "last" or p.is_dir() and p.name.startswith("."):
                continue
            if base.name == "figures" and p.suffix not in (".png", ".pdf", ".json"):
                continue
            if base.name == "paper_draft" and p.suffix not in (".docx",):
                continue
            if base.name == "logs" and p.suffix != ".json":
                continue
            last.mkdir(parents=True, exist_ok=True)
            dst = last / p.name
            if dst.exists():
                continue
            shutil.move(str(p), str(dst))
            moved += 1
        if moved:
            print(f"[archive] {base} -> {last} ({moved} files)")


def sync_confusion_figures() -> None:
    src_dir = ROOT / "results_remote" / "results" / "figures"
    if not src_dir.exists():
        src_dir = ROOT / "results" / "figures"
    out = ROOT / "paper_draft" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for stem, new_name in CM_MAP.items():
        for ext in ("png", "pdf", "json"):
            s = src_dir / f"{stem}.{ext}"
            if s.exists():
                shutil.copy2(s, out / f"{new_name}.{ext}")
                print(f"[cm] {s.name} -> {new_name}.{ext}")


def write_data_freeze() -> None:
    logs = ROOT / "results" / "logs"
    rec = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "ac_suite_2026-05",
        "source": "results/logs after merge_remote_logs",
        "experiments": {},
    }
    for name in autodl.CORE_EXPERIMENT_LOGS:
        p = logs / name
        if p.exists():
            rec["experiments"][name] = json.loads(p.read_text(encoding="utf-8"))
    ms = logs / "fau_multiseed_summary.json"
    if ms.exists():
        rec["fau_multiseed"] = json.loads(ms.read_text(encoding="utf-8"))
    (logs / "DATA_FREEZE.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("[freeze] wrote results/logs/DATA_FREEZE.json")


def update_figures_manifest() -> None:
    manifest = {
        "generated_by": "scripts/run_post_autodl_pipeline.py",
        "protocol": "ac_suite_2026-05",
        "figures": [
            {"file": "fig00_system_architecture", "use": "System pipeline"},
            {"file": "fig01_main_matrix_wa_uar", "use": "Main results overview"},
            {"file": "fig02_fau_prosody_vs_selfattn", "use": "FAU pooling compare"},
            {"file": "fig03_fd_vs_accuracy", "use": "FD vs accuracy"},
            {"file": "fig04_xai_saliency_triple", "use": "XAI saliency"},
            {"file": "fig05_cbesd_selfattn_vs_prosody", "use": "C-BESD bar compare"},
            {"file": "fig06_layer_fusion_weights", "use": "Layer 8 fusion weights"},
            {"file": "fig07_confusion_exp1_selfattn", "use": "Main text CM Exp1"},
            {"file": "fig08_confusion_exp2_prosody", "use": "Main text CM Exp2"},
            {"file": "figA1_confusion_exp3_iemocap", "use": "Appendix CM Exp3"},
            {"file": "figA2_confusion_exp4_zero_shot", "use": "Appendix CM Exp4"},
            {"file": "figA3_confusion_exp5_fau_prosody", "use": "Appendix CM Exp5"},
            {"file": "figA4_confusion_exp5b_fau_selfattn", "use": "Appendix CM Exp5b"},
        ],
        "placeholder_note": "Replace PNG/PDF with final scientific figures; keep filenames.",
    }
    path = ROOT / "paper_draft" / "figures" / "FIGURES_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def patch_experiments_tex() -> None:
    path = ROOT / "paper_draft" / "4_Experiments_and_Results.tex"
    text = path.read_text(encoding="utf-8")
    marker = "% ────────────────────────────────────────────────────────────\n\\subsection{Convergence"
    insert = r"""
% ────────────────────────────────────────────────────────────
\subsection{Confusion Structure on C-BESD}
\label{ssec:confusion_cbesd}

Figure~\ref{fig:cm_exp1} and Figure~\ref{fig:cm_exp2} show test-set confusion matrices for the two pooling variants on acted child speech (seed=42, speaker-disjoint split).
Self-attention (Exp1) achieves higher diagonal dominance on \textit{happy} and \textit{neutral}; prosody-guided pooling (Exp2) remains competitive with faster convergence (epoch 10 vs.\ 30).
Additional matrices for IEMOCAP, zero-shot FAU, and in-domain FAU are provided in the appendix (Figures~\ref{fig:cm_a1}--\ref{fig:cm_a4}).

\begin{figure}[t]
  \centering
  \begin{minipage}{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/fig07_confusion_exp1_selfattn.pdf}
    \caption{Exp1: Self-attention on C-BESD.}
    \label{fig:cm_exp1}
  \end{minipage}\hfill
  \begin{minipage}{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/fig08_confusion_exp2_prosody.pdf}
    \caption{Exp2: Prosody-guided on C-BESD.}
    \label{fig:cm_exp2}
  \end{minipage}
\end{figure}

\subsection{FAU Multi-Seed Robustness}
\label{ssec:fau_multiseed}

On spontaneous FAU Aibo we report mean$\pm$std over seeds $\{42,123,456\}$ under the \textit{fau} regularization profile (see \texttt{fau\_multiseed\_summary.json}).
This complements the single-seed main table (Exp5/5b) without replacing it.

"""
    if "ssec:confusion_cbesd" not in text and marker in text:
        text = text.replace(marker, insert + marker)
        path.write_text(text, encoding="utf-8")
        print("[patch] 4_Experiments_and_Results.tex")


def patch_main_appendix() -> None:
    path = ROOT / "paper_draft" / "main.tex"
    text = path.read_text(encoding="utf-8")
    appendix = r"""
\appendix
\section{Supplementary Confusion Matrices}
\label{app:confusion}

\begin{figure}[h]
  \centering
  \includegraphics[width=0.72\linewidth]{figures/figA1_confusion_exp3_iemocap.pdf}
  \caption{Exp3: IEMOCAP in-domain (prosody-guided).}
  \label{fig:cm_a1}
\end{figure}
\begin{figure}[h]
  \centering
  \includegraphics[width=0.72\linewidth]{figures/figA2_confusion_exp4_zero_shot.pdf}
  \caption{Exp4: Zero-shot C-BESD$\rightarrow$FAU Aibo.}
  \label{fig:cm_a2}
\end{figure}
\begin{figure}[h]
  \centering
  \includegraphics[width=0.72\linewidth]{figures/figA3_confusion_exp5_fau_prosody.pdf}
  \caption{Exp5: FAU in-domain (prosody-guided).}
  \label{fig:cm_a3}
\end{figure}
\begin{figure}[h]
  \centering
  \includegraphics[width=0.72\linewidth]{figures/figA4_confusion_exp5b_fau_selfattn.pdf}
  \caption{Exp5b: FAU in-domain (self-attention).}
  \label{fig:cm_a4}
\end{figure}

"""
    if "\\appendix" not in text:
        text = text.replace(
            "\\bibliographystyle{IEEEtran}",
            appendix + "\n\\bibliographystyle{IEEEtran}",
        )
        path.write_text(text, encoding="utf-8")
        print("[patch] main.tex appendix")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-poll", action="store_true")
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help="Use existing results_remote (AutoDL already off or data synced)",
    )
    args = parser.parse_args()

    if not args.skip_poll:
        poll_until_done()

    if not args.skip_pull:
        _run([sys.executable, "scripts/tmp_paramiko_autodl_runner.py", "--pull-all"], "Phase 0: pull-all")
    _run([sys.executable, "scripts/merge_remote_logs.py"], "Phase 0: merge logs")
    _run([sys.executable, "scripts/aggregate_fau_multiseed.py"], "Phase 0: multiseed summary")
    _run([sys.executable, "scripts/verify_experiment_jsons.py"], "Phase 1: verify JSONs")

    sync_confusion_figures()
    _run([sys.executable, "scripts/generate_paper_figures.py"], "Phase 2: placeholder figures")
    archive_to_last()

    patch_experiments_tex()
    patch_main_appendix()
    update_figures_manifest()
    write_data_freeze()

    if (ROOT / "scripts" / "generate_docx.py").exists():
        _run([sys.executable, "scripts/generate_docx.py"], "Phase 4: English docx")
    if (ROOT / "scripts" / "generate_docx_cn.py").exists():
        _run([sys.executable, "scripts/generate_docx_cn.py"], "Phase 4: Chinese docx")
    if (ROOT / "scripts" / "build_submission_bundle.py").exists():
        _run([sys.executable, "scripts/build_submission_bundle.py"], "Phase 4: submission bundle")

    print("\n=== POST-AUTODL PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
