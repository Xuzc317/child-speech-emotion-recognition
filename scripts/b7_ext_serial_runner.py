#!/usr/bin/env python3
"""B7-ext serial executor: 18 unfrozen transfer runs, one at a time.
Runs ON the AutoDL server. Checks best_epoch after each run, stops on failure.
Usage: python b7_ext_serial_runner.py [--resume-from N]
"""
import subprocess, time, json, sys, os, re
from datetime import datetime

REMOTE_ROOT = '/root/autodl-tmp/d-ser'
PYTHON_BIN = '/root/miniconda3/bin/python'
LOG_FILE = '/root/autodl-tmp/d-ser/b7_ext_execution.log'

# Parse --resume-from
resume_from = 0
args = sys.argv[1:]
for i, a in enumerate(args):
    if a == '--resume-from' and i+1 < len(args):
        resume_from = int(args[i+1])

def log(msg):
    line = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

# === Run definitions ===
runs = []
# E7-07: C-BESD -> FAU
for seed in [42, 123, 456]:
    runs.append(dict(exp=f'E7-07_s{seed}', train_data='fau-aibo', pooling='self_attention',
        ncls=4, reg='fau', ckpt='checkpoints/b1/E1-02_s42/best_model.pt',
        seed=seed, odir=f'checkpoints/b7_ext/E7-07_s{seed}'))
# E7-08: C-BESD -> IEMOCAP
for seed in [42, 123, 456]:
    runs.append(dict(exp=f'E7-08_s{seed}', train_data='iemocap', pooling='self_attention',
        ncls=4, reg='default', ckpt='checkpoints/b1/E1-02_s42/best_model.pt',
        seed=seed, odir=f'checkpoints/b7_ext/E7-08_s{seed}'))
# E7-09: FAU -> C-BESD
for seed in [42, 123, 456]:
    runs.append(dict(exp=f'E7-09_s{seed}', train_data='c-besd', pooling='self_attention',
        ncls=6, reg='default', ckpt='checkpoints/b1/E1-05_s42/best_model.pt',
        seed=seed, odir=f'checkpoints/b7_ext/E7-09_s{seed}'))
# E7-10: FAU -> IEMOCAP
for seed in [42, 123, 456]:
    runs.append(dict(exp=f'E7-10_s{seed}', train_data='iemocap', pooling='self_attention',
        ncls=4, reg='default', ckpt='checkpoints/b1/E1-05_s42/best_model.pt',
        seed=seed, odir=f'checkpoints/b7_ext/E7-10_s{seed}'))
# E7-11: IEMOCAP -> C-BESD
for seed in [42, 123, 456]:
    runs.append(dict(exp=f'E7-11_s{seed}', train_data='c-besd', pooling='self_attention',
        ncls=6, reg='default', ckpt='checkpoints/b1/E1-09_s42/best_model.pt',
        seed=seed, odir=f'checkpoints/b7_ext/E7-11_s{seed}'))
# E7-12: IEMOCAP -> FAU
for seed in [42, 123, 456]:
    runs.append(dict(exp=f'E7-12_s{seed}', train_data='fau-aibo', pooling='self_attention',
        ncls=4, reg='fau', ckpt='checkpoints/b1/E1-09_s42/best_model.pt',
        seed=seed, odir=f'checkpoints/b7_ext/E7-12_s{seed}'))

log('=== B7-ext serial execution START ===')
log(f'Total runs: {len(runs)}')
for i, r in enumerate(runs):
    log(f'  {i+1}. {r["exp"]}: {r["ckpt"]} -> {r["train_data"]} seed={r["seed"]}')

os.makedirs(f'{REMOTE_ROOT}/results/b7_ext', exist_ok=True)

results = []
for idx, r in enumerate(runs):
    if idx < resume_from:
        log(f'=== [{idx+1}/18] {r["exp"]}: SKIPPED (already completed) ===')
        continue
    log('')
    log(f'=== [{idx+1}/18] {r["exp"]}: {r["train_data"]} seed={r["seed"]} ===')

    train_cmd = (
        f'cd {REMOTE_ROOT} && '
        f'export PYTHONPATH={REMOTE_ROOT}:$PYTHONPATH && '
        f'export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY && '
        f'export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs && '
        f'export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav && '
        f'mkdir -p results/b7_ext && mkdir -p {r["odir"]} && '
        f'{PYTHON_BIN} -m src.train '
        f'--train_data {r["train_data"]} --pooling_type {r["pooling"]} '
        f'--num_classes {r["ncls"]} --reg_profile {r["reg"]} '
        f'--load_checkpoint {r["ckpt"]} '
        f'--unfreeze_ssl --ssl_lr 1e-5 --lr 3e-4 '
        f'--seed {r["seed"]} --data_split_seed 42 --epochs 100 --batch_size 8 --patience 15 '
        f'--exp_name {r["exp"]} --output_dir {r["odir"]}'
    )

    run_log = f'/tmp/{r["exp"]}.log'
    start_time = time.time()

    log(f'Launching...')
    proc = subprocess.Popen(
        ['bash', '-c', train_cmd],
        stdout=open(run_log, 'w'), stderr=subprocess.STDOUT,
        close_fds=True
    )
    pid = proc.pid
    log(f'PID={pid}, log={run_log}')

    # Wait for completion with timeout
    try:
        proc.wait(timeout=7200)  # 2h max
    except subprocess.TimeoutExpired:
        log(f'ERROR: {r["exp"]} timed out after 2h! Killing...')
        proc.kill()
        proc.wait()
        log('STOPPING due to timeout')
        sys.exit(1)

    elapsed = time.time() - start_time
    log(f'Completed in {elapsed/60:.1f} min (exit_code={proc.returncode})')

    # Check JSON output
    json_path = f'{REMOTE_ROOT}/results/logs/{r["exp"]}.json'
    if not os.path.exists(json_path):
        log(f'ERROR: JSON not found at {json_path}!')
        # Show tail of log
        try:
            with open(run_log) as f:
                tail = f.readlines()[-30:]
            log('Tail of training log:\n' + ''.join(tail))
        except:
            pass
        log('STOPPING due to missing JSON')
        sys.exit(1)

    with open(json_path) as f:
        result = json.load(f)

    best_epoch = result.get('best_epoch', -1)
    test_wa = result.get('test_wa', 0)
    test_uar = result.get('test_uar', 0)
    unfreeze = result.get('unfreeze_ssl', False)

    log(f'RESULT: best_epoch={best_epoch}, test_wa={test_wa:.4f}, test_uar={test_uar:.4f}, unfreeze_ssl={unfreeze}')

    if best_epoch < 0:  # only block negative (corrupted JSON); 0=fast convergence, ≥1=normal
        log(f'ERROR: Abnormal best_epoch={best_epoch} for {r["exp"]}!')
        try:
            with open(run_log) as f:
                tail = f.readlines()[-40:]
            log('Tail of training log:\n' + ''.join(tail))
        except:
            pass
        log('STOPPING due to abnormal best_epoch')
        sys.exit(1)

    results.append(dict(exp=r['exp'], test_wa=test_wa, test_uar=test_uar, best_epoch=best_epoch))

log('')
log('=== ALL 18 RUNS COMPLETED ===')
for r in results:
    log(f'  {r["exp"]}: WA={r["test_wa"]:.4f}, UAR={r["test_uar"]:.4f}, best_epoch={r["best_epoch"]}')
log('DONE')
