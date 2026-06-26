#!/usr/bin/env python3
"""B7-ext autonomous monitor — polls AutoDL, logs locally, auto-decides."""
import paramiko, time, json, sys, os, re
from datetime import datetime

LOCAL_LOG = r"D:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\b7_ext_monitor.log"
REMOTE_EXEC_LOG = "/root/autodl-tmp/d-ser/b7_ext_execution.log"
HOST = "connect.cqa1.seetacloud.com"
PORT = 25808
USER = "root"
PASS = "9HmcVfCXUFVD"

DECISIONS = []

def log(msg):
    line = f"[{datetime.now().strftime('%m-%d %H:%M')}] {msg}"
    print(line, flush=True)
    with open(LOCAL_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def decide(issue, action, reason):
    d = f"[DECISION] {issue}: {action} — {reason}"
    DECISIONS.append(d)
    log(d)

def connect(retries=3):
    for attempt in range(retries):
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=HOST, port=PORT, username=USER, password=PASS,
                look_for_keys=False, allow_agent=False,
                disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']},
                timeout=15, banner_timeout=20, auth_timeout=15)
            return client
        except Exception as e:
            log(f'SSH retry {attempt+1}/{retries}: {e}')
            time.sleep(10)
    return None

def exec_cmd(client, cmd):
    stdin, stdout, stderr = client.exec_command(cmd)
    code = stdout.channel.recv_exit_status()
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    return code, out, err

def get_completed_runs(client):
    """Count unique completed experiments by finding 'Completed in' lines."""
    _, out, _ = exec_cmd(client, f"grep -c 'Completed in' {REMOTE_EXEC_LOG} 2>/dev/null || echo 0")
    return int(out.strip().split()[0])

log('=== B7-ext autonomous monitor START ===')

# Record decision #1
decide('best_epoch=1 is normal', 'accept as valid',
       'E7-01/05 frozen runs also have best_epoch=1; transfer learning converges fast from pretrained checkpoint')

last_completed = 0
stall_count = 0
check_interval = 1800  # 30 min

for iteration in range(100):  # max 50 hours
    time.sleep(check_interval)

    client = connect()
    if client is None:
        log('SSH failed after retries, will try again next cycle')
        continue

    try:
        # Count completed
        n_done = get_completed_runs(client)

        # Check runner alive
        _, out, _ = exec_cmd(client, 'ps -ef | grep b7_ext_serial_runner.py | grep -v grep | wc -l')
        runner_alive = int(out.strip()) > 0

        # Check training alive
        _, out, _ = exec_cmd(client, 'ps -ef | grep "src.train" | grep -v grep | wc -l')
        training_alive = int(out.strip()) > 0

        # GPU
        _, out, _ = exec_cmd(client, 'nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo NA')
        gpu = out.strip()

        # Last exec log line
        _, out, _ = exec_cmd(client, f'tail -3 {REMOTE_EXEC_LOG}')
        last_lines = out.strip()

        # Check for ERROR or STOPPING
        has_error = 'ERROR' in last_lines or 'STOPPING' in last_lines
        has_all_done = 'ALL 18 RUNS' in last_lines

        status = f'[{n_done}/18 done] runner={runner_alive} train={training_alive} gpu={gpu}'

        if has_all_done:
            log(f'ALL 18 COMPLETED! {status}')
            break

        if has_error:
            # Auto-decide: read the error
            _, err_out, _ = exec_cmd(client, f'grep -A2 "ERROR\|STOPPING" {REMOTE_EXEC_LOG} | tail -6')
            err_detail = err_out.strip()
            log(f'ERROR DETECTED: {err_detail}')

            # If runner dead but training alive, it might be a false alarm from old log
            if not runner_alive and not training_alive:
                # Runner stopped, training not running — genuine stop
                if 'STOPPING due to abnormal best_epoch' in err_detail and 'best_epoch=1' in err_detail:
                    decide('Runner stopped for best_epoch=1', 'RESTART with threshold fix',
                           'best_epoch=1 confirmed normal for transfer learning; restart pending')
                    log('NEED MANUAL RESTART — but threshold should already be fixed')
                elif 'timed out' in err_detail.lower():
                    decide('Run timed out', 'RESTART from next index',
                           f'Run exceeded 2h limit; skip and continue')
                else:
                    log(f'UNKNOWN ERROR — needs investigation: {err_detail}')

        if n_done == last_completed and runner_alive and training_alive:
            stall_count += 1
            if stall_count >= 4:  # 2h of no progress
                log(f'WARNING: stalled for {stall_count * check_interval / 60:.0f} min')
                # Could add auto-kill-and-resume logic here if needed
        else:
            stall_count = 0

        log(f'{status} | {last_lines.split(chr(10))[-1][:120]}')
        last_completed = n_done

    finally:
        client.close()

    if n_done >= 18:
        log('ALL DONE!')
        break

log('=== MONITOR END ===')
log(f'Decisions made: {len(DECISIONS)}')
for d in DECISIONS:
    log(d)
