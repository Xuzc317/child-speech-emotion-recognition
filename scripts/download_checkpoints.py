#!/usr/bin/env python3
"""Download missing checkpoints from AutoDL cloud.
Downloads B5, B6, B7 (full) + B4 missing 7 files.
Total: 64 files, ~23 GB.
"""
import os, sys, time, paramiko

# ── Config ──
LOCAL_BASE = r'D:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\checkpoints\autodl'
REMOTE_BASE = '/root/autodl-tmp/d-ser/checkpoints'

SSH_HOST = 'connect.cqa1.seetacloud.com'
SSH_PORT = 25808
SSH_USER = 'root'
SSH_PASS = '9HmcVfCXUFVD'

# ── Files to download: list of (remote_subpath, local_subpath) ──
# Each remote_subpath is relative to REMOTE_BASE
# Each local_subpath is relative to LOCAL_BASE

FILES_TO_DOWNLOAD = []

# B4 missing: E5-07, E5-09, E5-08_L12
for eid in ['E5-07', 'E5-09']:
    for seed in ['s42', 's123', 's456']:
        FILES_TO_DOWNLOAD.append((f'b4/{eid}_{seed}/best_model.pt',
                                   f'b4/{eid}_{seed}/best_model.pt'))
FILES_TO_DOWNLOAD.append(('b4/E5-08_L12_s42/best_model.pt',
                           'b4/E5-08_L12_s42/best_model.pt'))

# B5 complete: E2-01, E2-02, E2-03
for eid in ['E2-01', 'E2-02', 'E2-03']:
    for seed in ['s42', 's123', 's456']:
        FILES_TO_DOWNLOAD.append((f'b5/{eid}_{seed}/best_model.pt',
                                   f'b5/{eid}_{seed}/best_model.pt'))

# B6 complete: E6-01 through E6-10
for i in range(1, 11):
    for seed in ['s42', 's123', 's456']:
        FILES_TO_DOWNLOAD.append((f'b6/E6-{i:02d}_{seed}/best_model.pt',
                                   f'b6/E6-{i:02d}_{seed}/best_model.pt'))

# B7 complete: E7-01 through E7-06
for i in range(1, 7):
    for seed in ['s42', 's123', 's456']:
        FILES_TO_DOWNLOAD.append((f'b7/E7-{i:02d}_{seed}/best_model.pt',
                                   f'b7/E7-{i:02d}_{seed}/best_model.pt'))


def download_checkpoints(dry_run=False):
    print(f'{"[DRY RUN] " if dry_run else ""}Downloading {len(FILES_TO_DOWNLOAD)} checkpoints...')
    print(f'Total size estimate: {len(FILES_TO_DOWNLOAD) * 363 / 1024:.1f} GB')
    print()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS,
                    look_for_keys=False, allow_agent=False,
                    disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
        sftp = ssh.open_sftp()

        ok, skip, fail = 0, 0, 0
        total_size = 0
        start = time.time()

        for i, (remote_path, local_path) in enumerate(FILES_TO_DOWNLOAD):
            full_remote = f'{REMOTE_BASE}/{remote_path}'
            full_local = os.path.join(LOCAL_BASE, local_path)

            # Check if already exists
            if os.path.exists(full_local):
                local_size = os.path.getsize(full_local)
                if local_size > 300 * 1024 * 1024:  # >300MB, likely valid
                    skip += 1
                    print(f'  [{i+1}/{len(FILES_TO_DOWNLOAD)}] SKIP (exists): {remote_path}')
                    total_size += local_size
                    continue

            # Check remote exists
            try:
                remote_stat = sftp.stat(full_remote)
            except FileNotFoundError:
                fail += 1
                print(f'  [{i+1}/{len(FILES_TO_DOWNLOAD)}] MISSING on cloud: {remote_path}')
                continue

            file_size_mb = remote_stat.st_size / (1024 * 1024)
            print(f'  [{i+1}/{len(FILES_TO_DOWNLOAD)}] DOWNLOAD {remote_path} ({file_size_mb:.0f} MB)...', end=' ', flush=True)

            if dry_run:
                print('(dry run)')
                continue

            # Ensure local directory exists
            os.makedirs(os.path.dirname(full_local), exist_ok=True)

            try:
                sftp.get(full_remote, full_local)
                ok += 1
                total_size += remote_stat.st_size
                elapsed = time.time() - start
                speed = total_size / elapsed / (1024 * 1024) if elapsed > 0 else 0
                print(f'OK ({speed:.1f} MB/s avg)')
            except Exception as e:
                fail += 1
                print(f'FAILED: {e}')
                # Clean up partial file
                if os.path.exists(full_local):
                    os.remove(full_local)

        sftp.close()
        ssh.close()

        elapsed = time.time() - start
        print(f'\n{"="*60}')
        print(f'Done in {elapsed:.1f}s ({total_size/(1024**3):.1f} GB total)')
        print(f'  OK: {ok}  Skip: {skip}  Fail: {fail}')
        print(f'{"="*60}')

    except Exception as e:
        print(f'Connection error: {e}')
        return 1

    return 0


if __name__ == '__main__':
    dry = '--dry-run' in sys.argv
    sys.exit(download_checkpoints(dry_run=dry))
