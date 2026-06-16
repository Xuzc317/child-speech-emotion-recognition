#!/usr/bin/env python3
"""Upload plot_bubbles_v7.py to remote server, execute, and download results."""
import paramiko
import sys
import os

HOST = 'connect.cqa1.seetacloud.com'
PORT = 25808
USER = 'root'
PASS = '9HmcVfCXUFVD'
REMOTE_DIR = '/root/autodl-tmp/acoustic_analysis'

LOCAL_SCRIPT = os.path.join(os.path.dirname(__file__), 'plot_bubbles_v7.py')
LOCAL_FIGURES = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'paper_draft', 'figures'
)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
print(f'Connected to {HOST}:{PORT}')

# 1. Upload the script
sftp = ssh.open_sftp()
remote_script = f'{REMOTE_DIR}/plot_bubbles_v7.py'
sftp.put(LOCAL_SCRIPT, remote_script)
sftp.close()
print(f'Uploaded: {remote_script}')

# 2. Run the script with environment variables
cmd = (
    'cd /root/autodl-tmp/acoustic_analysis && '
    'export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY && '
    'export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs && '
    'export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav && '
    '/root/miniconda3/bin/python3 plot_bubbles_v7.py 2>&1'
)

print(f'\nRunning: plot_bubbles_v7.py')
print('-' * 60)
stdin, stdout, stderr = ssh.exec_command(cmd)

# Print output line by line in real time
for line in iter(stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()

exit_code = stdout.channel.recv_exit_status()
print(f'Exit code: {exit_code}')

stderr_text = stderr.read().decode()
if stderr_text:
    print('STDERR:', stderr_text)

if exit_code != 0:
    print(f'ERROR: Script failed with exit code {exit_code}')
    ssh.close()
    sys.exit(1)

# 3. Download output files
print(f'\nDownloading figures to: {LOCAL_FIGURES}')
os.makedirs(LOCAL_FIGURES, exist_ok=True)

sftp = ssh.open_sftp()
output_files = [
    'fig_bubbles_v4_cbesd.png',
    'fig_bubbles_v4_fauaibo.png',
    'fig_bubbles_v4_iemocap.png',
    'fig_bubbles_v4_combined.png',
]

for fname in output_files:
    remote_path = f'{REMOTE_DIR}/{fname}'
    local_path = os.path.join(LOCAL_FIGURES, fname)
    try:
        sftp.get(remote_path, local_path)
        size_kb = os.path.getsize(local_path) / 1024
        print(f'  Downloaded: {fname} ({size_kb:.1f} KB)')
    except Exception as e:
        print(f'  SKIP {fname}: {e}')

sftp.close()
ssh.close()
print('\nDone!')
