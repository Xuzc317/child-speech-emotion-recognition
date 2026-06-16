#!/usr/bin/env python3
"""Upload plot_bubbles_v6.py to remote server and execute it."""
import paramiko
import sys
import os

HOST = 'connect.cqa1.seetacloud.com'
PORT = 25808
USER = 'root'
PASS = '9HmcVfCXUFVD'
REMOTE_DIR = '/root/autodl-tmp/acoustic_analysis'

# Local script to upload
LOCAL_SCRIPT = os.path.join(os.path.dirname(__file__), 'plot_bubbles_v6.py')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# Upload the script
sftp = ssh.open_sftp()
sftp.put(LOCAL_SCRIPT, f'{REMOTE_DIR}/plot_bubbles_v6.py')
sftp.close()
print('Uploaded plot_bubbles_v6.py')

# Run the script with environment variables
cmd = (
    'cd /root/autodl-tmp/acoustic_analysis && '
    'export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY && '
    'export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs && '
    'export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav && '
    '/root/miniconda3/bin/python3 plot_bubbles_v6.py 2>&1'
)

stdin, stdout, stderr = ssh.exec_command(cmd)

# Print output line by line in real time
for line in iter(stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()

exit_code = stdout.channel.recv_exit_status()
print(f'\nExit code: {exit_code}')

stderr_text = stderr.read().decode()
if stderr_text:
    print('STDERR:', stderr_text)

# List output files
stdin, stdout, stderr = ssh.exec_command(f'ls -la {REMOTE_DIR}/fig_bubbles_v4_*.png')
print('\nOutput files:')
print(stdout.read().decode())

ssh.close()
