"""Run P1 experiments on AutoDL: multi-seed training + APC."""
import paramiko, time, json, os

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('connect.cqa1.seetacloud.com', port=14393, username='root', password='9HmcVfCXUFVD', timeout=15)

P1_SCRIPT = r'''
import subprocess, json, os, sys

PYTHON = "/root/miniconda3/envs/speech/bin/python"
BASE = f"cd /root && {PYTHON} src/train.py --train_data c-besd --test_data c-besd --batch_size 16 --epochs 100 --patience 15 --lr 3e-4"

experiments = [
    ("exp2_prosody_s123", "--pooling_type prosody_guided --reg_profile default --seed 123"),
    ("exp2_prosody_s456", "--pooling_type prosody_guided --reg_profile default --seed 456"),
    ("exp1_selfattn_s123", "--pooling_type self_attention --reg_profile default --seed 123"),
    ("exp1_selfattn_s456", "--pooling_type self_attention --reg_profile default --seed 456"),
]

for exp_name, extra_args in experiments:
    cmd = f"{BASE} {extra_args} --exp_name {exp_name} --output_dir checkpoints/{exp_name}"
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"{'='*60}")
    sys.stdout.flush()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
    out = result.stdout
    print(out[-3000:] if len(out) > 3000 else out)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])
    print(f"Exit: {result.returncode}")
    sys.stdout.flush()

print("\n=== ALL P1 TRAINING DONE ===")
'''

sftp = c.open_sftp()
with sftp.file('/root/run_p1.py', 'w') as f:
    f.write(P1_SCRIPT.replace('\r\n', '\n'))
sftp.close()

print("Starting P1 experiments...")
stdin, stdout, stderr = c.exec_command(
    'source ~/miniconda3/etc/profile.d/conda.sh && conda activate speech && cd /root && python run_p1.py 2>&1'
)

channel = stdout.channel
while not channel.exit_status_ready():
    if channel.recv_ready():
        data = channel.recv(4096)
        print(data.decode(), end='', flush=True)
    time.sleep(0.5)
print(stdout.read().decode())

# Download results
print("\n=== Downloading P1 results ===")
sftp = c.open_sftp()
local_logs = r'D:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\results\logs'
os.makedirs(local_logs, exist_ok=True)

for exp_name in ['exp1_selfattn_s123', 'exp1_selfattn_s456', 'exp2_prosody_s123', 'exp2_prosody_s456']:
    remote_path = f'/root/results/logs/{exp_name}.json'
    local_path = os.path.join(local_logs, f'{exp_name}.json')
    try:
        sftp.get(remote_path, local_path)
        with open(local_path) as f:
            d = json.load(f)
        print(f'{exp_name}: WA={d.get("test_wa","?")}, best_epoch={d.get("best_epoch","?")}')
    except Exception as e:
        print(f'{exp_name}: FAILED - {e}')

sftp.close()
c.close()
print('\nP1 complete!')
