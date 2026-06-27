"""从云服务器下载 .pth 模型文件到本地 checkpoints/ 目录"""
import paramiko
import os
import sys

HOST = 'connect.cqa1.seetacloud.com'
PORT = 12112
USER = 'root'
PASSWORD = '9HmcVfCXUFVD'
REMOTE_MODEL_DIR = '/root/autodl-tmp/v5_data/models'
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'v5_622')

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)

sftp = ssh.open_sftp()

# 列出远程 .pth 文件
remote_files = []
for f in sftp.listdir(REMOTE_MODEL_DIR):
    if f.endswith('.pth'):
        path = f'{REMOTE_MODEL_DIR}/{f}'
        size_mb = sftp.stat(path).st_size / (1024 * 1024)
        remote_files.append((f, size_mb))

print(f"找到 {len(remote_files)} 个 .pth 文件:")
for name, size in sorted(remote_files):
    local_path = os.path.join(LOCAL_MODEL_DIR, name)
    status = "已存在" if os.path.exists(local_path) else "新文件"
    print(f"  {name} ({size:.1f} MB) — {status}")

print(f"\n开始下载到 {LOCAL_MODEL_DIR}...")
downloaded = 0
skipped = 0

for name, size in sorted(remote_files):
    local_path = os.path.join(LOCAL_MODEL_DIR, name)
    if os.path.exists(local_path):
        skipped += 1
        continue
    remote_path = f'{REMOTE_MODEL_DIR}/{name}'
    print(f"  下载 {name} ({size:.1f} MB)...", end=' ', flush=True)
    try:
        sftp.get(remote_path, local_path)
        print("OK")
        downloaded += 1
    except Exception as e:
        print(f"失败: {e}")

sftp.close()
ssh.close()
print(f"\n下载完成: {downloaded} 个新文件, {skipped} 个已跳过")
