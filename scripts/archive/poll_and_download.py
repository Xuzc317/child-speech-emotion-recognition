"""轮询云服务器，新 .pth 文件出现时自动下载。
目标: 24 个文件 (A1×3, A2b×3, A3×3, B3×3, C1×3, C2×3, C3×3, C4×3)
"""
import paramiko
import os, time, sys

HOST = 'connect.cqa1.seetacloud.com'
PORT = 12112
USER = 'root'
PASSWORD = '9HmcVfCXUFVD'
REMOTE_DIR = '/root/autodl-tmp/v5_data/models'
LOCAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'v5_622')

EXPECTED_TOTAL = 24  # 8 configs × 3 seeds
CHECK_INTERVAL = 120  # 2 minutes

os.makedirs(LOCAL_DIR, exist_ok=True)

def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
    return ssh

def get_remote_files(sftp):
    files = {}
    for f in sftp.listdir(REMOTE_DIR):
        if f.endswith('.pth'):
            path = f'{REMOTE_DIR}/{f}'
            files[f] = sftp.stat(path).st_size
    return files

def check_training_status(ssh):
    stdin, stdout, stderr = ssh.exec_command(
        "tail -5 /root/autodl-tmp/v5_data/logs/run_v5_save.log 2>/dev/null | grep -E '^(=|  [A-Z])' || echo waiting"
    )
    return stdout.read().decode('utf-8', errors='replace').strip()

print(f"开始轮询云服务器 (每 {CHECK_INTERVAL}s)")
print(f"目标: {EXPECTED_TOTAL} 个 .pth 文件")
print(f"本地目录: {LOCAL_DIR}\n")

iteration = 0
last_count = 0

while True:
    iteration += 1
    try:
        ssh = connect()
        sftp = ssh.open_sftp()
        remote = get_remote_files(sftp)

        downloaded = 0
        for name, size in remote.items():
            local_path = os.path.join(LOCAL_DIR, name)
            if not os.path.exists(local_path):
                rpath = f'{REMOTE_DIR}/{name}'
                try:
                    sftp.get(rpath, local_path)
                    print(f"  [下载] {name} ({size/1024/1024:.1f} MB)")
                    downloaded += 1
                except Exception as e:
                    print(f"  [失败] {name}: {e}")

        local_count = len([f for f in os.listdir(LOCAL_DIR) if f.endswith('.pth')])
        remote_count = len(remote)

        status_line = check_training_status(ssh)
        sftp.close()
        ssh.close()

        if downloaded > 0 or remote_count != last_count:
            print(f"[{time.strftime('%H:%M:%S')}] 第{iteration}次: 远程{remote_count}/本地{local_count}/{EXPECTED_TOTAL}目标 | 新下载{downloaded}个")
            if status_line:
                print(f"  当前: {status_line[:120]}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 第{iteration}次: 远程{remote_count}/本地{local_count} — 无变化")

        last_count = remote_count

        if remote_count >= EXPECTED_TOTAL and local_count >= EXPECTED_TOTAL:
            print(f"\n=== 全部 {EXPECTED_TOTAL} 个 .pth 文件已下载完成! ===")
            break

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 连接失败: {e}")

    time.sleep(CHECK_INTERVAL)

print("Done.")
