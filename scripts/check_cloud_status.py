"""检查云服务器训练状态、日志和模型文件"""
import paramiko
import sys

HOST = 'connect.cqa1.seetacloud.com'
PORT = 12112
USER = 'root'
PASSWORD = '9HmcVfCXUFVD'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
    print("=== SSH 连接成功 ===\n")
except Exception as e:
    print(f"SSH 连接失败: {e}")
    sys.exit(1)

def run(cmd, desc=None):
    if desc:
        print(f"\n--- {desc} ---")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out:
        print(out)
    if err:
        print(f"STDERR: {err}")
    return out

# 1. 检查 GPU 和训练进程
run('nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader', 'GPU 状态')
run('ps aux | grep -E "python|train" | grep -v grep', 'Python 训练进程')

# 2. 工作目录
run('ls -la /root/autodl-tmp/v5_data/', 'v5_data 目录')
run('ls -la /root/autodl-tmp/v5_data/models/ 2>/dev/null || echo "models/ 目录不存在"', '模型文件')

# 3. 最新日志
run('ls -la /root/autodl-tmp/v5_data/logs/', '日志目录')
run('tail -80 /root/autodl-tmp/v5_data/logs/run_v5_save.log 2>/dev/null || echo "run_v5_save.log 不存在"', '最新保存日志 (尾部80行)')

# 4. 查找所有 .pth 文件
run('find /root/autodl-tmp/v5_data/ -name "*.pth" -ls 2>/dev/null', '所有 .pth 文件')

# 5. 检查是否有正在运行的 nohup/screen/tmux
run('tmux ls 2>/dev/null; screen -ls 2>/dev/null; echo "---"; ls -la /root/autodl-tmp/v5_data/logs/nohup* 2>/dev/null || echo "no nohup output files"', '会话管理')

# 6. 检查 ser_project 目录
run('ls -la /root/ser_project/ 2>/dev/null || echo "ser_project 目录不存在"', 'ser_project 目录')
run('ls -la /root/autodl-tmp/ 2>/dev/null', 'autodl-tmp 目录')

# 7. 数据文件
run('ls -lh /root/autodl-tmp/v5_data/data/ 2>/dev/null || echo "data/ 目录不存在"', '数据文件')

ssh.close()
print("\n=== 检查完成 ===")
