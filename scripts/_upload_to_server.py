"""Upload updated launch scripts to AutoDL server and start chain monitor."""
import paramiko
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER = '/root/autodl-tmp/d-ser/scripts'
FILES = ['launch_b2.sh', 'launch_b3.sh', 'launch_b4.sh', 'launch_b5.sh', 'launch_b6.sh', 'launch_b7.sh']

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('connect.cqa1.seetacloud.com', port=25808, username='root',
               password='9HmcVfCXUFVD', timeout=30)

sftp = client.open_sftp()

for fname in FILES:
    local_path = os.path.join(SCRIPT_DIR, fname)
    remote_path = os.path.join(SERVER, fname)
    try:
        sftp.put(local_path, remote_path)
        print('OK: ' + fname)
    except Exception as e:
        print('FAIL: ' + fname + ' - ' + str(e))

sftp.close()

# Verify
stdin, stdout, stderr = client.exec_command(
    'ls -la /root/autodl-tmp/d-ser/scripts/launch_b*.sh && echo --- && '
    'head -1 /root/autodl-tmp/d-ser/scripts/launch_b3.sh && echo --- && '
    'grep -c data_split_seed /root/autodl-tmp/d-ser/scripts/launch_b*.sh'
)
out = stdout.read().decode()
err = stderr.read().decode()
print('\nVerify:')
print(out)
if err:
    print('ERR:', err)

# Create chain_starter.sh
chain = """#!/bin/bash
# Auto-starter: waits for B2 to finish, then launches B3->B7 chain
echo "Chain starter running at $(date)"
echo "Waiting for B2 to finish..."
while ps aux | grep -q "[l]aunch_b2.sh"; do
    sleep 60
done
echo "B2 finished at $(date)"
echo "Launching B3 chain..."
cd /root/autodl-tmp/d-ser
nohup bash scripts/launch_b3.sh > b3_output.log 2>&1 &
echo "B3 launched with PID $!"
echo "Chain B3->B4->B5->B6->B7 will proceed automatically"
"""

stdin, stdout, stderr = client.exec_command(
    'cat > /root/autodl-tmp/d-ser/scripts/chain_starter.sh && '
    'chmod +x /root/autodl-tmp/d-ser/scripts/chain_starter.sh'
)
stdin.write(chain)
stdin.close()
out = stdout.read().decode()
err = stderr.read().decode()
print('Chain starter:', out if out else 'created')

# Start the chain monitor in background
stdin2, stdout2, stderr2 = client.exec_command(
    'cd /root/autodl-tmp/d-ser && '
    'nohup bash scripts/chain_starter.sh > chain_starter_output.log 2>&1 &'
)
out2 = stdout2.read().decode()
err2 = stderr2.read().decode()
print('Chain monitor launched')

# Verify it's running
stdin3, stdout3, stderr3 = client.exec_command(
    'ps aux | grep chain_starter | grep -v grep'
)
print('Process check:')
print(stdout3.read().decode())

client.close()
print('\n=== ALL DONE ===')
