#!/usr/bin/env python3
"""Poll AutoDL until fix_4runs.sh completes. Exits 0 on success."""
import paramiko, time, sys

HOST = 'connect.cqa1.seetacloud.com'
PORT = 25808
USER = 'root'
PASS = '9HmcVfCXUFVD'
LOG = '/root/autodl-tmp/d-ser/fix_4runs_output.log'
REMOTE_LOGS = '/root/autodl-tmp/d-ser/results/logs'
FILES = ['E1-08_s42.json', 'E4-04_s42.json', 'E4-10_s42.json', 'E4-10_s123.json']

def ssh_cmd(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    return stdout.read().decode('utf-8', errors='replace').strip()

def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS,
                look_for_keys=False, allow_agent=False,
                disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']},
                timeout=15)
    return ssh

def main():
    ssh = connect()
    last_done = -1

    while True:
        try:
            # Check ALL DONE
            out = ssh_cmd(ssh, 'grep "ALL DONE" %s 2>/dev/null || echo ""' % LOG)

            if out:
                print('[%s] ALL DONE!' % time.strftime('%H:%M:%S'))
                tail = ssh_cmd(ssh, 'tail -30 %s' % LOG)
                print(tail)

                # Quick file check
                print('\n--- Remote JSON file check ---')
                for f in FILES:
                    r = ssh_cmd(ssh, 'test -f %s/%s && echo "EXISTS" || echo "MISSING"' % (REMOTE_LOGS, f))
                    print('  %s: %s' % (f, r))

                ssh.close()
                return 0

            # Done count
            done_out = ssh_cmd(ssh, 'grep -c "DONE (exit" %s 2>/dev/null || echo 0' % LOG)
            done = int(done_out.strip() or '0')

            # Training processes
            proc_out = ssh_cmd(ssh, 'pgrep -fc "src.train" || echo 0')
            n = int(proc_out.strip() or '0')

            # Last epoch line
            last = ssh_cmd(ssh, 'grep "^Epoch" %s | tail -1' % LOG)
            if not last:
                last = ssh_cmd(ssh, 'grep "^===" %s | tail -1' % LOG)

            # GPU
            gpu = ssh_cmd(ssh, 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null')

            if done != last_done:
                print('[%s] %d/4 done | GPU: %s | %s' % (time.strftime('%H:%M:%S'), done, gpu, last[:200]))
                last_done = done

            ssh.close()
        except Exception as e:
            print('[%s] Error: %s' % (time.strftime('%H:%M:%S'), str(e)[:150]))

        time.sleep(120)
        try:
            ssh = connect()
        except:
            time.sleep(30)
            ssh = connect()

if __name__ == '__main__':
    sys.exit(main())
