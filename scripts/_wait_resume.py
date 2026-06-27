import sys
import time
import paramiko

LOG = "/root/autodl-tmp/d-ser/ac_suite_logs/resume_20260526_223135.log"
for i in range(40):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect("connect.cqa1.seetacloud.com", 14393, "root", "9HmcVfCXUFVD", timeout=30)
        _, o, _ = c.exec_command(f"grep 'AC SUITE RESUME DONE' {LOG} && echo DONE || tail -n 3 {LOG}")
        out = o.read().decode(errors="replace").strip()
        _, o2, _ = c.exec_command("ps -ef | grep src.train | grep -v grep | wc -l")
        train = o2.read().decode().strip()
        print(f"[{i}] train_procs={train} | {out[-200:]}")
        if "DONE" in out:
            print("FINISHED")
            sys.exit(0)
    except Exception as e:
        print(f"[{i}] ssh error: {e}")
    finally:
        c.close()
    time.sleep(30)
sys.exit(1)
