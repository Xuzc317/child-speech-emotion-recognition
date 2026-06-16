"""Deploy plot_bubbles_v4.py to remote server, run it, and download results."""
import paramiko
import os
import sys
import time

HOST = 'connect.cqa1.seetacloud.com'
PORT = 25808
USER = 'root'
PASSWORD = '9HmcVfCXUFVD'
PYTHON = '/root/miniconda3/envs/speech/bin/python3.10'

REMOTE_DIR = '/root/autodl-tmp/acoustic_analysis'
REMOTE_SCRIPT = f'{REMOTE_DIR}/plot_bubbles_v4.py'

LOCAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'plot_bubbles_v4.py')
LOCAL_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'paper_draft', 'figures'
)

OUTPUT_FILES = [
    'fig_bubbles_v4_cbesd.png',
    'fig_bubbles_v4_fauaibo.png',
    'fig_bubbles_v4_iemocap.png',
    'fig_bubbles_v4_combined.png',
]


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {HOST}:{PORT}...")
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                   timeout=30, banner_timeout=30, auth_timeout=30)
    print("Connected.")
    transport = client.get_transport()

    def run(cmd, desc="", stream=False):
        if desc:
            print(f"\n--- {desc} ---")
        print(f"$ {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
        if stream:
            chan = transport.open_session()
            chan.exec_command(cmd)
            while not chan.exit_status_ready():
                if chan.recv_ready():
                    data = chan.recv(4096).decode('utf-8', errors='replace')
                    print(data, end='', flush=True)
                time.sleep(0.3)
            data = chan.recv(65536).decode('utf-8', errors='replace')
            if data:
                print(data)
            err = chan.recv_stderr(65536).decode('utf-8', errors='replace')
            if err:
                print(f"STDERR: {err}", file=sys.stderr)
            rc = chan.recv_exit_status()
            chan.close()
            return rc
        else:
            chan = transport.open_session()
            chan.exec_command(cmd)
            out = chan.recv(65536).decode('utf-8', errors='replace')
            err = chan.recv_stderr(65536).decode('utf-8', errors='replace')
            rc = chan.recv_exit_status()
            chan.close()
            if out:
                print(out)
            if err:
                print(f"STDERR: {err}", file=sys.stderr)
            return rc

    # Step 1: Create directory and upload script
    run(f"mkdir -p {REMOTE_DIR}", "Create remote directory")

    print("\n--- Uploading plot_bubbles_v4.py ---")
    with open(LOCAL_SCRIPT, 'r', encoding='utf-8') as f:
        script_content = f.read()
    sftp = client.open_sftp()
    with sftp.file(REMOTE_SCRIPT, 'w') as f:
        f.write(script_content)
    sftp.chmod(REMOTE_SCRIPT, 0o755)
    sftp.close()
    print(f"Uploaded ({len(script_content)} bytes)")

    # Step 2: Verify syntax
    rc = run(f"{PYTHON} -c \"compile(open('{REMOTE_SCRIPT}').read(), '{REMOTE_SCRIPT}', 'exec'); print('Syntax OK')\"",
             "Verify syntax")
    if rc != 0:
        print("ERROR: Syntax check failed!")
        client.close()
        return 1

    # Step 3: Run the plot script (long-running, stream output)
    cmd = (
        f"cd {REMOTE_DIR} && "
        f"CUDA_VISIBLE_DEVICES='' "
        f"SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY "
        f"SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs "
        f"SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav "
        f"{PYTHON} plot_bubbles_v4.py"
    )
    rc = run(cmd, "Running plot_bubbles_v4.py", stream=True)
    print(f"\nExit code: {rc}")

    # Step 4: List output files
    print("\n--- Remote output ---")
    run(f"ls -lh {REMOTE_DIR}/*v4*.png 2>/dev/null || echo 'No v4 PNG files found'")

    # Step 5: Download results
    os.makedirs(LOCAL_OUTPUT, exist_ok=True)
    sftp = client.open_sftp()
    for fname in OUTPUT_FILES:
        remote_path = f"{REMOTE_DIR}/{fname}"
        local_path = os.path.join(LOCAL_OUTPUT, fname)
        try:
            print(f"Downloading {fname} -> {local_path}")
            sftp.get(remote_path, local_path)
            size_kb = os.path.getsize(local_path) / 1024
            print(f"  Done ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"  ERROR: {e}")
    sftp.close()

    client.close()
    print("\nAll done!")


if __name__ == '__main__':
    main()
