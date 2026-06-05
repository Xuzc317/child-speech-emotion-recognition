"""Run all P0 zero-shot experiments on AutoDL cloud."""
import paramiko, time, json, os

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('connect.cqa1.seetacloud.com', port=14393, username='root', password='9HmcVfCXUFVD', timeout=15)

# Write remote script
remote_script = '''
import torch, json, os, sys, numpy as np
sys.path.insert(0, "/root")
from src.train import SERModel
from src.data import get_cross_corpus_dataloaders
from sklearn.metrics import accuracy_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_zeroshot(checkpoint_path, train_data, test_data, seed, output_path):
    print(f"Loading: {checkpoint_path}")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Fix state_dict key mismatch: old checkpoints have pooling without dropout
    # (prosody_proj has 2 Linear layers = keys end in .0 and .2)
    # New code adds Dropout layers = keys end in .0 and .3
    # Remap .2.* -> .3.* in state_dict
    sd = ck["model_state_dict"]
    new_sd = {}
    for k, v in list(sd.items()):
        if "pooler.prosody_proj.2." in k or "pooler.attn.2." in k:
            new_k = k.replace(".2.", ".3.")
            new_sd[new_k] = v
        elif "pooler.prosody_proj.3." in k or "pooler.attn.3." in k:
            pass  # skip new-style keys (we'll use the remapped old ones)
        else:
            new_sd[k] = v
    ck["model_state_dict"] = new_sd
    config = ck.get("config", {})
    config.setdefault("ssl_model", "wavlm")
    config.setdefault("num_classes", 4)
    config.setdefault("pooling_type", "prosody_guided")
    print(f"  Config: {config}")

    model = SERModel(config).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    dls = get_cross_corpus_dataloaders(train_data, test_data, seed=seed)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dls["test"]:
            waveforms, labels, lengths, _, _ = batch
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            logits = model(waveforms, lengths=lengths)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    wa = accuracy_score(all_labels, all_preds)
    uar = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    test_n = len(dls["test"].dataset)

    result = {
        "checkpoint": checkpoint_path, "train_data": train_data,
        "test_data": test_data, "seed": seed,
        "test_wa": float(wa), "test_uar": float(uar), "test_n": test_n,
        "type": "zero_shot"
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  RESULT: WA={wa:.4f} UAR={uar:.4f} N={test_n}")
    return result

print("=== P0-1: C-BESD -> IEMOCAP ===")
run_zeroshot("/root/checkpoints/exp2/best_model.pt", ["c-besd"], ["iemocap"], 42,
    "/root/results/logs/exp_zeroshot_cbesd2iemocap.json")

print("=== P0-2: FAU -> C-BESD ===")
run_zeroshot("/root/checkpoints/exp5/best_model.pt", ["fau-aibo"], ["c-besd"], 42,
    "/root/results/logs/exp_zeroshot_fau2cbesd.json")

print("=== P0-3: IEMOCAP -> C-BESD ===")
run_zeroshot("/root/checkpoints/exp3/best_model.pt", ["iemocap"], ["c-besd"], 42,
    "/root/results/logs/exp_zeroshot_iemocap2cbesd.json")

print("\\n=== ALL P0 DONE ===")
'''

sftp = c.open_sftp()
with sftp.file('/root/run_p0.py', 'w') as f:
    f.write(remote_script)
sftp.close()

print("Remote script uploaded. Executing...")
stdin, stdout, stderr = c.exec_command(
    'source ~/miniconda3/etc/profile.d/conda.sh && conda activate speech && cd /root && python run_p0.py 2>&1'
)

# Stream output
channel = stdout.channel
while not channel.exit_status_ready():
    if channel.recv_ready():
        data = channel.recv(4096)
        print(data.decode(), end='', flush=True)
    time.sleep(0.5)
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    print("STDERR:", err)

# Download results
print("\n=== Downloading results ===")
local_logs = r'D:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\results\logs'
os.makedirs(local_logs, exist_ok=True)

for fname in ['exp_zeroshot_cbesd2iemocap.json', 'exp_zeroshot_fau2cbesd.json', 'exp_zeroshot_iemocap2cbesd.json']:
    try:
        sftp.get(f'/root/results/logs/{fname}', os.path.join(local_logs, fname))
        print(f'Downloaded: {fname}')
    except Exception as e:
        print(f'Failed: {fname} - {e}')

sftp.close()
c.close()
print('\nP0 complete!')
