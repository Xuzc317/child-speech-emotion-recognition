"""Generate confusion matrices from cloud checkpoints."""
import paramiko, json, os, numpy as np

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('connect.cqa1.seetacloud.com', port=14393, username='root', password='9HmcVfCXUFVD', timeout=15)

CM_SCRIPT = r'''
import torch, json, os, sys, numpy as np
sys.path.insert(0, "/root")
from src.train import SERModel
from src.data import get_dataloaders, get_cross_corpus_dataloaders
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["angry", "happy", "neutral", "sad"]

def gen_cm(checkpoint_path, dataset, split, output_path, pooling_override=None):
    print(f"Loading: {checkpoint_path}")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ck["model_state_dict"]
    # Fix key mismatch
    new_sd = {}
    for k, v in list(sd.items()):
        if "pooler.prosody_proj.2." in k or "pooler.attn.2." in k:
            new_sd[k.replace(".2.", ".3.")] = v
        else:
            new_sd[k] = v
    ck["model_state_dict"] = new_sd

    config = ck.get("config", {})
    config.setdefault("ssl_model", "wavlm")
    config.setdefault("num_classes", 4)
    if pooling_override:
        config["pooling_type"] = pooling_override

    model = SERModel(config).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    dls = get_dataloaders(dataset, seed=42, splits=[split])
    dl = dls[split]
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            waveforms, labels, lengths, _, _ = batch
            waveforms, labels = waveforms.to(device), labels.to(device)
            lengths = lengths.to(device)
            logits = model(waveforms, lengths=lengths)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds).tolist()
    result = {
        "checkpoint": checkpoint_path,
        "dataset": dataset,
        "split": split,
        "confusion_matrix": cm,
        "classes": CLASSES[:len(cm)],
        "total": int(np.sum(cm)),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {output_path} (N={result['total']})")

# P1 new seeds — Exp1 Self-Attn
for seed in [123, 456]:
    gen_cm(f"/root/checkpoints/exp1_selfattn_s{seed}/best_model.pt",
           ["c-besd"], "test",
           f"/root/results/logs/cm_exp1_selfattn_s{seed}.json",
           pooling_override="self_attention")

# P1 new seeds — Exp2 Prosody
for seed in [123, 456]:
    gen_cm(f"/root/checkpoints/exp2_prosody_s{seed}/best_model.pt",
           ["c-besd"], "test",
           f"/root/results/logs/cm_exp2_prosody_s{seed}.json")

# P0 zero-shot — C-BESD->IEMOCAP
gen_cm("/root/checkpoints/exp2/best_model.pt",
       ["iemocap"], "test",
       "/root/results/logs/cm_zeroshot_cbesd2iemocap.json")

# P0 zero-shot — FAU->C-BESD
gen_cm("/root/checkpoints/exp5/best_model.pt",
       ["c-besd"], "test",
       "/root/results/logs/cm_zeroshot_fau2cbesd.json")

# P0 zero-shot — IEMOCAP->C-BESD
gen_cm("/root/checkpoints/exp3/best_model.pt",
       ["c-besd"], "test",
       "/root/results/logs/cm_zeroshot_iemocap2cbesd.json")

print("\nAll confusion matrices generated!")
'''

sftp = c.open_sftp()
with sftp.file('/root/gen_cm.py', 'w') as f:
    f.write(CM_SCRIPT)
sftp.close()

print("Generating confusion matrices on cloud...")
stdin, stdout, stderr = c.exec_command(
    'source ~/miniconda3/etc/profile.d/conda.sh && conda activate speech && cd /root && python gen_cm.py 2>&1'
)
channel = stdout.channel
import time
while not channel.exit_status_ready():
    if channel.recv_ready():
        print(channel.recv(4096).decode(), end='', flush=True)
    time.sleep(0.5)
print(stdout.read().decode())
err = stderr.read().decode()
if err: print("STDERR:", err[-500:])

# Download
print("\n=== Downloading CM files ===")
sftp = c.open_sftp()
local_cm = r'D:\大学\论文\儿童语音情绪识别\新方案-分布驱动儿童SER\results\logs'
os.makedirs(local_cm, exist_ok=True)
for fname in [
    'cm_exp1_selfattn_s123.json', 'cm_exp1_selfattn_s456.json',
    'cm_exp2_prosody_s123.json', 'cm_exp2_prosody_s456.json',
    'cm_zeroshot_cbesd2iemocap.json', 'cm_zeroshot_fau2cbesd.json',
    'cm_zeroshot_iemocap2cbesd.json'
]:
    try:
        sftp.get(f'/root/results/logs/{fname}', os.path.join(local_cm, fname))
        with open(os.path.join(local_cm, fname)) as f:
            d = json.load(f)
        cm = d['confusion_matrix']
        row_sums = [sum(r) for r in cm]
        diag = sum(cm[i][i] for i in range(len(cm)))
        wa = diag / sum(row_sums) * 100
        print(f'{fname}: {len(cm)}x{len(cm)}, N={sum(row_sums)}, WA={wa:.1f}%')
    except Exception as e:
        print(f'{fname}: FAILED - {e}')

sftp.close()
c.close()
print('\nDone!')
