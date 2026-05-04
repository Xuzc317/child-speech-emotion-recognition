"""Run A2 stat-prior independently on cloud."""
import sys, os, json, numpy as np
sys.path.insert(0, '/root/ser_project')
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset_ssl import SSLFeatureDataset, collate_fn_ssl_features
from src.models.adapter import AcousticCalibrationAdapter
from src.models.semlp import SEMLP

DEVICE = torch.device("cuda")
DATA = '/root/autodl-tmp/v5_data/'

# Load stat prior
init = dict(np.load('/root/ser_project/data/adapter_init.npz'))
print(f"Loaded stat prior: scale shape={init['scale'].shape}")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = AcousticCalibrationAdapter(
            dim=768, init_scale=init['scale'], init_bias=init['bias'])
        self.cls = SEMLP(input_dim=768, num_classes=6)
    def forward(self, x):
        return self.cls(self.adapter(x).mean(dim=1))

results = []
for seed in [42, 123, 456]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model().to(DEVICE)

    td = SSLFeatureDataset(DATA+'train_wavlm_feats.npy', DATA+'train_wavlm_labels.npy')
    vd = SSLFeatureDataset(DATA+'val_wavlm_feats.npy', DATA+'val_wavlm_labels.npy')
    sd = SSLFeatureDataset(DATA+'test_wavlm_feats.npy', DATA+'test_wavlm_labels.npy')

    tl = DataLoader(td, 128, shuffle=True, collate_fn=collate_fn_ssl_features, num_workers=4)
    vl = DataLoader(vd, 128, shuffle=False, collate_fn=collate_fn_ssl_features, num_workers=0)
    sl = DataLoader(sd, 128, shuffle=False, collate_fn=collate_fn_ssl_features, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_uar, patience, best_state = 0, 0, None
    for ep in range(60):
        model.train()
        for b in tl:
            f, l = b[0].to(DEVICE), b[1].to(DEVICE)
            opt.zero_grad()
            crit(model(f), l).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vc = sum((model(b[0].to(DEVICE)).argmax(dim=1).cpu() == b[1]).sum().item() for b in vl)
        vu = vc / len(vd)
        if vu > best_uar:
            best_uar = vu; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 15:
            break

    model.load_state_dict(best_state)
    with torch.no_grad():
        tc = sum((model(b[0].to(DEVICE)).argmax(dim=1).cpu() == b[1]).sum().item() for b in sl)
    twa = tc / len(sd)
    print(f"A2_stat_prior seed={seed}: val_uar={best_uar:.4f} test_wa={twa:.4f}")
    results.append({'seed': seed, 'val_uar': float(best_uar), 'test_wa': float(twa)})

print(json.dumps(results))

# Save
os.makedirs('/root/ser_project/experiments/v5_622', exist_ok=True)
with open('/root/ser_project/experiments/v5_622/a2_stat_prior_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved.")
