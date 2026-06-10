#!/bin/bash
# B1: In-Domain Pooling Baseline (9 experiments)
# Upload to AutoDL and run: bash run_b1.sh

export SER_C_BESD_PATH=/root/autodl-tmp/datasets/BESD/BESD/MY
export SER_IEMOCAP_PATH=/root/autodl-tmp/IEMOCAP/wavs
export SER_FAU_AIBO_PATH=/root/autodl-tmp/IS2009EmotionChallenge/IS2009EmotionChallenge/wav

cd /root/autodl-tmp/d-ser
PYTHON=/root/miniconda3/bin/python

echo "=== B1 Environment Check ==="
echo "Python: $($PYTHON --version)"
echo "BESD: $(find $SER_C_BESD_PATH -name '*.wav' 2>/dev/null | wc -l) wavs"
echo "IEMOCAP: $(find $SER_IEMOCAP_PATH -name '*.wav' 2>/dev/null | wc -l) wavs"
echo "FAU: $(find $SER_FAU_AIBO_PATH -name '*.wav' 2>/dev/null | wc -l) wavs"

echo ""
echo "=== Smoke Test: C-BESD 6-class ==="
$PYTHON -c "
import sys; sys.path.insert(0, '.')
from src.data.dataset import _collect_cbesd
import os
e = _collect_cbesd(os.environ['SER_C_BESD_PATH'])
print(f'C-BESD: {len(e)} entries')
from collections import Counter
c = Counter(x[1] for x in e)
print(f'Classes: {dict(c)}')
print(f'Speakers: {len(set(x[2] for x in e))}')
"

echo ""
echo "=== Smoke Test: FAU Aibo 4-class ==="
$PYTHON -c "
import sys; sys.path.insert(0, '.')
from src.data.dataset import _collect_fau_aibo
import os
from src.data.label_mapper import UniversalLabelMapper
from collections import Counter
e = _collect_fau_aibo(os.environ['SER_FAU_AIBO_PATH'])
print(f'FAU: {len(e)} entries')
mapper = UniversalLabelMapper('fau-aibo')
mapped = []
for fp, raw, sid in e:
    idx = mapper.to_index(raw)
    if idx is not None:
        mapped.append(idx)
print(f'Mapped: {len(mapped)} (4-class)')
idx_to_label = {0:'angry',1:'happy',2:'neutral',3:'sad'}
c = Counter(mapped)
for k,v in sorted(c.items()):
    print(f'  {idx_to_label[k]}: {v}')
"

echo ""
echo "=== B1 Ready ==="
echo "Next: run full B1 training (3 datasets x 3 pooling x 3 seeds)"
