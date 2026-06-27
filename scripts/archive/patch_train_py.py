"""Patch train.py on AutoDL to support mean pooling.
Run on AutoDL: python patch_train_py.py
"""
import os

TARGET = '/root/autodl-tmp/d-ser/src/train.py'

with open(TARGET, 'r') as f:
    content = f.read()

# Patch 1: Add 'mean' to pooling choices
old_choices = "choices=['prosody_guided', 'self_attention']"
new_choices = "choices=['prosody_guided', 'self_attention', 'mean']"
if old_choices in content:
    content = content.replace(old_choices, new_choices)
    print("PATCH 1 OK: added 'mean' to pooling choices")
else:
    print("PATCH 1 SKIP: choices line not found as expected")
    # Try to find the actual line
    for line in content.split('\n'):
        if 'choices' in line and 'pooling' in line.lower():
            print(f"  Found: {line.strip()}")

# Patch 2: Add mean pooling handling in forward() before prosody block
# Find the pattern to insert before
marker = "# M3: Pool with or without prosody"
if marker in content:
    mean_block = """        # M3: Pool with or without prosody
        if self.pooling_type == 'mean':
            # Simple mean pooling over time dimension (mask-aware)
            if mask is not None:
                fused_m = fused * mask.unsqueeze(-1).float()
                pooled = fused_m.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            else:
                pooled = fused.mean(dim=1)
        el"""
    old_block = """        # M3: Pool with or without prosody
        if self.pooling_type == 'prosody_guided':"""
    content = content.replace(old_block, mean_block + old_block[2:])
    print("PATCH 2 OK: added mean pooling in forward()")
else:
    print("PATCH 2 SKIP: marker not found")

with open(TARGET, 'w') as f:
    f.write(content)

print("Patched:", TARGET)
print("Contains 'mean':", "choices=['prosody_guided', 'self_attention', 'mean']" in open(TARGET).read())
print("Contains mean forward:", "self.pooling_type == 'mean'" in open(TARGET).read())
