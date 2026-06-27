"""Batch-update B2-B7 launch scripts with E1 best configs, data_split_seed, and auto-chaining."""
import os

SCRIPTS = os.path.dirname(os.path.abspath(__file__))

def fix_script(path, replacements):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    ok = 0
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            ok += 1
            print(f'  OK: {old[:60]}...')
        else:
            print(f'  MISS: {repr(old[:80])}')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'  -> {ok}/{len(replacements)} applied to {os.path.basename(path)}')

# E1 best:
#   C-BESD:  self_attention (E1-02_s42, WA=0.9292)
#   FAU:     self_attention (E1-05_s42, WA=0.6781)
#   IEMOCAP: prosody_guided (E1-09_s42, WA=0.6505)

# ── B2 ──
fix_script(os.path.join(SCRIPTS, 'launch_b2.sh'), [
    (
        '--seed 42 --epochs 100 --batch_size 16 --patience 15 \\',
        '--seed 42 --data_split_seed 42 --epochs 100 --batch_size 16 --patience 15 \\'
    ),
    (
        'echo " B2 COMPLETED: $(date)"\n'
        'echo "========================================="',
        'echo " B2 COMPLETED: $(date)"\n'
        'echo "========================================="\n'
        'echo ""\n'
        'echo "=== AUTO-CHAINING TO B3 ==="\n'
        'nohup bash scripts/launch_b3.sh > b3_output.log 2>&1 &\n'
        'echo "B3 launched in background"'
    ),
])

# ── B3 ──
fix_script(os.path.join(SCRIPTS, 'launch_b3.sh'), [
    (
        'IEMO_POOL="self_attention"  # placeholder, update from E1',
        'IEMO_POOL="prosody_guided"  # E1-09 best (WA=0.6505)'
    ),
    (
        '--seed "$seed" --epochs 100 --batch_size 16 --patience 15 \\\n'
        '            --exp_name "${exp_id}_s${seed}" \\',
        '--seed "$seed" --data_split_seed 42 --epochs 100 --batch_size 16 --patience 15 \\\n'
        '            --exp_name "${exp_id}_s${seed}" \\'
    ),
    (
        'echo " B3 COMPLETED: $(date)"\n'
        'echo "========================================="',
        'echo " B3 COMPLETED: $(date)"\n'
        'echo "========================================="\n'
        'echo ""\n'
        'echo "=== AUTO-CHAINING TO B4 ==="\n'
        'nohup bash scripts/launch_b4.sh > b4_output.log 2>&1 &\n'
        'echo "B4 launched in background"'
    ),
])

# ── B4 ──
fix_script(os.path.join(SCRIPTS, 'launch_b4.sh'), [
    (
        'IEMO_POOL="self_attention"',
        'IEMO_POOL="prosody_guided"  # E1-09 best (WA=0.6505)'
    ),
    (
        '--seed "$seed" --epochs 100 --batch_size 16 --patience 15 \\\n'
        '            --exp_name "${exp_id}_s${seed}" \\',
        '--seed "$seed" --data_split_seed 42 --epochs 100 --batch_size 16 --patience 15 \\\n'
        '            --exp_name "${exp_id}_s${seed}" \\'
    ),
    (
        '--seed 42 --epochs 100 --batch_size 16 --patience 15 \\\n'
        '            --exp_name "${eid}_s42" \\',
        '--seed 42 --data_split_seed 42 --epochs 100 --batch_size 16 --patience 15 \\\n'
        '            --exp_name "${eid}_s42" \\'
    ),
    (
        'echo " B4 COMPLETED: $(date)"\n'
        'echo "========================================="',
        'echo " B4 COMPLETED: $(date)"\n'
        'echo "========================================="\n'
        'echo ""\n'
        'echo "=== AUTO-CHAINING TO B5 ==="\n'
        'nohup bash scripts/launch_b5.sh > b5_output.log 2>&1 &\n'
        'echo "B5 launched in background"'
    ),
])

# ── B5 ──
fix_script(os.path.join(SCRIPTS, 'launch_b5.sh'), [
    (
        'IEMO_POOL="self_attention"',
        'IEMO_POOL="prosody_guided"  # E1-09 best (WA=0.6505)'
    ),
    (
        '--seed "$seed" --epochs 100 --batch_size 8 --patience 15 \\',
        '--seed "$seed" --data_split_seed 42 --epochs 100 --batch_size 8 --patience 15 \\'
    ),
    (
        'echo " B5 COMPLETED: $(date)"\n'
        'echo "========================================="',
        'echo " B5 COMPLETED: $(date)"\n'
        'echo "========================================="\n'
        'echo ""\n'
        'echo "=== AUTO-CHAINING TO B6 ==="\n'
        'nohup bash scripts/launch_b6.sh > b6_output.log 2>&1 &\n'
        'echo "B6 launched in background"'
    ),
])

# ── B6 ──
fix_script(os.path.join(SCRIPTS, 'launch_b6.sh'), [
    (
        '--seed "$seed" --epochs 100 --batch_size 16 --patience 15 \\\n'
        '            --exp_name "${exp_id}_s${seed}" \\',
        '--seed "$seed" --data_split_seed 42 --epochs 100 --batch_size 16 --patience 15 \\\n'
        '            --exp_name "${exp_id}_s${seed}" \\'
    ),
    (
        'echo " B6 COMPLETED: $(date)"\n'
        'echo "========================================="',
        'echo " B6 COMPLETED: $(date)"\n'
        'echo "========================================="\n'
        'echo ""\n'
        'echo "=== AUTO-CHAINING TO B7 ==="\n'
        'nohup bash scripts/launch_b7.sh > b7_output.log 2>&1 &\n'
        'echo "B7 launched in background"'
    ),
])

# ── B7 ──
fix_script(os.path.join(SCRIPTS, 'launch_b7.sh'), [
    (
        'CBESD_CKPT="checkpoints/b1/E1-02_s42/best_model.pt"      # self_attn seed42',
        'CBESD_CKPT="checkpoints/b1/E1-02_s42/best_model.pt"      # self_attn seed42 (WA=0.9292)'
    ),
    (
        'FAU_CKPT="checkpoints/b1/E1-05_s456/best_model.pt"       # self_attn seed456',
        'FAU_CKPT="checkpoints/b1/E1-05_s42/best_model.pt"        # self_attn seed42 (WA=0.6781)'
    ),
    (
        'IEMO_CKPT="checkpoints/b1/E1-08_s42/best_model.pt"       # placeholder',
        'IEMO_CKPT="checkpoints/b1/E1-09_s42/best_model.pt"       # prosody_guided seed42 (WA=0.6505)'
    ),
    (
        '--seed "$seed" --epochs 100 --batch_size 16 --patience 15 \\',
        '--seed "$seed" --data_split_seed 42 --epochs 100 --batch_size 16 --patience 15 \\'
    ),
])

print('\n=== ALL SCRIPTS UPDATED ===')
