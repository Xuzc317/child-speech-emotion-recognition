# Shutdown/Resume Quickstart

If you need to shut down now, use this one command next boot:

```bash
python -u scripts/tmp_paramiko_autodl_runner.py --auto-resume
```

On Windows you can also double-click:

`scripts/one_click_resume_and_sync.bat`

## What the command does

1. Connects to AutoDL via Paramiko.
2. Checks whether any training process is still running.
3. Checks which core logs are still missing:
   - `exp1_self_attention.json`
   - `exp2_prosody_guided.json`
   - `exp3_adult_iemocap.json`
   - `exp4_zero_shot_fau.json`
   - `exp5_fau_indomain.json`
   - `exp5b_self_attention_fau.json`
4. If logs are missing and no process is running, it relaunches only missing experiments.
5. Syncs remote `results/logs` back to local `results_remote/results/logs`.
6. Syncs `xai_raw_data.npz` to local `publication_package/xai_raw_data.npz` when available.

## Notes

- Remote jobs are started with `nohup`, so they continue after local shutdown.
- Auto-resume enforces safe thread env vars (`OMP_NUM_THREADS=1`, etc.) to reduce `libgomp` issues.
- C-BESD is expected at remote:
  `/root/autodl-tmp/datasets/BESD/BESD/MY`
