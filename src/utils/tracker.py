"""Unified experiment tracking: WandB online + local CSV fallback.

Usage:
    tracker = ExperimentTracker(model=model, project="child-speech-emotion",
                                config=flat_config_dict)
    tracker.log({"train/loss": 1.5, "val/acc": 0.72}, step=5)
    tracker.finish()
"""

import csv
import os
import time


class ExperimentTracker:
    def __init__(self, model=None, project="child-speech-emotion",
                 entity=None, config=None, log_dir="experiments/logs",
                 run_name=None):
        self.use_wandb = False
        self.wandb_run = None
        self.local_metrics = []
        self.start_time = time.time()

        os.makedirs(log_dir, exist_ok=True)

        # Try WandB
        try:
            import wandb
            mode = "online" if entity else "offline"
            self.wandb_run = wandb.init(
                project=project,
                entity=entity,
                config=config or {},
                name=run_name,
                mode=mode,
            )
            if model is not None:
                wandb.watch(model, log="gradients", log_freq=100)
            self.use_wandb = True
            print(f"[Tracker] WandB initialized (mode={mode}, project={project})")
        except Exception as e:
            print(f"[Tracker] WandB unavailable ({e}), using local CSV only.")

        # Local CSV file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = run_name or f"run_{timestamp}"
        self.csv_path = os.path.join(log_dir, f"{name}.csv")
        self._csv_written_header = False
        print(f"[Tracker] Local log: {self.csv_path}")

    def log(self, metrics, step=None):
        if step is None:
            step = len(self.local_metrics)
        entry = {**metrics, "step": step}

        if self.use_wandb:
            self.wandb_run.log(metrics, step=step)

        self.local_metrics.append(entry)
        self._write_csv_row(entry)

    def _write_csv_row(self, entry):
        mode = "a" if self._csv_written_header else "w"
        with open(self.csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(entry.keys()))
            if not self._csv_written_header:
                writer.writeheader()
                self._csv_written_header = True
            writer.writerow(entry)

    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"[Tracker] Finished. Duration: {elapsed/60:.1f} min. "
              f"Metrics saved to {self.csv_path}")
        if self.use_wandb:
            self.wandb_run.finish()
