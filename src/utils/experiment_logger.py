"""实验日志系统 — 每次运行自动记录，可追溯、可复现。

每次训练自动生成:
  experiments/runs/{timestamp}_{name}/
    ├── config.json      # 超参数快照
    ├── metrics.csv      # 每 epoch train/val 指标
    └── results.json     # 最终 val/test 结果 + 时长

全局检索:
  experiments/registry.csv  # 所有实验一行摘要，快速对比

用法:
    logger = ExperimentLogger(name="B3_seed42", config={...})
    logger.log_epoch(epoch, train_acc, val_acc)
    logger.finish(best_val, test_acc)
"""

import os, csv, json, time
from datetime import datetime


class ExperimentLogger:
    def __init__(self, name=None, config=None, base_dir="experiments"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = name or f"run_{self.timestamp}"
        self.run_dir = os.path.join(base_dir, "runs", f"{self.timestamp}_{self.name}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.config = config or {}
        self.metrics = []
        self.start_time = time.time()

        # Save config snapshot
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump({**self.config, "name": self.name, "timestamp": self.timestamp},
                      f, indent=2, ensure_ascii=False)

        # Init metrics CSV
        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        self._csv_header_written = False

        print(f"[Logger] {self.name} → {self.run_dir}")

    def log_epoch(self, epoch, train_acc=None, val_acc=None, test_acc=None, lr=None):
        entry = {"epoch": epoch, "timestamp": datetime.now().isoformat()}
        if train_acc is not None: entry["train_acc"] = round(float(train_acc), 6)
        if val_acc is not None:   entry["val_acc"] = round(float(val_acc), 6)
        if test_acc is not None:  entry["test_acc"] = round(float(test_acc), 6)
        if lr is not None:        entry["lr"] = float(lr)
        self.metrics.append(entry)

        # Write CSV row
        keys = list(entry.keys())
        mode = "a" if self._csv_header_written else "w"
        with open(self.csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(entry)

    def finish(self, best_val=None, test_acc=None):
        elapsed = time.time() - self.start_time
        results = {
            "name": self.name,
            "timestamp": self.timestamp,
            "duration_minutes": round(elapsed / 60, 1),
            "total_epochs": len(self.metrics),
            "best_val_acc": round(float(best_val), 6) if best_val is not None else None,
            "test_acc": round(float(test_acc), 6) if test_acc is not None else None,
        }

        # Save results
        with open(os.path.join(self.run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Append to global registry
        registry_path = os.path.join(os.path.dirname(self.run_dir), "..", "registry.csv")
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        registry_exists = os.path.exists(registry_path)
        with open(registry_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results.keys()))
            if not registry_exists:
                writer.writeheader()
            writer.writerow(results)

        print(f"[Logger] {self.name} finished in {elapsed/60:.1f}min")
        if best_val is not None:
            print(f"[Logger]   best_val={best_val:.4f}")
        if test_acc is not None:
            print(f"[Logger]   test_acc={test_acc:.4f}")
        print(f"[Logger]   log: {self.run_dir}")
        return results
