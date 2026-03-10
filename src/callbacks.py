import os
import torch
from glob import glob


class ModelCheckpoint:
    """Saves the best model checkpoint to disk based on a monitored metric."""

    def __init__(self, save_dir: str, monitor: str = "val_loss", mode: str = "min", max_models: int = 2):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.max_models = max_models
        self.best_score = float("inf") if mode == "min" else -float("inf")
        self.best_model_path = None
        os.makedirs(self.save_dir, exist_ok=True)

    def _remove_oldest(self):
        files = sorted(glob(f"{self.save_dir}/*.pth"), key=os.path.getmtime)
        if len(files) > self.max_models:
            os.remove(files[0])

    def step(self, epoch: int, metric: float, accuracy: float, model: torch.nn.Module):
        improved = (self.mode == "min" and metric < self.best_score) or (
            self.mode == "max" and metric > self.best_score
        )
        if improved:
            self.best_score = metric
            name = f"epoch={epoch}_val_loss={metric:.2f}_val_acc={accuracy:.2f}.pth"
            path = os.path.join(self.save_dir, name)
            torch.save(model.state_dict(), path)
            self.best_model_path = path
            self._remove_oldest()
            print(f"  Checkpoint saved: {path}")


class EarlyStopping:
    """Stops training when the monitored metric stops improving."""

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 0.003):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = float("inf") if mode == "min" else -float("inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        improved = (self.mode == "min" and metric < self.best_score - self.min_delta) or (
            self.mode == "max" and metric > self.best_score + self.min_delta
        )
        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
