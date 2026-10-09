"""Local PyTorch training on one NSL-KDD shard (M2 baseline + Flower client backend)."""

from __future__ import annotations

import argparse
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import MLPClassifier


@dataclass(frozen=True)
class TrainingSettings:
    local_epochs: int
    batch_size: int
    learning_rate: float
    random_seed: int


def load_project_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def shard_path(
    shards_dir: str | Path,
    partition: str,
    client_id: int,
) -> Path:
    return Path(shards_dir) / partition / f"client_{client_id}.npz"


def state_dict_to_ndarrays(state_dict: OrderedDict[str, torch.Tensor]) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy() for tensor in state_dict.values()]


def ndarrays_to_state_dict(
    keys: Sequence[str],
    ndarrays: Sequence[np.ndarray],
) -> OrderedDict[str, torch.Tensor]:
    if len(keys) != len(ndarrays):
        raise ValueError("keys and ndarrays length mismatch")
    return OrderedDict(
        (key, torch.tensor(array)) for key, array in zip(keys, ndarrays, strict=True)
    )


class LocalModel:
    """Train and evaluate the MLP on arrays (X, y) from one client shard."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        settings: TrainingSettings,
        device: torch.device | None = None,
    ) -> None:
        self.settings = settings
        self.device = device or torch.device("cpu")
        self._param_keys = tuple(
            MLPClassifier(input_dim, hidden_dims, num_classes).state_dict().keys()
        )
        self.model = MLPClassifier(input_dim, hidden_dims, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=settings.learning_rate
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any], device: torch.device | None = None) -> LocalModel:
        model_cfg = cfg["model"]
        train_cfg = cfg["training"]
        settings = TrainingSettings(
            local_epochs=int(train_cfg["local_epochs"]),
            batch_size=int(train_cfg["batch_size"]),
            learning_rate=float(train_cfg["learning_rate"]),
            random_seed=int(cfg["random_seed"]),
        )
        input_dim = int(model_cfg["input_dim"])
        if input_dim <= 0:
            raise ValueError("model.input_dim must be set in config (run preprocess first)")
        return cls(
            input_dim=input_dim,
            hidden_dims=list(model_cfg["hidden_dims"]),
            num_classes=int(model_cfg["num_classes"]),
            settings=settings,
            device=device,
        )

    def get_parameters(self) -> list[np.ndarray]:
        return state_dict_to_ndarrays(self.model.state_dict())

    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        state = ndarrays_to_state_dict(self._param_keys, parameters)
        self.model.load_state_dict(state, strict=True)

    def train_on(self, X: np.ndarray, y: np.ndarray) -> int:
        """Run local_epochs SGD on the shard; return number of training examples."""
        set_seed(self.settings.random_seed)
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.settings.batch_size,
            shuffle=True,
        )
        self.model.train()
        for _ in range(self.settings.local_epochs):
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()
        return int(len(y))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float]]:
        """Return (mean cross-entropy loss, sklearn metrics) on a fixed hold-out set."""
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            loss = float(self.criterion(logits, y_t).item())
            preds = logits.argmax(dim=1).cpu().numpy()
        metrics = {
            "f1": float(f1_score(y, preds, zero_division=0)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "accuracy": float((preds == y).mean()),
        }
        return loss, metrics


def run_local_baseline(
    client_id: int,
    config_path: str | Path = "src/config.yaml",
    *,
    override_shard_partition: str | None = None,
) -> dict[str, float]:
    """M2: train only on one shard, evaluate on the official global test set."""
    from src.preprocess import load_prepared_arrays
    from src.sharding import load_shard

    cfg = load_project_config(config_path)
    if client_id < 0 or client_id >= cfg["num_clients"]:
        raise ValueError(f"client_id must be between 0 and {cfg['num_clients'] - 1}")
    data_cfg = cfg["data"]
    if override_shard_partition is not None:
        data_cfg["shard_partition"] = override_shard_partition
    path = shard_path(data_cfg["shards_dir"], data_cfg["shard_partition"], client_id)
    shard = load_shard(path)
    X_train, y_train = shard["X"], shard["y"]

    processed = load_prepared_arrays(data_cfg["processed_dir"])
    X_test, y_test = processed["X_test"], processed["y_test"]

    local = LocalModel.from_config(cfg)
    n = local.train_on(X_train, y_train)
    loss, metrics = local.evaluate(X_test, y_test)
    print(f"client_{client_id}: trained on {n} samples from {path}")
    print(f"global test loss={loss:.4f}  F1={metrics['f1']:.4f}  acc={metrics['accuracy']:.4f}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Local baseline training on one FL shard (M2).")
    ap.add_argument("--cid", type=int, required=True, help="Client id 0..num_clients-1")
    ap.add_argument("--config", default="src/config.yaml")
    ap.add_argument("--shard_partition", default=None, help="Override config.data.shard_partition")
    args = ap.parse_args()
    run_local_baseline(args.cid, args.config, override_shard_partition=args.shard_partition)


if __name__ == "__main__":
    main()
