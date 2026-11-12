"""Local PyTorch training on one NSL-KDD shard (M2 baseline + Flower client backend)."""

from __future__ import annotations

import argparse
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_config import resolve_data_config
from src.evaluate import evaluate as score_predictions
from src.evaluate import format_report, stratified_holdout
from src.model import MLPClassifier

# Local holdout share (plan M2.1 / evaluation_protocol.md): train on 80%, validate on 20%.
VAL_FRACTION = 0.2
ATTACK_SCORE_THRESHOLD = 0.5


@dataclass(frozen=True)
class TrainingSettings:
    local_epochs: int
    batch_size: int
    learning_rate: float
    random_seed: int


def load_project_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve preprocessing paths / ``model.input_dim`` from data."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return resolve_data_config(raw)


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


def split_shard_holdout(
    X: np.ndarray,
    y: np.ndarray,
    categories: np.ndarray,
    seed: int,
    *,
    val_fraction: float = VAL_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split one client shard into local train / validation (stratified by category)."""
    train_idx, val_idx = stratified_holdout(categories, val_fraction, seed)
    return (
        X[train_idx],
        y[train_idx],
        categories[train_idx],
        X[val_idx],
        y[val_idx],
        categories[val_idx],
    )


def scalar_metrics(result: dict[str, object]) -> dict[str, float]:
    """Flatten ``evaluate.evaluate`` output to floats suitable for Flower aggregation."""
    out: dict[str, float] = {}
    for key, value in result.items():
        if key == "by_category":
            continue
        if isinstance(value, (bool, np.bool_)):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if math.isnan(number):
                continue
            out[key] = number
    return out


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
        raw_dim = model_cfg.get("input_dim")
        if raw_dim is None:
            raise ValueError(
                "model.input_dim is unset; run preprocess/build_data or set it in config"
            )
        input_dim = int(raw_dim)
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
        """Trenuje MLP na lokalnym zbiorze treningowym przez ``local_epochs`` epok SGD.

        Kroki:
        1. Ustaw seed i zbuduj ``DataLoader`` z deterministycznym tasowaniem batchy.
        2. Dla każdej epoki: pętla po batchach, forward, loss, backward, krok Adam.
        3. Zwróć ``len(y)`` — Flower traktuje to jako ``num_examples`` w weighted FedAvg.
        """
        set_seed(self.settings.random_seed)
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.settings.random_seed)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.settings.batch_size,
            shuffle=True,
            generator=generator,
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

    def predict_attack_proba(self, X: np.ndarray) -> np.ndarray:
        """Zwraca prawdopodobieństwo klasy „atak” (indeks 1) dla każdego wiersza.

        Kroki:
        1. Przełącz model w tryb eval i przenieś ``X`` na tensory na ``self.device``.
        2. Policz softmax na logitach i wybierz kolumnę klasy ataku.
        3. Zwróć wektor P(atak) jako ``numpy`` (wejście do ``evaluate.evaluate``).
        """
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(X_t), dim=1)[:, 1]
        return probs.cpu().numpy()

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categories: np.ndarray | None = None,
        *,
        threshold: float = ATTACK_SCORE_THRESHOLD,
    ) -> tuple[float, dict[str, float]]:
        """Score a hold-out set: CE loss plus ``evaluate.evaluate`` metrics (τ = 0.5)."""
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            loss = float(self.criterion(logits, y_t).item())
            scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        result = score_predictions(y, scores, categories, threshold=threshold)
        return loss, scalar_metrics(result)

    def evaluate_full(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categories: np.ndarray | None = None,
        *,
        threshold: float = ATTACK_SCORE_THRESHOLD,
    ) -> dict[str, object]:
        """Full ``evaluate.evaluate`` dict (incl. ``by_category``) for local reporting."""
        scores = self.predict_attack_proba(X)
        return score_predictions(y, scores, categories, threshold=threshold)


def run_local_baseline(
    client_id: int,
    config_path: str | Path = "src/config.yaml",
    *,
    override_shard_partition: str | None = None,
) -> dict[str, object]:
    """M2.1: train on 80% of one shard, evaluate on the local 20% validation split.

    Does not load ``KDDTest+`` (reserved for server-side / final reporting).
    """
    from src.sharding import load_shard

    cfg = load_project_config(config_path)
    if client_id < 0 or client_id >= cfg["num_clients"]:
        raise ValueError(f"client_id must be between 0 and {cfg['num_clients'] - 1}")
    data_cfg = cfg["data"]
    if override_shard_partition is not None:
        data_cfg["shard_partition"] = override_shard_partition
    path = shard_path(data_cfg["shards_dir"], data_cfg["shard_partition"], client_id)
    print(data_cfg["shards_dir"], data_cfg["shard_partition"], client_id)
    shard = load_shard(path)
    if "cat" not in shard:
        raise KeyError(f"shard {path} lacks 'cat'; re-run sharding / build_data")

    X_tr, y_tr, _cat_tr, X_va, y_va, cat_va = split_shard_holdout(
        shard["X"],
        shard["y"],
        shard["cat"],
        int(cfg["random_seed"]),
    )

    local = LocalModel.from_config(cfg)
    n = local.train_on(X_tr, y_tr)
    loss, metrics = local.evaluate(X_va, y_va, cat_va)
    full = local.evaluate_full(X_va, y_va, cat_va)
    print(f"client_{client_id}: trained on {n} samples from {path}")
    print(f"local validation loss={loss:.4f}  n_val={len(y_va)}")
    print(format_report(full, title=f"client_{client_id} local validation"))
    return {**full, "loss": loss, "n_train": n, "n_val": int(len(y_va)), **metrics}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Local baseline: train on 80% of one shard, validate on 20% (M2.1)."
    )
    ap.add_argument("--cid", type=int, required=True, help="Client id 0..num_clients-1")
    ap.add_argument("--config", default="src/config.yaml")
    ap.add_argument("--shard_partition", default=None, help="Override config.data.shard_partition")
    args = ap.parse_args()
    run_local_baseline(args.cid, args.config, override_shard_partition=args.shard_partition)


if __name__ == "__main__":
    main()
