"""Lokalny trening PyTorch na jednym shardzie NSL-KDD (baseline M2 + backend klienta Flower)."""

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
from src.evaluate import format_report, select_threshold_max_f1, stratified_holdout
from src.model import Autoencoder, MLPClassifier

# Udział holdoutu lokalnego (plan M2.1 / evaluation_protocol.md): trening 80%, walidacja 20%.
VAL_FRACTION = 0.2
ATTACK_SCORE_THRESHOLD = 0.5

# M2.2: po FedAvg wagi globalne „skaczą”; zachowanie momentów Adama między rundami
# oznaczałoby użycie momentum/wariancji z innej trajektorii parametrów. Reset przy każdym lokalnym fit.
DEFAULT_RESET_OPTIMIZER_EACH_ROUND = True

ARCHITECTURE_MLP = "mlp"
ARCHITECTURE_AUTOENCODER = "autoencoder"
ARCHITECTURES = (ARCHITECTURE_MLP, ARCHITECTURE_AUTOENCODER)


@dataclass(frozen=True)
class TrainingSettings:
    local_epochs: int
    batch_size: int
    learning_rate: float
    random_seed: int
    reset_optimizer_each_round: bool = DEFAULT_RESET_OPTIMIZER_EACH_ROUND


def load_project_config(path: str | Path) -> dict[str, Any]:
    """Wczytaj YAML i uzupełnij ścieżki preprocessingu / ``model.input_dim`` na podstawie danych."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return resolve_data_config(raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def round_seed(base_seed: int, server_round: int) -> int:
    """Ziarno RNG zależne od rundy: ``base_seed + server_round`` (Flower numeruje rundy od 1).

    Baseline lokalny (bez FL) używa ``server_round=0``, więc efektywne ziarno
    równa się ``config.random_seed``.
    """
    if server_round < 0:
        raise ValueError(f"server_round must be >= 0, got {server_round}")
    return int(base_seed) + int(server_round)


def server_round_from_config(config: dict[str, Any] | None) -> int:
    """Odczytaj ``server_round`` z konfiguracji fit/evaluate Flower (domyślnie 0)."""
    if not config:
        return 0
    if "server_round" not in config:
        return 0
    return int(config["server_round"])


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
    """Podziel shard klienta na lokalny train/val (stratyfikacja po kategorii)."""
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
    """Spłaszcz wynik ``evaluate.evaluate`` do floatów nadających się do agregacji Flower."""
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


def _resolve_architecture(model_cfg: dict[str, Any]) -> str:
    architecture = str(model_cfg.get("architecture", ARCHITECTURE_MLP)).lower()
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"model.architecture must be one of {ARCHITECTURES}, got '{architecture}'"
        )
    return architecture


def _default_latent_dim(hidden_dims: list[int], model_cfg: dict[str, Any]) -> int:
    raw = model_cfg.get("latent_dim")
    if raw is not None:
        return int(raw)
    # Domyślnie połowa ostatniej warstwy ukrytej (co najmniej 8).
    return max(8, int(hidden_dims[-1]) // 2)


class LocalModel:
    """Jednolity interfejs lokalnego modelu: MLP albo autoencoder (M2.3).

    Wspólne API wag: ``get_parameters`` / ``set_parameters`` (lista ``ndarray``).
    Wynik ataku: ``predict_attack_score`` (MLP: P(atak); AE: MSE rekonstrukcji).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        settings: TrainingSettings,
        device: torch.device | None = None,
        *,
        architecture: str = ARCHITECTURE_MLP,
        latent_dim: int | None = None,
    ) -> None:
        architecture = architecture.lower()
        if architecture not in ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {ARCHITECTURES}, got '{architecture}'"
            )
        self.architecture = architecture
        self.settings = settings
        self.device = device or torch.device("cpu")
        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims)
        self.num_classes = int(num_classes)

        if architecture == ARCHITECTURE_MLP:
            self.model: nn.Module = MLPClassifier(
                input_dim, self.hidden_dims, num_classes
            ).to(self.device)
            self.criterion: nn.Module = nn.CrossEntropyLoss()
            self.threshold = ATTACK_SCORE_THRESHOLD
            self.latent_dim: int | None = None
        else:
            if latent_dim is None:
                latent_dim = max(8, int(self.hidden_dims[-1]) // 2)
            self.latent_dim = int(latent_dim)
            self.model = Autoencoder(
                input_dim, self.hidden_dims, self.latent_dim
            ).to(self.device)
            self.criterion = nn.MSELoss()
            # Próg dobierany na walidacji przez ``fit_threshold``; startowo NaN = nieustawiony.
            self.threshold = float("nan")

        self._param_keys = tuple(self.model.state_dict().keys())
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
            reset_optimizer_each_round=bool(
                train_cfg.get(
                    "reset_optimizer_each_round",
                    DEFAULT_RESET_OPTIMIZER_EACH_ROUND,
                )
            ),
        )
        raw_dim = model_cfg.get("input_dim")
        if raw_dim is None:
            raise ValueError(
                "model.input_dim is unset; run preprocess/build_data or set it in config"
            )
        input_dim = int(raw_dim)
        if input_dim <= 0:
            raise ValueError("model.input_dim must be set in config (run preprocess first)")
        architecture = _resolve_architecture(model_cfg)
        hidden_dims = list(model_cfg["hidden_dims"])
        latent_dim = (
            _default_latent_dim(hidden_dims, model_cfg)
            if architecture == ARCHITECTURE_AUTOENCODER
            else None
        )
        return cls(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=int(model_cfg["num_classes"]),
            settings=settings,
            device=device,
            architecture=architecture,
            latent_dim=latent_dim,
        )

    @property
    def is_autoencoder(self) -> bool:
        return self.architecture == ARCHITECTURE_AUTOENCODER

    def get_parameters(self) -> list[np.ndarray]:
        return state_dict_to_ndarrays(self.model.state_dict())

    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        state = ndarrays_to_state_dict(self._param_keys, parameters)
        self.model.load_state_dict(state, strict=True)

    def reset_optimizer(self) -> None:
        """Utwórz Adama od nowa, żeby momenty nie przenosiły się po aktualizacji wag globalnych."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.settings.learning_rate
        )

    def train_on(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        server_round: int = 0,
    ) -> int:
        """Trenuje lokalny model przez ``local_epochs`` epok.

        MLP: klasyfikacja na całym ``(X, y)``.
        Autoencoder (M2.3): wyłącznie wiersze ``y == 0`` (ruch normalny), loss = MSE
        rekonstrukcji. ``num_examples`` = liczba próbek faktycznie użytych w treningu.
        """
        seed = round_seed(self.settings.random_seed, server_round)
        set_seed(seed)
        if self.settings.reset_optimizer_each_round:
            self.reset_optimizer()

        if self.is_autoencoder:
            mask = np.asarray(y).astype(int) == 0
            if not np.any(mask):
                raise ValueError(
                    "autoencoder wymaga co najmniej jednej próbki normalnej (y==0) w treningu"
                )
            X_fit = np.asarray(X, dtype=np.float32)[mask]
            X_t = torch.as_tensor(X_fit, dtype=torch.float32, device=self.device)
            dataset: TensorDataset = TensorDataset(X_t)
        else:
            X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
            dataset = TensorDataset(X_t, y_t)
            X_fit = X

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=self.settings.batch_size,
            shuffle=True,
            generator=generator,
        )
        self.model.train()
        for _ in range(self.settings.local_epochs):
            for batch in loader:
                self.optimizer.zero_grad()
                if self.is_autoencoder:
                    (batch_x,) = batch
                    recon = self.model(batch_x)
                    loss = self.criterion(recon, batch_x)
                else:
                    batch_x, batch_y = batch
                    logits = self.model(batch_x)
                    loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()
        return int(len(X_fit))

    def reconstruction_mse(self, X: np.ndarray) -> np.ndarray:
        """MSE rekonstrukcji per wiersz (wynik ataku autoencodera)."""
        if not self.is_autoencoder:
            raise TypeError("reconstruction_mse jest dostępne tylko dla autoencodera")
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            recon = self.model(X_t)
            mse = torch.mean((recon - X_t) ** 2, dim=1)
        return mse.cpu().numpy()

    def predict_attack_score(self, X: np.ndarray) -> np.ndarray:
        """Wynik ataku ``s``: MLP → P(atak); autoencoder → MSE rekonstrukcji."""
        if self.is_autoencoder:
            return self.reconstruction_mse(X)
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(X_t), dim=1)[:, 1]
        return probs.cpu().numpy()

    def predict_attack_proba(self, X: np.ndarray) -> np.ndarray:
        """Alias historyczny: to samo co ``predict_attack_score`` (dla MLP = prawdopodobieństwo)."""
        return self.predict_attack_score(X)

    def fit_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        max_candidates: int = 256,
    ) -> float:
        """Dobierz ``τ`` na walidacji (AE: max F1; MLP: zawsze 0,5)."""
        if not self.is_autoencoder:
            self.threshold = ATTACK_SCORE_THRESHOLD
            return self.threshold
        scores = self.predict_attack_score(X)
        self.threshold = select_threshold_max_f1(
            y, scores, max_candidates=max_candidates
        )
        return self.threshold

    def _active_threshold(self, threshold: float | None) -> float:
        if threshold is not None:
            return float(threshold)
        if self.is_autoencoder and (isinstance(self.threshold, float) and math.isnan(self.threshold)):
            raise ValueError(
                "próg autoencodera nieustawiony — wywołaj fit_threshold na walidacji"
            )
        return float(self.threshold)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categories: np.ndarray | None = None,
        *,
        threshold: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Oceń hold-out: loss treningowy + metryki z ``evaluate.evaluate``."""
        tau = self._active_threshold(threshold)
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.is_autoencoder:
                recon = self.model(X_t)
                loss = float(self.criterion(recon, X_t).item())
                scores = torch.mean((recon - X_t) ** 2, dim=1).cpu().numpy()
            else:
                y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
                logits = self.model(X_t)
                loss = float(self.criterion(logits, y_t).item())
                scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        result = score_predictions(y, scores, categories, threshold=tau)
        return loss, scalar_metrics(result)

    def evaluate_full(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categories: np.ndarray | None = None,
        *,
        threshold: float | None = None,
    ) -> dict[str, object]:
        """Pełny słownik ``evaluate.evaluate`` (w tym ``by_category``) do raportu lokalnego."""
        tau = self._active_threshold(threshold)
        scores = self.predict_attack_score(X)
        return score_predictions(y, scores, categories, threshold=tau)


def run_local_baseline(
    client_id: int,
    config_path: str | Path = "src/config.yaml",
    *,
    override_shard_partition: str | None = None,
    override_architecture: str | None = None,
) -> dict[str, object]:
    """M2.1/M2.3: trenuj na 80% shardu, oceń na lokalnych 20% walidacji.

    Nie ładuje ``KDDTest+`` (zarezerwowany na raport końcowy po stronie serwera).
    Dla autoencodera próg ``τ`` dobierany jest na walidacji (max F1).
    """
    from src.sharding import load_shard

    cfg = load_project_config(config_path)
    if client_id < 0 or client_id >= cfg["num_clients"]:
        raise ValueError(f"client_id must be between 0 and {cfg['num_clients'] - 1}")
    data_cfg = cfg["data"]
    if override_shard_partition is not None:
        data_cfg["shard_partition"] = override_shard_partition
    if override_architecture is not None:
        cfg["model"]["architecture"] = override_architecture
    path = shard_path(data_cfg["shards_dir"], data_cfg["shard_partition"], client_id)
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
    # Baseline lokalny to pojedynczy fit: server_round=0 → ziarno == config.random_seed.
    n = local.train_on(X_tr, y_tr, server_round=0)
    if local.is_autoencoder:
        local.fit_threshold(X_va, y_va)
    loss, metrics = local.evaluate(X_va, y_va, cat_va)
    full = local.evaluate_full(X_va, y_va, cat_va)
    arch = local.architecture
    print(f"client_{client_id} [{arch}]: trained on {n} samples from {path}")
    print(
        f"local validation loss={loss:.4f}  n_val={len(y_va)}  threshold={local.threshold:.6g}"
    )
    print(format_report(full, title=f"client_{client_id} local validation ({arch})"))
    return {
        **full,
        "architecture": arch,
        "loss": loss,
        "n_train": n,
        "n_val": int(len(y_va)),
        **metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Baseline lokalny: trening na 80% shardu, walidacja na 20% (M2)."
    )
    ap.add_argument("--cid", type=int, required=True, help="Id klienta 0..num_clients-1")
    ap.add_argument("--config", default="src/config.yaml")
    ap.add_argument(
        "--shard_partition",
        default=None,
        help="Nadpisz config.data.shard_partition",
    )
    ap.add_argument(
        "--architecture",
        choices=list(ARCHITECTURES),
        default=None,
        help="Nadpisz model.architecture (mlp | autoencoder)",
    )
    args = ap.parse_args()
    run_local_baseline(
        args.cid,
        args.config,
        override_shard_partition=args.shard_partition,
        override_architecture=args.architecture,
    )


if __name__ == "__main__":
    main()
