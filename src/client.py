"""Klient Flower: trening na lokalnym shardzie, komunikacja z serwerem tylko wagami."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from flwr.client import NumPyClient, start_client

from src.preprocess import load_prepared_arrays
from src.sharding import load_shard
from src.training import LocalModel, load_project_config, shard_path


class NSLKDDClient(NumPyClient):
    """Implementacja klienta Flower dla jednego shardu NSL-KDD."""

    def __init__(
        self,
        local: LocalModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
    ) -> None:
        """Przypina lokalny model i tablice treningowe / ewaluacyjne do instancji klienta."""
        self.local = local
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        """Zwraca bieżące wagi MLP (np. gdy serwer pyta przed rundą).

        Kroki:
        1. Odczytaj ``state_dict`` z ``LocalModel`` i serializuj do listy ``ndarray``.
        """
        return self.local.get_parameters()

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        """Wykonuje lokalną rundę treningu i zwraca zaktualizowane wagi.

        Kroki:
        1. Załaduj wagi globalne z serwera do ``LocalModel``.
        2. Trenuj wyłącznie na ``X_train`` / ``y_train`` (dane nie opuszczają procesu).
        3. Zwróć nowe wagi, ``len(y_train)`` jako ``num_examples`` oraz pusty słownik metryk.
        """
        self.local.set_parameters(parameters)
        num_examples = self.local.train_on(self.X_train, self.y_train)
        return self.local.get_parameters(), num_examples, {}

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        """Ocenia model globalny na wspólnym zbiorze testowym (KDDTest+).

        Kroki:
        1. Załaduj wagi z serwera.
        2. Policz loss i metryki sklearn na ``X_eval`` / ``y_eval``.
        3. Zwróć loss, liczbę próbek testowych (do ważenia na serwerze) i słownik metryk.
        """
        self.local.set_parameters(parameters)
        loss, metrics = self.local.evaluate(self.X_eval, self.y_eval)
        return loss, len(self.y_eval), metrics


def build_client(client_id: int, config_path: str | Path) -> NSLKDDClient:
    """Tworzy klienta Flower dla danego ``client_id``.

    Kroki:
    1. Wczytaj config i shard ``client_{id}.npz`` z katalogu partycji.
    2. Wczytaj globalny zbiór testowy z ``processed_dir``.
    3. Zbuduj ``LocalModel`` i opakuj dane w ``NSLKDDClient``.
    """
    cfg = load_project_config(config_path)
    data_cfg = cfg["data"]
    shard = load_shard(
        shard_path(data_cfg["shards_dir"], data_cfg["shard_partition"], client_id)
    )
    processed = load_prepared_arrays(data_cfg["processed_dir"])
    local = LocalModel.from_config(cfg)
    return NSLKDDClient(
        local=local,
        X_train=shard["X"],
        y_train=shard["y"],
        X_eval=processed["X_test"],
        y_eval=processed["y_test"],
    )


def main() -> None:
    """Uruchamia klienta Flower lub lokalny baseline bez serwera.

    Kroki:
    1. Parsuj ``--cid``, config, adres serwera i flagę ``--local-only``.
    2. W trybie ``local-only`` wywołaj ``run_local_baseline`` i zakończ.
    3. W trybie federacji zbuduj klienta i połącz się przez ``start_client``.
    """
    ap = argparse.ArgumentParser(description="Start a Flower client for NSL-KDD FL.")
    ap.add_argument("--cid", type=int, required=True, help="Client id 0..num_clients-1")
    ap.add_argument("--config", default="src/config.yaml")
    ap.add_argument("--server-address", default="localhost:8080")
    ap.add_argument(
        "--local-only",
        action="store_true",
        help="Skip Flower; run M2 local baseline on this shard only",
    )
    args = ap.parse_args()

    if args.local_only:
        from src.training import run_local_baseline

        run_local_baseline(args.cid, args.config)
        return

    flower_client = build_client(args.cid, args.config)
    start_client(
        server_address=args.server_address,
        client=flower_client.to_client(),
    )


if __name__ == "__main__":
    main()
