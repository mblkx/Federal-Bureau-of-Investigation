"""Klient Flower: trening na lokalnym shardzie, komunikacja z serwerem tylko wagami."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from flwr.client import NumPyClient, start_client

from src.sharding import load_shard
from src.training import (
    LocalModel,
    load_project_config,
    server_round_from_config,
    shard_path,
    split_shard_holdout,
)


class NSLKDDClient(NumPyClient):
    """Implementacja klienta Flower dla jednego shardu NSL-KDD."""

    def __init__(
        self,
        local: LocalModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        cat_eval: np.ndarray | None = None,
    ) -> None:
        """Przypina lokalny model oraz tablice treningowe / walidacyjne do klienta."""
        self.local = local
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.cat_eval = cat_eval

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        """Zwraca bieżące wagi modelu (MLP lub autoencoder).

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
        2. Trenuj na lokalnym 80% shardu z ziarnem zależnym od ``server_round`` (M2.2);
           Adam jest resetowany na początku fitu (domyślnie). AE trenuje tylko ``y==0``.
        3. Dla AE dobierz próg na lokalnej walidacji (max F1) — próg nie jest agregowany.
        4. Zwróć nowe wagi, ``num_examples`` oraz pusty słownik metryk.
        """
        self.local.set_parameters(parameters)
        num_examples = self.local.train_on(
            self.X_train,
            self.y_train,
            server_round=server_round_from_config(config),
        )
        if self.local.is_autoencoder:
            self.local.fit_threshold(self.X_eval, self.y_eval)
        return self.local.get_parameters(), num_examples, {}

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        """Ocenia model globalny na lokalnej walidacji (20% shardu), nie na KDDTest+.

        Kroki:
        1. Załaduj wagi z serwera.
        2. Dla AE dobierz ``τ`` na tej walidacji (dopasowanie do bieżących wag globalnych).
        3. Policz loss i metryki przez ``evaluate.evaluate``.
        """
        self.local.set_parameters(parameters)
        if self.local.is_autoencoder:
            self.local.fit_threshold(self.X_eval, self.y_eval)
        loss, metrics = self.local.evaluate(self.X_eval, self.y_eval, self.cat_eval)
        return loss, len(self.y_eval), metrics


def build_client(client_id: int, config_path: str | Path) -> NSLKDDClient:
    """Tworzy klienta Flower dla danego ``client_id`` (bez ładowania KDDTest+).

    Kroki:
    1. Wczytaj config i shard ``client_{id}.npz`` z katalogu partycji.
    2. Podziel shard 80/20 (stratyfikacja po kategorii) na trening i walidację lokalną.
    3. Zbuduj ``LocalModel`` (MLP lub AE wg configu) i opakuj dane w ``NSLKDDClient``.
    """
    cfg = load_project_config(config_path)
    data_cfg = cfg["data"]
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
    return NSLKDDClient(
        local=local,
        X_train=X_tr,
        y_train=y_tr,
        X_eval=X_va,
        y_eval=y_va,
        cat_eval=cat_va,
    )


def main() -> None:
    """Uruchamia klienta Flower lub lokalny baseline bez serwera.

    Kroki:
    1. Parsuj ``--cid``, config, adres serwera i flagę ``--local-only``.
    2. W trybie ``local-only`` wywołaj ``run_local_baseline`` i zakończ.
    3. W trybie federacji zbuduj klienta i połącz się przez ``start_client``.
    """
    ap = argparse.ArgumentParser(description="Uruchom klienta Flower dla FL na NSL-KDD.")
    ap.add_argument("--cid", type=int, required=True, help="Id klienta 0..num_clients-1")
    ap.add_argument("--config", default="src/config.yaml")
    ap.add_argument("--server-address", default="localhost:8080")
    ap.add_argument(
        "--local-only",
        action="store_true",
        help="Pomiń Flower; uruchom lokalny baseline M2 tylko na tym shardzie",
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
