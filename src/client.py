"""Flower client — local training on one NSL-KDD shard (M2/M3)."""

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
    """Federated client: fit on shard (X, y); evaluate on official global test."""

    def __init__(
        self,
        local: LocalModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
    ) -> None:
        self.local = local
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return self.local.get_parameters()

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        self.local.set_parameters(parameters)
        num_examples = self.local.train_on(self.X_train, self.y_train)
        return self.local.get_parameters(), num_examples, {}

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        self.local.set_parameters(parameters)
        loss, metrics = self.local.evaluate(self.X_eval, self.y_eval)
        return loss, len(self.y_eval), metrics


def build_client(client_id: int, config_path: str | Path) -> NSLKDDClient:
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
