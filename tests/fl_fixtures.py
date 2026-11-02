"""Syntetyczne shardy i config — wspólne dla testów integracyjnych i privacy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def write_processed(tmp_path: Path, dim: int = 6) -> Path:
    """Tworzy minimalny ``nsl_kdd_processed.npz`` z syntetycznym testem.

    Kroki:
    1. Wygeneruj ``X_test``, binarne ``y_test`` i kategorie ``cat_test``.
    2. Zapisz plik w ``tmp_path/processed/`` i zwróć ten katalog.
    """
    rng = np.random.default_rng(0)
    n_test = 40
    X_test = rng.standard_normal((n_test, dim), dtype=np.float32)
    y_test = (X_test[:, 0] > 0).astype(np.int64)
    cat_test = np.array(["normal" if y == 0 else "DoS" for y in y_test])
    out = tmp_path / "processed"
    out.mkdir()
    np.savez_compressed(
        out / "nsl_kdd_processed.npz",
        X_test=X_test,
        y_test=y_test,
        cat_test=cat_test,
    )
    return out


def write_shards(tmp_path: Path, dim: int = 6, num_clients: int = 3) -> Path:
    """Tworzy ``num_clients`` plików shardów IID w ``shards/iid/``.

    Kroki:
    1. Dla każdego ``client_id`` wygeneruj losowe ``X``, ``y`` i zapisz ``client_{id}.npz``.
    2. Zwróć katalog nadrzędny ``shards`` (ścieżka w configu ``data.shards_dir``).
    """
    rng = np.random.default_rng(1)
    shard_dir = tmp_path / "shards" / "iid"
    shard_dir.mkdir(parents=True)
    for cid in range(num_clients):
        n = 60
        X = rng.standard_normal((n, dim), dtype=np.float32)
        y = (X[:, 0] > 0).astype(np.int64)
        np.savez_compressed(shard_dir / f"client_{cid}.npz", X=X, y=y)
    return tmp_path / "shards"


def integration_fl_config_dict(
    tmp_path: Path,
    processed: Path,
    shards: Path,
    *,
    num_rounds: int = 1,
) -> dict:
    """Buduje słownik configu YAML zgodny z ``LocalModel`` i ścieżkami fixture.

    Kroki:
    1. Ustaw ``data.processed_dir``, ``data.shards_dir``, ``shard_partition=iid``.
    2. Uzupełnij sekcje ``model``, ``training``, ``federated``, ``evaluation``.
    3. Zwróć dict gotowy do zapisu lub bezpośredniego użycia w testach.
    """
    return {
        "random_seed": 0,
        "num_clients": 3,
        "data": {
            "processed_dir": str(processed),
            "shards_dir": str(shards),
            "shard_partition": "iid",
        },
        "model": {"input_dim": 6, "hidden_dims": [8], "num_classes": 2},
        "training": {"local_epochs": 1, "batch_size": 16, "learning_rate": 0.05},
        "federated": {"num_rounds": num_rounds},
        "evaluation": {"f1_threshold": 0.89},
    }


def write_integration_fl_config(tmp_path: Path, *, num_rounds: int = 1) -> Path:
    """Zapisuje kompletny ``config.yaml`` pod szybkie testy FL.

    Kroki:
    1. Wywołaj ``write_processed`` i ``write_shards``.
    2. Złóż config przez ``integration_fl_config_dict``.
    3. Zapisz YAML w ``tmp_path/config.yaml`` i zwróć ścieżkę do pliku.
    """
    processed = write_processed(tmp_path)
    shards = write_shards(tmp_path)
    config = integration_fl_config_dict(tmp_path, processed, shards, num_rounds=num_rounds)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    return path
