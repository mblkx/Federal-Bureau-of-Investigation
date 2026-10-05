"""NSL-KDD loading, preprocessing, and client sharding (to be implemented in M1)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def split_train_into_shards(
    n_samples: int,
    num_clients: int,
    seed: int,
) -> list[np.ndarray]:
    """Return disjoint index arrays partitioning range(n_samples) among clients."""
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return np.array_split(indices, num_clients)


def load_config_paths(shards_dir: Path) -> Tuple[Path, ...]:
    """Placeholder for shard file layout used by clients."""
    return tuple(shards_dir.glob("client_*.parquet"))
