"""Partition coverage (legacy name; logic lives in src/sharding.py)."""

import numpy as np

from src.sharding import make_partition


def test_split_covers_all_samples_once():
    n = 999
    cats = np.array(["normal", "DoS", "Probe"] * (n // 3), dtype=str)
    shards = make_partition("iid", cats, num_clients=3, seed=42)
    merged = np.concatenate(shards)
    assert len(merged) == n
    assert len(np.unique(merged)) == n
    assert merged.min() == 0
    assert merged.max() == n - 1
