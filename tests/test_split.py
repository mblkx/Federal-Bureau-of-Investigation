"""Unit tests for data partitioning."""

import numpy as np

from src.preprocess import split_train_into_shards


def test_split_covers_all_samples_once():
    n = 1000
    shards = split_train_into_shards(n, num_clients=3, seed=42)
    merged = np.concatenate(shards)
    assert len(merged) == n
    assert len(np.unique(merged)) == n
    assert merged.min() == 0
    assert merged.max() == n - 1
