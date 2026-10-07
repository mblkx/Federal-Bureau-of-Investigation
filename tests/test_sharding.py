"""Sharding tests on synthetic categories (no dataset needed, CI-friendly)."""

import numpy as np
import pytest

from src.sharding import (
    DEFAULT_ASSIGNMENT, KINDS, NORMAL, load_shard, make_partition,
    partition_by_category, partition_dirichlet, partition_iid,
    partition_stats, save_shards,
)

COUNTS = {"normal": 3000, "DoS": 1500, "Probe": 400, "R2L": 90, "U2R": 30}


@pytest.fixture()
def cats():
    arr = np.concatenate([np.full(n, c) for c, n in COUNTS.items()])
    return np.random.default_rng(0).permutation(arr)


@pytest.mark.parametrize("kind", KINDS)
def test_disjoint_and_cover_everything(kind, cats):
    shards = make_partition(kind, cats, num_clients=3, seed=1)
    assert len(shards) == 3
    all_idx = np.concatenate(shards)
    assert len(all_idx) == len(cats)
    assert np.array_equal(np.sort(all_idx), np.arange(len(cats)))


@pytest.mark.parametrize("kind", KINDS)
def test_deterministic_and_seed_sensitive(kind, cats):
    a = make_partition(kind, cats, seed=7)
    b = make_partition(kind, cats, seed=7)
    c = make_partition(kind, cats, seed=8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    if kind != "category":  # hard split differs only in normal traffic, still seed-dependent
        assert not all(np.array_equal(x, y) for x, y in zip(a, c))


@pytest.mark.parametrize("kind", KINDS)
def test_every_client_has_normal_and_attack(kind, cats):
    for idx in make_partition(kind, cats, seed=3):
        sub = cats[idx]
        assert (sub == NORMAL).sum() > 0
        assert (sub != NORMAL).sum() > 0


@pytest.mark.parametrize("kind", KINDS)
def test_normal_traffic_split_evenly(kind, cats):
    sizes = [(cats[idx] == NORMAL).sum() for idx in make_partition(kind, cats, seed=3)]
    assert max(sizes) - min(sizes) <= 1


def test_iid_is_stratified(cats):
    shards = partition_iid(cats, 3, seed=5)
    for cat in COUNTS:
        per_client = [(cats[idx] == cat).sum() for idx in shards]
        assert max(per_client) - min(per_client) <= 1


def test_category_split_isolates_attacks(cats):
    shards = partition_by_category(cats, 3, seed=5)
    for client, owned in DEFAULT_ASSIGNMENT.items():
        present = set(cats[shards[client]].tolist()) - {NORMAL}
        assert present == set(owned)


def test_category_split_validates_assignment(cats):
    with pytest.raises(ValueError):  # U2R missing
        partition_by_category(cats, 3, 0, {0: ["DoS"], 1: ["Probe"], 2: ["R2L"]})
    with pytest.raises(ValueError):  # DoS twice
        partition_by_category(cats, 3, 0, {0: ["DoS"], 1: ["DoS", "Probe"], 2: ["R2L", "U2R"]})
    with pytest.raises(ValueError):  # default assignment only for 3 clients
        partition_by_category(cats, 2, 0)


def test_dirichlet_alpha_controls_skew(cats):
    def spread(alpha):
        shards = partition_dirichlet(cats, 3, alpha, seed=11, min_attack_samples=1)
        dos = np.array([(cats[idx] == "DoS").sum() for idx in shards]) / COUNTS["DoS"]
        return dos.std()

    assert spread(0.1) > spread(100.0)


def test_dirichlet_enforces_min_attack_samples(cats):
    for idx in partition_dirichlet(cats, 3, 0.5, seed=2, min_attack_samples=50):
        assert (cats[idx] != NORMAL).sum() >= 50


def test_dirichlet_rejects_bad_params(cats):
    with pytest.raises(ValueError):
        partition_dirichlet(cats, 3, 0.0, seed=0)
    with pytest.raises(ValueError):  # impossible minimum
        partition_dirichlet(cats, 3, 0.5, seed=0, min_attack_samples=10**6, max_tries=3)


def test_invalid_inputs(cats):
    with pytest.raises(ValueError):
        make_partition("nope", cats)
    with pytest.raises(ValueError):
        partition_iid(cats, 0, seed=0)
    with pytest.raises(ValueError):
        partition_iid(np.array(["DoS", "Probe"]), 2, seed=0)  # no normal traffic


def test_stats_sum_to_total(cats):
    shards = partition_iid(cats, 3, seed=1)
    stats = partition_stats(cats, shards)
    assert stats["total"].sum() == len(cats)
    for cat, n in COUNTS.items():
        assert stats[cat].sum() == n


def test_save_and_load_roundtrip(cats, tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((len(cats), 4)).astype(np.float32)
    y = (cats != NORMAL).astype(np.int64)
    shards = partition_iid(cats, 3, seed=1)
    save_shards(X, y, cats, shards, tmp_path)

    assert (tmp_path / "stats.csv").exists()
    for k, idx in enumerate(shards):
        s = load_shard(tmp_path / f"client_{k}.npz")
        assert np.array_equal(s["X"], X[idx])
        assert np.array_equal(s["y"], y[idx])
        assert np.array_equal(s["cat"], cats[idx])
        assert np.array_equal(s["indices"], idx)