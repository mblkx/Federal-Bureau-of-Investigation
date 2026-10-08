"""Splitting the NSL-KDD training set among simulated FL clients.
 
Three partition kinds (all deterministic for a given seed, all disjoint and
covering every training row exactly once):
 
    iid         - stratified by attack category: every client gets the same
                  class mix (reference scenario, FedAvg should be close to the
                  centralized baseline).
    dirichlet   - non-IID with a tunable skew: each attack category is spread
                  over clients with proportions ~ Dirichlet(alpha). Small alpha
                  = strong skew, large alpha ~ IID.
    category    - hard non-IID: each client sees only specific attack
                  categories (default K0=DoS, K1=Probe, K2=R2L+U2R).
 
In every kind the *normal* traffic is split evenly, so no client ends up with a
single class (its local model would be useless).
 
Shard files contain X, y, cat and indices. ``cat`` (attack category) and
``indices`` exist only for analysis and tests; client training must use X and y.
 
CLI:  python -m src.sharding --partition iid|dirichlet|category
"""

from __future__ import annotations
 
import argparse
from pathlib import Path
from typing import Dict, List, Sequence
 
import numpy as np
import pandas as pd

NORMAL = "normal"
KINDS = ("iid", "dirichlet", "category")

#Hard non-IID default for 3 clients.
DEFAULT_ASSIGNMENT: Dict[int, List[str]] = {
    0: ["DoS"],
    1: ["Probe"],
    2: ["R2L", "U2R"],
}


"""HELPERS"""
 
def _check_args(categories: np.ndarray, num_clients: int) -> np.ndarray:
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    categories = np.asarray(categories)
    if NORMAL not in set(categories.tolist()):
        raise ValueError("no 'normal' samples in categories")
    return categories
 
 
def _split_even(idx: np.ndarray, k: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Shuffle indices and cut them into k near-equal chunks."""
    return np.array_split(rng.permutation(idx), k)
 
 
def _finish(parts: Sequence[Sequence[np.ndarray]]) -> List[np.ndarray]:
    """Concatenate chunks per client and sort indices (stable, readable files)."""
    return [np.sort(np.concatenate(list(p))) for p in parts]


"""PARTITIONS"""

def partition_iid(categories: np.ndarray, num_clients: int, seed: int) -> List[np.ndarray]:
    """Stratified IID split: each category is divided evenly among clients."""
    categories = _check_args(categories, num_clients)
    rng = np.random.default_rng(seed)
    parts: List[List[np.ndarray]] = [[] for _ in range(num_clients)]
    for cat in sorted(set(categories.tolist())):
        idx = np.flatnonzero(categories == cat)
        for c, chunk in enumerate(_split_even(idx, num_clients, rng)):
            parts[c].append(chunk)
    return _finish(parts)
 
 
def partition_dirichlet(
    categories: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_attack_samples: int = 50,
    max_tries: int = 100,
) -> List[np.ndarray]:
    """Non-IID split: attack categories spread with Dirichlet(alpha) proportions.
 
    Normal traffic is split evenly. The draw is repeated (same RNG stream, so
    still deterministic) until every client has at least `min_attack_samples`
    attack rows; otherwise a client could end up with only normal traffic.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    categories = _check_args(categories, num_clients)
    rng = np.random.default_rng(seed)
 
    normal_chunks = _split_even(np.flatnonzero(categories == NORMAL), num_clients, rng)
    attack_cats = sorted(set(categories.tolist()) - {NORMAL})
 
    for _ in range(max_tries):
        attack_parts: List[List[np.ndarray]] = [[] for _ in range(num_clients)]
        for cat in attack_cats:
            idx = rng.permutation(np.flatnonzero(categories == cat))
            props = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
            for c, chunk in enumerate(np.split(idx, cuts)):
                attack_parts[c].append(chunk)
        sizes = [sum(len(ch) for ch in ap) for ap in attack_parts]
        if min(sizes) >= min_attack_samples:
            break
    else:
        raise ValueError(
            f"could not give every client >= {min_attack_samples} attack samples "
            f"with alpha={alpha} in {max_tries} tries; increase alpha or lower the minimum"
        )
 
    return _finish([[normal_chunks[c], *attack_parts[c]] for c in range(num_clients)])
 
 
def partition_by_category(
    categories: np.ndarray,
    num_clients: int,
    seed: int,
    assignment: Dict[int, List[str]] | None = None,
) -> List[np.ndarray]:
    """Hard non-IID split: each attack category goes entirely to one client."""
    categories = _check_args(categories, num_clients)
    if assignment is None:
        if num_clients != 3:
            raise ValueError("default assignment is defined for 3 clients; pass `assignment`")
        assignment = DEFAULT_ASSIGNMENT
    if sorted(assignment) != list(range(num_clients)):
        raise ValueError("assignment keys must be 0..num_clients-1")
 
    assigned = [c for cats in assignment.values() for c in cats]
    if len(assigned) != len(set(assigned)):
        raise ValueError("an attack category is assigned to more than one client")
    present = set(categories.tolist()) - {NORMAL}
    if set(assigned) != present:
        raise ValueError(
            f"assignment must cover exactly the attack categories present: {sorted(present)}"
        )
 
    rng = np.random.default_rng(seed)
    normal_chunks = _split_even(np.flatnonzero(categories == NORMAL), num_clients, rng)
    parts = []
    for c in range(num_clients):
        own = [np.flatnonzero(categories == cat) for cat in assignment[c]]
        parts.append([normal_chunks[c], *own])
    return _finish(parts)
 
 
def make_partition(
    kind: str,
    categories: np.ndarray,
    num_clients: int = 3,
    seed: int = 42,
    alpha: float = 0.5,
    assignment: Dict[int, List[str]] | None = None,
) -> List[np.ndarray]:
    if kind == "iid":
        return partition_iid(categories, num_clients, seed)
    if kind == "dirichlet":
        return partition_dirichlet(categories, num_clients, alpha, seed)
    if kind == "category":
        return partition_by_category(categories, num_clients, seed, assignment)
    raise ValueError(f"unknown partition kind '{kind}', expected one of {KINDS}")


"""STATS AND I/O"""

def partition_stats(categories: np.ndarray, shards: Sequence[np.ndarray]) -> pd.DataFrame:
    """Rows = clients, columns = rows per category + total + attack_share."""
    categories = np.asarray(categories)
    cats = sorted(set(categories.tolist()))
    rows = []
    for k, idx in enumerate(shards):
        sub = categories[idx]
        row = {c: int((sub == c).sum()) for c in cats}
        row["total"] = int(len(idx))
        row["attack_share"] = round(1 - row.get(NORMAL, 0) / max(len(idx), 1), 3)
        rows.append(pd.Series(row, name=f"client_{k}"))
    return pd.DataFrame(rows)
 
 
def save_shards(
    X: np.ndarray,
    y: np.ndarray,
    categories: np.ndarray,
    shards: Sequence[np.ndarray],
    out_dir: str | Path,
) -> pd.DataFrame:
    """Write client_<k>.npz (X, y, cat, indices) and stats.csv; return stats."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    categories = np.asarray(categories)
    for k, idx in enumerate(shards):
        np.savez_compressed(
            out / f"client_{k}.npz",
            X=X[idx], y=y[idx], cat=categories[idx], indices=idx,
        )
    stats = partition_stats(categories, shards)
    stats.to_csv(out / "stats.csv")
    return stats
 
 
def load_shard(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


"""MAIN"""

def main() -> None:
    from src.preprocess import load_prepared_arrays  # local import keeps module light
 
    ap = argparse.ArgumentParser(description="Split preprocessed NSL-KDD train set among clients.")
    ap.add_argument("--partition", choices=KINDS, required=True)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--num-clients", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha (dirichlet only)")
    args = ap.parse_args()
 
    data = load_prepared_arrays(args.processed)
    shards = make_partition(
        args.partition, data["cat_train"], args.num_clients, args.seed, args.alpha
    )
    name = f"dirichlet_a{args.alpha}" if args.partition == "dirichlet" else args.partition
    out_dir = Path(args.out) / name
    stats = save_shards(data["X_train"], data["y_train"], data["cat_train"], shards, out_dir)
 
    print(f"saved {len(shards)} shards to {out_dir}")
    print(stats.to_string())
 
 
if __name__ == "__main__":
    main()