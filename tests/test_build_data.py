"""One-command data build on synthetic raw files (no dataset needed)."""

import json

import numpy as np
import pandas as pd

from src.build_data import build_all
from src.data_config import resolve_data_config
from src.preprocess import COLUMNS, FEATURE_COLUMNS

LABELS = ["normal"] * 5 + ["neptune", "smurf", "satan", "ipsweep", "guess_passwd", "buffer_overflow"]


def make_raw_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df["num_outbound_cmds"] = 0
    df["serror_rate"] = df["rerror_rate"]  # perfectly correlated pair
    df["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n)
    df["service"] = rng.choice(["http", "ftp", "smtp", "domain_u"], n)
    df["flag"] = rng.choice(["SF", "S0", "REJ"], n)
    for c in ("duration", "src_bytes", "dst_bytes"):
        df[c] = rng.integers(0, 1000, n)
    df["label"] = np.resize(LABELS, n)
    df["difficulty"] = 10
    df["is_attack"] = (df["label"] != "normal").astype(int)
    df["attack_category"] = df["label"].map(
        {"normal": "normal", "neptune": "DoS", "smurf": "DoS", "satan": "Probe", "ipsweep": "Probe",
         "guess_passwd": "R2L", "buffer_overflow": "U2R"})
    return df


def _cfg(tmp_path):
    train, test = tmp_path / "train.txt", tmp_path / "test.txt"
    make_raw_df(660, 0)[COLUMNS].to_csv(train, header=False, index=False)
    make_raw_df(220, 1)[COLUMNS].to_csv(test, header=False, index=False)
    return {
        "random_seed": 42,
        "num_clients": 3,
        "data": {
            "train_path": str(train), "test_path": str(test),
            "processed_dir": str(tmp_path / "processed"), "shards_dir": str(tmp_path / "shards"),
            "preprocessing": "full", "shard_partition": "iid", "dirichlet_alpha": 0.5,
        },
        "model": {"input_dim": None, "hidden_dims": [4], "num_classes": 2},
        "features": {"method": "pca", "corr_threshold": 0.95, "n_components": 0.95},
    }


def test_build_all_creates_both_methods_and_three_partitions(tmp_path):
    cfg = _cfg(tmp_path)
    before = (tmp_path / "train.txt").read_bytes()
    summary = build_all(cfg)
    assert set(summary) == {"full", "extracted"}
    assert summary["extracted"]["input_dim"] < summary["full"]["input_dim"]
    for method in summary:
        for part in ("iid", "dirichlet_a0.5", "category"):
            d = tmp_path / "shards" / method / part
            assert sorted(p.name for p in d.glob("client_*.npz")) == [f"client_{k}.npz" for k in range(3)]
        meta = json.loads((tmp_path / "processed" / method / "meta.json").read_text())
        assert meta["input_dim"] == summary[method]["input_dim"] == len(meta["feature_names"])
    assert (tmp_path / "train.txt").read_bytes() == before  # raw data untouched


def test_shards_have_dynamic_input_dim(tmp_path):
    cfg = _cfg(tmp_path)
    summary = build_all(cfg)
    for method in ("full", "extracted"):
        cfg["data"]["preprocessing"] = method
        resolved = resolve_data_config(cfg)
        dim = resolved["model"]["input_dim"]
        assert dim == summary[method]["input_dim"]
        with np.load(tmp_path / "shards" / method / "iid" / "client_0.npz") as f:
            assert f["X"].shape[1] == dim


def test_single_method_only(tmp_path):
    summary = build_all(_cfg(tmp_path), methods=("extracted",), partitions=("iid",))
    assert list(summary) == ["extracted"]
    assert not (tmp_path / "processed" / "full").exists()