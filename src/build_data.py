"""One command that builds all data variants: 2 preprocessing methods x 3 partitions.

    python -m src.build_data

method `full`       122 features (log1p + scaling + one-hot), src/preprocess.py
method `extracted`  constant/correlated filter + PCA or selection, src/features.py
partitions          iid, dirichlet (alpha from config), category -> 3 client shards each

`model.input_dim` is not set by hand: it is written to meta.json for each method
and read back by src/data_config.py. Raw files are only read, never modified.
Does not import torch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.data_config import METHODS, write_meta
from src.features import extract_prepared
from src.preprocess import prepare_data, save_prepared
from src.sharding import KINDS, make_partition, save_shards

DEFAULT_ALPHA = 0.5


def partition_name(kind: str, alpha: float) -> str:
    return f"dirichlet_a{alpha}" if kind == "dirichlet" else kind


def build_all(
    cfg: dict,
    methods: tuple[str, ...] = METHODS,
    partitions: tuple[str, ...] = KINDS,
) -> dict:
    data_cfg = cfg["data"]
    seed = int(cfg["random_seed"])
    num_clients = int(cfg["num_clients"])
    alpha = float(data_cfg.get("dirichlet_alpha", DEFAULT_ALPHA))

    full = prepare_data(data_cfg["train_path"], data_cfg["test_path"])
    names = list(full.preprocessor.get_feature_names_out())
    variants = {}
    if "full" in methods:
        variants["full"] = (full, names, None)
    if "extracted" in methods:
        extracted, extractor = extract_prepared(full, dict(cfg.get("features", {})))
        variants["extracted"] = (extracted, extractor.get_feature_names_out(), extractor.summary())

    summary = {}
    for method, (prepared, feature_names, details) in variants.items():
        proc_dir = Path(data_cfg["processed_dir"]) / method
        save_prepared(prepared, proc_dir)
        write_meta(proc_dir, method, feature_names)
        if details is not None:
            (proc_dir / "features.json").write_text(
                json.dumps(details, indent=2) + "\n", encoding="utf-8", newline="\n"
            )
        parts = []
        for kind in partitions:
            shards = make_partition(kind, prepared.cat_train, num_clients, seed, alpha)
            out = Path(data_cfg["shards_dir"]) / method / partition_name(kind, alpha)
            save_shards(prepared.X_train, prepared.y_train, prepared.cat_train, shards, out)
            parts.append(partition_name(kind, alpha))
        summary[method] = {
            "input_dim": prepared.input_dim,
            "processed_dir": proc_dir.as_posix(),
            "partitions": parts,
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build processed data and client shards for both methods.")
    ap.add_argument("--config", default="src/config.yaml")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--partitions", nargs="+", choices=KINDS, default=list(KINDS))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary = build_all(cfg, tuple(args.methods), tuple(args.partitions))
    for method, info in summary.items():
        print(f"{method}: input_dim = {info['input_dim']}, data in {info['processed_dir']}, "
              f"shards: {', '.join(info['partitions'])}")


if __name__ == "__main__":
    main()