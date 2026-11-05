"""Data paths and `input_dim` resolved from the prepared data (no hand-set dimension).

Layout written by `python -m src.build_data`:
    data/processed/<method>/            nsl_kdd_processed.npz, preprocessor.joblib, meta.json
    data/shards/<method>/<partition>/   client_<k>.npz, stats.csv
where <method> is `full` (method no. 1, all features) or `extracted` (method no. 2).

Does not import torch, so it runs without PyTorch and in CI.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

METHODS = ("full", "extracted")
META_FILE = "meta.json"
_AUTO = (None, 0, "auto")


def write_meta(out_dir: str | Path, method: str, feature_names: Sequence[str]) -> dict:
    """Store `input_dim` and feature names next to the processed arrays."""
    meta = {"method": method, "input_dim": len(feature_names), "feature_names": list(feature_names)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return meta


def read_input_dim(processed_dir: str | Path) -> int | None:
    """`input_dim` from meta.json, else from the saved train matrix; None if no data."""
    d = Path(processed_dir)
    meta = d / META_FILE
    if meta.exists():
        return int(json.loads(meta.read_text(encoding="utf-8"))["input_dim"])
    npz = d / "nsl_kdd_processed.npz"
    if npz.exists():
        with np.load(npz) as f:
            return int(f["X_train"].shape[1])
    return None


def resolve_data_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the config with method-specific dirs and `model.input_dim` filled in.

    * `data.preprocessing` (full | extracted) appends the method to `processed_dir`
      and `shards_dir`; without the key the paths stay as they are.
    * `model.input_dim` set to null / 0 / "auto" is read from the processed data;
      an explicit positive value is left untouched.
    """
    cfg = copy.deepcopy(cfg)
    data = cfg.get("data") or {}
    method = data.get("preprocessing")
    if method is not None and not data.get("_resolved"):
        if method not in METHODS:
            raise ValueError(f"data.preprocessing must be one of {METHODS}, got '{method}'")
        for key in ("processed_dir", "shards_dir"):
            if key in data:
                data[key] = (Path(data[key]) / method).as_posix()
        data["_resolved"] = True

    model = cfg.get("model")
    if model is not None and model.get("input_dim") in _AUTO and data.get("processed_dir"):
        dim = read_input_dim(data["processed_dir"])
        if dim:
            model["input_dim"] = dim
    return cfg