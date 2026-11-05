"""resolve_data_config tests (paths per preprocessing method, dynamic input_dim)."""

import json

import numpy as np
import pytest

from src.data_config import read_input_dim, resolve_data_config, write_meta


def _cfg(tmp_path, **data):
    return {
        "data": {"processed_dir": str(tmp_path / "processed"), "shards_dir": str(tmp_path / "shards"), **data},
        "model": {"input_dim": None, "hidden_dims": [4], "num_classes": 2},
    }


def test_preprocessing_appends_method_to_dirs(tmp_path):
    cfg = resolve_data_config(_cfg(tmp_path, preprocessing="extracted"))
    assert cfg["data"]["processed_dir"].endswith("processed/extracted")
    assert cfg["data"]["shards_dir"].endswith("shards/extracted")


def test_resolve_is_idempotent_and_does_not_mutate_input(tmp_path):
    raw = _cfg(tmp_path, preprocessing="full")
    once = resolve_data_config(raw)
    twice = resolve_data_config(once)
    assert once["data"]["processed_dir"] == twice["data"]["processed_dir"]
    assert raw["data"]["processed_dir"] == str(tmp_path / "processed")


def test_unknown_preprocessing_raises(tmp_path):
    with pytest.raises(ValueError):
        resolve_data_config(_cfg(tmp_path, preprocessing="pca"))


def test_input_dim_from_meta(tmp_path):
    write_meta(tmp_path / "processed" / "full", "full", ["a", "b", "c"])
    cfg = resolve_data_config(_cfg(tmp_path, preprocessing="full"))
    assert cfg["model"]["input_dim"] == 3


def test_input_dim_from_npz_fallback(tmp_path):
    d = tmp_path / "processed"
    d.mkdir()
    np.savez(d / "nsl_kdd_processed.npz", X_train=np.zeros((5, 7), dtype=np.float32))
    assert read_input_dim(d) == 7
    assert resolve_data_config(_cfg(tmp_path))["model"]["input_dim"] == 7


def test_explicit_input_dim_is_kept(tmp_path):
    write_meta(tmp_path / "processed", "full", ["a", "b"])
    cfg = _cfg(tmp_path)
    cfg["model"]["input_dim"] = 11
    assert resolve_data_config(cfg)["model"]["input_dim"] == 11


def test_missing_data_leaves_input_dim_unset(tmp_path):
    assert resolve_data_config(_cfg(tmp_path))["model"]["input_dim"] is None
    assert read_input_dim(tmp_path / "nothing") is None


def test_config_without_data_section_is_tolerated():
    cfg = {"model": {"input_dim": 5}}
    assert resolve_data_config(cfg) == cfg


def test_meta_content(tmp_path):
    meta = write_meta(tmp_path, "extracted", ["pc1", "pc2"])
    assert meta["input_dim"] == 2
    assert json.loads((tmp_path / "meta.json").read_text())["feature_names"] == ["pc1", "pc2"]