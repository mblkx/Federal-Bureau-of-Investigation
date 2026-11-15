"""Testy siatki eksperymentów M2.4 (bez pełnego NSL-KDD)."""

from pathlib import Path

import json
import yaml

from src.run_baselines import (
    list_available_partitions,
    make_json_safe,
    run_centralized_baseline,
    run_local_client_baseline,
    run_baselines,
)
from tests.fl_fixtures import (
    integration_fl_config_dict,
    write_processed,
    write_shards,
)


def _write_config(tmp_path: Path) -> Path:
    processed = write_processed(tmp_path)
    shards = write_shards(tmp_path)
    cfg = integration_fl_config_dict(tmp_path, processed, shards, num_rounds=1)
    cfg["model"]["architecture"] = "mlp"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


def test_make_json_safe_handles_numpy_and_nan():
    import math
    import numpy as np

    payload = {"f1": np.float64(0.5), "bad": float("nan"), "n": np.int64(3)}
    out = make_json_safe(payload)
    assert out["f1"] == 0.5
    assert out["bad"] is None
    assert out["n"] == 3
    assert math.isnan(payload["bad"])


def test_list_available_partitions(tmp_path: Path):
    shards = tmp_path / "shards"
    (shards / "iid").mkdir(parents=True)
    (shards / "category").mkdir()
    (shards / "other").mkdir()
    assert list_available_partitions(shards) == ["iid", "category"]


def test_run_centralized_baseline_writes_val_and_test(tmp_path: Path):
    from src.training import load_project_config

    config_path = _write_config(tmp_path)
    cfg = load_project_config(config_path)
    result = run_centralized_baseline(cfg, architecture="mlp")
    assert result["model"] == "centralized"
    assert result["architecture"] == "mlp"
    assert "validation" in result and "test" in result
    assert result["test"]["split"] == "KDDTest+"
    assert "f1" in result["validation"] and "f1" in result["test"]
    assert "meets_project_criterion" in result["test"]


def test_run_local_client_baseline_for_autoencoder(tmp_path: Path):
    from src.training import load_project_config

    config_path = _write_config(tmp_path)
    cfg = load_project_config(config_path)
    result = run_local_client_baseline(
        cfg,
        client_id=0,
        architecture="autoencoder",
        shard_partition="iid",
    )
    assert result["model"] == "local"
    assert result["architecture"] == "autoencoder"
    assert result["client_id"] == 0
    assert result["shard_partition"] == "iid"
    assert result["n_train"] == int((result["n_train"]))  # json-safe int
    assert result["test"]["split"] == "KDDTest+"
    assert result["threshold"] is not None


def test_run_m2_baselines_matrix_writes_files(tmp_path: Path):
    config_path = _write_config(tmp_path)
    results_dir = tmp_path / "results_m2"
    summary = run_m2_baselines(
        config_path,
        results_dir=results_dir,
        architectures=["mlp", "autoencoder"],
        partitions=["iid"],
        run_centralized=True,
        run_local=True,
    )

    assert (results_dir / "centralized" / "mlp.json").is_file()
    assert (results_dir / "centralized" / "autoencoder.json").is_file()
    for arch in ("mlp", "autoencoder"):
        for cid in range(3):
            assert (results_dir / "local" / "iid" / f"client_{cid}_{arch}.json").is_file()

    summary_disk = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(summary_disk["centralized"]) == 2
    assert len(summary_disk["local"]) == 6
    assert summary["results_dir"] == str(results_dir)
