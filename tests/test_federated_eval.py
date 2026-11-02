"""Ewaluacja globalna, FedAvg in-process, zapis/odczyt wag."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.federated_eval import (
    evaluate_global_parameters,
    load_global_weights,
    run_inprocess_fedavg,
    save_global_weights,
    write_federated_results,
)
from src.training import LocalModel, TrainingSettings
from tests.fl_fixtures import integration_fl_config_dict, write_processed, write_shards


def _config(tmp_path: Path, processed: Path, shards: Path) -> dict:
    return integration_fl_config_dict(tmp_path, processed, shards, num_rounds=2)


def test_save_load_global_weights(tmp_path: Path) -> None:
    arrays = [np.array([1.0, 2.0]), np.array([[3.0]])]
    path = tmp_path / "w.npz"
    save_global_weights(path, arrays)
    loaded = load_global_weights(path)
    assert len(loaded) == 2
    assert np.allclose(loaded[0], arrays[0])
    assert np.allclose(loaded[1], arrays[1])


def test_evaluate_global_parameters(tmp_path: Path) -> None:
    processed = write_processed(tmp_path)
    cfg = _config(tmp_path, processed, write_shards(tmp_path))
    local = LocalModel(
        input_dim=6,
        hidden_dims=[8],
        num_classes=2,
        settings=TrainingSettings(1, 16, 0.05, 0),
    )
    params = local.get_parameters()
    result = evaluate_global_parameters(cfg, params)
    assert result["split"] == "KDDTest+"
    assert "f1" in result
    assert "meets_project_criterion" in result


def test_run_inprocess_fedavg_and_write_results(tmp_path: Path) -> None:
    processed = write_processed(tmp_path)
    shards = write_shards(tmp_path)
    cfg = _config(tmp_path, processed, shards)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    parameters = run_inprocess_fedavg(config_path)
    metrics_path = tmp_path / "metrics.json"
    weights_path = tmp_path / "global.npz"
    write_federated_results(cfg, parameters, metrics_path, weights_path=weights_path)

    assert weights_path.is_file()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["model"] == "fedavg_global"
    assert "f1" in saved
