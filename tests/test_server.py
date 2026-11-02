"""Serwer FedAvg — bez żywego gRPC."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from flwr.common import parameters_to_ndarrays

import pytest

from src.server import (
    build_fedavg_strategy,
    initial_parameters_from_config,
    log_weighted_evaluate_metrics,
    main,
    weighted_evaluate_metrics,
)


def test_initial_parameters_from_config(tmp_path: Path) -> None:
    config = {
        "random_seed": 0,
        "num_clients": 3,
        "model": {"input_dim": 8, "hidden_dims": [4], "num_classes": 2},
        "training": {"local_epochs": 1, "batch_size": 16, "learning_rate": 0.01},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")

    params = initial_parameters_from_config(path)
    ndarrays = parameters_to_ndarrays(params)
    assert len(ndarrays) > 0


def test_build_fedavg_strategy_uses_num_clients_default(tmp_path: Path) -> None:
    config = {
        "random_seed": 0,
        "model": {"input_dim": 8, "hidden_dims": [4], "num_classes": 2},
        "training": {"local_epochs": 1, "batch_size": 16, "learning_rate": 0.01},
        "federated": {"num_rounds": 2},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")

    strategy = build_fedavg_strategy(path)
    assert strategy.min_fit_clients == 3
    assert strategy.min_evaluate_clients == 3
    assert strategy.min_available_clients == 3
    assert strategy.initial_parameters is not None


def test_build_fedavg_strategy_respects_config_num_clients(tmp_path: Path) -> None:
    config = {
        "random_seed": 0,
        "num_clients": 5,
        "model": {"input_dim": 8, "hidden_dims": [4], "num_classes": 2},
        "training": {"local_epochs": 1, "batch_size": 16, "learning_rate": 0.01},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")

    strategy = build_fedavg_strategy(path, num_clients=3)
    assert strategy.min_fit_clients == 5


def test_weighted_evaluate_metrics() -> None:
    metrics = [
        (100, {"f1": 0.8, "accuracy": 0.9}),
        (300, {"f1": 0.6, "accuracy": 0.7}),
    ]
    out = weighted_evaluate_metrics(metrics)
    assert out["f1"] == pytest.approx(0.65)
    assert out["accuracy"] == pytest.approx(0.75)


def test_weighted_evaluate_metrics_empty() -> None:
    assert weighted_evaluate_metrics([]) == {}


def test_build_fedavg_strategy_uses_metrics_aggregation(tmp_path: Path) -> None:
    config = {
        "random_seed": 0,
        "model": {"input_dim": 8, "hidden_dims": [4], "num_classes": 2},
        "training": {"local_epochs": 1, "batch_size": 16, "learning_rate": 0.01},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")

    strategy = build_fedavg_strategy(path)
    assert strategy.evaluate_metrics_aggregation_fn is log_weighted_evaluate_metrics


def test_main_calls_start_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "random_seed": 0,
        "num_clients": 3,
        "model": {"input_dim": 8, "hidden_dims": [4], "num_classes": 2},
        "training": {"local_epochs": 1, "batch_size": 16, "learning_rate": 0.01},
        "federated": {"num_rounds": 2},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")

    captured: dict[str, object] = {}
    mock_start = MagicMock(side_effect=lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr("src.server.start_server", mock_start)
    monkeypatch.setattr(
        sys,
        "argv",
        ["server", "--config", str(path), "--server-address", "127.0.0.1:9999"],
    )

    main()

    mock_start.assert_called_once()
    assert captured["server_address"] == "127.0.0.1:9999"
    assert captured["config"].num_rounds == 2
    assert captured["strategy"].min_fit_clients == 3


def test_log_weighted_evaluate_metrics(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="src.server")
    metrics = [(50, {"f1": 0.5})]
    assert log_weighted_evaluate_metrics(metrics) == {"f1": 0.5}
    assert "Federated evaluate" in caplog.text
    assert "f1=0.5000" in caplog.text
