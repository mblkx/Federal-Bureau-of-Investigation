"""Integracja: subprocess serwer + 3 klienty, powtarzalność, opcjonalnie Ray."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.federated_eval import evaluate_global_parameters, run_inprocess_fedavg
from tests.fl_fixtures import write_integration_fl_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(host: str, port: int, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.15)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


@pytest.mark.integration
def test_fl_server_and_three_clients_one_round(tmp_path: Path) -> None:
    config_path = write_integration_fl_config(tmp_path, num_rounds=1)
    port = _free_port()
    address = f"127.0.0.1:{port}"
    python = sys.executable

    server = subprocess.Popen(
        [
            python,
            "-m",
            "src.server",
            "--config",
            str(config_path),
            "--server-address",
            address,
            "--num-clients",
            "3",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_port("127.0.0.1", port)
        clients = [
            subprocess.Popen(
                [
                    python,
                    "-m",
                    "src.client",
                    "--cid",
                    str(cid),
                    "--config",
                    str(config_path),
                    "--server-address",
                    address,
                ],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for cid in range(3)
        ]
        client_outputs: list[str] = []
        for proc in clients:
            out, _ = proc.communicate(timeout=120)
            client_outputs.append(out or "")
            assert proc.returncode == 0, f"client failed:\n{out}"

        server_out, _ = server.communicate(timeout=120)
        assert server.returncode == 0, f"server failed:\n{server_out}"
    finally:
        if server.poll() is None:
            server.kill()
            server.communicate(timeout=10)


@pytest.mark.integration
def test_inprocess_fl_f1_reproducible_with_same_seed(tmp_path: Path) -> None:
    config_path = write_integration_fl_config(tmp_path, num_rounds=1)
    cfg_dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params_a = run_inprocess_fedavg(config_path)
    params_b = run_inprocess_fedavg(config_path)
    for left, right in zip(params_a, params_b, strict=True):
        assert np.allclose(left, right, rtol=0.0, atol=1e-6)

    f1_a = float(evaluate_global_parameters(cfg_dict, params_a)["f1"])
    f1_b = float(evaluate_global_parameters(cfg_dict, params_b)["f1"])
    assert abs(f1_a - f1_b) <= 0.01


@pytest.mark.integration
def test_fl_flower_simulation_when_ray_available(tmp_path: Path) -> None:
    pytest.importorskip("ray")
    from flwr.simulation import start_simulation

    from src.client import build_client
    from src.server import build_fedavg_strategy
    from src.training import load_project_config

    config_path = write_integration_fl_config(tmp_path, num_rounds=1)
    cfg = load_project_config(config_path)
    num_clients = int(cfg["num_clients"])

    def client_fn(cid: str):
        return build_client(int(cid), config_path).to_client()

    strategy = build_fedavg_strategy(config_path)
    start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=__import__("flwr").server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )
