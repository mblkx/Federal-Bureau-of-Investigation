"""Payload ``fit`` — wyłącznie wagi modelu."""

from pathlib import Path

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from src.client import build_client
from src.privacy import assert_fit_payload_is_model_only, estimate_payload_bytes
from tests.fl_fixtures import write_integration_fl_config


def test_fit_payload_shapes_match_model_only(tmp_path: Path) -> None:
    config_path = write_integration_fl_config(tmp_path, num_rounds=1)
    client = build_client(0, config_path)
    reference = client.get_parameters({})
    out_params, num_examples, metrics = client.fit(reference, {})

    assert num_examples == len(client.y_train)
    assert metrics == {}
    assert_fit_payload_is_model_only(
        out_params,
        reference,
        forbidden_feature_shape=tuple(client.X_train.shape),
    )


def test_flower_parameters_roundtrip_preserves_model_shapes(tmp_path: Path) -> None:
    config_path = write_integration_fl_config(tmp_path, num_rounds=1)
    client = build_client(1, config_path)
    reference = client.get_parameters({})
    out_params, _, _ = client.fit(reference, {})

    round_trip = parameters_to_ndarrays(ndarrays_to_parameters(out_params))
    assert_fit_payload_is_model_only(
        round_trip,
        reference,
        forbidden_feature_shape=tuple(client.X_train.shape),
    )


def test_fit_payload_is_not_shard_feature_block(tmp_path: Path) -> None:
    config_path = write_integration_fl_config(tmp_path, num_rounds=1)
    client = build_client(2, config_path)
    reference = client.get_parameters({})
    out_params, _, _ = client.fit(reference, {})

    assert not any(np.array_equal(arr, client.X_train) for arr in out_params)
    assert not any(np.array_equal(arr, client.X_train.T) for arr in out_params)
    payload_bytes = estimate_payload_bytes(out_params)
    shard_bytes = int(client.X_train.nbytes + client.y_train.nbytes)
    assert payload_bytes < shard_bytes * 10
