"""``NSLKDDClient.fit`` i asercje privacy (bez serwera)."""

import numpy as np
import pytest

from src.privacy import assert_fit_payload_is_model_only
from src.training import LocalModel, TrainingSettings
from src.client import NSLKDDClient


def _client() -> NSLKDDClient:
    rng = np.random.default_rng(1)
    n_train, n_eval, dim = 80, 40, 6
    X_train = rng.standard_normal((n_train, dim), dtype=np.float32)
    y_train = (X_train[:, 0] > 0).astype(np.int64)
    X_eval = rng.standard_normal((n_eval, dim), dtype=np.float32)
    y_eval = (X_eval[:, 0] > 0).astype(np.int64)
    local = LocalModel(
        input_dim=dim,
        hidden_dims=[8],
        num_classes=2,
        settings=TrainingSettings(2, 16, 0.05, 0),
    )
    return NSLKDDClient(local, X_train, y_train, X_eval, y_eval)


def test_fit_returns_shard_size_not_raw_features():
    client = _client()
    params = client.get_parameters({})
    out_params, num_examples, metrics = client.fit(params, {})
    assert num_examples == len(client.y_train)
    assert metrics == {}
    assert_fit_payload_is_model_only(
        out_params,
        params,
        forbidden_feature_shape=tuple(client.X_train.shape),
    )


def test_assert_fit_payload_rejects_feature_matrix_shape():
    client = _client()
    params = client.get_parameters({})
    leaked = [client.X_train.astype(np.float32)]
    with pytest.raises(ValueError, match="expected"):
        assert_fit_payload_is_model_only(leaked, params)


def test_evaluate_returns_loss_and_metrics():
    client = _client()
    params = client.get_parameters({})
    loss, num_examples, metrics = client.evaluate(params, {})
    assert num_examples == len(client.y_eval)
    assert isinstance(loss, float)
    assert "f1" in metrics
