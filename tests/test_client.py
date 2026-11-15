"""``NSLKDDClient.fit`` i asercje privacy (bez serwera)."""

from pathlib import Path

import numpy as np
import pytest

from src.client import NSLKDDClient, build_client
from src.privacy import assert_fit_payload_is_model_only
from src.training import LocalModel, TrainingSettings, VAL_FRACTION
from tests.fl_fixtures import write_integration_fl_config


def _client() -> NSLKDDClient:
    rng = np.random.default_rng(1)
    n_train, n_eval, dim = 80, 40, 6
    X_train = rng.standard_normal((n_train, dim), dtype=np.float32)
    y_train = (X_train[:, 0] > 0).astype(np.int64)
    X_eval = rng.standard_normal((n_eval, dim), dtype=np.float32)
    y_eval = (X_eval[:, 0] > 0).astype(np.int64)
    cat_eval = np.array(
        ["normal" if label == 0 else "DoS" for label in y_eval],
        dtype="U16",
    )
    local = LocalModel(
        input_dim=dim,
        hidden_dims=[8],
        num_classes=2,
        settings=TrainingSettings(2, 16, 0.05, 0),
    )
    return NSLKDDClient(local, X_train, y_train, X_eval, y_eval, cat_eval=cat_eval)


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


def test_fit_uses_server_round_for_seed(monkeypatch: pytest.MonkeyPatch):
    client = _client()
    seen: dict[str, int] = {}

    def fake_train_on(X, y, *, server_round: int = 0):
        seen["server_round"] = server_round
        return len(y)

    monkeypatch.setattr(client.local, "train_on", fake_train_on)
    client.fit(client.get_parameters({}), {"server_round": 4})
    assert seen["server_round"] == 4


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
    assert metrics["threshold"] == pytest.approx(0.5)


def test_build_client_uses_local_holdout_not_kddtest(tmp_path: Path, monkeypatch):
    config_path = write_integration_fl_config(tmp_path, num_rounds=1)

    def boom(*_args, **_kwargs):
        raise AssertionError("client must not load KDDTest+ / prepared arrays (M2.1)")

    monkeypatch.setattr("src.preprocess.load_prepared_arrays", boom)
    monkeypatch.setattr("src.client.load_prepared_arrays", boom, raising=False)

    client = build_client(0, config_path)
    shard_n = 60
    assert len(client.y_train) + len(client.y_eval) == shard_n
    assert abs(len(client.y_eval) / shard_n - VAL_FRACTION) < 0.08
    assert client.cat_eval is not None
    assert len(client.cat_eval) == len(client.y_eval)


def test_build_client_requires_cat_in_shard(tmp_path: Path):
    from tests.fl_fixtures import integration_fl_config_dict, write_processed
    import yaml

    processed = write_processed(tmp_path)
    # Legacy shard without cat:
    shard_dir = tmp_path / "shards" / "iid"
    shard_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6), dtype=np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    np.savez_compressed(shard_dir / "client_0.npz", X=X, y=y)
    for cid in (1, 2):
        np.savez_compressed(shard_dir / f"client_{cid}.npz", X=X, y=y, cat=np.array(["normal"] * 40))

    cfg = integration_fl_config_dict(tmp_path, processed, tmp_path / "shards")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    with pytest.raises(KeyError, match="cat"):
        build_client(0, path)
