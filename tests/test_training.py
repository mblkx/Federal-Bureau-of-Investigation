"""Unit tests for local training (no NSL-KDD download)."""

import numpy as np

from src.training import LocalModel, TrainingSettings, ndarrays_to_state_dict, state_dict_to_ndarrays


def _tiny_local(input_dim: int = 8) -> LocalModel:
    return LocalModel(
        input_dim=input_dim,
        hidden_dims=[16, 8],
        num_classes=2,
        settings=TrainingSettings(
            local_epochs=3,
            batch_size=32,
            learning_rate=0.05,
            random_seed=0,
        ),
    )


def test_train_improves_loss_on_synthetic_data():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.standard_normal((n, 8), dtype=np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    model = _tiny_local()
    loss_before, _ = model.evaluate(X, y)
    model.train_on(X, y)
    loss_after, metrics = model.evaluate(X, y)
    assert loss_after < loss_before
    assert metrics["f1"] > 0.5


def test_parameter_roundtrip():
    model = _tiny_local(input_dim=4)
    original = model.get_parameters()
    model.set_parameters([p + 0.1 for p in original])
    restored_keys = list(model.model.state_dict().keys())
    roundtrip = ndarrays_to_state_dict(restored_keys, original)
    assert len(state_dict_to_ndarrays(roundtrip)) == len(original)


def test_train_on_returns_sample_count():
    X = np.random.randn(50, 8).astype(np.float32)
    y = np.zeros(50, dtype=np.int64)
    model = _tiny_local()
    assert model.train_on(X, y) == 50


def test_evaluate_returns_sklearn_metrics():
    model = _tiny_local()
    X = np.random.randn(20, 8).astype(np.float32)
    y = np.array([0, 1] * 10, dtype=np.int64)
    loss, metrics = model.evaluate(X, y)
    assert isinstance(loss, float)
    assert set(metrics) >= {"f1", "precision", "recall", "accuracy"}


def test_from_config_rejects_missing_input_dim():
    cfg = {
        "random_seed": 1,
        "model": {"input_dim": None, "hidden_dims": [8], "num_classes": 2},
        "training": {"local_epochs": 1, "batch_size": 8, "learning_rate": 0.01},
    }
    try:
        LocalModel.from_config(cfg)
        assert False, "expected ValueError"
    except (ValueError, TypeError):
        pass
