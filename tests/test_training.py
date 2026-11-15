"""Unit tests for local training (no NSL-KDD download)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.training import (
    VAL_FRACTION,
    LocalModel,
    TrainingSettings,
    load_project_config,
    ndarrays_to_state_dict,
    round_seed,
    run_local_baseline,
    server_round_from_config,
    split_shard_holdout,
    state_dict_to_ndarrays,
)


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


def test_evaluate_uses_threshold_metrics_from_evaluate_module():
    model = _tiny_local()
    X = np.random.randn(20, 8).astype(np.float32)
    y = np.array([0, 1] * 10, dtype=np.int64)
    cats = np.array(["normal", "DoS"] * 10)
    loss, metrics = model.evaluate(X, y, cats)
    assert isinstance(loss, float)
    assert set(metrics) >= {"f1", "precision", "recall", "accuracy", "threshold", "tp", "tn"}
    assert metrics["threshold"] == pytest.approx(0.5)
    assert "by_category" not in metrics


def test_evaluate_full_includes_by_category():
    model = _tiny_local()
    X = np.random.randn(20, 8).astype(np.float32)
    y = np.array([0, 1] * 10, dtype=np.int64)
    cats = np.array(["normal", "DoS"] * 10)
    full = model.evaluate_full(X, y, cats)
    assert "by_category" in full
    assert "DoS" in full["by_category"]


def test_split_shard_holdout_is_disjoint_and_covers_all():
    rng = np.random.default_rng(2)
    n = 100
    X = rng.standard_normal((n, 4), dtype=np.float32)
    y = np.array([0] * 50 + [1] * 50)
    cats = np.array(["normal"] * 50 + ["DoS"] * 30 + ["Probe"] * 20)
    X_tr, y_tr, cat_tr, X_va, y_va, cat_va = split_shard_holdout(X, y, cats, seed=7)
    assert len(X_tr) + len(X_va) == n
    assert len(y_tr) == len(X_tr) == len(cat_tr)
    assert len(y_va) == len(X_va) == len(cat_va)
    assert abs(len(X_va) / n - VAL_FRACTION) < 0.05
    # No index overlap via feature-row identity on this synthetic set:
    train_rows = {tuple(row.tolist()) for row in X_tr}
    val_rows = {tuple(row.tolist()) for row in X_va}
    assert train_rows.isdisjoint(val_rows)


def test_from_config_rejects_missing_input_dim():
    cfg = {
        "random_seed": 1,
        "model": {"input_dim": None, "hidden_dims": [8], "num_classes": 2},
        "training": {"local_epochs": 1, "batch_size": 8, "learning_rate": 0.01},
    }
    with pytest.raises(ValueError, match="input_dim"):
        LocalModel.from_config(cfg)


def test_run_local_baseline_uses_shard_validation_not_kddtest(tmp_path: Path, monkeypatch):
    from tests.fl_fixtures import write_integration_fl_config

    config_path = write_integration_fl_config(tmp_path, num_rounds=1)

    def boom(*_args, **_kwargs):
        raise AssertionError("KDDTest+ must not be loaded for local baseline (M2.1)")

    monkeypatch.setattr("src.preprocess.load_prepared_arrays", boom)

    metrics = run_local_baseline(0, config_path)
    assert metrics["n_train"] + metrics["n_val"] == 60
    assert metrics["n_val"] > 0
    assert "f1" in metrics


def test_load_project_config_resolves_preprocessing_paths(tmp_path: Path):
    processed = tmp_path / "processed" / "full"
    processed.mkdir(parents=True)
    (processed / "meta.json").write_text(
        '{"method": "full", "input_dim": 9, "feature_names": []}\n',
        encoding="utf-8",
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "random_seed": 1,
                "num_clients": 3,
                "data": {
                    "processed_dir": str(tmp_path / "processed"),
                    "shards_dir": str(tmp_path / "shards"),
                    "shard_partition": "iid",
                    "preprocessing": "full",
                },
                "model": {"input_dim": None, "hidden_dims": [4], "num_classes": 2},
                "training": {"local_epochs": 1, "batch_size": 8, "learning_rate": 0.01},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_project_config(cfg_path)
    assert cfg["model"]["input_dim"] == 9
    assert cfg["data"]["processed_dir"].endswith("full")


def test_round_seed_is_base_plus_server_round():
    assert round_seed(42, 0) == 42
    assert round_seed(42, 1) == 43
    assert round_seed(42, 5) == 47
    with pytest.raises(ValueError, match="server_round"):
        round_seed(42, -1)


def test_server_round_from_config_defaults_and_parses():
    assert server_round_from_config(None) == 0
    assert server_round_from_config({}) == 0
    assert server_round_from_config({"server_round": 3}) == 3
    assert server_round_from_config({"server_round": 2.0}) == 2


def test_train_on_different_rounds_change_updates():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 8), dtype=np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    base = _tiny_local()
    start = base.get_parameters()

    a = _tiny_local()
    a.set_parameters(start)
    a.train_on(X, y, server_round=1)

    b = _tiny_local()
    b.set_parameters(start)
    b.train_on(X, y, server_round=2)

    assert not all(np.allclose(pa, pb) for pa, pb in zip(a.get_parameters(), b.get_parameters()))


def test_train_on_same_round_is_deterministic():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((64, 8), dtype=np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    start = _tiny_local().get_parameters()

    a = _tiny_local()
    a.set_parameters(start)
    a.train_on(X, y, server_round=3)

    b = _tiny_local()
    b.set_parameters(start)
    b.train_on(X, y, server_round=3)

    for pa, pb in zip(a.get_parameters(), b.get_parameters()):
        np.testing.assert_allclose(pa, pb)


def test_train_on_resets_adam_state_by_default():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((48, 8), dtype=np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    model = _tiny_local()
    model.train_on(X, y, server_round=0)
    assert model.optimizer.state  # moments after first fit

    opt_before = model.optimizer
    model.train_on(X, y, server_round=1)
    assert model.optimizer is not opt_before
    # Fresh Adam: state fills only after the new fit's steps.
    assert model.optimizer.state


def test_train_on_can_keep_adam_when_disabled():
    settings = TrainingSettings(
        local_epochs=1,
        batch_size=16,
        learning_rate=0.05,
        random_seed=0,
        reset_optimizer_each_round=False,
    )
    model = LocalModel(8, [16], 2, settings)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((32, 8), dtype=np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    model.train_on(X, y, server_round=0)
    opt_before = model.optimizer
    state_len_before = len(model.optimizer.state)
    model.train_on(X, y, server_round=1)
    assert model.optimizer is opt_before
    assert len(model.optimizer.state) == state_len_before
