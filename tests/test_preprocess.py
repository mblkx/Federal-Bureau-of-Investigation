"""Preprocessing tests on synthetic data (no NSL-KDD download needed, CI-friendly)."""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    COLUMNS, FEATURE_COLUMNS, CATEGORICAL, ATTACK_CATEGORY,
    load_nsl_kdd, fit_preprocessor, transform, save_preprocessor,
    load_preprocessor, prepare_data,
)

LABELS = ["normal", "neptune", "satan", "guess_passwd", "buffer_overflow"]


def _write_fake(path, n, seed, services=("http", "ftp", "smtp"), labels=LABELS):
    rng = np.random.default_rng(seed)
    rows = {}
    for col in FEATURE_COLUMNS:
        if col == "protocol_type":
            rows[col] = rng.choice(["tcp", "udp", "icmp"], n)
        elif col == "service":
            rows[col] = rng.choice(list(services), n)
        elif col == "flag":
            rows[col] = rng.choice(["SF", "S0", "REJ"], n)
        elif col in ("duration", "src_bytes", "dst_bytes"):
            rows[col] = rng.integers(0, 10**6, n)
        else:
            rows[col] = rng.random(n)
    rows["num_outbound_cmds"] = 0  # constant column, like in the real data
    rows["label"] = rng.choice(labels, n)
    rows["difficulty"] = rng.integers(0, 21, n)
    pd.DataFrame(rows)[COLUMNS].to_csv(path, header=False, index=False)


@pytest.fixture()
def files(tmp_path):
    tr, te = tmp_path / "train.txt", tmp_path / "test.txt"
    _write_fake(tr, 500, seed=1)
    _write_fake(te, 200, seed=2)
    return tr, te


def test_load_adds_labels(files):
    df = load_nsl_kdd(files[0])
    assert df.shape[1] == len(COLUMNS) + 2
    assert set(df["is_attack"].unique()) <= {0, 1}
    assert (df.loc[df["label"] == "normal", "is_attack"] == 0).all()
    assert (df.loc[df["label"] == "neptune", "attack_category"] == "DoS").all()


def test_unknown_label_raises(tmp_path):
    p = tmp_path / "bad.txt"
    _write_fake(p, 50, seed=3, labels=["normal", "totally_new_attack"])
    with pytest.raises(ValueError, match="Unknown attack labels"):
        load_nsl_kdd(p)


def test_no_nan_and_same_width(files):
    p = prepare_data(*files)
    assert not np.isnan(p.X_train).any() and not np.isnan(p.X_test).any()
    assert p.X_train.shape[1] == p.X_test.shape[1] == p.input_dim
    assert p.X_train.dtype == np.float32
    assert set(np.unique(p.y_train)) <= {0, 1}


def test_unseen_category_in_test_does_not_change_width(tmp_path):
    tr, te = tmp_path / "tr.txt", tmp_path / "te.txt"
    _write_fake(tr, 300, seed=1, services=("http", "ftp"))
    _write_fake(te, 100, seed=2, services=("http", "ftp", "brand_new_service"))
    p = prepare_data(tr, te)
    assert p.X_train.shape[1] == p.X_test.shape[1]


def test_fitted_on_train_only(files):
    train_df = load_nsl_kdd(files[0])
    test_df = load_nsl_kdd(files[1])
    pre = fit_preprocessor(train_df)
    Xtr, _ = transform(train_df, pre)
    Xte, _ = transform(test_df, pre)
    # train numeric columns are centred; test ones are (generally) not exactly
    assert abs(Xtr[:, 0].mean()) < 1e-4
    assert abs(Xte[:, 0].mean()) > 1e-4


def test_one_hot_width(files):
    train_df = load_nsl_kdd(files[0])
    pre = fit_preprocessor(train_df)
    n_cat = sum(train_df[c].nunique() for c in CATEGORICAL)
    n_num = len(FEATURE_COLUMNS) - len(CATEGORICAL)
    assert transform(train_df, pre)[0].shape[1] == n_num + n_cat


def test_save_load_roundtrip(files, tmp_path):
    df = load_nsl_kdd(files[0])
    pre = fit_preprocessor(df)
    save_preprocessor(pre, tmp_path / "pre.joblib")
    X1, _ = transform(df, pre)
    X2, _ = transform(df, load_preprocessor(tmp_path / "pre.joblib"))
    assert np.allclose(X1, X2)


def test_deterministic(files):
    a, b = prepare_data(*files), prepare_data(*files)
    assert np.array_equal(a.X_train, b.X_train)


def test_taxonomy_complete():
    assert set(ATTACK_CATEGORY.values()) == {"normal", "DoS", "Probe", "R2L", "U2R"}