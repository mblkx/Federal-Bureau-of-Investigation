"""Feature extraction tests on synthetic data (no dataset needed, CI-friendly)."""

import numpy as np
import pytest

from src.features import FeatureExtractor, extraction_pipeline


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    a = rng.normal(size=n) + 2.0 * y        # related to the target
    b = a + rng.normal(scale=0.01, size=n)  # almost a copy of a
    c = rng.normal(size=n)                  # noise
    d = rng.normal(size=n)                  # noise
    const = np.full(n, 3.0)
    X = np.column_stack([const, b, a, c, d])
    return X, y, ["const", "b", "a", "c", "d"]


def test_constant_column_dropped(data):
    X, y, names = data
    ext = FeatureExtractor(method="filter").fit(X, y, names)
    assert ext.constant_ == ["const"]
    assert "const" not in ext.get_feature_names_out()


def test_correlated_drops_less_target_related_column(data):
    X, y, names = data
    # make b clearly worse than a for the target
    X = X.copy()
    X[:, 1] = X[:, 2] + np.random.default_rng(1).normal(scale=0.3, size=len(y))
    ext = FeatureExtractor(method="filter", corr_threshold=0.9).fit(X, y, names)
    assert set(ext.correlated_) == {"b"}
    assert ext.correlated_["b"]["kept"] == "a"
    assert ext.get_feature_names_out() == ["a", "c", "d"]


def test_filter_output_shape_and_dtype(data):
    X, y, names = data
    ext = FeatureExtractor(method="filter").fit(X, y, names)
    out = ext.transform(X)
    assert out.dtype == np.float32
    assert out.shape == (len(X), len(ext.get_feature_names_out()))


def test_select_keeps_k_best(data):
    X, y, names = data
    ext = FeatureExtractor(method="select", k=2).fit(X, y, names)
    out = ext.transform(X)
    assert out.shape[1] == 2
    assert "a" in ext.get_feature_names_out() or "b" in ext.get_feature_names_out()


def test_select_k_larger_than_columns_is_clipped(data):
    X, y, names = data
    ext = FeatureExtractor(method="select", k=100).fit(X, y, names)
    assert ext.transform(X).shape[1] == len(ext.filter_idx_)


def test_pca_reduces_and_reports_variance(data):
    X, y, names = data
    ext = FeatureExtractor(method="pca", n_components=0.9).fit(X, y, names)
    out = ext.transform(X)
    assert out.shape[1] == len(ext.get_feature_names_out()) <= len(ext.filter_idx_)
    assert ext.summary()["explained_variance"] >= 0.9
    assert ext.get_feature_names_out()[0] == "pc1"


def test_fit_uses_train_statistics_only(data):
    X, y, names = data
    ext = FeatureExtractor(method="pca", n_components=2).fit(X[:300], y[:300], names)
    shifted = X[300:] + 100.0
    out = ext.transform(shifted)  # must not refit on new data
    assert out.shape == (100, 2)
    assert np.abs(out).max() > 10


def test_deterministic(data):
    X, y, names = data
    a = FeatureExtractor(method="pca").fit(X, y, names).transform(X)
    b = FeatureExtractor(method="pca").fit(X, y, names).transform(X)
    assert np.array_equal(a, b)


def test_errors(data):
    X, y, names = data
    with pytest.raises(ValueError):
        FeatureExtractor(method="nope").fit(X, y, names)
    with pytest.raises(ValueError):
        FeatureExtractor().fit(X, y, ["only_one"])
    ext = FeatureExtractor(method="filter").fit(X, y, names)
    with pytest.raises(ValueError):
        ext.transform(X[:, :3])


def test_pipeline_from_raw_frame():
    from tests.test_build_data import make_raw_df
    from src.preprocess import FEATURE_COLUMNS, fit_preprocessor, transform

    df = make_raw_df(500, seed=0)
    pre = fit_preprocessor(df)
    X, y = transform(df, pre)
    ext = FeatureExtractor(method="filter").fit(X, y, list(pre.get_feature_names_out()))
    pipe = extraction_pipeline(pre, ext)
    out = pipe.transform(df[FEATURE_COLUMNS])
    assert out.shape[1] == len(ext.get_feature_names_out()) < X.shape[1]