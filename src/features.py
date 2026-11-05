"""Feature extraction (preprocessing method no. 2), fit on train only.

Works on the output of method no. 1 (log1p + scaling + one-hot, 122 columns):

1. drop constant columns,
2. drop highly correlated columns (|r| > corr_threshold; of a correlated pair the
   column less related to the target is dropped),
3. optionally reduce further: `select` (SelectKBest, ANOVA F) or `pca`.

Does not import torch, so it runs without PyTorch and in CI.

CLI:
    python -m src.features             # write results/feature_extraction.json
    python -m src.features --compare   # + compare with the full 122 features on validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

METHODS = ("filter", "select", "pca")


"""EXTRACTOR (fit on train only)"""

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Constant + correlated column filter, optionally followed by selection or PCA.

    method:
        filter  only steps 1 and 2
        select  steps 1, 2 and the `k` best columns by ANOVA F-score vs the target
        pca     steps 1, 2 and PCA keeping `n_components` of the variance (default)
    """

    def __init__(
        self,
        method: str = "pca",
        corr_threshold: float = 0.95,
        k: int = 40,
        n_components: float | int = 0.95,
    ) -> None:
        self.method = method
        self.corr_threshold = corr_threshold
        self.k = k
        self.n_components = n_components

    def fit(self, X, y, feature_names: Sequence[str] | None = None) -> "FeatureExtractor":
        if self.method not in METHODS:
            raise ValueError(f"unknown method '{self.method}', expected one of {METHODS}")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_cols = X.shape[1]
        names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_cols)]
        if len(names) != n_cols:
            raise ValueError("feature_names length does not match X")
        self.input_names_ = names
        self.n_features_in_ = n_cols

        # 1. constant columns
        const = X.var(axis=0) <= 1e-12
        self.constant_ = [names[i] for i in np.flatnonzero(const)]
        alive = np.flatnonzero(~const)

        # 2. correlated columns (greedy, strongest pairs first, deterministic)
        self.correlated_: dict[str, dict] = {}
        if len(alive) > 1:
            Xa = X[:, alive]
            corr = np.corrcoef(Xa, rowvar=False)
            yc = y.astype(np.float64) - y.mean()
            Xc = Xa - Xa.mean(axis=0)
            denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
            target_corr = np.abs(Xc.T @ yc) / np.where(denom == 0, 1, denom)
            i_idx, j_idx = np.triu_indices_from(corr, k=1)
            r = np.abs(corr[i_idx, j_idx])
            order = np.argsort(-r, kind="stable")
            dropped = np.zeros(len(alive), dtype=bool)
            parent: dict[int, int] = {}
            for o in order:
                if r[o] <= self.corr_threshold:
                    break
                i, j = i_idx[o], j_idx[o]
                if dropped[i] or dropped[j]:
                    continue
                drop, keep = (i, j) if target_corr[i] < target_corr[j] else (j, i)
                dropped[drop] = True
                parent[drop] = keep
            for drop, keep in parent.items():
                while dropped[keep]:  # the partner may itself have been dropped later
                    keep = parent[keep]
                self.correlated_[names[alive[drop]]] = {
                    "kept": names[alive[keep]],
                    "r": round(float(corr[drop, keep]), 4),
                }
            alive = alive[~dropped]
        self.filter_idx_ = alive
        Xf = X[:, self.filter_idx_]

        # 3. optional reduction
        self.selector_ = None
        self.pca_ = None
        if self.method == "select":
            k = min(int(self.k), Xf.shape[1])
            self.selector_ = SelectKBest(f_classif, k=k).fit(Xf, y)
            kept = self.filter_idx_[self.selector_.get_support()]
            self.output_names_ = [names[i] for i in kept]
        elif self.method == "pca":
            self.pca_ = PCA(n_components=self.n_components, svd_solver="full", random_state=0).fit(Xf)
            self.output_names_ = [f"pc{i + 1}" for i in range(self.pca_.n_components_)]
        else:
            self.output_names_ = [names[i] for i in self.filter_idx_]
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} columns, got {X.shape[1]}")
        Xf = X[:, self.filter_idx_]
        if self.selector_ is not None:
            Xf = self.selector_.transform(Xf)
        elif self.pca_ is not None:
            Xf = self.pca_.transform(Xf)
        return np.ascontiguousarray(Xf, dtype=np.float32)

    def get_feature_names_out(self) -> list[str]:
        return list(self.output_names_)

    def summary(self) -> dict:
        out = {
            "method": self.method,
            "input_dim": int(self.n_features_in_),
            "output_dim": len(self.output_names_),
            "constant_dropped": self.constant_,
            "correlated_dropped": self.correlated_,
            "after_filter": int(len(self.filter_idx_)),
            "corr_threshold": self.corr_threshold,
        }
        if self.method == "select":
            out["k"] = len(self.output_names_)
            out["selected"] = self.output_names_
        if self.method == "pca":
            out["explained_variance"] = round(float(self.pca_.explained_variance_ratio_.sum()), 4)
        return out


"""APPLY TO PREPARED DATA"""

def extraction_pipeline(pre, extractor: FeatureExtractor) -> Pipeline:
    """Raw feature frame -> method 1 -> method 2 (one object for the whole transform)."""
    return Pipeline([("pre", pre), ("extract", extractor)])


def extract_prepared(prepared, params: dict | None = None):
    """Method no. 2 on a `Prepared` set from src.preprocess (fit on its train part only).

    Returns (new Prepared with extracted features, fitted FeatureExtractor).
    """
    from dataclasses import replace

    extractor = FeatureExtractor(**(params or {})).fit(
        prepared.X_train,
        prepared.y_train,
        feature_names=list(prepared.preprocessor.get_feature_names_out()),
    )
    new = replace(
        prepared,
        X_train=extractor.transform(prepared.X_train),
        X_test=extractor.transform(prepared.X_test),
        preprocessor=extraction_pipeline(prepared.preprocessor, extractor),
    )
    return new, extractor


"""COMPARISON WITH THE FULL FEATURE SET (validation only)"""

def compare_on_validation(prepared, variants: dict[str, dict | None], seed: int = 42) -> dict:
    """Train the same small MLP on each variant, score on a stratified 20% holdout of train.

    `None` = the full 122 features. KDDTest+ is not used here (see docs/evaluation_protocol.md).
    Proxy model: scikit-learn MLP, same hidden sizes as the project MLP.
    """
    from sklearn.neural_network import MLPClassifier

    from src.evaluate import compute_metrics, stratified_holdout

    names = list(prepared.preprocessor.get_feature_names_out())
    tr_idx, va_idx = stratified_holdout(prepared.cat_train, 0.2, seed)
    X_tr, X_va = prepared.X_train[tr_idx], prepared.X_train[va_idx]
    y_tr, y_va = prepared.y_train[tr_idx], prepared.y_train[va_idx]

    results = {}
    for name, params in variants.items():
        if params is None:
            A, B, dim = X_tr, X_va, X_tr.shape[1]
        else:
            ext = FeatureExtractor(**params).fit(X_tr, y_tr, feature_names=names)
            A, B, dim = ext.transform(X_tr), ext.transform(X_va), len(ext.get_feature_names_out())
        clf = MLPClassifier((128, 64), max_iter=20, random_state=seed).fit(A, y_tr)
        scores = clf.predict_proba(B)[:, 1]
        m = compute_metrics(y_va, (scores >= 0.5).astype(int), scores)
        results[name] = {
            "input_dim": dim,
            "f1": round(float(m["f1"]), 4),
            "roc_auc": round(float(m["roc_auc"]), 4),
            "accuracy": round(float(m["accuracy"]), 4),
        }
    return results


"""MAIN, CLI: python -m src.features"""

def main() -> None:
    from src.preprocess import prepare_data

    ap = argparse.ArgumentParser(description="Feature extraction report (method no. 2).")
    ap.add_argument("--config", default="src/config.yaml")
    ap.add_argument("--out", default="results/feature_extraction.json")
    ap.add_argument("--compare", action="store_true", help="also compare with the full 122 features")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    params = dict(cfg.get("features", {}))
    prepared = prepare_data(cfg["data"]["train_path"], cfg["data"]["test_path"])
    _, extractor = extract_prepared(prepared, params)

    report = {"config": params, "extractor": extractor.summary()}
    if args.compare:
        variants = {
            "full": None,
            "filter": {**params, "method": "filter"},
            "select": {**params, "method": "select"},
            "pca": {**params, "method": "pca"},
        }
        report["validation_comparison"] = compare_on_validation(prepared, variants, cfg["random_seed"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()