"""Evaluation metrics and validation split for the NSL-KDD / FedAvg project.
 
Framework-agnostic: works on numpy arrays, so the same code scores the
centralized baseline, local models and the global FedAvg model.
 
Positive class = attack (is_attack == 1). Models output an attack score
(probability); a sample is predicted as attack when score >= threshold.
 
Rules (see docs/evaluation_protocol.md):
* tune hyperparameters / threshold / number of rounds on VALIDATION only,
* KDDTest+ is used for final reporting only.
"""
 
from __future__ import annotations
 
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
 
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
 
NORMAL = "normal"


"""VALIDATION SPLIT"""

def stratified_holdout(
    categories: np.ndarray, val_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split row indices into (train_idx, val_idx), stratified by category.
 
    Use it on the full train set (centralized baseline) or on one client's
    shard (local holdout; the global validation set is the union of the local
    ones). A category with a single row stays in train.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    categories = np.asarray(categories)
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for cat in sorted(set(categories.tolist())):
        idx = rng.permutation(np.flatnonzero(categories == cat))
        n_val = int(round(len(idx) * val_fraction))
        n_val = min(max(n_val, 1), len(idx) - 1) if len(idx) > 1 else 0
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))


"""METRICS"""

def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Binary metrics with attack = positive class."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
 
    auc = float("nan")
    if y_score is not None and len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_score))
 
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
 
 
def detection_rate_by_category(y_pred: np.ndarray, categories: np.ndarray) -> Dict[str, float]:
    """Share of rows predicted correctly, per category.
 
    For an attack category it is the detection rate (recall); for 'normal'
    it is the true-negative rate (1 - false-alarm rate).
    """
    y_pred = np.asarray(y_pred).astype(int)
    categories = np.asarray(categories)
    out: Dict[str, float] = {}
    for cat in sorted(set(categories.tolist())):
        mask = categories == cat
        correct = (y_pred[mask] == (0 if cat == NORMAL else 1)).mean()
        out[cat] = float(correct)
    return out
 
 
def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    categories: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Threshold scores, compute all metrics and (optionally) per-category rates."""
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    result: Dict[str, object] = {"threshold": threshold, **compute_metrics(y_true, y_pred, y_score)}
    if categories is not None:
        result["by_category"] = detection_rate_by_category(y_pred, categories)
    return result


"""REPORTING"""

def format_report(result: Dict[str, object], title: str = "") -> str:
    lines = [f"== {title} ==" if title else "== evaluation =="]
    for key in ("f1", "precision", "recall", "accuracy", "roc_auc"):
        lines.append(f"{key:>10}: {result[key]:.4f}")
    lines.append(f"{'confusion':>10}: tn={result['tn']} fp={result['fp']} fn={result['fn']} tp={result['tp']}")
    for cat, rate in (result.get("by_category") or {}).items():
        label = "true-negative rate" if cat == NORMAL else "detection rate"
        lines.append(f"{cat:>10}: {rate:.3f} ({label})")
    return "\n".join(lines)
 
 
def save_results(result: Dict[str, object], path: str | Path) -> None:
    """Write metrics to JSON (e.g. experiments/results/<run>.json)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")