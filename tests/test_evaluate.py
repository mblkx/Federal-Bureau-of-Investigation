"""Metric and validation-split tests on small hand-computed examples."""
 
import json
import math
 
import numpy as np
import pytest
 
from src.evaluate import (
    compute_metrics, detection_rate_by_category, evaluate, format_report,
    save_results, stratified_holdout,
)
 
#tp=3, fn=1, fp=1, tn=5
Y_TRUE = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
Y_PRED = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])

def test_metrics_hand_computed():
    m = compute_metrics(Y_TRUE, Y_PRED)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 5)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(0.8)
    assert math.isnan(m["roc_auc"])  # no scores given
 
 
def test_f1_is_harmonic_mean():
    # precision 0.9, recall 0.72 -> F1 0.8 (example from the project notes)
    y_true = np.array([1] * 100 + [0] * 100)
    y_pred = np.array([1] * 72 + [0] * 28 + [1] * 8 + [0] * 92)
    m = compute_metrics(y_true, y_pred)
    assert m["precision"] == pytest.approx(0.9)
    assert m["recall"] == pytest.approx(0.72)
    assert m["f1"] == pytest.approx(0.8)
 
 
def test_always_normal_model_has_zero_f1():
    m = compute_metrics(Y_TRUE, np.zeros_like(Y_TRUE))
    assert m["f1"] == 0.0 and m["recall"] == 0.0 and m["precision"] == 0.0
 
 
def test_auc_perfect_and_single_class():
    scores = np.array([0.9, 0.8, 0.95, 0.7, 0.1, 0.2, 0.05, 0.3, 0.15, 0.25])
    assert compute_metrics(Y_TRUE, Y_PRED, scores)["roc_auc"] == pytest.approx(1.0)
    only_attacks = np.ones(4, dtype=int)
    assert math.isnan(compute_metrics(only_attacks, only_attacks, np.ones(4))["roc_auc"])
 
 
def test_detection_rate_by_category():
    cats = np.array(["DoS", "DoS", "Probe", "R2L", "normal", "normal", "normal", "normal"])
    pred = np.array([1, 0, 1, 0, 0, 0, 0, 1])
    rates = detection_rate_by_category(pred, cats)
    assert rates["DoS"] == pytest.approx(0.5)
    assert rates["Probe"] == pytest.approx(1.0)
    assert rates["R2L"] == pytest.approx(0.0)
    assert rates["normal"] == pytest.approx(0.75)  # true-negative rate
 
 
def test_evaluate_threshold_and_categories():
    cats = np.array(["DoS", "DoS", "normal", "normal"])
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.2, 0.6])
    r = evaluate(y, scores, cats, threshold=0.5)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 1)
    assert r["by_category"]["DoS"] == pytest.approx(0.5)
    assert evaluate(y, scores, cats, threshold=0.3)["recall"] == pytest.approx(1.0)
 
 
def test_report_and_save(tmp_path):
    cats = np.array(["DoS", "normal"])
    r = evaluate(np.array([1, 0]), np.array([0.9, 0.1]), cats)
    assert "f1" in format_report(r, "demo")
    save_results(r, tmp_path / "sub" / "r.json")
    assert json.loads((tmp_path / "sub" / "r.json").read_text())["f1"] == pytest.approx(1.0)


#Validation
OUNTS = {"normal": 600, "DoS": 300, "Probe": 80, "R2L": 20, "U2R": 5}
 
 
@pytest.fixture()
def cats():
    arr = np.concatenate([np.full(n, c) for c, n in COUNTS.items()])
    return np.random.default_rng(0).permutation(arr)
 
 
def test_holdout_disjoint_and_covers(cats):
    tr, va = stratified_holdout(cats, 0.2, seed=1)
    assert len(set(tr) & set(va)) == 0
    assert np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(len(cats)))
 
 
def test_holdout_is_stratified(cats):
    _, va = stratified_holdout(cats, 0.2, seed=1)
    for cat, n in COUNTS.items():
        assert (cats[va] == cat).sum() == pytest.approx(0.2 * n, abs=1)
    assert (cats[va] == "U2R").sum() >= 1  # rare class still present in validation
 
 
def test_holdout_deterministic(cats):
    a = stratified_holdout(cats, 0.2, seed=3)
    b = stratified_holdout(cats, 0.2, seed=3)
    c = stratified_holdout(cats, 0.2, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[1], c[1])
 
 
def test_holdout_edge_cases():
    tiny = np.array(["normal"] * 10 + ["U2R"])  # single-row category stays in train
    tr, va = stratified_holdout(tiny, 0.2, seed=0)
    assert (tiny[va] == "U2R").sum() == 0 and (tiny[tr] == "U2R").sum() == 1
    for bad in (0, 1, -0.1, 1.5):
        with pytest.raises(ValueError):
            stratified_holdout(tiny, bad, seed=0)