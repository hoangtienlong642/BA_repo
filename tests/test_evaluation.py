import numpy as np
import pytest

from app.evaluation import cost_curve, evaluate


class _StubModel:
    """Returns fixed probabilities regardless of X, for deterministic tests."""

    def __init__(self, probabilities):
        self._probabilities = np.asarray(probabilities)

    def predict_proba(self, X):
        return np.column_stack([1 - self._probabilities, self._probabilities])


def test_evaluate_confusion_matrix_and_metrics():
    y_test = [0, 0, 1, 1]
    model = _StubModel([0.1, 0.4, 0.6, 0.9])  # threshold 0.5 -> preds [0, 0, 1, 1]

    result = evaluate(model, X_test=None, y_test=y_test, threshold=0.5)

    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["auc_pr"] == pytest.approx(1.0)


def test_evaluate_single_class_y_test_returns_none_auc():
    y_test = [0, 0, 0, 0]
    model = _StubModel([0.1, 0.4, 0.6, 0.9])

    result = evaluate(model, X_test=None, y_test=y_test, threshold=0.5)

    assert result["auc_pr"] is None
    assert result["roc_auc"] is None
    assert result["confusion_matrix"] == {"tn": 2, "fp": 2, "fn": 0, "tp": 0}
    assert result["precision"] == pytest.approx(0.0)
    assert result["recall"] == pytest.approx(0.0)


def test_cost_curve_finds_zero_cost_threshold():
    y_true = [1, 0]
    y_proba = [0.9, 0.1]
    amounts = [100, 0]
    fp_cost = 10

    curve, best_threshold = cost_curve(
        y_true, y_proba, amounts, fp_cost, thresholds=[0.0, 0.5, 1.0]
    )

    assert best_threshold == pytest.approx(0.5)
    best_row = next(r for r in curve if r["threshold"] == pytest.approx(0.5))
    assert best_row["total_cost"] == pytest.approx(0.0)

    zero_row = next(r for r in curve if r["threshold"] == pytest.approx(0.0))
    assert zero_row["fp_cost"] == pytest.approx(10.0)
    assert zero_row["fn_cost"] == pytest.approx(0.0)

    one_row = next(r for r in curve if r["threshold"] == pytest.approx(1.0))
    assert one_row["fn_cost"] == pytest.approx(100.0)


def test_cost_curve_default_thresholds():
    y_true = [0, 1, 0, 1, 0, 1]
    y_proba = [0.1, 0.8, 0.3, 0.6, 0.2, 0.9]
    amounts = [50, 200, 10, 150, 20, 300]
    fp_cost = 5

    curve, best_threshold = cost_curve(y_true, y_proba, amounts, fp_cost)

    assert len(curve) == 101

    curve_thresholds = [row["threshold"] for row in curve]
    assert curve_thresholds == sorted(curve_thresholds)
    assert curve_thresholds[0] == pytest.approx(0.0)
    assert curve_thresholds[-1] == pytest.approx(1.0)

    assert isinstance(best_threshold, float)
    assert 0.0 <= best_threshold <= 1.0


def test_cost_curve_chunked_matches_unchunked_reference():
    rng = np.random.default_rng(42)
    n_samples = 500
    y_true = rng.integers(0, 2, size=n_samples)
    y_proba = rng.random(n_samples)
    amounts = rng.uniform(1, 1000, size=n_samples)
    fp_cost = 7.5
    thresholds = np.linspace(0.0, 1.0, 137)  # not a multiple of chunk_size

    curve_chunked, best_chunked = cost_curve(
        y_true, y_proba, amounts, fp_cost, thresholds=thresholds, chunk_size=10
    )
    curve_unchunked, best_unchunked = cost_curve(
        y_true, y_proba, amounts, fp_cost, thresholds=thresholds, chunk_size=len(thresholds)
    )

    assert best_chunked == pytest.approx(best_unchunked)
    for row_chunked, row_unchunked in zip(curve_chunked, curve_unchunked):
        assert row_chunked == pytest.approx(row_unchunked)
