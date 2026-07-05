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
