import pandas as pd
import pytest

from app.imbalance import get_class_weight, get_scale_pos_weight


def test_get_class_weight_known_counts():
    y_train = pd.Series([0] * 90 + [1] * 10)
    weights = get_class_weight(y_train)

    assert weights[0] == pytest.approx(100 / (2 * 90))
    assert weights[1] == pytest.approx(100 / (2 * 10))


def test_get_scale_pos_weight_known_counts():
    y_train = pd.Series([0] * 90 + [1] * 10)
    ratio = get_scale_pos_weight(y_train)
    assert ratio == pytest.approx(9.0)


def test_get_scale_pos_weight_no_positives_raises():
    y_train = pd.Series([0] * 10)
    with pytest.raises(ValueError, match="No positive samples"):
        get_scale_pos_weight(y_train)
