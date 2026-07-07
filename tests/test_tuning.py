import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.tuning import time_series_search


def test_time_series_search_returns_fitted_best_estimator():
    rng = np.random.RandomState(0)
    X_train = pd.DataFrame(rng.rand(60, 3), columns=["a", "b", "c"])
    y_train = pd.Series(([0] * 9 + [1]) * 6)  # imbalanced but spread across the series so every TimeSeriesSplit fold contains both classes

    estimator = LogisticRegression()
    param_distributions = {"C": [0.1, 1.0, 10.0]}

    best_estimator, best_params, best_score = time_series_search(
        estimator, param_distributions, X_train, y_train, n_iter=3, n_splits=3, random_state=0
    )

    assert hasattr(best_estimator, "predict_proba")
    assert "C" in best_params
    assert isinstance(best_score, float)
