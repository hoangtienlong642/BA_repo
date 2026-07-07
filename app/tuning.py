from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


def time_series_search(
    estimator,
    param_distributions: dict,
    X_train,
    y_train,
    n_iter: int = 25,
    n_splits: int = 5,
    random_state: int = 42,
):
    cv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=random_state,
        n_jobs=6,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_
