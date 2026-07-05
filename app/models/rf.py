from sklearn.ensemble import RandomForestClassifier


def build_estimator(class_weight=None, random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )


def param_distributions() -> dict:
    return {
        "n_estimators": [20, 40, 60, 80],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", 0.5],
    }
