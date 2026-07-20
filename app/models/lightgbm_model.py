from lightgbm import LGBMClassifier


def build_estimator(
    class_weight=None,
    random_state: int = 42,
    device: str = "cpu",
) -> LGBMClassifier:
    return LGBMClassifier(
        class_weight=class_weight,
        random_state=random_state,
        device_type="gpu" if device == "cuda" else "cpu",
        max_bin=63 if device == "cuda" else 255,
        metric=["average_precision", "binary_logloss"],
        n_jobs=-1,
        verbosity=-1,
    )


def param_distributions() -> dict:
    return {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [-1, 4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.6, 0.8, 1.0],
    }
