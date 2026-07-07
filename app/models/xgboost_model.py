from xgboost import XGBClassifier


def build_estimator(scale_pos_weight: float = 1.0, random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="aucpr",
        n_jobs=-1,
    )


def param_distributions() -> dict:
    return {
        "n_estimators": [20, 40, 60, 80],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
