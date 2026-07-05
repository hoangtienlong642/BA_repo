import argparse

from app import config, data, evaluation, imbalance, mlflow_utils, tuning
from app.models import lightgbm_model, rf, xgboost_model

MODEL_REGISTRY = {
    "rf": rf,
    "xgboost": xgboost_model,
    "lightgbm": lightgbm_model,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a fraud detection model.")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True)
    parser.add_argument("--n-iter", type=int, default=25)
    args = parser.parse_args()

    print(f"Loading features from {config.FEATURES_PATH}...")
    df = data.load_features(config.FEATURES_PATH)
    X_train, X_test, y_train, y_test = data.time_based_split(df, test_frac=config.TEST_FRAC)
    print(f"Train: {len(X_train):,} rows, Test: {len(X_test):,} rows.")

    model_module = MODEL_REGISTRY[args.model]
    if args.model == "xgboost":
        weight = imbalance.get_scale_pos_weight(y_train)
        estimator = model_module.build_estimator(scale_pos_weight=weight, random_state=config.RANDOM_SEED)
    else:
        weight = imbalance.get_class_weight(y_train)
        estimator = model_module.build_estimator(class_weight=weight, random_state=config.RANDOM_SEED)

    print(f"Running hyperparameter search ({args.n_iter} iterations)...")
    best_estimator, best_params, best_cv_score = tuning.time_series_search(
        estimator,
        model_module.param_distributions(),
        X_train,
        y_train,
        n_iter=args.n_iter,
        random_state=config.RANDOM_SEED,
    )
    print(f"Best CV AUC-PR: {best_cv_score:.4f}, params: {best_params}")

    metrics_default = evaluation.evaluate(best_estimator, X_test, y_test, threshold=0.5)
    y_proba = best_estimator.predict_proba(X_test)[:, 1]
    amounts_test = X_test["amount"].to_numpy()
    _curve, best_threshold = evaluation.cost_curve(
        y_test.to_numpy(), y_proba, amounts_test, config.FP_COST
    )
    metrics_best_threshold = evaluation.evaluate(best_estimator, X_test, y_test, threshold=best_threshold)

    mlflow_utils.init_tracking(config.MLRUNS_DIR)
    mlflow_utils.log_run(args.model, best_params, metrics_best_threshold, best_threshold, best_estimator)

    print(f"\n=== {args.model} ===")
    print(f"Metrics @ threshold 0.5: {metrics_default}")
    print(f"Cost-minimizing threshold: {best_threshold:.3f}")
    print(f"Metrics @ cost-minimizing threshold: {metrics_best_threshold}")


if __name__ == "__main__":
    main()
