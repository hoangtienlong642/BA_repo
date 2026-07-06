import argparse
import json
import tempfile

import joblib
from sklearn.model_selection import TimeSeriesSplit

from app import config, data, evaluation, imbalance, mlflow_utils, monitoring, tuning
from app.learning_curve import plot_learning_curve
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
    parser.add_argument("--plot-learning-curve", action="store_true")
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="JSON dict of fixed hyperparams, e.g. '{\"n_estimators\": 80}'. "
        "Skips the RandomizedSearchCV and fits this exact config directly.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=None,
        help="Optional path to also save the fitted model with joblib (in addition to the MLflow artifact).",
    )
    args = parser.parse_args()

    print(f"Loading features from {config.FEATURES_PATH}...")
    df = data.load_features(config.FEATURES_PATH)
    X_train, X_test, y_train, y_test = data.time_based_split(df, test_frac=config.TEST_FRAC)
    print(f"Train: {len(X_train):,} rows, Test: {len(X_test):,} rows.")

    X_train_selected = data.select_features(X_train, config.SELECTED_FEATURES)
    X_test_selected = data.select_features(X_test, config.SELECTED_FEATURES)

    model_module = MODEL_REGISTRY[args.model]
    if args.model == "xgboost":
        weight = imbalance.get_scale_pos_weight(y_train)
        estimator = model_module.build_estimator(scale_pos_weight=weight, random_state=config.RANDOM_SEED)
    else:
        weight = imbalance.get_class_weight(y_train)
        estimator = model_module.build_estimator(class_weight=weight, random_state=config.RANDOM_SEED)

    if args.params:
        best_params = json.loads(args.params)
        print(f"Skipping search, fitting fixed params: {best_params}")
        best_estimator = estimator.set_params(**best_params)
        best_estimator.fit(X_train_selected, y_train)
        best_cv_score = None
    else:
        print(f"Running hyperparameter search ({args.n_iter} iterations)...")
        best_estimator, best_params, best_cv_score = tuning.time_series_search(
            estimator,
            model_module.param_distributions(),
            X_train_selected,
            y_train,
            n_iter=args.n_iter,
            random_state=config.RANDOM_SEED,
        )
        print(f"Best CV AUC-PR: {best_cv_score:.4f}, params: {best_params}")

    metrics_default = evaluation.evaluate(best_estimator, X_test_selected, y_test, threshold=0.5)
    y_proba = best_estimator.predict_proba(X_test_selected)[:, 1]
    _curve, best_threshold = evaluation.cost_curve(
        y_test.to_numpy(), y_proba, config.FN_COST, config.FP_COST
    )
    metrics_best_threshold = evaluation.evaluate(best_estimator, X_test_selected, y_test, threshold=best_threshold)

    print("Computing reference stats for drift monitoring...")
    reference_stats = monitoring.compute_reference_stats(X_train_selected)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REFERENCE_STATS_PATH, "w") as f:
        json.dump(reference_stats, f)
    print(f"Reference stats saved to {config.REFERENCE_STATS_PATH}")

    artifact_paths = []
    if args.plot_learning_curve:
        print("Generating learning curve...")
        fig = plot_learning_curve(best_estimator, X_train_selected, y_train, cv=TimeSeriesSplit(n_splits=5))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig.savefig(f.name)
            artifact_paths.append(f.name)

    mlflow_utils.init_tracking(config.MLRUNS_DIR)
    mlflow_utils.log_run(args.model, best_params, metrics_best_threshold, best_threshold, best_estimator, artifact_paths=artifact_paths)

    if args.model_out:
        joblib.dump(best_estimator, args.model_out)
        print(f"Model saved to {args.model_out}")

    print(f"\n=== {args.model} ===")
    print(f"Metrics @ threshold 0.5: {metrics_default}")
    print(f"Cost-minimizing threshold: {best_threshold:.3f}")
    print(f"Metrics @ cost-minimizing threshold: {metrics_best_threshold}")


if __name__ == "__main__":
    main()
