import argparse
import json
import tempfile

import joblib
from sklearn.model_selection import TimeSeriesSplit

from app import config, data, evaluation, imbalance, mlflow_utils, tuning, wandb_utils
from app.learning_curve import plot_learning_curve
from app.models import lightgbm_model, rf, xgboost_model

MODEL_REGISTRY = {
    "rf": rf,
    "xgboost": xgboost_model,
    "lightgbm": lightgbm_model,
}
VALIDATION_FRAC = 0.2


def _temporal_validation_split(X, y, validation_frac: float = VALIDATION_FRAC):
    split_idx = int(len(X) * (1 - validation_frac))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def _parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a fraud detection model.")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True)
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--plot-learning-curve", action="store_true")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Training device. CUDA supports XGBoost and LightGBM; sklearn RF is CPU-only.",
    )
    parser.add_argument(
        "--cv-jobs",
        type=int,
        default=1,
        help="Concurrent CV fits. Keep at 1 for GPU and RF to avoid resource contention.",
    )
    parser.add_argument(
        "--search-verbose",
        type=int,
        default=2,
        help="RandomizedSearchCV progress verbosity (default: 2).",
    )
    parser.add_argument(
        "--tracker",
        choices=["mlflow", "wandb", "none"],
        default="mlflow",
        help="Experiment tracker to use (default: mlflow).",
    )
    parser.add_argument("--wandb-project", default="fraud-detection")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="JSON dict of fixed hyperparameters. Skips RandomizedSearchCV.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=None,
        help="Optional path to save an additional estimator-only joblib file.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and args.model == "rf":
        parser.error(
            "RandomForestClassifier is CPU-only. Use --device cpu for rf, "
            "or choose xgboost/lightgbm for GPU training."
        )
    if args.cv_jobs < 1:
        parser.error("--cv-jobs must be at least 1.")
    if args.device == "cuda" and args.cv_jobs != 1:
        parser.error("Use --cv-jobs 1 with --device cuda to avoid GPU memory contention.")
    return args


def _build_estimator(model_name, model_module, y_train, device):
    if model_name == "xgboost":
        return model_module.build_estimator(
            scale_pos_weight=imbalance.get_scale_pos_weight(y_train),
            random_state=config.RANDOM_SEED,
            device=device,
        )

    estimator_kwargs = {
        "class_weight": imbalance.get_class_weight(y_train),
        "random_state": config.RANDOM_SEED,
    }
    if model_name == "lightgbm":
        estimator_kwargs["device"] = device
    return model_module.build_estimator(**estimator_kwargs)


def main() -> None:
    args = _parse_args()
    model_module = MODEL_REGISTRY[args.model]
    wandb_run = None

    if args.tracker == "wandb":
        wandb_run = wandb_utils.start_run(
            model_name=args.model,
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            config={
                "model": args.model,
                "training_device": args.device,
                "n_iter": args.n_iter,
                "cv_jobs": args.cv_jobs,
                "random_seed": config.RANDOM_SEED,
                "test_fraction": config.TEST_FRAC,
                "validation_fraction_within_train": VALIDATION_FRAC,
                "selected_features": config.SELECTED_FEATURES,
                "fixed_parameters": json.loads(args.params) if args.params else None,
            },
        )
        wandb_utils.log_phase(wandb_run, "loading_data", 0.05)

    print(f"Training device: {args.device}")
    print(f"Loading features from {config.FEATURES_PATH}...")
    df = data.load_features(config.FEATURES_PATH)
    X_train_val, X_test, y_train_val, y_test = data.time_based_split(
        df, test_frac=config.TEST_FRAC
    )
    X_train, X_val, y_train, y_val = _temporal_validation_split(X_train_val, y_train_val)
    print(
        f"Train: {len(X_train):,}, Validation: {len(X_val):,}, "
        f"Test: {len(X_test):,} rows."
    )

    X_train_selected = data.select_features(X_train, config.SELECTED_FEATURES)
    X_val_selected = data.select_features(X_val, config.SELECTED_FEATURES)
    X_test_selected = data.select_features(X_test, config.SELECTED_FEATURES)
    estimator = _build_estimator(args.model, model_module, y_train, args.device)

    wandb_utils.log_phase(wandb_run, "hyperparameter_search", 0.15)
    if args.params:
        best_params = json.loads(args.params)
        print(f"Skipping search, using fixed params: {best_params}")
        best_estimator = estimator.set_params(**best_params)
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
            n_jobs=args.cv_jobs,
            verbose=args.search_verbose,
        )
        print(f"Best CV AUC-PR: {best_cv_score:.4f}, params: {best_params}")
        if wandb_run is not None:
            wandb_run.log({"search/best_cv_auc_pr": best_cv_score})

    print("Fitting selected configuration with temporal validation monitoring...")
    wandb_utils.log_phase(wandb_run, "monitored_fit", 0.65)
    if args.model == "xgboost":
        if wandb_run is not None:
            best_estimator.set_params(callbacks=[wandb_utils.XGBoostMetricsCallback(wandb_run)])
        best_estimator.fit(
            X_train_selected,
            y_train,
            eval_set=[(X_train_selected, y_train), (X_val_selected, y_val)],
            verbose=False,
        )
        best_estimator.callbacks = None
    elif args.model == "lightgbm":
        callbacks = [wandb_utils.lightgbm_metrics_callback(wandb_run)] if wandb_run else None
        best_estimator.fit(
            X_train_selected,
            y_train,
            eval_set=[(X_train_selected, y_train), (X_val_selected, y_val)],
            eval_names=["training", "validation"],
            callbacks=callbacks,
        )
    else:
        best_estimator.fit(X_train_selected, y_train)

    # GPU training is complete. CPU prediction avoids device-mismatch fallback warnings
    # for the pandas-resident train, validation, and test frames.
    if args.model == "xgboost" and args.device == "cuda":
        best_estimator.get_booster().set_param({"device": "cpu"})

    wandb_utils.log_phase(wandb_run, "evaluation", 0.85)
    val_proba = best_estimator.predict_proba(X_val_selected)[:, 1]
    _, best_threshold = evaluation.cost_curve(
        y_val.to_numpy(), val_proba, X_val["amount"].to_numpy(), config.FP_COST
    )
    split_metrics = {
        "train": evaluation.evaluate(best_estimator, X_train_selected, y_train, best_threshold),
        "validation": evaluation.evaluate(best_estimator, X_val_selected, y_val, best_threshold),
        "test": evaluation.evaluate(best_estimator, X_test_selected, y_test, best_threshold),
    }

    artifact_paths = []
    if args.plot_learning_curve:
        print("Generating learning curve...")
        fig = plot_learning_curve(
            best_estimator, X_train_selected, y_train, cv=TimeSeriesSplit(n_splits=5)
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as file:
            fig.savefig(file.name)
            artifact_paths.append(file.name)

    if args.tracker == "mlflow":
        mlflow_utils.init_tracking(config.MLRUNS_DIR)
        mlflow_utils.log_run(
            args.model,
            best_params,
            split_metrics["test"],
            best_threshold,
            best_estimator,
            artifact_paths=artifact_paths,
        )
    elif wandb_run is not None:
        wandb_utils.log_results(
            run=wandb_run,
            params=best_params,
            split_metrics=split_metrics,
            best_threshold=best_threshold,
            best_cv_score=best_cv_score,
            model=best_estimator,
            selected_features=config.SELECTED_FEATURES,
            training_device=args.device,
            artifact_paths=artifact_paths,
        )
        wandb_utils.log_phase(wandb_run, "complete", 1.0)
        wandb_run.finish()

    if args.model_out:
        joblib.dump(best_estimator, args.model_out)
        print(f"Model saved to {args.model_out}")

    print(f"\n=== {args.model} @ threshold {best_threshold:.3f} ===")
    for split_name, metrics in split_metrics.items():
        print(f"{split_name.title()}: {metrics}")


if __name__ == "__main__":
    main()
