import os

import mlflow
import mlflow.sklearn


def init_tracking(tracking_dir) -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file:{tracking_dir}")


def log_run(model_name: str, params: dict, metrics: dict, best_threshold: float, model) -> None:
    flat_metrics = {
        k: v for k, v in metrics.items() if isinstance(v, (int, float))
    }
    if "confusion_matrix" in metrics:
        for k, v in metrics["confusion_matrix"].items():
            flat_metrics[f"confusion_matrix_{k}"] = v

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(flat_metrics)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.sklearn.log_model(model, artifact_path="model")
