import tempfile
from pathlib import Path

import joblib
import wandb
from xgboost.callback import TrainingCallback


def _flatten_metrics(metrics: dict, prefix: str) -> dict:
    flattened = {
        f"{prefix}/{key}": value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and value is not None
    }
    for key, value in metrics.get("confusion_matrix", {}).items():
        flattened[f"{prefix}/confusion_matrix_{key}"] = value
    return flattened


def start_run(
    model_name: str,
    project: str,
    entity: str | None,
    mode: str,
    config: dict,
):
    return wandb.init(
        project=project,
        entity=entity,
        name=model_name,
        job_type="training",
        config=config,
        mode=mode,
    )


def log_phase(run, phase: str, progress: float) -> None:
    if run is not None:
        run.log({"pipeline/phase": phase, "pipeline/progress": progress})


class XGBoostMetricsCallback(TrainingCallback):
    def __init__(self, run):
        self.run = run

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        metrics = {"boosting_round": epoch}
        for dataset_name, dataset_metrics in evals_log.items():
            split_name = "train" if dataset_name == "validation_0" else "validation"
            for metric_name, values in dataset_metrics.items():
                metrics[f"round/{split_name}/{metric_name}"] = values[-1]
        self.run.log(metrics)
        return False


def lightgbm_metrics_callback(run):
    def callback(environment) -> None:
        metrics = {"boosting_round": environment.iteration}
        for dataset_name, metric_name, value, _ in environment.evaluation_result_list:
            split_name = "train" if dataset_name == "training" else "validation"
            metrics[f"round/{split_name}/{metric_name}"] = value
        run.log(metrics)

    callback.order = 20
    callback.before_iteration = False
    return callback


def log_results(
    run,
    params: dict,
    split_metrics: dict[str, dict],
    best_threshold: float,
    best_cv_score: float | None,
    model,
    selected_features: list[str],
    training_device: str,
    artifact_paths=None,
) -> None:
    run.config.update({"best_parameters": params}, allow_val_change=True)
    logged_metrics = {
        key: value
        for split_name, metrics in split_metrics.items()
        for key, value in _flatten_metrics(metrics, split_name).items()
    }
    logged_metrics["best_threshold"] = best_threshold
    if best_cv_score is not None:
        logged_metrics["best_cv_auc_pr"] = best_cv_score
    run.log(logged_metrics)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model.joblib"
        joblib.dump(
            {
                "model": model,
                "threshold": best_threshold,
                "selected_features": selected_features,
            },
            model_path,
        )
        model_artifact = wandb.Artifact(
            name=f"{run.config['model']}-model",
            type="model",
            metadata={
                "best_threshold": best_threshold,
                "training_device": training_device,
            },
        )
        model_artifact.add_file(str(model_path))
        run.log_artifact(model_artifact)

    for path in artifact_paths or []:
        run.log({Path(path).stem: wandb.Image(path)})
