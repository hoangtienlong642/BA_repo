from pathlib import Path
from types import SimpleNamespace

from sklearn.dummy import DummyClassifier

from app import wandb_utils


class _FakeConfig(dict):
    def update(self, values, allow_val_change=False):
        super().update(values)


class _FakeRun:
    def __init__(self):
        self.config = _FakeConfig(model="rf")
        self.logged = []
        self.artifacts = []

    def log(self, values):
        self.logged.append(values)

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)


class _FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path):
        assert Path(path).exists()
        self.files.append(path)


def test_flatten_metrics_includes_scalar_and_confusion_matrix_values():
    metrics = {
        "precision": 0.75,
        "auc_pr": None,
        "confusion_matrix": {"tn": 10, "fp": 2, "fn": 1, "tp": 3},
    }

    result = wandb_utils._flatten_metrics(metrics, "validation")

    assert result == {
        "validation/precision": 0.75,
        "validation/confusion_matrix_tn": 10,
        "validation/confusion_matrix_fp": 2,
        "validation/confusion_matrix_fn": 1,
        "validation/confusion_matrix_tp": 3,
    }


def test_xgboost_callback_logs_each_boosting_round():
    run = _FakeRun()
    callback = wandb_utils.XGBoostMetricsCallback(run)

    callback.after_iteration(
        model=None,
        epoch=3,
        evals_log={
            "validation_0": {"aucpr": [0.5, 0.6], "logloss": [0.4, 0.3]},
            "validation_1": {"aucpr": [0.4, 0.55], "logloss": [0.5, 0.35]},
        },
    )

    assert run.logged[-1] == {
        "boosting_round": 3,
        "round/train/aucpr": 0.6,
        "round/train/logloss": 0.3,
        "round/validation/aucpr": 0.55,
        "round/validation/logloss": 0.35,
    }


def test_lightgbm_callback_logs_each_boosting_round():
    run = _FakeRun()
    callback = wandb_utils.lightgbm_metrics_callback(run)
    environment = SimpleNamespace(
        iteration=2,
        evaluation_result_list=[
            ("training", "binary_logloss", 0.2, False),
            ("validation", "binary_logloss", 0.3, False),
        ],
    )

    callback(environment)

    assert run.logged[-1]["boosting_round"] == 2
    assert run.logged[-1]["round/train/binary_logloss"] == 0.2
    assert run.logged[-1]["round/validation/binary_logloss"] == 0.3


def test_log_results_logs_all_splits_and_model_bundle(monkeypatch):
    run = _FakeRun()
    monkeypatch.setattr(wandb_utils.wandb, "Artifact", _FakeArtifact)
    metrics = {"precision": 0.8, "confusion_matrix": {"tn": 2}}

    wandb_utils.log_results(
        run=run,
        params={"n_estimators": 10},
        split_metrics={"train": metrics, "validation": metrics, "test": metrics},
        best_threshold=0.42,
        best_cv_score=0.7,
        model=DummyClassifier(strategy="most_frequent"),
        selected_features=["amount"],
        training_device="cuda",
    )

    assert run.config["best_parameters"] == {"n_estimators": 10}
    assert run.logged[-1]["train/precision"] == 0.8
    assert run.logged[-1]["validation/precision"] == 0.8
    assert run.logged[-1]["test/precision"] == 0.8
    assert run.logged[-1]["best_cv_auc_pr"] == 0.7
    assert len(run.artifacts) == 1
    assert run.artifacts[0].metadata["training_device"] == "cuda"
