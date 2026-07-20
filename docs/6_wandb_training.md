# Training all models with Weights & Biases

This guide runs the existing time-aware fraud-detection training pipeline and records each run in a W&B project. MLflow remains the default tracker; pass `--tracker wandb` explicitly to use W&B.

## 1. Prerequisites

The engineered feature dataset must exist at the repository root:

```text
Synthetic_Financial_datasets_features.parquet
```

If it is missing, run the data preparation and feature-engineering notebooks first. Install the locked project dependencies:

```bash
uv sync --dev
```

For online tracking, create a W&B account and authenticate once:

```bash
uv run wandb login
```

For CI or a remote training machine, provide the key through the environment instead of committing it:

```bash
export WANDB_API_KEY="<your-api-key>"
export WANDB_PROJECT="fraud-detection"
```

`WANDB_API_KEY` must never be placed in source control or an `.env` file that is committed.

## 2. Train one model with automatic hyperparameter search

Random Forest:

```bash
uv run python -m app.train \
  --model rf \
  --n-iter 25 \
  --tracker wandb \
  --wandb-project fraud-detection
```

XGBoost:

```bash
uv run python -m app.train \
  --model xgboost \
  --n-iter 25 \
  --tracker wandb \
  --wandb-project fraud-detection
```

LightGBM:

```bash
uv run python -m app.train \
  --model lightgbm \
  --n-iter 25 \
  --tracker wandb \
  --wandb-project fraud-detection
```

Each automatic run uses `RandomizedSearchCV` with `TimeSeriesSplit` and AUC-PR scoring. Increase `--n-iter` for a broader search or reduce it for a smoke test.

## 3. Train all models sequentially

The following loop creates one W&B run and one versioned model artifact for each model:

```bash
for model in rf xgboost lightgbm; do
  uv run python -m app.train \
    --model "$model" \
    --n-iter 25 \
    --tracker wandb \
    --wandb-project fraud-detection
done
```

These models can consume substantial CPU and memory on the full dataset. Run them sequentially unless the machine has enough capacity for multiple concurrent searches.

## 4. Train with fixed parameters

Use `--params` to skip hyperparameter search and reproduce a specific configuration:

```bash
uv run python -m app.train \
  --model xgboost \
  --params '{"n_estimators": 80, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}' \
  --tracker wandb \
  --wandb-project fraud-detection
```

A fixed-parameter run has no `best_cv_auc_pr` metric because cross-validation search is skipped.

## 5. Optional flags

Log the learning-curve image to the run:

```bash
uv run python -m app.train \
  --model rf \
  --n-iter 10 \
  --plot-learning-curve \
  --tracker wandb \
  --wandb-project fraud-detection
```

Assign the run to a W&B team:

```bash
uv run python -m app.train \
  --model lightgbm \
  --tracker wandb \
  --wandb-project fraud-detection \
  --wandb-entity <team-name>
```

Save an additional estimator-only joblib file locally:

```bash
mkdir -p models
uv run python -m app.train \
  --model rf \
  --tracker wandb \
  --model-out models/rf.joblib
```

The W&B artifact is more deployment-ready than `--model-out`: its `model.joblib` bundle contains the estimator, selected decision threshold, and selected feature names.

## 6. Offline and disabled modes

Record locally without synchronizing during training:

```bash
uv run python -m app.train \
  --model rf \
  --tracker wandb \
  --wandb-mode offline
```

W&B writes offline runs under `wandb/`, which is ignored by Git. Synchronize a completed run later:

```bash
uv run wandb sync wandb/offline-run-<run-id>
```

Exercise the W&B code path without recording or uploading a run:

```bash
uv run python -m app.train \
  --model rf \
  --tracker wandb \
  --wandb-mode disabled
```

Skip experiment tracking entirely with `--tracker none`. Omit `--tracker` to retain the existing local MLflow behavior.

## 7. Data recorded for each W&B run

Each run records:

- Model name, fitted hyperparameters, and selected features
- Metrics at threshold `0.5`
- Metrics at the cost-minimizing threshold
- Confusion-matrix counts for both thresholds
- Best threshold and, for search runs, best cross-validation AUC-PR
- An optional learning-curve image
- A versioned model artifact containing the estimator, threshold, and feature list

The feature parquet dataset is not uploaded to W&B.

## 8. GPU training and Random Forest performance

The `--device` option controls model training:

- `xgboost --device cuda` uses XGBoost's native CUDA histogram trainer.
- `lightgbm --device cuda` uses LightGBM's GPU backend. With the standard Linux wheel in this environment, that backend uses OpenCL on the NVIDIA GPU; the wheel was not compiled with LightGBM's separate native CUDA tree learner.
- `rf` uses scikit-learn and is CPU-only. The CLI rejects `--model rf --device cuda` instead of silently running it on CPU.

Keep `--cv-jobs 1` for GPU training. Parallel CV fits would compete for the same GPU and can exhaust GPU memory.

XGBoost on GPU:

```bash
uv run python -m app.train \
  --model xgboost \
  --device cuda \
  --cv-jobs 1 \
  --n-iter 25 \
  --tracker wandb \
  --wandb-project fraud-detection
```

LightGBM on GPU:

```bash
uv run python -m app.train \
  --model lightgbm \
  --device cuda \
  --cv-jobs 1 \
  --n-iter 25 \
  --tracker wandb \
  --wandb-project fraud-detection
```

Random Forest uses every available CPU core inside each fit. CV defaults to one fit at a time to prevent nested parallelism and excessive memory usage:

```bash
uv run python -m app.train \
  --model rf \
  --device cpu \
  --cv-jobs 1 \
  --n-iter 10 \
  --tracker wandb \
  --wandb-project fraud-detection
```

For faster RF iteration:

1. Start with `--n-iter 5` or `--n-iter 10`; 25 iterations with five time-series folds requires up to 125 model fits.
2. Use `--params` after identifying a promising configuration to perform only one full-data fit.
3. Prefer finite `max_depth`, larger `min_samples_leaf`, fewer `n_estimators`, and `max_samples` below `1.0` when the validation score remains acceptable.
4. Do not raise `--cv-jobs` while each RF estimator uses all CPU cores. Parallel full-dataset RF fits substantially increase memory pressure and usually oversubscribe the CPU.

A quick RF baseline is:

```bash
uv run python -m app.train \
  --model rf \
  --device cpu \
  --params '{"n_estimators": 80, "max_depth": 10, "min_samples_leaf": 5, "max_features": "sqrt", "max_samples": 0.75}' \
  --tracker wandb \
  --wandb-project fraud-detection
```

## 9. Live run observability

A W&B run is created before data loading, rather than after training. The run records these pipeline phases while the command is active:

```text
loading_data -> hyperparameter_search -> monitored_fit -> evaluation -> complete
```

The run configuration is visible immediately and includes the model, requested training device, search iterations, CV concurrency, random seed, temporal split fractions, selected features, and fixed parameters when supplied. The selected search parameters and best CV AUC-PR are added after search.

The temporal data flow is:

```text
earliest 64% of rows  -> training and time-series parameter search
next 16% of rows      -> validation monitoring and threshold selection
latest 20% of rows    -> one final unbiased test evaluation
```

For XGBoost, each boosting round streams these charts to W&B during the monitored fit:

- `round/train/aucpr`
- `round/train/logloss`
- `round/validation/aucpr`
- `round/validation/logloss`

LightGBM provides the corresponding `average_precision` and `binary_logloss` charts. Random Forest does not have boosting epochs, so it records pipeline phases and final train/validation/test metrics but no per-round loss curve.

After training, every model logs precision, recall, F1, AUC-PR, ROC-AUC, log loss, cost-selected threshold, and confusion-matrix counts for train, validation, and test. The test set is not evaluated for every hyperparameter candidate because doing so would leak test information into model selection.

`RandomizedSearchCV` also prints candidate/fold completion in the terminal. Control that output with `--search-verbose`:

```bash
uv run python -m app.train \
  --model xgboost \
  --device cuda \
  --n-iter 25 \
  --search-verbose 2 \
  --tracker wandb
```
