# 3. Model Training & Evaluation

## Scope

Training + evaluation pipeline in `app/` for fraud models on top of the feature-engineered parquet (`Synthetic_Financial_datasets_features.parquet`, from `2_feature_engineer.ipynb`). Covers: time-based split, class-imbalance handling, RF/XGBoost/LightGBM training with hyperparameter tuning, evaluation (confusion matrix, AUC-PR, cost-based metric, threshold selection), experiment tracking via MLflow.

**Out of scope**: FastAPI serving layer, Streamlit/HTML demo. Separate spec once models exist.

## Architecture

Flat, script-driven pipeline, run via CLI:

```
parquet -> load_features -> time_based_split -> class weights (train only)
  -> RandomizedSearchCV (TimeSeriesSplit, train only) -> best estimator
  -> evaluate on test -> cost-based threshold search -> MLflow log
```

## Modules (`app/`)

- `config.py` — data path, `mlruns` dir, cost assumptions (`FP_COST` constant, `FN_COST` = transaction `amount`), split fraction (0.8 train), random seed.
- `data.py`
  - `load_features()` — read parquet. Raise clear error if missing, pointing to `2_feature_engineer.ipynb`.
  - `time_based_split(df, test_frac)` — sort by `step`, first `1-test_frac` rows = train, rest = test. No shuffling. Drops ID columns (`nameOrig`, `nameDest`, raw `type`) from the feature matrix.
- `imbalance.py`
  - `get_class_weight(y_train)` — balanced weight dict.
  - Per-model adapters: RF/LightGBM take `class_weight` dict; XGBoost takes `scale_pos_weight` (ratio of negative/positive counts).
- `models/rf.py`, `models/xgboost_model.py`, `models/lightgbm_model.py` — each exposes:
  - `build_estimator()` — base sklearn-API estimator.
  - `param_distributions()` — dict for `RandomizedSearchCV`.
- `tuning.py`
  - `time_series_search(estimator, param_dist, X_train, y_train, n_iter)` — `RandomizedSearchCV` + `TimeSeriesSplit`, scored on `average_precision` (AUC-PR). Returns best estimator.
- `evaluation.py`
  - `evaluate(model, X_test, y_test)` — confusion matrix, precision/recall/F1, AUC-PR, ROC-AUC (secondary).
  - `cost_curve(y_true, y_proba, fn_cost, fp_cost)` — sweep thresholds, total cost = `fn_cost * count(FN) + fp_cost * count(FP)`. Returns curve + threshold minimizing total cost.
  - `cost_at_threshold(y_true, y_pred, fn_cost, fp_cost)` — same cost math at a single fixed threshold (no sweep); reusable by the monitoring module.
- `train.py` — CLI entrypoint: `uv run python -m app.train --model rf|xgboost|lightgbm --n-iter 25`. Orchestrates load -> split -> tune -> evaluate -> log to MLflow (params, metrics, confusion matrix plot, cost curve plot, model artifact) -> prints summary.
- `mlflow_utils.py` — sets local tracking URI (`./mlruns`, gitignored), `log_run(...)` helper.

## Data flow / leakage guard

Test set touched only once, at final evaluation. `RandomizedSearchCV`'s internal CV (`TimeSeriesSplit`) operates within train only — no test leakage into tuning.

## Cost-based metric

`cost(FN) = $500/incident` (missed fraud), `cost(FP) = $5/incident` (false alarm / customer friction) — both flat per-incident constants set in `config.py` (`FN_COST`, `FP_COST`). Threshold chosen by minimizing total cost on the cost curve; reported alongside default 0.5 threshold for comparison.

## Error handling

- Missing parquet -> clear error message, no silent fallback.
- Invalid hyperparameter combo during search -> let `RandomizedSearchCV`'s `error_score` handle/skip, no custom retry.
- `mlruns` not writable -> raise, no silent fallback.

## Testing

- Unit tests (pytest, synthetic fixtures, no full dataset):
  - `time_based_split` — ordering / no-leakage assertion.
  - `cost_curve` — threshold math on tiny synthetic arrays.
  - `get_class_weight` — correctness on known counts.
- No test for MLflow wrapper itself or full train.py end-to-end — verified by running `train.py` once per model during implementation.

## MLflow

Local file-based tracking store (`./mlruns`, gitignored). View via `mlflow ui`.
