# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Talk like a caveman.

## Project

Fraud detection on the PaySim **Synthetic Financial Datasets** (~6.3M transactions, ~0.1% fraud). Two halves:

1. **Offline pipeline** (notebooks): data cleaning → EDA → feature engineering → model training (Random Forest, XGBoost, LightGBM, Logistic Regression) with class-imbalance handling → evaluation (AUC-PR, confusion matrix, cost-based metric, cost-minimizing threshold).
2. **Online serving app** (`app/` + `webapp/`): FastAPI real-time scoring API with a review queue, drift/retraining monitoring, and a Streamlit multi-tab dashboard (model comparison, EDA, feature docs, live ticker streaming).

Tech stack: pandas, numpy, scikit-learn, xgboost, lightgbm, mlflow, FastAPI, Streamlit, SQLite. Package manager: **`uv`**.

## Spec-driven workflow (required)

Write the plan and solution to `docs/` **before** implementing or training. Documents are numbered by pipeline stage, e.g. `docs/1_cleaning_data.md`, `docs/2_feature_engineering.md`, … Design first, then build against the spec.

## Package management — use `uv`

Use `uv` for all dependency and execution needs, not `pip`/`venv`.
- `uv add <pkg>` to add a dependency; `uv sync` to install.
- `uv run <cmd>` to run scripts/servers/tests in the project environment.

Note: `requirements.txt` documents an older pip/`venv` setup and diverges from `pyproject.toml` (e.g. `fastapi`, `streamlit`, `scipy`, `matplotlib` are in `requirements.txt` but missing from `pyproject.toml`'s `dependencies`) — prefer `uv`/`pyproject.toml` and reconcile these when touched.

## Commands

```bash
# Train a model (writes to MLflow under mlruns/, optionally also joblib)
uv run python -m app.train --model rf --n-iter 25
uv run python -m app.train --model xgboost --params '{"n_estimators": 80, "max_depth": 8}' --model-out model/model.joblib

# Run the scoring API (dev)
PYTHONPATH=. uv run uvicorn app.api:app --reload --port 12075

# Run the Streamlit dashboard (dev)
uv run streamlit run webapp/app.py --server.port 12076

# Full stack via Docker Compose (api on :12075, webapp on :12076)
docker-compose up --build

# Tests
uv run pytest
uv run pytest tests/test_evaluation.py -k cost_curve   # single test
```

`--model` in `app/train.py` must be one of the keys in `MODEL_REGISTRY` (`app/train.py`): `rf`, `xgboost`, `lightgbm`, `lr`. Hyperparameter search spaces live per-model in `app/models/{rf,xgboost_model,lightgbm_model,lr}.py::param_distributions()`.

## Data pipeline (notebooks, run in order)

1. `0_get_data.ipynb` / `get_data.py` — download the Kaggle dataset → `Synthetic_Financial_datasets_log.csv` (~178 MB).
2. `1_clean_data.ipynb` — type/missing/duplicate checks, IQR outlier analysis, balance business-logic validation; drops `isFlaggedFraud`.
3. `2_feature_engineer.ipynb` — produces `Synthetic_Financial_datasets_features.parquet` (~800 MB, ~48 features). **Heavy: ~2 hours on the full 6.3M rows.** Don't re-run casually; work on a sample when iterating.
4. `3_eda_data.ipynb` — exploratory analysis and plots.
5. `4_feature_investigation.ipynb` — feature selection/retuning follow-up; see `docs/2b_feature_selection*.md`.

Data files (`*.csv`, `*.parquet`) and `mlruns/` are gitignored — regenerate via the pipeline. Output is stored as compressed Parquet (pyarrow).

## Critical domain conventions

- **Temporal ordering prevents leakage.** Data is sorted chronologically by `step` before any cumulative/velocity feature is computed. Splits must be time-based (`app/data.py::time_based_split`), not random.
- **Fraud lives only in `TRANSFER` and `CASH_OUT`** transaction types, flagged via `is_transfer_or_cashout`.
- **Merchant accounts** (`nameDest` starting with `M`) never get customer balance fields; handled via `is_merchant_dest`.
- Feature groups: one-hot `type_*`, cyclical time (`hour_of_day`, `day_of_month`, `day_of_week`), balance-discrepancy (`errorBalanceOrig`/`errorBalanceDest`, ratios, zero-balance flags), per-account cumulative history + 24h velocity (`orig_cum_*`, `dest_cum_*`, `dest_txn_last_24h`, …).
- **`step` is hours**, not a timestamp: `hour_of_day = step % 24`, `day_of_month = step // 24`.
- The **serving-time feature extractor is a separate, simplified implementation**: `app/features.py::extract_features` computes a reduced feature set on the fly from raw transaction fields (no historical/cumulative context, since there's no per-account history at request time). `app/config.py::SELECTED_FEATURES` (20 features) is the single source of truth for column order/presence and is shared by both training (`app/data.py::select_features`) and serving — keep the two extractors' outputs consistent with this list when adding features.

## Modeling & evaluation guidance

- Extreme imbalance (~0.1%): prefer **AUC-PR** over ROC-AUC (`app/evaluation.py::evaluate`); report confusion matrix + cost-based metric. Do not tune to raw accuracy.
- `app/evaluation.py::cost_curve` sweeps thresholds against `config.FP_COST` (fixed friction cost per false alarm) vs. missed-fraud loss = the transaction `amount`, and returns the cost-minimizing threshold — this is the "operating threshold," a business decision, not a fixed 0.5.
- Imbalance handling differs by model family (`app/imbalance.py`): `class_weight` for sklearn-style estimators (rf, lr), `scale_pos_weight` for xgboost.
- Every training run is logged to MLflow (`app/mlflow_utils.py`, file store under `mlruns/`) with params, metrics (including per-cell confusion matrix and `best_threshold`), and the fitted model artifact.

## Serving app architecture (`app/api.py`)

- Model loading is lazy + cached (`load_scoring_model`, module-level `_cached_model`) from `model/model.joblib`; if absent, `/predict` falls back to a hardcoded heuristic (not the trained model) — check which path is active when debugging serving behavior.
- `/predict` runs `app/features.py::extract_features_single`, scores it, persists the transaction + score to SQLite (`app/db.py`, `data/fraud.db`) via `db.insert_transaction`, and returns a fixed **threshold=0.05** decision plus rule-based `anomaly_highlights` (separate from the model score, used for UI explainability).
- `/queue` + `/queue/{id}/review` implement human-in-the-loop review: an analyst's `APPROVED`/`DECLINED`/`ESCALATED` action backfills `ground_truth` on the transaction (`DECLINED` → 1, `APPROVED` → 0), which is what powers `/monitoring/metrics`.
- `/monitoring/metrics` computes precision/recall/F1 only over reviewed transactions (no ground truth otherwise). `/monitoring/drift` does a KS-test per feature against synthetic baseline distributions (hardcoded in `get_data_drift`, not derived from real training data) to flag drift. `/monitoring/triggers` combines a recall-drop trigger (recall < 80% with ≥10 reviews) and the drift trigger into a single retrain-or-not signal.
- `app/db.py` runs `init_db()` at import time in addition to the FastAPI startup event — both the API process and any script importing `app.db` will create `data/fraud.db` and its schema as a side effect.

## Webapp (`webapp/app.py`)

Streamlit dashboard with 4 tabs: model results/comparison, data source & EDA, feature list (documents `SELECTED_FEATURES`), and real-time streaming (CSV upload replay, random transaction push, continuous auto-stream, multi-model ticker chart). Talks to the FastAPI backend via `API_URL` (from `.env`, see `.env.example`), not by importing `app/` directly — treat it as an independent HTTP client when tracing request flow.
