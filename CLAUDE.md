# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fraud detection on the PaySim **Synthetic Financial Datasets** (~6.3M transactions, ~0.1% fraud). Scope: data cleaning → EDA → feature engineering → several fraud models (Random Forest, XGBoost, LightGBM, …) with class-imbalance handling → evaluation (AUC-PR, confusion matrix, **cost-based** metric weighing missed-fraud loss vs. false-alarm friction, and an operating threshold reflecting that trade-off) → a demo front-end for inference and multi-model comparison.

Tech stack: pandas, numpy, scikit-learn, matplotlib, seaborn, FastAPI (serving), Streamlit / HTML (demo). Package manager: **`uv`**.

## Spec-driven workflow (required)

Write the plan and solution to `docs/` **before** implementing or training. Documents are numbered by pipeline stage, e.g. `docs/1_cleaning_data.md`, `docs/2_feature_engineering.md`, … Design first, then build against the spec.

## Package management — use `uv`

Use `uv` for all dependency and execution needs, not `pip`/`venv`.
- `uv add <pkg>` to add a dependency; `uv sync` to install.
- `uv run <cmd>` to run scripts/servers in the project environment.

Note: `requirements.txt` and the Vietnamese `README.md` document an older pip/`venv` + Jupyter kernel setup and are out of date relative to the `uv` direction — prefer `uv` and update these when touched.

## Data pipeline (notebooks, run in order)

The pipeline is a sequence of numbered notebooks; each consumes the previous stage's output. Run top-to-bottom in order:

1. `0_get_data.ipynb` / `get_data.py` — download the Kaggle dataset → `Synthetic_Financial_datasets_log.csv` (~178 MB). `python get_data.py` handles download, unzip, rename, and preview.
2. `1_clean_data.ipynb` — type/missing/duplicate checks, IQR outlier analysis, balance business-logic validation; drops `isFlaggedFraud`.
3. `2_feature_engineer.ipynb` — produces `Synthetic_Financial_datasets_features.parquet` (~800 MB, ~48 features). **Heavy: ~2 hours on the full 6.3M rows** (cumulative + 24h sliding-window features). Don't re-run casually; work on a sample when iterating.
4. `3_eda_data.ipynb` — exploratory analysis and plots.

Data files (`*.csv`, `*.parquet`) are gitignored and never committed — regenerate them via the pipeline. Output is stored as compressed Parquet (pyarrow) for memory/speed.

## Critical domain conventions

- **Temporal ordering prevents leakage.** The data is sorted chronologically by `step` (hours since start) before any cumulative/velocity feature is computed. Any train/test split must be time-based, not random — a shuffle would leak future information into the past.
- **Fraud lives only in `TRANSFER` and `CASH_OUT`** transaction types. Feature engineering keeps all types (so models learn context) but flags this via `is_transfer_or_cashout`.
- **Merchant accounts** (`nameDest` starting with `M`) never receive the balance fields the dataset populates for customers; handled via the `is_merchant_dest` indicator.
- Engineered features fall into groups: one-hot `type_*`, cyclical time (`hour_of_day`, `day_of_month`, `day_of_week`), balance-discrepancy (`errorBalanceOrig`, `errorBalanceDest`, ratios, zero-balance flags), and per-account cumulative history + 24h velocity (`orig_cum_*`, `dest_cum_*`, `dest_txn_last_24h`, …).
- **`step` is hours**, not a timestamp: `hour_of_day = step % 24`, `day_of_month = step // 24`.

## Modeling & evaluation guidance

- Extreme imbalance (~0.1%): prefer **AUC-PR** over ROC-AUC; report the confusion matrix and a cost-based metric. Do not tune to raw accuracy.
- The operating threshold is a business decision (fraud loss vs. customer friction), not a fixed 0.5 — surface it as a tunable parameter driven by the assumed cost matrix.
