# Drift & performance monitoring — design

## Scope

Offline batch monitoring module: compare incoming transaction feature
distributions to training data (PSI), track rolling precision/recall/cost on
a simulated incoming stream, surface both via a Streamlit dashboard, and
define retraining triggers. Depends on the cost-based metric module
(`app/evaluation.py`'s `cost_at_threshold`).

**Out of scope**: live serving integration (FastAPI), real label-delay
handling (ground truth is available immediately since the data is synthetic),
automated retraining execution (trigger only flags, does not retrain).

## Architecture

```
train.py (after fit) -> compute_reference_stats(X_train_selected) -> reports/reference_stats.json

monitor.py --window <batch> ->
  load reference_stats.json + model
  -> drift_report(reference_stats, incoming_df)          # PSI per feature
  -> rolling_metrics(y_true, y_pred, window=10_000)      # sliding precision/recall/cost
  -> check_retrain_trigger(drift_report, rolling_metrics, train_recall, cost_budget)
  -> reports/monitor_<run>.json

dashboard.py (Streamlit) -> reads latest reports/monitor_*.json -> renders 3 tabs
```

"Incoming" data is simulated by replaying the held-out test set in `step`
order — there is no live feed yet. Ground truth is available immediately
(synthetic data), so no label-delay simulation is needed.

## Modules

### `app/monitoring.py`

- `compute_reference_stats(X_train_selected, n_bins=10) -> dict` — for each
  feature in `config.SELECTED_FEATURES`, compute decile bin edges from the
  training data. Returns `{feature: {"bin_edges": [...], "bin_pcts": [...]}}`.
- `compute_psi(bin_edges, bin_pcts_train, incoming_values) -> float` — bins
  `incoming_values` using `bin_edges`, computes
  `PSI = sum((pct_incoming - pct_train) * ln(pct_incoming / pct_train))`
  over bins. Bins with zero count on either side get an epsilon floor to
  avoid `log(0)` / divide-by-zero.
- `drift_report(reference_stats, incoming_df) -> dict` — `compute_psi` per
  feature, returns `{feature_psis: {...}, max_psi: float, drifted_features:
  [feature names with PSI > 0.25]}`.
- `rolling_metrics(y_true, y_pred, fn_cost, fp_cost, window=10_000) -> list[dict]`
  — sliding window over row-ordered arrays (step size = window, non-overlapping
  chunks), each entry: `{window_start, window_end, precision, recall,
  cost_at_threshold(...)}`. Reuses `evaluation.cost_at_threshold`.
- `check_retrain_trigger(drift_report, rolling_metrics, train_recall, cost_budget) -> dict`
  — `{triggered: bool, reasons: [str]}`. Trigger if: any feature PSI > 0.25,
  OR latest rolling-window recall < `train_recall - 0.10`, OR latest
  rolling-window `total_cost` > `cost_budget`. Each true condition appends a
  human-readable reason string.

### `app/monitor.py` (CLI)

`uv run python -m app.monitor --window <path to parquet/csv>` —
loads `reports/reference_stats.json` and the trained model (joblib path via
`--model-path`), loads the window batch, runs `drift_report` +
`rolling_metrics` + `check_retrain_trigger`, writes
`reports/monitor_<timestamp-arg-or-count>.json`. No new config beyond
`config.MONITOR_WINDOW_SIZE = 10_000` and `config.COST_BUDGET`.

### `app/dashboard.py` (Streamlit)

`uv run streamlit run app/dashboard.py` — reads the most recent
`reports/monitor_*.json`. Three tabs:
- **Drift** — PSI-per-feature table + bar/heatmap, `drifted_features` called
  out.
- **Rolling Metrics** — line charts of precision, recall, total cost across
  windows.
- **Retraining Triggers** — status banner (triggered / not triggered) +
  bullet list of `reasons`.

### `app/train.py`

After fitting `best_estimator`, call
`monitoring.compute_reference_stats(X_train_selected)` and write it to
`reports/reference_stats.json` (mkdir `reports/` if absent).

### `app/config.py`

- `MONITOR_WINDOW_SIZE = 10_000`
- `COST_BUDGET` — placeholder threshold for `check_retrain_trigger`'s cost
  condition; set from a representative training-set rolling-window cost
  during implementation (not hardcoded here).

## Data flow / leakage guard

Reference stats are computed from `X_train_selected` only, at train time —
never touched again until the next retrain. Monitoring runs read-only against
that snapshot; no leakage risk since monitoring never feeds back into
training within this module's scope.

## Error handling

- Missing `reference_stats.json` when running `monitor.py` -> clear error,
  point to running `train.py` first.
- Zero-count bins in PSI calc -> epsilon floor, no divide-by-zero.
- Missing `reports/monitor_*.json` when launching dashboard -> clear message
  in the Streamlit UI, no crash.

## Testing

- Unit tests (pytest, synthetic fixtures):
  - `compute_psi` — known bin distributions, verify PSI value and the
    zero-count epsilon-floor path.
  - `drift_report` — drifted vs non-drifted feature classification.
  - `rolling_metrics` — precision/recall/cost math on tiny synthetic windows.
  - `check_retrain_trigger` — each of the three trigger conditions
    individually and combined.
- No test for `monitor.py` CLI or Streamlit dashboard end-to-end — verified
  by running once during implementation.
