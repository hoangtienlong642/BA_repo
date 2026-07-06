# 4. Drift & Performance Monitoring

## Scope

Offline batch monitoring: compare incoming transaction feature distributions
to training data (PSI), track rolling precision/recall/cost on a simulated
incoming stream, surface both via a Streamlit dashboard, and flag retraining
triggers. Depends on the cost-based metric module (`app/evaluation.py`'s
`cost_at_threshold`).

**Out of scope**: live serving integration, real label-delay handling
(ground truth is available immediately since PaySim is synthetic), automated
retraining execution (the trigger only flags — it does not retrain).

## Pipeline

1. `train.py` (after fitting) calls `monitoring.compute_reference_stats` on
   `X_train_selected` and writes `reports/reference_stats.json` — per-feature
   decile bin edges and percentages from the training distribution.
2. `uv run python -m app.monitor --window <batch> --model-path <model.joblib> --train-recall <float>`
   loads the reference stats + model, runs:
   - `monitoring.drift_report` — PSI per feature (`config.SELECTED_FEATURES`
     only), flags features with PSI > 0.25.
   - `monitoring.rolling_metrics` — precision/recall/cost over fixed
     `config.MONITOR_WINDOW_SIZE` (10,000-row) windows.
   - `monitoring.check_retrain_trigger` — fires if any feature PSI > 0.25, OR
     latest rolling recall drops more than 0.10 below the given
     `--train-recall`, OR latest rolling total cost exceeds
     `config.COST_BUDGET`.
   Writes `reports/monitor_<n>.json`.
3. `uv run streamlit run app/dashboard.py` reads the most recent
   `reports/monitor_*.json` and renders three tabs: Drift, Rolling Metrics,
   Retraining Triggers.

"Incoming" data for `--window` is, until a live feed exists, the held-out
test set (or a fresh batch) replayed in `step` order.

## Error handling

- Missing `reference_stats.json` -> `monitor.py` raises with a message
  pointing to `app.train`.
- Zero-count PSI bins -> epsilon floor, no divide-by-zero.
- No `monitor_*.json` reports -> dashboard shows an error message, no crash.

## Testing

Unit tests in `tests/test_monitoring.py` cover `compute_psi`,
`compute_reference_stats`, `drift_report`, `rolling_metrics`, and
`check_retrain_trigger` on synthetic fixtures. `monitor.py` and
`dashboard.py` are verified by running once during implementation — no
automated end-to-end test, consistent with `train.py`'s convention.
