# Cost-based metric & threshold selection — design

## Scope

Replace amount-weighted false-negative cost in `app/evaluation.py` with a flat
per-incident cost model: missed fraud costs $500/incident, false alarm costs
$5/incident. Threshold selection (`cost_curve`) picks the threshold that
minimizes total cost under this model.

**Out of scope**: drift monitoring, performance-over-time tracking, dashboard,
retraining triggers — separate spec.

## Config (`app/config.py`)

- `FN_COST = 500.0` — cost of one missed fraud incident.
- `FP_COST = 5.0` — cost of one false alarm (replaces current `10.0`).

## `app/evaluation.py`

- `cost_curve(y_true, y_proba, fn_cost, fp_cost, thresholds=None, chunk_size=20)`
  — drop the `amounts` parameter. For each threshold:
  `total_cost = fn_count(t) * fn_cost + fp_count(t) * fp_cost`.
  Returns the same curve-list + best-threshold shape as today, with
  `fn_cost`/`fp_cost` keys holding the totals (count × unit cost) instead of
  amount-derived sums.
- New `cost_at_threshold(y_true, y_pred, fn_cost, fp_cost) -> dict` — single-point
  cost calc: `{fn_count, fp_count, fn_cost_total, fp_cost_total, total_cost}`.
  Reusable outside the threshold sweep (e.g. by the future monitoring module to
  score a rolling window against the chosen threshold).

## `app/train.py`

- Drop `amounts_test = X_test["amount"].to_numpy()`.
- Call: `cost_curve(y_test.to_numpy(), y_proba, config.FN_COST, config.FP_COST)`.

## Tests (`tests/test_evaluation.py`)

- Rewrite the 4 existing `cost_curve` tests for the new signature (no
  `amounts`), verifying flat-cost math on tiny synthetic arrays.
- Add a test for `cost_at_threshold`.

## Docs

- Update `docs/3_model_training_evaluation.md`'s cost-based-metric section to
  describe flat per-incident costs instead of amount-weighted.

## Error handling

No change from existing conventions — no new failure modes introduced.
