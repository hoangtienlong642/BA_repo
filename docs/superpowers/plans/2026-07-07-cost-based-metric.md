# Cost-Based Metric & Threshold Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace amount-weighted false-negative cost with a flat per-incident cost model ($500/missed fraud, $5/false alarm) in the cost curve and threshold selection.

**Architecture:** `app/evaluation.py`'s `cost_curve` drops its `amounts` parameter and computes cost purely from FN/FP counts × flat unit costs. A new `cost_at_threshold` helper exposes the same per-point math for reuse outside the sweep. `app/train.py` and `app/config.py` are updated to match.

**Tech Stack:** Python, numpy, pytest. No new dependencies.

## Global Constraints

- `FN_COST = 500.0` (missed fraud, per incident) — from spec.
- `FP_COST = 5.0` (false alarm, per incident) — from spec, replaces current `10.0`.
- No `amounts` parameter anywhere in the cost model going forward (spec: fully replace amount-weighting).

---

### Task 1: Flat-cost `cost_at_threshold` helper

**Files:**
- Modify: `app/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `cost_at_threshold(y_true, y_pred, fn_cost, fp_cost) -> dict` with keys `fn_count`, `fp_count`, `fn_cost_total`, `fp_cost_total`, `total_cost` (all values `int`/`float`, not numpy scalars).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluation.py`:

```python
from app.evaluation import cost_at_threshold


def test_cost_at_threshold_counts_and_totals():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 0, 0]  # 1 FP, 2 FN, 1 TP, 1 TN

    result = cost_at_threshold(y_true, y_pred, fn_cost=500.0, fp_cost=5.0)

    assert result == {
        "fn_count": 2,
        "fp_count": 1,
        "fn_cost_total": 1000.0,
        "fp_cost_total": 5.0,
        "total_cost": 1005.0,
    }


def test_cost_at_threshold_zero_errors():
    y_true = [0, 1]
    y_pred = [0, 1]

    result = cost_at_threshold(y_true, y_pred, fn_cost=500.0, fp_cost=5.0)

    assert result["fn_count"] == 0
    assert result["fp_count"] == 0
    assert result["total_cost"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation.py::test_cost_at_threshold_counts_and_totals -v`
Expected: FAIL with `ImportError: cannot import name 'cost_at_threshold'`

- [ ] **Step 3: Write minimal implementation**

Add to `app/evaluation.py` (near the top, after imports, before `evaluate`):

```python
def cost_at_threshold(y_true, y_pred, fn_cost: float, fp_cost: float) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fn_count = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp_count = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn_cost_total = fn_count * fn_cost
    fp_cost_total = fp_count * fp_cost

    return {
        "fn_count": fn_count,
        "fp_count": fp_count,
        "fn_cost_total": float(fn_cost_total),
        "fp_cost_total": float(fp_cost_total),
        "total_cost": float(fn_cost_total + fp_cost_total),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluation.py::test_cost_at_threshold_counts_and_totals tests/test_evaluation.py::test_cost_at_threshold_zero_errors -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/evaluation.py tests/test_evaluation.py
git commit -m "feat: add cost_at_threshold flat-cost helper"
```

---

### Task 2: Rewrite `cost_curve` to flat per-incident cost, drop `amounts`

**Files:**
- Modify: `app/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: nothing new from Task 1 (parallel logic, not a call site).
- Produces: `cost_curve(y_true, y_proba, fn_cost, fp_cost, thresholds=None, chunk_size=20) -> (list[dict], float)`. Each curve entry: `{"threshold": float, "fn_cost": float, "fp_cost": float, "total_cost": float}` (fn_cost/fp_cost are cost *totals* at that threshold — same key names as today, new meaning).

- [ ] **Step 1: Write the failing tests**

Replace the four existing `cost_curve` tests in `tests/test_evaluation.py` (the ones passing `amounts`) with:

```python
def test_cost_curve_finds_zero_cost_threshold():
    y_true = [1, 0]
    y_proba = [0.9, 0.1]
    fn_cost = 500.0
    fp_cost = 5.0

    curve, best_threshold = cost_curve(
        y_true, y_proba, fn_cost, fp_cost, thresholds=[0.0, 0.5, 1.0]
    )

    assert best_threshold == pytest.approx(0.5)
    best_row = next(r for r in curve if r["threshold"] == pytest.approx(0.5))
    assert best_row["total_cost"] == pytest.approx(0.0)

    zero_row = next(r for r in curve if r["threshold"] == pytest.approx(0.0))
    assert zero_row["fp_cost"] == pytest.approx(5.0)  # both flagged fraud -> 1 FP
    assert zero_row["fn_cost"] == pytest.approx(0.0)

    one_row = next(r for r in curve if r["threshold"] == pytest.approx(1.0))
    assert one_row["fn_cost"] == pytest.approx(500.0)  # nothing flagged -> 1 FN


def test_cost_curve_default_thresholds():
    y_true = [0, 1, 0, 1, 0, 1]
    y_proba = [0.1, 0.8, 0.3, 0.6, 0.2, 0.9]
    fn_cost = 500.0
    fp_cost = 5.0

    curve, best_threshold = cost_curve(y_true, y_proba, fn_cost, fp_cost)

    assert len(curve) == 101

    curve_thresholds = [row["threshold"] for row in curve]
    assert curve_thresholds == sorted(curve_thresholds)
    assert curve_thresholds[0] == pytest.approx(0.0)
    assert curve_thresholds[-1] == pytest.approx(1.0)

    assert isinstance(best_threshold, float)
    assert 0.0 <= best_threshold <= 1.0


def test_cost_curve_chunked_matches_unchunked_reference():
    rng = np.random.default_rng(42)
    n_samples = 500
    y_true = rng.integers(0, 2, size=n_samples)
    y_proba = rng.random(n_samples)
    fn_cost = 500.0
    fp_cost = 5.0
    thresholds = np.linspace(0.0, 1.0, 137)  # not a multiple of chunk_size

    curve_chunked, best_chunked = cost_curve(
        y_true, y_proba, fn_cost, fp_cost, thresholds=thresholds, chunk_size=10
    )
    curve_unchunked, best_unchunked = cost_curve(
        y_true, y_proba, fn_cost, fp_cost, thresholds=thresholds, chunk_size=len(thresholds)
    )

    assert best_chunked == pytest.approx(best_unchunked)
    for row_chunked, row_unchunked in zip(curve_chunked, curve_unchunked):
        assert row_chunked == pytest.approx(row_unchunked)
```

Remove the old `test_cost_curve_finds_zero_cost_threshold`, `test_cost_curve_default_thresholds`, `test_cost_curve_chunked_matches_unchunked_reference` definitions (the ones with `amounts` params) — these three names replace them.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation.py -k cost_curve -v`
Expected: FAIL — `cost_curve() missing 1 required positional argument` or similar signature mismatch (old implementation still takes `amounts`).

- [ ] **Step 3: Write minimal implementation**

Replace the entire `cost_curve` function body in `app/evaluation.py`:

```python
def cost_curve(y_true, y_proba, fn_cost: float, fp_cost: float, thresholds=None, chunk_size: int = 20):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    thresholds = np.asarray(thresholds)

    is_fraud = y_true == 1
    is_legit = y_true == 0

    n_thresholds = thresholds.shape[0]
    fn_cost_totals = np.empty(n_thresholds, dtype=float)
    fp_cost_totals = np.empty(n_thresholds, dtype=float)

    # Process thresholds in chunks so peak memory is bounded by
    # n_samples * chunk_size instead of n_samples * n_thresholds.
    for start in range(0, n_thresholds, chunk_size):
        end = min(start + chunk_size, n_thresholds)
        chunk_thresholds = thresholds[start:end]

        # (n_samples, chunk_len) boolean prediction matrix
        y_pred_chunk = y_proba[:, None] >= chunk_thresholds[None, :]

        fn_mask_chunk = is_fraud[:, None] & ~y_pred_chunk
        fp_mask_chunk = is_legit[:, None] & y_pred_chunk

        fn_cost_totals[start:end] = fn_mask_chunk.sum(axis=0) * fn_cost
        fp_cost_totals[start:end] = fp_mask_chunk.sum(axis=0) * fp_cost

    total_costs = fn_cost_totals + fp_cost_totals

    curve = [
        {
            "threshold": float(t),
            "fn_cost": float(fn_c),
            "fp_cost": float(fp_c),
            "total_cost": float(total_c),
        }
        for t, fn_c, fp_c, total_c in zip(thresholds, fn_cost_totals, fp_cost_totals, total_costs)
    ]

    best_idx = int(np.argmin(total_costs))
    return curve, curve[best_idx]["threshold"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: PASS (all tests in the file, including Task 1's)

- [ ] **Step 5: Commit**

```bash
git add app/evaluation.py tests/test_evaluation.py
git commit -m "feat: switch cost_curve to flat per-incident cost model"
```

---

### Task 3: Update `config.py` and `train.py` call sites

**Files:**
- Modify: `app/config.py`
- Modify: `app/train.py`

**Interfaces:**
- Consumes: `config.FN_COST`, `config.FP_COST` (new/changed constants); `evaluation.cost_curve(y_true, y_proba, fn_cost, fp_cost, ...)` from Task 2.

- [ ] **Step 1: Update config constants**

In `app/config.py`, replace:

```python
# Cost assumption: missed fraud costs the full transaction amount lost.
# False alarm costs a fixed friction cost (review + customer contact).
FP_COST = 10.0
```

with:

```python
# Cost assumption: flat per-incident costs (business-defined trade-off).
FN_COST = 500.0  # missed fraud
FP_COST = 5.0  # false alarm
```

- [ ] **Step 2: Update train.py call site**

In `app/train.py`, remove this line:

```python
    amounts_test = X_test["amount"].to_numpy()
```

And change:

```python
    _curve, best_threshold = evaluation.cost_curve(
        y_test.to_numpy(), y_proba, amounts_test, config.FP_COST
    )
```

to:

```python
    _curve, best_threshold = evaluation.cost_curve(
        y_test.to_numpy(), y_proba, config.FN_COST, config.FP_COST
    )
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: PASS (all tests)

- [ ] **Step 4: Sanity-check train.py imports/runs without error up to the cost_curve call**

Run: `uv run python -c "from app import config; print(config.FN_COST, config.FP_COST)"`
Expected: `500.0 5.0`

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/train.py
git commit -m "feat: wire flat per-incident cost constants into train.py"
```

---

### Task 4: Update docs

**Files:**
- Modify: `docs/3_model_training_evaluation.md`

- [ ] **Step 1: Update the cost-based metric section**

Replace this line:

```
`cost(FN) = transaction amount` (money lost), `cost(FP) = fixed constant` (customer friction / review cost, set in `config.py`). Threshold chosen by minimizing total cost on the cost curve; reported alongside default 0.5 threshold for comparison.
```

with:

```
`cost(FN) = $500/incident` (missed fraud), `cost(FP) = $5/incident` (false alarm / customer friction) — both flat per-incident constants set in `config.py` (`FN_COST`, `FP_COST`). Threshold chosen by minimizing total cost on the cost curve; reported alongside default 0.5 threshold for comparison.
```

Also update this line in the `evaluation.py` module bullet:

```
  - `cost_curve(y_true, y_proba, amounts, fp_cost)` — sweep thresholds, total cost = `sum(amount for FN) + fp_cost * count(FP)`. Returns curve + threshold minimizing total cost.
```

to:

```
  - `cost_curve(y_true, y_proba, fn_cost, fp_cost)` — sweep thresholds, total cost = `fn_cost * count(FN) + fp_cost * count(FP)`. Returns curve + threshold minimizing total cost.
  - `cost_at_threshold(y_true, y_pred, fn_cost, fp_cost)` — same cost math at a single fixed threshold (no sweep); reusable by the monitoring module.
```

- [ ] **Step 2: Commit**

```bash
git add docs/3_model_training_evaluation.md
git commit -m "docs: update cost-based metric section for flat per-incident model"
```
