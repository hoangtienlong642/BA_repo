# Drift & Performance Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Offline batch module that computes PSI-based feature drift vs. training data, tracks rolling precision/recall/cost on a simulated incoming stream, exposes a Streamlit dashboard, and flags retraining triggers.

**Architecture:** `app/monitoring.py` holds all pure functions (PSI, drift report, rolling metrics, trigger check). `app/train.py` saves a reference-stats snapshot after fitting. `app/monitor.py` is a CLI that loads the snapshot + a model + an "incoming" batch, runs the monitoring functions, writes a JSON report. `app/dashboard.py` is a Streamlit app that reads the latest report and renders it.

**Tech Stack:** Python, numpy, pandas, scikit-learn (`precision_score`/`recall_score`), Streamlit (new dependency), matplotlib (chart rendering inside Streamlit via `st.pyplot`), pytest.

## Global Constraints

- PSI drift threshold: `> 0.25` = drifted feature (spec).
- Rolling window: fixed row count, `MONITOR_WINDOW_SIZE = 10_000` (spec).
- PSI computed only over `config.SELECTED_FEATURES` (spec), not all engineered features.
- Retrain trigger fires if: any feature PSI > 0.25, OR latest rolling recall < `train_recall - 0.10`, OR latest rolling `total_cost` > `cost_budget` (spec).
- Depends on `app.evaluation.cost_at_threshold(y_true, y_pred, fn_cost, fp_cost) -> dict` with keys `fn_count, fp_count, fn_cost_total, fp_cost_total, total_cost` — from the cost-based-metric plan (`docs/superpowers/plans/2026-07-07-cost-based-metric.md`). That plan must land first.
- No live label-delay simulation — ground truth is available immediately (synthetic data).
- Chart colors (dataviz skill, validated palette in `references/palette.md`): sequential magnitude = blue `#2a78d6`; categorical series precision=`#2a78d6` (slot 1), recall=`#1baf7a` (slot 2); status good=`#0ca30c`, critical=`#d03b3b`. No dual-axis charts — cost gets its own chart, not overlaid on precision/recall.

---

### Task 1: `compute_reference_stats` and `compute_psi`

**Files:**
- Create: `app/monitoring.py`
- Test: `tests/test_monitoring.py`

**Interfaces:**
- Produces: `compute_reference_stats(X_train_selected: pd.DataFrame, n_bins: int = 10) -> dict` → `{feature: {"bin_edges": list[float], "bin_pcts": list[float]}}`.
- Produces: `compute_psi(bin_edges: list[float], bin_pcts_train: list[float], incoming_values) -> float`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_monitoring.py`:

```python
import numpy as np
import pandas as pd
import pytest

from app.monitoring import compute_psi, compute_reference_stats


def test_compute_reference_stats_returns_edges_and_pcts_per_feature():
    df = pd.DataFrame({
        "a": np.arange(100, dtype=float),
        "b": np.arange(100, 200, dtype=float),
    })

    stats = compute_reference_stats(df, n_bins=10)

    assert set(stats.keys()) == {"a", "b"}
    for feature_stats in stats.values():
        assert len(feature_stats["bin_edges"]) == 11  # n_bins + 1
        assert len(feature_stats["bin_pcts"]) == 10
        assert sum(feature_stats["bin_pcts"]) == pytest.approx(1.0)


def test_compute_psi_zero_when_distributions_match():
    train_values = np.arange(100, dtype=float)
    stats = compute_reference_stats(pd.DataFrame({"a": train_values}), n_bins=10)["a"]

    psi = compute_psi(stats["bin_edges"], stats["bin_pcts"], train_values)

    assert psi == pytest.approx(0.0, abs=1e-6)


def test_compute_psi_positive_when_distribution_shifts():
    train_values = np.arange(100, dtype=float)
    stats = compute_reference_stats(pd.DataFrame({"a": train_values}), n_bins=10)["a"]
    shifted_values = np.arange(100, 200, dtype=float)  # all in the last bin's open range

    psi = compute_psi(stats["bin_edges"], stats["bin_pcts"], shifted_values)

    assert psi > 0.25


def test_compute_psi_handles_zero_count_bins_without_error():
    train_values = np.concatenate([np.zeros(50), np.arange(50, dtype=float) + 50])
    stats = compute_reference_stats(pd.DataFrame({"a": train_values}), n_bins=10)["a"]
    incoming_all_zero = np.zeros(50)

    psi = compute_psi(stats["bin_edges"], stats["bin_pcts"], incoming_all_zero)

    assert np.isfinite(psi)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_monitoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.monitoring'`

- [ ] **Step 3: Write minimal implementation**

Create `app/monitoring.py`:

```python
import numpy as np
import pandas as pd


def compute_reference_stats(X_train_selected: pd.DataFrame, n_bins: int = 10) -> dict:
    stats = {}
    for feature in X_train_selected.columns:
        values = X_train_selected[feature].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([values.min(), values.min() + 1e-9])
        counts, edges = np.histogram(values, bins=edges)
        pcts = counts / counts.sum()
        stats[feature] = {
            "bin_edges": edges.tolist(),
            "bin_pcts": pcts.tolist(),
        }
    return stats


def compute_psi(bin_edges, bin_pcts_train, incoming_values) -> float:
    edges = np.array(bin_edges, dtype=float)
    edges_open = edges.copy()
    edges_open[0] = -np.inf
    edges_open[-1] = np.inf

    incoming = np.asarray(incoming_values, dtype=float)
    counts, _ = np.histogram(incoming, bins=edges_open)
    total = counts.sum()
    pcts_incoming = counts / total if total > 0 else np.zeros_like(counts, dtype=float)

    pcts_train = np.array(bin_pcts_train, dtype=float)

    epsilon = 1e-6
    pcts_incoming = np.clip(pcts_incoming, epsilon, None)
    pcts_train = np.clip(pcts_train, epsilon, None)

    psi = np.sum((pcts_incoming - pcts_train) * np.log(pcts_incoming / pcts_train))
    return float(psi)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_monitoring.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app/monitoring.py tests/test_monitoring.py
git commit -m "feat: add PSI drift computation to monitoring module"
```

---

### Task 2: `drift_report`

**Files:**
- Modify: `app/monitoring.py`
- Test: `tests/test_monitoring.py`

**Interfaces:**
- Consumes: `compute_psi` from Task 1.
- Produces: `drift_report(reference_stats: dict, incoming_df: pd.DataFrame, psi_threshold: float = 0.25) -> dict` → `{"feature_psis": {feature: float}, "max_psi": float, "drifted_features": list[str]}`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_monitoring.py`:

```python
from app.monitoring import drift_report


def test_drift_report_flags_drifted_features():
    train_values_a = np.arange(100, dtype=float)
    train_values_b = np.arange(100, dtype=float)
    reference_stats = compute_reference_stats(
        pd.DataFrame({"a": train_values_a, "b": train_values_b}), n_bins=10
    )

    incoming_df = pd.DataFrame({
        "a": np.arange(100, 200, dtype=float),  # shifted -> drift
        "b": np.arange(100, dtype=float),  # unchanged -> no drift
    })

    report = drift_report(reference_stats, incoming_df)

    assert report["drifted_features"] == ["a"]
    assert report["max_psi"] == pytest.approx(report["feature_psis"]["a"])
    assert report["feature_psis"]["b"] == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_monitoring.py::test_drift_report_flags_drifted_features -v`
Expected: FAIL with `ImportError: cannot import name 'drift_report'`

- [ ] **Step 3: Write minimal implementation**

Add to `app/monitoring.py`:

```python
def drift_report(reference_stats: dict, incoming_df: pd.DataFrame, psi_threshold: float = 0.25) -> dict:
    feature_psis = {}
    for feature, stats in reference_stats.items():
        incoming_values = incoming_df[feature].to_numpy(dtype=float)
        feature_psis[feature] = compute_psi(stats["bin_edges"], stats["bin_pcts"], incoming_values)

    max_psi = max(feature_psis.values()) if feature_psis else 0.0
    drifted_features = [f for f, psi in feature_psis.items() if psi > psi_threshold]

    return {
        "feature_psis": feature_psis,
        "max_psi": max_psi,
        "drifted_features": drifted_features,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_monitoring.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app/monitoring.py tests/test_monitoring.py
git commit -m "feat: add drift_report aggregation over reference stats"
```

---

### Task 3: `rolling_metrics`

**Files:**
- Modify: `app/monitoring.py`
- Test: `tests/test_monitoring.py`

**Interfaces:**
- Consumes: `app.evaluation.cost_at_threshold(y_true, y_pred, fn_cost, fp_cost) -> dict` (from cost-based-metric plan, Task 1).
- Produces: `rolling_metrics(y_true, y_pred, fn_cost: float, fp_cost: float, window: int = 10_000) -> list[dict]`. Each entry: `{"window_start": int, "window_end": int, "precision": float, "recall": float, "fn_count": int, "fp_count": int, "fn_cost_total": float, "fp_cost_total": float, "total_cost": float}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_monitoring.py`:

```python
from app.monitoring import rolling_metrics


def test_rolling_metrics_splits_into_correct_windows():
    y_true = [0] * 5 + [1] * 5
    y_pred = [0] * 5 + [1] * 5  # all correct

    windows = rolling_metrics(y_true, y_pred, fn_cost=500.0, fp_cost=5.0, window=4)

    assert [w["window_start"] for w in windows] == [0, 4, 8]
    assert [w["window_end"] for w in windows] == [4, 8, 10]
    for w in windows:
        assert w["precision"] == pytest.approx(1.0) or w["precision"] == pytest.approx(0.0)


def test_rolling_metrics_computes_precision_recall_and_cost():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]  # 1 FP, 1 FN, 1 TP, 1 TN

    windows = rolling_metrics(y_true, y_pred, fn_cost=500.0, fp_cost=5.0, window=4)

    assert len(windows) == 1
    w = windows[0]
    assert w["precision"] == pytest.approx(0.5)
    assert w["recall"] == pytest.approx(0.5)
    assert w["fn_count"] == 1
    assert w["fp_count"] == 1
    assert w["total_cost"] == pytest.approx(505.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_monitoring.py::test_rolling_metrics_splits_into_correct_windows -v`
Expected: FAIL with `ImportError: cannot import name 'rolling_metrics'`

- [ ] **Step 3: Write minimal implementation**

Add to `app/monitoring.py`, alongside a new import line at the top:

```python
from sklearn.metrics import precision_score, recall_score

from app.evaluation import cost_at_threshold
```

```python
def rolling_metrics(y_true, y_pred, fn_cost: float, fp_cost: float, window: int = 10_000) -> list[dict]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    results = []
    for start in range(0, n, window):
        end = min(start + window, n)
        window_true = y_true[start:end]
        window_pred = y_pred[start:end]

        cost = cost_at_threshold(window_true, window_pred, fn_cost, fp_cost)

        results.append({
            "window_start": start,
            "window_end": end,
            "precision": float(precision_score(window_true, window_pred, zero_division=0)),
            "recall": float(recall_score(window_true, window_pred, zero_division=0)),
            **cost,
        })

    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_monitoring.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add app/monitoring.py tests/test_monitoring.py
git commit -m "feat: add rolling_metrics window computation"
```

---

### Task 4: `check_retrain_trigger`

**Files:**
- Modify: `app/monitoring.py`
- Test: `tests/test_monitoring.py`

**Interfaces:**
- Consumes: output shapes of `drift_report` (Task 2) and `rolling_metrics` (Task 3).
- Produces: `check_retrain_trigger(drift_report: dict, rolling_metrics: list, train_recall: float, cost_budget: float, psi_threshold: float = 0.25, recall_drop_threshold: float = 0.10) -> dict` → `{"triggered": bool, "reasons": list[str]}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_monitoring.py`:

```python
from app.monitoring import check_retrain_trigger


def test_check_retrain_trigger_fires_on_drift():
    drift = {"feature_psis": {"a": 0.5}, "max_psi": 0.5, "drifted_features": ["a"]}
    rolling = [{"recall": 0.9, "total_cost": 10.0}]

    result = check_retrain_trigger(drift, rolling, train_recall=0.9, cost_budget=1000.0)

    assert result["triggered"] is True
    assert any("PSI" in r for r in result["reasons"])


def test_check_retrain_trigger_fires_on_recall_drop():
    drift = {"feature_psis": {}, "max_psi": 0.0, "drifted_features": []}
    rolling = [{"recall": 0.5, "total_cost": 10.0}]

    result = check_retrain_trigger(drift, rolling, train_recall=0.9, cost_budget=1000.0)

    assert result["triggered"] is True
    assert any("recall" in r.lower() for r in result["reasons"])


def test_check_retrain_trigger_fires_on_cost_budget():
    drift = {"feature_psis": {}, "max_psi": 0.0, "drifted_features": []}
    rolling = [{"recall": 0.9, "total_cost": 5000.0}]

    result = check_retrain_trigger(drift, rolling, train_recall=0.9, cost_budget=1000.0)

    assert result["triggered"] is True
    assert any("cost" in r.lower() for r in result["reasons"])


def test_check_retrain_trigger_does_not_fire_when_all_clear():
    drift = {"feature_psis": {"a": 0.05}, "max_psi": 0.05, "drifted_features": []}
    rolling = [{"recall": 0.9, "total_cost": 10.0}]

    result = check_retrain_trigger(drift, rolling, train_recall=0.9, cost_budget=1000.0)

    assert result["triggered"] is False
    assert result["reasons"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_monitoring.py::test_check_retrain_trigger_fires_on_drift -v`
Expected: FAIL with `ImportError: cannot import name 'check_retrain_trigger'`

- [ ] **Step 3: Write minimal implementation**

Add to `app/monitoring.py`:

```python
def check_retrain_trigger(
    drift_report: dict,
    rolling_metrics: list,
    train_recall: float,
    cost_budget: float,
    psi_threshold: float = 0.25,
    recall_drop_threshold: float = 0.10,
) -> dict:
    reasons = []

    drifted = [f for f, psi in drift_report["feature_psis"].items() if psi > psi_threshold]
    if drifted:
        reasons.append(f"PSI drift on features: {', '.join(drifted)}")

    if rolling_metrics:
        latest = rolling_metrics[-1]
        if latest["recall"] < train_recall - recall_drop_threshold:
            reasons.append(
                f"Rolling recall {latest['recall']:.3f} below train recall "
                f"{train_recall:.3f} minus drop threshold {recall_drop_threshold:.3f}"
            )
        if latest["total_cost"] > cost_budget:
            reasons.append(
                f"Rolling window cost {latest['total_cost']:.2f} exceeds budget {cost_budget:.2f}"
            )

    return {"triggered": bool(reasons), "reasons": reasons}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_monitoring.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add app/monitoring.py tests/test_monitoring.py
git commit -m "feat: add check_retrain_trigger combining drift and performance signals"
```

---

### Task 5: `config.py` additions and `train.py` reference-stats snapshot

**Files:**
- Modify: `app/config.py`
- Modify: `app/train.py`

**Interfaces:**
- Consumes: `monitoring.compute_reference_stats` from Task 1.
- Produces: `reports/reference_stats.json` on disk after every `train.py` run.

- [ ] **Step 1: Add config constants**

In `app/config.py`, add after the `FN_COST`/`FP_COST` block (from the cost-based-metric plan):

```python
REPORTS_DIR = BASE_DIR / "reports"
REFERENCE_STATS_PATH = REPORTS_DIR / "reference_stats.json"
MONITOR_WINDOW_SIZE = 10_000
COST_BUDGET = 5_000.0  # placeholder; tune from a representative training-set rolling-window cost
```

- [ ] **Step 2: Wire reference-stats saving into train.py**

In `app/train.py`, add to the imports:

```python
import json

from app import monitoring
```

After the line computing `metrics_best_threshold` (right before the `artifact_paths = []` block), add:

```python
    print("Computing reference stats for drift monitoring...")
    reference_stats = monitoring.compute_reference_stats(X_train_selected)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REFERENCE_STATS_PATH, "w") as f:
        json.dump(reference_stats, f)
    print(f"Reference stats saved to {config.REFERENCE_STATS_PATH}")
```

- [ ] **Step 3: Verify config imports cleanly**

Run: `uv run python -c "from app import config; print(config.REPORTS_DIR, config.MONITOR_WINDOW_SIZE, config.COST_BUDGET)"`
Expected: prints the reports dir path, `10000`, `5000.0`

- [ ] **Step 4: Run full test suite (no regressions)**

Run: `uv run pytest tests/ -v`
Expected: PASS (all tests, train.py itself has no unit test per this project's convention)

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/train.py
git commit -m "feat: save reference stats snapshot after training"
```

---

### Task 6: `monitor.py` CLI

**Files:**
- Create: `app/monitor.py`

**Interfaces:**
- Consumes: `data.load_features`, `data.select_features` (existing, `app/data.py`); `monitoring.drift_report`, `monitoring.rolling_metrics`, `monitoring.check_retrain_trigger` (Tasks 2-4); `config.SELECTED_FEATURES`, `config.FN_COST`, `config.FP_COST`, `config.MONITOR_WINDOW_SIZE`, `config.COST_BUDGET`, `config.REFERENCE_STATS_PATH`, `config.REPORTS_DIR`.
- Produces: `reports/monitor_<n>.json` on disk, where `<n>` is an incrementing integer (count of existing `monitor_*.json` files).

- [ ] **Step 1: Write the CLI**

Create `app/monitor.py`:

```python
import argparse
import json

import joblib
import pandas as pd

from app import config, data, monitoring


def _next_report_path():
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(config.REPORTS_DIR.glob("monitor_*.json"))
    return config.REPORTS_DIR / f"monitor_{len(existing)}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run drift and performance monitoring on a batch.")
    parser.add_argument("--window", type=str, required=True, help="Path to the incoming batch (parquet or csv).")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the joblib model artifact.")
    parser.add_argument("--train-recall", type=float, required=True, help="Recall on the training-time test set, for trigger comparison.")
    args = parser.parse_args()

    if not config.REFERENCE_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Reference stats not found at {config.REFERENCE_STATS_PATH}. "
            "Run app.train first to generate it."
        )
    with open(config.REFERENCE_STATS_PATH) as f:
        reference_stats = json.load(f)

    model = joblib.load(args.model_path)

    if args.window.endswith(".csv"):
        incoming_df = pd.read_csv(args.window)
    else:
        incoming_df = pd.read_parquet(args.window)

    X_incoming = data.select_features(incoming_df, config.SELECTED_FEATURES)
    y_true = incoming_df[data.TARGET_COLUMN].to_numpy()
    y_pred = model.predict(X_incoming)

    drift = monitoring.drift_report(reference_stats, X_incoming)
    rolling = monitoring.rolling_metrics(
        y_true, y_pred, config.FN_COST, config.FP_COST, window=config.MONITOR_WINDOW_SIZE
    )
    trigger = monitoring.check_retrain_trigger(
        drift, rolling, train_recall=args.train_recall, cost_budget=config.COST_BUDGET
    )

    report = {
        "drift": drift,
        "rolling_metrics": rolling,
        "trigger": trigger,
    }

    report_path = _next_report_path()
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Monitoring report written to {report_path}")
    print(f"Max PSI: {drift['max_psi']:.3f}, drifted features: {drift['drifted_features']}")
    print(f"Retrain triggered: {trigger['triggered']} ({trigger['reasons']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports without error**

Run: `uv run python -c "from app import monitor; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add app/monitor.py
git commit -m "feat: add monitor.py CLI for offline drift and performance monitoring"
```

(No automated test for this task per the spec's testing section — CLI is verified by running once during implementation once a trained model + reference stats exist, same convention as `train.py`.)

---

### Task 7: Add Streamlit dependency and `dashboard.py`

**Files:**
- Modify: `pyproject.toml` (via `uv add`)
- Create: `app/dashboard.py`

**Interfaces:**
- Consumes: `reports/monitor_*.json` files written by `app/monitor.py` (Task 6).

- [ ] **Step 1: Add the dependency**

Run: `uv add streamlit`
Expected: `pyproject.toml` gains a `streamlit>=...` entry in `dependencies`.

- [ ] **Step 2: Write the dashboard**

Create `app/dashboard.py`. Colors below follow the project's validated dataviz palette: sequential PSI bars use blue `#2a78d6` with the drift-threshold line and drifted bars called out in critical red `#d03b3b`; the two rolling-metric series (precision, recall) use fixed categorical slots blue `#2a78d6` and aqua `#1baf7a`; cost gets its own chart (no dual-axis); the trigger banner uses status colors good `#0ca30c` / critical `#d03b3b` with an icon + text label, never color alone.

```python
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app import config

st.set_page_config(page_title="Fraud Model Monitoring", layout="wide")

report_files = sorted(
    glob.glob(str(config.REPORTS_DIR / "monitor_*.json")),
    key=os.path.getmtime,
)

if not report_files:
    st.error("No monitoring reports found. Run `uv run python -m app.monitor` first.")
    st.stop()

with open(report_files[-1]) as f:
    report = json.load(f)

drift = report["drift"]
rolling = report["rolling_metrics"]
trigger = report["trigger"]

tab_drift, tab_metrics, tab_trigger = st.tabs(["Drift", "Rolling Metrics", "Retraining Triggers"])

with tab_drift:
    st.subheader("Feature drift (PSI)")
    psi_df = pd.DataFrame(
        {"feature": list(drift["feature_psis"].keys()), "psi": list(drift["feature_psis"].values())}
    ).sort_values("psi", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(psi_df))))
    colors = ["#d03b3b" if f in drift["drifted_features"] else "#2a78d6" for f in psi_df["feature"]]
    ax.barh(psi_df["feature"], psi_df["psi"], color=colors)
    ax.axvline(0.25, color="#898781", linestyle="--", linewidth=1)
    ax.set_xlabel("PSI")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

    st.caption("Bars in red exceed the PSI > 0.25 drift threshold (dashed line).")
    st.dataframe(psi_df.sort_values("psi", ascending=False), use_container_width=True)

with tab_metrics:
    st.subheader("Rolling precision / recall")
    metrics_df = pd.DataFrame(rolling)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics_df.index, metrics_df["precision"], color="#2a78d6", linewidth=2, label="Precision")
    ax.plot(metrics_df.index, metrics_df["recall"], color="#1baf7a", linewidth=2, label="Recall")
    ax.set_xlabel("Window index")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

    st.subheader("Rolling total cost")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(metrics_df.index, metrics_df["total_cost"], color="#2a78d6", linewidth=2)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Total cost ($)")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

    st.dataframe(metrics_df, use_container_width=True)

with tab_trigger:
    if trigger["triggered"]:
        st.error("🔴 Retraining triggered")
    else:
        st.success("🟢 No retraining trigger")

    if trigger["reasons"]:
        st.write("Reasons:")
        for reason in trigger["reasons"]:
            st.write(f"- {reason}")
```

- [ ] **Step 3: Verify it imports without error**

Run: `uv run python -c "import app.dashboard" 2>&1 | tail -5`
Expected: either clean exit or a Streamlit "missing ScriptRunContext" warning only (harmless outside `streamlit run`) — no `ImportError`/`ModuleNotFoundError`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock app/dashboard.py
git commit -m "feat: add Streamlit monitoring dashboard"
```

---

### Task 8: Update docs

**Files:**
- Create: `docs/4_monitoring.md`

- [ ] **Step 1: Write the monitoring doc**

Create `docs/4_monitoring.md`:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/4_monitoring.md
git commit -m "docs: add drift and performance monitoring module doc"
```
