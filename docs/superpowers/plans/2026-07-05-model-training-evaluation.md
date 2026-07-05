# Model Training & Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `app/` training + evaluation pipeline for RF/XGBoost/LightGBM fraud models: time-based split, class-weight imbalance handling, hyperparameter tuning, cost-based evaluation, MLflow tracking.

**Architecture:** Flat script-driven pipeline. `app/train.py` CLI orchestrates: `data.load_features` -> `data.time_based_split` -> `imbalance` weight computation -> `tuning.time_series_search` (RandomizedSearchCV + TimeSeriesSplit) -> `evaluation.evaluate` / `evaluation.cost_curve` -> `mlflow_utils.log_run`.

**Tech Stack:** pandas, scikit-learn, xgboost, lightgbm, mlflow, pytest. Package manager `uv`.

## Global Constraints

- Spec: `docs/3_model_training_evaluation.md` — follow exactly.
- Test set touched only at final evaluation; all tuning CV happens inside train split via `TimeSeriesSplit`.
- `cost(FN) = transaction amount`, `cost(FP) = config.FP_COST` (fixed constant).
- Split: 80% train / 20% test, ordered by `step`, no shuffling.
- Class imbalance handled via class weighting only (`class_weight` for RF/LightGBM, `scale_pos_weight` for XGBoost) — no resampling.
- Hyperparameter tuning: `RandomizedSearchCV` scored on `average_precision`, CV = `TimeSeriesSplit`.
- MLflow: local file-based tracking store at `./mlruns` (gitignored).
- No unit tests for MLflow wrapper or full `train.py` end-to-end — verified manually per spec.

---

## File Structure

```
app/
  __init__.py
  config.py              # paths, cost assumptions, split fraction, seed
  data.py                 # load_features, time_based_split
  imbalance.py            # get_class_weight, get_scale_pos_weight
  models/
    __init__.py
    rf.py                 # build_estimator, param_distributions
    xgboost_model.py
    lightgbm_model.py
  tuning.py               # time_series_search
  evaluation.py           # evaluate, cost_curve
  mlflow_utils.py         # init_tracking, log_run
  train.py                # CLI entrypoint
tests/
  __init__.py
  test_data.py
  test_imbalance.py
  test_models.py
  test_evaluation.py
  test_tuning.py
```

---

### Task 1: Project setup — dependencies and package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `app/__init__.py`, `app/models/__init__.py`, `app/config.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `app.config.FEATURES_PATH`, `app.config.MLRUNS_DIR`, `app.config.TEST_FRAC`, `app.config.RANDOM_SEED`, `app.config.FP_COST` — all later tasks import these.

- [ ] **Step 1: Add dependencies via uv**

Run:
```bash
uv add scikit-learn xgboost lightgbm mlflow
uv add --dev pytest
```
Expected: `pyproject.toml` and `uv.lock` updated, no errors.

- [ ] **Step 2: Create package skeleton**

Create `app/__init__.py` (empty file):
```python
```

Create `app/models/__init__.py` (empty file):
```python
```

- [ ] **Step 3: Create config.py**

Create `app/config.py`:
```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES_PATH = BASE_DIR / "Synthetic_Financial_datasets_features.parquet"
MLRUNS_DIR = BASE_DIR / "mlruns"

TEST_FRAC = 0.2
RANDOM_SEED = 42

# Cost assumption: missed fraud costs the full transaction amount lost.
# False alarm costs a fixed friction cost (review + customer contact).
FP_COST = 10.0
```

- [ ] **Step 4: Update .gitignore**

Add to `.gitignore` (append):
```
mlruns/
app/results/
```

- [ ] **Step 5: Create tests package**

Create `tests/__init__.py` (empty file):
```python
```

- [ ] **Step 6: Verify imports work**

Run: `uv run python -c "from app import config; print(config.FP_COST)"`
Expected: `10.0`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock app/__init__.py app/models/__init__.py app/config.py .gitignore tests/__init__.py
git commit -m "chore: scaffold app/ package and add ML dependencies"
```

---

### Task 2: `data.py` — load and time-based split

**Files:**
- Create: `app/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing (pure pandas + `pathlib`).
- Produces: `load_features(path) -> pd.DataFrame`; `time_based_split(df, test_frac=0.2) -> (X_train, X_test, y_train, y_test)` (all `pd.DataFrame`/`pd.Series`) for Task 8 (`train.py`).

- [ ] **Step 1: Write failing tests**

Create `tests/test_data.py`:
```python
import pandas as pd
import pytest

from app.data import load_features, time_based_split


def test_load_features_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError, match="2_feature_engineer.ipynb"):
        load_features(missing_path)


def _make_df():
    return pd.DataFrame({
        "step": [3, 1, 2, 5, 4],
        "type": ["TRANSFER", "PAYMENT", "CASH_OUT", "TRANSFER", "PAYMENT"],
        "nameOrig": ["C1", "C2", "C3", "C4", "C5"],
        "nameDest": ["C6", "M1", "C7", "C8", "M2"],
        "amount": [100.0, 20.0, 30.0, 400.0, 50.0],
        "isFraud": [0, 0, 1, 1, 0],
    })


def test_time_based_split_orders_by_step_no_leakage():
    df = _make_df()
    X_train, X_test, y_train, y_test = time_based_split(df, test_frac=0.4)

    assert len(X_train) == 3
    assert len(X_test) == 2
    # train rows must all have step <= min step of test rows (time-based, no shuffling)
    train_steps = df.loc[df["amount"].isin(X_train["amount"]), "step"]
    test_steps = df.loc[df["amount"].isin(X_test["amount"]), "step"]
    assert train_steps.max() <= test_steps.min()


def test_time_based_split_drops_id_columns():
    df = _make_df()
    X_train, X_test, y_train, y_test = time_based_split(df, test_frac=0.4)

    for col in ["step", "type", "nameOrig", "nameDest", "isFraud"]:
        assert col not in X_train.columns
        assert col not in X_test.columns

    assert "amount" in X_train.columns
    assert y_train.name == "isFraud"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.data'`

- [ ] **Step 3: Implement data.py**

Create `app/data.py`:
```python
from pathlib import Path

import pandas as pd

ID_COLUMNS = ["step", "type", "nameOrig", "nameDest"]
TARGET_COLUMN = "isFraud"


def load_features(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found at {path}. "
            "Run 2_feature_engineer.ipynb first to generate it."
        )
    return pd.read_parquet(path)


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    df_sorted = df.sort_values("step").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_frac))

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    feature_cols = [
        c for c in df_sorted.columns if c not in ID_COLUMNS + [TARGET_COLUMN]
    ]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/data.py tests/test_data.py
git commit -m "feat: add feature loading and time-based train/test split"
```

---

### Task 3: `imbalance.py` — class weight computation

**Files:**
- Create: `app/imbalance.py`
- Test: `tests/test_imbalance.py`

**Interfaces:**
- Consumes: `pd.Series` of 0/1 labels.
- Produces: `get_class_weight(y_train) -> dict[int, float]` (for Task 4, 6 — RF/LightGBM); `get_scale_pos_weight(y_train) -> float` (for Task 5 — XGBoost).

- [ ] **Step 1: Write failing tests**

Create `tests/test_imbalance.py`:
```python
import pandas as pd
import pytest

from app.imbalance import get_class_weight, get_scale_pos_weight


def test_get_class_weight_known_counts():
    y_train = pd.Series([0] * 90 + [1] * 10)
    weights = get_class_weight(y_train)

    assert weights[0] == pytest.approx(100 / (2 * 90))
    assert weights[1] == pytest.approx(100 / (2 * 10))


def test_get_scale_pos_weight_known_counts():
    y_train = pd.Series([0] * 90 + [1] * 10)
    ratio = get_scale_pos_weight(y_train)
    assert ratio == pytest.approx(9.0)


def test_get_scale_pos_weight_no_positives_raises():
    y_train = pd.Series([0] * 10)
    with pytest.raises(ValueError, match="No positive samples"):
        get_scale_pos_weight(y_train)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_imbalance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.imbalance'`

- [ ] **Step 3: Implement imbalance.py**

Create `app/imbalance.py`:
```python
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def get_class_weight(y_train: pd.Series) -> dict:
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def get_scale_pos_weight(y_train: pd.Series) -> float:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    if pos == 0:
        raise ValueError("No positive samples in y_train; cannot compute scale_pos_weight.")
    return neg / pos
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_imbalance.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/imbalance.py tests/test_imbalance.py
git commit -m "feat: add class-weight imbalance handling"
```

---

### Task 4: `models/rf.py` — Random Forest estimator + search space

**Files:**
- Create: `app/models/rf.py`
- Test: `tests/test_models.py` (new file, shared across Tasks 4-6)

**Interfaces:**
- Consumes: `imbalance.get_class_weight` output (dict) as `class_weight` kwarg.
- Produces: `build_estimator(class_weight=None, random_state=42) -> RandomForestClassifier`; `param_distributions() -> dict` for Task 8 (`tuning.time_series_search`).

- [ ] **Step 1: Write failing test**

Create `tests/test_models.py`:
```python
from app.models import rf


def test_rf_build_estimator_applies_class_weight():
    estimator = rf.build_estimator(class_weight={0: 0.5, 1: 5.0}, random_state=42)
    assert estimator.class_weight == {0: 0.5, 1: 5.0}
    assert estimator.random_state == 42


def test_rf_param_distributions_is_nonempty_dict():
    params = rf.param_distributions()
    assert isinstance(params, dict)
    assert "n_estimators" in params
    assert "max_depth" in params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v -k rf`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.rf'`

- [ ] **Step 3: Implement models/rf.py**

Create `app/models/rf.py`:
```python
from sklearn.ensemble import RandomForestClassifier


def build_estimator(class_weight=None, random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )


def param_distributions() -> dict:
    return {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", 0.5],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v -k rf`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add app/models/rf.py tests/test_models.py
git commit -m "feat: add Random Forest estimator and search space"
```

---

### Task 5: `models/xgboost_model.py` — XGBoost estimator + search space

**Files:**
- Create: `app/models/xgboost_model.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: `imbalance.get_scale_pos_weight` output (float) as `scale_pos_weight` kwarg.
- Produces: `build_estimator(scale_pos_weight=1.0, random_state=42) -> XGBClassifier`; `param_distributions() -> dict` for Task 8.

- [ ] **Step 1: Write failing test**

Append to `tests/test_models.py`:
```python
from app.models import xgboost_model


def test_xgboost_build_estimator_applies_scale_pos_weight():
    estimator = xgboost_model.build_estimator(scale_pos_weight=9.0, random_state=42)
    assert estimator.get_params()["scale_pos_weight"] == 9.0
    assert estimator.get_params()["random_state"] == 42


def test_xgboost_param_distributions_is_nonempty_dict():
    params = xgboost_model.param_distributions()
    assert isinstance(params, dict)
    assert "max_depth" in params
    assert "learning_rate" in params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v -k xgboost`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.xgboost_model'`

- [ ] **Step 3: Implement models/xgboost_model.py**

Create `app/models/xgboost_model.py`:
```python
from xgboost import XGBClassifier


def build_estimator(scale_pos_weight: float = 1.0, random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="aucpr",
        n_jobs=-1,
    )


def param_distributions() -> dict:
    return {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v -k xgboost`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add app/models/xgboost_model.py tests/test_models.py
git commit -m "feat: add XGBoost estimator and search space"
```

---

### Task 6: `models/lightgbm_model.py` — LightGBM estimator + search space

**Files:**
- Create: `app/models/lightgbm_model.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: `imbalance.get_class_weight` output (dict) as `class_weight` kwarg.
- Produces: `build_estimator(class_weight=None, random_state=42) -> LGBMClassifier`; `param_distributions() -> dict` for Task 8.

- [ ] **Step 1: Write failing test**

Append to `tests/test_models.py`:
```python
from app.models import lightgbm_model


def test_lightgbm_build_estimator_applies_class_weight():
    estimator = lightgbm_model.build_estimator(class_weight={0: 0.5, 1: 5.0}, random_state=42)
    assert estimator.get_params()["class_weight"] == {0: 0.5, 1: 5.0}
    assert estimator.get_params()["random_state"] == 42


def test_lightgbm_param_distributions_is_nonempty_dict():
    params = lightgbm_model.param_distributions()
    assert isinstance(params, dict)
    assert "num_leaves" in params
    assert "learning_rate" in params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v -k lightgbm`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.lightgbm_model'`

- [ ] **Step 3: Implement models/lightgbm_model.py**

Create `app/models/lightgbm_model.py`:
```python
from lightgbm import LGBMClassifier


def build_estimator(class_weight=None, random_state: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def param_distributions() -> dict:
    return {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [-1, 4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.6, 0.8, 1.0],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v -k lightgbm`
Expected: 2 passed

- [ ] **Step 5: Run full test_models.py suite and commit**

Run: `uv run pytest tests/test_models.py -v`
Expected: 6 passed

```bash
git add app/models/lightgbm_model.py tests/test_models.py
git commit -m "feat: add LightGBM estimator and search space"
```

---

### Task 7: `evaluation.py` — metrics, confusion matrix, cost curve

**Files:**
- Create: `app/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: a fitted sklearn-API estimator (`.predict_proba`), `X_test`, `y_test` (Task 8 provides these from `data.time_based_split` + `tuning.time_series_search`).
- Produces: `evaluate(model, X_test, y_test, threshold=0.5) -> dict`; `cost_curve(y_true, y_proba, amounts, fp_cost, thresholds=None) -> (list[dict], float)` for Task 9 (`train.py`).

- [ ] **Step 1: Write failing tests**

Create `tests/test_evaluation.py`:
```python
import numpy as np
import pytest

from app.evaluation import cost_curve, evaluate


class _StubModel:
    """Returns fixed probabilities regardless of X, for deterministic tests."""

    def __init__(self, probabilities):
        self._probabilities = np.asarray(probabilities)

    def predict_proba(self, X):
        return np.column_stack([1 - self._probabilities, self._probabilities])


def test_evaluate_confusion_matrix_and_metrics():
    y_test = [0, 0, 1, 1]
    model = _StubModel([0.1, 0.4, 0.6, 0.9])  # threshold 0.5 -> preds [0, 0, 1, 1]

    result = evaluate(model, X_test=None, y_test=y_test, threshold=0.5)

    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["auc_pr"] == pytest.approx(1.0)


def test_cost_curve_finds_zero_cost_threshold():
    y_true = [1, 0]
    y_proba = [0.9, 0.1]
    amounts = [100, 0]
    fp_cost = 10

    curve, best_threshold = cost_curve(
        y_true, y_proba, amounts, fp_cost, thresholds=[0.0, 0.5, 1.0]
    )

    assert best_threshold == pytest.approx(0.5)
    best_row = next(r for r in curve if r["threshold"] == pytest.approx(0.5))
    assert best_row["total_cost"] == pytest.approx(0.0)

    zero_row = next(r for r in curve if r["threshold"] == pytest.approx(0.0))
    assert zero_row["fp_cost"] == pytest.approx(10.0)
    assert zero_row["fn_cost"] == pytest.approx(0.0)

    one_row = next(r for r in curve if r["threshold"] == pytest.approx(1.0))
    assert one_row["fn_cost"] == pytest.approx(100.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.evaluation'`

- [ ] **Step 3: Implement evaluation.py**

Create `app/evaluation.py`:
```python
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_pr": float(average_precision_score(y_test, y_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def cost_curve(y_true, y_proba, amounts, fp_cost: float, thresholds=None):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    amounts = np.asarray(amounts)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    curve = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fn_mask = (y_true == 1) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)

        fn_cost_total = float(amounts[fn_mask].sum())
        fp_cost_total = float(fp_mask.sum() * fp_cost)

        curve.append({
            "threshold": float(t),
            "fn_cost": fn_cost_total,
            "fp_cost": fp_cost_total,
            "total_cost": fn_cost_total + fp_cost_total,
        })

    best = min(curve, key=lambda r: r["total_cost"])
    return curve, best["threshold"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add app/evaluation.py tests/test_evaluation.py
git commit -m "feat: add confusion matrix, AUC-PR, and cost-based threshold evaluation"
```

---

### Task 8: `tuning.py` — RandomizedSearchCV with TimeSeriesSplit

**Files:**
- Create: `app/tuning.py`
- Test: `tests/test_tuning.py`

**Interfaces:**
- Consumes: an unfitted sklearn-API estimator (from Task 4/5/6 `build_estimator()`), a `param_distributions()` dict, `X_train`/`y_train` (from Task 2 `time_based_split`).
- Produces: `time_series_search(estimator, param_distributions, X_train, y_train, n_iter=25, n_splits=5, random_state=42) -> (best_estimator, best_params, best_score)` for Task 9 (`train.py`).

- [ ] **Step 1: Write failing test**

Create `tests/test_tuning.py`:
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.tuning import time_series_search


def test_time_series_search_returns_fitted_best_estimator():
    rng = np.random.RandomState(0)
    X_train = pd.DataFrame(rng.rand(60, 3), columns=["a", "b", "c"])
    y_train = pd.Series([0] * 54 + [1] * 6)  # imbalanced, deterministic order

    estimator = LogisticRegression()
    param_distributions = {"C": [0.1, 1.0, 10.0]}

    best_estimator, best_params, best_score = time_series_search(
        estimator, param_distributions, X_train, y_train, n_iter=3, n_splits=3, random_state=0
    )

    assert hasattr(best_estimator, "predict_proba")
    assert "C" in best_params
    assert isinstance(best_score, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tuning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.tuning'`

- [ ] **Step 3: Implement tuning.py**

Create `app/tuning.py`:
```python
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit


def time_series_search(
    estimator,
    param_distributions: dict,
    X_train,
    y_train,
    n_iter: int = 25,
    n_splits: int = 5,
    random_state: int = 42,
):
    cv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tuning.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add app/tuning.py tests/test_tuning.py
git commit -m "feat: add time-series-aware hyperparameter search"
```

---

### Task 9: `mlflow_utils.py` — experiment tracking

**Files:**
- Create: `app/mlflow_utils.py`

**Interfaces:**
- Consumes: `config.MLRUNS_DIR` (Task 1); `metrics` dict (Task 7 `evaluate` output, flattened); a fitted model (Task 8 output).
- Produces: `init_tracking(tracking_dir)`; `log_run(model_name, params, metrics, best_threshold, model)` for Task 10 (`train.py`).

No automated test per spec (`docs/3_model_training_evaluation.md`, Testing section) — verified via manual smoke run in this task and end-to-end in Task 10.

- [ ] **Step 1: Implement mlflow_utils.py**

Create `app/mlflow_utils.py`:
```python
import mlflow
import mlflow.sklearn


def init_tracking(tracking_dir) -> None:
    mlflow.set_tracking_uri(f"file:{tracking_dir}")


def log_run(model_name: str, params: dict, metrics: dict, best_threshold: float, model) -> None:
    flat_metrics = {
        k: v for k, v in metrics.items() if isinstance(v, (int, float))
    }
    if "confusion_matrix" in metrics:
        for k, v in metrics["confusion_matrix"].items():
            flat_metrics[f"confusion_matrix_{k}"] = v

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(flat_metrics)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.sklearn.log_model(model, artifact_path="model")
```

- [ ] **Step 2: Manual smoke test**

Run:
```bash
uv run python -c "
from sklearn.linear_model import LogisticRegression
import pandas as pd
from app import mlflow_utils

model = LogisticRegression().fit(pd.DataFrame({'a': [0, 1, 0, 1]}), [0, 1, 0, 1])
mlflow_utils.init_tracking('./mlruns_smoke_test')
mlflow_utils.log_run(
    'smoke_test',
    {'C': 1.0},
    {'precision': 0.9, 'confusion_matrix': {'tn': 1, 'fp': 0, 'fn': 0, 'tp': 1}},
    0.5,
    model,
)
print('OK')
"
```
Expected: prints `OK`, creates `./mlruns_smoke_test/` directory with a run.

- [ ] **Step 3: Clean up smoke test artifacts**

Run: `rm -rf mlruns_smoke_test`

- [ ] **Step 4: Commit**

```bash
git add app/mlflow_utils.py
git commit -m "feat: add MLflow experiment tracking wrapper"
```

---

### Task 10: `train.py` — CLI entrypoint

**Files:**
- Create: `app/train.py`

**Interfaces:**
- Consumes: `data.load_features`, `data.time_based_split` (Task 2); `imbalance.get_class_weight`, `imbalance.get_scale_pos_weight` (Task 3); `models.rf/xgboost_model/lightgbm_model.build_estimator/param_distributions` (Tasks 4-6); `tuning.time_series_search` (Task 8); `evaluation.evaluate`, `evaluation.cost_curve` (Task 7); `mlflow_utils.init_tracking`, `mlflow_utils.log_run` (Task 9); `config.*` (Task 1).
- Produces: CLI `uv run python -m app.train --model {rf,xgboost,lightgbm} --n-iter N`. No return value consumed elsewhere — this is the pipeline's terminal entrypoint.

No automated test per spec — verified by running once per model against real data (Step 3).

- [ ] **Step 1: Implement train.py**

Create `app/train.py`:
```python
import argparse

from app import config, data, evaluation, imbalance, mlflow_utils, tuning
from app.models import lightgbm_model, rf, xgboost_model

MODEL_REGISTRY = {
    "rf": rf,
    "xgboost": xgboost_model,
    "lightgbm": lightgbm_model,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a fraud detection model.")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True)
    parser.add_argument("--n-iter", type=int, default=25)
    args = parser.parse_args()

    print(f"Loading features from {config.FEATURES_PATH}...")
    df = data.load_features(config.FEATURES_PATH)
    X_train, X_test, y_train, y_test = data.time_based_split(df, test_frac=config.TEST_FRAC)
    print(f"Train: {len(X_train):,} rows, Test: {len(X_test):,} rows.")

    model_module = MODEL_REGISTRY[args.model]
    if args.model == "xgboost":
        weight = imbalance.get_scale_pos_weight(y_train)
        estimator = model_module.build_estimator(scale_pos_weight=weight, random_state=config.RANDOM_SEED)
    else:
        weight = imbalance.get_class_weight(y_train)
        estimator = model_module.build_estimator(class_weight=weight, random_state=config.RANDOM_SEED)

    print(f"Running hyperparameter search ({args.n_iter} iterations)...")
    best_estimator, best_params, best_cv_score = tuning.time_series_search(
        estimator,
        model_module.param_distributions(),
        X_train,
        y_train,
        n_iter=args.n_iter,
        random_state=config.RANDOM_SEED,
    )
    print(f"Best CV AUC-PR: {best_cv_score:.4f}, params: {best_params}")

    metrics_default = evaluation.evaluate(best_estimator, X_test, y_test, threshold=0.5)
    y_proba = best_estimator.predict_proba(X_test)[:, 1]
    amounts_test = X_test["amount"].to_numpy()
    _curve, best_threshold = evaluation.cost_curve(
        y_test.to_numpy(), y_proba, amounts_test, config.FP_COST
    )
    metrics_best_threshold = evaluation.evaluate(best_estimator, X_test, y_test, threshold=best_threshold)

    mlflow_utils.init_tracking(config.MLRUNS_DIR)
    mlflow_utils.log_run(args.model, best_params, metrics_best_threshold, best_threshold, best_estimator)

    print(f"\n=== {args.model} ===")
    print(f"Metrics @ threshold 0.5: {metrics_default}")
    print(f"Cost-minimizing threshold: {best_threshold:.3f}")
    print(f"Metrics @ cost-minimizing threshold: {metrics_best_threshold}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Check whether feature parquet exists**

Run: `ls Synthetic_Financial_datasets_features.parquet 2>/dev/null && echo EXISTS || echo MISSING`

- [ ] **Step 3a: If EXISTS — run end-to-end for all three models**

Run:
```bash
uv run python -m app.train --model rf --n-iter 10
uv run python -m app.train --model xgboost --n-iter 10
uv run python -m app.train --model lightgbm --n-iter 10
```
Expected: each prints train/test row counts, best CV AUC-PR, and metrics at both thresholds, with no exceptions. Use `--n-iter 10` for a fast manual check; increase later for real experiments.

- [ ] **Step 3b: If MISSING — smoke test with a synthetic parquet fixture**

Run:
```bash
uv run python -c "
import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
n = 2000
df = pd.DataFrame({
    'step': np.sort(rng.randint(1, 200, n)),
    'type': rng.choice(['TRANSFER', 'CASH_OUT', 'PAYMENT'], n),
    'nameOrig': [f'C{i}' for i in range(n)],
    'nameDest': [f'C{i+1}' for i in range(n)],
    'amount': rng.exponential(200, n),
    'isFraud': (rng.rand(n) < 0.01).astype(int),
})
for col in ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER',
            'is_merchant_dest', 'hour_of_day', 'day_of_month', 'day_of_week']:
    df[col] = rng.rand(n)
df.to_parquet('Synthetic_Financial_datasets_features.parquet', index=False)
print('fixture written')
"
uv run python -m app.train --model rf --n-iter 5
rm Synthetic_Financial_datasets_features.parquet
```
Expected: `train.py` completes without exceptions against the synthetic fixture; fixture removed after. Note: real full-scale runs against actual PaySim data are the user's responsibility once `0_get_data.ipynb`/`1_clean_data.ipynb`/`2_feature_engineer.ipynb` have been run — this smoke test only verifies the pipeline's wiring.

- [ ] **Step 4: Commit**

```bash
git add app/train.py
git commit -m "feat: add train.py CLI entrypoint for model training and evaluation"
```

---

## Post-plan note

This plan covers `docs/3_model_training_evaluation.md` in full: split (Task 2), imbalance (Task 3), all three models (Tasks 4-6), evaluation + cost metric (Task 7), tuning (Task 8), tracking (Task 9), orchestration (Task 10). FastAPI serving and the Streamlit/HTML demo are out of scope here and need their own spec once trained models exist (per brainstorming decision).
