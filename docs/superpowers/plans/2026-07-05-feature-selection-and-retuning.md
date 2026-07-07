# Feature Selection and Retuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Investigate the 44 candidate features, select a justified subset, wire it into the training pipeline, retune RF and XGBoost with aligned complexity and learning curves, and write a Vietnamese results/next-steps doc.

**Architecture:** A new investigation notebook produces a concrete feature list consumed by a small `app/` addition (`select_features`), which `train.py` applies before fitting. Retuning reruns the existing `train.py` CLI against the selected features with the already-lowered `n_estimators` search spaces.

**Tech Stack:** pandas, scikit-learn, matplotlib, MLflow, pytest. Package manager `uv`.

## Global Constraints

- Spec: `docs/2b_feature_selection_and_retuning.md` — follow exactly.
- Feature selection: RF importance ranking + correlation pruning at `|corr| > 0.9`, keeping the higher-importance member of each correlated pair.
- `time_based_split` stays unchanged (returns all candidate columns); a separate `select_features` helper applies the subset — no forced selection on other callers.
- Learning curve: `sklearn.model_selection.learning_curve` with `cv=TimeSeriesSplit(n_splits=5)`, `scoring="average_precision"`, train-size fractions `np.linspace(0.2, 1.0, 5)`.
- RF `param_distributions()` `n_estimators` already lowered to `[20, 40, 60, 80]` (done, committed in `cfb6d48`) — do not change further in this plan.
- XGBoost `param_distributions()` `n_estimators` must be changed to match: `[20, 40, 60, 80]`. Leave `max_depth`/`learning_rate`/`subsample`/`colsample_bytree` unchanged.
- Target AUC-PR >= 0.75 on the held-out test set for both RF and XGBoost — report the actual achieved number regardless of outcome.
- No silent fallback if a `SELECTED_FEATURES` column is missing from the loaded DataFrame — raise clearly.
- `app/learning_curve.py` has no automated unit test (matplotlib figure, verified manually) per spec's Testing section. `app/data.select_features` DOES get a unit test (TDD).

---

## File Structure

```
4_feature_investigation.ipynb      # new notebook, produces docs/2b_feature_selection.md + selected feature list
docs/2b_feature_selection.md       # new doc, written by Task 1's notebook run
app/config.py                       # add SELECTED_FEATURES
app/data.py                         # add select_features()
app/learning_curve.py               # new: plot_learning_curve()
app/mlflow_utils.py                 # extend log_run() to accept optional artifact paths
app/models/xgboost_model.py         # change n_estimators range
app/train.py                        # apply select_features(); add --plot-learning-curve flag
tests/test_data.py                  # add test for select_features
docs/4_ket_qua_va_de_xuat.md         # new Vietnamese results doc, written last
```

---

### Task 1: Feature investigation notebook and selection doc

**Files:**
- Create: `4_feature_investigation.ipynb`
- Create: `docs/2b_feature_selection.md`

**Interfaces:**
- Consumes: `Synthetic_Financial_datasets_features.parquet` (real data, already on disk), `app.data.time_based_split` (existing, from Task 2 of the prior plan).
- Produces: a concrete Python list of selected feature names, written verbatim into `docs/2b_feature_selection.md` — Task 2 copies this list into `app/config.py:SELECTED_FEATURES`.

- [ ] **Step 1: Create the notebook and load data**

Create `4_feature_investigation.ipynb` with a first code cell:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from app.data import time_based_split, load_features
from app.config import FEATURES_PATH, TEST_FRAC, RANDOM_SEED

df = load_features(FEATURES_PATH)
X_train, X_test, y_train, y_test = time_based_split(df, test_frac=TEST_FRAC)
print(f"Train: {len(X_train):,} rows, {X_train.shape[1]} candidate features")
```

Expected: prints train row count and confirms 44 candidate features (all columns minus `step`, `type`, `nameOrig`, `nameDest`, `isFraud`).

- [ ] **Step 2: Fit a quick RF and extract importances**

```python
rf_probe = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
rf_probe.fit(X_train, y_train)

importances = pd.Series(rf_probe.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(importances)
```

Expected: a ranked Series of all 44 features by importance. This step's runtime is real — expect roughly 1-2 minutes on the full 5M-row training set with 50 trees.

- [ ] **Step 3: Correlation pruning**

```python
corr_matrix = X_train.corr().abs()
to_drop = set()
cols = list(corr_matrix.columns)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        c1, c2 = cols[i], cols[j]
        if corr_matrix.loc[c1, c2] > 0.9:
            loser = c1 if importances[c1] < importances[c2] else c2
            to_drop.add(loser)

print(f"Dropping {len(to_drop)} correlated features: {sorted(to_drop)}")
remaining = [c for c in importances.index if c not in to_drop]
print(f"{len(remaining)} features remain after correlation pruning")
```

Expected: prints the dropped set and remaining count. Read the actual output — the exact features dropped depend on the real correlation values in this dataset.

- [ ] **Step 4: Pick the importance elbow and finalize the list**

```python
remaining_importances = importances[remaining]
print(remaining_importances)
```

Look at the printed ranked list. Visually identify where importance drops off sharply (the "elbow") — targeting roughly 15-25 features, but let the actual data decide, not a hardcoded count. Write the final list as a Python variable:

```python
SELECTED_FEATURES = remaining_importances.head(N).index.tolist()  # N = your chosen cutoff, justified by the elbow
print(SELECTED_FEATURES)
```

- [ ] **Step 5: Write the selection doc**

Create `docs/2b_feature_selection.md` with:
- A table: feature name, importance score, kept/dropped, reason (correlation-pruned vs below-elbow vs selected).
- The final `SELECTED_FEATURES` Python list, verbatim, in a code block — this exact list is what Task 2 will copy into `app/config.py`.
- One paragraph explaining where you drew the elbow cutoff and why.

- [ ] **Step 6: Commit**

```bash
git add 4_feature_investigation.ipynb docs/2b_feature_selection.md
git commit -m "docs: investigate and select features via RF importance + correlation pruning"
```

**Report to controller:** after this task, report the exact `SELECTED_FEATURES` list back — the controller needs it verbatim to brief Task 2.

---

### Task 2: Wire feature selection into `app/`

**Files:**
- Modify: `app/config.py`
- Modify: `app/data.py`
- Test: `tests/test_data.py`
- Modify: `app/train.py`

**Interfaces:**
- Consumes: the `SELECTED_FEATURES` list from Task 1's `docs/2b_feature_selection.md` (the controller will inject the exact list into this task's dispatch — it cannot be known until Task 1 runs).
- Produces: `app.config.SELECTED_FEATURES: list[str]`; `app.data.select_features(X: pd.DataFrame, features: list[str]) -> pd.DataFrame` for `train.py` and any future caller.

- [ ] **Step 1: Add SELECTED_FEATURES to config.py**

Add to `app/config.py` (the controller will provide the exact list from Task 1 — insert it verbatim, do not invent placeholder names):

```python
SELECTED_FEATURES = [
    # exact list from docs/2b_feature_selection.md
]
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_data.py`:

```python
from app.data import select_features


def test_select_features_returns_requested_columns_in_order():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]})
    result = select_features(df, features=["c", "a"])
    assert list(result.columns) == ["c", "a"]
    assert result["c"].tolist() == [5, 6]
    assert result["a"].tolist() == [1, 2]


def test_select_features_missing_column_raises():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(KeyError):
        select_features(df, features=["a", "does_not_exist"])
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_data.py -v -k select_features`
Expected: FAIL with `ImportError: cannot import name 'select_features'`

- [ ] **Step 4: Implement select_features**

Add to `app/data.py`:

```python
def select_features(X: pd.DataFrame, features: list) -> pd.DataFrame:
    return X[features]
```

(Indexing with a list of missing column names already raises `KeyError` in pandas — no extra error handling needed.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_data.py -v -k select_features`
Expected: 2 passed

- [ ] **Step 6: Apply selection in train.py**

In `app/train.py`, after the `time_based_split` call and before building the estimator, add:

```python
    X_train_selected = data.select_features(X_train, config.SELECTED_FEATURES)
    X_test_selected = data.select_features(X_test, config.SELECTED_FEATURES)
```

Replace all subsequent uses of `X_train`/`X_test` for fitting and evaluating with `X_train_selected`/`X_test_selected` — EXCEPT the `amounts_test = X_test["amount"].to_numpy()` line, which must keep reading from the original unfiltered `X_test` (so `cost_curve` still gets real amounts even if `amount` isn't in `SELECTED_FEATURES`).

The updated relevant section of `app/train.py` should read:

```python
    X_train, X_test, y_train, y_test = data.time_based_split(df, test_frac=config.TEST_FRAC)
    print(f"Train: {len(X_train):,} rows, Test: {len(X_test):,} rows.")

    X_train_selected = data.select_features(X_train, config.SELECTED_FEATURES)
    X_test_selected = data.select_features(X_test, config.SELECTED_FEATURES)

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
        X_train_selected,
        y_train,
        n_iter=args.n_iter,
        random_state=config.RANDOM_SEED,
    )
    print(f"Best CV AUC-PR: {best_cv_score:.4f}, params: {best_params}")

    metrics_default = evaluation.evaluate(best_estimator, X_test_selected, y_test, threshold=0.5)
    y_proba = best_estimator.predict_proba(X_test_selected)[:, 1]
    amounts_test = X_test["amount"].to_numpy()
    _curve, best_threshold = evaluation.cost_curve(
        y_test.to_numpy(), y_proba, amounts_test, config.FP_COST
    )
    metrics_best_threshold = evaluation.evaluate(best_estimator, X_test_selected, y_test, threshold=best_threshold)
```

- [ ] **Step 7: Run the full test suite to confirm no regressions**

Run: `uv run pytest tests/ -v`
Expected: all tests pass (18 previous + 2 new = 20)

- [ ] **Step 8: Commit**

```bash
git add app/config.py app/data.py app/train.py tests/test_data.py
git commit -m "feat: wire selected-feature subset into training pipeline"
```

---

### Task 3: Learning curve module and MLflow artifact logging

**Files:**
- Create: `app/learning_curve.py`
- Modify: `app/mlflow_utils.py`
- Modify: `app/train.py`

**Interfaces:**
- Consumes: a fitted-or-unfitted sklearn-API estimator, `X_train`/`y_train` (Task 2's selected-feature versions), a CV splitter.
- Produces: `plot_learning_curve(estimator, X_train, y_train, cv, scoring="average_precision") -> matplotlib.figure.Figure` for `train.py`; `mlflow_utils.log_run(..., artifact_paths=None)` extended signature for Task 4/5's real runs.

No automated test for `plot_learning_curve` per spec (matplotlib figure, verified manually below).

- [ ] **Step 1: Implement learning_curve.py**

Create `app/learning_curve.py`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, X_train, y_train, cv, scoring: str = "average_precision"):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_mean, "o-", label="Train score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(train_sizes, test_mean, "o-", label="CV score")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
    ax.set_xlabel("Training set size")
    ax.set_ylabel(scoring)
    ax.set_title("Learning Curve")
    ax.legend(loc="best")
    return fig
```

- [ ] **Step 2: Manual verification**

Run:
```bash
uv run python -c "
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from app.learning_curve import plot_learning_curve

rng = np.random.RandomState(0)
X = pd.DataFrame(rng.rand(200, 3), columns=['a', 'b', 'c'])
y = pd.Series((rng.rand(200) < 0.3).astype(int))

fig = plot_learning_curve(LogisticRegression(), X, y, cv=TimeSeriesSplit(n_splits=3))
fig.savefig('/tmp/lc_smoke_test.png')
print('saved')
"
ls -la /tmp/lc_smoke_test.png
rm /tmp/lc_smoke_test.png
```
Expected: prints `saved`, file exists with nonzero size, cleaned up after.

- [ ] **Step 3: Extend mlflow_utils.log_run for artifacts**

Modify `app/mlflow_utils.py`'s `log_run` signature and body:

```python
def log_run(model_name: str, params: dict, metrics: dict, best_threshold: float, model, artifact_paths=None) -> None:
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
        if artifact_paths:
            for path in artifact_paths:
                mlflow.log_artifact(path)
```

- [ ] **Step 4: Add --plot-learning-curve flag to train.py**

In `app/train.py`, add the import and argument:

```python
from app.learning_curve import plot_learning_curve
from sklearn.model_selection import TimeSeriesSplit
import tempfile
```

Add to the argument parser:
```python
    parser.add_argument("--plot-learning-curve", action="store_true")
```

Before the `mlflow_utils.log_run(...)` call, add:

```python
    artifact_paths = []
    if args.plot_learning_curve:
        print("Generating learning curve...")
        fig = plot_learning_curve(best_estimator, X_train_selected, y_train, cv=TimeSeriesSplit(n_splits=5))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig.savefig(f.name)
            artifact_paths.append(f.name)
```

Update the `log_run` call to pass `artifact_paths=artifact_paths`:

```python
    mlflow_utils.log_run(args.model, best_params, metrics_best_threshold, best_threshold, best_estimator, artifact_paths=artifact_paths)
```

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `uv run pytest tests/ -v`
Expected: 20 passed (no new tests added this task, confirming nothing broke)

- [ ] **Step 6: Commit**

```bash
git add app/learning_curve.py app/mlflow_utils.py app/train.py
git commit -m "feat: add learning curve plotting and MLflow artifact logging"
```

---

### Task 4: RF retuning run

**Files:** none created/modified — this task executes the pipeline and verifies results.

**Interfaces:**
- Consumes: everything from Tasks 1-3 — `app/train.py --model rf --n-iter N --plot-learning-curve`.

- [ ] **Step 1: Run RF retuning in the background**

RF fits at this scale take real time (empirically ~12.6s for 10 trees/10 features on 5M rows in an earlier quick check — scale accordingly for up to 80 trees and the selected feature count). Use a bounded `--n-iter` and run in the background so it doesn't block:

```bash
nohup uv run python -m app.train --model rf --n-iter 5 --plot-learning-curve > /tmp/rf_retune.log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 2: Poll until complete**

```bash
tail -f /tmp/rf_retune.log
```
(or periodically `tail -30 /tmp/rf_retune.log` and check `ps aux | grep app.train`). Expected completion output includes `Best CV AUC-PR: ...`, `Metrics @ threshold 0.5: ...`, `Cost-minimizing threshold: ...`.

- [ ] **Step 3: Verify MLflow logged the run**

```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
runs = mlflow.search_runs(experiment_ids=['0'])
print(runs[['tags.mlflow.runName', 'metrics.auc_pr', 'metrics.best_threshold']].tail(3))
"
```
Expected: a row for this RF run with a real `auc_pr` value.

- [ ] **Step 4: Report the result**

Record in your report: the achieved AUC-PR, whether it met the >= 0.75 target, the cost-minimizing threshold, and the confusion matrix at that threshold. No commit needed for this task (no code changed) — just report findings for use in Task 6's Vietnamese doc.

---

### Task 5: XGBoost complexity alignment and retuning run

**Files:**
- Modify: `app/models/xgboost_model.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: updated `param_distributions()` for `app/train.py --model xgboost`.

- [ ] **Step 1: Change n_estimators range**

In `app/models/xgboost_model.py`, change:
```python
        "n_estimators": [100, 200, 300, 500],
```
to:
```python
        "n_estimators": [20, 40, 60, 80],
```

- [ ] **Step 2: Run the existing model test to confirm no regression**

Run: `uv run pytest tests/test_models.py -v -k xgboost`
Expected: 2 passed (the existing test only checks the dict has the right keys, not specific values, so this should still pass)

- [ ] **Step 3: Commit the config change**

```bash
git add app/models/xgboost_model.py
git commit -m "feat: align XGBoost n_estimators range with RF for comparable complexity"
```

- [ ] **Step 4: Run XGBoost retuning in the background**

```bash
nohup uv run python -m app.train --model xgboost --n-iter 5 > /tmp/xgboost_retune.log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 5: Poll until complete**

Same pattern as Task 4 Step 2.

- [ ] **Step 6: Verify MLflow logged the run and report the result**

Same verification as Task 4 Step 3. Record: achieved AUC-PR, whether >= 0.75, cost-minimizing threshold, confusion matrix. No further commit needed.

---

### Task 6: Vietnamese results and next-steps doc

**Files:**
- Create: `docs/4_ket_qua_va_de_xuat.md`

**Interfaces:**
- Consumes: Task 4's RF results and Task 5's XGBoost results (the controller provides these numbers from the prior tasks' reports — they cannot be known until those tasks run).

- [ ] **Step 1: Write the doc**

Create `docs/4_ket_qua_va_de_xuat.md` in Vietnamese, covering (the controller will supply the actual numbers from Tasks 4-5 to insert):

```markdown
# Kết quả và Đề xuất

## Phương pháp tốt nhất hiện tại

[Tên model: Random Forest hoặc XGBoost — model có AUC-PR cao hơn giữa hai kết quả từ Task 4 và Task 5] được chọn vì [lý do dựa trên số liệu thực tế].

## Kết quả đánh giá

- AUC-PR: [số liệu thực tế]
- Ma trận nhầm lẫn (confusion matrix) tại ngưỡng tối ưu chi phí: [số liệu thực tế]
- Ngưỡng (threshold) tối ưu chi phí: [số liệu thực tế], với giả định chi phí bỏ lỡ gian lận = số tiền giao dịch, chi phí báo động giả = hằng số cố định (FP_COST).

## Đề xuất cải thiện thêm

1. Xem lại giả định FP_COST — quan sát ban đầu cho thấy ngưỡng tối ưu chi phí có thể bị đẩy về 0 (đánh dấu mọi giao dịch là gian lận) nếu FP_COST quá nhỏ so với số tiền gian lận thực tế; cần điều chỉnh hằng số này cho thực tế hơn.
2. Thử nghiệm SMOTE hoặc undersampling như một phương án thay thế cho class-weighting.
3. Tinh chỉnh (tune) LightGBM — hiện tại module đã có nhưng chưa được retuning trong vòng này.
4. Mở rộng không gian tìm kiếm siêu tham số (hyperparameter search space) khi có nhiều thời gian/tài nguyên tính toán hơn.
5. [any additional finding specific to what Task 4/5 actually revealed]
```

- [ ] **Step 2: Commit**

```bash
git add docs/4_ket_qua_va_de_xuat.md
git commit -m "docs: add Vietnamese results and next-steps report"
```

---

## Post-plan note

This plan covers `docs/2b_feature_selection_and_retuning.md` in full: feature investigation (Task 1), pipeline wiring (Task 2), learning curves + MLflow artifacts (Task 3), RF retuning (Task 4), XGBoost alignment + retuning (Task 5), Vietnamese report (Task 6). LightGBM retuning and the FP_COST re-assumption are explicitly deferred to the Vietnamese doc's next-steps section, not executed here.
