# 2b. Feature Investigation, Selection, and Retuning

## Scope

Deep-dive investigation of the 44 candidate features (from `2_feature_engineer.ipynb`'s output), a data-driven feature selection, and a retuning pass for RF and XGBoost with aligned model complexity. Ends with a Vietnamese results/next-steps doc.

**Out of scope**: LightGBM retuning (may be covered by the "next steps" section of the Vietnamese doc, not executed here), FastAPI/Streamlit demo.

## 1. Feature investigation (`4_feature_investigation.ipynb`)

- Load `Synthetic_Financial_datasets_features.parquet`.
- Time-based split via `app.data.time_based_split` (reused, not reimplemented).
- Drop ID columns (`step`, `type`, `nameOrig`, `nameDest`) — same as `app/data.py`'s existing `ID_COLUMNS`.
- Fit `RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)` on all 44 candidate features (train split only).
- Extract `feature_importances_`, rank descending.
- Compute Pearson correlation matrix across the 44 features. For any pair with `|corr| > 0.9`, drop the lower-importance member of the pair.
- From the remaining ranked list, select the top features up to the importance "elbow" (visually justified in the notebook, targeting roughly 15-25 features — not a hardcoded count).
- Output: a table (feature name, importance, kept/dropped + reason) written to `docs/2b_feature_selection.md`, and the final Python list of selected feature names recorded there verbatim (for copying into `app/config.py`).

## 2. Wiring selection into `app/`

- `app/config.py`: add `SELECTED_FEATURES: list[str]` — the literal list from step 1's output.
- `app/data.py`: add `select_features(X: pd.DataFrame, features: list[str]) -> pd.DataFrame` — returns `X[features]`. Does not modify `time_based_split`, which keeps returning all candidate columns (so the investigation notebook and any future caller aren't forced into a fixed selection).
- `app/train.py`: after `time_based_split`, call `data.select_features(X_train, config.SELECTED_FEATURES)` / same for `X_test`, before fitting/evaluating. `amount` must remain available for `cost_curve` even if not in `SELECTED_FEATURES` — pull it from the unfiltered `X_test` (already the existing pattern in `train.py`).

## 3. Learning curves

- `app/learning_curve.py`: `plot_learning_curve(estimator, X_train, y_train, cv, scoring='average_precision') -> matplotlib.figure.Figure`. Uses `sklearn.model_selection.learning_curve` with `cv=TimeSeriesSplit(n_splits=5)` (consistent with the rest of the pipeline's no-shuffle stance), 5 train-size fractions from 0.2 to 1.0. Plots train score and CV score vs train-set size with shaded std-dev bands.
- `app/train.py`: add `--plot-learning-curve` flag. When set, after fitting the best estimator, call `plot_learning_curve`, save as PNG to a temp path, and log it to the active MLflow run via `mlflow.log_artifact`.

## 4. RF retuning

- `app/models/rf.py`'s `param_distributions()` already updated by the user: `n_estimators: [20, 40, 60, 80]` (reduced from `[100,200,300,500]` for tractable search time on 5M+ rows).
- Rerun `uv run python -m app.train --model rf --n-iter <N> --plot-learning-curve` against the selected feature set. Target: AUC-PR >= 0.75 on the held-out test set. Report the actual achieved number regardless of whether the target is hit.

## 5. XGBoost complexity alignment

- `app/models/xgboost_model.py`'s `param_distributions()`: change `n_estimators` to `[20, 40, 60, 80]` (matching RF's reduced range — boosted trees need far fewer estimators than a forest for comparable capacity, so no depth/learning-rate compensation needed). Leave `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` unchanged.
- Rerun `uv run python -m app.train --model xgboost --n-iter <N>` against the selected feature set. Same AUC-PR >= 0.75 target.

## 6. Vietnamese results doc (`docs/4_ket_qua_va_de_xuat.md`)

Written after steps 1-5 produce real numbers. Contents:
- Current best method (RF or XGBoost, whichever scores higher) and why.
- Evaluation metrics achieved: AUC-PR, confusion matrix, cost-based threshold and its rationale.
- 3-5 concrete next steps for further improvement, e.g.: revisit the `FP_COST` assumption (observed to drive the cost-optimal threshold to 0 in an earlier quick check — worth investigating whether the constant is realistic), try SMOTE/undersampling as an alternative to class-weighting, tune LightGBM, expand the hyperparameter search once compute allows.

## Error handling

- If `SELECTED_FEATURES` in config references a column not present in the loaded DataFrame (e.g. after a future re-run of feature engineering changes column names), `select_features` should raise a clear `KeyError`-derived message — no silent fallback to all columns.

## Testing

- `app/learning_curve.py`: no unit test (matplotlib figure generation, verified visually/manually per existing plan convention for non-automatable modules).
- `app/data.select_features`: unit test with a small synthetic DataFrame verifying it returns exactly the requested columns in the requested order.
