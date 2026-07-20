import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    y_test_arr = np.asarray(y_test)
    if len(np.unique(y_test_arr)) < 2:
        auc_pr = None
        roc_auc = None
    else:
        auc_pr = float(average_precision_score(y_test, y_proba))
        roc_auc = float(roc_auc_score(y_test, y_proba))

    return {
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_test, y_proba, labels=[0, 1])),
        "auc_pr": auc_pr,
        "roc_auc": roc_auc,
    }


def cost_curve(y_true, y_proba, amounts, fp_cost: float, thresholds=None, chunk_size: int = 20):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    amounts = np.asarray(amounts)

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

        fn_cost_totals[start:end] = (amounts[:, None] * fn_mask_chunk).sum(axis=0)
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
