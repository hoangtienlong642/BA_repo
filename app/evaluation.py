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
