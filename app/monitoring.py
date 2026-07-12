import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from app.evaluation import cost_at_threshold


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


def rolling_metrics(y_true, y_pred, fn_cost: float, fp_cost: float, window: int = 10_000) -> list:
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
