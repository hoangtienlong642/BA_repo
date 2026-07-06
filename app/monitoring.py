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
