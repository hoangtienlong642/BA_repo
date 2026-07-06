import numpy as np
import pandas as pd
import pytest

from app.monitoring import compute_psi, compute_reference_stats, drift_report


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
