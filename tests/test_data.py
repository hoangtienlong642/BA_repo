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
