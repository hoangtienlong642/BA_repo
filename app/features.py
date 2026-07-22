import numpy as np
import pandas as pd
from app.config import SELECTED_FEATURES


def extract_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes model features from raw transaction DataFrame.
    Required input columns:
    ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
    """
    df = raw_df.copy()

    # Cast numeric columns
    numeric_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 1. Hour of day
    df["hour_of_day"] = df["step"].astype(int) % 24

    # 2. Amount flags
    df["is_large_amount"] = (df["amount"] > 200000).astype(int)
    df["is_transfer_or_cashout"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)

    # 3. Origin balance features
    df["is_amount_equal_oldbalanceOrig"] = np.isclose(df["amount"], df["oldbalanceOrg"], atol=1e-2).astype(int)
    df["orig_balance_change"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["orig_balance_after_expected"] = df["oldbalanceOrg"] - df["amount"]
    df["orig_balance_change_abs_error"] = (df["newbalanceOrig"] - df["orig_balance_after_expected"]).abs()
    df["isOrigBalanceEnough"] = (df["oldbalanceOrg"] >= df["amount"]).astype(int)
    df["amount_to_orig_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    df["balance_drop_ratio"] = (df["oldbalanceOrg"] - df["newbalanceOrig"]) / (df["oldbalanceOrg"] + 1.0)
    df["isNewBalanceOrigZero"] = np.isclose(df["newbalanceOrig"], 0.0, atol=1e-2).astype(int)

    # 4. Destination balance features
    df["dest_balance_change"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["dest_balance_after_expected"] = df["oldbalanceDest"] + df["amount"]
    df["dest_balance_change_abs_error"] = (df["newbalanceDest"] - df["dest_balance_after_expected"]).abs()
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["amount_to_dest_ratio"] = df["amount"] / (df["oldbalanceDest"] + 1.0)
    df["amount_to_dest_cum_avg_ratio"] = df["amount_to_dest_ratio"]  # Simplified for real-time scoring

    # 5. One-hot transaction types
    df["type_PAYMENT"] = (df["type"] == "PAYMENT").astype(int)
    df["type_TRANSFER"] = (df["type"] == "TRANSFER").astype(int)
    df["type_CASH_OUT"] = (df["type"] == "CASH_OUT").astype(int)

    # Ensure all SELECTED_FEATURES are present and in correct order
    for feature in SELECTED_FEATURES:
        if feature not in df.columns:
            df[feature] = 0.0

    return df[SELECTED_FEATURES]


def extract_features_single(transaction: dict) -> pd.DataFrame:
    """Utility for single transaction dict"""
    raw_df = pd.DataFrame([transaction])
    return extract_features(raw_df)
