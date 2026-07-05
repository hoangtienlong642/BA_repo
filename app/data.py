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
