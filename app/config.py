from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES_PATH = BASE_DIR / "Synthetic_Financial_datasets_features.parquet"
MLRUNS_DIR = BASE_DIR / "mlruns"

TEST_FRAC = 0.2
RANDOM_SEED = 42

# Cost assumption: missed fraud costs the full transaction amount lost.
# False alarm costs a fixed friction cost (review + customer contact).
FP_COST = 10.0

SELECTED_FEATURES = [
    'is_amount_equal_oldbalanceOrig',
    'orig_balance_change_abs_error',
    'isOrigBalanceEnough',
    'amount_to_orig_ratio',
    'balance_drop_ratio',
    'isNewBalanceOrigZero',
    'orig_balance_after_expected',
    'orig_balance_change',
    'is_transfer_or_cashout',
    'amount_to_dest_cum_avg_ratio',
    'amount_to_dest_ratio',
    'type_PAYMENT',
    'errorBalanceDest',
    'type_TRANSFER',
    'type_CASH_OUT',
    'is_large_amount',
    'hour_of_day',
    'dest_balance_change',
    'dest_balance_after_expected',
    'dest_balance_change_abs_error',
]
