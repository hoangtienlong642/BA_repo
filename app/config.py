from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES_PATH = BASE_DIR / "Synthetic_Financial_datasets_features.parquet"
MLRUNS_DIR = BASE_DIR / "mlruns"

TEST_FRAC = 0.2
RANDOM_SEED = 42

# Cost assumption: flat per-incident costs (business-defined trade-off).
FN_COST = 500.0  # missed fraud
FP_COST = 5.0  # false alarm

REPORTS_DIR = BASE_DIR / "reports"
REFERENCE_STATS_PATH = REPORTS_DIR / "reference_stats.json"
MONITOR_WINDOW_SIZE = 10_000
COST_BUDGET = 5_000.0  # placeholder; tune from a representative training-set rolling-window cost

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
