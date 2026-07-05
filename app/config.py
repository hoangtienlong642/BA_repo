from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES_PATH = BASE_DIR / "Synthetic_Financial_datasets_features.parquet"
MLRUNS_DIR = BASE_DIR / "mlruns"

TEST_FRAC = 0.2
RANDOM_SEED = 42

# Cost assumption: missed fraud costs the full transaction amount lost.
# False alarm costs a fixed friction cost (review + customer contact).
FP_COST = 10.0
