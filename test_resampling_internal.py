import pandas as pd
import numpy as np
from app.imbalance import apply_hybrid_resampling, HybridResampler, ResamplingConfig

def test_hybrid_resampling():
    # Create dummy data: 10,000 samples, 100 fraud (1%)
    n_samples = 10000
    n_fraud = 100
    X = pd.DataFrame(np.random.rand(n_samples, 5), columns=[f'feat_{i}' for i in range(5)])
    y = pd.Series([0] * (n_samples - n_fraud) + [1] * n_fraud)
    
    # Target limit for majority: 500
    # Note: RUS will keep 500 non-fraud. SMOTE will then increase fraud (100) to match non-fraud (500).
    X_res, y_res = apply_hybrid_resampling(X, y, under_sample_limit=500)
    
    print(f"Original shape: {X.shape}, Fraud: {y.sum()}")
    print(f"Resampled shape: {X_res.shape}, Fraud: {y_res.sum()}")
    
    assert y_res.sum() == 500
    assert (y_res == 0).sum() == 500
    print("Test passed!")

if __name__ == "__main__":
    test_hybrid_resampling()
