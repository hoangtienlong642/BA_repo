import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def get_class_weight(y_train: pd.Series) -> dict:
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def get_scale_pos_weight(y_train: pd.Series) -> float:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    if pos == 0:
        raise ValueError("No positive samples in y_train; cannot compute scale_pos_weight.")
    return neg / pos
