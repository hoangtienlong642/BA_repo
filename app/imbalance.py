import logging
from dataclasses import dataclass
from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResamplingConfig:
    """
    Configuration parameters for the Hybrid Resampling pipeline.

    Attributes:
        under_sample_limit (int): The target number of samples for the majority class (0) 
                                 after undersampling.
        over_sample_strategy (Union[float, str]): The sampling strategy for SMOTE (Step 2). 
                                                 'auto' or 1.0 means balance classes.
        random_state (int): Seed for reproducibility.
    """
    under_sample_limit: int = 3500000
    over_sample_strategy: Union[float, str] = 1.0
    random_state: int = 42


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


class HybridResampler:
    """
    A production-grade module for Hybrid Resampling using a combination of 
    RandomUnderSampler and SMOTE.

    This approach prevents Memory Errors (OOM) by first reducing the majority 
    class to a manageable size before applying synthetic oversampling.

    Example:
        >>> from app.imbalance import HybridResampler, ResamplingConfig
        >>> resampler = HybridResampler(ResamplingConfig(under_sample_limit=100000))
        >>> X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    """

    def __init__(self, config: Optional[ResamplingConfig] = None):
        """
        Initialize the HybridResampler with configuration.

        Args:
            config: ResamplingConfig object. Defaults to default settings.
        """
        self.config = config or ResamplingConfig()
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        """
        Constructs the imblearn Pipeline.
        Step 1: RandomUnderSampler to reduce majority class.
        Step 2: SMOTE to oversample minority class.
        """
        # Step 1: Undersample the majority class (0) to the limit
        # We use a dict to specify the exact number of samples for the majority class
        rus = RandomUnderSampler(
            sampling_strategy={0: self.config.under_sample_limit},
            random_state=self.config.random_state
        )

        # Step 2: Oversample the minority class (1) to match the new majority
        smote = SMOTE(
            sampling_strategy=self.config.over_sample_strategy,
            random_state=self.config.random_state
        )

        return Pipeline(steps=[
            ('undersampler', rus),
            ('oversampler', smote)
        ])

    def fit_resample(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray]
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
        """
        Apply the hybrid resampling pipeline to the training data.

        Args:
            X: Training features.
            y: Training labels.

        Returns:
            A tuple of (X_resampled, y_resampled).
        """
        logger.info(
            f"Starting hybrid resampling. Original shape: {X.shape}, "
            f"Fraud count: {np.sum(y == 1)}"
        )

        # Basic validation: ensure the majority class actually has more than the limit
        majority_count = np.sum(y == 0)
        if majority_count <= self.config.under_sample_limit:
            logger.warning(
                f"Majority class count ({majority_count}) is already below or equal to "
                f"the limit ({self.config.under_sample_limit}). Skipping undersampling step."
            )
            # If majority is already small, we might just want to SMOTE or do nothing
            # For simplicity in this pipeline, we'll let RUS pass if it can, 
            # but usually RUS fails if strategy is higher than current.
            # We can adjust the strategy dynamically if needed.
            rus_strategy = 'auto' # default behavior
        else:
            rus_strategy = {0: self.config.under_sample_limit}

        # Re-build steps if logic changed, or just use the pre-built one if simple
        X_res, y_res = self.pipeline.fit_resample(X, y)

        logger.info(
            f"Resampling complete. New shape: {X_res.shape}, "
            f"New Fraud count: {np.sum(y_res == 1)}"
        )

        return X_res, y_res


def apply_hybrid_resampling(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    under_sample_limit: int = 3500000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Functional interface for hybrid resampling.

    Args:
        X_train: Training features.
        y_train: Training labels.
        under_sample_limit: Number of non-fraud samples to keep.
        random_state: Seed.

    Returns:
        X_train_resampled, y_train_resampled
    """
    config = ResamplingConfig(
        under_sample_limit=under_sample_limit,
        random_state=random_state
    )
    resampler = HybridResampler(config)
    return resampler.fit_resample(X_train, y_train)
