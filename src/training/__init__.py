"""Training utilities for the optical digits CNN."""

from src.training.pipeline import (
    CNNTrainingConfig,
    EpochMetrics,
    EvaluationMetrics,
    TrainingResult,
    train_optical_digits_cnn,
)
from src.training.tracking import log_experiment_result

__all__ = [
    "CNNTrainingConfig",
    "EpochMetrics",
    "EvaluationMetrics",
    "TrainingResult",
    "log_experiment_result",
    "train_optical_digits_cnn",
]
