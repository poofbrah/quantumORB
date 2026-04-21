"""Model training and inference package."""

from .baselines import (
    BaselineFoldResult,
    BaselineRunSummary,
    fit_baseline_model,
    evaluate_baseline_fold,
    prepare_baseline_features,
    summarize_baseline_results,
)
from .lstm import (
    LSTMConfig,
    LSTMFoldResult,
    LSTMRunSummary,
    evaluate_lstm_walk_forward,
    fit_lstm_sequence_model,
    predict_lstm_probabilities,
    summarize_lstm_results,
)
from .setup_quality_dataset import (
    SetupQualityDatasetConfig,
    add_walk_forward_columns,
    build_setup_quality_dataset,
    flatten_labeled_setup,
    labeled_setups_to_frame,
)
from .setup_quality_sequences import (
    SetupSequenceDataset,
    SetupSequenceDatasetConfig,
    build_setup_quality_sequence_dataset,
    save_setup_sequence_dataset,
)
from .walk_forward import WalkForwardSplit, build_walk_forward_splits, summarize_walk_forward_splits

__all__ = [
    "BaselineFoldResult",
    "BaselineRunSummary",
    "LSTMConfig",
    "LSTMFoldResult",
    "LSTMRunSummary",
    "SetupSequenceDataset",
    "SetupSequenceDatasetConfig",
    "SetupQualityDatasetConfig",
    "WalkForwardSplit",
    "add_walk_forward_columns",
    "build_setup_quality_dataset",
    "build_setup_quality_sequence_dataset",
    "build_walk_forward_splits",
    "evaluate_baseline_fold",
    "evaluate_lstm_walk_forward",
    "fit_baseline_model",
    "fit_lstm_sequence_model",
    "flatten_labeled_setup",
    "labeled_setups_to_frame",
    "predict_lstm_probabilities",
    "prepare_baseline_features",
    "summarize_lstm_results",
    "save_setup_sequence_dataset",
    "summarize_baseline_results",
    "summarize_walk_forward_splits",
]
