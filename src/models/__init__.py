"""Model training and inference package."""

from .baselines import (
    BaselineFoldResult,
    BaselineRunSummary,
    fit_baseline_model,
    evaluate_baseline_fold,
    prepare_baseline_features,
    summarize_baseline_results,
)
from .setup_quality_dataset import (
    SetupQualityDatasetConfig,
    add_walk_forward_columns,
    build_setup_quality_dataset,
    flatten_labeled_setup,
    labeled_setups_to_frame,
)
from .walk_forward import WalkForwardSplit, build_walk_forward_splits, summarize_walk_forward_splits

__all__ = [
    "BaselineFoldResult",
    "BaselineRunSummary",
    "SetupQualityDatasetConfig",
    "WalkForwardSplit",
    "add_walk_forward_columns",
    "build_setup_quality_dataset",
    "build_walk_forward_splits",
    "evaluate_baseline_fold",
    "fit_baseline_model",
    "flatten_labeled_setup",
    "labeled_setups_to_frame",
    "prepare_baseline_features",
    "summarize_baseline_results",
    "summarize_walk_forward_splits",
]
