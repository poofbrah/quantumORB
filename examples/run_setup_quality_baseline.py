from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.models import StrategyConfig
from features.pipeline import build_feature_frame
from labeling.future_path import ForwardSetupLabeler, LabelerConfig
from models import (
    SetupQualityDatasetConfig,
    build_setup_quality_dataset,
    build_walk_forward_splits,
    evaluate_baseline_fold,
    summarize_baseline_results,
)
from setups.orb import ORBConfig, ORBSetupDetector

INPUT_PATH = ROOT / "data" / "processed" / "nq_1min_2022_2025.parquet"
OUTPUT_DIR = ROOT / "outputs" / "setup_quality_baseline_nq_orb"
DATASET_PATH = OUTPUT_DIR / "setup_quality_dataset.parquet"
FOLD_RESULTS_PATH = OUTPUT_DIR / "fold_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "baseline_summary.json"
MODEL_NAMES = ["logistic_regression", "gradient_boosting"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(INPUT_PATH)
    featured = build_feature_frame(frame, opening_range_minutes=15)
    dataset = build_dataset(featured)
    dataset.to_parquet(DATASET_PATH, index=False)

    splits = build_walk_forward_splits(
        dataset,
        frequency="Q",
        train_periods=4,
        test_periods=1,
        step_periods=1,
        min_train_rows=25,
        min_test_rows=10,
    )
    if not splits:
        raise RuntimeError("No walk-forward splits were produced. Check the dataset coverage or split parameters.")

    fold_rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    for model_name in MODEL_NAMES:
        fold_results = []
        for split in splits:
            train_frame = dataset.iloc[split.train_indices].reset_index(drop=True)
            test_frame = dataset.iloc[split.test_indices].reset_index(drop=True)
            if train_frame["label"].nunique() < 2 or test_frame["label"].nunique() < 2:
                continue
            result = evaluate_baseline_fold(model_name, split.fold_id, train_frame, test_frame, threshold=0.55)
            fold_results.append(result)
            fold_rows.append(
                {
                    "model_name": result.model_name,
                    "fold_id": result.fold_id,
                    "train_rows": result.train_rows,
                    "test_rows": result.test_rows,
                    "roc_auc": result.roc_auc,
                    "log_loss": result.log_loss,
                    "brier_score": result.brier_score,
                    "threshold": result.threshold,
                    "selected_setups": result.selected_setups,
                    "selected_win_rate": result.selected_win_rate,
                    "selected_average_r": result.selected_average_r,
                    "selected_profit_factor": result.selected_profit_factor,
                    "train_period_start": split.train_period_start,
                    "train_period_end": split.train_period_end,
                    "test_period_start": split.test_period_start,
                    "test_period_end": split.test_period_end,
                }
            )
        summaries[model_name] = asdict(summarize_baseline_results(fold_results))

    pd.DataFrame(fold_rows).to_csv(FOLD_RESULTS_PATH, index=False)
    payload = {
        "rows": len(featured),
        "dataset_rows": len(dataset),
        "walk_forward_folds": len(splits),
        "models": summaries,
        "dataset_path": str(DATASET_PATH),
        "fold_results_path": str(FOLD_RESULTS_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"rows={payload['rows']}")
    print(f"dataset_rows={payload['dataset_rows']}")
    print(f"walk_forward_folds={payload['walk_forward_folds']}")
    for model_name in MODEL_NAMES:
        summary = summaries[model_name]
        print(f"model={model_name}")
        print(f"  folds_used={summary['folds_used']}")
        print(f"  average_roc_auc={format_metric(summary['average_roc_auc'])}")
        print(f"  average_log_loss={format_metric(summary['average_log_loss'])}")
        print(f"  average_brier_score={format_metric(summary['average_brier_score'])}")
        print(f"  selected_setups={summary['selected_setups']}")
        print(f"  selected_win_rate={format_metric(summary['selected_win_rate'])}")
        print(f"  selected_average_r={format_metric(summary['selected_average_r'])}")
        print(f"  selected_profit_factor={format_metric(summary['selected_profit_factor'])}")
    print(f"saved={OUTPUT_DIR}")


def build_dataset(featured: pd.DataFrame) -> pd.DataFrame:
    strategy_config = StrategyConfig(
        strategy_profile="nq_am_displacement_orb",
        instrument="NQ",
        latest_entry_time="11:30",
        target_rule="r_multiple",
        target_r_multiple=2.0,
        require_retest=False,
        runner_trail_rule=None,
        breakeven_enabled=False,
        breakeven_after_first_draw=True,
        partial_take_profit_enabled=False,
    )
    detector = ORBSetupDetector(ORBConfig.from_strategy_config(strategy_config))
    labeler = ForwardSetupLabeler(LabelerConfig(horizon_bars=20))
    return build_setup_quality_dataset(
        featured,
        detector,
        labeler,
        config=SetupQualityDatasetConfig(include_context=True, include_label_metadata=True),
    )


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
