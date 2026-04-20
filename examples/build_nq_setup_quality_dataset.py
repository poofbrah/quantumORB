from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.models import StrategyConfig
from features.pipeline import build_feature_frame
from labeling.future_path import ForwardSetupLabeler, LabelerConfig
from models import SetupQualityDatasetConfig, build_setup_quality_dataset
from setups.orb import ORBConfig, ORBSetupDetector

INPUT_PATH = ROOT / "data" / "processed" / "nq_1min_2022_2025.parquet"
OUTPUT_DIR = ROOT / "outputs" / "setup_quality_dataset_nq_orb"
FEATURED_PATH = OUTPUT_DIR / "featured_nq_1min_2022_2025.parquet"
DATASET_PATH = OUTPUT_DIR / "setup_quality_dataset.parquet"
SUMMARY_PATH = OUTPUT_DIR / "dataset_summary.json"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(INPUT_PATH)
    featured = build_feature_frame(frame, opening_range_minutes=15)

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
    dataset = build_setup_quality_dataset(
        featured,
        detector,
        labeler,
        config=SetupQualityDatasetConfig(include_context=True, include_label_metadata=True),
    )

    featured.to_parquet(FEATURED_PATH, index=False)
    dataset.to_parquet(DATASET_PATH, index=False)

    summary = {
        "rows": len(featured),
        "setups": len(dataset),
        "positive_rate": float(dataset["label"].mean()) if not dataset.empty else None,
        "feature_column_count": len([column for column in dataset.columns if column.startswith("feature_")]),
        "context_column_count": len([column for column in dataset.columns if column.startswith("context_")]),
        "label_metadata_column_count": len([column for column in dataset.columns if column.startswith("label_metadata_")]),
        "start": str(dataset["setup_timestamp"].min()) if not dataset.empty else None,
        "end": str(dataset["setup_timestamp"].max()) if not dataset.empty else None,
        "dataset_path": str(DATASET_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"rows={summary['rows']}")
    print(f"setups={summary['setups']}")
    print(f"positive_rate={summary['positive_rate']}")
    print(f"feature_columns={summary['feature_column_count']}")
    print(f"context_columns={summary['context_column_count']}")
    print(f"label_metadata_columns={summary['label_metadata_column_count']}")
    print(f"saved={DATASET_PATH}")


if __name__ == "__main__":
    main()
