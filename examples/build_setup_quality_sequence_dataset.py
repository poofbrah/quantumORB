from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.preprocess import filter_session_hours
from features.pipeline import build_feature_frame
from labeling.future_path import ForwardSetupLabeler, LabelerConfig
from models import (
    SetupSequenceDatasetConfig,
    build_setup_quality_sequence_dataset,
    save_setup_sequence_dataset,
)
from setups.orb_session_vwap_retest import ORBSessionVWAPRetestConfig, ORBSessionVWAPRetestDetector

INPUT_PATH = ROOT / "data" / "processed" / "nq_1min_2022_2025.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an LSTM-ready setup-quality sequence dataset.")
    parser.add_argument("--lookback-bars", type=int, default=30)
    parser.add_argument("--target-mode", choices=["fixed_r", "liquidity"], default="liquidity")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--exclude-setup-bar", action="store_true")
    parser.add_argument("--require-fvg-context", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / f"setup_quality_sequence_dataset_orb_session_vwap_retest_{args.target_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(INPUT_PATH)
    rth_frame = filter_session_hours(frame, session_start="09:30", session_end="16:00")
    featured = build_feature_frame(rth_frame, opening_range_minutes=15)
    featured.to_parquet(output_dir / "featured_nq_1min_2022_2025.parquet", index=False)

    detector = ORBSessionVWAPRetestDetector(
        ORBSessionVWAPRetestConfig(
            opening_range_start_time="09:30",
            opening_range_end_time="09:45",
            opening_range_minutes=15,
            latest_entry_time="11:30",
            allowed_trade_windows=("09:45-11:30",),
            target_r_multiple=2.0,
            target_mode=args.target_mode,
            require_trend_alignment=True,
            require_fvg_context=args.require_fvg_context,
        )
    )
    setups = detector.detect(featured)
    labeled_setups = ForwardSetupLabeler(LabelerConfig(horizon_bars=20)).label(featured, setups)

    dataset = build_setup_quality_sequence_dataset(
        featured,
        labeled_setups,
        config=SetupSequenceDatasetConfig(
            lookback_bars=args.lookback_bars,
            include_setup_bar=not args.exclude_setup_bar,
        ),
    )
    paths = save_setup_sequence_dataset(dataset, output_dir)

    summary = {
        "rows": len(featured),
        "setups_detected": len(setups),
        "sequence_count": int(dataset.sequences.shape[0]),
        "lookback_bars": args.lookback_bars,
        "feature_count": len(dataset.feature_columns),
        "include_setup_bar": not args.exclude_setup_bar,
        "require_fvg_context": args.require_fvg_context,
        **paths,
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"rows={summary['rows']}")
    print(f"setups_detected={summary['setups_detected']}")
    print(f"sequence_count={summary['sequence_count']}")
    print(f"lookback_bars={summary['lookback_bars']}")
    print(f"feature_count={summary['feature_count']}")
    print(f"include_setup_bar={summary['include_setup_bar']}")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
