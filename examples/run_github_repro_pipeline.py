from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET = ROOT / "data" / "raw" / "Dataset_NQ_1min_2022_2025.csv"
PROCESSED_DATASET = ROOT / "data" / "processed" / "nq_1min_2022_2025.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproducible QuantumORB research pipeline end to end."
    )
    parser.add_argument(
        "--with-preprocess",
        action="store_true",
        help="Run CSV preprocessing first. Use this when the raw dataset is present and the processed parquet is missing.",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip the LSTM stage if you only want the tabular/model comparison outputs.",
    )
    parser.add_argument(
        "--skip-regime-compare",
        action="store_true",
        help="Skip the continuation/range-reversion/hybrid regime comparison stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.with_preprocess:
        if not RAW_DATASET.exists():
            raise FileNotFoundError(
                f"Raw dataset not found at {RAW_DATASET}. Place Dataset_NQ_1min_2022_2025.csv there or rerun without --with-preprocess."
            )
        run_step("preprocess_nq_dataset.py")
    elif not PROCESSED_DATASET.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_DATASET}. Run with --with-preprocess or add the parquet first."
        )

    run_step("run_setup_quality_research_pipeline.py")
    run_step("build_setup_quality_sequence_dataset.py")

    if not args.skip_lstm:
        run_step("run_setup_quality_lstm.py")

    run_step("run_setup_quality_model_comparison.py")

    if not args.skip_regime_compare:
        run_step("run_orb_regime_mode_compare.py")

    print("pipeline=complete")
    print(f"outputs={ROOT / 'outputs'}")


def run_step(script_name: str) -> None:
    script_path = ROOT / "examples" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    print(f"stage={script_name}")
    subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
