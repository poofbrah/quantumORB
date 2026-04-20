from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.pipeline import load_and_preprocess_ohlcv, save_processed_ohlcv

RAW_PATH = ROOT / "data" / "raw" / "Dataset_NQ_1min_2022_2025.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "nq_1min_2022_2025.parquet"


def main() -> None:
    frame = load_and_preprocess_ohlcv(
        RAW_PATH,
        timezone="America/New_York",
        session_start=None,
        session_end=None,
        resample_rule=None,
    )
    save_processed_ohlcv(frame, PROCESSED_PATH)
    print(f"rows={len(frame)}")
    print(f"columns={list(frame.columns)}")
    print(f"saved={PROCESSED_PATH}")


if __name__ == "__main__":
    main()
