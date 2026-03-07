from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import load_ohlcv, save_dataset
from .preprocess import preprocess_ohlcv


def load_and_preprocess_ohlcv(
    path: str | Path,
    symbol: str | None = None,
    timezone: str = "America/New_York",
    session_start: str | None = None,
    session_end: str | None = None,
    resample_rule: str | None = None,
) -> pd.DataFrame:
    raw = load_ohlcv(path, symbol=symbol)
    return preprocess_ohlcv(
        raw,
        timezone=timezone,
        session_start=session_start,
        session_end=session_end,
        resample_rule=resample_rule,
    )


def save_processed_ohlcv(frame: pd.DataFrame, path: str | Path) -> None:
    save_dataset(frame, path)
