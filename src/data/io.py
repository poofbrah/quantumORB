from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".parquet"}


def load_ohlcv(path: str | Path, symbol: str | None = None) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".csv":
        frame = pd.read_csv(file_path)
    else:
        frame = pd.read_parquet(file_path)

    if symbol is not None and "symbol" not in frame.columns:
        frame["symbol"] = symbol

    return frame


def save_dataset(frame: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(file_path, index=index)
        return
    if suffix == ".parquet":
        frame.to_parquet(file_path, index=index)
        return
    raise ValueError(f"Unsupported file type: {suffix}")
