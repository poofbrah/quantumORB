from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".parquet"}
KNOWN_SYMBOLS = ("MNQ", "NQ", "MES", "ES", "CL")


def load_ohlcv(path: str | Path, symbol: str | None = None) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".csv":
        frame = pd.read_csv(file_path)
    else:
        frame = pd.read_parquet(file_path)

    inferred_symbol = symbol or infer_symbol_from_path(file_path)
    if inferred_symbol is not None and "symbol" not in {column.lower() for column in frame.columns}:
        frame["symbol"] = inferred_symbol

    return frame


def infer_symbol_from_path(path: str | Path) -> str | None:
    stem = Path(path).stem.upper()
    for symbol in KNOWN_SYMBOLS:
        if re.search(rf"(?:^|[_\-]){symbol}(?:[_\-]|$)", stem):
            return symbol
    return None


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
