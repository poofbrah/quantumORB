from __future__ import annotations

import numpy as np
import pandas as pd


def add_candle_anatomy_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["candle_range"] = enriched["high"] - enriched["low"]
    enriched["candle_body"] = (enriched["close"] - enriched["open"]).abs()
    enriched["upper_wick"] = enriched["high"] - enriched[["open", "close"]].max(axis=1)
    enriched["lower_wick"] = enriched[["open", "close"]].min(axis=1) - enriched["low"]
    enriched["body_range_ratio"] = enriched["candle_body"] / enriched["candle_range"].replace(0.0, np.nan)
    return enriched
