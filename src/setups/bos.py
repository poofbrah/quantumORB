from __future__ import annotations

import pandas as pd


def add_bos_columns(
    frame: pd.DataFrame,
    lookback_bars: int = 3,
    require_same_direction_candle: bool = True,
) -> pd.DataFrame:
    enriched = frame.copy()
    prior_high = enriched["high"].shift(1).rolling(window=lookback_bars, min_periods=lookback_bars).max()
    prior_low = enriched["low"].shift(1).rolling(window=lookback_bars, min_periods=lookback_bars).min()

    bull = enriched["close"] > prior_high
    bear = enriched["close"] < prior_low
    if require_same_direction_candle:
        bull &= enriched["close"] > enriched["open"]
        bear &= enriched["close"] < enriched["open"]

    enriched["bullish_bos"] = bull.fillna(False)
    enriched["bearish_bos"] = bear.fillna(False)
    return enriched
