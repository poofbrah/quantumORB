from __future__ import annotations

import numpy as np
import pandas as pd


def add_orb_features(
    frame: pd.DataFrame,
    opening_range_minutes: int = 5,
    by: str = "symbol",
    session_column: str = "session_date",
    atr_column: str = "atr",
) -> pd.DataFrame:
    enriched = frame.copy()
    group_keys = [by, session_column]
    session_start = enriched.groupby(group_keys, sort=False)["timestamp"].transform("min")
    opening_range_end = session_start + pd.to_timedelta(opening_range_minutes, unit="m")
    in_opening_range = enriched["timestamp"] < opening_range_end

    opening_high = enriched["high"].where(in_opening_range)
    opening_low = enriched["low"].where(in_opening_range)

    orb_high = opening_high.groupby([enriched[key] for key in group_keys], sort=False).transform("max")
    orb_low = opening_low.groupby([enriched[key] for key in group_keys], sort=False).transform("min")

    orb_ready = enriched["timestamp"] >= opening_range_end
    enriched["or_high"] = orb_high.where(orb_ready)
    enriched["or_low"] = orb_low.where(orb_ready)
    enriched["or_width"] = enriched["or_high"] - enriched["or_low"]

    if atr_column in enriched.columns:
        enriched["or_width_atr"] = enriched["or_width"] / enriched[atr_column].replace(0.0, np.nan)
    else:
        enriched["or_width_atr"] = np.nan

    enriched["distance_from_or_high"] = enriched["close"] - enriched["or_high"]
    enriched["distance_from_or_low"] = enriched["close"] - enriched["or_low"]

    width = enriched["or_width"].replace(0.0, np.nan)
    breakout_above = ((enriched["close"] - enriched["or_high"]).clip(lower=0.0)) / width
    breakout_below = ((enriched["or_low"] - enriched["close"]).clip(lower=0.0)) / width
    enriched["breakout_strength"] = breakout_above.fillna(0.0) + breakout_below.fillna(0.0)

    historical_volume = enriched.groupby(by, sort=False)["volume"].transform(
        lambda series: series.shift(1).rolling(window=max(5, opening_range_minutes), min_periods=1).mean()
    )
    enriched["relative_volume"] = enriched["volume"] / historical_volume.replace(0.0, np.nan)
    return enriched
