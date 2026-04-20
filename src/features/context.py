from __future__ import annotations

import numpy as np
import pandas as pd


def add_trend_context(
    frame: pd.DataFrame,
    fast_ema_column: str = "ema_20",
    slow_ema_column: str = "ema_50",
) -> pd.DataFrame:
    enriched = frame.copy()
    if fast_ema_column not in enriched.columns or slow_ema_column not in enriched.columns:
        enriched["trend_bias"] = 0
        enriched["trend_spread"] = np.nan
        enriched["trend_spread_pct"] = np.nan
        enriched["close_vs_fast_ema"] = np.nan
        return enriched

    fast = enriched[fast_ema_column]
    slow = enriched[slow_ema_column]
    enriched["trend_spread"] = fast - slow
    enriched["trend_spread_pct"] = enriched["trend_spread"] / slow.replace(0.0, np.nan)
    enriched["close_vs_fast_ema"] = enriched["close"] - fast
    enriched["trend_bias"] = np.select(
        [fast > slow, fast < slow],
        [1, -1],
        default=0,
    )
    return enriched


def add_fair_value_gap_features(
    frame: pd.DataFrame,
    by: str = "symbol",
    lookback_bars: int = 8,
) -> pd.DataFrame:
    enriched = frame.copy()
    prior_high_2 = enriched.groupby(by, sort=False)["high"].shift(2)
    prior_low_2 = enriched.groupby(by, sort=False)["low"].shift(2)

    enriched["bullish_fvg_size"] = (enriched["low"] - prior_high_2).clip(lower=0.0)
    enriched["bearish_fvg_size"] = (prior_low_2 - enriched["high"]).clip(lower=0.0)
    enriched["has_bullish_fvg"] = enriched["bullish_fvg_size"] > 0.0
    enriched["has_bearish_fvg"] = enriched["bearish_fvg_size"] > 0.0

    grouped_bullish = enriched.groupby(by, sort=False)["bullish_fvg_size"]
    grouped_bearish = enriched.groupby(by, sort=False)["bearish_fvg_size"]
    enriched["recent_bullish_fvg_size"] = grouped_bullish.transform(
        lambda series: series.shift(1).rolling(window=lookback_bars, min_periods=1).max()
    )
    enriched["recent_bearish_fvg_size"] = grouped_bearish.transform(
        lambda series: series.shift(1).rolling(window=lookback_bars, min_periods=1).max()
    )
    enriched["recent_any_fvg_size"] = (
        enriched[["recent_bullish_fvg_size", "recent_bearish_fvg_size"]].max(axis=1)
    )
    return enriched
