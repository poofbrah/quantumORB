from __future__ import annotations

import pandas as pd

from .candles import add_candle_anatomy_features
from .orb import add_orb_features
from .time_features import add_time_features, add_volatility_regime
from indicators.technical import (
    add_atr,
    add_ema,
    add_intraday_vwap,
    add_log_returns,
    add_momentum,
    add_returns,
    add_rolling_volatility,
    add_rsi,
    add_sma,
)


def build_feature_frame(
    frame: pd.DataFrame,
    opening_range_minutes: int = 5,
    volatility_window: int = 20,
    atr_window: int = 14,
    rsi_window: int = 14,
    ema_span: int = 20,
    sma_window: int = 20,
    momentum_periods: int = 5,
    include_vwap: bool = True,
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched = add_returns(enriched)
    enriched = add_log_returns(enriched)
    enriched = add_rolling_volatility(enriched, window=volatility_window)
    enriched = add_atr(enriched, window=atr_window)
    enriched = add_rsi(enriched, window=rsi_window)
    enriched = add_ema(enriched, span=ema_span)
    enriched = add_sma(enriched, window=sma_window)
    enriched = add_momentum(enriched, periods=momentum_periods)
    if include_vwap and "session_date" in enriched.columns:
        enriched = add_intraday_vwap(enriched)
    enriched = add_candle_anatomy_features(enriched)
    enriched = add_time_features(enriched)
    enriched = add_volatility_regime(enriched)
    enriched = add_orb_features(enriched, opening_range_minutes=opening_range_minutes)
    return enriched
