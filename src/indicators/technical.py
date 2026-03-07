from __future__ import annotations

import numpy as np
import pandas as pd


def _grouped_series(frame: pd.DataFrame, column: str, by: str = "symbol"):
    return frame.groupby(by, sort=False)[column]


def add_returns(frame: pd.DataFrame, by: str = "symbol", column_name: str = "returns") -> pd.DataFrame:
    enriched = frame.copy()
    enriched[column_name] = _grouped_series(enriched, "close", by).pct_change()
    return enriched


def add_log_returns(frame: pd.DataFrame, by: str = "symbol", column_name: str = "log_returns") -> pd.DataFrame:
    enriched = frame.copy()
    enriched[column_name] = np.log(enriched["close"] / _grouped_series(enriched, "close", by).shift(1))
    return enriched


def add_rolling_volatility(
    frame: pd.DataFrame,
    window: int = 20,
    by: str = "symbol",
    returns_column: str = "returns",
    column_name: str = "rolling_volatility",
) -> pd.DataFrame:
    enriched = frame.copy()
    if returns_column not in enriched.columns:
        enriched = add_returns(enriched, by=by, column_name=returns_column)
    enriched[column_name] = (
        enriched.groupby(by, sort=False)[returns_column]
        .transform(lambda series: series.rolling(window=window, min_periods=window).std())
    )
    return enriched


def add_atr(frame: pd.DataFrame, window: int = 14, by: str = "symbol", column_name: str = "atr") -> pd.DataFrame:
    enriched = frame.copy()
    prev_close = _grouped_series(enriched, "close", by).shift(1)
    true_range = pd.concat(
        [
            enriched["high"] - enriched["low"],
            (enriched["high"] - prev_close).abs(),
            (enriched["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    enriched[column_name] = (
        pd.Series(true_range, index=enriched.index)
        .groupby(enriched[by], sort=False)
        .transform(lambda series: series.rolling(window=window, min_periods=window).mean())
    )
    return enriched


def add_rsi(frame: pd.DataFrame, window: int = 14, by: str = "symbol", column_name: str = "rsi") -> pd.DataFrame:
    enriched = frame.copy()
    delta = _grouped_series(enriched, "close", by).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.groupby(enriched[by], sort=False).transform(
        lambda series: series.rolling(window=window, min_periods=window).mean()
    )
    avg_loss = loss.groupby(enriched[by], sort=False).transform(
        lambda series: series.rolling(window=window, min_periods=window).mean()
    )
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    enriched[column_name] = 100.0 - (100.0 / (1.0 + rs))
    return enriched


def add_ema(
    frame: pd.DataFrame,
    span: int = 20,
    by: str = "symbol",
    source_column: str = "close",
    column_name: str | None = None,
) -> pd.DataFrame:
    enriched = frame.copy()
    target_column = column_name or f"ema_{span}"
    enriched[target_column] = _grouped_series(enriched, source_column, by).transform(
        lambda series: series.ewm(span=span, adjust=False, min_periods=span).mean()
    )
    return enriched


def add_sma(
    frame: pd.DataFrame,
    window: int = 20,
    by: str = "symbol",
    source_column: str = "close",
    column_name: str | None = None,
) -> pd.DataFrame:
    enriched = frame.copy()
    target_column = column_name or f"sma_{window}"
    enriched[target_column] = _grouped_series(enriched, source_column, by).transform(
        lambda series: series.rolling(window=window, min_periods=window).mean()
    )
    return enriched


def add_intraday_vwap(
    frame: pd.DataFrame,
    session_column: str = "session_date",
    by: list[str] | None = None,
    column_name: str = "vwap",
) -> pd.DataFrame:
    enriched = frame.copy()
    keys = by or ["symbol", session_column]
    typical_price = (enriched["high"] + enriched["low"] + enriched["close"]) / 3.0
    price_volume = typical_price * enriched["volume"]
    cumulative_pv = price_volume.groupby([enriched[key] for key in keys], sort=False).cumsum()
    cumulative_volume = enriched["volume"].groupby([enriched[key] for key in keys], sort=False).cumsum()
    enriched[column_name] = cumulative_pv / cumulative_volume.replace(0.0, np.nan)
    return enriched


def add_momentum(
    frame: pd.DataFrame,
    periods: int = 5,
    by: str = "symbol",
    source_column: str = "close",
    column_name: str | None = None,
) -> pd.DataFrame:
    enriched = frame.copy()
    target_column = column_name or f"momentum_{periods}"
    enriched[target_column] = _grouped_series(enriched, source_column, by).diff(periods)
    return enriched
