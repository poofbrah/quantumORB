from __future__ import annotations

import math

import pandas as pd

from indicators.technical import add_atr, add_ema, add_log_returns, add_returns, add_rsi, add_sma


def make_indicator_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=6, freq="min", tz="America/New_York"),
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105],
            "volume": [10, 11, 12, 13, 14, 15],
            "symbol": ["ES"] * 6,
            "session_date": [pd.Timestamp("2024-01-02", tz="America/New_York")] * 6,
        }
    )


def test_returns_and_log_returns() -> None:
    frame = make_indicator_frame()
    frame = add_returns(frame)
    frame = add_log_returns(frame)

    assert math.isclose(frame.loc[1, "returns"], 0.01)
    assert math.isclose(frame.loc[1, "log_returns"], math.log(101 / 100))


def test_sma_ema_atr_and_rsi_columns_exist() -> None:
    frame = make_indicator_frame()
    frame = add_sma(frame, window=3)
    frame = add_ema(frame, span=3)
    frame = add_atr(frame, window=3)
    frame = add_rsi(frame, window=3)

    assert "sma_3" in frame.columns
    assert "ema_3" in frame.columns
    assert "atr" in frame.columns
    assert "rsi" in frame.columns
    assert frame["atr"].notna().sum() >= 3
