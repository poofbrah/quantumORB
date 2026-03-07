from __future__ import annotations

import pandas as pd

from features.pipeline import build_feature_frame


def make_feature_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30", periods=8, freq="min", tz="America/New_York")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 100.5, 101.0, 101.6, 102.2, 102.8, 103.0, 103.4],
            "high": [100.7, 101.1, 101.8, 102.5, 102.9, 103.2, 103.6, 104.0],
            "low": [99.8, 100.2, 100.8, 101.2, 101.9, 102.5, 102.8, 103.2],
            "close": [100.5, 101.0, 101.6, 102.2, 102.7, 103.0, 103.5, 103.8],
            "volume": [100, 110, 120, 130, 140, 150, 160, 170],
            "symbol": ["ES"] * 8,
            "session_date": [pd.Timestamp("2024-01-02", tz="America/New_York")] * 8,
        }
    )


def test_feature_pipeline_generates_expected_columns() -> None:
    frame = build_feature_frame(
        make_feature_frame(),
        opening_range_minutes=2,
        volatility_window=3,
        atr_window=3,
        rsi_window=3,
        ema_span=3,
        sma_window=3,
        momentum_periods=2,
    )

    expected_columns = {
        "returns",
        "log_returns",
        "rolling_volatility",
        "atr",
        "rsi",
        "ema_3",
        "sma_3",
        "momentum_2",
        "vwap",
        "candle_range",
        "candle_body",
        "upper_wick",
        "lower_wick",
        "body_range_ratio",
        "hour",
        "minute",
        "day_of_week",
        "session_bucket",
        "volatility_regime",
        "or_high",
        "or_low",
        "or_width",
        "or_width_atr",
        "distance_from_or_high",
        "distance_from_or_low",
        "breakout_strength",
        "relative_volume",
    }
    assert expected_columns.issubset(frame.columns)


def test_orb_features_do_not_populate_before_range_complete() -> None:
    frame = build_feature_frame(
        make_feature_frame(),
        opening_range_minutes=2,
        volatility_window=3,
        atr_window=3,
        rsi_window=3,
        ema_span=3,
        sma_window=3,
        momentum_periods=2,
    )

    assert pd.isna(frame.loc[0, "or_high"])
    assert pd.isna(frame.loc[1, "or_high"])
    assert frame.loc[2, "or_high"] == max(frame.loc[0:1, "high"])
    assert frame.loc[2, "or_low"] == min(frame.loc[0:1, "low"])
