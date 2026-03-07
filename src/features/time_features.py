from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_BUCKETS = [
    ("09:30", "10:29", "open"),
    ("10:30", "11:59", "mid_morning"),
    ("12:00", "13:59", "midday"),
    ("14:00", "16:00", "close"),
]


def add_time_features(frame: pd.DataFrame, timezone: str = "America/New_York") -> pd.DataFrame:
    enriched = frame.copy()
    timestamps = enriched["timestamp"]
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert(timezone)
    enriched["hour"] = timestamps.dt.hour
    enriched["minute"] = timestamps.dt.minute
    enriched["day_of_week"] = timestamps.dt.dayofweek
    enriched["session_bucket"] = _session_bucket(timestamps)
    return enriched


def _session_bucket(timestamps: pd.Series) -> pd.Series:
    labels = []
    for ts in timestamps:
        current = ts.strftime("%H:%M")
        label = "off_session"
        for start, end, bucket in DEFAULT_BUCKETS:
            if start <= current <= end:
                label = bucket
                break
        labels.append(label)
    return pd.Series(labels, index=timestamps.index, dtype="object")


def add_volatility_regime(
    frame: pd.DataFrame,
    volatility_column: str = "rolling_volatility",
    by: str = "symbol",
    window: int = 50,
    column_name: str = "volatility_regime",
) -> pd.DataFrame:
    enriched = frame.copy()
    baseline = enriched.groupby(by, sort=False)[volatility_column].transform(
        lambda series: series.shift(1).rolling(window=window, min_periods=max(5, window // 5)).median()
    )
    enriched[column_name] = np.where(
        enriched[volatility_column] > baseline,
        "high",
        np.where(enriched[volatility_column] < baseline, "low", "normal"),
    )
    enriched.loc[baseline.isna(), column_name] = pd.NA
    return enriched
