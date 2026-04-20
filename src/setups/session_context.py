from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta

import pandas as pd


@dataclass(slots=True)
class SessionWindow:
    start: time
    end: time


def parse_session_window(window: str) -> SessionWindow:
    start_text, end_text = window.split("-")
    return SessionWindow(start=time.fromisoformat(start_text), end=time.fromisoformat(end_text))


def in_session(timestamp: pd.Timestamp, window: str) -> bool:
    session = parse_session_window(window)
    current = timestamp.time()
    if session.start <= session.end:
        return session.start <= current <= session.end
    return current >= session.start or current <= session.end


def add_trading_day(frame: pd.DataFrame, reset_time: str = "18:00", timezone: str = "America/New_York") -> pd.DataFrame:
    enriched = frame.copy()
    timestamps = pd.to_datetime(enriched["timestamp"])
    if timestamps.dt.tz is not None:
        local_ts = timestamps.dt.tz_convert(timezone)
    else:
        local_ts = timestamps.dt.tz_localize(timezone)
    enriched["local_timestamp"] = local_ts

    reset_at = time.fromisoformat(reset_time)
    trade_day = local_ts.dt.normalize()
    after_reset = local_ts.dt.time >= reset_at
    trade_day = trade_day.where(~after_reset, trade_day + pd.Timedelta(days=1))
    enriched["trade_day"] = trade_day
    return enriched


def within_any_trade_window(timestamp: pd.Timestamp, windows: tuple[str, ...]) -> bool:
    if not windows:
        return True
    return any(in_session(timestamp, window) for window in windows)


def window_end_timestamp(trade_day: pd.Timestamp, window: str, timezone: str = "America/New_York") -> pd.Timestamp:
    session = parse_session_window(window)
    session_date = pd.Timestamp(trade_day)
    end_dt = pd.Timestamp.combine(session_date.date(), session.end)
    localized = pd.Timestamp(end_dt).tz_localize(timezone)
    if session.start > session.end:
        localized += pd.Timedelta(days=1)
    return localized
