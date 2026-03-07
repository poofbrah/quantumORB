from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class LiquidityLevel(str, Enum):
    PDH = "pdh"
    PDL = "pdl"
    DAY_HIGH = "day_high"
    DAY_LOW = "day_low"
    H4_HIGH = "h4_high"
    H4_LOW = "h4_low"
    LONDON_HIGH = "london_high"
    LONDON_LOW = "london_low"
    LONDON_SWEEP_CONTEXT = "london_sweep_context"


@dataclass(slots=True)
class LiquidityFrameworkSpec:
    mode: str
    priority: tuple[str, ...]
    london_sweep_context_mode: str | None = None


def add_liquidity_levels(frame: pd.DataFrame, timezone: str = "America/New_York") -> pd.DataFrame:
    enriched = frame.copy().sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)
    timestamps = enriched["timestamp"]
    if timestamps.dt.tz is not None:
        local_ts = timestamps.dt.tz_convert(timezone)
    else:
        local_ts = timestamps
    enriched["local_timestamp"] = local_ts
    enriched["trade_date"] = local_ts.dt.floor("D")

    enriched = _add_previous_day_levels(enriched)
    enriched = _add_current_day_levels(enriched)
    enriched = _add_4h_levels(enriched)
    enriched = _add_london_levels(enriched)
    return enriched.drop(columns=["trade_date"], errors="ignore")


def select_first_liquidity_target(row: pd.Series, direction: str, priority: tuple[str, ...]) -> tuple[str | None, float | None]:
    if not priority:
        return None, None

    entry_price = float(row["close"])
    candidates: list[tuple[float, int, str, float]] = []
    for rank, level_name in enumerate(priority):
        value = row.get(level_name)
        if pd.isna(value):
            continue
        price = float(value)
        if direction == "long" and price > entry_price:
            candidates.append((price - entry_price, rank, level_name, price))
        if direction == "short" and price < entry_price:
            candidates.append((entry_price - price, rank, level_name, price))

    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, name, price = candidates[0]
    return name, price


def select_runner_targets(row: pd.Series, direction: str, priority: tuple[str, ...], first_name: str | None) -> list[tuple[str, float]]:
    entry_price = float(row["close"])
    targets: list[tuple[str, float, int]] = []
    for rank, level_name in enumerate(priority):
        if level_name == first_name:
            continue
        value = row.get(level_name)
        if pd.isna(value):
            continue
        price = float(value)
        if direction == "long" and price > entry_price:
            targets.append((level_name, price, rank))
        if direction == "short" and price < entry_price:
            targets.append((level_name, price, rank))
    if direction == "long":
        targets.sort(key=lambda item: (item[1], item[2]))
    else:
        targets.sort(key=lambda item: (-item[1], item[2]))
    return [(name, price) for name, price, _ in targets]


def _add_previous_day_levels(frame: pd.DataFrame) -> pd.DataFrame:
    daily = (
        frame.groupby(["symbol", "trade_date"], sort=False)
        .agg(prev_high=("high", "max"), prev_low=("low", "min"))
        .reset_index()
        .sort_values(["symbol", "trade_date"], kind="stable")
    )
    daily["pdh"] = daily.groupby("symbol", sort=False)["prev_high"].shift(1)
    daily["pdl"] = daily.groupby("symbol", sort=False)["prev_low"].shift(1)
    merged = frame.merge(daily[["symbol", "trade_date", "pdh", "pdl"]], on=["symbol", "trade_date"], how="left")
    return merged


def _add_current_day_levels(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["day_high"] = (
        enriched.groupby(["symbol", "trade_date"], sort=False)["high"]
        .cummax()
        .groupby([enriched["symbol"], enriched["trade_date"]], sort=False)
        .shift(1)
    )
    enriched["day_low"] = (
        enriched.groupby(["symbol", "trade_date"], sort=False)["low"]
        .cummin()
        .groupby([enriched["symbol"], enriched["trade_date"]], sort=False)
        .shift(1)
    )
    return enriched


def _add_4h_levels(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for symbol, group in frame.groupby("symbol", sort=False):
        indexed = group.sort_values("timestamp").set_index("timestamp")
        high_4h = indexed["high"].rolling("4h", closed="left").max()
        low_4h = indexed["low"].rolling("4h", closed="left").min()
        group = group.copy()
        group["h4_high"] = high_4h.to_numpy()
        group["h4_low"] = low_4h.to_numpy()
        parts.append(group)
    return pd.concat(parts, ignore_index=True) if parts else frame.copy()


def _add_london_levels(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    local_ts = enriched["local_timestamp"]
    london_mask = (local_ts.dt.time >= pd.Timestamp("03:00").time()) & (local_ts.dt.time <= pd.Timestamp("08:29").time())
    london = (
        enriched.loc[london_mask]
        .groupby(["symbol", "trade_date"], sort=False)
        .agg(london_high=("high", "max"), london_low=("low", "min"))
        .reset_index()
    )
    enriched = enriched.merge(london, on=["symbol", "trade_date"], how="left")
    return enriched
