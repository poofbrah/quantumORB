from __future__ import annotations

from datetime import time

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
COLUMN_ALIASES = {
    "datetime": "timestamp",
    "date": "timestamp",
    "time": "timestamp",
    "timestamp et": "timestamp",
    "timestamp_et": "timestamp",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "ticker": "symbol",
}
ET_TIMESTAMP_COLUMNS = {"timestamp et", "timestamp_et"}
DEFAULT_NAIVE_TIMESTAMP_TIMEZONE = "UTC"


def standardize_ohlcv_schema(frame: pd.DataFrame) -> pd.DataFrame:
    source_columns = {column.lower(): column for column in frame.columns}
    renamed = frame.rename(
        columns={column: COLUMN_ALIASES.get(column.lower(), column.lower()) for column in frame.columns}
    )
    missing = [column for column in REQUIRED_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ordered_columns = REQUIRED_COLUMNS + [col for col in renamed.columns if col not in REQUIRED_COLUMNS]
    standardized = renamed.loc[:, list(dict.fromkeys(ordered_columns))].copy()
    standardized["timestamp"] = _parse_timestamp_column(standardized["timestamp"], source_columns)

    numeric_columns = [column for column in ["open", "high", "low", "close", "volume", "vwap_rth", "vwap_eth"] if column in standardized.columns]
    for column in numeric_columns:
        standardized[column] = pd.to_numeric(standardized[column], errors="raise")

    standardized["symbol"] = standardized["symbol"].astype(str)
    return standardized


def _parse_timestamp_column(series: pd.Series, source_columns: dict[str, str]) -> pd.Series:
    parsed = pd.to_datetime(series, errors="raise")
    if getattr(parsed.dt, "tz", None) is not None:
        return parsed
    if any(column in ET_TIMESTAMP_COLUMNS for column in source_columns):
        return parsed.dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
    return parsed.dt.tz_localize(DEFAULT_NAIVE_TIMESTAMP_TIMEZONE)


def clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.dropna(subset=REQUIRED_COLUMNS).copy()
    cleaned = cleaned.sort_values(["symbol", "timestamp"], kind="stable")
    cleaned = cleaned.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def localize_timestamps(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    localized = frame.copy()
    localized["timestamp"] = localized["timestamp"].dt.tz_convert(timezone)
    return localized


def add_session_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    # v1 simplification: session_date is derived from regular-trading-hours local calendar date.
    enriched["session_date"] = enriched["timestamp"].dt.floor("D")
    return enriched


def filter_session_hours(
    frame: pd.DataFrame,
    session_start: str | None = None,
    session_end: str | None = None,
) -> pd.DataFrame:
    if not session_start and not session_end:
        return frame.copy()

    filtered = frame.copy()
    start_time = time.fromisoformat(session_start) if session_start else None
    end_time = time.fromisoformat(session_end) if session_end else None
    times = filtered["timestamp"].dt.time

    mask = pd.Series(True, index=filtered.index)
    if start_time is not None:
        mask &= times >= start_time
    if end_time is not None:
        mask &= times <= end_time
    return filtered.loc[mask].reset_index(drop=True)


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    if not rule:
        return frame.copy()

    pieces: list[pd.DataFrame] = []
    for symbol, group in frame.groupby("symbol", sort=False):
        indexed = group.sort_values("timestamp").set_index("timestamp")
        agg_map = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        for optional_column in ("vwap_rth", "vwap_eth"):
            if optional_column in indexed.columns:
                agg_map[optional_column] = "last"
        resampled = indexed.resample(rule).agg(agg_map)
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        resampled["symbol"] = symbol
        for column in indexed.columns:
            if column in {*agg_map.keys(), "symbol"}:
                continue
            resampled[column] = indexed[column].resample(rule).last()
        pieces.append(resampled.reset_index())

    if not pieces:
        return frame.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def preprocess_ohlcv(
    frame: pd.DataFrame,
    timezone: str = "America/New_York",
    session_start: str | None = None,
    session_end: str | None = None,
    resample_rule: str | None = None,
) -> pd.DataFrame:
    processed = standardize_ohlcv_schema(frame)
    processed = clean_ohlcv(processed)
    processed = localize_timestamps(processed, timezone)
    processed = add_session_columns(processed)
    processed = filter_session_hours(processed, session_start=session_start, session_end=session_end)
    if resample_rule:
        processed = resample_ohlcv(processed, resample_rule)
        processed = add_session_columns(processed)
    processed = processed.reset_index(drop=True)
    return processed
