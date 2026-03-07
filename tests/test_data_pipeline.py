from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.io import load_ohlcv, save_dataset
from data.pipeline import load_and_preprocess_ohlcv
from data.preprocess import preprocess_ohlcv, standardize_ohlcv_schema


@pytest.fixture()
def raw_intraday_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DateTime": [
                "2024-01-02 14:31:00",
                "2024-01-02 14:30:00",
                "2024-01-02 14:31:00",
                "2024-01-02 21:05:00",
            ],
            "Open": [101.0, 100.0, 101.5, 103.0],
            "High": [102.0, 101.0, 102.5, 104.0],
            "Low": [100.5, 99.5, 100.0, 102.5],
            "Close": [101.5, 100.5, 102.0, 103.5],
            "Volume": [20, 10, 30, 5],
            "Symbol": ["ES", "ES", "ES", "ES"],
        }
    )


def test_load_csv_and_standardize(tmp_path: Path, raw_intraday_frame: pd.DataFrame) -> None:
    csv_path = tmp_path / "sample.csv"
    raw_intraday_frame.to_csv(csv_path, index=False)

    loaded = load_ohlcv(csv_path)
    standardized = standardize_ohlcv_schema(loaded)

    assert list(standardized.columns[:7]) == ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    assert str(standardized["timestamp"].dtype).startswith("datetime64")


def test_load_parquet(tmp_path: Path, raw_intraday_frame: pd.DataFrame) -> None:
    pytest.importorskip("pyarrow")
    parquet_path = tmp_path / "sample.parquet"
    raw_intraday_frame.to_parquet(parquet_path, index=False)

    loaded = load_ohlcv(parquet_path)
    assert len(loaded) == 4


def test_preprocess_sorts_deduplicates_and_filters_session(raw_intraday_frame: pd.DataFrame) -> None:
    processed = preprocess_ohlcv(
        raw_intraday_frame,
        timezone="America/New_York",
        session_start="09:30",
        session_end="16:00",
    )

    assert len(processed) == 2
    assert processed["timestamp"].is_monotonic_increasing
    assert processed.iloc[0]["timestamp"].hour == 9
    assert processed.iloc[0]["timestamp"].minute == 30
    assert "session_date" in processed.columns


def test_resample_and_save_processed_dataset(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02 14:30:00+00:00",
                    "2024-01-02 14:31:00+00:00",
                    "2024-01-02 14:32:00+00:00",
                    "2024-01-02 14:33:00+00:00",
                ]
            ),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [10, 20, 30, 40],
            "symbol": ["ES", "ES", "ES", "ES"],
        }
    )

    csv_path = tmp_path / "processed.csv"
    processed = preprocess_ohlcv(frame, timezone="America/New_York", resample_rule="2min")
    save_dataset(processed, csv_path)

    assert len(processed) == 2
    assert csv_path.exists()


def test_load_and_preprocess_pipeline(tmp_path: Path, raw_intraday_frame: pd.DataFrame) -> None:
    csv_path = tmp_path / "sample.csv"
    raw_intraday_frame.to_csv(csv_path, index=False)

    processed = load_and_preprocess_ohlcv(csv_path, timezone="America/New_York", session_start="09:30", session_end="16:00")
    assert len(processed) == 2
