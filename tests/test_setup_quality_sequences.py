from __future__ import annotations

import pytest
import pandas as pd

from data.schemas import SetupEvent, SetupStatus, Side
from labeling.future_path import ForwardSetupLabeler, LabelerConfig
from models.setup_quality_sequences import (
    SetupSequenceDatasetConfig,
    build_setup_quality_sequence_dataset,
)


def make_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30", periods=8, freq="min", tz="America/New_York")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["NQ"] * 8,
            "contract": ["NQH4"] * 8,
            "session_date": [timestamps[0].normalize()] * 8,
            "open": [100.0, 100.4, 100.9, 101.2, 101.8, 102.1, 102.4, 102.8],
            "high": [100.5, 100.9, 101.3, 101.8, 102.2, 102.6, 103.0, 103.4],
            "low": [99.8, 100.2, 100.7, 101.0, 101.5, 101.9, 102.2, 102.6],
            "close": [100.3, 100.8, 101.1, 101.7, 102.0, 102.4, 102.8, 103.1],
            "volume": [100, 110, 120, 130, 140, 150, 160, 170],
            "vwap": [100.1, 100.4, 100.8, 101.1, 101.5, 101.9, 102.2, 102.6],
            "atr": [0.5, 0.55, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7],
            "returns": [0.0, 0.01, 0.003, 0.006, 0.003, 0.004, 0.004, 0.003],
            "rolling_volatility": [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
            "rsi": [50, 52, 55, 58, 60, 62, 64, 66],
            "body_range_ratio": [0.6, 0.65, 0.66, 0.7, 0.72, 0.7, 0.75, 0.76],
            "or_high": [None, None, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0],
            "or_low": [None, None, 99.8, 99.8, 99.8, 99.8, 99.8, 99.8],
            "distance_from_or_high": [None, None, 0.1, 0.7, 1.0, 1.4, 1.8, 2.1],
            "distance_from_or_low": [None, None, 1.3, 1.9, 2.2, 2.6, 3.0, 3.3],
            "breakout_strength": [0.0, 0.0, 0.05, 0.2, 0.3, 0.35, 0.4, 0.45],
            "trend_bias": [0, 0, 1, 1, 1, 1, 1, 1],
            "trend_spread": [0.0, 0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            "recent_bullish_fvg_size": [0.0, 0.0, 0.0, 0.15, 0.2, 0.25, 0.3, 0.35],
            "recent_bearish_fvg_size": [0.0] * 8,
        }
    )


def make_setup(frame: pd.DataFrame, setup_index: int) -> SetupEvent:
    timestamp = frame.loc[setup_index, "timestamp"]
    return SetupEvent(
        setup_id=f"setup-{setup_index}",
        setup_name="orb_session_vwap_retest",
        symbol="NQ",
        contract="NQH4",
        timestamp=timestamp.to_pydatetime(),
        session_date=timestamp.normalize().to_pydatetime(),
        direction=Side.LONG,
        status=SetupStatus.CANDIDATE,
        entry_reference=float(frame.loc[setup_index, "close"]),
        stop_reference=100.0,
        target_reference=104.0,
        features={"breakout_strength": float(frame.loc[setup_index, "breakout_strength"])},
        context={"entry_family": "orb_session_vwap_retest"},
    )


def test_build_setup_quality_sequence_dataset_extracts_last_bars_in_order() -> None:
    frame = make_frame()
    labeled = ForwardSetupLabeler(LabelerConfig(horizon_bars=2)).label(frame, [make_setup(frame, 5)])

    dataset = build_setup_quality_sequence_dataset(
        frame,
        labeled,
        config=SetupSequenceDatasetConfig(
            lookback_bars=3,
            sequence_columns=("close", "trend_bias"),
        ),
    )

    assert dataset.sequences.shape == (1, 3, 2)
    closes = dataset.sequences[0, :, 0].tolist()
    assert closes == pytest.approx([101.7, 102.0, 102.4])
    assert dataset.metadata.loc[0, "setup_id"] == "setup-5"


def test_build_setup_quality_sequence_dataset_can_exclude_setup_bar() -> None:
    frame = make_frame()
    labeled = ForwardSetupLabeler(LabelerConfig(horizon_bars=2)).label(frame, [make_setup(frame, 5)])

    dataset = build_setup_quality_sequence_dataset(
        frame,
        labeled,
        config=SetupSequenceDatasetConfig(
            lookback_bars=3,
            include_setup_bar=False,
            sequence_columns=("close",),
        ),
    )

    closes = dataset.sequences[0, :, 0].tolist()
    assert closes == pytest.approx([101.1, 101.7, 102.0])


def test_build_setup_quality_sequence_dataset_drops_short_history_when_requested() -> None:
    frame = make_frame()
    labeled = ForwardSetupLabeler(LabelerConfig(horizon_bars=2)).label(frame, [make_setup(frame, 2)])

    dataset = build_setup_quality_sequence_dataset(
        frame,
        labeled,
        config=SetupSequenceDatasetConfig(
            lookback_bars=5,
            sequence_columns=("close",),
        ),
    )

    assert dataset.sequences.shape[0] == 0
    assert dataset.metadata.empty
