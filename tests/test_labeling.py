from __future__ import annotations

from datetime import timedelta

import pandas as pd

from data.schemas import SetupEvent, SetupStatus, Side
from labeling.future_path import ForwardSetupLabeler, IntrabarConflictPolicy, LabelerConfig


def make_future_bars() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30", periods=6, freq="min", tz="America/New_York")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["ES"] * 6,
            "high": [100.4, 100.6, 101.2, 101.9, 102.8, 103.0],
            "low": [99.8, 100.0, 100.4, 100.9, 101.8, 102.0],
            "close": [100.1, 100.5, 101.0, 101.7, 102.5, 102.7],
        }
    )


def make_setup(timestamp: pd.Timestamp, direction: Side = Side.LONG) -> SetupEvent:
    return SetupEvent(
        setup_id="orb-test",
        setup_name="orb",
        symbol="ES",
        contract="ESH4",
        timestamp=timestamp.to_pydatetime(),
        session_date=timestamp.normalize().to_pydatetime(),
        direction=direction,
        status=SetupStatus.CANDIDATE,
        entry_reference=101.0,
        stop_reference=100.0 if direction is Side.LONG else 102.0,
        target_reference=103.0 if direction is Side.LONG else 99.0,
        features={"or_high": 100.6},
        context={},
    )


def test_label_generation_binary_and_regression() -> None:
    frame = make_future_bars()
    setup = make_setup(frame.loc[2, "timestamp"])

    labeled = ForwardSetupLabeler(LabelerConfig(horizon_bars=3)).label(frame, [setup])

    assert len(labeled) == 1
    item = labeled[0]
    assert item.label == 1
    assert item.realized_return == 2.0
    assert item.quality_bucket == "great"
    assert item.metadata["regression_label"] == 2.0
    assert item.metadata["intrabar_conflict_policy"] == "stop_first"


def test_label_generation_uses_only_future_bars() -> None:
    frame = make_future_bars().copy()
    setup_time = frame.loc[2, "timestamp"]
    frame.loc[2, "high"] = 103.5
    setup = make_setup(setup_time)

    labeled = ForwardSetupLabeler(LabelerConfig(horizon_bars=1)).label(frame, [setup])

    assert labeled[0].label == 0
    assert labeled[0].realized_return == 0.7


def test_short_label_generation() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=5, freq="min", tz="America/New_York"),
            "symbol": ["ES"] * 5,
            "high": [100.4, 100.3, 100.1, 99.7, 99.2],
            "low": [99.8, 99.6, 99.3, 98.8, 98.7],
            "close": [100.1, 99.8, 99.4, 99.0, 98.9],
        }
    )
    setup = SetupEvent(
        setup_id="orb-short",
        setup_name="orb",
        symbol="ES",
        contract="ESH4",
        timestamp=frame.loc[1, "timestamp"].to_pydatetime(),
        session_date=frame.loc[1, "timestamp"].normalize().to_pydatetime(),
        direction=Side.SHORT,
        status=SetupStatus.CANDIDATE,
        entry_reference=99.8,
        stop_reference=100.8,
        target_reference=97.8,
        features={},
        context={},
    )

    labeled = ForwardSetupLabeler(LabelerConfig(horizon_bars=3)).label(frame, [setup])
    assert labeled[0].label == 0
    assert labeled[0].realized_return == 0.9


def test_same_bar_conflict_defaults_to_stop_first() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=3, freq="min", tz="America/New_York"),
            "symbol": ["ES"] * 3,
            "high": [100.4, 100.5, 103.2],
            "low": [99.8, 100.0, 99.8],
            "close": [100.1, 101.0, 101.5],
        }
    )
    setup = make_setup(frame.loc[1, "timestamp"])

    labeled = ForwardSetupLabeler(LabelerConfig(horizon_bars=1)).label(frame, [setup])
    assert labeled[0].label == 0
    assert labeled[0].realized_return == -1.0


def test_same_bar_conflict_can_be_made_target_first() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=3, freq="min", tz="America/New_York"),
            "symbol": ["ES"] * 3,
            "high": [100.4, 100.5, 103.2],
            "low": [99.8, 100.0, 99.8],
            "close": [100.1, 101.0, 101.5],
        }
    )
    setup = make_setup(frame.loc[1, "timestamp"])

    labeled = ForwardSetupLabeler(
        LabelerConfig(horizon_bars=1, intrabar_conflict_policy=IntrabarConflictPolicy.TARGET_FIRST)
    ).label(frame, [setup])
    assert labeled[0].label == 1
    assert labeled[0].realized_return == 2.0


def test_unresolved_label_can_be_neutral() -> None:
    frame = make_future_bars().iloc[:4].copy()
    setup = make_setup(frame.loc[2, "timestamp"] + timedelta(minutes=10))

    labeled = ForwardSetupLabeler(LabelerConfig(horizon_bars=2, neutral_if_unresolved=True)).label(frame, [setup])
    assert labeled[0].label == -1
    assert labeled[0].realized_return == 0.0
