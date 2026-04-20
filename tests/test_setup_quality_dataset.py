from __future__ import annotations

import pandas as pd

from data.schemas import SetupEvent, SetupStatus, Side
from labeling.future_path import ForwardSetupLabeler, LabelerConfig
from models.setup_quality_dataset import add_walk_forward_columns, build_setup_quality_dataset, labeled_setups_to_frame
from setups.base import BaseSetupDetector


class StaticDetector(BaseSetupDetector):
    def __init__(self, setups: list[SetupEvent]) -> None:
        self._setups = setups

    def detect(self, frame: pd.DataFrame) -> list[SetupEvent]:
        return list(self._setups)


def make_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30", periods=6, freq="min", tz="America/New_York")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["NQ"] * 6,
            "open": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5],
            "high": [100.4, 100.9, 101.6, 102.3, 103.2, 103.5],
            "low": [99.8, 100.2, 100.8, 101.1, 101.8, 102.2],
            "close": [100.2, 100.8, 101.4, 102.0, 102.8, 103.0],
        }
    )


def make_setup(frame: pd.DataFrame) -> SetupEvent:
    timestamp = frame.loc[2, "timestamp"]
    return SetupEvent(
        setup_id="setup-1",
        setup_name="orb",
        symbol="NQ",
        contract="NQH4",
        timestamp=timestamp.to_pydatetime(),
        session_date=timestamp.normalize().to_pydatetime(),
        direction=Side.LONG,
        status=SetupStatus.CANDIDATE,
        entry_reference=101.0,
        stop_reference=100.0,
        target_reference=103.0,
        features={"or_high": 100.9, "breakout_strength": 1.5},
        context={"first_liquidity_target": "prior_day_high", "risk_bucket": None, "ignored": [1, 2, 3]},
    )


def test_labeled_setups_to_frame_flattens_features_and_context() -> None:
    frame = make_frame()
    setup = make_setup(frame)
    labeled = ForwardSetupLabeler(LabelerConfig(horizon_bars=2)).label(frame, [setup])

    dataset = labeled_setups_to_frame(labeled)

    assert len(dataset) == 1
    row = dataset.iloc[0]
    assert row["setup_id"] == "setup-1"
    assert row["direction"] == "long"
    assert row["feature_or_high"] == 100.9
    assert row["feature_breakout_strength"] == 1.5
    assert row["context_first_liquidity_target"] == "prior_day_high"
    assert pd.isna(row["context_risk_bucket"])
    assert "context_ignored" not in dataset.columns
    assert row["label"] == 1
    assert row["realized_return"] == 2.0
    assert row["label_metadata_binary_label"] == 1


def test_build_setup_quality_dataset_uses_detector_and_labeler() -> None:
    frame = make_frame()
    detector = StaticDetector([make_setup(frame)])
    labeler = ForwardSetupLabeler(LabelerConfig(horizon_bars=2))

    dataset = build_setup_quality_dataset(frame, detector, labeler)

    assert len(dataset) == 1
    assert dataset.loc[0, "setup_name"] == "orb"
    assert dataset.loc[0, "target_r_multiple"] == 2.0
    assert dataset.loc[0, "walk_forward_fold"] == "2024Q1"


def test_add_walk_forward_columns_sorts_and_derives_periods() -> None:
    dataset = pd.DataFrame(
        {
            "setup_id": ["b", "a"],
            "setup_timestamp": [pd.Timestamp("2024-04-03 10:00", tz="America/New_York"), pd.Timestamp("2024-01-02 09:45", tz="America/New_York")],
        }
    )

    enriched = add_walk_forward_columns(dataset)

    assert list(enriched["setup_id"]) == ["a", "b"]
    assert list(enriched["setup_year_month"]) == ["2024-01", "2024-04"]
    assert list(enriched["walk_forward_fold"]) == ["2024Q1", "2024Q2"]
