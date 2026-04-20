from __future__ import annotations

import pandas as pd

from models.walk_forward import build_walk_forward_splits, summarize_walk_forward_splits


def make_dataset() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2023-01-05 09:45",
            "2023-02-06 09:45",
            "2023-04-10 09:45",
            "2023-05-11 09:45",
            "2023-07-12 09:45",
            "2023-08-14 09:45",
            "2023-10-02 09:45",
            "2023-11-03 09:45",
            "2024-01-08 09:45",
            "2024-02-09 09:45",
        ],
        utc=True,
    ).tz_convert("America/New_York")
    return pd.DataFrame(
        {
            "setup_id": [f"s{i}" for i in range(len(timestamps))],
            "setup_timestamp": timestamps,
            "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )


def test_build_walk_forward_splits_returns_time_ordered_quarterly_folds() -> None:
    dataset = make_dataset()

    splits = build_walk_forward_splits(dataset, train_periods=2, test_periods=1, frequency="Q")

    assert len(splits) == 3
    assert splits[0].train_period_start == "2023Q1"
    assert splits[0].train_period_end == "2023Q2"
    assert splits[0].test_period_start == "2023Q3"
    assert splits[0].test_period_end == "2023Q3"
    assert max(splits[0].train_indices) < min(splits[0].test_indices)
    assert splits[1].train_period_start == "2023Q2"
    assert splits[1].test_period_start == "2023Q4"
    assert splits[2].test_period_start == "2024Q1"


def test_build_walk_forward_splits_can_use_monthly_frequency() -> None:
    dataset = make_dataset()

    splits = build_walk_forward_splits(dataset, train_periods=3, test_periods=2, step_periods=2, frequency="M")

    assert len(splits) >= 1
    assert splits[0].train_period_start == "2023-01"
    assert splits[0].train_period_end == "2023-04"
    assert splits[0].test_period_start == "2023-05"


def test_summarize_walk_forward_splits_returns_fold_frame() -> None:
    dataset = make_dataset()
    splits = build_walk_forward_splits(dataset, train_periods=2, test_periods=1, frequency="Q")

    summary = summarize_walk_forward_splits(splits)

    assert list(summary.columns) == [
        "fold_id",
        "train_period_start",
        "train_period_end",
        "test_period_start",
        "test_period_end",
        "train_rows",
        "test_rows",
    ]
    assert len(summary) == len(splits)
    assert summary.loc[0, "train_rows"] > 0
    assert summary.loc[0, "test_rows"] > 0
