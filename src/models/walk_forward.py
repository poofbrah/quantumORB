from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class WalkForwardSplit:
    fold_id: int
    train_period_start: str
    train_period_end: str
    test_period_start: str
    test_period_end: str
    train_indices: list[int]
    test_indices: list[int]


def build_walk_forward_splits(
    dataset: pd.DataFrame,
    *,
    timestamp_column: str = "setup_timestamp",
    frequency: str = "Q",
    train_periods: int = 4,
    test_periods: int = 1,
    step_periods: int = 1,
    min_train_rows: int = 1,
    min_test_rows: int = 1,
) -> list[WalkForwardSplit]:
    if train_periods < 1:
        raise ValueError("train_periods must be at least 1")
    if test_periods < 1:
        raise ValueError("test_periods must be at least 1")
    if step_periods < 1:
        raise ValueError("step_periods must be at least 1")
    if dataset.empty:
        return []
    if timestamp_column not in dataset.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")

    ordered = dataset.copy()
    ordered[timestamp_column] = pd.to_datetime(ordered[timestamp_column])
    ordered = ordered.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    period_index = ordered[timestamp_column].dt.tz_localize(None).dt.to_period(frequency)
    period_labels = period_index.astype(str)
    unique_periods = list(dict.fromkeys(period_labels.tolist()))

    required_window = train_periods + test_periods
    if len(unique_periods) < required_window:
        return []

    splits: list[WalkForwardSplit] = []
    fold_id = 1
    max_start = len(unique_periods) - required_window
    for start in range(0, max_start + 1, step_periods):
        train_window = unique_periods[start : start + train_periods]
        test_window = unique_periods[start + train_periods : start + required_window]

        train_mask = period_labels.isin(train_window)
        test_mask = period_labels.isin(test_window)
        train_indices = ordered.index[train_mask].tolist()
        test_indices = ordered.index[test_mask].tolist()

        if len(train_indices) < min_train_rows or len(test_indices) < min_test_rows:
            continue

        splits.append(
            WalkForwardSplit(
                fold_id=fold_id,
                train_period_start=train_window[0],
                train_period_end=train_window[-1],
                test_period_start=test_window[0],
                test_period_end=test_window[-1],
                train_indices=train_indices,
                test_indices=test_indices,
            )
        )
        fold_id += 1
    return splits


def summarize_walk_forward_splits(splits: list[WalkForwardSplit]) -> pd.DataFrame:
    rows = [
        {
            "fold_id": split.fold_id,
            "train_period_start": split.train_period_start,
            "train_period_end": split.train_period_end,
            "test_period_start": split.test_period_start,
            "test_period_end": split.test_period_end,
            "train_rows": len(split.train_indices),
            "test_rows": len(split.test_indices),
        }
        for split in splits
    ]
    return pd.DataFrame(rows)
