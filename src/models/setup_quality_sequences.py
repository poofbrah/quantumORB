from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.schemas import LabeledSetup
from .setup_quality_dataset import add_walk_forward_columns, labeled_setups_to_frame


DEFAULT_SEQUENCE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "atr",
    "returns",
    "rolling_volatility",
    "rsi",
    "body_range_ratio",
    "or_high",
    "or_low",
    "distance_from_or_high",
    "distance_from_or_low",
    "breakout_strength",
    "trend_bias",
    "trend_spread",
    "recent_bullish_fvg_size",
    "recent_bearish_fvg_size",
)


@dataclass(slots=True)
class SetupSequenceDatasetConfig:
    lookback_bars: int = 30
    include_setup_bar: bool = True
    drop_incomplete_sequences: bool = True
    symbol_column: str = "symbol"
    timestamp_column: str = "timestamp"
    setup_timestamp_column: str = "setup_timestamp"
    sequence_columns: tuple[str, ...] = DEFAULT_SEQUENCE_COLUMNS
    dtype: str = "float32"


@dataclass(slots=True)
class SetupSequenceDataset:
    sequences: np.ndarray
    metadata: pd.DataFrame
    feature_columns: list[str]
    lookback_bars: int
    include_setup_bar: bool


def build_setup_quality_sequence_dataset(
    frame: pd.DataFrame,
    labeled_setups: list[LabeledSetup],
    config: SetupSequenceDatasetConfig | None = None,
) -> SetupSequenceDataset:
    settings = config or SetupSequenceDatasetConfig()
    metadata = labeled_setups_to_frame(labeled_setups)
    if metadata.empty:
        return SetupSequenceDataset(
            sequences=np.empty((0, settings.lookback_bars, 0), dtype=settings.dtype),
            metadata=metadata,
            feature_columns=[],
            lookback_bars=settings.lookback_bars,
            include_setup_bar=settings.include_setup_bar,
        )

    available_columns = [column for column in settings.sequence_columns if column in frame.columns]
    if not available_columns:
        raise ValueError("No requested sequence feature columns are present in the frame")

    ordered = frame.copy()
    ordered[settings.timestamp_column] = pd.to_datetime(ordered[settings.timestamp_column])
    ordered = ordered.sort_values([settings.symbol_column, settings.timestamp_column], kind="stable").reset_index(drop=True)
    grouped = {
        symbol: group.reset_index(drop=True)
        for symbol, group in ordered.groupby(settings.symbol_column, sort=False)
    }

    sequence_rows: list[np.ndarray] = []
    metadata_rows: list[pd.Series] = []
    effective_indexes: list[int] = []

    for idx, row in metadata.iterrows():
        symbol = row["symbol"]
        if symbol not in grouped:
            continue
        setup_frame = grouped[symbol]
        sequence = _extract_sequence_for_setup(setup_frame, row, available_columns, settings)
        if sequence is None:
            if settings.drop_incomplete_sequences:
                continue
            sequence = np.full((settings.lookback_bars, len(available_columns)), np.nan, dtype=settings.dtype)
        sequence_rows.append(sequence.astype(settings.dtype, copy=False))
        metadata_rows.append(row)
        effective_indexes.append(idx)

    if not sequence_rows:
        empty_metadata = metadata.iloc[0:0].copy()
        return SetupSequenceDataset(
            sequences=np.empty((0, settings.lookback_bars, len(available_columns)), dtype=settings.dtype),
            metadata=empty_metadata,
            feature_columns=available_columns,
            lookback_bars=settings.lookback_bars,
            include_setup_bar=settings.include_setup_bar,
        )

    sequence_array = np.stack(sequence_rows).astype(settings.dtype, copy=False)
    result_metadata = pd.DataFrame(metadata_rows).reset_index(drop=True)
    result_metadata = add_walk_forward_columns(result_metadata, timestamp_column=settings.setup_timestamp_column)
    result_metadata["sequence_length"] = settings.lookback_bars
    result_metadata["sequence_feature_count"] = len(available_columns)
    result_metadata["sequence_index"] = np.arange(len(result_metadata))
    return SetupSequenceDataset(
        sequences=sequence_array,
        metadata=result_metadata,
        feature_columns=available_columns,
        lookback_bars=settings.lookback_bars,
        include_setup_bar=settings.include_setup_bar,
    )


def save_setup_sequence_dataset(dataset: SetupSequenceDataset, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "sequence_metadata.parquet"
    sequences_path = output_dir / "sequence_data.npz"
    summary_path = output_dir / "sequence_summary.json"

    dataset.metadata.to_parquet(metadata_path, index=False)
    np.savez_compressed(
        sequences_path,
        sequences=dataset.sequences,
        feature_columns=np.array(dataset.feature_columns, dtype=object),
    )

    summary = {
        "sequence_count": int(dataset.sequences.shape[0]),
        "lookback_bars": dataset.lookback_bars,
        "feature_count": int(dataset.sequences.shape[2]) if dataset.sequences.ndim == 3 else 0,
        "feature_columns": list(dataset.feature_columns),
        "include_setup_bar": dataset.include_setup_bar,
        "metadata_path": str(metadata_path),
        "sequences_path": str(sequences_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "metadata_path": str(metadata_path),
        "sequences_path": str(sequences_path),
        "summary_path": str(summary_path),
    }


def _extract_sequence_for_setup(
    setup_frame: pd.DataFrame,
    setup_row: pd.Series,
    feature_columns: list[str],
    config: SetupSequenceDatasetConfig,
) -> np.ndarray | None:
    setup_timestamp = pd.Timestamp(setup_row[config.setup_timestamp_column])
    timestamp_series = pd.to_datetime(setup_frame[config.timestamp_column])
    if config.include_setup_bar:
        eligible = setup_frame.loc[timestamp_series <= setup_timestamp]
    else:
        eligible = setup_frame.loc[timestamp_series < setup_timestamp]
    if len(eligible) < config.lookback_bars:
        return None
    window = eligible.iloc[-config.lookback_bars:].copy()
    return window.loc[:, feature_columns].to_numpy(dtype=config.dtype, copy=True)
