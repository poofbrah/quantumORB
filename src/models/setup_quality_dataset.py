from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

import pandas as pd

from data.schemas import LabeledSetup, SetupEvent
from labeling.base import BaseSetupLabeler
from setups.base import BaseSetupDetector


@dataclass(slots=True)
class SetupQualityDatasetConfig:
    feature_prefix: str = "feature_"
    context_prefix: str = "context_"
    metadata_prefix: str = "label_metadata_"
    include_context: bool = True
    include_label_metadata: bool = True
    include_null_scalars: bool = True


SCALAR_TYPES = (str, int, float, bool, type(None), pd.Timestamp, datetime, date)


def build_setup_quality_dataset(
    frame: pd.DataFrame,
    detector: BaseSetupDetector,
    labeler: BaseSetupLabeler,
    config: SetupQualityDatasetConfig | None = None,
) -> pd.DataFrame:
    settings = config or SetupQualityDatasetConfig()
    setups = detector.detect(frame)
    labeled_setups = labeler.label(frame, setups)
    return labeled_setups_to_frame(labeled_setups, config=settings)


def labeled_setups_to_frame(
    labeled_setups: list[LabeledSetup],
    config: SetupQualityDatasetConfig | None = None,
) -> pd.DataFrame:
    settings = config or SetupQualityDatasetConfig()
    rows = [flatten_labeled_setup(item, config=settings) for item in labeled_setups]
    dataset = pd.DataFrame(rows)
    if dataset.empty:
        return dataset
    return add_walk_forward_columns(dataset)


def flatten_labeled_setup(
    labeled_setup: LabeledSetup,
    config: SetupQualityDatasetConfig | None = None,
) -> dict[str, Any]:
    settings = config or SetupQualityDatasetConfig()
    setup = labeled_setup.setup
    row: dict[str, Any] = {
        "setup_id": setup.setup_id,
        "setup_name": setup.setup_name,
        "symbol": setup.symbol,
        "contract": setup.contract,
        "setup_timestamp": pd.Timestamp(setup.timestamp),
        "session_date": pd.Timestamp(setup.session_date),
        "direction": _normalize_scalar(setup.direction),
        "status": _normalize_scalar(setup.status),
        "entry_reference": setup.entry_reference,
        "stop_reference": setup.stop_reference,
        "target_reference": setup.target_reference,
        "risk_points": abs(setup.entry_reference - setup.stop_reference),
        "reward_points": abs(setup.target_reference - setup.entry_reference),
        "target_r_multiple": _safe_r_multiple(setup),
        "label": labeled_setup.label,
        "label_name": labeled_setup.label_name,
        "label_source": _normalize_scalar(labeled_setup.label_source),
        "horizon_bars": labeled_setup.horizon_bars,
        "realized_return": labeled_setup.realized_return,
        "realized_mae": labeled_setup.realized_mae,
        "realized_mfe": labeled_setup.realized_mfe,
        "quality_bucket": labeled_setup.quality_bucket,
    }

    row.update(_flatten_mapping(setup.features, settings.feature_prefix, settings.include_null_scalars))
    if settings.include_context:
        row.update(_flatten_mapping(setup.context, settings.context_prefix, settings.include_null_scalars))
    if settings.include_label_metadata:
        row.update(_flatten_mapping(labeled_setup.metadata, settings.metadata_prefix, settings.include_null_scalars))
    return row


def add_walk_forward_columns(dataset: pd.DataFrame, timestamp_column: str = "setup_timestamp") -> pd.DataFrame:
    if dataset.empty:
        return dataset.copy()
    enriched = dataset.copy()
    enriched[timestamp_column] = pd.to_datetime(enriched[timestamp_column])
    enriched = enriched.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    timestamps = pd.to_datetime(enriched[timestamp_column])
    enriched["setup_date"] = timestamps.dt.date
    enriched["setup_year"] = timestamps.dt.year
    enriched["setup_month"] = timestamps.dt.month
    enriched["setup_year_month"] = timestamps.dt.strftime("%Y-%m")
    enriched["walk_forward_fold"] = timestamps.dt.tz_localize(None).dt.to_period("Q").astype(str)
    return enriched


def _flatten_mapping(mapping: dict[str, Any], prefix: str, include_null_scalars: bool) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        if not _is_scalar(value):
            continue
        if value is None and not include_null_scalars:
            continue
        flattened[f"{prefix}{key}"] = _normalize_scalar(value)
    return flattened


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.Timestamp):
        return value
    return value


def _is_scalar(value: Any) -> bool:
    if isinstance(value, Enum):
        return True
    return isinstance(value, SCALAR_TYPES)


def _safe_r_multiple(setup: SetupEvent) -> float | None:
    risk = abs(setup.entry_reference - setup.stop_reference)
    if risk == 0.0:
        return None
    reward = abs(setup.target_reference - setup.entry_reference)
    return reward / risk

