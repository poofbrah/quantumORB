from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from data.schemas import LabeledSetup, LabelSource, SetupEvent, Side
from .base import BaseSetupLabeler


class IntrabarConflictPolicy(str, Enum):
    STOP_FIRST = "stop_first"
    TARGET_FIRST = "target_first"


@dataclass(slots=True)
class QualityBucketRule:
    poor_threshold: float = -0.5
    good_threshold: float = 1.0
    great_threshold: float = 2.0

    def bucket(self, realized_r: float) -> str:
        if realized_r < self.poor_threshold:
            return "poor"
        if realized_r < self.good_threshold:
            return "average"
        if realized_r < self.great_threshold:
            return "good"
        return "great"


@dataclass(slots=True)
class LabelerConfig:
    horizon_bars: int = 20
    include_quality_bucket: bool = True
    neutral_if_unresolved: bool = False
    intrabar_conflict_policy: IntrabarConflictPolicy = IntrabarConflictPolicy.STOP_FIRST


class ForwardSetupLabeler(BaseSetupLabeler):
    """Forward-only setup labeler.

    v1 intrabar conflict policy is explicit and conservative by default: if both the
    target and stop are touched in the same future bar, resolve the bar according to
    `intrabar_conflict_policy`, which defaults to `STOP_FIRST`.
    """

    def __init__(
        self,
        config: LabelerConfig | None = None,
        quality_rule: QualityBucketRule | None = None,
    ) -> None:
        self.config = config or LabelerConfig()
        self.quality_rule = quality_rule or QualityBucketRule()

    def label(self, frame: pd.DataFrame, setups: list[SetupEvent]) -> list[LabeledSetup]:
        if not setups:
            return []

        ordered = frame.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)
        labeled: list[LabeledSetup] = []
        for setup in setups:
            future_bars = self._future_bars(ordered, setup)
            outcome = self._evaluate_outcome(future_bars, setup)
            metadata = {
                "binary_label": outcome["binary_label"],
                "regression_label": outcome["realized_r"],
                "quality_bucket": outcome["quality_bucket"],
                "target_hit_index": outcome["target_hit_index"],
                "stop_hit_index": outcome["stop_hit_index"],
                "resolved": outcome["resolved"],
                "intrabar_conflict_policy": self.config.intrabar_conflict_policy.value,
            }
            labeled.append(
                LabeledSetup(
                    setup=setup,
                    label=outcome["binary_label"],
                    label_name="target_before_stop",
                    label_source=LabelSource.RULE,
                    horizon_bars=self.config.horizon_bars,
                    realized_return=outcome["realized_r"],
                    realized_mae=outcome["mae_r"],
                    realized_mfe=outcome["mfe_r"],
                    quality_bucket=outcome["quality_bucket"],
                    metadata=metadata,
                )
            )
        return labeled

    def _future_bars(self, frame: pd.DataFrame, setup: SetupEvent) -> pd.DataFrame:
        mask = (frame["symbol"] == setup.symbol) & (frame["timestamp"] > pd.Timestamp(setup.timestamp))
        future = frame.loc[mask].head(self.config.horizon_bars).reset_index(drop=True)
        return future

    def _evaluate_outcome(self, future_bars: pd.DataFrame, setup: SetupEvent) -> dict[str, float | int | str | bool | None]:
        risk = abs(setup.entry_reference - setup.stop_reference)
        if risk == 0.0:
            raise ValueError("Labeling requires non-zero setup risk.")

        target_hit_index: int | None = None
        stop_hit_index: int | None = None
        for idx, row in future_bars.iterrows():
            if setup.direction is Side.LONG:
                target_hit = row["high"] >= setup.target_reference
                stop_hit = row["low"] <= setup.stop_reference
            else:
                target_hit = row["low"] <= setup.target_reference
                stop_hit = row["high"] >= setup.stop_reference

            if target_hit and stop_hit:
                if self.config.intrabar_conflict_policy is IntrabarConflictPolicy.TARGET_FIRST:
                    target_hit_index = idx
                else:
                    stop_hit_index = idx
                break
            if target_hit:
                target_hit_index = idx
                break
            if stop_hit:
                stop_hit_index = idx
                break

        resolved = target_hit_index is not None or stop_hit_index is not None
        binary_label = self._binary_label(target_hit_index, stop_hit_index, resolved)
        realized_r = self._realized_r(future_bars, setup, target_hit_index, stop_hit_index)
        mfe_r, mae_r = self._excursions(future_bars, setup, risk)
        quality_bucket = self.quality_rule.bucket(realized_r) if self.config.include_quality_bucket else None

        return {
            "binary_label": binary_label,
            "realized_r": realized_r,
            "mfe_r": mfe_r,
            "mae_r": mae_r,
            "quality_bucket": quality_bucket,
            "target_hit_index": target_hit_index,
            "stop_hit_index": stop_hit_index,
            "resolved": resolved,
        }

    def _binary_label(self, target_hit_index: int | None, stop_hit_index: int | None, resolved: bool) -> int:
        if target_hit_index is not None and stop_hit_index is None:
            return 1
        if target_hit_index is None and stop_hit_index is not None:
            return 0
        if target_hit_index is not None and stop_hit_index is not None:
            return 0 if target_hit_index >= stop_hit_index else 1
        if not resolved and self.config.neutral_if_unresolved:
            return -1
        return 0

    def _realized_r(
        self,
        future_bars: pd.DataFrame,
        setup: SetupEvent,
        target_hit_index: int | None,
        stop_hit_index: int | None,
    ) -> float:
        risk = abs(setup.entry_reference - setup.stop_reference)
        if target_hit_index is not None and (stop_hit_index is None or target_hit_index < stop_hit_index):
            return self._signed_r(setup, setup.target_reference, risk)
        if stop_hit_index is not None:
            return self._signed_r(setup, setup.stop_reference, risk)
        if future_bars.empty:
            return 0.0
        terminal_price = float(future_bars.iloc[-1]["close"])
        return self._signed_r(setup, terminal_price, risk)

    def _excursions(self, future_bars: pd.DataFrame, setup: SetupEvent, risk: float) -> tuple[float, float]:
        if future_bars.empty:
            return 0.0, 0.0
        if setup.direction is Side.LONG:
            mfe = (future_bars["high"].max() - setup.entry_reference) / risk
            mae = (future_bars["low"].min() - setup.entry_reference) / risk
            return float(mfe), float(mae)
        mfe = (setup.entry_reference - future_bars["low"].min()) / risk
        mae = (setup.entry_reference - future_bars["high"].max()) / risk
        return float(mfe), float(mae)

    def _signed_r(self, setup: SetupEvent, price: float, risk: float) -> float:
        if setup.direction is Side.LONG:
            return round(float((price - setup.entry_reference) / risk), 10)
        return round(float((setup.entry_reference - price) / risk), 10)

