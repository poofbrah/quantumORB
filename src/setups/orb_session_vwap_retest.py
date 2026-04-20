from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from uuid import uuid4

import numpy as np
import pandas as pd

from data.schemas import SetupEvent, SetupStatus, Side
from .base import BaseSetupDetector
from .liquidity import LiquidityLevel, add_liquidity_levels, select_first_liquidity_target

TARGET_MODE_FIXED_R = "fixed_r"
TARGET_MODE_LIQUIDITY = "liquidity"
SUPPORTED_TARGET_MODES = {TARGET_MODE_FIXED_R, TARGET_MODE_LIQUIDITY}
DEFAULT_LIQUIDITY_PRIORITY = (
    LiquidityLevel.PDH.value,
    LiquidityLevel.PDL.value,
    LiquidityLevel.H4_HIGH.value,
    LiquidityLevel.H4_LOW.value,
    LiquidityLevel.DAY_HIGH.value,
    LiquidityLevel.DAY_LOW.value,
    LiquidityLevel.LONDON_HIGH.value,
    LiquidityLevel.LONDON_LOW.value,
)


@dataclass(slots=True)
class ORBSessionVWAPRetestConfig:
    opening_range_start_time: str = "09:30"
    opening_range_end_time: str = "09:45"
    opening_range_minutes: int = 15
    latest_entry_time: str | None = "11:30"
    max_trades_per_session: int = 1
    enable_long: bool = True
    enable_short: bool = True
    target_r_multiple: float = 2.0
    target_mode: str = TARGET_MODE_FIXED_R
    liquidity_target_priority: tuple[str, ...] = DEFAULT_LIQUIDITY_PRIORITY
    stop_mode: str = "or_boundary"
    allowed_vwap_columns: tuple[str, ...] = ("vwap_rth", "vwap_eth", "vwap")
    setup_name: str = "orb_session_vwap_retest"
    timezone: str = "America/New_York"
    allowed_trade_windows: tuple[str, ...] = field(default_factory=tuple)
    require_trend_alignment: bool = False
    trend_bias_column: str = "trend_bias"
    trend_spread_column: str = "trend_spread"
    require_fvg_context: bool = False
    bullish_fvg_column: str = "recent_bullish_fvg_size"
    bearish_fvg_column: str = "recent_bearish_fvg_size"


class ORBSessionVWAPRetestDetector(BaseSetupDetector):
    def __init__(self, config: ORBSessionVWAPRetestConfig | None = None) -> None:
        self.config = config or ORBSessionVWAPRetestConfig()
        if self.config.target_mode not in SUPPORTED_TARGET_MODES:
            raise ValueError(f"Unsupported target_mode: {self.config.target_mode}")

    def detect(self, frame: pd.DataFrame) -> list[SetupEvent]:
        if frame.empty:
            return []
        required = {"timestamp", "symbol", "session_date", "open", "high", "low", "close", "or_high", "or_low"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"ORB session VWAP retest detection requires columns: {sorted(missing)}")

        ordered = add_liquidity_levels(frame, timezone=self.config.timezone)
        ordered = ordered.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)
        setups: list[SetupEvent] = []
        for _, session_frame in ordered.groupby(["symbol", "session_date"], sort=False):
            setups.extend(self._detect_session(session_frame.reset_index(drop=True)))
        return setups

    def _detect_session(self, session_frame: pd.DataFrame) -> list[SetupEvent]:
        detected: list[SetupEvent] = []
        long_breakout_time: pd.Timestamp | None = None
        short_breakout_time: pd.Timestamp | None = None

        for idx in range(len(session_frame)):
            row = session_frame.iloc[idx]
            if pd.isna(row.get("or_high")) or pd.isna(row.get("or_low")):
                continue

            local_ts = pd.Timestamp(row.get("local_timestamp", row["timestamp"]))
            if not self._is_time_allowed(local_ts):
                continue

            if self.config.enable_long and float(row["close"]) > float(row["or_high"]) and long_breakout_time is None:
                long_breakout_time = pd.Timestamp(row["timestamp"])
            if self.config.enable_short and float(row["close"]) < float(row["or_low"]) and short_breakout_time is None:
                short_breakout_time = pd.Timestamp(row["timestamp"])

            touched_vwap = self._touched_session_vwap(row)
            if touched_vwap is None:
                continue
            vwap_name, vwap_price = touched_vwap

            current_ts = pd.Timestamp(row["timestamp"])
            if self.config.enable_long and long_breakout_time is not None and current_ts > long_breakout_time:
                if float(row["close"]) >= vwap_price and self._supports_direction(row, Side.LONG):
                    detected.append(self._build_setup(row, Side.LONG, long_breakout_time, vwap_name, vwap_price))
                    if len(detected) >= self.config.max_trades_per_session:
                        break
            if self.config.enable_short and short_breakout_time is not None and current_ts > short_breakout_time:
                if float(row["close"]) <= vwap_price and self._supports_direction(row, Side.SHORT):
                    detected.append(self._build_setup(row, Side.SHORT, short_breakout_time, vwap_name, vwap_price))
                    if len(detected) >= self.config.max_trades_per_session:
                        break
        return detected

    def _build_setup(
        self,
        row: pd.Series,
        direction: Side,
        breakout_time: pd.Timestamp,
        vwap_name: str,
        vwap_price: float,
    ) -> SetupEvent:
        entry_reference = float(row["close"])
        stop_reference = float(row["or_low"] if direction is Side.LONG else row["or_high"])
        risk = abs(entry_reference - stop_reference)
        target_name, target_reference = self._resolve_target(row, direction, entry_reference, stop_reference)
        zone_mid = float((row["or_high"] + row["or_low"]) / 2.0)
        range_width = float(row["or_high"] - row["or_low"])
        trend_bias = self._float_or_none(row.get(self.config.trend_bias_column))
        trend_spread = self._float_or_none(row.get(self.config.trend_spread_column))
        recent_bullish_fvg = self._float_or_none(row.get(self.config.bullish_fvg_column))
        recent_bearish_fvg = self._float_or_none(row.get(self.config.bearish_fvg_column))
        return SetupEvent(
            setup_id=f"orb-vwap-{uuid4().hex[:12]}",
            setup_name=self.config.setup_name,
            symbol=str(row["symbol"]),
            contract=str(row.get("contract", row["symbol"])),
            timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
            session_date=pd.Timestamp(row["session_date"]).to_pydatetime(),
            direction=direction,
            status=SetupStatus.CANDIDATE,
            entry_reference=entry_reference,
            stop_reference=stop_reference,
            target_reference=float(target_reference),
            features={
                "or_high": float(row["or_high"]),
                "or_low": float(row["or_low"]),
                "or_mid": zone_mid,
                "range_width": range_width,
                "range_width_atr": self._float_or_none(row.get("or_width_atr")),
                "retest_vwap_price": vwap_price,
                "distance_to_vwap": abs(entry_reference - vwap_price),
                "trend_bias": trend_bias,
                "trend_spread": trend_spread,
                "close_vs_fast_ema": self._float_or_none(row.get("close_vs_fast_ema")),
                "recent_bullish_fvg_size": recent_bullish_fvg,
                "recent_bearish_fvg_size": recent_bearish_fvg,
                "recent_directional_fvg_size": recent_bullish_fvg if direction is Side.LONG else recent_bearish_fvg,
                "distance_to_pdh": self._distance_to_level(row, entry_reference, "pdh"),
                "distance_to_pdl": self._distance_to_level(row, entry_reference, "pdl"),
                "distance_to_h4_high": self._distance_to_level(row, entry_reference, "h4_high"),
                "distance_to_h4_low": self._distance_to_level(row, entry_reference, "h4_low"),
                "distance_to_day_high": self._distance_to_level(row, entry_reference, "day_high"),
                "distance_to_day_low": self._distance_to_level(row, entry_reference, "day_low"),
                "distance_to_london_high": self._distance_to_level(row, entry_reference, "london_high"),
                "distance_to_london_low": self._distance_to_level(row, entry_reference, "london_low"),
                "target_mode": self.config.target_mode,
                "target_name": target_name,
            },
            context={
                "strategy_profile": self.config.setup_name,
                "breakout_time": breakout_time.isoformat(),
                "retest_vwap_name": vwap_name,
                "retest_vwap_price": vwap_price,
                "entry_family": "orb_session_vwap_retest",
                "stop_mode": self.config.stop_mode,
                "target_rule": self.config.target_mode,
                "target_r_multiple": self.config.target_r_multiple,
                "liquidity_target_priority": list(self.config.liquidity_target_priority),
                "target_name": target_name,
                "target_source": "liquidity" if self.config.target_mode == TARGET_MODE_LIQUIDITY and target_name != f"{TARGET_MODE_FIXED_R}_{self.config.target_r_multiple}R" else "fixed_r",
                "trend_alignment_required": self.config.require_trend_alignment,
                "fvg_context_required": self.config.require_fvg_context,
            },
        )

    def _resolve_target(
        self,
        row: pd.Series,
        direction: Side,
        entry_reference: float,
        stop_reference: float,
    ) -> tuple[str, float]:
        fixed_target = self._fixed_r_target(entry_reference, stop_reference, direction)
        fixed_name = f"{TARGET_MODE_FIXED_R}_{self.config.target_r_multiple}R"
        if self.config.target_mode != TARGET_MODE_LIQUIDITY:
            return fixed_name, fixed_target

        target_name, liquidity_target = select_first_liquidity_target(
            row,
            direction.value,
            self.config.liquidity_target_priority,
        )
        if target_name is None or liquidity_target is None:
            return fixed_name, fixed_target
        return target_name, float(liquidity_target)

    def _fixed_r_target(self, entry_reference: float, stop_reference: float, direction: Side) -> float:
        risk = abs(entry_reference - stop_reference)
        if direction is Side.LONG:
            return entry_reference + (risk * self.config.target_r_multiple)
        return entry_reference - (risk * self.config.target_r_multiple)

    def _touched_session_vwap(self, row: pd.Series) -> tuple[str, float] | None:
        low = float(row["low"])
        high = float(row["high"])
        for column in self.config.allowed_vwap_columns:
            if column not in row.index or pd.isna(row.get(column)):
                continue
            value = float(row.get(column))
            if low <= value <= high:
                return column, value
        return None

    def _supports_direction(self, row: pd.Series, direction: Side) -> bool:
        if self.config.require_trend_alignment and not self._trend_allows_direction(row, direction):
            return False
        if self.config.require_fvg_context and not self._fvg_allows_direction(row, direction):
            return False
        return True

    def _trend_allows_direction(self, row: pd.Series, direction: Side) -> bool:
        bias = row.get(self.config.trend_bias_column)
        if pd.isna(bias):
            return False
        return (direction is Side.LONG and float(bias) > 0) or (direction is Side.SHORT and float(bias) < 0)

    def _fvg_allows_direction(self, row: pd.Series, direction: Side) -> bool:
        column = self.config.bullish_fvg_column if direction is Side.LONG else self.config.bearish_fvg_column
        value = row.get(column)
        return not pd.isna(value) and float(value) > 0.0

    def _distance_to_level(self, row: pd.Series, entry_reference: float, column: str) -> float | None:
        value = row.get(column)
        if pd.isna(value):
            return None
        return float(value) - entry_reference

    def _float_or_none(self, value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        result = float(value)
        if np.isnan(result):
            return None
        return result

    def _is_time_allowed(self, timestamp: pd.Timestamp) -> bool:
        current_time = timestamp.time()
        if self.config.latest_entry_time is not None and current_time > time.fromisoformat(self.config.latest_entry_time):
            return False
        if self.config.allowed_trade_windows:
            for window in self.config.allowed_trade_windows:
                start, end = window.split("-")
                if time.fromisoformat(start) <= current_time <= time.fromisoformat(end):
                    return True
            return False
        return True
