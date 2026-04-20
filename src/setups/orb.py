from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import time
from uuid import uuid4

import pandas as pd

from config.models import StrategyConfig
from data.schemas import SetupEvent, SetupStatus, Side
from .base import BaseSetupDetector
from .liquidity import add_liquidity_levels, select_first_liquidity_target, select_runner_targets
from .specification import ORBStrategySpec, strategy_spec_from_config

RAW_SETUP_COLUMNS = {
    "timestamp",
    "local_timestamp",
    "symbol",
    "contract",
    "session_date",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
}
EXCLUDED_SNAPSHOT_COLUMNS = {"entry_reference", "stop_reference", "target_reference"}
ADVANCED_FORMAL_ONLY_FIELDS = {
    "liquidity_targeting",
    "partial_take_profit",
    "breakeven_management",
    "runner_target_sequence",
    "runner_trail",
}
DIAGNOSTIC_KEYS = (
    "candidate_windows_scanned",
    "breakout_candidates",
    "setups_emitted",
    "breakout_not_achieved",
    "wick_only_breakout",
    "failed_strong_close_threshold",
    "failed_retest_requirement",
    "failed_latest_entry_cutoff",
    "failed_trend_filter",
    "failed_volatility_filter",
    "failed_minimum_rr_threshold",
    "failed_liquidity_target_lookup",
    "failed_stop_calculation",
    "duplicate_same_side_session_suppression",
)


@dataclass(slots=True)
class ORBConfig:
    opening_range_minutes: int = 5
    opening_range_start_time: str = "09:30"
    opening_range_end_time: str = "09:35"
    latest_entry_time: str | None = None
    max_trades_per_session: int = 1
    enable_long: bool = True
    enable_short: bool = True
    require_breakout_close: bool = True
    require_retest: bool = False
    min_breakout_strength: float = 0.0
    trend_filter: str = "none"
    trend_column: str = "ema_20"
    volatility_filter_enabled: bool = False
    min_or_width_atr: float | None = None
    max_or_width_atr: float | None = None
    displacement_rule: str = "none"
    strong_close_min_body_range_ratio: float | None = None
    strong_close_min_boundary_distance: float | None = None
    strong_close_min_boundary_distance_atr: float | None = None
    invalidate_on_early_counter_liquidity_consumption: bool = False
    allow_retest_rescue_after_early_liquidity_consumption: bool = False
    stop_mode: str = "or_boundary"
    formal_stop_rule: str = "or_boundary"
    stop_atr_multiple: float = 1.0
    stop_buffer_points: float | None = None
    stop_buffer_atr_multiple: float | None = None
    wick_reference_mode: str | None = None
    fallback_stop_mode: str | None = "or_boundary"
    target_r_multiple: float = 2.0
    target_rule: str = "r_multiple"
    liquidity_target_mode: str = "none"
    liquidity_target_priority: tuple[str, ...] = field(default_factory=tuple)
    first_draw_target_rule: str | None = None
    partial_take_profit_rule: str | None = None
    partial_take_profit_fraction: float | None = None
    runner_target_rule: str | None = None
    minimum_rr_threshold: float | None = None
    breakeven_after_first_draw: bool = False
    runner_trail_rule: str | None = None
    runner_trail_atr_multiple: float | None = None
    setup_feature_whitelist: tuple[str, ...] | None = None
    instrument: str = "ES"
    session_name: str = "rth"
    session_start: str = "09:30"
    session_end: str = "16:00"
    ny_open_anchored: bool = False
    allowed_trade_windows: tuple[str, ...] = field(default_factory=tuple)
    allowed_days_of_week: tuple[str, ...] = field(default_factory=tuple)
    news_skip_rules: tuple[str, ...] = field(default_factory=tuple)
    discretionary_skip_reasons: tuple[str, ...] = field(default_factory=tuple)
    formalized_only_fields: tuple[str, ...] = field(default_factory=tuple)
    timezone: str = "America/New_York"

    @classmethod
    def from_strategy_config(cls, config: StrategyConfig) -> "ORBConfig":
        return cls.from_strategy_spec(strategy_spec_from_config(config))

    @classmethod
    def from_strategy_spec(cls, spec: ORBStrategySpec) -> "ORBConfig":
        operational_only: list[str] = []
        if spec.liquidity.mode != "none":
            operational_only.append("liquidity_targeting")
        if spec.management.partial_take_profit_rule is not None:
            operational_only.append("partial_take_profit")
        if spec.management.breakeven_rule is not None:
            operational_only.append("breakeven_management")
        if spec.management.runner_target_rule is not None:
            operational_only.append("runner_target_sequence")
        if spec.management.runner_trail_rule is not None:
            operational_only.append("runner_trail")

        return cls(
            opening_range_minutes=spec.opening_range_minutes,
            opening_range_start_time=spec.opening_range_start_time,
            opening_range_end_time=spec.opening_range_end_time,
            latest_entry_time=spec.latest_entry_time,
            max_trades_per_session=spec.max_trades_per_session,
            enable_long=spec.enable_long,
            enable_short=spec.enable_short,
            require_breakout_close=spec.breakout_confirmation_rule == "close_beyond_level",
            require_retest=spec.require_retest,
            min_breakout_strength=spec.min_breakout_strength,
            trend_filter=spec.trend_filter,
            trend_column=spec.trend_column or "ema_20",
            volatility_filter_enabled=spec.volatility_filter_enabled,
            min_or_width_atr=spec.min_or_width_atr,
            max_or_width_atr=spec.max_or_width_atr,
            displacement_rule=spec.displacement.rule,
            strong_close_min_body_range_ratio=spec.displacement.strong_close_min_body_range_ratio,
            strong_close_min_boundary_distance=spec.displacement.strong_close_min_boundary_distance,
            strong_close_min_boundary_distance_atr=spec.displacement.strong_close_min_boundary_distance_atr,
            invalidate_on_early_counter_liquidity_consumption=spec.displacement.invalidate_on_early_counter_liquidity_consumption,
            allow_retest_rescue_after_early_liquidity_consumption=spec.displacement.allow_retest_rescue_after_early_liquidity_consumption,
            stop_mode=spec.stop.rule,
            formal_stop_rule=spec.stop.rule,
            stop_atr_multiple=spec.stop.stop_atr_multiple or 1.0,
            stop_buffer_points=spec.stop.stop_buffer_points,
            stop_buffer_atr_multiple=spec.stop.stop_buffer_atr_multiple,
            wick_reference_mode=spec.stop.wick_reference_mode,
            fallback_stop_mode=spec.stop.fallback_stop_mode,
            target_r_multiple=spec.target_r_multiple or 2.0,
            target_rule=spec.target_rule,
            liquidity_target_mode=spec.liquidity.mode,
            liquidity_target_priority=spec.liquidity.priority,
            first_draw_target_rule=spec.management.first_draw_target_rule,
            partial_take_profit_rule=spec.management.partial_take_profit_rule,
            partial_take_profit_fraction=spec.management.partial_take_profit_fraction,
            runner_target_rule=spec.management.runner_target_rule,
            minimum_rr_threshold=spec.management.minimum_rr_threshold,
            breakeven_after_first_draw=spec.management.breakeven_after_first_draw,
            runner_trail_rule=spec.management.runner_trail_rule,
            runner_trail_atr_multiple=spec.management.runner_trail_atr_multiple,
            setup_feature_whitelist=spec.setup_feature_whitelist,
            instrument=spec.instrument,
            session_name=spec.session_name,
            session_start=spec.session_start,
            session_end=spec.session_end,
            ny_open_anchored=spec.ny_open_anchored,
            allowed_trade_windows=spec.allowed_trade_windows,
            allowed_days_of_week=spec.allowed_days_of_week,
            news_skip_rules=spec.news_skip_rules,
            discretionary_skip_reasons=spec.discretionary_skip_reasons,
            formalized_only_fields=tuple(field for field in operational_only if field not in ADVANCED_FORMAL_ONLY_FIELDS),
            timezone=spec.session_timezone,
        )


@dataclass(slots=True)
class ORBDetectionDiagnostics:
    counts: dict[str, int]
    audit_frame: pd.DataFrame


class ORBSetupDetector(BaseSetupDetector):
    def __init__(self, config: ORBConfig | None = None) -> None:
        self.config = config or ORBConfig()

    def detect(self, frame: pd.DataFrame) -> list[SetupEvent]:
        setups, _ = self.detect_with_diagnostics(frame)
        return setups

    def detect_with_diagnostics(self, frame: pd.DataFrame) -> tuple[list[SetupEvent], ORBDetectionDiagnostics]:
        if frame.empty:
            return [], ORBDetectionDiagnostics(counts={key: 0 for key in DIAGNOSTIC_KEYS}, audit_frame=pd.DataFrame())

        required_columns = {"timestamp", "symbol", "session_date", "close", "high", "low", "or_high", "or_low", "or_width", "open"}
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"ORB detection requires columns: {sorted(missing)}")

        ordered = add_liquidity_levels(frame, timezone=self.config.timezone)
        ordered = ordered.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)
        setups: list[SetupEvent] = []
        audit_rows: list[dict[str, object]] = []
        counts = Counter({key: 0 for key in DIAGNOSTIC_KEYS})
        for _, session_frame in ordered.groupby(["symbol", "session_date"], sort=False):
            session_setups, session_audit, session_counts = self._detect_session(session_frame.reset_index(drop=True))
            setups.extend(session_setups)
            audit_rows.extend(session_audit)
            counts.update(session_counts)
        return setups, ORBDetectionDiagnostics(counts=dict(counts), audit_frame=pd.DataFrame(audit_rows))

    def _detect_session(self, session_frame: pd.DataFrame) -> tuple[list[SetupEvent], list[dict[str, object]], Counter]:
        detected: list[SetupEvent] = []
        audit_rows: list[dict[str, object]] = []
        counts = Counter({key: 0 for key in DIAGNOSTIC_KEYS})
        long_emitted = False
        short_emitted = False

        for idx in range(len(session_frame)):
            row = session_frame.iloc[idx]
            if pd.isna(row.get("or_high")) or pd.isna(row.get("or_low")):
                continue

            counts["candidate_windows_scanned"] += 1
            history = session_frame.iloc[: idx + 1]
            breakout_side, body_close_outside, wick_only = self._classify_breakout(row)
            latest_entry_pass = self._is_time_allowed(row)
            strong_close_pass = None
            retest_pass = None
            trend_pass = None
            volatility_pass = None
            rr_estimate = None
            emitted = False
            rejection_reason = None

            if not latest_entry_pass:
                counts["failed_latest_entry_cutoff"] += 1
                rejection_reason = "failed_latest_entry_cutoff"
            elif breakout_side is None:
                if wick_only:
                    counts["wick_only_breakout"] += 1
                    rejection_reason = "wick_only_breakout"
                else:
                    counts["breakout_not_achieved"] += 1
                    rejection_reason = "breakout_not_achieved"
            else:
                counts["breakout_candidates"] += 1
                if (breakout_side is Side.LONG and long_emitted) or (breakout_side is Side.SHORT and short_emitted):
                    counts["duplicate_same_side_session_suppression"] += 1
                    rejection_reason = "duplicate_same_side_session_suppression"
                elif len(detected) >= self.config.max_trades_per_session:
                    counts["duplicate_same_side_session_suppression"] += 1
                    rejection_reason = "duplicate_same_side_session_suppression"
                else:
                    strong_close_pass = self._passes_strong_close(row, breakout_side)
                    retest_pass = self._passes_retest(row, breakout_side)
                    trend_pass = self._passes_trend_filter(row, breakout_side)
                    volatility_pass = self._passes_volatility_filter(row)
                    if not strong_close_pass:
                        counts["failed_strong_close_threshold"] += 1
                        rejection_reason = "failed_strong_close_threshold"
                    elif not retest_pass:
                        counts["failed_retest_requirement"] += 1
                        rejection_reason = "failed_retest_requirement"
                    elif not trend_pass:
                        counts["failed_trend_filter"] += 1
                        rejection_reason = "failed_trend_filter"
                    elif not volatility_pass:
                        counts["failed_volatility_filter"] += 1
                        rejection_reason = "failed_volatility_filter"
                    else:
                        setup, setup_meta = self._prepare_setup(history, breakout_side)
                        rr_estimate = setup_meta.get("rr_estimate")
                        rejection_reason = setup_meta.get("rejection_reason")
                        if rejection_reason == "failed_minimum_rr_threshold":
                            counts["failed_minimum_rr_threshold"] += 1
                        elif rejection_reason == "failed_liquidity_target_lookup":
                            counts["failed_liquidity_target_lookup"] += 1
                        elif rejection_reason == "failed_stop_calculation":
                            counts["failed_stop_calculation"] += 1
                        if setup is not None:
                            detected.append(setup)
                            emitted = True
                            counts["setups_emitted"] += 1
                            if breakout_side is Side.LONG:
                                long_emitted = True
                            else:
                                short_emitted = True

            audit_rows.append(
                self._build_audit_row(
                    row=row,
                    breakout_side=breakout_side,
                    body_close_outside=body_close_outside,
                    latest_entry_pass=latest_entry_pass,
                    strong_close_pass=strong_close_pass,
                    retest_pass=retest_pass,
                    trend_pass=trend_pass,
                    volatility_pass=volatility_pass,
                    rr_estimate=rr_estimate,
                    emitted=emitted,
                    rejection_reason=rejection_reason,
                )
            )
        return detected, audit_rows, counts

    def _classify_breakout(self, row: pd.Series) -> tuple[Side | None, bool, bool]:
        if float(row["close"]) > float(row["or_high"]):
            return Side.LONG, True, False
        if float(row["close"]) < float(row["or_low"]):
            return Side.SHORT, True, False
        if float(row["high"]) > float(row["or_high"]):
            return None, False, True
        if float(row["low"]) < float(row["or_low"]):
            return None, False, True
        return None, False, False

    def _passes_strong_close(self, row: pd.Series, direction: Side) -> bool:
        if self.config.displacement_rule == "none":
            return True
        candle_range = float(row.get("candle_range", row["high"] - row["low"]))
        candle_body = float(row.get("candle_body", abs(row["close"] - row["open"])))
        ratio = (candle_body / candle_range) if candle_range > 0 else 0.0
        if self.config.strong_close_min_body_range_ratio is not None and ratio < self.config.strong_close_min_body_range_ratio:
            return False
        boundary_distance = float(row["close"] - row["or_high"]) if direction is Side.LONG else float(row["or_low"] - row["close"])
        if self.config.strong_close_min_boundary_distance is not None and boundary_distance < self.config.strong_close_min_boundary_distance:
            return False
        atr = row.get("atr")
        if self.config.strong_close_min_boundary_distance_atr is not None:
            if pd.isna(atr) or float(atr) <= 0:
                return False
            if boundary_distance / float(atr) < self.config.strong_close_min_boundary_distance_atr:
                return False
        return True

    def _passes_retest(self, row: pd.Series, direction: Side) -> bool:
        if not self.config.require_retest:
            return True
        if direction is Side.LONG:
            return bool(row["low"] <= row["or_high"] and row["close"] > row["or_high"])
        return bool(row["high"] >= row["or_low"] and row["close"] < row["or_low"])

    def _passes_volatility_filter(self, row: pd.Series) -> bool:
        if not self.config.volatility_filter_enabled:
            return True
        width_atr = row.get("or_width_atr")
        if pd.isna(width_atr):
            return False
        if self.config.min_or_width_atr is not None and width_atr < self.config.min_or_width_atr:
            return False
        if self.config.max_or_width_atr is not None and width_atr > self.config.max_or_width_atr:
            return False
        return True

    def _passes_trend_filter(self, row: pd.Series, direction: Side) -> bool:
        filter_mode = self.config.trend_filter.lower()
        if filter_mode == "none":
            return True
        trend_column = self.config.trend_column
        if trend_column not in row.index or pd.isna(row[trend_column]):
            return False
        if filter_mode == "above_below_ma":
            return bool(row["close"] > row[trend_column]) if direction is Side.LONG else bool(row["close"] < row[trend_column])
        raise ValueError(f"Unsupported trend filter: {self.config.trend_filter}")

    def _is_time_allowed(self, row: pd.Series) -> bool:
        local_ts = pd.Timestamp(row.get("local_timestamp", row["timestamp"]))
        current_time = local_ts.time()
        if self.config.latest_entry_time is not None and current_time > time.fromisoformat(self.config.latest_entry_time):
            return False
        if self.config.allowed_days_of_week:
            current_day = local_ts.day_name().lower()
            if current_day not in {day.lower() for day in self.config.allowed_days_of_week}:
                return False
        if self.config.allowed_trade_windows:
            for window in self.config.allowed_trade_windows:
                start, end = window.split("-")
                if time.fromisoformat(start) <= current_time <= time.fromisoformat(end):
                    return True
            return False
        return True

    def _prepare_setup(self, history: pd.DataFrame, direction: Side) -> tuple[SetupEvent | None, dict[str, object]]:
        row = history.iloc[-1]
        try:
            stop_reference = self._compute_stop(history, direction)
        except Exception:
            return None, {"rejection_reason": "failed_stop_calculation", "rr_estimate": None}

        entry_reference = float(row["close"])
        risk = abs(entry_reference - stop_reference)
        if risk == 0.0:
            return None, {"rejection_reason": "failed_stop_calculation", "rr_estimate": None}

        first_target_name, first_target_price = select_first_liquidity_target(row, direction.value, self.config.liquidity_target_priority)
        if self.config.target_rule == "liquidity_sequence":
            if first_target_price is None:
                return None, {"rejection_reason": "failed_liquidity_target_lookup", "rr_estimate": None}
            target_reference = float(first_target_price)
            rr_estimate = abs(target_reference - entry_reference) / risk
            runner_targets = select_runner_targets(row, direction.value, self.config.liquidity_target_priority, first_target_name)
        else:
            target_reference = self._compute_default_target(entry_reference, stop_reference, direction)
            rr_estimate = abs(target_reference - entry_reference) / risk
            runner_targets = []

        if self.config.minimum_rr_threshold is not None and rr_estimate < self.config.minimum_rr_threshold:
            return None, {"rejection_reason": "failed_minimum_rr_threshold", "rr_estimate": rr_estimate}

        feature_snapshot = self._build_feature_snapshot(row)
        if self.config.setup_feature_whitelist is None:
            feature_snapshot.update(
                {
                    "rr_to_first_target": rr_estimate,
                    "first_liquidity_target": first_target_name,
                }
            )

        setup = SetupEvent(
            setup_id=f"orb-{uuid4().hex[:12]}",
            setup_name="orb",
            symbol=str(row["symbol"]),
            contract=str(row.get("contract", row["symbol"])),
            timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
            session_date=pd.Timestamp(row["session_date"]).to_pydatetime(),
            direction=direction,
            status=SetupStatus.CANDIDATE,
            entry_reference=entry_reference,
            stop_reference=stop_reference,
            target_reference=target_reference,
            features=feature_snapshot,
            context={
                "ny_open_anchored": self.config.ny_open_anchored,
                "or_high": float(row["or_high"]),
                "or_low": float(row["or_low"]),
                "or_midpoint": float((row["or_high"] + row["or_low"]) / 2.0),
                "decision_close": float(row["close"]),
                "stop_mode": self.config.stop_mode,
                "formal_stop_rule": self.config.formal_stop_rule,
                "target_r_multiple": self.config.target_r_multiple,
                "target_rule": self.config.target_rule,
                "session_name": self.config.session_name,
                "liquidity_target_mode": self.config.liquidity_target_mode,
                "liquidity_target_priority": list(self.config.liquidity_target_priority),
                "first_liquidity_target": first_target_name,
                "first_liquidity_target_price": first_target_price,
                "rr_to_first_target": rr_estimate,
                "first_draw_target_rule": self.config.first_draw_target_rule,
                "minimum_rr_threshold": self.config.minimum_rr_threshold,
                "partial_take_profit_rule": self.config.partial_take_profit_rule,
                "partial_take_profit_fraction": self.config.partial_take_profit_fraction,
                "breakeven_after_first_draw": self.config.breakeven_after_first_draw,
                "runner_target_rule": self.config.runner_target_rule if runner_targets else None,
                "runner_targets": [{"name": name, "price": price} for name, price in runner_targets],
                "runner_trail_rule": self.config.runner_trail_rule,
                "runner_trail_atr_multiple": self.config.runner_trail_atr_multiple,
                "strong_close_min_body_range_ratio": self.config.strong_close_min_body_range_ratio,
                "strong_close_min_boundary_distance": self.config.strong_close_min_boundary_distance,
                "strong_close_min_boundary_distance_atr": self.config.strong_close_min_boundary_distance_atr,
                "invalidate_on_early_counter_liquidity_consumption": self.config.invalidate_on_early_counter_liquidity_consumption,
                "allow_retest_rescue_after_early_liquidity_consumption": self.config.allow_retest_rescue_after_early_liquidity_consumption,
                "news_skip_rules": list(self.config.news_skip_rules),
                "discretionary_skip_reasons": list(self.config.discretionary_skip_reasons),
                "formalized_only_fields": list(self.config.formalized_only_fields),
            },
        )
        return setup, {"rejection_reason": None, "rr_estimate": rr_estimate}

    def _build_audit_row(
        self,
        row: pd.Series,
        breakout_side: Side | None,
        body_close_outside: bool,
        latest_entry_pass: bool,
        strong_close_pass: bool | None,
        retest_pass: bool | None,
        trend_pass: bool | None,
        volatility_pass: bool | None,
        rr_estimate: float | None,
        emitted: bool,
        rejection_reason: str | None,
    ) -> dict[str, object]:
        candle_range = float(row.get("candle_range", row["high"] - row["low"]))
        candle_body = float(row.get("candle_body", abs(row["close"] - row["open"])))
        body_range_ratio = candle_body / candle_range if candle_range > 0 else 0.0
        if breakout_side is Side.LONG:
            boundary_distance = float(row["close"] - row["or_high"])
            breakout_label = "long"
        elif breakout_side is Side.SHORT:
            boundary_distance = float(row["or_low"] - row["close"])
            breakout_label = "short"
        elif float(row["high"]) > float(row["or_high"]):
            boundary_distance = float(row["high"] - row["or_high"])
            breakout_label = "long_wick_only"
        elif float(row["low"]) < float(row["or_low"]):
            boundary_distance = float(row["or_low"] - row["low"])
            breakout_label = "short_wick_only"
        else:
            boundary_distance = 0.0
            breakout_label = "none"
        return {
            "timestamp": row["timestamp"],
            "symbol": row["symbol"],
            "session_date": row["session_date"],
            "or_high": row["or_high"],
            "or_low": row["or_low"],
            "close": row["close"],
            "candle_body_size": candle_body,
            "body_range_ratio": body_range_ratio,
            "boundary_distance_beyond_or": boundary_distance,
            "breakout_side": breakout_label,
            "body_close_outside_or": body_close_outside,
            "latest_entry_pass": latest_entry_pass,
            "strong_close_pass": strong_close_pass,
            "retest_pass": retest_pass,
            "trend_pass": trend_pass,
            "volatility_pass": volatility_pass,
            "estimated_rr": rr_estimate,
            "emitted_setup": emitted,
            "rejection_reason": rejection_reason or "emitted",
        }

    def _build_feature_snapshot(self, row: pd.Series) -> dict[str, float | int | str | bool | None]:
        if self.config.setup_feature_whitelist is not None:
            candidate_columns = [column for column in self.config.setup_feature_whitelist if column in row.index]
        else:
            candidate_columns = [column for column in row.index if column not in RAW_SETUP_COLUMNS and column not in EXCLUDED_SNAPSHOT_COLUMNS and not column.endswith("_id")]
        snapshot: dict[str, float | int | str | bool | None] = {}
        for column in candidate_columns:
            value = row[column]
            if pd.isna(value):
                continue
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, pd.Timestamp):
                value = value.isoformat()
            snapshot[column] = value
        return snapshot

    def _compute_stop(self, history: pd.DataFrame, direction: Side) -> float:
        row = history.iloc[-1]
        if self.config.formal_stop_rule == "or_boundary":
            return float(row["or_low"] if direction is Side.LONG else row["or_high"])
        if self.config.formal_stop_rule == "atr":
            atr_value = row.get("atr")
            if pd.isna(atr_value):
                raise ValueError("ATR-based stop requires atr column.")
            return float(row["close"] - self.config.stop_atr_multiple * atr_value) if direction is Side.LONG else float(row["close"] + self.config.stop_atr_multiple * atr_value)

        or_bars = self._opening_range_bars(history)
        midpoint_stop = self._midpoint_buffer_stop(row, direction)
        wick_stop = self._range_wick_buffer_stop(or_bars, direction, row)
        if self.config.formal_stop_rule == "or_midpoint_buffer":
            return midpoint_stop
        if self.config.formal_stop_rule == "range_wick_buffer":
            return wick_stop if wick_stop is not None else self._fallback_stop(row, direction)
        if self.config.formal_stop_rule == "midpoint_or_wick_buffer":
            if wick_stop is None:
                return midpoint_stop
            return min(midpoint_stop, wick_stop) if direction is Side.LONG else max(midpoint_stop, wick_stop)
        return self._fallback_stop(row, direction)

    def _midpoint_buffer_stop(self, row: pd.Series, direction: Side) -> float:
        midpoint = float((row["or_high"] + row["or_low"]) / 2.0)
        buffer = self._stop_buffer_value(row)
        return midpoint - buffer if direction is Side.LONG else midpoint + buffer

    def _range_wick_buffer_stop(self, or_bars: pd.DataFrame, direction: Side, row: pd.Series) -> float | None:
        if or_bars.empty:
            return None
        buffer = self._stop_buffer_value(row)
        if direction is Side.LONG:
            wick_series = or_bars.get("lower_wick")
            if wick_series is None or wick_series.isna().all():
                reference = float(or_bars["low"].min())
            else:
                reference_idx = wick_series.fillna(-1.0).idxmax()
                reference = float(or_bars.loc[reference_idx, "low"])
            return reference - buffer
        wick_series = or_bars.get("upper_wick")
        if wick_series is None or wick_series.isna().all():
            reference = float(or_bars["high"].max())
        else:
            reference_idx = wick_series.fillna(-1.0).idxmax()
            reference = float(or_bars.loc[reference_idx, "high"])
        return reference + buffer

    def _fallback_stop(self, row: pd.Series, direction: Side) -> float:
        if self.config.fallback_stop_mode == "atr":
            atr_value = row.get("atr")
            if pd.isna(atr_value):
                return float(row["or_low"] if direction is Side.LONG else row["or_high"])
            return float(row["close"] - self.config.stop_atr_multiple * atr_value) if direction is Side.LONG else float(row["close"] + self.config.stop_atr_multiple * atr_value)
        return float(row["or_low"] if direction is Side.LONG else row["or_high"])

    def _stop_buffer_value(self, row: pd.Series) -> float:
        buffer = float(self.config.stop_buffer_points or 0.0)
        if self.config.stop_buffer_atr_multiple is not None:
            atr_value = row.get("atr")
            if not pd.isna(atr_value):
                buffer += float(self.config.stop_buffer_atr_multiple) * float(atr_value)
        return buffer

    def _opening_range_bars(self, history: pd.DataFrame) -> pd.DataFrame:
        timestamps = pd.to_datetime(history.get("local_timestamp", history["timestamp"]))
        start = time.fromisoformat(self.config.opening_range_start_time)
        end = time.fromisoformat(self.config.opening_range_end_time)
        mask = (timestamps.dt.time >= start) & (timestamps.dt.time < end)
        return history.loc[mask]

    def _compute_default_target(self, entry_reference: float, stop_reference: float, direction: Side) -> float:
        risk = abs(entry_reference - stop_reference)
        return float(entry_reference + risk * self.config.target_r_multiple) if direction is Side.LONG else float(entry_reference - risk * self.config.target_r_multiple)
