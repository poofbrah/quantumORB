from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from uuid import uuid4

import pandas as pd

from config.models import StrategyConfig
from data.schemas import SetupEvent, SetupStatus, Side
from indicators.technical import add_atr
from .base import BaseSetupDetector
from .session_context import add_trading_day, within_any_trade_window, window_end_timestamp

RP_PROFILE_NAME = "rp_profits_8am_orb"
ENTRY_FAMILY_DISPLACEMENT = "displacement"
ENTRY_FAMILY_REENTRY = "range_reentry"
ENTRY_MODE_DISPLACEMENT = "displacement_vwap_pullback"
ENTRY_MODE_REENTRY = "range_reentry_vwap"
ENTRY_MODE_HYBRID = "hybrid"
DIAGNOSTIC_KEYS = (
    "days_scanned",
    "zone_ready",
    "zone_missing",
    "zone_out_of_band",
    "strong_close_failed",
    "displacement_bullish",
    "displacement_bearish",
    "displacement_not_found",
    "vwap_missing",
    "vwap_retracement_found",
    "vwap_pullback_not_reached",
    "reentry_top_break",
    "reentry_bottom_break",
    "reentry_close_back_inside_failed",
    "reentry_short_signal",
    "reentry_long_signal",
    "reentry_not_found",
    "trade_window_expired",
    "invalid_risk_geometry",
    "setups_emitted",
)


@dataclass(slots=True)
class RPProfits8AMConfig:
    instrument: str = "NQ"
    timezone: str = "America/New_York"
    daily_reset_time: str = "18:00"
    key_zone_start_time: str = "08:00"
    key_zone_end_time: str = "08:15"
    key_zone_min_width_points: float | None = None
    key_zone_max_width_points: float | None = None
    key_zone_min_width_atr: float | None = None
    key_zone_max_width_atr: float | None = None
    trade_window_required: bool = True
    trade_windows: tuple[str, ...] = ("09:00-10:30",)
    entry_mode: str = ENTRY_MODE_DISPLACEMENT
    strong_close_min_body_range_ratio: float = 0.6
    strong_close_min_boundary_distance: float = 0.25
    strong_close_min_boundary_distance_atr: float | None = 0.5
    rr_multiple: float = 2.0
    stop_mode: str = "zone_boundary"
    stop_buffer_points: float = 0.0
    partial_at_r: float = 1.0
    partial_fraction: float = 0.5
    move_stop_to_breakeven_after_partial: bool = True
    runner_target_r: float | None = None
    runner_trailing_stop_mode: str = "atr"
    runner_trailing_atr_multiple: float = 1.5
    reentry_body_ratio_min: float = 0.5
    reentry_min_close_distance_points: float = 0.25
    reentry_min_close_distance_atr: float | None = 0.25
    reentry_stop_mode: str = "signal_or_range"
    reentry_stop_buffer_points: float = 0.0
    reentry_max_entry_time: str = "10:30"
    reentry_tp1_mode: str = "vwap_or_mid"
    reentry_partial_fraction: float = 0.5
    reentry_breakeven_after_tp1: bool = True
    max_trades_per_day: int = 1

    @classmethod
    def from_strategy_config(cls, config: StrategyConfig) -> "RPProfits8AMConfig":
        return cls(
            instrument=config.instrument,
            timezone=config.session_timezone,
            daily_reset_time=config.daily_reset_time,
            key_zone_start_time=config.key_zone_start_time,
            key_zone_end_time=config.key_zone_end_time,
            key_zone_min_width_points=config.key_zone_min_width_points,
            key_zone_max_width_points=config.key_zone_max_width_points,
            key_zone_min_width_atr=config.key_zone_min_width_atr,
            key_zone_max_width_atr=config.key_zone_max_width_atr,
            trade_window_required=config.trade_window_required,
            trade_windows=tuple(config.allowed_trade_windows or ("09:00-10:30",)),
            entry_mode=config.rp_entry_mode,
            strong_close_min_body_range_ratio=config.strong_close_min_body_range_ratio or 0.6,
            strong_close_min_boundary_distance=(config.strong_close_min_boundary_distance if config.strong_close_min_boundary_distance is not None else 0.25),
            strong_close_min_boundary_distance_atr=config.strong_close_min_boundary_distance_atr,
            rr_multiple=config.rr_multiple,
            stop_mode=config.rp_stop_mode,
            stop_buffer_points=config.rp_stop_buffer_points,
            partial_at_r=config.rp_partial_at_r,
            partial_fraction=config.rp_partial_fraction,
            move_stop_to_breakeven_after_partial=config.rp_move_stop_to_breakeven_after_partial,
            runner_target_r=config.rp_runner_target_r,
            runner_trailing_stop_mode=config.rp_runner_trailing_stop_mode,
            runner_trailing_atr_multiple=config.rp_runner_trailing_atr_multiple,
            reentry_body_ratio_min=config.reentry_body_ratio_min,
            reentry_min_close_distance_points=config.reentry_min_close_distance_points,
            reentry_min_close_distance_atr=config.reentry_min_close_distance_atr,
            reentry_stop_mode=config.rp_reentry_stop_mode,
            reentry_stop_buffer_points=config.rp_reentry_stop_buffer_points,
            reentry_max_entry_time=config.rp_reentry_max_entry_time,
            reentry_tp1_mode=config.rp_reentry_tp1_mode,
            reentry_partial_fraction=config.rp_reentry_partial_fraction,
            reentry_breakeven_after_tp1=config.rp_reentry_breakeven_after_tp1,
            max_trades_per_day=config.max_trades_per_session,
        )


@dataclass(slots=True)
class RPProfitsDiagnostics:
    counts: dict[str, int]
    audit_frame: pd.DataFrame


class RPProfits8AMSetupDetector(BaseSetupDetector):
    def __init__(self, config: RPProfits8AMConfig | None = None) -> None:
        self.config = config or RPProfits8AMConfig()

    def detect(self, frame: pd.DataFrame) -> list[SetupEvent]:
        setups, _ = self.detect_with_diagnostics(frame)
        return setups

    def detect_with_diagnostics(self, frame: pd.DataFrame) -> tuple[list[SetupEvent], RPProfitsDiagnostics]:
        if frame.empty:
            empty_counts = {key: 0 for key in DIAGNOSTIC_KEYS}
            return [], RPProfitsDiagnostics(counts=empty_counts, audit_frame=pd.DataFrame())
        required = {"timestamp", "symbol", "open", "high", "low", "close"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"RP Profits setup detection requires columns: {sorted(missing)}")
        prepared = add_trading_day(frame, reset_time=self.config.daily_reset_time, timezone=self.config.timezone)
        prepared = prepared.sort_values(["symbol", "local_timestamp"], kind="stable").reset_index(drop=True)
        if "atr" not in prepared.columns:
            prepared = add_atr(prepared)
        setups: list[SetupEvent] = []
        audit_rows: list[dict[str, object]] = []
        counts = Counter({key: 0 for key in DIAGNOSTIC_KEYS})
        for (symbol, trade_day), day_frame in prepared.groupby(["symbol", "trade_day"], sort=False):
            day_setups, day_audit, day_counts = self._detect_day(day_frame.reset_index(drop=True), symbol=str(symbol), trade_day=pd.Timestamp(trade_day))
            setups.extend(day_setups)
            audit_rows.extend(day_audit)
            counts.update(day_counts)
        return setups, RPProfitsDiagnostics(counts=dict(counts), audit_frame=pd.DataFrame(audit_rows))

    def _detect_day(self, day_frame: pd.DataFrame, *, symbol: str, trade_day: pd.Timestamp) -> tuple[list[SetupEvent], list[dict[str, object]], Counter]:
        counts = Counter({key: 0 for key in DIAGNOSTIC_KEYS})
        counts["days_scanned"] += 1
        zone_start = pd.Timestamp(self.config.key_zone_start_time).time()
        zone_end = pd.Timestamp(self.config.key_zone_end_time).time()
        zone_rows = day_frame[(day_frame["local_timestamp"].dt.time >= zone_start) & (day_frame["local_timestamp"].dt.time < zone_end)]
        if zone_rows.empty:
            counts["zone_missing"] += 1
            return [], [self._audit_row(symbol, trade_day, zone_formed=False, rejection_reason="missing_zone")], counts
        zone_high = float(zone_rows["high"].max())
        zone_low = float(zone_rows["low"].min())
        zone_width = zone_high - zone_low
        zone_mid = (zone_high + zone_low) / 2.0
        zone_atr = self._last_valid(zone_rows, "atr")
        range_width_atr = (zone_width / zone_atr) if zone_atr not in (None, 0.0) else None
        counts["zone_ready"] += 1
        if not self._range_allowed(zone_width, range_width_atr):
            counts["zone_out_of_band"] += 1
            return [], [self._audit_row(symbol, trade_day, zone_formed=True, range_width=zone_width, range_width_atr=range_width_atr, rejection_reason="zone_out_of_band")], counts
        disp_setup, disp_info, disp_counts = self._find_displacement_setup(day_frame, trade_day, zone_high, zone_low, zone_mid, zone_width, range_width_atr)
        counts.update(disp_counts)
        reentry_setup, reentry_info, reentry_counts = self._find_reentry_setup(day_frame, trade_day, zone_high, zone_low, zone_mid, zone_width, range_width_atr)
        counts.update(reentry_counts)
        chosen_setup = self._choose_setup(disp_setup, reentry_setup)
        if chosen_setup is not None:
            counts["setups_emitted"] += 1
        audit = self._audit_row(
            symbol,
            trade_day,
            zone_formed=True,
            range_width=zone_width,
            range_width_atr=range_width_atr,
            displacement_found=disp_info["displacement_found"],
            displacement_side=disp_info["displacement_side"],
            displacement_time=disp_info["displacement_time"],
            displacement_strength=disp_info["displacement_strength"],
            vwap_retracement_found=disp_info["vwap_retracement_found"],
            setup_emitted=chosen_setup is not None,
            vwap_at_entry=chosen_setup.context.get("vwap_at_entry") if chosen_setup is not None else None,
            retracement_depth=chosen_setup.features.get("retracement_depth") if chosen_setup is not None else None,
            rejection_reason=self._choose_rejection_reason(chosen_setup, disp_info, reentry_info),
            entry_family=chosen_setup.context.get("entry_family") if chosen_setup is not None else None,
            range_break_side=reentry_info["range_break_side"],
            broke_outside_range=reentry_info["broke_outside_range"],
            closed_back_inside=reentry_info["closed_back_inside"],
            reentry_signal_time=reentry_info["reentry_signal_time"],
            tp1_price=chosen_setup.context.get("first_liquidity_target_price") if chosen_setup is not None else None,
            runner_target_price=self._runner_target_from_setup(chosen_setup),
            stop_anchor_used=chosen_setup.context.get("stop_anchor_used") if chosen_setup is not None else None,
        )
        return ([chosen_setup] if chosen_setup is not None else []), [audit], counts
    def _find_displacement_setup(self, day_frame: pd.DataFrame, trade_day: pd.Timestamp, zone_high: float, zone_low: float, zone_mid: float, zone_width: float, range_width_atr: float | None) -> tuple[SetupEvent | None, dict[str, object], Counter]:
        info = {"displacement_found": False, "displacement_side": None, "displacement_time": None, "displacement_strength": None, "vwap_retracement_found": False, "rejection_reason": "displacement_not_found"}
        counts = Counter({key: 0 for key in DIAGNOSTIC_KEYS})
        if self.config.entry_mode not in {ENTRY_MODE_DISPLACEMENT, ENTRY_MODE_HYBRID}:
            return None, info, counts
        zone_end_ts = self._localized_timestamp(trade_day, self.config.key_zone_end_time)
        entry_expiration = self._entry_expiration(trade_day)
        displacement_side = None
        displacement_time = None
        displacement_close = None
        displacement_strength = None
        displacement_boundary_distance = None
        displacement_atr = None
        for _, row in day_frame.iterrows():
            ts = pd.Timestamp(row["local_timestamp"])
            if ts < zone_end_ts:
                continue
            atr_value = row.get("atr")
            if displacement_side is None:
                displacement = self._displacement_signal(row, zone_high, zone_low, atr_value)
                if displacement is None:
                    if float(row["high"]) > zone_high or float(row["low"]) < zone_low:
                        counts["strong_close_failed"] += 1
                    continue
                displacement_side = str(displacement["side"])
                displacement_time = ts
                displacement_close = float(row["close"])
                displacement_strength = float(displacement["strength"])
                displacement_boundary_distance = float(displacement["boundary_distance"])
                displacement_atr = float(atr_value) if atr_value is not None and not pd.isna(atr_value) else None
                info.update({"displacement_found": True, "displacement_side": displacement_side, "displacement_time": displacement_time, "displacement_strength": displacement_strength, "rejection_reason": "vwap_pullback_not_reached"})
                counts[f"displacement_{displacement_side}"] += 1
                continue
            if entry_expiration is not None and ts > entry_expiration:
                counts["trade_window_expired"] += 1
                info["rejection_reason"] = "trade_window_expired"
                break
            if self.config.trade_window_required and not within_any_trade_window(ts, self.config.trade_windows):
                continue
            active_vwap = self._active_vwap(row)
            if active_vwap is None or pd.isna(active_vwap):
                counts["vwap_missing"] += 1
                info["rejection_reason"] = "vwap_missing"
                continue
            if not (float(row["low"]) <= float(active_vwap) <= float(row["high"])):
                continue
            info["vwap_retracement_found"] = True
            counts["vwap_retracement_found"] += 1
            entry_price = float(active_vwap)
            stop_price, stop_anchor_used = self._compute_displacement_stop(displacement_side, zone_high, zone_low, zone_mid)
            risk = entry_price - stop_price if displacement_side == "bullish" else stop_price - entry_price
            if risk <= 0:
                counts["invalid_risk_geometry"] += 1
                info["rejection_reason"] = "invalid_risk_geometry"
                return None, info, counts
            direction = Side.LONG if displacement_side == "bullish" else Side.SHORT
            partial_price = entry_price + self.config.partial_at_r * risk if direction is Side.LONG else entry_price - self.config.partial_at_r * risk
            runner_targets: list[dict[str, float | str]] = []
            disable_default_target_exit = True
            target_price = entry_price + self.config.rr_multiple * risk if direction is Side.LONG else entry_price - self.config.rr_multiple * risk
            if self.config.runner_target_r is not None:
                runner_price = entry_price + self.config.runner_target_r * risk if direction is Side.LONG else entry_price - self.config.runner_target_r * risk
                runner_targets.append({"name": "runner_r", "price": float(runner_price)})
                disable_default_target_exit = False
                target_price = float(runner_price)
            retracement_depth = abs(displacement_close - entry_price) if displacement_close is not None else None
            setup = SetupEvent(
                setup_id=f"rp-8am-{uuid4().hex[:12]}",
                setup_name=RP_PROFILE_NAME,
                symbol=str(row["symbol"]),
                contract=str(row.get("contract", row["symbol"])),
                timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                session_date=trade_day.to_pydatetime(),
                direction=direction,
                status=SetupStatus.CANDIDATE,
                entry_reference=entry_price,
                stop_reference=float(stop_price),
                target_reference=float(target_price),
                features={"zone_high": zone_high, "zone_low": zone_low, "zone_mid": zone_mid, "range_width": zone_width, "range_width_atr": range_width_atr, "displacement_strength": displacement_strength, "displacement_boundary_distance": displacement_boundary_distance, "atr": displacement_atr, "vwap_at_entry": entry_price, "retracement_depth": retracement_depth},
                context={"strategy_profile": RP_PROFILE_NAME, "entry_family": ENTRY_FAMILY_DISPLACEMENT, "entry_model": ENTRY_MODE_DISPLACEMENT, "entry_fill_mode": "limit_touch", "skip_entry_bar_management": True, "entry_expiration_time": entry_expiration.isoformat() if entry_expiration is not None else None, "trade_day": trade_day.isoformat(), "displacement_side": displacement_side, "displacement_time": displacement_time.isoformat() if displacement_time is not None else None, "displacement_strength": displacement_strength, "range_width": zone_width, "range_width_atr": range_width_atr, "vwap_at_entry": entry_price, "active_vwap": entry_price, "retracement_depth": retracement_depth, "stop_anchor_used": stop_anchor_used, "stop_mode": self.config.stop_mode, "formal_stop_rule": self.config.stop_mode, "initial_risk": risk, "rr_multiple": self.config.rr_multiple, "partial_take_profit_fraction": self.config.partial_fraction, "first_liquidity_target": "one_r", "first_liquidity_target_price": float(partial_price), "breakeven_after_first_draw": self.config.move_stop_to_breakeven_after_partial, "disable_default_target_exit": disable_default_target_exit, "runner_targets": runner_targets, "runner_trail_rule": "atr_placeholder" if self.config.runner_trailing_stop_mode == "atr" else None, "runner_trail_atr_multiple": self.config.runner_trailing_atr_multiple},
            )
            info["rejection_reason"] = "setup_emitted"
            return setup, info, counts
        if info["displacement_found"] and not info["vwap_retracement_found"] and info["rejection_reason"] == "vwap_pullback_not_reached":
            counts["vwap_pullback_not_reached"] += 1
        elif not info["displacement_found"]:
            counts["displacement_not_found"] += 1
        return None, info, counts

    def _find_reentry_setup(self, day_frame: pd.DataFrame, trade_day: pd.Timestamp, zone_high: float, zone_low: float, zone_mid: float, zone_width: float, range_width_atr: float | None) -> tuple[SetupEvent | None, dict[str, object], Counter]:
        info = {"range_break_side": None, "broke_outside_range": False, "closed_back_inside": False, "reentry_signal_time": None, "rejection_reason": "reentry_not_found"}
        counts = Counter({key: 0 for key in DIAGNOSTIC_KEYS})
        if self.config.entry_mode not in {ENTRY_MODE_REENTRY, ENTRY_MODE_HYBRID}:
            return None, info, counts
        zone_end_ts = self._localized_timestamp(trade_day, self.config.key_zone_end_time)
        max_entry_ts = self._localized_timestamp(trade_day, self.config.reentry_max_entry_time)
        top_break_time = None
        bottom_break_time = None
        for _, row in day_frame.iterrows():
            ts = pd.Timestamp(row["local_timestamp"])
            if ts < zone_end_ts:
                continue
            if ts > max_entry_ts:
                if top_break_time is not None or bottom_break_time is not None:
                    counts["trade_window_expired"] += 1
                    info["rejection_reason"] = "trade_window_expired"
                break
            high_price = float(row["high"])
            low_price = float(row["low"])
            if top_break_time is None and high_price > zone_high:
                top_break_time = ts
                info["range_break_side"] = "top"
                info["broke_outside_range"] = True
                counts["reentry_top_break"] += 1
            if bottom_break_time is None and low_price < zone_low:
                bottom_break_time = ts
                info["range_break_side"] = "bottom"
                info["broke_outside_range"] = True
                counts["reentry_bottom_break"] += 1
            if self.config.trade_window_required and not within_any_trade_window(ts, self.config.trade_windows):
                continue
            candidate_side = None
            if top_break_time is not None and ts > top_break_time and self._valid_reentry_close_inside(row, "short", zone_high, zone_low):
                candidate_side = "short"
            elif bottom_break_time is not None and ts > bottom_break_time and self._valid_reentry_close_inside(row, "long", zone_high, zone_low):
                candidate_side = "long"
            elif (top_break_time is not None and ts > top_break_time) or (bottom_break_time is not None and ts > bottom_break_time):
                counts["reentry_close_back_inside_failed"] += 1
                continue
            if candidate_side is None:
                continue
            info["closed_back_inside"] = True
            info["reentry_signal_time"] = ts
            info["rejection_reason"] = "setup_emitted"
            if candidate_side == "short":
                counts["reentry_short_signal"] += 1
                direction = Side.SHORT
                range_break_side = "top"
                runner_target_price = zone_low
            else:
                counts["reentry_long_signal"] += 1
                direction = Side.LONG
                range_break_side = "bottom"
                runner_target_price = zone_high
            entry_price = float(row["close"])
            stop_price, stop_anchor_used = self._compute_reentry_stop(row, candidate_side, zone_high, zone_low)
            risk = entry_price - stop_price if direction is Side.LONG else stop_price - entry_price
            if risk <= 0:
                counts["invalid_risk_geometry"] += 1
                info["rejection_reason"] = "invalid_risk_geometry"
                return None, info, counts
            tp1_price, tp1_mode = self._reentry_tp1_price(row, direction, entry_price, zone_mid, runner_target_price)
            setup = SetupEvent(
                setup_id=f"rp-8am-{uuid4().hex[:12]}",
                setup_name=RP_PROFILE_NAME,
                symbol=str(row["symbol"]),
                contract=str(row.get("contract", row["symbol"])),
                timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                session_date=trade_day.to_pydatetime(),
                direction=direction,
                status=SetupStatus.CANDIDATE,
                entry_reference=entry_price,
                stop_reference=float(stop_price),
                target_reference=float(runner_target_price),
                features={"zone_high": zone_high, "zone_low": zone_low, "zone_mid": zone_mid, "range_width": zone_width, "range_width_atr": range_width_atr, "displacement_strength": None, "vwap_at_entry": self._ny_vwap(row), "retracement_depth": None, "tp1_price": tp1_price, "runner_target_price": runner_target_price},
                context={"strategy_profile": RP_PROFILE_NAME, "entry_family": ENTRY_FAMILY_REENTRY, "entry_model": ENTRY_MODE_REENTRY, "entry_fill_mode": "signal_close", "trade_day": trade_day.isoformat(), "range_break_side": range_break_side, "broke_outside_range": True, "closed_back_inside": True, "reentry_signal_time": ts.isoformat(), "tp1_price": float(tp1_price), "tp1_mode": tp1_mode, "runner_target_price": float(runner_target_price), "stop_anchor_used": stop_anchor_used, "stop_mode": self.config.reentry_stop_mode, "formal_stop_rule": self.config.reentry_stop_mode, "initial_risk": risk, "partial_take_profit_fraction": self.config.reentry_partial_fraction, "first_liquidity_target": tp1_mode, "first_liquidity_target_price": float(tp1_price), "breakeven_after_first_draw": self.config.reentry_breakeven_after_tp1, "disable_default_target_exit": True, "runner_targets": [{"name": "range_runner", "price": float(runner_target_price)}], "runner_trail_rule": None},
            )
            return setup, info, counts
        counts["reentry_not_found"] += 1
        return None, info, counts
    def _choose_setup(self, disp_setup: SetupEvent | None, reentry_setup: SetupEvent | None) -> SetupEvent | None:
        if self.config.entry_mode == ENTRY_MODE_DISPLACEMENT:
            return disp_setup
        if self.config.entry_mode == ENTRY_MODE_REENTRY:
            return reentry_setup
        candidates = [setup for setup in (disp_setup, reentry_setup) if setup is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda setup: (pd.Timestamp(setup.timestamp), setup.setup_id))

    def _choose_rejection_reason(self, chosen_setup: SetupEvent | None, disp_info: dict[str, object], reentry_info: dict[str, object]) -> str:
        if chosen_setup is not None:
            return "setup_emitted"
        if self.config.entry_mode == ENTRY_MODE_DISPLACEMENT:
            return str(disp_info["rejection_reason"])
        if self.config.entry_mode == ENTRY_MODE_REENTRY:
            return str(reentry_info["rejection_reason"])
        for reason in (reentry_info["rejection_reason"], disp_info["rejection_reason"]):
            if reason not in {"reentry_not_found", "displacement_not_found"}:
                return str(reason)
        return "no_branch_signal"

    def _range_allowed(self, zone_width: float, range_width_atr: float | None) -> bool:
        if self.config.key_zone_min_width_points is not None and zone_width < self.config.key_zone_min_width_points:
            return False
        if self.config.key_zone_max_width_points is not None and zone_width > self.config.key_zone_max_width_points:
            return False
        if range_width_atr is not None:
            if self.config.key_zone_min_width_atr is not None and range_width_atr < self.config.key_zone_min_width_atr:
                return False
            if self.config.key_zone_max_width_atr is not None and range_width_atr > self.config.key_zone_max_width_atr:
                return False
        return True

    def _displacement_signal(self, row: pd.Series, zone_high: float, zone_low: float, atr_value: object) -> dict[str, float | str] | None:
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        candle_range = high_price - low_price
        if candle_range <= 0:
            return None
        body_size = abs(close_price - open_price)
        body_ratio = body_size / candle_range
        atr_float = None if atr_value is None or pd.isna(atr_value) else float(atr_value)
        bull_distance = close_price - zone_high
        bear_distance = zone_low - close_price
        ratio_ok = body_ratio >= self.config.strong_close_min_body_range_ratio
        bull_close_ok = close_price > zone_high and close_price > open_price
        bear_close_ok = close_price < zone_low and close_price < open_price
        bull_distance_ok = bull_distance >= self.config.strong_close_min_boundary_distance
        bear_distance_ok = bear_distance >= self.config.strong_close_min_boundary_distance
        if self.config.strong_close_min_boundary_distance_atr is not None:
            if atr_float is None or atr_float <= 0:
                bull_atr_ok = False
                bear_atr_ok = False
            else:
                bull_atr_ok = (bull_distance / atr_float) >= self.config.strong_close_min_boundary_distance_atr
                bear_atr_ok = (bear_distance / atr_float) >= self.config.strong_close_min_boundary_distance_atr
        else:
            bull_atr_ok = True
            bear_atr_ok = True
        if bull_close_ok and ratio_ok and bull_distance_ok and bull_atr_ok:
            strength = (bull_distance / atr_float) if atr_float and atr_float > 0 else bull_distance
            return {"side": "bullish", "strength": strength, "boundary_distance": bull_distance}
        if bear_close_ok and ratio_ok and bear_distance_ok and bear_atr_ok:
            strength = (bear_distance / atr_float) if atr_float and atr_float > 0 else bear_distance
            return {"side": "bearish", "strength": strength, "boundary_distance": bear_distance}
        return None

    def _valid_reentry_close_inside(self, row: pd.Series, side: str, zone_high: float, zone_low: float) -> bool:
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        candle_range = high_price - low_price
        if candle_range <= 0:
            return False
        body_ratio = abs(close_price - open_price) / candle_range
        atr_value = row.get("atr")
        atr_float = None if atr_value is None or pd.isna(atr_value) else float(atr_value)
        if body_ratio < self.config.reentry_body_ratio_min:
            return False
        if side == "short":
            if not (close_price < zone_high and close_price > zone_low and close_price < open_price):
                return False
            distance_inside = zone_high - close_price
        else:
            if not (close_price > zone_low and close_price < zone_high and close_price > open_price):
                return False
            distance_inside = close_price - zone_low
        if distance_inside < self.config.reentry_min_close_distance_points:
            return False
        if self.config.reentry_min_close_distance_atr is not None:
            if atr_float is None or atr_float <= 0:
                return False
            if (distance_inside / atr_float) < self.config.reentry_min_close_distance_atr:
                return False
        return True

    def _compute_displacement_stop(self, displacement_side: str, zone_high: float, zone_low: float, zone_mid: float) -> tuple[float, str]:
        buffer_points = float(self.config.stop_buffer_points)
        if self.config.stop_mode == "zone_midpoint":
            stop_price = zone_mid - buffer_points if displacement_side == "bullish" else zone_mid + buffer_points
            return float(stop_price), "zone_midpoint"
        stop_price = zone_low - buffer_points if displacement_side == "bullish" else zone_high + buffer_points
        return float(stop_price), "zone_boundary"

    def _compute_reentry_stop(self, row: pd.Series, side: str, zone_high: float, zone_low: float) -> tuple[float, str]:
        buffer_points = float(self.config.reentry_stop_buffer_points)
        entry_price = float(row["close"])
        if self.config.reentry_stop_mode in {"signal_or_range", "signal_candle"}:
            signal_stop = float(row["low"] - buffer_points) if side == "long" else float(row["high"] + buffer_points)
            if (side == "long" and signal_stop < entry_price) or (side == "short" and signal_stop > entry_price):
                return signal_stop, "signal_candle"
        fallback_stop = float(zone_low - buffer_points) if side == "long" else float(zone_high + buffer_points)
        return fallback_stop, "range_boundary"

    def _reentry_tp1_price(self, row: pd.Series, direction: Side, entry_price: float, range_mid: float, runner_target_price: float) -> tuple[float, str]:
        if self.config.reentry_tp1_mode == "mid":
            return float(range_mid), "range_mid"
        vwap_value = self._ny_vwap(row)
        if vwap_value is not None and self._value_on_path(direction, entry_price, float(vwap_value), runner_target_price):
            return float(vwap_value), "ny_vwap"
        return float(range_mid), "range_mid"

    def _ny_vwap(self, row: pd.Series) -> float | None:
        if "vwap_rth" in row and not pd.isna(row.get("vwap_rth")):
            return float(row.get("vwap_rth"))
        return None

    def _active_vwap(self, row: pd.Series) -> float | None:
        local_ts = pd.Timestamp(row["local_timestamp"])
        before_rth = local_ts.time() < pd.Timestamp("09:30").time()
        if before_rth and "vwap_eth" in row and not pd.isna(row.get("vwap_eth")):
            return float(row.get("vwap_eth"))
        if "vwap_rth" in row and not pd.isna(row.get("vwap_rth")):
            return float(row.get("vwap_rth"))
        if "vwap_eth" in row and not pd.isna(row.get("vwap_eth")):
            return float(row.get("vwap_eth"))
        if "vwap" in row and not pd.isna(row.get("vwap")):
            return float(row.get("vwap"))
        return None

    def _value_on_path(self, direction: Side, entry_price: float, candidate: float, target: float) -> bool:
        if direction is Side.LONG:
            return entry_price < candidate < target
        return target < candidate < entry_price

    def _entry_expiration(self, trade_day: pd.Timestamp) -> pd.Timestamp | None:
        if not self.config.trade_windows:
            return None
        expiry = max(window_end_timestamp(trade_day, window, timezone=self.config.timezone) for window in self.config.trade_windows)
        return pd.Timestamp(expiry)

    def _localized_timestamp(self, trade_day: pd.Timestamp, time_text: str) -> pd.Timestamp:
        return pd.Timestamp.combine(trade_day.date(), pd.Timestamp(time_text).time()).tz_localize(self.config.timezone)

    def _last_valid(self, frame: pd.DataFrame, column: str) -> float | None:
        if column not in frame.columns:
            return None
        values = frame[column].dropna()
        if values.empty:
            return None
        return float(values.iloc[-1])

    def _runner_target_from_setup(self, setup: SetupEvent | None) -> float | None:
        if setup is None:
            return None
        runner_targets = setup.context.get("runner_targets", [])
        if runner_targets:
            return float(runner_targets[0]["price"])
        return None

    def _audit_row(self, symbol: str, trade_day: pd.Timestamp, *, zone_formed: bool, range_width: float | None = None, range_width_atr: float | None = None, displacement_found: bool = False, displacement_side: str | None = None, displacement_time: pd.Timestamp | None = None, displacement_strength: float | None = None, vwap_retracement_found: bool = False, setup_emitted: bool = False, vwap_at_entry: float | None = None, retracement_depth: float | None = None, rejection_reason: str = "", entry_family: str | None = None, range_break_side: str | None = None, broke_outside_range: bool = False, closed_back_inside: bool = False, reentry_signal_time: pd.Timestamp | None = None, tp1_price: float | None = None, runner_target_price: float | None = None, stop_anchor_used: str | None = None) -> dict[str, object]:
        return {"symbol": symbol, "trade_day": trade_day, "trade_window": "|".join(self.config.trade_windows), "zone_formed": zone_formed, "range_width": range_width, "range_width_atr": range_width_atr, "displacement_found": displacement_found, "displacement_side": displacement_side, "displacement_time": displacement_time, "displacement_strength": displacement_strength, "vwap_retracement_found": vwap_retracement_found, "entry_family": entry_family, "range_break_side": range_break_side, "broke_outside_range": broke_outside_range, "closed_back_inside": closed_back_inside, "reentry_signal_time": reentry_signal_time, "setup_emitted": setup_emitted, "trade_executed": False, "vwap_at_entry": vwap_at_entry, "tp1_price": tp1_price, "runner_target_price": runner_target_price, "stop_anchor_used": stop_anchor_used, "retracement_depth": retracement_depth, "rejection_reason": rejection_reason}
