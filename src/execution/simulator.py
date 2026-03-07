from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

import pandas as pd

from data.schemas import SetupEvent, Side, Trade


class EntryConvention(str, Enum):
    NEXT_BAR_OPEN = "next_bar_open"


class IntrabarExitConflictPolicy(str, Enum):
    STOP_FIRST = "stop_first"
    TARGET_FIRST = "target_first"


class ExitReason(str, Enum):
    STOP = "stop"
    TARGET = "target"
    SESSION_END = "session_end"
    DATA_END = "data_end"
    RUNNER_TARGET = "runner_target"
    BREAKEVEN = "breakeven"
    TRAIL_STOP = "trail_stop"


@dataclass(slots=True)
class ExecutionConfig:
    entry_convention: EntryConvention = EntryConvention.NEXT_BAR_OPEN
    slippage_per_unit: float = 0.0
    commission_per_unit: float = 0.0
    position_size: float = 1.0
    one_position_only: bool = True
    exit_on_session_end: bool = True
    intrabar_exit_conflict_policy: IntrabarExitConflictPolicy = IntrabarExitConflictPolicy.STOP_FIRST


@dataclass(slots=True)
class ActivePosition:
    setup: SetupEvent
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    remaining_size: float
    current_stop: float
    bars_held: int = 0
    realized_pnl: float = 0.0
    realized_fees: float = 0.0
    partial_taken: bool = False
    breakeven_activated: bool = False
    runner_target_hit: bool = False
    trail_activated: bool = False
    first_liquidity_target: str | None = None
    first_liquidity_price: float | None = None
    runner_targets: list[dict[str, float | str]] = field(default_factory=list)
    trail_stop: float | None = None


def enter_position(setup: SetupEvent, bar: pd.Series, config: ExecutionConfig) -> ActivePosition:
    if config.entry_convention is not EntryConvention.NEXT_BAR_OPEN:
        raise ValueError(f"Unsupported entry convention: {config.entry_convention}")
    raw_price = float(bar["open"])
    entry_price = _apply_entry_slippage(raw_price, setup.direction, config.slippage_per_unit)
    first_target = setup.context.get("first_liquidity_target")
    first_target_price = setup.context.get("first_liquidity_target_price")
    runner_targets = list(setup.context.get("runner_targets", []))
    return ActivePosition(
        setup=setup,
        entry_time=pd.Timestamp(bar["timestamp"]),
        entry_price=entry_price,
        size=float(config.position_size),
        remaining_size=float(config.position_size),
        current_stop=float(setup.stop_reference),
        first_liquidity_target=first_target,
        first_liquidity_price=float(first_target_price) if first_target_price is not None else None,
        runner_targets=runner_targets,
    )


def evaluate_position_on_bar(active: ActivePosition, bar: pd.Series, config: ExecutionConfig) -> dict | None:
    side = active.setup.direction
    events: list[dict] = []

    if not active.partial_taken and active.first_liquidity_price is not None:
        if _target_hit(side, bar, active.first_liquidity_price):
            partial_fraction = float(active.setup.context.get("partial_take_profit_fraction", 0.5))
            partial_size = active.size * partial_fraction
            partial_size = min(partial_size, active.remaining_size)
            partial_exit_price = _trigger_exit_price(active.setup, active.first_liquidity_price, config)
            partial_pnl = _segment_pnl(active.entry_price, partial_exit_price, side, partial_size)
            partial_fee = config.commission_per_unit * partial_size
            active.realized_pnl += partial_pnl - partial_fee
            active.realized_fees += partial_fee
            active.remaining_size -= partial_size
            active.partial_taken = partial_size > 0
            events.append({"type": "partial", "price": partial_exit_price, "size": partial_size})
            if active.setup.context.get("breakeven_after_first_draw"):
                active.current_stop = active.entry_price
                active.breakeven_activated = True

    if not active.trail_activated and _rr_threshold_reached(active, bar, threshold=2.0):
        if active.setup.context.get("runner_trail_rule") == "atr_placeholder":
            active.trail_activated = True
            _update_trail_stop(active, bar)

    if active.trail_activated:
        _update_trail_stop(active, bar)
        if active.trail_stop is not None:
            if side is Side.LONG:
                active.current_stop = max(active.current_stop, active.trail_stop)
            else:
                active.current_stop = min(active.current_stop, active.trail_stop)

    if active.remaining_size <= 0:
        return {
            "closed": False,
            "events": events,
        }

    stop_hit = _target_hit(side, bar, active.current_stop, stop_check=True)
    runner_target = _next_runner_target(active)
    if runner_target is not None:
        target_price = float(runner_target["price"])
        target_reason = ExitReason.RUNNER_TARGET
        target_hit = _target_hit(side, bar, target_price)
        runner_target_hit = True
    elif active.first_liquidity_price is None:
        target_price = float(active.setup.target_reference)
        target_reason = ExitReason.TARGET
        target_hit = _target_hit(side, bar, target_price)
        runner_target_hit = False
    else:
        target_price = None
        target_reason = ExitReason.TARGET
        target_hit = False
        runner_target_hit = False

    if stop_hit and target_hit:
        if config.intrabar_exit_conflict_policy is IntrabarExitConflictPolicy.TARGET_FIRST:
            return _close_remaining(active, bar, float(target_price), target_reason, config, events, runner_target_hit=runner_target_hit)
        reason = ExitReason.BREAKEVEN if active.breakeven_activated and abs(active.current_stop - active.entry_price) < 1e-9 else (ExitReason.TRAIL_STOP if active.trail_activated else ExitReason.STOP)
        return _close_remaining(active, bar, active.current_stop, reason, config, events)
    if target_hit:
        return _close_remaining(active, bar, float(target_price), target_reason, config, events, runner_target_hit=runner_target_hit)
    if stop_hit:
        reason = ExitReason.BREAKEVEN if active.breakeven_activated and abs(active.current_stop - active.entry_price) < 1e-9 else (ExitReason.TRAIL_STOP if active.trail_activated else ExitReason.STOP)
        return _close_remaining(active, bar, active.current_stop, reason, config, events)
    return {"closed": False, "events": events}


def exit_position(active: ActivePosition, exit_time: pd.Timestamp, close_result: dict, config: ExecutionConfig) -> Trade:
    exit_price = float(close_result["exit_price"])
    exit_reason = close_result["exit_reason"]
    final_size = float(close_result["final_size"])
    side = active.setup.direction
    final_pnl = _segment_pnl(active.entry_price, exit_price, side, final_size)
    final_fee = config.commission_per_unit * final_size
    gross_pnl = active.realized_pnl + final_pnl + active.realized_fees
    net_pnl = active.realized_pnl + final_pnl - final_fee
    total_fees = active.realized_fees + final_fee
    total_slippage = (2.0 * config.slippage_per_unit * active.size)
    risk_per_unit = abs(active.setup.entry_reference - active.setup.stop_reference)
    pnl_r = net_pnl / (risk_per_unit * active.size) if risk_per_unit > 0 else None
    return_pct = ((exit_price - active.entry_price) / active.entry_price) * (1.0 if side is Side.LONG else -1.0) if active.entry_price else None
    quality_score = active.setup.features.get("quality_score")
    if quality_score is None:
        quality_score = active.setup.context.get("quality_score")
    return Trade(
        trade_id=f"trade-{uuid4().hex[:12]}",
        setup_id=active.setup.setup_id,
        setup_name=active.setup.setup_name,
        symbol=active.setup.symbol,
        contract=active.setup.contract,
        side=side,
        entry_time=active.entry_time.to_pydatetime(),
        entry_price=float(active.entry_price),
        size=float(active.size),
        exit_time=exit_time.to_pydatetime(),
        exit_price=exit_price,
        fees=float(total_fees),
        slippage=float(total_slippage),
        pnl=float(net_pnl),
        pnl_r=float(pnl_r) if pnl_r is not None else None,
        return_pct=float(return_pct) if return_pct is not None else None,
        bars_held=active.bars_held,
        exit_reason=exit_reason.value,
        quality_score=float(quality_score) if isinstance(quality_score, (int, float)) else None,
        first_liquidity_target=active.first_liquidity_target,
        partial_taken=active.partial_taken,
        breakeven_activated=active.breakeven_activated,
        runner_target_hit=close_result.get("runner_target_hit", False),
        trail_activated=active.trail_activated,
        stop_mode_used=str(active.setup.context.get("formal_stop_rule", active.setup.context.get("stop_mode"))),
        metadata={
            "gross_pnl": float(gross_pnl),
            "partial_events": close_result.get("events", []),
            "runner_targets": active.runner_targets,
        },
    )


def close_on_bar_value(active: ActivePosition, bar: pd.Series, field: str, reason: ExitReason, config: ExecutionConfig) -> dict:
    raw_price = float(bar[field])
    exit_price = raw_price - config.slippage_per_unit if active.setup.direction is Side.LONG else raw_price + config.slippage_per_unit
    return {
        "exit_price": exit_price,
        "exit_reason": reason,
        "final_size": active.remaining_size,
        "runner_target_hit": False,
        "events": [],
    }


def _close_remaining(active: ActivePosition, bar: pd.Series, trigger_price: float, reason: ExitReason, config: ExecutionConfig, events: list[dict], runner_target_hit: bool = False) -> dict:
    exit_price = _trigger_exit_price(active.setup, trigger_price, config)
    return {
        "exit_price": exit_price,
        "exit_reason": reason,
        "final_size": active.remaining_size,
        "runner_target_hit": runner_target_hit,
        "events": events,
    }


def _target_hit(side: Side, bar: pd.Series, price: float, stop_check: bool = False) -> bool:
    if side is Side.LONG:
        return float(bar["low"]) <= price if stop_check else float(bar["high"]) >= price
    return float(bar["high"]) >= price if stop_check else float(bar["low"]) <= price


def _segment_pnl(entry_price: float, exit_price: float, side: Side, size: float) -> float:
    multiplier = 1.0 if side is Side.LONG else -1.0
    return (exit_price - entry_price) * multiplier * size


def _rr_threshold_reached(active: ActivePosition, bar: pd.Series, threshold: float) -> bool:
    risk = abs(active.entry_price - active.setup.stop_reference)
    if risk <= 0:
        return False
    if active.setup.direction is Side.LONG:
        return (float(bar["high"]) - active.entry_price) / risk >= threshold
    return (active.entry_price - float(bar["low"])) / risk >= threshold


def _update_trail_stop(active: ActivePosition, bar: pd.Series) -> None:
    atr_mult = active.setup.context.get("runner_trail_atr_multiple")
    atr_value = bar.get("atr")
    if atr_mult is None or pd.isna(atr_value):
        return
    if active.setup.direction is Side.LONG:
        trail = float(bar["close"] - float(atr_mult) * float(atr_value))
        active.trail_stop = trail if active.trail_stop is None else max(active.trail_stop, trail)
    else:
        trail = float(bar["close"] + float(atr_mult) * float(atr_value))
        active.trail_stop = trail if active.trail_stop is None else min(active.trail_stop, trail)


def _next_runner_target(active: ActivePosition) -> dict[str, float | str] | None:
    return active.runner_targets[0] if active.runner_targets else None


def _apply_entry_slippage(price: float, side: Side, slippage_per_unit: float) -> float:
    return price + slippage_per_unit if side is Side.LONG else price - slippage_per_unit


def _trigger_exit_price(setup: SetupEvent, trigger_price: float, config: ExecutionConfig) -> float:
    return trigger_price - config.slippage_per_unit if setup.direction is Side.LONG else trigger_price + config.slippage_per_unit

