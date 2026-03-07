from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SetupStatus(str, Enum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    FILTERED = "filtered"
    EXECUTED = "executed"
    REJECTED = "rejected"


class LabelSource(str, Enum):
    RULE = "rule"
    SIMULATION = "simulation"
    MANUAL = "manual"


class PredictionKind(str, Enum):
    PROBABILITY = "probability"
    SCORE = "score"
    CLASS = "class"


@dataclass(slots=True)
class MarketBar:
    symbol: str
    contract: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_interval: str
    session_date: datetime | None = None
    open_interest: float | None = None
    vwap: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SetupEvent:
    setup_id: str
    setup_name: str
    symbol: str
    contract: str
    timestamp: datetime
    session_date: datetime
    direction: Side
    status: SetupStatus
    entry_reference: float
    stop_reference: float
    target_reference: float
    features: dict[str, float | int | str | bool | None] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def side(self) -> Side:
        return self.direction


@dataclass(slots=True)
class LabeledSetup:
    setup: SetupEvent
    label: int | float
    label_name: str
    label_source: LabelSource
    horizon_bars: int | None = None
    realized_return: float | None = None
    realized_mae: float | None = None
    realized_mfe: float | None = None
    quality_bucket: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trade:
    trade_id: str
    setup_id: str | None
    setup_name: str
    symbol: str
    contract: str
    side: Side
    entry_time: datetime
    entry_price: float
    size: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    fees: float = 0.0
    slippage: float = 0.0
    pnl: float | None = None
    pnl_r: float | None = None
    return_pct: float | None = None
    bars_held: int = 0
    exit_reason: str | None = None
    quality_score: float | None = None
    first_liquidity_target: str | None = None
    partial_taken: bool = False
    breakeven_activated: bool = False
    runner_target_hit: bool = False
    trail_activated: bool = False
    stop_mode_used: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def quantity(self) -> float:
        return self.size


@dataclass(slots=True)
class Prediction:
    prediction_id: str
    setup_id: str
    model_name: str
    timestamp: datetime
    kind: PredictionKind
    value: float | int | str
    confidence: float | None = None
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestResult:
    run_id: str
    strategy_name: str
    symbol: str
    start: datetime
    end: datetime
    total_trades: int
    net_pnl: float
    gross_pnl: float
    total_return: float | None = None
    win_rate: float | None = None
    average_win: float | None = None
    average_loss: float | None = None
    profit_factor: float | None = None
    expectancy: float | None = None
    max_drawdown: float | None = None
    sharpe_ratio: float | None = None
    average_trade_duration: float | None = None
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    drawdown_curve: list[tuple[datetime, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
