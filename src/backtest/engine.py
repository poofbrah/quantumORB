from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pandas as pd

from data.schemas import BacktestResult, SetupEvent, Trade
from evaluation.metrics import calculate_summary_metrics
from execution.simulator import (
    ActivePosition,
    EntryConvention,
    ExecutionConfig,
    ExitReason,
    IntrabarExitConflictPolicy,
    close_on_bar_value,
    enter_position,
    evaluate_position_on_bar,
    exit_position,
)
from portfolio.ledger import build_drawdown_curve, build_equity_curve


@dataclass(slots=True)
class BacktestConfig:
    initial_capital: float = 100000.0
    position_size: float = 1.0
    commission_per_unit: float = 0.0
    slippage_per_unit: float = 0.0
    one_position_only: bool = True
    exit_on_session_end: bool = True
    entry_convention: EntryConvention = EntryConvention.NEXT_BAR_OPEN
    intrabar_exit_conflict_policy: IntrabarExitConflictPolicy = IntrabarExitConflictPolicy.STOP_FIRST

    def to_execution_config(self) -> ExecutionConfig:
        return ExecutionConfig(
            entry_convention=self.entry_convention,
            slippage_per_unit=self.slippage_per_unit,
            commission_per_unit=self.commission_per_unit,
            position_size=self.position_size,
            one_position_only=self.one_position_only,
            exit_on_session_end=self.exit_on_session_end,
            intrabar_exit_conflict_policy=self.intrabar_exit_conflict_policy,
        )


@dataclass(slots=True)
class BacktestRunConfig:
    strategy_name: str = "orb"


class BarBacktestEngine:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.execution_config = self.config.to_execution_config()

    def run(self, price_frame: pd.DataFrame, setups: list[SetupEvent], run_config: BacktestRunConfig | None = None) -> BacktestResult:
        run_config = run_config or BacktestRunConfig()
        ordered = price_frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
        pending = self._build_pending_entries(ordered, setups)
        trades: list[Trade] = []
        active: ActivePosition | None = None
        exited_this_bar = False

        for idx, bar in ordered.iterrows():
            exited_this_bar = False
            if active is not None and bar["symbol"] == active.setup.symbol:
                active.bars_held += 1
                result = evaluate_position_on_bar(active, bar, self.execution_config)
                if result is not None and result.get("closed") is not False:
                    trades.append(exit_position(active, pd.Timestamp(bar["timestamp"]), result, self.execution_config))
                    active = None
                    exited_this_bar = True
                elif self.config.exit_on_session_end and self._is_session_end_bar(ordered, idx, active.setup.symbol):
                    close_result = close_on_bar_value(active, bar, "close", ExitReason.SESSION_END, self.execution_config)
                    trades.append(exit_position(active, pd.Timestamp(bar["timestamp"]), close_result, self.execution_config))
                    active = None
                    exited_this_bar = True

            if active is None and not exited_this_bar:
                for setup in pending.get(idx, []):
                    active = enter_position(setup, bar, self.execution_config)
                    active.bars_held += 1
                    result = evaluate_position_on_bar(active, bar, self.execution_config)
                    if result is not None and result.get("closed") is not False:
                        trades.append(exit_position(active, pd.Timestamp(bar["timestamp"]), result, self.execution_config))
                        active = None
                        exited_this_bar = True
                    elif self.config.exit_on_session_end and self._is_session_end_bar(ordered, idx, setup.symbol):
                        close_result = close_on_bar_value(active, bar, "close", ExitReason.SESSION_END, self.execution_config)
                        trades.append(exit_position(active, pd.Timestamp(bar["timestamp"]), close_result, self.execution_config))
                        active = None
                        exited_this_bar = True
                    break

        if active is not None:
            last_bar = ordered[ordered["symbol"] == active.setup.symbol].iloc[-1]
            close_result = close_on_bar_value(active, last_bar, "close", ExitReason.DATA_END, self.execution_config)
            trades.append(exit_position(active, pd.Timestamp(last_bar["timestamp"]), close_result, self.execution_config))

        equity_curve = build_equity_curve(ordered, trades, self.config.initial_capital)
        drawdown_curve = build_drawdown_curve(equity_curve)
        metrics = calculate_summary_metrics(trades, equity_curve, self.config.initial_capital)
        symbol = setups[0].symbol if setups else (str(ordered.iloc[0]["symbol"]) if not ordered.empty else "")
        return BacktestResult(
            run_id=f"run-{uuid4().hex[:12]}",
            strategy_name=run_config.strategy_name,
            symbol=symbol,
            start=pd.Timestamp(ordered.iloc[0]["timestamp"]).to_pydatetime() if not ordered.empty else pd.Timestamp.utcnow().to_pydatetime(),
            end=pd.Timestamp(ordered.iloc[-1]["timestamp"]).to_pydatetime() if not ordered.empty else pd.Timestamp.utcnow().to_pydatetime(),
            total_trades=int(metrics["number_of_trades"] or 0),
            net_pnl=float(metrics["net_pnl"] or 0.0),
            gross_pnl=float(metrics["gross_pnl"] or 0.0),
            total_return=metrics["total_return"],
            win_rate=metrics["win_rate"],
            average_win=metrics["average_win"],
            average_loss=metrics["average_loss"],
            profit_factor=metrics["profit_factor"],
            expectancy=metrics["expectancy"],
            max_drawdown=metrics["max_drawdown"],
            sharpe_ratio=metrics["sharpe_ratio"],
            average_trade_duration=metrics["average_trade_duration"],
            trades=trades,
            equity_curve=[(timestamp.to_pydatetime(), value) for timestamp, value in equity_curve],
            drawdown_curve=[(timestamp.to_pydatetime(), value) for timestamp, value in drawdown_curve],
            metadata={
                "entry_convention": self.config.entry_convention.value,
                "intrabar_exit_conflict_policy": self.config.intrabar_exit_conflict_policy.value,
                "one_position_only": self.config.one_position_only,
                "initial_capital": self.config.initial_capital,
            },
        )

    def _build_pending_entries(self, ordered: pd.DataFrame, setups: list[SetupEvent]) -> dict[int, list[SetupEvent]]:
        pending: dict[int, list[SetupEvent]] = {}
        ordered_setups = sorted(setups, key=lambda setup: (setup.timestamp, setup.symbol, setup.setup_id))
        for setup in ordered_setups:
            mask = (ordered["symbol"] == setup.symbol) & (ordered["timestamp"] > pd.Timestamp(setup.timestamp))
            eligible = ordered.index[mask]
            if len(eligible) == 0:
                continue
            pending.setdefault(int(eligible[0]), []).append(setup)
        return pending

    def _is_session_end_bar(self, ordered: pd.DataFrame, idx: int, symbol: str) -> bool:
        current = ordered.iloc[idx]
        later_same_symbol = ordered[(ordered.index > idx) & (ordered["symbol"] == symbol)]
        if later_same_symbol.empty:
            return True
        next_row = later_same_symbol.iloc[0]
        return pd.Timestamp(next_row["session_date"]) != pd.Timestamp(current["session_date"])
