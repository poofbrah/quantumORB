from __future__ import annotations

import pandas as pd

from data.schemas import Trade


def build_equity_curve(
    price_frame: pd.DataFrame,
    trades: list[Trade],
    initial_capital: float,
) -> list[tuple[pd.Timestamp, float]]:
    ordered_bars = price_frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    realized_pnl_by_time: dict[pd.Timestamp, float] = {}
    for trade in trades:
        exit_time = pd.Timestamp(trade.exit_time)
        realized_pnl_by_time[exit_time] = realized_pnl_by_time.get(exit_time, 0.0) + float(trade.pnl or 0.0)

    equity = float(initial_capital)
    curve: list[tuple[pd.Timestamp, float]] = []
    for _, row in ordered_bars.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        equity += realized_pnl_by_time.get(timestamp, 0.0)
        curve.append((timestamp, equity))
    return curve


def build_drawdown_curve(equity_curve: list[tuple[pd.Timestamp, float]]) -> list[tuple[pd.Timestamp, float]]:
    drawdowns: list[tuple[pd.Timestamp, float]] = []
    running_peak: float | None = None
    for timestamp, equity in equity_curve:
        if running_peak is None or equity > running_peak:
            running_peak = equity
        drawdown = 0.0 if running_peak == 0 else (equity - running_peak) / running_peak
        drawdowns.append((timestamp, drawdown))
    return drawdowns
