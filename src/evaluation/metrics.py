from __future__ import annotations

import math
from statistics import mean

from data.schemas import Trade


def calculate_summary_metrics(
    trades: list[Trade],
    equity_curve: list[tuple[object, float]],
    initial_capital: float,
) -> dict[str, float | int | None]:
    pnls = [float(trade.pnl or 0.0) for trade in trades]
    gross_trade_pnls = [float(trade.metadata.get("gross_pnl", trade.pnl or 0.0)) for trade in trades]
    gross_profit = sum(pnl for pnl in gross_trade_pnls if pnl > 0)
    gross_loss = sum(pnl for pnl in gross_trade_pnls if pnl < 0)
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    return_series = [float(trade.return_pct or 0.0) for trade in trades]
    durations = [trade.bars_held for trade in trades]

    ending_equity = equity_curve[-1][1] if equity_curve else initial_capital
    max_drawdown = min((value for _, value in _build_drawdowns(equity_curve)), default=0.0)

    return {
        "total_return": ((ending_equity - initial_capital) / initial_capital) if initial_capital else None,
        "cumulative_pnl": sum(pnls),
        "win_rate": (len(wins) / len(trades)) if trades else None,
        "average_win": mean(wins) if wins else 0.0,
        "average_loss": mean(losses) if losses else 0.0,
        "profit_factor": (gross_profit / abs(gross_loss)) if gross_loss < 0 else None,
        "expectancy": mean(pnls) if trades else None,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": _sharpe_ratio(return_series),
        "average_trade_duration": mean(durations) if durations else 0.0,
        "number_of_trades": len(trades),
        "gross_pnl": sum(gross_trade_pnls),
        "net_pnl": sum(pnls),
    }


def _sharpe_ratio(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    avg = mean(returns)
    variance = sum((value - avg) ** 2 for value in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return None
    return math.sqrt(len(returns)) * avg / std


def _build_drawdowns(equity_curve: list[tuple[object, float]]) -> list[tuple[object, float]]:
    drawdowns: list[tuple[object, float]] = []
    peak: float | None = None
    for timestamp, equity in equity_curve:
        if peak is None or equity > peak:
            peak = equity
        drawdown = 0.0 if peak == 0 else (equity - peak) / peak
        drawdowns.append((timestamp, drawdown))
    return drawdowns
