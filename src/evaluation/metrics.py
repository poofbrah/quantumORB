from __future__ import annotations

import math
from statistics import mean

from data.schemas import Trade


STOP_LIKE_EXIT_REASONS = {"stop", "breakeven"}
TARGET_LIKE_EXIT_REASONS = {"target", "runner_target"}


def calculate_summary_metrics(
    trades: list[Trade],
    equity_curve: list[tuple[object, float]],
    initial_capital: float,
) -> dict[str, float | int | None]:
    pnls = [float(trade.pnl or 0.0) for trade in trades]
    return_series = [float(trade.return_pct or 0.0) for trade in trades]
    durations = [trade.bars_held for trade in trades]
    pnl_rs = [float(trade.pnl_r) for trade in trades if trade.pnl_r is not None]
    mfe_values = [float(trade.max_favorable_excursion) for trade in trades if trade.max_favorable_excursion is not None]
    mae_values = [float(trade.max_adverse_excursion) for trade in trades if trade.max_adverse_excursion is not None]
    mfe_r_values = [float(trade.max_favorable_excursion_r) for trade in trades if trade.max_favorable_excursion_r is not None]
    mae_r_values = [float(trade.max_adverse_excursion_r) for trade in trades if trade.max_adverse_excursion_r is not None]
    unrealized_profit_values = [float(trade.max_unrealized_profit) for trade in trades if trade.max_unrealized_profit is not None]
    unrealized_loss_values = [float(trade.max_unrealized_loss) for trade in trades if trade.max_unrealized_loss is not None]

    gross_profit = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = sum(pnl for pnl in pnls if pnl < 0)
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]

    ending_equity = equity_curve[-1][1] if equity_curve else initial_capital
    drawdowns = _build_drawdowns(equity_curve)
    max_drawdown_abs = min((value for _, value, _ in drawdowns), default=0.0)
    max_drawdown_pct = min((value for _, _, value in drawdowns), default=0.0)
    total_return = ((ending_equity - initial_capital) / initial_capital) if initial_capital else None

    target_exit_count = _count_exits(trades, TARGET_LIKE_EXIT_REASONS)
    stop_exit_count = _count_exits(trades, STOP_LIKE_EXIT_REASONS)
    trail_exit_count = _count_exits(trades, {"trail_stop"})
    session_end_count = _count_exits(trades, {"session_end"})
    total_trades = len(trades)

    sharpe = _sharpe_ratio(return_series)
    sortino = _sortino_ratio(return_series)
    calmar = _calmar_ratio(total_return, max_drawdown_pct)

    return {
        "total_return": total_return,
        "cumulative_pnl": sum(pnls),
        "gross_pnl": sum(pnls),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": sum(pnls),
        "win_rate": (len(wins) / total_trades) if total_trades else None,
        "loss_rate": (len(losses) / total_trades) if total_trades else None,
        "average_win": mean(wins) if wins else 0.0,
        "average_loss": mean(losses) if losses else 0.0,
        "average_trade_pnl": mean(pnls) if total_trades else None,
        "average_r": mean(pnl_rs) if pnl_rs else None,
        "profit_factor": (gross_profit / abs(gross_loss)) if gross_loss < 0 else None,
        "expectancy": mean(pnls) if total_trades else None,
        "max_drawdown": max_drawdown_abs,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "sharpe_ratio": sharpe,
        "sortino": sortino,
        "sortino_ratio": sortino,
        "calmar": calmar,
        "calmar_ratio": calmar,
        "average_trade_duration": mean(durations) if durations else 0.0,
        "number_of_trades": total_trades,
        "max_unrealized_profit": max(unrealized_profit_values, default=0.0),
        "max_unrealized_loss": max(unrealized_loss_values, default=0.0),
        "average_mfe": mean(mfe_values) if mfe_values else 0.0,
        "average_mae": mean(mae_values) if mae_values else 0.0,
        "average_mfe_r": mean(mfe_r_values) if mfe_r_values else 0.0,
        "average_mae_r": mean(mae_r_values) if mae_r_values else 0.0,
        "percent_partial_taken": (sum(1 for trade in trades if trade.partial_taken) / total_trades) if total_trades else None,
        "percent_target_exits": (target_exit_count / total_trades) if total_trades else None,
        "percent_stop_exits": (stop_exit_count / total_trades) if total_trades else None,
        "percent_trailing_stop_exits": (trail_exit_count / total_trades) if total_trades else None,
        "percent_session_end_exits": (session_end_count / total_trades) if total_trades else None,
    }


def _count_exits(trades: list[Trade], reasons: set[str]) -> int:
    return sum(1 for trade in trades if trade.exit_reason in reasons)


def _sharpe_ratio(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    avg = mean(returns)
    variance = sum((value - avg) ** 2 for value in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return None
    return math.sqrt(len(returns)) * avg / std


def _sortino_ratio(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    avg = mean(returns)
    downside = [min(value, 0.0) for value in returns]
    downside_variance = sum(value**2 for value in downside) / len(downside)
    downside_std = math.sqrt(downside_variance)
    if downside_std == 0:
        return None
    return math.sqrt(len(returns)) * avg / downside_std


def _calmar_ratio(total_return: float | None, max_drawdown_pct: float) -> float | None:
    if total_return is None or max_drawdown_pct >= 0:
        return None
    if max_drawdown_pct == 0:
        return None
    return total_return / abs(max_drawdown_pct)


def _build_drawdowns(equity_curve: list[tuple[object, float]]) -> list[tuple[object, float, float]]:
    drawdowns: list[tuple[object, float, float]] = []
    peak: float | None = None
    for timestamp, equity in equity_curve:
        if peak is None or equity > peak:
            peak = equity
        drawdown_abs = equity - peak
        drawdown_pct = 0.0 if peak == 0 else drawdown_abs / peak
        drawdowns.append((timestamp, drawdown_abs, drawdown_pct))
    return drawdowns
