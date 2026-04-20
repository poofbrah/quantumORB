from __future__ import annotations

from datetime import datetime, timedelta

from data.schemas import Side, Trade
from evaluation.metrics import calculate_summary_metrics


def make_trade(trade_id: str, pnl: float, return_pct: float) -> Trade:
    now = datetime(2024, 1, 2, 9, 30)
    return Trade(
        trade_id=trade_id,
        setup_id=trade_id,
        setup_name="orb",
        symbol="NQ",
        contract="NQH4",
        side=Side.LONG,
        entry_time=now,
        entry_price=100.0,
        size=1.0,
        exit_time=now + timedelta(minutes=5),
        exit_price=101.0,
        pnl=pnl,
        return_pct=return_pct,
        bars_held=5,
        exit_reason="target" if pnl > 0 else "stop",
    )


def test_summary_metrics_include_sortino_and_calmar() -> None:
    trades = [
        make_trade("t1", 100.0, 0.01),
        make_trade("t2", -50.0, -0.005),
        make_trade("t3", 120.0, 0.012),
        make_trade("t4", -20.0, -0.002),
    ]
    equity_curve = [
        (datetime(2024, 1, 2, 9, 35), 100100.0),
        (datetime(2024, 1, 2, 9, 40), 100050.0),
        (datetime(2024, 1, 2, 9, 45), 100170.0),
        (datetime(2024, 1, 2, 9, 50), 100150.0),
    ]

    metrics = calculate_summary_metrics(trades, equity_curve, 100000.0)

    assert metrics["sharpe"] is not None
    assert metrics["sortino"] is not None
    assert metrics["calmar"] is not None
    assert metrics["sortino_ratio"] == metrics["sortino"]
    assert metrics["calmar_ratio"] == metrics["calmar"]
