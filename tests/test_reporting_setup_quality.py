from __future__ import annotations

import pandas as pd

from reporting.setup_quality_research import select_best_threshold_row


def test_select_best_threshold_row_prefers_primary_metrics_then_trade_count() -> None:
    frame = pd.DataFrame(
        [
            {"threshold": 0.55, "profit_factor": 1.10, "sharpe": 0.70, "net_pnl": 120.0, "trades_executed": 15},
            {"threshold": 0.60, "profit_factor": 1.10, "sharpe": 0.70, "net_pnl": 120.0, "trades_executed": 20},
            {"threshold": 0.65, "profit_factor": 1.20, "sharpe": 0.50, "net_pnl": 80.0, "trades_executed": 8},
        ]
    )

    best = select_best_threshold_row(frame)

    assert float(best["threshold"]) == 0.65


def test_select_best_threshold_row_uses_lower_threshold_as_final_tiebreak() -> None:
    frame = pd.DataFrame(
        [
            {"threshold": 0.55, "profit_factor": 1.10, "sharpe": 0.70, "net_pnl": 120.0, "trades_executed": 15},
            {"threshold": 0.60, "profit_factor": 1.10, "sharpe": 0.70, "net_pnl": 120.0, "trades_executed": 15},
        ]
    )

    best = select_best_threshold_row(frame)

    assert float(best["threshold"]) == 0.55
