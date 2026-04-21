from __future__ import annotations

import pandas as pd

from reporting.setup_quality_research import build_paper_experiment_comparison_frame, select_best_threshold_row


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


def test_build_paper_experiment_comparison_frame_includes_raw_and_filtered_rows() -> None:
    frame = build_paper_experiment_comparison_frame(
        baseline_metrics={
            "trades_executed": 100,
            "win_rate": 0.5,
            "profit_factor": 1.2,
            "net_pnl": 250.0,
            "max_drawdown": -100.0,
            "sharpe": 0.8,
            "sortino": 1.0,
            "calmar": 1.5,
        },
        baseline_best_thresholds={
            "logistic_regression": {
                "best_threshold": 0.7,
                "trades_executed": 40,
                "win_rate": 0.6,
                "profit_factor": 1.4,
                "net_pnl": 300.0,
                "max_drawdown": -80.0,
                "sharpe": 1.0,
                "sortino": 1.2,
                "calmar": 1.8,
            }
        },
        lstm_best_threshold={
            "threshold": 0.75,
            "trades_executed": 25,
            "win_rate": 0.64,
            "profit_factor": 1.5,
            "net_pnl": 280.0,
            "max_drawdown": -70.0,
            "sharpe": 1.1,
            "sortino": 1.3,
            "calmar": 2.0,
        },
    )

    assert frame["model_name"].tolist() == ["raw_strategy", "logistic_regression", "lstm"]
    assert frame["selection_rule"].tolist() == ["all_setups", "probability_filter", "probability_filter"]
    assert pd.isna(frame.loc[0, "probability_threshold"])
    assert frame.loc[1, "probability_threshold"] == 0.7
    assert frame.loc[2, "probability_threshold"] == 0.75
