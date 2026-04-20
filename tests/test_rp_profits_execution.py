from __future__ import annotations

import pandas as pd

from backtest.engine import BarBacktestEngine, BacktestConfig, BacktestRunConfig
from data.schemas import SetupEvent, SetupStatus, Side
from evaluation.metrics import calculate_summary_metrics


TIMEZONE = "America/New_York"


def make_price_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 08:59", periods=4, freq="min", tz=TIMEZONE)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["NQ"] * 4,
            "open": [99.9, 100.0, 100.1, 101.4],
            "high": [100.0, 100.2, 102.2, 101.6],
            "low": [99.8, 99.8, 100.0, 100.4],
            "close": [99.9, 100.0, 101.6, 100.6],
            "volume": [100, 100, 100, 100],
            "session_date": [timestamps[0].normalize()] * 4,
            "atr": [1.0, 1.0, 1.0, 1.0],
        }
    )


def make_setup() -> SetupEvent:
    timestamp = pd.Timestamp("2024-01-02 08:59", tz=TIMEZONE)
    return SetupEvent(
        setup_id="rp-test",
        setup_name="rp_profits_8am_orb",
        symbol="NQ",
        contract="NQH4",
        timestamp=timestamp.to_pydatetime(),
        session_date=timestamp.normalize().to_pydatetime(),
        direction=Side.LONG,
        status=SetupStatus.CANDIDATE,
        entry_reference=100.0,
        stop_reference=98.0,
        target_reference=104.0,
        features={"range_width": 2.0, "range_width_atr": 2.0, "displacement_strength": 0.8, "vwap_at_entry": 100.0, "retracement_depth": 1.3},
        context={"entry_fill_mode": "next_bar_open", "first_liquidity_target": "one_r", "first_liquidity_target_price": 102.0, "partial_take_profit_fraction": 0.5, "breakeven_after_first_draw": True, "runner_trail_rule": "atr_placeholder", "runner_trail_atr_multiple": 1.0, "disable_default_target_exit": True, "strategy_profile": "rp_profits_8am_orb"},
    )


def make_reentry_setup() -> SetupEvent:
    timestamp = pd.Timestamp("2024-01-02 09:05", tz=TIMEZONE)
    return SetupEvent(
        setup_id="rp-reentry",
        setup_name="rp_profits_8am_orb",
        symbol="NQ",
        contract="NQH4",
        timestamp=timestamp.to_pydatetime(),
        session_date=timestamp.normalize().to_pydatetime(),
        direction=Side.SHORT,
        status=SetupStatus.CANDIDATE,
        entry_reference=100.6,
        stop_reference=101.6,
        target_reference=99.0,
        features={"range_width": 2.0, "range_width_atr": 2.0},
        context={"entry_fill_mode": "signal_close", "skip_entry_bar_management": True, "first_liquidity_target": "ny_vwap", "first_liquidity_target_price": 100.0, "partial_take_profit_fraction": 0.5, "breakeven_after_first_draw": True, "disable_default_target_exit": True, "runner_targets": [{"name": "range_runner", "price": 99.0}], "strategy_profile": "rp_profits_8am_orb", "entry_family": "range_reentry"},
    )


def make_reentry_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:04", periods=3, freq="min", tz=TIMEZONE)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["NQ"] * 3,
            "open": [101.2, 101.4, 99.8],
            "high": [101.4, 101.6, 100.1],
            "low": [100.8, 100.2, 98.8],
            "close": [101.1, 100.6, 99.0],
            "volume": [100, 100, 100],
            "session_date": [timestamps[0].normalize()] * 3,
            "atr": [1.0, 1.0, 1.0],
        }
    )


def test_partial_at_1r_and_breakeven_and_trailing_stop() -> None:
    frame = make_price_frame()
    result = BarBacktestEngine(BacktestConfig(position_size=2.0)).run(frame, [make_setup()], BacktestRunConfig(strategy_name="rp-test"))
    trade = result.trades[0]
    assert trade.partial_taken is True
    assert trade.partial_exit_price == 102.0
    assert trade.partial_exit_fraction == 0.5
    assert trade.moved_to_breakeven is True
    assert trade.trailing_stop_used is True
    assert trade.exit_reason == "trail_stop"
    assert trade.runner_exit_price == 100.6
    assert round(float(trade.exit_price_blended), 4) == 101.3


def test_mfe_mae_and_unrealized_tracking() -> None:
    frame = make_price_frame()
    result = BarBacktestEngine(BacktestConfig(position_size=2.0)).run(frame, [make_setup()], BacktestRunConfig(strategy_name="rp-test"))
    trade = result.trades[0]
    assert round(float(trade.max_favorable_excursion), 4) == 2.2
    assert round(float(trade.max_adverse_excursion), 4) == 0.2
    assert round(float(trade.max_favorable_excursion_r), 4) == 1.1
    assert round(float(trade.max_adverse_excursion_r), 4) == 0.1
    assert round(float(trade.max_unrealized_profit), 4) == 4.4
    assert round(float(trade.max_unrealized_loss), 4) == 0.4


def test_range_reentry_tp1_partial_then_runner_to_opposite_side() -> None:
    frame = make_reentry_frame()
    result = BarBacktestEngine(BacktestConfig(position_size=2.0)).run(frame, [make_reentry_setup()], BacktestRunConfig(strategy_name="rp-reentry"))
    trade = result.trades[0]
    assert trade.partial_taken is True
    assert trade.partial_exit_price == 100.0
    assert trade.moved_to_breakeven is True
    assert trade.exit_reason == "runner_target"
    assert trade.runner_exit_price == 99.0


def test_summary_metric_generation_for_real_trader_fields() -> None:
    frame = make_price_frame()
    result = BarBacktestEngine(BacktestConfig(position_size=2.0)).run(frame, [make_setup()], BacktestRunConfig(strategy_name="rp-test"))
    metrics = calculate_summary_metrics(result.trades, result.equity_curve, 100000.0)
    assert metrics["number_of_trades"] == 1
    assert metrics["percent_partial_taken"] == 1.0
    assert metrics["percent_trailing_stop_exits"] == 1.0
    assert round(float(metrics["average_mfe"]), 4) == 2.2
    assert round(float(metrics["average_mae"]), 4) == 0.2

