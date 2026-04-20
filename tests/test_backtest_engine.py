from __future__ import annotations

import pandas as pd

from backtest.engine import BarBacktestEngine, BacktestConfig, BacktestRunConfig
from data.schemas import SetupEvent, SetupStatus, Side, Trade
from evaluation.metrics import calculate_summary_metrics
from execution.simulator import IntrabarExitConflictPolicy


def make_price_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30", periods=6, freq="min", tz="America/New_York")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["ES"] * 6,
            "session_date": [pd.Timestamp("2024-01-02", tz="America/New_York")] * 6,
            "open": [100.0, 101.0, 101.2, 101.8, 102.1, 102.4],
            "high": [100.5, 101.4, 101.9, 102.5, 103.2, 103.0],
            "low": [99.8, 100.7, 100.9, 101.5, 101.9, 102.0],
            "close": [100.2, 101.1, 101.7, 102.2, 102.8, 102.6],
            "volume": [100, 110, 120, 130, 140, 150],
            "contract": ["ESH4"] * 6,
        }
    )


def make_setup(timestamp: pd.Timestamp, side: Side = Side.LONG, target: float = 103.0, stop: float = 100.0) -> SetupEvent:
    return SetupEvent(
        setup_id=f"setup-{timestamp.minute}-{side.value}",
        setup_name="orb",
        symbol="ES",
        contract="ESH4",
        timestamp=timestamp.to_pydatetime(),
        session_date=timestamp.normalize().to_pydatetime(),
        direction=side,
        status=SetupStatus.CANDIDATE,
        entry_reference=101.0 if side is Side.LONG else 101.5,
        stop_reference=stop,
        target_reference=target,
        features={},
        context={},
    )


def test_trade_entry_occurs_at_next_bar_open() -> None:
    frame = make_price_frame()
    setup = make_setup(frame.loc[0, "timestamp"])

    result = BarBacktestEngine(BacktestConfig()).run(frame, [setup], BacktestRunConfig(strategy_name="orb"))

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_time == frame.loc[1, "timestamp"].to_pydatetime()
    assert trade.entry_price == frame.loc[1, "open"]


def test_limit_touch_entry_uses_entry_reference_on_touch_bar() -> None:
    frame = make_price_frame().copy()
    frame.loc[1, "low"] = 100.0
    frame.loc[1, "high"] = 102.0
    setup = make_setup(frame.loc[1, "timestamp"], side=Side.LONG, target=103.0, stop=99.0)
    setup.entry_reference = 100.5
    setup.context["entry_fill_mode"] = "limit_touch"
    setup.context["skip_entry_bar_management"] = True

    result = BarBacktestEngine(BacktestConfig()).run(frame, [setup])

    assert result.trades[0].entry_time == frame.loc[1, "timestamp"].to_pydatetime()
    assert result.trades[0].entry_price == 100.5


def test_signal_close_entry_uses_same_bar_close() -> None:
    frame = make_price_frame().copy()
    setup = make_setup(frame.loc[1, "timestamp"], side=Side.LONG, target=103.0, stop=99.0)
    setup.context["entry_fill_mode"] = "signal_close"
    setup.context["skip_entry_bar_management"] = True

    result = BarBacktestEngine(BacktestConfig()).run(frame, [setup])

    assert result.trades[0].entry_time == frame.loc[1, "timestamp"].to_pydatetime()
    assert result.trades[0].entry_price == frame.loc[1, "close"]


def test_target_exit_for_long_trade() -> None:
    frame = make_price_frame()
    setup = make_setup(frame.loc[0, "timestamp"], side=Side.LONG, target=103.0, stop=100.0)

    result = BarBacktestEngine(BacktestConfig()).run(frame, [setup])
    trade = result.trades[0]

    assert trade.exit_reason == "target"
    assert trade.exit_price == 103.0
    assert trade.pnl > 0


def test_stop_exit_for_long_trade() -> None:
    frame = make_price_frame().copy()
    frame.loc[2, "low"] = 99.5
    setup = make_setup(frame.loc[0, "timestamp"], side=Side.LONG, target=103.0, stop=100.0)

    result = BarBacktestEngine(BacktestConfig()).run(frame, [setup])
    trade = result.trades[0]

    assert trade.exit_reason == "stop"
    assert trade.exit_price == 100.0
    assert trade.pnl < 0


def test_short_trade_handling() -> None:
    frame = make_price_frame().copy()
    frame["open"] = [103.0, 102.5, 102.0, 101.4, 100.8, 100.4]
    frame["high"] = [103.2, 102.7, 102.1, 101.6, 101.0, 100.7]
    frame["low"] = [102.6, 101.8, 101.2, 100.5, 99.8, 99.4]
    frame["close"] = [102.8, 102.0, 101.4, 100.8, 100.2, 99.7]
    setup = make_setup(frame.loc[0, "timestamp"], side=Side.SHORT, target=100.0, stop=103.0)

    result = BarBacktestEngine(BacktestConfig()).run(frame, [setup])
    trade = result.trades[0]

    assert trade.side is Side.SHORT
    assert trade.entry_price == frame.loc[1, "open"]
    assert trade.exit_reason == "target"
    assert trade.pnl > 0


def test_one_position_mode_prevents_overlapping_trades() -> None:
    frame = make_price_frame()
    setup_a = make_setup(frame.loc[0, "timestamp"], side=Side.LONG, target=103.0, stop=100.0)
    setup_b = make_setup(frame.loc[1, "timestamp"], side=Side.LONG, target=104.0, stop=100.5)

    result = BarBacktestEngine(BacktestConfig(one_position_only=True)).run(frame, [setup_a, setup_b])

    assert len(result.trades) == 1
    assert result.trades[0].setup_id == setup_a.setup_id


def test_no_lookahead_entry_does_not_use_setup_bar_close() -> None:
    frame = make_price_frame().copy()
    frame.loc[0, "close"] = 110.0
    frame.loc[1, "open"] = 101.0
    setup = make_setup(frame.loc[0, "timestamp"], side=Side.LONG, target=103.0, stop=100.0)

    result = BarBacktestEngine(BacktestConfig()).run(frame, [setup])
    assert result.trades[0].entry_price == 101.0


def test_same_bar_exit_conflict_is_conservative_by_default() -> None:
    frame = make_price_frame().copy()
    frame.loc[1, "high"] = 103.2
    frame.loc[1, "low"] = 99.8
    setup = make_setup(frame.loc[0, "timestamp"], side=Side.LONG, target=103.0, stop=100.0)

    result = BarBacktestEngine(BacktestConfig()).run(frame, [setup])
    assert result.trades[0].exit_reason == "stop"


def test_same_bar_exit_conflict_can_be_target_first() -> None:
    frame = make_price_frame().copy()
    frame.loc[1, "high"] = 103.2
    frame.loc[1, "low"] = 99.8
    setup = make_setup(frame.loc[0, "timestamp"], side=Side.LONG, target=103.0, stop=100.0)

    result = BarBacktestEngine(
        BacktestConfig(intrabar_exit_conflict_policy=IntrabarExitConflictPolicy.TARGET_FIRST)
    ).run(frame, [setup])
    assert result.trades[0].exit_reason == "target"


def test_session_end_exit_uses_precomputed_lookup() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 15:58",
            "2024-01-02 15:59",
            "2024-01-03 09:30",
            "2024-01-03 09:31",
        ],
        utc=False,
    ).tz_localize("America/New_York")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["ES"] * 4,
            "session_date": [
                pd.Timestamp("2024-01-02", tz="America/New_York"),
                pd.Timestamp("2024-01-02", tz="America/New_York"),
                pd.Timestamp("2024-01-03", tz="America/New_York"),
                pd.Timestamp("2024-01-03", tz="America/New_York"),
            ],
            "open": [100.0, 100.2, 100.4, 100.6],
            "high": [100.3, 100.4, 100.7, 100.8],
            "low": [99.9, 100.0, 100.3, 100.4],
            "close": [100.1, 100.25, 100.5, 100.7],
            "volume": [100, 100, 100, 100],
            "contract": ["ESH4"] * 4,
        }
    )
    setup = make_setup(frame.loc[0, "timestamp"], side=Side.LONG, target=110.0, stop=95.0)

    result = BarBacktestEngine(BacktestConfig(exit_on_session_end=True)).run(frame, [setup])

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "session_end"
    assert result.trades[0].exit_time == frame.loc[1, "timestamp"].to_pydatetime()


def test_metric_calculation_basics() -> None:
    trades = [
        Trade(
            trade_id="t1",
            setup_id="s1",
            setup_name="orb",
            symbol="ES",
            contract="ESH4",
            side=Side.LONG,
            entry_time=pd.Timestamp("2024-01-02 09:31", tz="America/New_York").to_pydatetime(),
            entry_price=100.0,
            size=1.0,
            exit_time=pd.Timestamp("2024-01-02 09:32", tz="America/New_York").to_pydatetime(),
            exit_price=101.0,
            pnl=1.0,
            return_pct=0.01,
            bars_held=1,
            exit_reason="target",
            fees=0.0,
            slippage=0.0,
            metadata={"gross_pnl": 1.0},
        ),
        Trade(
            trade_id="t2",
            setup_id="s2",
            setup_name="orb",
            symbol="ES",
            contract="ESH4",
            side=Side.SHORT,
            entry_time=pd.Timestamp("2024-01-02 09:33", tz="America/New_York").to_pydatetime(),
            entry_price=101.0,
            size=1.0,
            exit_time=pd.Timestamp("2024-01-02 09:34", tz="America/New_York").to_pydatetime(),
            exit_price=102.0,
            pnl=-1.0,
            return_pct=-0.0099009901,
            bars_held=1,
            exit_reason="stop",
            fees=0.0,
            slippage=0.0,
            metadata={"gross_pnl": -1.0},
        ),
    ]
    equity_curve = [
        (pd.Timestamp("2024-01-02 09:31", tz="America/New_York"), 100000.0),
        (pd.Timestamp("2024-01-02 09:32", tz="America/New_York"), 100001.0),
        (pd.Timestamp("2024-01-02 09:34", tz="America/New_York"), 100000.0),
    ]

    metrics = calculate_summary_metrics(trades, equity_curve, 100000.0)
    assert metrics["number_of_trades"] == 2
    assert metrics["cumulative_pnl"] == 0.0
    assert metrics["win_rate"] == 0.5
    assert metrics["average_win"] == 1.0
    assert metrics["average_loss"] == -1.0
    assert metrics["expectancy"] == 0.0
