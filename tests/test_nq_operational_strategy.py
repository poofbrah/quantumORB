from __future__ import annotations

import pandas as pd

from backtest.engine import BarBacktestEngine, BacktestConfig
from config.models import StrategyConfig
from data.schemas import SetupEvent, SetupStatus, Side
from setups.liquidity import add_liquidity_levels, select_first_liquidity_target
from setups.orb import ORBConfig, ORBSetupDetector


def make_detector_frame() -> pd.DataFrame:
    timestamps = list(pd.date_range("2024-01-01 09:30", periods=4, freq="15min", tz="America/New_York"))
    timestamps += list(pd.date_range("2024-01-02 09:30", periods=5, freq="15min", tz="America/New_York"))
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["NQ"] * 9,
            "contract": ["NQH4"] * 9,
            "session_date": [pd.Timestamp("2024-01-01", tz="America/New_York")] * 4
            + [pd.Timestamp("2024-01-02", tz="America/New_York")] * 5,
            "open": [15280, 15300, 15310, 15320, 15100, 15110, 15120, 15135, 15150],
            "high": [15310, 15340, 15350, 15330, 15140, 15140, 15155, 15180, 15210],
            "low": [15240, 15270, 15290, 15300, 15090, 15100, 15110, 15125, 15140],
            "close": [15300, 15320, 15340, 15310, 15110, 15120, 15130, 15170, 15200],
            "volume": [100] * 9,
        }
    )
    frame["or_high"] = [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 15140.0, 15140.0, 15140.0, 15140.0]
    frame["or_low"] = [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 15090.0, 15090.0, 15090.0, 15090.0]
    frame["or_width"] = [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 50.0, 50.0, 50.0, 50.0]
    frame["or_width_atr"] = [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1.0, 1.0, 1.0, 1.0]
    frame["atr"] = [20.0] * 9
    frame["candle_range"] = frame["high"] - frame["low"]
    frame["candle_body"] = (frame["close"] - frame["open"]).abs()
    frame["lower_wick"] = frame[["open", "close"]].min(axis=1) - frame["low"]
    frame["upper_wick"] = frame["high"] - frame[["open", "close"]].max(axis=1)
    frame["ema_20"] = frame["close"] - 10
    frame["breakout_strength"] = [0.0] * 7 + [0.7, 0.9]
    return frame


def make_liquidity_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-01 09:30",
            "2024-01-01 10:00",
            "2024-01-02 03:00",
            "2024-01-02 04:00",
            "2024-01-02 09:30",
            "2024-01-02 09:45",
            "2024-01-02 10:00",
        ],
        utc=False,
    ).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["NQ"] * len(timestamps),
            "contract": ["NQH4"] * len(timestamps),
            "session_date": [pd.Timestamp("2024-01-01", tz="America/New_York")] * 2
            + [pd.Timestamp("2024-01-02", tz="America/New_York")] * 5,
            "open": [15220.0, 15280.0, 15140.0, 15160.0, 15170.0, 15195.0, 15205.0],
            "high": [15290.0, 15350.0, 15180.0, 15190.0, 15200.0, 15210.0, 15300.0],
            "low": [15120.0, 15240.0, 15120.0, 15130.0, 15150.0, 15170.0, 15190.0],
            "close": [15270.0, 15320.0, 15160.0, 15180.0, 15190.0, 15205.0, 15280.0],
            "volume": [100] * len(timestamps),
        }
    )


def make_execution_setup() -> SetupEvent:
    return SetupEvent(
        setup_id="nq-op",
        setup_name="orb",
        symbol="NQ",
        contract="NQH4",
        timestamp=pd.Timestamp("2024-01-02 09:45", tz="America/New_York").to_pydatetime(),
        session_date=pd.Timestamp("2024-01-02", tz="America/New_York").to_pydatetime(),
        direction=Side.LONG,
        status=SetupStatus.CANDIDATE,
        entry_reference=15170.0,
        stop_reference=15120.0,
        target_reference=15200.0,
        features={},
        context={
            "first_liquidity_target": "pdh",
            "first_liquidity_target_price": 15200.0,
            "partial_take_profit_rule": "first_draw_on_liquidity",
            "partial_take_profit_fraction": 0.5,
            "breakeven_after_first_draw": True,
            "runner_target_rule": "liquidity_sequence",
            "runner_targets": [{"name": "day_high", "price": 15240.0}],
            "runner_trail_rule": "atr_placeholder",
            "runner_trail_atr_multiple": 1.0,
            "formal_stop_rule": "midpoint_or_wick_buffer",
        },
    )


def make_partial_breakeven_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 10:00", periods=3, freq="15min", tz="America/New_York"),
            "symbol": ["NQ"] * 3,
            "session_date": [pd.Timestamp("2024-01-02", tz="America/New_York")] * 3,
            "open": [15170.0, 15190.0, 15185.0],
            "high": [15195.0, 15205.0, 15195.0],
            "low": [15160.0, 15180.0, 15170.0],
            "close": [15190.0, 15195.0, 15172.0],
            "atr": [10.0, 10.0, 10.0],
            "contract": ["NQH4"] * 3,
        }
    )


def make_trailing_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 10:00", periods=3, freq="15min", tz="America/New_York"),
            "symbol": ["NQ"] * 3,
            "session_date": [pd.Timestamp("2024-01-02", tz="America/New_York")] * 3,
            "open": [15170.0, 15190.0, 15255.0],
            "high": [15195.0, 15280.0, 15260.0],
            "low": [15160.0, 15195.0, 15245.0],
            "close": [15190.0, 15260.0, 15248.0],
            "atr": [10.0, 10.0, 10.0],
            "contract": ["NQH4"] * 3,
        }
    )


def test_latest_entry_time_enforcement() -> None:
    frame = make_detector_frame()
    config = ORBConfig.from_strategy_config(
        StrategyConfig(strategy_profile="nq_am_displacement_orb", latest_entry_time="10:00")
    )
    assert ORBSetupDetector(config).detect(frame) == []


def test_body_close_breakout_enforcement() -> None:
    frame = make_detector_frame().copy()
    frame.loc[7, "open"] = 15135.0
    frame.loc[7, "high"] = 15190.0
    frame.loc[7, "low"] = 15120.0
    frame.loc[7, "close"] = 15139.0
    frame.loc[7, "candle_range"] = 70.0
    frame.loc[7, "candle_body"] = 4.0

    config = ORBConfig.from_strategy_config(StrategyConfig(strategy_profile="nq_am_displacement_orb", latest_entry_time="10:15"))
    assert ORBSetupDetector(config).detect(frame) == []


def test_strong_close_threshold_enforcement() -> None:
    frame = make_detector_frame().copy()
    frame.loc[7, "open"] = 15168.0
    frame.loc[7, "close"] = 15142.0
    frame.loc[7, "high"] = 15190.0
    frame.loc[7, "low"] = 15120.0
    frame.loc[7, "candle_range"] = 70.0
    frame.loc[7, "candle_body"] = 26.0

    config = ORBConfig.from_strategy_config(StrategyConfig(strategy_profile="nq_am_displacement_orb", latest_entry_time="10:15"))
    assert ORBSetupDetector(config).detect(frame) == []


def test_liquidity_level_calculation() -> None:
    frame = add_liquidity_levels(make_liquidity_frame())
    row = frame.iloc[5]

    assert row["pdh"] == 15350.0
    assert row["pdl"] == 15120.0
    assert row["day_high"] == 15200.0
    assert row["day_low"] == 15120.0
    assert row["h4_high"] == 15200.0
    assert row["h4_low"] == 15150.0
    assert row["london_high"] == 15190.0
    assert row["london_low"] == 15120.0


def test_first_draw_selection_uses_closest_valid_target() -> None:
    frame = add_liquidity_levels(make_liquidity_frame())
    row = frame.iloc[5]
    name, price = select_first_liquidity_target(row, "long", ("day_high", "pdh", "h4_high", "london_high"))

    assert name == "pdh"
    assert price == 15350.0


def test_minimum_rr_filter() -> None:
    frame = make_detector_frame()
    config = ORBConfig.from_strategy_config(
        StrategyConfig(strategy_profile="nq_am_displacement_orb", minimum_rr_threshold=3.0)
    )
    assert ORBSetupDetector(config).detect(frame) == []


def test_midpoint_wick_stop_calculation() -> None:
    frame = make_detector_frame()
    config = ORBConfig.from_strategy_config(StrategyConfig(strategy_profile="nq_am_displacement_orb", liquidity_target_priority=("pdh",), minimum_rr_threshold=2.0))

    setups = ORBSetupDetector(config).detect(frame)

    assert len(setups) == 1
    setup = setups[0]
    assert setup.stop_reference == 15086.0
    assert setup.context["formal_stop_rule"] == "midpoint_or_wick_buffer"
    assert setup.context["first_liquidity_target"] == "pdh"


def test_partial_and_breakeven_behavior() -> None:
    result = BarBacktestEngine(BacktestConfig()).run(make_partial_breakeven_frame(), [make_execution_setup()])
    trade = result.trades[0]

    assert trade.partial_taken is True
    assert trade.breakeven_activated is True
    assert trade.exit_reason == "breakeven"


def test_runner_trail_activation_after_2r() -> None:
    setup = make_execution_setup()
    setup.context["runner_targets"] = [{"name": "h4_high", "price": 15320.0}]

    result = BarBacktestEngine(BacktestConfig()).run(make_trailing_frame(), [setup])
    trade = result.trades[0]

    assert trade.trail_activated is True
    assert trade.exit_reason == "trail_stop"


def test_no_lookahead_in_liquidity_targeting() -> None:
    frame = add_liquidity_levels(make_liquidity_frame())
    row = frame.iloc[5].copy()

    assert row["day_high"] == 15200.0
    assert row["pdh"] == 15350.0

    name, price = select_first_liquidity_target(row, "long", ("day_high", "pdh"))

    assert name == "pdh"
    assert price == 15350.0

