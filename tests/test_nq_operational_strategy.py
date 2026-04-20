from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.engine import BarBacktestEngine, BacktestConfig
from config.models import StrategyConfig
from data.schemas import SetupEvent, SetupStatus, Side
from setups.orb import ORBConfig, ORBSetupDetector


def make_detector_frame() -> pd.DataFrame:
    timestamps = list(pd.date_range("2024-01-01 09:30", periods=4, freq="15min", tz="America/New_York"))
    timestamps += list(pd.date_range("2024-01-02 09:30", periods=10, freq="15min", tz="America/New_York"))
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["NQ"] * len(timestamps),
            "contract": ["NQH4"] * len(timestamps),
            "session_date": [pd.Timestamp("2024-01-01", tz="America/New_York")] * 4
            + [pd.Timestamp("2024-01-02", tz="America/New_York")] * 10,
            "open": [15280, 15300, 15310, 15320, 15100, 15110, 15120, 15135, 15150, 15180, 15195, 15210, 15205, 15215],
            "high": [15310, 15340, 15350, 15330, 15140, 15140, 15155, 15180, 15210, 15240, 15255, 15270, 15245, 15235],
            "low": [15240, 15270, 15290, 15300, 15090, 15100, 15110, 15125, 15140, 15170, 15190, 15200, 15195, 15190],
            "close": [15300, 15320, 15340, 15310, 15110, 15120, 15130, 15170, 15200, 15230, 15245, 15260, 15210, 15200],
            "volume": [100] * len(timestamps),
        }
    )
    frame["or_high"] = [pd.NA] * 5 + [15140.0] * 9
    frame["or_low"] = [pd.NA] * 5 + [15090.0] * 9
    frame["or_width"] = [pd.NA] * 5 + [50.0] * 9
    frame["or_width_atr"] = [pd.NA] * 5 + [1.0] * 9
    frame["atr"] = [20.0] * len(timestamps)
    frame["candle_range"] = frame["high"] - frame["low"]
    frame["candle_body"] = (frame["close"] - frame["open"]).abs()
    frame["lower_wick"] = frame[["open", "close"]].min(axis=1) - frame["low"]
    frame["upper_wick"] = frame["high"] - frame[["open", "close"]].max(axis=1)
    frame["ema_20"] = frame["close"] - 10
    frame["breakout_strength"] = [0.0] * 7 + [0.7] * (len(timestamps) - 7)
    return frame


def make_after_cutoff_frame() -> pd.DataFrame:
    frame = make_detector_frame().copy()
    frame.loc[7:, "timestamp"] = pd.to_datetime(
        [
            "2024-01-02 11:31",
            "2024-01-02 11:45",
            "2024-01-02 12:00",
            "2024-01-02 12:15",
            "2024-01-02 12:30",
            "2024-01-02 12:45",
            "2024-01-02 13:00",
        ]
    ).tz_localize("America/New_York")
    return frame


def make_simple_execution_setup() -> SetupEvent:
    return SetupEvent(
        setup_id="nq-two-lot",
        setup_name="orb",
        symbol="NQ",
        contract="NQH4",
        timestamp=pd.Timestamp("2024-01-02 09:45", tz="America/New_York").to_pydatetime(),
        session_date=pd.Timestamp("2024-01-02", tz="America/New_York").to_pydatetime(),
        direction=Side.LONG,
        status=SetupStatus.CANDIDATE,
        entry_reference=15170.0,
        stop_reference=15120.0,
        target_reference=15270.0,
        features={},
        context={
            "formal_stop_rule": "midpoint_or_wick_buffer",
            "first_liquidity_target": "pdh",
            "first_liquidity_target_price": 15220.0,
            "breakeven_after_first_draw": True,
        },
    )


def make_execution_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 10:00", periods=3, freq="15min", tz="America/New_York"),
            "symbol": ["NQ"] * 3,
            "session_date": [pd.Timestamp("2024-01-02", tz="America/New_York")] * 3,
            "open": [15170.0, 15190.0, 15210.0],
            "high": [15195.0, 15220.0, 15280.0],
            "low": [15160.0, 15170.0, 15200.0],
            "close": [15190.0, 15210.0, 15270.0],
            "atr": [10.0, 10.0, 10.0],
            "contract": ["NQH4"] * 3,
        }
    )


def test_profile_default_latest_entry_time_is_1130() -> None:
    config = ORBConfig.from_strategy_config(StrategyConfig(strategy_profile="nq_am_displacement_orb"))
    assert config.latest_entry_time == "11:30"


def test_profile_breakeven_after_first_draw_is_enabled() -> None:
    config = ORBConfig.from_strategy_config(StrategyConfig(strategy_profile="nq_am_displacement_orb"))
    assert config.breakeven_after_first_draw is True


def test_latest_entry_time_enforcement() -> None:
    frame = make_after_cutoff_frame()
    config = ORBConfig.from_strategy_config(StrategyConfig(strategy_profile="nq_am_displacement_orb"))
    setups, diagnostics = ORBSetupDetector(config).detect_with_diagnostics(frame)

    assert setups == []
    assert diagnostics.counts["failed_latest_entry_cutoff"] >= 1


def test_body_close_breakout_enforcement() -> None:
    frame = make_detector_frame().copy()
    frame.loc[7, "open"] = 15135.0
    frame.loc[7, "high"] = 15190.0
    frame.loc[7, "low"] = 15120.0
    frame.loc[7, "close"] = 15139.0
    frame.loc[7, "candle_range"] = 70.0
    frame.loc[7, "candle_body"] = 4.0

    config = ORBConfig.from_strategy_config(StrategyConfig(strategy_profile="nq_am_displacement_orb", latest_entry_time="10:15"))
    setups, diagnostics = ORBSetupDetector(config).detect_with_diagnostics(frame)

    assert setups == []
    assert diagnostics.counts["wick_only_breakout"] >= 1


def test_midpoint_wick_stop_calculation() -> None:
    frame = make_detector_frame()
    config = ORBConfig.from_strategy_config(StrategyConfig(strategy_profile="nq_am_displacement_orb"))

    setups = ORBSetupDetector(config).detect(frame)

    assert len(setups) == 1
    setup = setups[0]
    assert setup.stop_reference == 15086.0
    assert setup.context["formal_stop_rule"] == "midpoint_or_wick_buffer"


def test_fixed_2r_target_logic() -> None:
    frame = make_detector_frame()
    config = ORBConfig.from_strategy_config(StrategyConfig(strategy_profile="nq_am_displacement_orb"))
    setup = ORBSetupDetector(config).detect(frame)[0]

    risk = abs(setup.entry_reference - setup.stop_reference)
    assert setup.target_reference == setup.entry_reference + (2.0 * risk)
    assert setup.context["target_rule"] == "r_multiple"
    assert setup.context["target_r_multiple"] == 2.0


def test_two_contract_position_size_propagation() -> None:
    result = BarBacktestEngine(BacktestConfig(position_size=2.0)).run(make_execution_frame(), [make_simple_execution_setup()])

    assert result.trades[0].size == 2.0


def test_breakeven_after_first_draw_can_exit_remaining_position() -> None:
    result = BarBacktestEngine(BacktestConfig(position_size=2.0)).run(make_execution_frame(), [make_simple_execution_setup()])

    assert result.trades[0].breakeven_activated is True
    assert result.trades[0].exit_reason == "breakeven"


def test_audit_csv_generation(tmp_path: Path) -> None:
    frame = make_detector_frame()
    config = ORBConfig.from_strategy_config(StrategyConfig(strategy_profile="nq_am_displacement_orb"))
    _, diagnostics = ORBSetupDetector(config).detect_with_diagnostics(frame)

    audit_path = tmp_path / "candidate_audit.csv"
    diagnostics.audit_frame.to_csv(audit_path, index=False)

    assert audit_path.exists()
    audit = pd.read_csv(audit_path)
    assert {"timestamp", "or_high", "or_low", "close", "body_range_ratio", "estimated_rr", "emitted_setup", "rejection_reason"}.issubset(audit.columns)
