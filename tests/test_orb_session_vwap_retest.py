from __future__ import annotations

import pandas as pd

from data.schemas import Side
from setups.orb_session_vwap_retest import ORBSessionVWAPRetestConfig, ORBSessionVWAPRetestDetector


def make_long_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30", periods=20, freq="min", tz="America/New_York")
    rows: list[dict[str, object]] = []
    for idx, ts in enumerate(timestamps):
        if idx < 15:
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": "NQ",
                    "contract": "NQH4",
                    "session_date": ts.normalize(),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 100,
                    "or_high": pd.NA,
                    "or_low": pd.NA,
                    "vwap_eth": 100.0,
                    "vwap_rth": 100.0,
                }
            )
        elif idx == 15:
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": "NQ",
                    "contract": "NQH4",
                    "session_date": ts.normalize(),
                    "open": 101.2,
                    "high": 102.5,
                    "low": 101.1,
                    "close": 102.0,
                    "volume": 100,
                    "or_high": 101.0,
                    "or_low": 99.0,
                    "vwap_eth": 100.0,
                    "vwap_rth": 100.5,
                    "pdh": 104.0,
                    "h4_high": 103.0,
                    "day_high": 102.5,
                    "london_high": 102.8,
                    "trend_bias": 1,
                    "trend_spread": 0.5,
                    "close_vs_fast_ema": 0.3,
                    "recent_bullish_fvg_size": 0.8,
                    "recent_bearish_fvg_size": 0.0,
                }
            )
        elif idx == 16:
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": "NQ",
                    "contract": "NQH4",
                    "session_date": ts.normalize(),
                    "open": 101.9,
                    "high": 102.0,
                    "low": 100.4,
                    "close": 101.3,
                    "volume": 100,
                    "or_high": 101.0,
                    "or_low": 99.0,
                    "vwap_eth": 100.0,
                    "vwap_rth": 100.5,
                    "pdh": 104.0,
                    "h4_high": 103.0,
                    "day_high": 102.5,
                    "london_high": 102.8,
                    "trend_bias": 1,
                    "trend_spread": 0.4,
                    "close_vs_fast_ema": 0.2,
                    "recent_bullish_fvg_size": 0.6,
                    "recent_bearish_fvg_size": 0.0,
                }
            )
        else:
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": "NQ",
                    "contract": "NQH4",
                    "session_date": ts.normalize(),
                    "open": 101.4,
                    "high": 102.0,
                    "low": 101.2,
                    "close": 101.8,
                    "volume": 100,
                    "or_high": 101.0,
                    "or_low": 99.0,
                    "vwap_eth": 100.0,
                    "vwap_rth": 100.6,
                    "pdh": 104.0,
                    "h4_high": 103.0,
                    "day_high": 102.5,
                    "london_high": 102.8,
                    "trend_bias": 1,
                    "trend_spread": 0.3,
                    "close_vs_fast_ema": 0.1,
                    "recent_bullish_fvg_size": 0.4,
                    "recent_bearish_fvg_size": 0.0,
                }
            )
    return pd.DataFrame(rows)


def make_short_frame() -> pd.DataFrame:
    frame = make_long_frame().copy()
    frame.loc[15, ["open", "high", "low", "close", "pdl", "h4_low", "day_low", "london_low"]] = [98.8, 98.9, 97.5, 97.8, 95.0, 96.0, 97.4, 96.8]
    frame.loc[16, ["open", "high", "low", "close", "vwap_rth", "pdl", "h4_low", "day_low", "london_low"]] = [97.9, 99.6, 97.7, 98.7, 99.2, 95.0, 96.0, 97.4, 96.8]
    frame.loc[15:, "or_high"] = 101.0
    frame.loc[15:, "or_low"] = 99.0
    frame.loc[15:, "trend_bias"] = -1
    frame.loc[15:, "trend_spread"] = -0.5
    frame.loc[15:, "close_vs_fast_ema"] = -0.3
    frame.loc[15:, "recent_bullish_fvg_size"] = 0.0
    frame.loc[15:, "recent_bearish_fvg_size"] = 0.7
    return frame


def test_orb_session_vwap_retest_detects_long_after_breakout_then_vwap_touch() -> None:
    detector = ORBSessionVWAPRetestDetector(
        ORBSessionVWAPRetestConfig(opening_range_minutes=15, allowed_trade_windows=("09:45-11:30",))
    )

    setups = detector.detect(make_long_frame())

    assert len(setups) == 1
    setup = setups[0]
    assert setup.direction is Side.LONG
    assert setup.timestamp.minute == 46
    assert setup.context["retest_vwap_name"] == "vwap_rth"
    assert setup.stop_reference == 99.0
    assert setup.target_reference > setup.entry_reference
    assert setup.context["target_name"] == "fixed_r_2.0R"
    assert setup.features["trend_bias"] == 1.0
    assert setup.features["recent_directional_fvg_size"] == 0.6


def test_orb_session_vwap_retest_detects_short_after_breakout_then_vwap_touch() -> None:
    detector = ORBSessionVWAPRetestDetector(
        ORBSessionVWAPRetestConfig(opening_range_minutes=15, allowed_trade_windows=("09:45-11:30",))
    )

    setups = detector.detect(make_short_frame())

    assert len(setups) == 1
    setup = setups[0]
    assert setup.direction is Side.SHORT
    assert setup.timestamp.minute == 46
    assert setup.context["retest_vwap_name"] == "vwap_rth"
    assert setup.stop_reference == 101.0
    assert setup.features["trend_bias"] == -1.0


def test_orb_session_vwap_retest_requires_touch_of_session_vwap() -> None:
    frame = make_long_frame()
    frame.loc[16, "high"] = 100.4
    frame.loc[16, "low"] = 100.1

    detector = ORBSessionVWAPRetestDetector(
        ORBSessionVWAPRetestConfig(opening_range_minutes=15, allowed_trade_windows=("09:45-11:30",))
    )

    setups = detector.detect(frame)

    assert setups == []


def test_orb_session_vwap_retest_can_use_liquidity_target() -> None:
    detector = ORBSessionVWAPRetestDetector(
        ORBSessionVWAPRetestConfig(
            opening_range_minutes=15,
            allowed_trade_windows=("09:45-11:30",),
            target_mode="liquidity",
            liquidity_target_priority=("day_high", "h4_high", "pdh"),
        )
    )

    setups = detector.detect(make_long_frame())

    assert len(setups) == 1
    setup = setups[0]
    assert setup.target_reference == 102.5
    assert setup.context["target_name"] == "day_high"
    assert setup.context["target_source"] == "liquidity"


def test_orb_session_vwap_retest_liquidity_mode_falls_back_to_fixed_r() -> None:
    frame = make_long_frame()
    detector = ORBSessionVWAPRetestDetector(
        ORBSessionVWAPRetestConfig(
            opening_range_minutes=15,
            allowed_trade_windows=("09:45-11:30",),
            target_mode="liquidity",
            liquidity_target_priority=("pdl", "h4_low", "day_low"),
        )
    )

    setups = detector.detect(frame)

    assert len(setups) == 1
    setup = setups[0]
    assert setup.context["target_name"] == "fixed_r_2.0R"
    assert setup.context["target_source"] == "fixed_r"
    assert round(setup.target_reference, 10) == 105.9


def test_orb_session_vwap_retest_can_require_trend_alignment() -> None:
    frame = make_long_frame()
    frame.loc[16, "trend_bias"] = -1
    detector = ORBSessionVWAPRetestDetector(
        ORBSessionVWAPRetestConfig(
            opening_range_minutes=15,
            allowed_trade_windows=("09:45-11:30",),
            require_trend_alignment=True,
        )
    )

    setups = detector.detect(frame)

    assert setups == []


def test_orb_session_vwap_retest_can_require_recent_fvg_context() -> None:
    frame = make_long_frame()
    frame.loc[16, "recent_bullish_fvg_size"] = 0.0
    detector = ORBSessionVWAPRetestDetector(
        ORBSessionVWAPRetestConfig(
            opening_range_minutes=15,
            allowed_trade_windows=("09:45-11:30",),
            require_fvg_context=True,
        )
    )

    setups = detector.detect(frame)

    assert setups == []


