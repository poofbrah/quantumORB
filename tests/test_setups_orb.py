from __future__ import annotations

import pandas as pd

from data.schemas import Side
from setups.orb import ORBConfig, ORBSetupDetector


def make_orb_detection_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:30", periods=7, freq="min", tz="America/New_York")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["ES"] * 7,
            "contract": ["ESH4"] * 7,
            "session_date": [pd.Timestamp("2024-01-02", tz="America/New_York")] * 7,
            "open": [100.0, 100.2, 100.4, 100.9, 101.1, 101.4, 101.5],
            "high": [100.5, 100.6, 101.4, 101.2, 101.5, 101.7, 101.9],
            "low": [99.8, 100.0, 100.7, 100.7, 101.0, 101.2, 101.3],
            "close": [100.2, 100.4, 101.1, 101.0, 101.4, 101.6, 101.8],
            "volume": [100, 110, 120, 130, 140, 150, 160],
            "or_high": [pd.NA, pd.NA, 100.6, 100.6, 100.6, 100.6, 100.6],
            "or_low": [pd.NA, pd.NA, 99.8, 99.8, 99.8, 99.8, 99.8],
            "or_width": [pd.NA, pd.NA, 0.8, 0.8, 0.8, 0.8, 0.8],
            "or_width_atr": [pd.NA, pd.NA, 0.5, 0.5, 0.5, 0.5, 0.5],
            "breakout_strength": [0.0, 0.0, 0.625, 0.5, 1.0, 1.25, 1.5],
            "ema_20": [100.0, 100.1, 100.5, 100.7, 100.9, 101.0, 101.2],
            "atr": [0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6],
            "rolling_volatility": [0.1] * 7,
            "relative_volume": [1.0, 1.0, 1.3, 1.1, 1.0, 1.0, 1.0],
            "hour": [9] * 7,
            "minute": [30, 31, 32, 33, 34, 35, 36],
            "day_of_week": [1] * 7,
            "session_bucket": ["open"] * 7,
            "volatility_regime": ["normal"] * 7,
            "distance_from_or_high": [pd.NA, pd.NA, 0.5, 0.4, 0.8, 1.0, 1.2],
            "distance_from_or_low": [pd.NA, pd.NA, 1.3, 1.2, 1.6, 1.8, 2.0],
        }
    )


def make_orb_short_frame() -> pd.DataFrame:
    frame = make_orb_detection_frame().copy()
    frame["close"] = [100.2, 100.0, 99.5, 99.2, 99.1, 99.0, 98.9]
    frame["high"] = [100.5, 100.4, 100.1, 99.9, 99.7, 99.5, 99.4]
    frame["low"] = [99.8, 99.9, 99.3, 99.0, 98.8, 98.7, 98.6]
    frame["or_high"] = [pd.NA, pd.NA, 100.5, 100.5, 100.5, 100.5, 100.5]
    frame["or_low"] = [pd.NA, pd.NA, 99.8, 99.8, 99.8, 99.8, 99.8]
    frame["breakout_strength"] = [0.0, 0.0, 0.375, 0.75, 0.875, 1.0, 1.1]
    frame["distance_from_or_high"] = [pd.NA, pd.NA, -1.0, -1.3, -1.4, -1.5, -1.6]
    frame["distance_from_or_low"] = [pd.NA, pd.NA, -0.3, -0.6, -0.7, -0.8, -0.9]
    frame["ema_20"] = [100.2, 100.1, 100.0, 99.8, 99.6, 99.4, 99.2]
    return frame


def test_orb_long_setup_detection() -> None:
    detector = ORBSetupDetector(ORBConfig(opening_range_minutes=2, max_trades_per_session=1))
    setups = detector.detect(make_orb_detection_frame())

    assert len(setups) == 1
    setup = setups[0]
    assert setup.setup_name == "orb"
    assert setup.direction is Side.LONG
    assert setup.entry_reference == 101.1
    assert setup.stop_reference == 99.8
    assert setup.target_reference > setup.entry_reference
    assert setup.timestamp.minute == 32
    assert "or_high" in setup.features
    assert "close" not in setup.features
    assert "timestamp" not in setup.features


def test_orb_short_setup_detection() -> None:
    detector = ORBSetupDetector(ORBConfig(opening_range_minutes=2, enable_long=False, enable_short=True))
    setups = detector.detect(make_orb_short_frame())

    assert len(setups) == 1
    assert setups[0].direction is Side.SHORT
    assert setups[0].stop_reference == 100.5


def test_orb_config_filters_change_detection() -> None:
    detector = ORBSetupDetector(
        ORBConfig(
            opening_range_minutes=2,
            trend_filter="above_below_ma",
            trend_column="ema_20",
            volatility_filter_enabled=True,
            min_or_width_atr=0.6,
        )
    )
    setups = detector.detect(make_orb_detection_frame())
    assert setups == []


def test_orb_retest_filter_requires_touch_back_to_level() -> None:
    detector = ORBSetupDetector(ORBConfig(opening_range_minutes=2, require_retest=True))
    setups = detector.detect(make_orb_detection_frame())
    assert setups == []


def test_orb_emits_only_first_same_side_breakout_per_session() -> None:
    detector = ORBSetupDetector(ORBConfig(opening_range_minutes=2, max_trades_per_session=2))
    setups = detector.detect(make_orb_detection_frame())

    assert len(setups) == 1
    assert setups[0].timestamp.minute == 32


def test_orb_feature_whitelist_is_respected() -> None:
    detector = ORBSetupDetector(
        ORBConfig(
            opening_range_minutes=2,
            setup_feature_whitelist=("or_high", "or_low", "breakout_strength"),
        )
    )
    setups = detector.detect(make_orb_detection_frame())

    assert set(setups[0].features) == {"or_high", "or_low", "breakout_strength"}

