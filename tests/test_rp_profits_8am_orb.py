from __future__ import annotations

import pandas as pd

from config.models import StrategyConfig
from setups.rp_profits_8am_orb import RPProfits8AMConfig, RPProfits8AMSetupDetector


TIMEZONE = "America/New_York"


def _base_rows(zone_high: float = 101.0, zone_low: float = 99.0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for minute in range(15):
        ts = pd.Timestamp("2024-01-02 08:00", tz=TIMEZONE) + pd.Timedelta(minutes=minute)
        high = zone_high if minute == 5 else zone_high - 0.2
        low = zone_low if minute == 8 else zone_low + 0.2
        rows.append(
            {
                "timestamp": ts,
                "symbol": "NQ",
                "contract": "NQH4",
                "open": 100.0,
                "high": high,
                "low": low,
                "close": 100.0,
                "volume": 100,
                "session_date": ts.normalize(),
                "atr": 1.0,
                "vwap_eth": 100.0,
                "vwap_rth": None,
            }
        )
    return rows


def make_displacement_frame(*, touch_time: str = "2024-01-02 09:05", include_touch: bool = True, pre_displacement_touch: bool = False) -> pd.DataFrame:
    rows = _base_rows()
    if pre_displacement_touch:
        ts = pd.Timestamp("2024-01-02 08:16", tz=TIMEZONE)
        rows.append({"timestamp": ts, "symbol": "NQ", "contract": "NQH4", "open": 100.2, "high": 100.6, "low": 100.0, "close": 100.3, "volume": 100, "session_date": ts.normalize(), "atr": 1.0, "vwap_eth": 100.5, "vwap_rth": None})
    ts = pd.Timestamp("2024-01-02 08:20", tz=TIMEZONE)
    rows.append({"timestamp": ts, "symbol": "NQ", "contract": "NQH4", "open": 100.8, "high": 102.0, "low": 100.7, "close": 101.8, "volume": 100, "session_date": ts.normalize(), "atr": 1.0, "vwap_eth": 100.5, "vwap_rth": None})
    if include_touch:
        ts = pd.Timestamp(touch_time, tz=TIMEZONE)
        rows.append({"timestamp": ts, "symbol": "NQ", "contract": "NQH4", "open": 100.8, "high": 101.0, "low": 100.4, "close": 100.7, "volume": 100, "session_date": ts.normalize(), "atr": 1.0, "vwap_eth": 100.5, "vwap_rth": None})
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def make_reentry_frame(*, side: str = "short") -> pd.DataFrame:
    rows = _base_rows()
    if side == "short":
        ts = pd.Timestamp("2024-01-02 08:40", tz=TIMEZONE)
        rows.append({"timestamp": ts, "symbol": "NQ", "contract": "NQH4", "open": 101.1, "high": 101.8, "low": 100.9, "close": 101.4, "volume": 100, "session_date": ts.normalize(), "atr": 1.0, "vwap_eth": 100.7, "vwap_rth": None})
        ts = pd.Timestamp("2024-01-02 09:05", tz=TIMEZONE)
        rows.append({"timestamp": ts, "symbol": "NQ", "contract": "NQH4", "open": 101.4, "high": 101.6, "low": 100.2, "close": 100.6, "volume": 100, "session_date": ts.normalize(), "atr": 1.0, "vwap_eth": 100.5, "vwap_rth": 100.0})
    else:
        ts = pd.Timestamp("2024-01-02 08:40", tz=TIMEZONE)
        rows.append({"timestamp": ts, "symbol": "NQ", "contract": "NQH4", "open": 98.9, "high": 99.1, "low": 98.2, "close": 98.6, "volume": 100, "session_date": ts.normalize(), "atr": 1.0, "vwap_eth": 99.3, "vwap_rth": None})
        ts = pd.Timestamp("2024-01-02 09:05", tz=TIMEZONE)
        rows.append({"timestamp": ts, "symbol": "NQ", "contract": "NQH4", "open": 98.6, "high": 99.8, "low": 98.4, "close": 99.4, "volume": 100, "session_date": ts.normalize(), "atr": 1.0, "vwap_eth": 99.3, "vwap_rth": 100.0})
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def make_hybrid_frame() -> pd.DataFrame:
    frame = pd.concat([make_displacement_frame(touch_time="2024-01-02 09:04"), make_reentry_frame(side="short")], ignore_index=True)
    return frame.sort_values("timestamp").drop_duplicates(subset=["timestamp", "open", "high", "low", "close"], keep="first").reset_index(drop=True)


def make_detector(**overrides) -> RPProfits8AMSetupDetector:
    config_kwargs: dict[str, object] = {
        "strategy_profile": "rp_profits_8am_orb",
        "instrument": "NQ",
        "session_timezone": TIMEZONE,
        "allowed_trade_windows": ("09:00-10:30",),
        "trade_window_required": True,
        "key_zone_start_time": "08:00",
        "key_zone_end_time": "08:15",
        "key_zone_min_width_points": None,
        "key_zone_max_width_points": None,
        "strong_close_min_body_range_ratio": 0.6,
        "strong_close_min_boundary_distance": 0.25,
        "strong_close_min_boundary_distance_atr": 0.5,
        "rr_multiple": 2.0,
        "rp_stop_mode": "zone_boundary",
        "rp_partial_at_r": 1.0,
        "rp_partial_fraction": 0.5,
        "rp_move_stop_to_breakeven_after_partial": True,
        "rp_runner_target_r": None,
        "rp_runner_trailing_stop_mode": "atr",
        "rp_runner_trailing_atr_multiple": 1.5,
        "rp_entry_mode": "displacement_vwap_pullback",
        "reentry_body_ratio_min": 0.5,
        "reentry_min_close_distance_points": 0.25,
        "reentry_min_close_distance_atr": 0.25,
        "rp_reentry_stop_mode": "signal_or_range",
        "rp_reentry_stop_buffer_points": 0.0,
        "rp_reentry_max_entry_time": "10:30",
        "rp_reentry_tp1_mode": "vwap_or_mid",
        "rp_reentry_partial_fraction": 0.5,
        "rp_reentry_breakeven_after_tp1": True,
    }
    config_kwargs.update(overrides)
    return RPProfits8AMSetupDetector(RPProfits8AMConfig.from_strategy_config(StrategyConfig(**config_kwargs)))


def test_displacement_branch_still_emits_setup() -> None:
    setup = make_detector().detect(make_displacement_frame())[0]
    assert setup.context["entry_family"] == "displacement"
    assert setup.entry_reference == 100.5
    assert setup.stop_reference == 99.0


def test_short_from_top_of_range_reentry_emits_setup() -> None:
    setup = make_detector(rp_entry_mode="range_reentry_vwap").detect(make_reentry_frame(side="short"))[0]
    assert setup.context["entry_family"] == "range_reentry"
    assert setup.direction.value == "short"
    assert setup.context["range_break_side"] == "top"
    assert setup.context["tp1_price"] == 100.0
    assert setup.context["runner_target_price"] == 99.0


def test_long_from_bottom_of_range_reentry_emits_setup() -> None:
    setup = make_detector(rp_entry_mode="range_reentry_vwap").detect(make_reentry_frame(side="long"))[0]
    assert setup.context["entry_family"] == "range_reentry"
    assert setup.direction.value == "long"
    assert setup.context["range_break_side"] == "bottom"
    assert setup.context["runner_target_price"] == 101.0


def test_no_lookahead_pre_displacement_touch_does_not_trigger_displacement() -> None:
    setups = make_detector(allowed_trade_windows=("08:15-10:30",)).detect(make_displacement_frame(include_touch=False, pre_displacement_touch=True))
    assert setups == []


def test_hybrid_mode_selects_earliest_valid_branch() -> None:
    setup = make_detector(rp_entry_mode="hybrid").detect(make_hybrid_frame())[0]
    assert setup.context["entry_family"] == "displacement"


def test_reentry_stop_falls_back_to_range_boundary_when_requested() -> None:
    setup = make_detector(rp_entry_mode="range_reentry_vwap", rp_reentry_stop_mode="range_buffer").detect(make_reentry_frame(side="short"))[0]
    assert setup.stop_reference == 101.0
    assert setup.context["stop_anchor_used"] == "range_boundary"

