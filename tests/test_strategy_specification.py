from __future__ import annotations

from pathlib import Path

import pytest

from config.models import StrategyConfig
from setups.documentation import format_strategy_spec, write_strategy_spec
from setups.orb import ORBConfig
from setups.profiles import get_orb_profile
from setups.specification import ORB_PROFILE_NAMES, strategy_spec_from_config


def test_valid_strategy_config_parsing() -> None:
    config = StrategyConfig(
        strategy_profile="orb_retest",
        instrument="NQ",
        opening_range_start_time="09:30",
        opening_range_end_time="09:45",
        opening_range_minutes=15,
        require_retest=True,
        max_trades_per_session=2,
        allowed_days_of_week=("monday", "tuesday"),
        allowed_trade_windows=("09:45-11:00",),
    )

    spec = strategy_spec_from_config(config)
    assert spec.profile_name == "orb_retest"
    assert spec.instrument == "NQ"
    assert spec.opening_range_minutes == 15
    assert spec.require_retest is True
    assert spec.max_trades_per_session == 2


def test_invalid_strategy_config_is_rejected() -> None:
    config = StrategyConfig(
        strategy_profile="orb_basic",
        session_start="16:00",
        session_end="09:30",
        max_trades_per_session=0,
    )

    with pytest.raises(ValueError):
        strategy_spec_from_config(config)


def test_orb_profile_loading() -> None:
    profile = get_orb_profile("orb_trend_filtered")
    assert profile.profile_name == "orb_trend_filtered"
    assert profile.trend_filter == "above_below_ma"
    assert "orb_trend_filtered" in ORB_PROFILE_NAMES


def test_nq_am_displacement_orb_profile_loading() -> None:
    profile = get_orb_profile("nq_am_displacement_orb")
    assert profile.profile_name == "nq_am_displacement_orb"
    assert profile.instrument == "NQ"
    assert profile.ny_open_anchored is True
    assert profile.opening_range_start_time == "09:30"
    assert profile.opening_range_end_time == "09:45"
    assert profile.latest_entry_time == "11:30"
    assert profile.displacement.rule == "candle_displacement"
    assert profile.require_retest is False
    assert profile.target_rule == "r_multiple"
    assert profile.target_r_multiple == 2.0
    assert profile.stop.rule == "midpoint_or_wick_buffer"
    assert profile.management.runner_trail_rule is None


def test_opening_range_start_end_config_parsing() -> None:
    spec = strategy_spec_from_config(
        StrategyConfig(
            instrument="MNQ",
            ny_open_anchored=True,
            opening_range_start_time="09:30",
            opening_range_end_time="09:45",
            opening_range_minutes=15,
            latest_entry_time="10:15",
        )
    )
    assert spec.opening_range_start_time == "09:30"
    assert spec.opening_range_end_time == "09:45"
    assert spec.latest_entry_time == "10:15"


def test_displacement_config_validation() -> None:
    with pytest.raises(ValueError):
        strategy_spec_from_config(
            StrategyConfig(
                strategy_profile="nq_am_displacement_orb",
                strong_close_min_body_range_ratio=1.2,
            )
        )


def test_minimum_rr_validation() -> None:
    with pytest.raises(ValueError):
        strategy_spec_from_config(
            StrategyConfig(
                strategy_profile="nq_am_displacement_orb",
                minimum_rr_threshold=1.5,
            )
        )


def test_stop_mode_validation() -> None:
    with pytest.raises(ValueError):
        strategy_spec_from_config(
            StrategyConfig(
                strategy_profile="nq_am_displacement_orb",
                stop_mode="midpoint_or_wick_buffer",
                stop_buffer_points=0.0,
                stop_buffer_atr_multiple=None,
            )
        )


def test_liquidity_target_priority_validation() -> None:
    with pytest.raises(ValueError):
        strategy_spec_from_config(
            StrategyConfig(
                strategy_profile="nq_am_displacement_orb",
                liquidity_target_mode="directional_continuation",
                liquidity_target_priority=("pdh", "invalid_level"),
            )
        )


def test_strategy_documentation_output() -> None:
    spec = strategy_spec_from_config(StrategyConfig(strategy_profile="nq_am_displacement_orb"))
    text = format_strategy_spec(spec)

    assert "Strategy Profile: nq_am_displacement_orb" in text
    assert "NY Open Anchored: yes" in text
    assert "Latest Entry Time: 11:30" in text
    assert "Opening Range: 09:30-09:45 (15 minutes)" in text
    assert "Retest Requirement: no" in text
    assert "Target Rule: r_multiple (2.0R if applicable)" in text
    assert "Runner Trail Rule: none atr_multiple=None" in text
    assert "Trade Windows: 09:45-11:30" in text


def test_strategy_documentation_can_be_written(tmp_path: Path) -> None:
    spec = strategy_spec_from_config(StrategyConfig())
    path = tmp_path / "strategy_rules.txt"
    write_strategy_spec(spec, path)

    assert path.exists()
    assert "Strategy Profile: orb_basic" in path.read_text(encoding="utf-8")


def test_orb_config_bridge_from_strategy_spec() -> None:
    spec = strategy_spec_from_config(
        StrategyConfig(
            strategy_profile="nq_am_displacement_orb",
            instrument="MNQ",
            max_trades_per_session=3,
            setup_feature_whitelist=("or_high", "or_low"),
        )
    )
    orb_config = ORBConfig.from_strategy_spec(spec)

    assert orb_config.instrument == "MNQ"
    assert orb_config.ny_open_anchored is True
    assert orb_config.stop_mode == "midpoint_or_wick_buffer"
    assert orb_config.formal_stop_rule == "midpoint_or_wick_buffer"
    assert orb_config.latest_entry_time == "11:30"
    assert orb_config.target_rule == "r_multiple"
    assert orb_config.target_r_multiple == 2.0
    assert orb_config.breakeven_after_first_draw is True
    assert orb_config.runner_trail_rule is None

