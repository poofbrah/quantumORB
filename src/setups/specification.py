from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, time

from config.models import StrategyConfig
from .liquidity import LiquidityFrameworkSpec, LiquidityLevel

DAY_NAMES = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
ORB_PROFILE_NAMES = (
    "orb_basic",
    "orb_retest",
    "orb_trend_filtered",
    "orb_volatility_filtered",
    "nq_am_displacement_orb",
)
SUPPORTED_LIQUIDITY_LEVELS = {level.value for level in LiquidityLevel}
SUPPORTED_STOP_RULES = {
    "or_boundary",
    "atr",
    "or_midpoint_buffer",
    "range_wick_buffer",
    "midpoint_or_wick_buffer",
}
SUPPORTED_WICK_REFERENCE_MODES = {None, "largest_internal_wick", "nearest_internal_wick"}
SUPPORTED_RUNNER_RULES = {None, "liquidity_sequence", "r_multiple"}
SUPPORTED_DISPLACEMENT_RULES = {"none", "candle_displacement"}
SUPPORTED_TARGET_RULES = {"r_multiple", "liquidity_sequence"}
SUPPORTED_FIRST_DRAW_RULES = {None, "first_liquidity_in_priority", "first_liquidity_in_direction"}
SUPPORTED_RUNNER_TRAIL_RULES = {None, "atr_placeholder", "swing_placeholder"}


@dataclass(slots=True)
class DisplacementSpec:
    rule: str
    min_body_size: float | None
    min_body_range_ratio: float | None
    min_close_distance: float | None
    min_close_distance_atr: float | None
    strong_close_min_body_range_ratio: float | None
    strong_close_min_boundary_distance: float | None
    strong_close_min_boundary_distance_atr: float | None
    invalidate_on_early_counter_liquidity_consumption: bool
    allow_retest_rescue_after_early_liquidity_consumption: bool


@dataclass(slots=True)
class StopSpec:
    rule: str
    stop_atr_multiple: float | None
    stop_buffer_points: float | None
    stop_buffer_atr_multiple: float | None
    wick_reference_mode: str | None
    fallback_stop_mode: str | None


@dataclass(slots=True)
class ManagementSpec:
    first_draw_target_rule: str | None
    minimum_rr_threshold: float | None
    breakeven_rule: str | None
    breakeven_trigger_r: float | None
    breakeven_after_partial: bool
    breakeven_after_first_draw: bool
    partial_take_profit_rule: str | None
    partial_take_profit_r: float | None
    partial_take_profit_fraction: float | None
    runner_target_rule: str | None
    runner_target_priority: tuple[str, ...]
    trailing_stop_rule: str | None
    trailing_stop_atr_multiple: float | None
    runner_trail_rule: str | None
    runner_trail_atr_multiple: float | None


@dataclass(slots=True)
class ORBStrategySpec:
    profile_name: str
    instrument: str
    session_name: str
    session_timezone: str
    ny_open_anchored: bool
    session_start: str
    session_end: str
    latest_entry_time: str | None
    opening_range_start_time: str
    opening_range_end_time: str
    opening_range_minutes: int
    enable_long: bool
    enable_short: bool
    long_trigger: str
    short_trigger: str
    breakout_confirmation_rule: str
    require_retest: bool
    retest_rule: str | None
    trend_filter: str
    trend_column: str | None
    volatility_filter_enabled: bool
    min_or_width_atr: float | None
    max_or_width_atr: float | None
    displacement: DisplacementSpec
    liquidity: LiquidityFrameworkSpec
    entry_rule: str
    stop: StopSpec
    target_rule: str
    target_r_multiple: float | None
    management: ManagementSpec
    max_trades_per_session: int
    allowed_trade_windows: tuple[str, ...]
    allowed_days_of_week: tuple[str, ...]
    news_skip_rules: tuple[str, ...]
    discretionary_skip_reasons: tuple[str, ...]
    min_breakout_strength: float
    setup_feature_whitelist: tuple[str, ...] | None


def strategy_spec_from_config(config: StrategyConfig) -> ORBStrategySpec:
    base = _base_spec(config.strategy_profile)
    defaults = StrategyConfig()

    def pick(field_name: str, base_value):
        value = getattr(config, field_name)
        default_value = getattr(defaults, field_name)
        return base_value if value == default_value else value

    spec = replace(
        base,
        profile_name=config.strategy_profile,
        instrument=pick("instrument", base.instrument),
        session_name=pick("session_name", base.session_name),
        session_timezone=pick("session_timezone", base.session_timezone),
        ny_open_anchored=pick("ny_open_anchored", base.ny_open_anchored),
        session_start=pick("session_start", base.session_start),
        session_end=pick("session_end", base.session_end),
        latest_entry_time=pick("latest_entry_time", base.latest_entry_time),
        opening_range_start_time=pick("opening_range_start_time", base.opening_range_start_time),
        opening_range_end_time=pick("opening_range_end_time", base.opening_range_end_time),
        opening_range_minutes=pick("opening_range_minutes", base.opening_range_minutes),
        enable_long=pick("enable_long", base.enable_long),
        enable_short=pick("enable_short", base.enable_short),
        long_trigger=pick("long_trigger", base.long_trigger),
        short_trigger=pick("short_trigger", base.short_trigger),
        breakout_confirmation_rule=pick("breakout_confirmation_rule", base.breakout_confirmation_rule),
        require_retest=pick("require_retest", base.require_retest),
        retest_rule=pick("retest_rule", base.retest_rule),
        trend_filter=pick("trend_filter", base.trend_filter),
        trend_column=pick("trend_column", base.trend_column),
        volatility_filter_enabled=pick("volatility_filter_enabled", base.volatility_filter_enabled),
        min_or_width_atr=pick("min_or_width_atr", base.min_or_width_atr),
        max_or_width_atr=pick("max_or_width_atr", base.max_or_width_atr),
        displacement=DisplacementSpec(
            rule=pick("displacement_rule", base.displacement.rule),
            min_body_size=pick("displacement_min_body_size", base.displacement.min_body_size),
            min_body_range_ratio=pick("displacement_min_body_range_ratio", base.displacement.min_body_range_ratio),
            min_close_distance=pick("displacement_min_close_distance", base.displacement.min_close_distance),
            min_close_distance_atr=pick("displacement_min_close_distance_atr", base.displacement.min_close_distance_atr),
            strong_close_min_body_range_ratio=pick("strong_close_min_body_range_ratio", base.displacement.strong_close_min_body_range_ratio),
            strong_close_min_boundary_distance=pick("strong_close_min_boundary_distance", base.displacement.strong_close_min_boundary_distance),
            strong_close_min_boundary_distance_atr=pick("strong_close_min_boundary_distance_atr", base.displacement.strong_close_min_boundary_distance_atr),
            invalidate_on_early_counter_liquidity_consumption=pick("invalidate_on_early_counter_liquidity_consumption", base.displacement.invalidate_on_early_counter_liquidity_consumption),
            allow_retest_rescue_after_early_liquidity_consumption=pick("allow_retest_rescue_after_early_liquidity_consumption", base.displacement.allow_retest_rescue_after_early_liquidity_consumption),
        ),
        liquidity=LiquidityFrameworkSpec(
            mode=pick("liquidity_target_mode", base.liquidity.mode),
            priority=tuple(pick("liquidity_target_priority", base.liquidity.priority) or ()),
            london_sweep_context_mode=pick("london_sweep_context_mode", base.liquidity.london_sweep_context_mode),
        ),
        entry_rule=pick("entry_rule", base.entry_rule),
        stop=StopSpec(
            rule=pick("stop_mode", base.stop.rule),
            stop_atr_multiple=pick("stop_atr_multiple", base.stop.stop_atr_multiple),
            stop_buffer_points=pick("stop_buffer_points", base.stop.stop_buffer_points),
            stop_buffer_atr_multiple=pick("stop_buffer_atr_multiple", base.stop.stop_buffer_atr_multiple),
            wick_reference_mode=pick("wick_reference_mode", base.stop.wick_reference_mode),
            fallback_stop_mode=pick("fallback_stop_mode", base.stop.fallback_stop_mode),
        ),
        target_rule=pick("target_rule", base.target_rule),
        target_r_multiple=pick("target_r_multiple", base.target_r_multiple),
        management=ManagementSpec(
            first_draw_target_rule=pick("first_draw_target_rule", base.management.first_draw_target_rule),
            minimum_rr_threshold=pick("minimum_rr_threshold", base.management.minimum_rr_threshold),
            breakeven_rule=base.management.breakeven_rule if pick("breakeven_enabled", base.management.breakeven_rule is not None) else None,
            breakeven_trigger_r=pick("breakeven_trigger_r", base.management.breakeven_trigger_r),
            breakeven_after_partial=pick("breakeven_after_partial", base.management.breakeven_after_partial),
            breakeven_after_first_draw=pick("breakeven_after_first_draw", base.management.breakeven_after_first_draw),
            partial_take_profit_rule=pick("partial_take_profit_rule", base.management.partial_take_profit_rule),
            partial_take_profit_r=pick("partial_take_profit_r", base.management.partial_take_profit_r),
            partial_take_profit_fraction=pick("partial_take_profit_fraction", base.management.partial_take_profit_fraction),
            runner_target_rule=pick("runner_target_rule", base.management.runner_target_rule),
            runner_target_priority=tuple(pick("runner_target_priority", base.management.runner_target_priority) or ()),
            trailing_stop_rule=pick("trailing_stop_rule", base.management.trailing_stop_rule) if pick("trailing_stop_enabled", base.management.trailing_stop_rule is not None) else None,
            trailing_stop_atr_multiple=pick("trailing_stop_atr_multiple", base.management.trailing_stop_atr_multiple),
            runner_trail_rule=pick("runner_trail_rule", base.management.runner_trail_rule),
            runner_trail_atr_multiple=pick("runner_trail_atr_multiple", base.management.runner_trail_atr_multiple),
        ),
        max_trades_per_session=pick("max_trades_per_session", base.max_trades_per_session),
        allowed_trade_windows=tuple(pick("allowed_trade_windows", base.allowed_trade_windows) or ()),
        allowed_days_of_week=tuple(pick("allowed_days_of_week", base.allowed_days_of_week) or ()),
        news_skip_rules=tuple(pick("news_skip_rules", base.news_skip_rules) or ()),
        discretionary_skip_reasons=tuple(pick("discretionary_skip_reasons", base.discretionary_skip_reasons) or ()),
        min_breakout_strength=pick("min_breakout_strength", base.min_breakout_strength),
        setup_feature_whitelist=pick("setup_feature_whitelist", base.setup_feature_whitelist),
    )
    validate_strategy_spec(spec)
    return spec


def validate_strategy_spec(spec: ORBStrategySpec) -> None:
    if spec.profile_name not in ORB_PROFILE_NAMES:
        raise ValueError(f"Unsupported ORB profile: {spec.profile_name}")
    if spec.instrument not in {"ES", "NQ", "MNQ", "CL", "MES"}:
        raise ValueError("Unsupported instrument for current ORB spec")
    if spec.opening_range_minutes <= 0:
        raise ValueError("opening_range_minutes must be greater than 0")
    if spec.max_trades_per_session < 1:
        raise ValueError("max_trades_per_session must be at least 1")
    if not spec.enable_long and not spec.enable_short:
        raise ValueError("At least one direction must be enabled")
    if spec.long_trigger != "break_above_or_high":
        raise ValueError("v1 only supports long_trigger='break_above_or_high'")
    if spec.short_trigger != "break_below_or_low":
        raise ValueError("v1 only supports short_trigger='break_below_or_low'")
    if spec.breakout_confirmation_rule not in {"close_beyond_level", "none"}:
        raise ValueError("Unsupported breakout_confirmation_rule")
    if spec.entry_rule != "next_bar_open":
        raise ValueError("Current engine only supports entry_rule='next_bar_open'")
    if spec.stop.rule not in SUPPORTED_STOP_RULES:
        raise ValueError("Unsupported stop rule")
    if spec.stop.rule == "atr" and (spec.stop.stop_atr_multiple is None or spec.stop.stop_atr_multiple <= 0):
        raise ValueError("ATR stop requires stop_atr_multiple > 0")
    if spec.stop.rule in {"or_midpoint_buffer", "range_wick_buffer", "midpoint_or_wick_buffer"}:
        if spec.stop.stop_buffer_points is None and spec.stop.stop_buffer_atr_multiple is None:
            raise ValueError("Buffered stop modes require stop_buffer_points and/or stop_buffer_atr_multiple")
    if spec.stop.stop_buffer_points is not None and spec.stop.stop_buffer_points <= 0:
        raise ValueError("stop_buffer_points must be positive")
    if spec.stop.stop_buffer_atr_multiple is not None and spec.stop.stop_buffer_atr_multiple <= 0:
        raise ValueError("stop_buffer_atr_multiple must be positive")
    if spec.stop.rule in {"range_wick_buffer", "midpoint_or_wick_buffer"} and spec.stop.wick_reference_mode not in SUPPORTED_WICK_REFERENCE_MODES - {None}:
        raise ValueError("Wick-based stop mode requires a valid wick_reference_mode")
    if spec.stop.fallback_stop_mode is not None and spec.stop.fallback_stop_mode not in {"or_boundary", "atr"}:
        raise ValueError("Unsupported fallback_stop_mode")
    if spec.target_rule not in SUPPORTED_TARGET_RULES:
        raise ValueError("Unsupported target_rule")
    if spec.target_rule == "r_multiple" and (spec.target_r_multiple is None or spec.target_r_multiple <= 0):
        raise ValueError("target_rule='r_multiple' requires a positive target_r_multiple")
    if spec.target_rule == "liquidity_sequence" and not spec.liquidity.priority:
        raise ValueError("liquidity_sequence target_rule requires liquidity_target_priority")
    if spec.min_or_width_atr is not None and spec.max_or_width_atr is not None and spec.min_or_width_atr > spec.max_or_width_atr:
        raise ValueError("min_or_width_atr cannot be greater than max_or_width_atr")
    if spec.volatility_filter_enabled and spec.min_or_width_atr is None and spec.max_or_width_atr is None:
        raise ValueError("volatility_filter_enabled requires min_or_width_atr and/or max_or_width_atr")
    if spec.trend_filter not in {"none", "above_below_ma"}:
        raise ValueError("Unsupported trend_filter")
    if spec.trend_filter != "none" and not spec.trend_column:
        raise ValueError("trend_filter requires trend_column")
    if spec.displacement.rule not in SUPPORTED_DISPLACEMENT_RULES:
        raise ValueError("Unsupported displacement_rule")
    if spec.displacement.rule == "candle_displacement":
        if all(
            value is None
            for value in (
                spec.displacement.min_body_size,
                spec.displacement.min_body_range_ratio,
                spec.displacement.min_close_distance,
                spec.displacement.min_close_distance_atr,
                spec.displacement.strong_close_min_body_range_ratio,
                spec.displacement.strong_close_min_boundary_distance,
                spec.displacement.strong_close_min_boundary_distance_atr,
            )
        ):
            raise ValueError("candle_displacement requires at least one threshold")
        if spec.displacement.min_body_range_ratio is not None and not (0 < spec.displacement.min_body_range_ratio <= 1):
            raise ValueError("displacement min_body_range_ratio must be in (0, 1]")
        if spec.displacement.strong_close_min_body_range_ratio is not None and not (0 < spec.displacement.strong_close_min_body_range_ratio <= 1):
            raise ValueError("strong_close_min_body_range_ratio must be in (0, 1]")
        if spec.displacement.min_body_size is not None and spec.displacement.min_body_size <= 0:
            raise ValueError("displacement min_body_size must be positive")
        if spec.displacement.min_close_distance is not None and spec.displacement.min_close_distance <= 0:
            raise ValueError("displacement min_close_distance must be positive")
        if spec.displacement.min_close_distance_atr is not None and spec.displacement.min_close_distance_atr <= 0:
            raise ValueError("displacement min_close_distance_atr must be positive")
        if spec.displacement.strong_close_min_boundary_distance is not None and spec.displacement.strong_close_min_boundary_distance <= 0:
            raise ValueError("strong_close_min_boundary_distance must be positive")
        if spec.displacement.strong_close_min_boundary_distance_atr is not None and spec.displacement.strong_close_min_boundary_distance_atr <= 0:
            raise ValueError("strong_close_min_boundary_distance_atr must be positive")
    if spec.displacement.allow_retest_rescue_after_early_liquidity_consumption and not spec.require_retest:
        raise ValueError("retest rescue requires require_retest=True")
    if spec.liquidity.mode not in {"none", "directional_continuation"}:
        raise ValueError("Unsupported liquidity_target_mode")
    invalid_priority = [level for level in spec.liquidity.priority if level not in SUPPORTED_LIQUIDITY_LEVELS]
    if invalid_priority:
        raise ValueError(f"Invalid liquidity target priority: {invalid_priority}")
    if spec.liquidity.mode == "directional_continuation" and not spec.liquidity.priority:
        raise ValueError("directional_continuation liquidity mode requires a priority list")
    invalid_runner_priority = [level for level in spec.management.runner_target_priority if level not in SUPPORTED_LIQUIDITY_LEVELS]
    if invalid_runner_priority:
        raise ValueError(f"Invalid runner target priority: {invalid_runner_priority}")
    if spec.management.first_draw_target_rule not in SUPPORTED_FIRST_DRAW_RULES:
        raise ValueError("Unsupported first_draw_target_rule")
    if spec.management.minimum_rr_threshold is not None and spec.management.minimum_rr_threshold < 2.0:
        raise ValueError("minimum_rr_threshold must be at least 2.0 for this strategy family")
    if spec.management.breakeven_rule is not None and not any(
        [spec.management.breakeven_after_partial, spec.management.breakeven_after_first_draw, spec.management.breakeven_trigger_r is not None]
    ):
        raise ValueError("breakeven rule requires partial, first-draw, or R trigger")
    if spec.management.partial_take_profit_rule is not None:
        if spec.management.partial_take_profit_rule == "r_multiple":
            if spec.management.partial_take_profit_r is None or spec.management.partial_take_profit_r <= 0:
                raise ValueError("r_multiple partial take-profit requires partial_take_profit_r > 0")
        if spec.management.partial_take_profit_fraction is None or not (0 < spec.management.partial_take_profit_fraction < 1):
            raise ValueError("partial_take_profit_fraction must be between 0 and 1")
    if spec.management.breakeven_after_partial and spec.management.partial_take_profit_rule is None:
        raise ValueError("breakeven_after_partial requires a partial take-profit rule")
    if spec.management.breakeven_after_first_draw and spec.management.first_draw_target_rule is None:
        raise ValueError("breakeven_after_first_draw requires first_draw_target_rule")
    if spec.management.runner_target_rule not in SUPPORTED_RUNNER_RULES:
        raise ValueError("Unsupported runner_target_rule")
    if spec.management.runner_target_rule == "liquidity_sequence" and not spec.management.runner_target_priority:
        raise ValueError("runner_target_rule='liquidity_sequence' requires runner_target_priority")
    if spec.management.trailing_stop_rule is not None and spec.management.trailing_stop_rule not in {"atr"}:
        raise ValueError("Unsupported trailing_stop_rule")
    if spec.management.trailing_stop_rule == "atr" and (
        spec.management.trailing_stop_atr_multiple is None or spec.management.trailing_stop_atr_multiple <= 0
    ):
        raise ValueError("ATR trailing stop requires trailing_stop_atr_multiple > 0")
    if spec.management.runner_trail_rule not in SUPPORTED_RUNNER_TRAIL_RULES:
        raise ValueError("Unsupported runner_trail_rule")
    if spec.management.runner_trail_rule == "atr_placeholder" and (
        spec.management.runner_trail_atr_multiple is None or spec.management.runner_trail_atr_multiple <= 0
    ):
        raise ValueError("atr_placeholder runner trail requires runner_trail_atr_multiple > 0")
    _validate_session(spec.session_start, spec.session_end)
    _validate_opening_range(spec)
    _validate_trade_windows(spec.allowed_trade_windows)
    _validate_days(spec.allowed_days_of_week)


def _validate_session(start: str, end: str) -> None:
    start_time = time.fromisoformat(start)
    end_time = time.fromisoformat(end)
    if start_time >= end_time:
        raise ValueError("session_start must be earlier than session_end")


def _validate_opening_range(spec: ORBStrategySpec) -> None:
    session_start = time.fromisoformat(spec.session_start)
    session_end = time.fromisoformat(spec.session_end)
    or_start = time.fromisoformat(spec.opening_range_start_time)
    or_end = time.fromisoformat(spec.opening_range_end_time)
    if not (session_start <= or_start < or_end <= session_end):
        raise ValueError("Opening range must be fully contained inside the session")
    actual_minutes = int((datetime.combine(datetime.min, or_end) - datetime.combine(datetime.min, or_start)).total_seconds() / 60)
    if actual_minutes != spec.opening_range_minutes:
        raise ValueError("opening_range_minutes must match opening_range_start_time/opening_range_end_time")
    if spec.latest_entry_time is not None:
        latest_entry = time.fromisoformat(spec.latest_entry_time)
        if latest_entry <= or_end or latest_entry > session_end:
            raise ValueError("latest_entry_time must be after opening range end and within the session")


def _validate_trade_windows(windows: tuple[str, ...]) -> None:
    for window in windows:
        try:
            start, end = window.split("-")
        except ValueError as exc:
            raise ValueError(f"Invalid trade window: {window}") from exc
        _validate_session(start, end)


def _validate_days(days: tuple[str, ...]) -> None:
    invalid = [day for day in days if day.lower() not in DAY_NAMES]
    if invalid:
        raise ValueError(f"Invalid day names: {invalid}")


def _base_spec(profile_name: str) -> ORBStrategySpec:
    profile_name = profile_name or "orb_basic"
    base = ORBStrategySpec(
        profile_name="orb_basic",
        instrument="ES",
        session_name="rth",
        session_timezone="America/New_York",
        ny_open_anchored=False,
        session_start="09:30",
        session_end="16:00",
        latest_entry_time=None,
        opening_range_start_time="09:30",
        opening_range_end_time="09:35",
        opening_range_minutes=5,
        enable_long=True,
        enable_short=True,
        long_trigger="break_above_or_high",
        short_trigger="break_below_or_low",
        breakout_confirmation_rule="close_beyond_level",
        require_retest=False,
        retest_rule="touch_level_then_close_through",
        trend_filter="none",
        trend_column="ema_20",
        volatility_filter_enabled=False,
        min_or_width_atr=None,
        max_or_width_atr=None,
        displacement=DisplacementSpec(
            rule="none",
            min_body_size=None,
            min_body_range_ratio=None,
            min_close_distance=None,
            min_close_distance_atr=None,
            strong_close_min_body_range_ratio=None,
            strong_close_min_boundary_distance=None,
            strong_close_min_boundary_distance_atr=None,
            invalidate_on_early_counter_liquidity_consumption=False,
            allow_retest_rescue_after_early_liquidity_consumption=False,
        ),
        liquidity=LiquidityFrameworkSpec(
            mode="none",
            priority=(),
            london_sweep_context_mode=None,
        ),
        entry_rule="next_bar_open",
        stop=StopSpec(
            rule="or_boundary",
            stop_atr_multiple=1.0,
            stop_buffer_points=None,
            stop_buffer_atr_multiple=None,
            wick_reference_mode=None,
            fallback_stop_mode="or_boundary",
        ),
        target_rule="r_multiple",
        target_r_multiple=2.0,
        management=ManagementSpec(
            first_draw_target_rule=None,
            minimum_rr_threshold=None,
            breakeven_rule=None,
            breakeven_trigger_r=None,
            breakeven_after_partial=False,
            breakeven_after_first_draw=False,
            partial_take_profit_rule=None,
            partial_take_profit_r=None,
            partial_take_profit_fraction=None,
            runner_target_rule=None,
            runner_target_priority=(),
            trailing_stop_rule=None,
            trailing_stop_atr_multiple=None,
            runner_trail_rule=None,
            runner_trail_atr_multiple=None,
        ),
        max_trades_per_session=1,
        allowed_trade_windows=(),
        allowed_days_of_week=(),
        news_skip_rules=(),
        discretionary_skip_reasons=(),
        min_breakout_strength=0.0,
        setup_feature_whitelist=None,
    )
    if profile_name == "orb_basic":
        return base
    if profile_name == "orb_retest":
        return replace(base, profile_name="orb_retest", require_retest=True)
    if profile_name == "orb_trend_filtered":
        return replace(base, profile_name="orb_trend_filtered", trend_filter="above_below_ma")
    if profile_name == "orb_volatility_filtered":
        return replace(base, profile_name="orb_volatility_filtered", volatility_filter_enabled=True, min_or_width_atr=0.3, max_or_width_atr=2.5)
    if profile_name == "nq_am_displacement_orb":
        return replace(
            base,
            profile_name="nq_am_displacement_orb",
            instrument="NQ",
            session_name="new_york_open",
            ny_open_anchored=True,
            session_start="09:30",
            session_end="12:00",
            latest_entry_time="11:30",
            opening_range_start_time="09:30",
            opening_range_end_time="09:45",
            opening_range_minutes=15,
            require_retest=False,
            displacement=DisplacementSpec(
                rule="candle_displacement",
                min_body_size=None,
                min_body_range_ratio=0.6,
                min_close_distance=2.0,
                min_close_distance_atr=None,
                strong_close_min_body_range_ratio=0.6,
                strong_close_min_boundary_distance=2.0,
                strong_close_min_boundary_distance_atr=None,
                invalidate_on_early_counter_liquidity_consumption=False,
                allow_retest_rescue_after_early_liquidity_consumption=False,
            ),
            liquidity=LiquidityFrameworkSpec(
                mode="directional_continuation",
                priority=(LiquidityLevel.PDH.value, LiquidityLevel.PDL.value, LiquidityLevel.DAY_HIGH.value, LiquidityLevel.DAY_LOW.value, LiquidityLevel.H4_HIGH.value, LiquidityLevel.H4_LOW.value, LiquidityLevel.LONDON_HIGH.value, LiquidityLevel.LONDON_LOW.value),
                london_sweep_context_mode=LiquidityLevel.LONDON_SWEEP_CONTEXT.value,
            ),
            stop=StopSpec(
                rule="midpoint_or_wick_buffer",
                stop_atr_multiple=1.0,
                stop_buffer_points=4.0,
                stop_buffer_atr_multiple=None,
                wick_reference_mode="largest_internal_wick",
                fallback_stop_mode="or_boundary",
            ),
            target_rule="r_multiple",
            target_r_multiple=2.0,
            management=ManagementSpec(
                first_draw_target_rule="first_liquidity_in_priority",
                minimum_rr_threshold=None,
                breakeven_rule=None,
                breakeven_trigger_r=None,
                breakeven_after_partial=False,
                breakeven_after_first_draw=True,
                partial_take_profit_rule=None,
                partial_take_profit_r=None,
                partial_take_profit_fraction=None,
                runner_target_rule=None,
                runner_target_priority=(LiquidityLevel.PDH.value, LiquidityLevel.DAY_HIGH.value, LiquidityLevel.H4_HIGH.value, LiquidityLevel.LONDON_HIGH.value),
                trailing_stop_rule=None,
                trailing_stop_atr_multiple=None,
                runner_trail_rule=None,
                runner_trail_atr_multiple=None,
            ),
            allowed_trade_windows=("09:45-11:30",),
            news_skip_rules=("major_fed_event", "major_us_macro_release"),
            discretionary_skip_reasons=("unusual_price_action", "poor_displacement_quality"),
        )
    raise ValueError(f"Unsupported ORB profile: {profile_name}")
