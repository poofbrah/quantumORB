from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ProjectConfig:
    name: str = "quantumORB"
    mode: str = "research"
    timezone: str = "America/New_York"
    root_dir: Path = Path(".")


@dataclass(slots=True)
class DataConfig:
    base_currency: str = "USD"
    asset_class: str = "futures"
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    session_template: str = "regular_trading_hours"
    timestamp_column: str = "timestamp"
    symbol_column: str = "symbol"
    timezone: str = "America/New_York"
    session_start: str | None = "09:30"
    session_end: str | None = "16:00"
    resample_rule: str | None = None


@dataclass(slots=True)
class BacktestConfig:
    engine_name: str = "custom"
    initial_capital: float = 100000.0
    commission_model: str = "fixed_per_unit"
    slippage_model: str = "fixed_per_unit"
    allow_short: bool = True
    entry_convention: str = "next_bar_open"
    one_position_only: bool = True
    position_size: float = 1.0
    commission_per_unit: float = 0.0
    slippage_per_unit: float = 0.0
    exit_on_session_end: bool = True
    intrabar_exit_conflict_policy: str = "stop_first"


@dataclass(slots=True)
class StrategyConfig:
    active_strategy: str = "orb"
    strategy_profile: str = "orb_basic"
    instrument: str = "ES"
    session_name: str = "rth"
    session_timezone: str = "America/New_York"
    ny_open_anchored: bool = False
    session_start: str = "09:30"
    session_end: str = "16:00"
    latest_entry_time: str | None = None
    opening_range_start_time: str = "09:30"
    opening_range_end_time: str = "09:35"
    opening_range_minutes: int = 5
    enable_long: bool = True
    enable_short: bool = True
    long_trigger: str = "break_above_or_high"
    short_trigger: str = "break_below_or_low"
    breakout_confirmation_rule: str = "close_beyond_level"
    require_breakout_close: bool = True
    require_retest: bool = False
    retest_rule: str = "touch_level_then_close_through"
    trend_filter: str = "none"
    trend_column: str = "ema_20"
    volatility_filter_enabled: bool = False
    min_or_width_atr: float | None = None
    max_or_width_atr: float | None = None
    displacement_rule: str = "none"
    displacement_min_body_size: float | None = None
    displacement_min_body_range_ratio: float | None = None
    displacement_min_close_distance: float | None = None
    displacement_min_close_distance_atr: float | None = None
    strong_close_min_body_range_ratio: float | None = None
    strong_close_min_boundary_distance: float | None = None
    strong_close_min_boundary_distance_atr: float | None = None
    invalidate_on_early_counter_liquidity_consumption: bool = False
    allow_retest_rescue_after_early_liquidity_consumption: bool = False
    liquidity_target_mode: str = "none"
    liquidity_target_priority: tuple[str, ...] | None = None
    london_sweep_context_mode: str | None = None
    first_draw_target_rule: str | None = None
    minimum_rr_threshold: float | None = None
    entry_rule: str = "next_bar_open"
    stop_mode: str = "or_boundary"
    stop_atr_multiple: float = 1.0
    stop_buffer_points: float | None = None
    stop_buffer_atr_multiple: float | None = None
    wick_reference_mode: str | None = None
    fallback_stop_mode: str | None = "or_boundary"
    target_rule: str = "r_multiple"
    target_r_multiple: float = 2.0
    breakeven_enabled: bool = False
    breakeven_trigger_r: float | None = None
    breakeven_after_partial: bool = False
    breakeven_after_first_draw: bool = False
    partial_take_profit_enabled: bool = False
    partial_take_profit_rule: str | None = None
    partial_take_profit_r: float | None = None
    partial_take_profit_fraction: float | None = None
    runner_target_rule: str | None = None
    runner_target_priority: tuple[str, ...] | None = None
    trailing_stop_enabled: bool = False
    trailing_stop_rule: str | None = None
    trailing_stop_atr_multiple: float | None = None
    runner_trail_rule: str | None = None
    runner_trail_atr_multiple: float | None = None
    max_trades_per_session: int = 1
    allowed_trade_windows: tuple[str, ...] | None = None
    allowed_days_of_week: tuple[str, ...] | None = None
    news_skip_rules: tuple[str, ...] | None = None
    discretionary_skip_reasons: tuple[str, ...] | None = None
    min_breakout_strength: float = 0.0
    label_horizon_bars: int = 20
    setup_feature_whitelist: tuple[str, ...] | None = None


@dataclass(slots=True)
class ModelingConfig:
    setup_quality_enabled: bool = False
    ga_enabled: bool = False
    rl_enabled: bool = False


@dataclass(slots=True)
class IntegrationsConfig:
    qlib_enabled: bool = False
    finrl_enabled: bool = False


@dataclass(slots=True)
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    integrations: IntegrationsConfig = field(default_factory=IntegrationsConfig)

    def to_dict(self) -> dict:
        return asdict(self)
