"""Setup detection package."""

from .base import BaseSetupDetector
from .documentation import format_strategy_spec, write_strategy_spec
from .liquidity import (
    LiquidityFrameworkSpec,
    LiquidityLevel,
    add_liquidity_levels,
    select_first_liquidity_target,
    select_runner_targets,
)
from .orb import ORBConfig, ORBSetupDetector
from .orb_session_vwap_retest import ORBSessionVWAPRetestConfig, ORBSessionVWAPRetestDetector
from .profiles import ORB_PROFILE_NAMES, get_orb_profile
from .rp_profits_8am_orb import RP_PROFILE_NAME, RPProfits8AMConfig, RPProfits8AMSetupDetector
from .session_context import add_trading_day, in_session, parse_session_window, within_any_trade_window
from .specification import DisplacementSpec, ManagementSpec, ORBStrategySpec, StopSpec, strategy_spec_from_config

__all__ = [
    "BaseSetupDetector",
    "DisplacementSpec",
    "LiquidityFrameworkSpec",
    "LiquidityLevel",
    "ManagementSpec",
    "ORBConfig",
    "ORBSetupDetector",
    "ORBSessionVWAPRetestConfig",
    "ORBSessionVWAPRetestDetector",
    "ORBStrategySpec",
    "ORB_PROFILE_NAMES",
    "RP_PROFILE_NAME",
    "RPProfits8AMConfig",
    "RPProfits8AMSetupDetector",
    "StopSpec",
    "add_liquidity_levels",
    "add_trading_day",
    "format_strategy_spec",
    "get_orb_profile",
    "in_session",
    "parse_session_window",
    "select_first_liquidity_target",
    "select_runner_targets",
    "strategy_spec_from_config",
    "within_any_trade_window",
    "write_strategy_spec",
]
