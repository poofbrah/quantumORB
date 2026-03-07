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
from .profiles import ORB_PROFILE_NAMES, get_orb_profile
from .specification import DisplacementSpec, ManagementSpec, ORBStrategySpec, StopSpec, strategy_spec_from_config

__all__ = [
    "BaseSetupDetector",
    "DisplacementSpec",
    "LiquidityFrameworkSpec",
    "LiquidityLevel",
    "ManagementSpec",
    "ORBConfig",
    "ORBSetupDetector",
    "ORBStrategySpec",
    "ORB_PROFILE_NAMES",
    "StopSpec",
    "add_liquidity_levels",
    "format_strategy_spec",
    "get_orb_profile",
    "select_first_liquidity_target",
    "select_runner_targets",
    "strategy_spec_from_config",
    "write_strategy_spec",
]
