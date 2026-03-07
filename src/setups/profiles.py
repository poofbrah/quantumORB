from __future__ import annotations

from .specification import ORB_PROFILE_NAMES, ORBStrategySpec, _base_spec


def get_orb_profile(name: str) -> ORBStrategySpec:
    return _base_spec(name)
