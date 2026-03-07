"""Optional Qlib integration scaffolding.

This module is intentionally dependency-light. The base framework must run
without Qlib installed. Any real adapter implementation should guard imports
and expose a narrow translation layer into the custom project schemas.
"""


def is_available() -> bool:
    """Return True when Qlib is installed in the active environment."""
    try:
        import qlib  # noqa: F401
    except ImportError:
        return False
    return True
