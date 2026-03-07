"""Optional FinRL integration scaffolding.

This module is intentionally dependency-light. The base framework must run
without FinRL installed. Any real adapter implementation should wrap the
custom environment and portfolio interfaces rather than replacing them.
"""


def is_available() -> bool:
    """Return True when FinRL is installed in the active environment."""
    try:
        import finrl  # noqa: F401
    except ImportError:
        return False
    return True
