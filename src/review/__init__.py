"""SmartMoneyEngine review dashboard package."""

from typing import Any

__all__ = [
    "ReviewDashboard",
    "ReviewDashboardError",
    "build_review_dashboard",
]


def __getattr__(name: str) -> Any:
    """Lazy export to avoid import side effects during module execution."""
    if name in __all__:
        from src.review import review_dashboard as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
