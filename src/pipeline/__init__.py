"""SmartMoneyEngine market integration pipeline."""

from typing import Any

__all__ = [
    "MarketPipelineError",
    "MarketPipelineReport",
    "MarketPipelineRunner",
    "PipelineStageResult",
    "run_market_pipeline",
]


def __getattr__(name: str) -> Any:
    """Lazy export to avoid import side effects during module execution."""
    if name in __all__:
        from src.pipeline import market_pipeline as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
