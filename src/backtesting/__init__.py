"""SmartMoneyEngine backtesting layer."""

from typing import Any

__all__ = [
    "BacktestEngine",
    "BacktestEngineError",
    "BacktestReport",
    "TradeResult",
    "run_backtest",
]


def __getattr__(name: str) -> Any:
    """Lazy export to avoid import side effects during module execution."""
    if name in __all__:
        from src.backtesting import backtest_engine as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
