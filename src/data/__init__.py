"""Data layer package for SmartMoneyEngine."""

from typing import Any

__all__ = ["SymbolManager"]


def __getattr__(name: str) -> Any:
    """Lazy export to avoid import side effects during module execution."""
    if name == "SymbolManager":
        from src.data.symbols.symbol_manager import SymbolManager

        return SymbolManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
