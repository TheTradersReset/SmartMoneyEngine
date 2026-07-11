"""Historical market data loader package."""

from typing import Any

__all__ = [
    "DataLoaderError",
    "DuplicateTimestampError",
    "EmptyDatasetError",
    "HistoricalDataLoader",
    "HistoricalDataNotFoundError",
    "InvalidTimestampError",
    "MissingColumnsError",
]


def __getattr__(name: str) -> Any:
    """Lazy export to avoid import side effects during module execution."""
    if name in __all__:
        from src.data.loader import data_loader as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
