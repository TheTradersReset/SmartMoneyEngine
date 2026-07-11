"""Symbol management package."""

from typing import Any

__all__ = [
    "BulkDownloadError",
    "BulkDownloadReport",
    "BulkHistoricalDownloader",
    "DuplicateSymbolError",
    "InvalidSymbolRowError",
    "SymbolCSVFormatError",
    "SymbolCSVNotFoundError",
    "SymbolDownloadResult",
    "SymbolManager",
    "SymbolManagerError",
    "SymbolRecord",
    "download_all_symbols",
    "to_fyers_symbol",
]


def __getattr__(name: str) -> Any:
    """Lazy export to avoid import side effects during module execution."""
    if name in __all__:
        if name in {
            "BulkDownloadError",
            "BulkDownloadReport",
            "BulkHistoricalDownloader",
            "SymbolDownloadResult",
            "download_all_symbols",
            "to_fyers_symbol",
        }:
            from src.data.symbols import bulk_downloader as module
        else:
            from src.data.symbols import symbol_manager as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
