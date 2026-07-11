"""
Symbol management for SmartMoneyEngine.

Provides built-in index symbols and CSV-backed equity symbol loading with
validation, deduplication, and search utilities.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.core.logger import logger

REQUIRED_CSV_COLUMNS = ("symbol", "exchange", "instrument_type")
VALID_EXCHANGES = frozenset({"NSE", "BSE", "MCX"})
VALID_INSTRUMENT_TYPES = frozenset({"EQ", "INDEX", "FUT", "OPT"})

DEFAULT_INDEX_SYMBOLS: tuple[tuple[str, str, str], ...] = (
    ("NIFTY50", "NSE", "INDEX"),
    ("BANKNIFTY", "NSE", "INDEX"),
    ("FINNIFTY", "NSE", "INDEX"),
    ("MIDCPNIFTY", "NSE", "INDEX"),
)


class SymbolManagerError(Exception):
    """Base exception for symbol management failures."""


class SymbolCSVNotFoundError(SymbolManagerError):
    """Raised when a symbol CSV file does not exist."""


class SymbolCSVFormatError(SymbolManagerError):
    """Raised when a symbol CSV file has invalid structure."""


class InvalidSymbolRowError(SymbolManagerError):
    """Raised when a CSV row fails symbol validation."""


class DuplicateSymbolError(SymbolManagerError):
    """Raised when a duplicate symbol is encountered."""


@dataclass(frozen=True, slots=True)
class SymbolRecord:
    """Normalized tradable symbol definition."""

    symbol: str
    exchange: str
    instrument_type: str

    @property
    def key(self) -> str:
        """Return the case-insensitive lookup key."""
        return self.symbol.casefold()


class SymbolManager:
    """
    Manage index and equity symbols for the trading engine.

    Built-in index symbols are loaded at initialization. Additional equity
    symbols can be loaded from CSV files with validation and deduplication.
    """

    def __init__(self) -> None:
        self._symbols: dict[str, SymbolRecord] = {}
        self._last_loaded_count = 0
        self._last_duplicates_removed = 0
        self._last_invalid_rows = 0
        self._load_default_indices()

    def _load_default_indices(self) -> None:
        """Load predefined index symbols."""
        for symbol, exchange, instrument_type in DEFAULT_INDEX_SYMBOLS:
            record = self._build_record(symbol, exchange, instrument_type)
            self._symbols[record.key] = record
        logger.info("Loaded %s default index symbol(s).", len(DEFAULT_INDEX_SYMBOLS))

    @staticmethod
    def _build_record(symbol: str, exchange: str, instrument_type: str) -> SymbolRecord:
        """Normalize and construct a symbol record."""
        return SymbolRecord(
            symbol=symbol.strip().upper(),
            exchange=exchange.strip().upper(),
            instrument_type=instrument_type.strip().upper(),
        )

    @staticmethod
    def _sorted(records: Iterable[SymbolRecord]) -> list[SymbolRecord]:
        """Return records sorted alphabetically by symbol."""
        return sorted(records, key=lambda item: item.symbol)

    def load_csv(self, path: str | Path) -> int:
        """
        Load equity symbols from a CSV file.

        Parameters
        ----------
        path : str | Path
            CSV path with columns ``symbol``, ``exchange``, ``instrument_type``.

        Returns
        -------
        int
            Number of newly loaded symbols.

        Raises
        ------
        SymbolCSVNotFoundError
            If the CSV file does not exist.
        SymbolCSVFormatError
            If required columns are missing.
        """
        csv_path = Path(path)
        if not csv_path.exists():
            raise SymbolCSVNotFoundError(f"Symbol CSV file not found: {csv_path}")

        loaded_count = 0
        duplicates_removed = 0
        invalid_rows = 0
        seen_in_file: set[str] = set()

        logger.info("Loading symbols from %s", csv_path)

        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise SymbolCSVFormatError(f"Symbol CSV file is empty: {csv_path}")

            normalized_columns = {
                column.strip().casefold(): column for column in reader.fieldnames
            }
            missing_columns = [
                column
                for column in REQUIRED_CSV_COLUMNS
                if column not in normalized_columns
            ]
            if missing_columns:
                raise SymbolCSVFormatError(
                    f"Symbol CSV missing required columns {missing_columns}: {csv_path}"
                )

            for row_number, row in enumerate(reader, start=2):
                symbol = (row.get(normalized_columns["symbol"]) or "").strip()
                exchange = (row.get(normalized_columns["exchange"]) or "").strip()
                instrument_type = (
                    row.get(normalized_columns["instrument_type"]) or ""
                ).strip()

                if not symbol or not exchange or not instrument_type:
                    invalid_rows += 1
                    logger.warning("Rejected blank row at line %s in %s.", row_number, csv_path)
                    continue

                if exchange.upper() not in VALID_EXCHANGES:
                    invalid_rows += 1
                    logger.warning(
                        "Rejected invalid exchange '%s' at line %s in %s.",
                        exchange,
                        row_number,
                        csv_path,
                    )
                    continue

                if instrument_type.upper() not in VALID_INSTRUMENT_TYPES:
                    invalid_rows += 1
                    logger.warning(
                        "Rejected invalid instrument type '%s' at line %s in %s.",
                        instrument_type,
                        row_number,
                        csv_path,
                    )
                    continue

                record = self._build_record(symbol, exchange, instrument_type)
                if record.key in seen_in_file:
                    duplicates_removed += 1
                    logger.warning(
                        "Rejected duplicate symbol '%s' at line %s in %s.",
                        record.symbol,
                        row_number,
                        csv_path,
                    )
                    continue

                seen_in_file.add(record.key)

                if record.key in self._symbols:
                    duplicates_removed += 1
                    logger.warning(
                        "Rejected duplicate symbol '%s' already loaded at line %s in %s.",
                        record.symbol,
                        row_number,
                        csv_path,
                    )
                    continue

                self._symbols[record.key] = record
                loaded_count += 1

        self._last_loaded_count = loaded_count
        self._last_duplicates_removed = duplicates_removed
        self._last_invalid_rows = invalid_rows

        logger.info("Loaded count: %s", loaded_count)
        logger.info("Duplicates removed: %s", duplicates_removed)
        logger.info("Invalid rows: %s", invalid_rows)

        return loaded_count

    def get_all(self) -> list[SymbolRecord]:
        """Return all symbols sorted alphabetically."""
        return self._sorted(self._symbols.values())

    def get_indices(self) -> list[SymbolRecord]:
        """Return index symbols sorted alphabetically."""
        indices = (
            record
            for record in self._symbols.values()
            if record.instrument_type == "INDEX"
        )
        return self._sorted(indices)

    def get_equities(self) -> list[SymbolRecord]:
        """Return equity symbols sorted alphabetically."""
        equities = (
            record
            for record in self._symbols.values()
            if record.instrument_type == "EQ"
        )
        return self._sorted(equities)

    def exists(self, symbol: str) -> bool:
        """
        Check whether a symbol exists.

        Matching is case-insensitive.
        """
        if not symbol or not symbol.strip():
            return False
        return symbol.strip().casefold() in self._symbols

    def search(self, text: str) -> list[SymbolRecord]:
        """
        Search symbols by case-insensitive substring match.

        Parameters
        ----------
        text : str
            Search text applied to symbol, exchange, and instrument type.

        Returns
        -------
        list[SymbolRecord]
            Matching symbols sorted alphabetically.
        """
        if not text or not text.strip():
            return []

        needle = text.strip().casefold()
        matches = (
            record
            for record in self._symbols.values()
            if needle in record.symbol.casefold()
            or needle in record.exchange.casefold()
            or needle in record.instrument_type.casefold()
        )
        return self._sorted(matches)

    def count(self) -> int:
        """Return the total number of loaded symbols."""
        return len(self._symbols)


def main() -> int:
    """
    CLI entry point.

    Loads built-in indices, imports a sample equity CSV, and prints summary
    information including a sample search result.
    """
    sample_csv = "\n".join(
        [
            "symbol,exchange,instrument_type",
            "RELIANCE,NSE,EQ",
            "SBIN,NSE,EQ",
            "RELIANCE,NSE,EQ",
            ",NSE,EQ",
            "TCS,INVALID,EQ",
            "INFY,NSE,XX",
        ]
    )

    manager = SymbolManager()

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as handle:
        handle.write(sample_csv)
        temp_path = Path(handle.name)

    try:
        manager.load_csv(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Total Symbols: {manager.count()}")
    print("Indices:")
    for record in manager.get_indices():
        print(f"  - {record.symbol} ({record.exchange}, {record.instrument_type})")

    print("Stocks:")
    for record in manager.get_equities():
        print(f"  - {record.symbol} ({record.exchange}, {record.instrument_type})")

    search_text = "bank"
    print(f"Search Result ('{search_text}'):")
    for record in manager.search(search_text):
        print(f"  - {record.symbol} ({record.exchange}, {record.instrument_type})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
