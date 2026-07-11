"""
Production historical data loader for SmartMoneyEngine.

Loads FYERS-downloaded OHLCV candles from yearly CSV/Parquet files under
``data/historical/<symbol>/<timeframe>/``.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HISTORICAL_DIR = PROJECT_ROOT / "data" / "historical"
IST = ZoneInfo("Asia/Kolkata")

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
OHLC_COLUMNS = ("open", "high", "low", "close")

STORAGE_DIR_MAP: dict[str, str] = {
    "NIFTY50": "NSE_NIFTY50-INDEX",
    "BANKNIFTY": "NSE_NIFTYBANK-INDEX",
    "FINNIFTY": "NSE_FINNIFTY-INDEX",
    "MIDCPNIFTY": "NSE_MIDCPNIFTY-INDEX",
}


class DataLoaderError(Exception):
    """Base exception for data loader failures."""


class HistoricalDataNotFoundError(DataLoaderError):
    """Raised when no historical files exist for the requested query."""


class MissingColumnsError(DataLoaderError):
    """Raised when required OHLCV columns are missing."""


class EmptyDatasetError(DataLoaderError):
    """Raised when the filtered dataset contains no rows."""


class InvalidTimestampError(DataLoaderError):
    """Raised when timestamps are invalid or not timezone-aware."""


class DuplicateTimestampError(DataLoaderError):
    """Raised when duplicate timestamps remain after deduplication."""


class HistoricalDataLoader:
    """
    Load historical OHLCV data from yearly CSV/Parquet files.

    Parameters
    ----------
    base_dir : Path | None, optional
        Root historical data directory. Defaults to ``data/historical``.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir if base_dir is not None else DEFAULT_HISTORICAL_DIR

    @staticmethod
    def _parse_date(value: date | str) -> date:
        """Parse a date object or ISO date string."""
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value.strip())
        raise DataLoaderError(f"Unsupported date value: {value!r}")

    @staticmethod
    def _normalize_timeframe(timeframe: str) -> str:
        """Normalize timeframe strings to storage directory names."""
        normalized = timeframe.strip().lower()
        replacements = {
            "5-minute": "5",
            "5min": "5",
            "5m": "5",
            "1-minute": "1",
            "1min": "1",
            "1m": "1",
            "15-minute": "15",
            "15min": "15",
            "15m": "15",
            "1d": "D",
            "day": "D",
            "daily": "D",
        }
        return replacements.get(normalized, timeframe.strip())

    @staticmethod
    def _years_for_range(start: date, end: date) -> list[int]:
        """Return calendar years overlapping an inclusive date range."""
        return list(range(start.year, end.year + 1))

    def _resolve_storage_dir(self, symbol: str) -> str:
        """
        Resolve an internal symbol name to a historical storage directory.

        Parameters
        ----------
        symbol : str
            Symbol such as ``NIFTY50`` or ``NSE_NIFTY50-INDEX``.

        Returns
        -------
        str
            Directory name under ``data/historical/``.
        """
        normalized = symbol.strip().upper()
        mapped = STORAGE_DIR_MAP.get(normalized)
        if mapped is not None:
            return mapped

        direct = symbol.strip().replace(":", "_").replace("/", "_")
        if (self.base_dir / direct).exists():
            return direct

        equity_candidate = f"NSE_{normalized}-EQ"
        if (self.base_dir / equity_candidate).exists():
            return equity_candidate

        index_candidate = f"NSE_{normalized}-INDEX"
        if (self.base_dir / index_candidate).exists():
            return index_candidate

        return mapped if mapped is not None else direct

    def _symbol_path(self, symbol: str, timeframe: str) -> Path:
        """Build the symbol/timeframe directory path."""
        storage_dir = self._resolve_storage_dir(symbol)
        normalized_timeframe = self._normalize_timeframe(timeframe)
        return self.base_dir / storage_dir / normalized_timeframe

    def _select_yearly_file(
        self,
        symbol_path: Path,
        year: int,
        prefer_parquet: bool,
    ) -> Path | None:
        """
        Select a yearly data file, preferring Parquet when available.

        Returns
        -------
        Path | None
            Selected file path, or ``None`` if no file exists for the year.
        """
        parquet_path = symbol_path / f"{year}.parquet"
        csv_path = symbol_path / f"{year}.csv"

        if prefer_parquet and parquet_path.exists():
            return parquet_path
        if csv_path.exists():
            return csv_path
        if parquet_path.exists():
            return parquet_path
        return None

    def _discover_files(
        self,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
        prefer_parquet: bool,
    ) -> list[Path]:
        """
        Discover only the yearly files required for the requested date range.

        Raises
        ------
        HistoricalDataNotFoundError
            If no files exist for any required year.
        """
        symbol_path = self._symbol_path(symbol, timeframe)
        if not symbol_path.exists():
            raise HistoricalDataNotFoundError(
                f"Historical path not found: {symbol_path}"
            )

        selected_files: list[Path] = []
        for year in self._years_for_range(start, end):
            selected = self._select_yearly_file(symbol_path, year, prefer_parquet)
            if selected is not None:
                selected_files.append(selected)

        if not selected_files:
            raise HistoricalDataNotFoundError(
                f"No historical files found for {symbol} ({timeframe}) "
                f"between {start.isoformat()} and {end.isoformat()}."
            )

        logger.info(
            "Discovered %s yearly file(s) for %s/%s: %s",
            len(selected_files),
            symbol,
            self._normalize_timeframe(timeframe),
            ", ".join(path.name for path in selected_files),
        )
        return selected_files

    @staticmethod
    def _read_file(path: Path) -> pd.DataFrame:
        """Read a yearly CSV or Parquet file."""
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    @staticmethod
    def _validate_columns(frame: pd.DataFrame, source: Path) -> None:
        """Ensure required OHLCV columns are present."""
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise MissingColumnsError(
                f"Missing required columns {missing} in {source}."
            )

    @staticmethod
    def _normalize_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
        """
        Convert timestamps to timezone-aware Asia/Kolkata datetimes.

        Raises
        ------
        InvalidTimestampError
            If timestamps are invalid or timezone conversion fails.
        """
        working = frame.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")

        if working["timestamp"].isna().any():
            raise InvalidTimestampError("Dataset contains invalid timestamp values.")

        if working["timestamp"].dt.tz is None:
            working["timestamp"] = working["timestamp"].dt.tz_localize(IST)
        else:
            working["timestamp"] = working["timestamp"].dt.tz_convert(IST)

        return working

    @staticmethod
    def _validate_timestamp_ordering(frame: pd.DataFrame) -> None:
        """Raise when timestamps are not strictly increasing after sorting."""
        if frame["timestamp"].duplicated().any():
            duplicates = int(frame["timestamp"].duplicated().sum())
            raise DuplicateTimestampError(
                f"Dataset contains {duplicates} duplicate timestamp(s)."
            )

        if not frame["timestamp"].is_monotonic_increasing:
            raise InvalidTimestampError("Timestamps are not in ascending order.")

    def _filter_date_range(
        self,
        frame: pd.DataFrame,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Filter rows to an inclusive date range."""
        start_ts = pd.Timestamp(datetime.combine(start, datetime.min.time()), tz=IST)
        end_ts = pd.Timestamp(datetime.combine(end, datetime.max.time()), tz=IST)
        mask = (frame["timestamp"] >= start_ts) & (frame["timestamp"] <= end_ts)
        return frame.loc[mask].reset_index(drop=True)

    def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: date | str,
        end_date: date | str,
        prefer_parquet: bool = True,
    ) -> pd.DataFrame:
        """
        Load historical OHLCV candles for a symbol and date range.

        Parameters
        ----------
        symbol : str
            Symbol name such as ``NIFTY50``.
        timeframe : str
            Candle timeframe such as ``5`` or ``5-minute``.
        start_date : date | str
            Inclusive start date.
        end_date : date | str
            Inclusive end date.
        prefer_parquet : bool, optional
            Prefer Parquet over CSV when both exist.

        Returns
        -------
        pd.DataFrame
            Timezone-aware OHLCV DataFrame sorted by timestamp.

        Raises
        ------
        HistoricalDataNotFoundError
            If required files are missing.
        MissingColumnsError
            If required columns are absent.
        EmptyDatasetError
            If no rows remain after filtering.
        InvalidTimestampError
            If timestamps are invalid or unordered.
        DuplicateTimestampError
            If duplicate timestamps remain after deduplication.
        """
        started = time.perf_counter()
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)

        if start > end:
            raise DataLoaderError("start_date must be on or before end_date.")

        files = self._discover_files(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            prefer_parquet=prefer_parquet,
        )

        frames: list[pd.DataFrame] = []
        rows_before_dedupe = 0

        for path in files:
            logger.info("Loading historical file: %s", path)
            frame = self._read_file(path)
            self._validate_columns(frame, path)
            frame = self._normalize_timestamps(frame)
            rows_before_dedupe += len(frame)
            frames.append(frame)
            logger.info("Loaded %s row(s) from %s", len(frame), path.name)

        merged = pd.concat(frames, ignore_index=True)
        duplicates_removed = int(merged.duplicated(subset=["timestamp"]).sum())
        if duplicates_removed:
            logger.info("Duplicates removed: %s", duplicates_removed)
            merged = merged.drop_duplicates(subset=["timestamp"], keep="first")

        merged = merged.sort_values("timestamp").reset_index(drop=True)
        merged = self._filter_date_range(merged, start, end)

        if merged.empty:
            raise EmptyDatasetError(
                f"No rows found for {symbol} ({timeframe}) between "
                f"{start.isoformat()} and {end.isoformat()}."
            )

        self._validate_timestamp_ordering(merged)

        elapsed = time.perf_counter() - started
        logger.info(
            "Historical load completed for %s/%s: rows=%s files=%s elapsed=%.3fs",
            symbol,
            self._normalize_timeframe(timeframe),
            len(merged),
            len(files),
            elapsed,
        )
        return merged


def main() -> int:
    """
    CLI demo.

    Loads NIFTY50 5-minute candles for the last 30 calendar days and prints
    row count, timestamp range, and memory usage.
    """
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        loader = HistoricalDataLoader()
        frame = loader.load(
            symbol="NIFTY50",
            timeframe="5",
            start_date=start_date,
            end_date=end_date,
            prefer_parquet=True,
        )

        memory_bytes = int(frame.memory_usage(deep=True).sum())
        print(f"Rows: {len(frame)}")
        print(f"First timestamp: {frame['timestamp'].iloc[0]}")
        print(f"Last timestamp: {frame['timestamp'].iloc[-1]}")
        print(f"Memory usage: {memory_bytes / 1024 / 1024:.2f} MB")
        return 0
    except DataLoaderError as exc:
        logger.error("Data loader error: %s", exc)
        print(f"Data loader error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected data loader failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
