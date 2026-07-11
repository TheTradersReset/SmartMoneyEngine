"""
Fyers historical candle downloader for SmartMoneyEngine.

Fetches OHLCV data through ``FyersClient``, splits large date ranges into
FYERS-compliant chunks, validates the merged dataset, and persists yearly
CSV/Parquet files under ``data/historical/``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.brokers.fyers.client import FyersClient, FyersClientError
from src.core.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HISTORICAL_DIR = PROJECT_ROOT / "data" / "historical"
IST = ZoneInfo("Asia/Kolkata")

INTRADAY_RESOLUTIONS = {
    "1", "2", "3", "5", "10", "15", "20", "30", "45", "60", "120", "180", "240",
}
DAILY_RESOLUTIONS = {"D", "1D", "DAY"}
SECONDS_RESOLUTIONS = {"5S", "10S", "15S", "30S", "45S"}

MAX_DAYS_INTRADAY = 100
MAX_DAYS_DAILY = 366
MAX_DAYS_SECONDS = 30

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class HistoricalDownloadError(Exception):
    """Raised when historical data download or validation fails."""


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of post-download candle validation."""

    duplicate_candles: int
    missing_candles: int
    invalid_ohlc_count: int
    negative_volume_count: int
    ordering_issues: int


@dataclass
class DownloadSummary:
    """Summary report for a historical download run."""

    symbol: str
    resolution: str
    from_date: date
    to_date: date
    total_candles: int
    downloaded_candles: int
    missing_candles: int
    duplicate_candles: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    invalid_ohlc_count: int = 0
    negative_volume_count: int = 0
    ordering_issues: int = 0
    saved_files: list[Path] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable summary dictionary."""
        return {
            "symbol": self.symbol,
            "resolution": self.resolution,
            "from_date": self.from_date.isoformat(),
            "to_date": self.to_date.isoformat(),
            "total_candles": self.total_candles,
            "downloaded_candles": self.downloaded_candles,
            "missing_candles": self.missing_candles,
            "duplicate_candles": self.duplicate_candles,
            "first_timestamp": (
                self.first_timestamp.isoformat() if self.first_timestamp else None
            ),
            "last_timestamp": (
                self.last_timestamp.isoformat() if self.last_timestamp else None
            ),
            "invalid_ohlc_count": self.invalid_ohlc_count,
            "negative_volume_count": self.negative_volume_count,
            "ordering_issues": self.ordering_issues,
            "saved_files": [str(path) for path in self.saved_files],
        }


def _parse_date(value: date | str) -> date:
    """Parse a date object or ISO date string."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value.strip())
    raise HistoricalDownloadError(f"Unsupported date value: {value!r}")


def _sanitize_symbol(symbol: str) -> str:
    """Convert a Fyers symbol into a filesystem-safe directory name."""
    return symbol.strip().replace(":", "_").replace("/", "_")


def _normalize_resolution(resolution: str) -> str:
    """Normalize resolution strings for FYERS API calls."""
    normalized = resolution.strip().upper()
    if normalized == "DAY":
        return "D"
    return normalized


def _max_chunk_days(resolution: str) -> int:
    """Return the maximum inclusive calendar days allowed per FYERS history request."""
    normalized = _normalize_resolution(resolution)
    if normalized in DAILY_RESOLUTIONS:
        return MAX_DAYS_DAILY
    if normalized in SECONDS_RESOLUTIONS:
        return MAX_DAYS_SECONDS
    if normalized in INTRADAY_RESOLUTIONS:
        return MAX_DAYS_INTRADAY
    raise HistoricalDownloadError(f"Unsupported resolution: {resolution}")


def _resolution_interval_seconds(resolution: str) -> int | None:
    """
    Return candle interval in seconds for intraday/seconds resolutions.

    Returns ``None`` for daily resolutions.
    """
    normalized = _normalize_resolution(resolution)
    if normalized in DAILY_RESOLUTIONS:
        return None
    if normalized in SECONDS_RESOLUTIONS:
        return int(normalized[:-1])
    if normalized in INTRADAY_RESOLUTIONS:
        return int(normalized) * 60
    raise HistoricalDownloadError(f"Unsupported resolution: {resolution}")


def split_date_range(
    from_date: date | str,
    to_date: date | str,
    resolution: str,
) -> list[tuple[date, date]]:
    """
    Split a date range into FYERS-compliant request windows.

    Parameters
    ----------
    from_date : date | str
        Inclusive start date.
    to_date : date | str
        Inclusive end date.
    resolution : str
        Candle resolution.

    Returns
    -------
    list[tuple[date, date]]
        Ordered inclusive date chunks.
    """
    start = _parse_date(from_date)
    end = _parse_date(to_date)
    if start > end:
        raise HistoricalDownloadError("from_date must be on or before to_date.")

    chunk_limit = _max_chunk_days(resolution)
    chunks: list[tuple[date, date]] = []
    cursor = start

    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_limit - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)

    logger.info(
        "Split date range %s to %s (%s) into %s chunk(s).",
        start.isoformat(),
        end.isoformat(),
        resolution,
        len(chunks),
    )
    return chunks


def _candles_to_dataframe(candles: list[list[Any]]) -> pd.DataFrame:
    """Convert raw FYERS candle arrays into a typed DataFrame."""
    if not candles:
        return pd.DataFrame(columns=CANDLE_COLUMNS)

    frame = pd.DataFrame(candles, columns=CANDLE_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True).dt.tz_convert(IST)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _validate_dataframe(frame: pd.DataFrame, resolution: str) -> ValidationResult:
    """
    Validate merged candle data for duplicates, gaps, OHLC integrity, and ordering.

    Parameters
    ----------
    frame : pd.DataFrame
        Candle DataFrame sorted by timestamp.
    resolution : str
        Candle resolution.

    Returns
    -------
    ValidationResult
        Validation metrics.
    """
    if frame.empty:
        return ValidationResult(
            duplicate_candles=0,
            missing_candles=0,
            invalid_ohlc_count=0,
            negative_volume_count=0,
            ordering_issues=0,
        )

    duplicate_candles = int(frame.duplicated(subset=["timestamp"], keep="first").sum())

    invalid_ohlc_mask = (
        frame[["open", "high", "low", "close"]].isna().any(axis=1)
        | (frame["high"] < frame["low"])
        | (frame["high"] < frame["open"])
        | (frame["high"] < frame["close"])
        | (frame["low"] > frame["open"])
        | (frame["low"] > frame["close"])
    )
    invalid_ohlc_count = int(invalid_ohlc_mask.sum())

    negative_volume_count = int((frame["volume"] < 0).sum())

    ordering_issues = int((frame["timestamp"].diff().dropna() <= pd.Timedelta(0)).sum())

    interval_seconds = _resolution_interval_seconds(resolution)
    missing_candles = 0
    if interval_seconds is not None:
        timestamps = frame["timestamp"].sort_values().reset_index(drop=True)
        for index in range(1, len(timestamps)):
            previous = timestamps.iloc[index - 1]
            current = timestamps.iloc[index]
            if previous.date() != current.date():
                continue
            gap_seconds = (current - previous).total_seconds()
            if gap_seconds > interval_seconds * 1.5:
                missing_candles += max(int(round(gap_seconds / interval_seconds)) - 1, 0)

    logger.info(
        "Validation complete: duplicates=%s missing=%s invalid_ohlc=%s "
        "negative_volume=%s ordering_issues=%s",
        duplicate_candles,
        missing_candles,
        invalid_ohlc_count,
        negative_volume_count,
        ordering_issues,
    )

    return ValidationResult(
        duplicate_candles=duplicate_candles,
        missing_candles=missing_candles,
        invalid_ohlc_count=invalid_ohlc_count,
        negative_volume_count=negative_volume_count,
        ordering_issues=ordering_issues,
    )


class HistoricalDownloader:
    """
    Download, merge, validate, and persist FYERS historical candles.

    Parameters
    ----------
    client : FyersClient
        Authenticated Fyers client.
    output_dir : Path, optional
        Root directory for persisted files.
    max_retries : int, optional
        Retry attempts per chunk download.
    retry_backoff_seconds : float, optional
        Initial retry backoff in seconds.
    """

    def __init__(
        self,
        client: FyersClient,
        output_dir: Path | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self.client = client
        self.output_dir = output_dir if output_dir is not None else DEFAULT_HISTORICAL_DIR
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def _fetch_chunk(
        self,
        symbol: str,
        resolution: str,
        chunk_from: date,
        chunk_to: date,
    ) -> list[list[Any]]:
        """
        Fetch one history chunk with retry logic.

        Returns
        -------
        list[list[Any]]
            Raw FYERS candle rows.
        """
        action = (
            f"history chunk {symbol} {resolution} "
            f"{chunk_from.isoformat()}->{chunk_to.isoformat()}"
        )
        backoff = self.retry_backoff_seconds
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            logger.info("Downloading %s (attempt %s/%s)", action, attempt, self.max_retries)
            try:
                response = self.client.get_history(
                    symbol=symbol,
                    resolution=resolution,
                    date_from=chunk_from.isoformat(),
                    date_to=chunk_to.isoformat(),
                )
                candles = response.get("candles", [])
                if not isinstance(candles, list):
                    raise HistoricalDownloadError(
                        f"Unexpected candles payload for {action}."
                    )
                logger.info(
                    "Downloaded %s candle(s) for %s to %s.",
                    len(candles),
                    chunk_from.isoformat(),
                    chunk_to.isoformat(),
                )
                return candles
            except (FyersClientError, HistoricalDownloadError) as exc:
                last_error = exc
                logger.warning("Chunk download failed: %s", exc)
                if attempt < self.max_retries:
                    logger.info("Retrying %s in %.1f seconds.", action, backoff)
                    time.sleep(backoff)
                    backoff *= 2

        assert last_error is not None
        raise HistoricalDownloadError(
            f"Failed to download history chunk after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def download(
        self,
        symbol: str,
        resolution: str,
        from_date: date | str,
        to_date: date | str,
        save: bool = True,
    ) -> tuple[pd.DataFrame, DownloadSummary]:
        """
        Download historical candles for a symbol and optional persistence.

        Parameters
        ----------
        symbol : str
            Fyers symbol, for example ``NSE:NIFTY50-INDEX``.
        resolution : str
            Candle resolution such as ``5`` or ``1D``.
        from_date : date | str
            Inclusive start date.
        to_date : date | str
            Inclusive end date.
        save : bool, optional
            Persist yearly CSV and Parquet files when ``True``.

        Returns
        -------
        tuple[pd.DataFrame, DownloadSummary]
            Merged candle DataFrame and summary report.
        """
        if not symbol or not symbol.strip():
            raise HistoricalDownloadError("Symbol is required.")
        if not resolution or not str(resolution).strip():
            raise HistoricalDownloadError("Resolution is required.")

        normalized_symbol = symbol.strip()
        normalized_resolution = _normalize_resolution(str(resolution))
        start = _parse_date(from_date)
        end = _parse_date(to_date)

        logger.info(
            "Starting historical download for %s (%s) from %s to %s.",
            normalized_symbol,
            normalized_resolution,
            start.isoformat(),
            end.isoformat(),
        )

        chunks = split_date_range(start, end, normalized_resolution)
        raw_candles: list[list[Any]] = []
        for chunk_from, chunk_to in chunks:
            raw_candles.extend(
                self._fetch_chunk(
                    symbol=normalized_symbol,
                    resolution=normalized_resolution,
                    chunk_from=chunk_from,
                    chunk_to=chunk_to,
                )
            )

        frame = _candles_to_dataframe(raw_candles)
        duplicate_before_dedupe = int(frame.duplicated(subset=["timestamp"]).sum()) if not frame.empty else 0

        frame = frame.drop_duplicates(subset=["timestamp"], keep="first")
        frame = frame.sort_values("timestamp").reset_index(drop=True)

        validation = _validate_dataframe(frame, normalized_resolution)
        duplicate_candles = max(duplicate_before_dedupe, validation.duplicate_candles)

        first_timestamp = frame["timestamp"].iloc[0].to_pydatetime() if not frame.empty else None
        last_timestamp = frame["timestamp"].iloc[-1].to_pydatetime() if not frame.empty else None
        downloaded_candles = len(frame)
        total_candles = downloaded_candles + validation.missing_candles

        saved_files: list[Path] = []
        if save and not frame.empty:
            saved_files = self._save_by_year(
                frame=frame,
                symbol=normalized_symbol,
                resolution=normalized_resolution,
            )

        summary = DownloadSummary(
            symbol=normalized_symbol,
            resolution=normalized_resolution,
            from_date=start,
            to_date=end,
            total_candles=total_candles,
            downloaded_candles=downloaded_candles,
            missing_candles=validation.missing_candles,
            duplicate_candles=duplicate_candles,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            invalid_ohlc_count=validation.invalid_ohlc_count,
            negative_volume_count=validation.negative_volume_count,
            ordering_issues=validation.ordering_issues,
            saved_files=saved_files,
        )

        logger.info(
            "Historical download completed for %s (%s): downloaded=%s missing=%s duplicates=%s.",
            normalized_symbol,
            normalized_resolution,
            summary.downloaded_candles,
            summary.missing_candles,
            summary.duplicate_candles,
        )
        return frame, summary

    def _save_by_year(
        self,
        frame: pd.DataFrame,
        symbol: str,
        resolution: str,
    ) -> list[Path]:
        """
        Persist yearly CSV and Parquet files.

        Layout::

            data/historical/<symbol>/<resolution>/YYYY.csv
            data/historical/<symbol>/<resolution>/YYYY.parquet
        """
        symbol_dir = self.output_dir / _sanitize_symbol(symbol) / resolution
        symbol_dir.mkdir(parents=True, exist_ok=True)

        saved_files: list[Path] = []
        working = frame.copy()
        working["year"] = working["timestamp"].dt.year

        for year, year_frame in working.groupby("year", sort=True):
            export_frame = year_frame.drop(columns=["year"]).reset_index(drop=True)
            csv_path = symbol_dir / f"{year}.csv"
            parquet_path = symbol_dir / f"{year}.parquet"

            export_frame.to_csv(csv_path, index=False)
            export_frame.to_parquet(parquet_path, index=False)

            saved_files.extend([csv_path, parquet_path])
            logger.info("Saved %s candles to %s and %s.", len(export_frame), csv_path, parquet_path)

        return saved_files


def download_history(
    symbol: str,
    resolution: str,
    from_date: date | str,
    to_date: date | str,
    save: bool = True,
    client: FyersClient | None = None,
) -> tuple[pd.DataFrame, DownloadSummary]:
    """
    Download historical candles using ``HistoricalDownloader``.

    Parameters
    ----------
    symbol : str
        Fyers symbol.
    resolution : str
        Candle resolution.
    from_date : date | str
        Inclusive start date.
    to_date : date | str
        Inclusive end date.
    save : bool, optional
        Persist yearly CSV/Parquet files when ``True``.
    client : FyersClient | None, optional
        Existing client instance. Created from token file when omitted.

    Returns
    -------
    tuple[pd.DataFrame, DownloadSummary]
        Merged candles and summary report.
    """
    active_client = client if client is not None else FyersClient.from_token_file()
    downloader = HistoricalDownloader(client=active_client)
    return downloader.download(
        symbol=symbol,
        resolution=resolution,
        from_date=from_date,
        to_date=to_date,
        save=save,
    )


def print_summary(summary: DownloadSummary) -> None:
    """Print a human-readable download summary."""
    print("Historical Download Summary")
    print(f"Symbol: {summary.symbol}")
    print(f"Resolution: {summary.resolution}")
    print(f"Date Range: {summary.from_date.isoformat()} to {summary.to_date.isoformat()}")
    print(f"Total Candles: {summary.total_candles}")
    print(f"Downloaded Candles: {summary.downloaded_candles}")
    print(f"Missing Candles: {summary.missing_candles}")
    print(f"Duplicate Candles: {summary.duplicate_candles}")
    print(
        "First Timestamp: "
        f"{summary.first_timestamp.isoformat() if summary.first_timestamp else 'N/A'}"
    )
    print(
        "Last Timestamp: "
        f"{summary.last_timestamp.isoformat() if summary.last_timestamp else 'N/A'}"
    )
    if summary.saved_files:
        print("Saved Files:")
        for path in summary.saved_files:
            print(f"  - {path}")


def main() -> int:
    """
    CLI entry point.

    Downloads NIFTY 5-minute candles for the last 30 calendar days,
    saves yearly files, and prints the summary report.
    """
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        logger.info(
            "Starting CLI historical download for NIFTY 5-minute (%s to %s).",
            start_date.isoformat(),
            end_date.isoformat(),
        )

        _, summary = download_history(
            symbol="NSE:NIFTY50-INDEX",
            resolution="5",
            from_date=start_date,
            to_date=end_date,
            save=True,
        )
        print_summary(summary)
        return 0
    except HistoricalDownloadError as exc:
        logger.error("Historical download error: %s", exc)
        print(f"Historical download error: {exc}", file=sys.stderr)
        return 1
    except FyersClientError as exc:
        logger.error("Fyers client error: %s", exc)
        print(f"Fyers client error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected historical download failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
