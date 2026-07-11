"""
Bulk historical downloader for all managed symbols.

Orchestrates ``SymbolManager`` and the FYERS ``HistoricalDownloader`` to
download every configured symbol, retry failures, continue on errors, and
emit structured reports.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.brokers.fyers.client import FyersClient, FyersClientError
from src.brokers.fyers.historical import (
    DownloadSummary,
    HistoricalDownloadError,
    HistoricalDownloader,
)
from src.core.logger import logger
from src.data.symbols.symbol_manager import SymbolCSVNotFoundError, SymbolManager, SymbolRecord

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EQUITIES_CSV = PROJECT_ROOT / "data" / "symbols" / "equities.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "historical"

INDEX_FYERS_MAP: dict[str, str] = {
    "NIFTY50": "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "FINNIFTY": "NSE:FINNIFTY-INDEX",
    "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX",
}

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0


class BulkDownloadError(Exception):
    """Raised when bulk download orchestration fails."""


@dataclass
class SymbolDownloadResult:
    """Download outcome for a single symbol."""

    symbol: str
    fyers_symbol: str
    exchange: str
    instrument_type: str
    status: str
    attempts: int
    downloaded_candles: int = 0
    missing_candles: int = 0
    duplicate_candles: int = 0
    saved_files: list[str] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable result dictionary."""
        return asdict(self)


@dataclass
class BulkDownloadReport:
    """Aggregate report for a bulk download run."""

    started_at: datetime
    finished_at: datetime
    resolution: str
    from_date: date
    to_date: date
    total_symbols: int
    successful_symbols: int
    failed_symbols: int
    total_candles: int
    results: list[SymbolDownloadResult] = field(default_factory=list)
    report_path: Path | None = None
    failed_symbols_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable report dictionary."""
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "resolution": self.resolution,
            "from_date": self.from_date.isoformat(),
            "to_date": self.to_date.isoformat(),
            "total_symbols": self.total_symbols,
            "successful_symbols": self.successful_symbols,
            "failed_symbols": self.failed_symbols,
            "total_candles": self.total_candles,
            "report_path": str(self.report_path) if self.report_path else None,
            "failed_symbols_path": (
                str(self.failed_symbols_path) if self.failed_symbols_path else None
            ),
            "results": [result.as_dict() for result in self.results],
        }


def to_fyers_symbol(record: SymbolRecord) -> str:
    """
    Convert an internal symbol record to a FYERS API symbol string.

    Parameters
    ----------
    record : SymbolRecord
        Managed symbol record.

    Returns
    -------
    str
        FYERS symbol identifier.
    """
    if record.instrument_type == "INDEX":
        mapped = INDEX_FYERS_MAP.get(record.symbol)
        if mapped is not None:
            return mapped
        return f"{record.exchange}:{record.symbol}-INDEX"

    if record.instrument_type == "EQ":
        return f"{record.exchange}:{record.symbol}-EQ"

    if record.instrument_type == "FUT":
        return f"{record.exchange}:{record.symbol}-FUT"

    if record.instrument_type == "OPT":
        return f"{record.exchange}:{record.symbol}-OPT"

    return f"{record.exchange}:{record.symbol}"


class BulkHistoricalDownloader:
    """
    Download historical candles for every symbol in a ``SymbolManager``.

    Parameters
    ----------
    symbol_manager : SymbolManager
        Symbol registry to iterate.
    client : FyersClient | None, optional
        Authenticated FYERS client. Created from token file when omitted.
    report_dir : Path | None, optional
        Directory for generated reports.
    max_retries : int, optional
        Retry attempts per symbol.
    retry_backoff_seconds : float, optional
        Initial retry backoff in seconds.
    """

    def __init__(
        self,
        symbol_manager: SymbolManager,
        client: FyersClient | None = None,
        report_dir: Path | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self.symbol_manager = symbol_manager
        self.client = client if client is not None else FyersClient.from_token_file()
        self.report_dir = report_dir if report_dir is not None else DEFAULT_REPORT_DIR
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._historical = HistoricalDownloader(client=self.client)

    def _download_symbol(
        self,
        record: SymbolRecord,
        resolution: str,
        from_date: date,
        to_date: date,
        save: bool,
    ) -> SymbolDownloadResult:
        """
        Download one symbol with retry logic.

        Failures are returned as failed results instead of raising, so the
        bulk run can continue with remaining symbols.
        """
        fyers_symbol = to_fyers_symbol(record)
        backoff = self.retry_backoff_seconds
        last_error = "Unknown error"

        for attempt in range(1, self.max_retries + 1):
            logger.info(
                "Downloading %s as %s (attempt %s/%s).",
                record.symbol,
                fyers_symbol,
                attempt,
                self.max_retries,
            )
            try:
                _, summary = self._historical.download(
                    symbol=fyers_symbol,
                    resolution=resolution,
                    from_date=from_date,
                    to_date=to_date,
                    save=save,
                )
                logger.info(
                    "Download succeeded for %s: %s candle(s).",
                    record.symbol,
                    summary.downloaded_candles,
                )
                return SymbolDownloadResult(
                    symbol=record.symbol,
                    fyers_symbol=fyers_symbol,
                    exchange=record.exchange,
                    instrument_type=record.instrument_type,
                    status="success",
                    attempts=attempt,
                    downloaded_candles=summary.downloaded_candles,
                    missing_candles=summary.missing_candles,
                    duplicate_candles=summary.duplicate_candles,
                    saved_files=[str(path) for path in summary.saved_files],
                )
            except (HistoricalDownloadError, FyersClientError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Download failed for %s on attempt %s/%s: %s",
                    record.symbol,
                    attempt,
                    self.max_retries,
                    exc,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.exception(
                    "Unexpected download failure for %s on attempt %s/%s.",
                    record.symbol,
                    attempt,
                    self.max_retries,
                )

            if attempt < self.max_retries:
                logger.info(
                    "Retrying %s in %.1f seconds.",
                    record.symbol,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2

        logger.error(
            "Download failed for %s after %s attempts.",
            record.symbol,
            self.max_retries,
        )
        return SymbolDownloadResult(
            symbol=record.symbol,
            fyers_symbol=fyers_symbol,
            exchange=record.exchange,
            instrument_type=record.instrument_type,
            status="failed",
            attempts=self.max_retries,
            error=last_error,
        )

    def download_all(
        self,
        resolution: str,
        from_date: date | str,
        to_date: date | str,
        save: bool = True,
    ) -> BulkDownloadReport:
        """
        Download historical data for every managed symbol.

        Parameters
        ----------
        resolution : str
            Candle resolution such as ``5`` or ``1D``.
        from_date : date | str
            Inclusive start date.
        to_date : date | str
            Inclusive end date.
        save : bool, optional
            Persist yearly CSV/Parquet files when ``True``.

        Returns
        -------
        BulkDownloadReport
            Aggregate download report with per-symbol results.
        """
        if isinstance(from_date, str):
            start = date.fromisoformat(from_date.strip())
        else:
            start = from_date

        if isinstance(to_date, str):
            end = date.fromisoformat(to_date.strip())
        else:
            end = to_date

        if start > end:
            raise BulkDownloadError("from_date must be on or before to_date.")

        symbols = self.symbol_manager.get_all()
        started_at = datetime.now()
        results: list[SymbolDownloadResult] = []

        logger.info(
            "Starting bulk historical download for %s symbol(s), resolution=%s, %s to %s.",
            len(symbols),
            resolution,
            start.isoformat(),
            end.isoformat(),
        )

        for record in symbols:
            result = self._download_symbol(
                record=record,
                resolution=resolution,
                from_date=start,
                to_date=end,
                save=save,
            )
            results.append(result)

        finished_at = datetime.now()
        successful = [item for item in results if item.status == "success"]
        failed = [item for item in results if item.status == "failed"]

        report = BulkDownloadReport(
            started_at=started_at,
            finished_at=finished_at,
            resolution=str(resolution),
            from_date=start,
            to_date=end,
            total_symbols=len(results),
            successful_symbols=len(successful),
            failed_symbols=len(failed),
            total_candles=sum(item.downloaded_candles for item in successful),
            results=results,
        )

        report.report_path, report.failed_symbols_path = self._write_reports(report)
        logger.info(
            "Bulk download completed: success=%s failed=%s total_candles=%s",
            report.successful_symbols,
            report.failed_symbols,
            report.total_candles,
        )
        return report

    def _write_reports(self, report: BulkDownloadReport) -> tuple[Path, Path | None]:
        """
        Persist the download report and failed symbol list.

        Returns
        -------
        tuple[Path, Path | None]
            Paths to the JSON report and failed-symbol CSV (if any failures).
        """
        self.report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = report.started_at.strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"download_report_{timestamp}.json"
        failed_path = self.report_dir / f"failed_symbols_{timestamp}.csv"

        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report.as_dict(), handle, indent=2)

        logger.info("Download report saved to %s", report_path)

        failed_results = [item for item in report.results if item.status == "failed"]
        if failed_results:
            with failed_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "symbol",
                        "fyers_symbol",
                        "exchange",
                        "instrument_type",
                        "attempts",
                        "error",
                    ],
                )
                writer.writeheader()
                for item in failed_results:
                    writer.writerow(
                        {
                            "symbol": item.symbol,
                            "fyers_symbol": item.fyers_symbol,
                            "exchange": item.exchange,
                            "instrument_type": item.instrument_type,
                            "attempts": item.attempts,
                            "error": item.error or "",
                        }
                    )
            logger.info("Failed symbol list saved to %s", failed_path)
            return report_path, failed_path

        logger.info("No failed symbols; failed symbol list not created.")
        return report_path, None


def download_all_symbols(
    resolution: str,
    from_date: date | str,
    to_date: date | str,
    equities_csv: Path | str | None = DEFAULT_EQUITIES_CSV,
    save: bool = True,
) -> BulkDownloadReport:
    """
    Load symbols and download historical data for each one.

    Parameters
    ----------
    resolution : str
        Candle resolution.
    from_date : date | str
        Inclusive start date.
    to_date : date | str
        Inclusive end date.
    equities_csv : Path | str | None, optional
        Optional equity symbol CSV path.
    save : bool, optional
        Persist yearly files when ``True``.

    Returns
    -------
    BulkDownloadReport
        Aggregate download report.
    """
    manager = SymbolManager()
    if equities_csv is not None:
        csv_path = Path(equities_csv)
        if csv_path.exists():
            manager.load_csv(csv_path)
        else:
            logger.info("Equity symbol CSV not found at %s; using indices only.", csv_path)

    downloader = BulkHistoricalDownloader(symbol_manager=manager)
    return downloader.download_all(
        resolution=resolution,
        from_date=from_date,
        to_date=to_date,
        save=save,
    )


def print_summary(report: BulkDownloadReport) -> None:
    """Print a human-readable bulk download summary."""
    duration = report.finished_at - report.started_at
    print("Bulk Historical Download Summary")
    print(f"Resolution: {report.resolution}")
    print(f"Date Range: {report.from_date.isoformat()} to {report.to_date.isoformat()}")
    print(f"Total Symbols: {report.total_symbols}")
    print(f"Successful: {report.successful_symbols}")
    print(f"Failed: {report.failed_symbols}")
    print(f"Total Candles: {report.total_candles}")
    print(f"Duration: {duration}")
    if report.report_path is not None:
        print(f"Download Report: {report.report_path}")
    if report.failed_symbols_path is not None:
        print(f"Failed Symbol List: {report.failed_symbols_path}")

    if report.failed_symbols:
        print("Failed Symbols:")
        for item in report.results:
            if item.status == "failed":
                print(f"  - {item.symbol} ({item.fyers_symbol}): {item.error}")


def main() -> int:
    """
    CLI entry point.

    Downloads 5-minute history for all managed symbols over the last 30
    calendar days, saves yearly files, and emits reports.
    """
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        report = download_all_symbols(
            resolution="5",
            from_date=start_date,
            to_date=end_date,
            save=True,
        )
        print_summary(report)
        return 0 if report.failed_symbols == 0 else 2
    except BulkDownloadError as exc:
        logger.error("Bulk download error: %s", exc)
        print(f"Bulk download error: {exc}", file=sys.stderr)
        return 1
    except (FyersClientError, HistoricalDownloadError) as exc:
        logger.error("Historical download error: %s", exc)
        print(f"Historical download error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected bulk download failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
