"""
Historical expansion validation for SmartMoneyEngine.

Downloads, validates, and summarizes expanded FYERS historical datasets
without modifying any signal or strategy modules.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.brokers.fyers.client import FyersClient, FyersClientError
from src.brokers.fyers.historical import HistoricalDownloadError, HistoricalDownloader
from src.data.loader.data_loader import DataLoaderError, HistoricalDataLoader
from src.data.validation.dataset_validator import DatasetValidator, ValidationResult

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "dataset_summary.json"

EXPANSION_DAYS = 365
SUPPORTED_SYMBOLS: tuple[str, ...] = ("NIFTY50", "BANKNIFTY", "FINNIFTY")
TIMEFRAME_MAP: dict[str, str] = {
    "5M": "5",
    "15M": "15",
    "1H": "60",
}
FYERS_SYMBOL_MAP: dict[str, str] = {
    "NIFTY50": "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "FINNIFTY": "NSE:FINNIFTY-INDEX",
}
IST = ZoneInfo("Asia/Kolkata")


class HistoricalExpansionError(Exception):
    """Raised when historical expansion validation fails."""


@dataclass
class DatasetQualityReport:
    """Validation summary for one symbol/timeframe dataset."""

    symbol: str
    timeframe: str
    storage_timeframe: str
    fyers_symbol: str
    bar_count: int
    start_timestamp: str | None
    end_timestamp: str | None
    date_range_start: str
    date_range_end: str
    download_success: bool
    download_candles: int
    download_missing_candles: int
    download_duplicate_candles: int
    duplicates: int
    missing_candles: int
    invalid_ohlc: int
    timezone_consistent: bool
    timezone: str
    gap_count: int
    validation_valid: bool
    data_quality_score: float
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    saved_files: list[str] = field(default_factory=list)
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable dataset report."""
        return asdict(self)


@dataclass
class ExpansionSummaryReport:
    """Master historical expansion validation report."""

    expansion_days: int
    start_date: str
    end_date: str
    symbols: list[str]
    timeframes: list[str]
    total_datasets: int
    successful_datasets: int
    failed_datasets: int
    overall_quality_score: float
    bars_by_symbol: dict[str, int]
    bars_by_timeframe: dict[str, int]
    execution_time_seconds: float
    datasets: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable expansion summary."""
        return asdict(self)


class HistoricalExpansionValidator:
    """
    Download and validate expanded historical datasets.

    Parameters
    ----------
    expansion_days : int, optional
        Calendar days of history to request.
    auto_download : bool, optional
        Download missing FYERS history before validation.
    """

    def __init__(
        self,
        expansion_days: int = EXPANSION_DAYS,
        auto_download: bool = True,
    ) -> None:
        if expansion_days < 1:
            raise HistoricalExpansionError("expansion_days must be at least 1.")
        self.expansion_days = expansion_days
        self.auto_download = auto_download
        self._loader = HistoricalDataLoader()

    def _resolve_range(self, end_date: date | None = None) -> tuple[date, date]:
        end = end_date if end_date is not None else date.today()
        start = end - timedelta(days=self.expansion_days)
        return start, end

    @staticmethod
    def _storage_timeframe(timeframe_label: str) -> str:
        if timeframe_label not in TIMEFRAME_MAP:
            raise HistoricalExpansionError(f"Unsupported timeframe label: {timeframe_label}")
        return TIMEFRAME_MAP[timeframe_label]

    def _download_dataset(
        self,
        symbol: str,
        storage_tf: str,
        start: date,
        end: date,
    ) -> tuple[bool, dict[str, Any]]:
        fyers_symbol = FYERS_SYMBOL_MAP.get(symbol)
        if fyers_symbol is None:
            return False, {"error_message": f"No FYERS mapping for {symbol}."}

        try:
            client = FyersClient.from_token_file()
            downloader = HistoricalDownloader(client=client)
            _, summary = downloader.download(
                symbol=fyers_symbol,
                resolution=storage_tf,
                from_date=start,
                to_date=end,
                save=True,
            )
            return True, {
                "download_candles": summary.downloaded_candles,
                "download_missing_candles": summary.missing_candles,
                "download_duplicate_candles": summary.duplicate_candles,
                "invalid_ohlc": summary.invalid_ohlc_count,
                "saved_files": [str(path) for path in summary.saved_files],
            }
        except (FyersClientError, HistoricalDownloadError) as exc:
            logger.error("Download failed for %s/%s: %s", symbol, storage_tf, exc)
            return False, {"error_message": str(exc)}

    def _load_dataset(
        self,
        symbol: str,
        storage_tf: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        return self._loader.load(
            symbol=symbol,
            timeframe=storage_tf,
            start_date=start,
            end_date=end,
            prefer_parquet=True,
        )

    @staticmethod
    def _validate_timezone(frame: pd.DataFrame) -> tuple[bool, list[str]]:
        errors: list[str] = []
        if frame.empty:
            return False, ["Dataset is empty."]

        timestamps = frame["timestamp"]
        if timestamps.dt.tz is None:
            errors.append("Timestamps are timezone-naive.")
            return False, errors

        timezones = {str(ts.tz) for ts in timestamps.dropna().head(100)}
        if len(timezones) > 1:
            errors.append(f"Mixed timezones detected: {sorted(timezones)}")
            return False, errors

        sample_tz = str(timestamps.iloc[0].tz)
        if sample_tz not in {"Asia/Kolkata", "Asia/Calcutta"}:
            errors.append(f"Expected Asia/Kolkata timezone, found {sample_tz}.")
            return False, errors

        return True, errors

    @staticmethod
    def _count_missing_candles(validation: ValidationResult) -> int:
        return int(validation.missing_timestamps + validation.gap_count)

    @staticmethod
    def _quality_score(
        bar_count: int,
        validation: ValidationResult,
        timezone_consistent: bool,
        download_missing: int,
        download_duplicates: int,
    ) -> float:
        if bar_count <= 0:
            return 0.0

        score = 100.0
        duplicate_rate = validation.duplicates / bar_count
        score -= min(duplicate_rate * 500.0, 25.0)
        score -= min(download_duplicates * 0.5, 10.0)

        missing_total = HistoricalExpansionValidator._count_missing_candles(validation)
        missing_rate = missing_total / bar_count
        score -= min(missing_rate * 400.0, 25.0)
        score -= min(download_missing * 0.05, 15.0)

        invalid_rate = validation.invalid_ohlc / bar_count
        score -= min(invalid_rate * 1000.0, 30.0)

        if not timezone_consistent:
            score -= 20.0
        if not validation.is_valid:
            score -= 10.0
        score -= min(len(validation.errors) * 5.0, 15.0)
        score -= min(len(validation.warnings) * 1.0, 5.0)

        return round(max(0.0, min(score, 100.0)), 1)

    def validate_dataset(
        self,
        symbol: str,
        timeframe_label: str,
        start: date,
        end: date,
    ) -> DatasetQualityReport:
        """Download (optional) and validate one symbol/timeframe dataset."""
        storage_tf = self._storage_timeframe(timeframe_label)
        fyers_symbol = FYERS_SYMBOL_MAP.get(symbol, "")

        download_success = False
        download_meta: dict[str, Any] = {
            "download_candles": 0,
            "download_missing_candles": 0,
            "download_duplicate_candles": 0,
            "saved_files": [],
        }
        error_message: str | None = None

        if self.auto_download:
            download_success, download_meta = self._download_dataset(
                symbol, storage_tf, start, end
            )
            if not download_success:
                error_message = str(download_meta.get("error_message", "Download failed."))

        try:
            frame = self._load_dataset(symbol, storage_tf, start, end)
        except DataLoaderError as exc:
            if error_message is None:
                error_message = str(exc)
            return DatasetQualityReport(
                symbol=symbol,
                timeframe=timeframe_label,
                storage_timeframe=storage_tf,
                fyers_symbol=fyers_symbol,
                bar_count=0,
                start_timestamp=None,
                end_timestamp=None,
                date_range_start=start.isoformat(),
                date_range_end=end.isoformat(),
                download_success=download_success,
                download_candles=int(download_meta.get("download_candles", 0)),
                download_missing_candles=int(download_meta.get("download_missing_candles", 0)),
                download_duplicate_candles=int(download_meta.get("download_duplicate_candles", 0)),
                duplicates=0,
                missing_candles=0,
                invalid_ohlc=0,
                timezone_consistent=False,
                timezone="Asia/Kolkata",
                gap_count=0,
                validation_valid=False,
                data_quality_score=0.0,
                error_message=error_message,
            )

        duplicate_count = int(frame.duplicated(subset=["timestamp"]).sum()) if not frame.empty else 0
        validator = DatasetValidator(timeframe=storage_tf)
        validation = validator.validate(frame)
        timezone_consistent, timezone_errors = self._validate_timezone(frame)

        missing_candles = self._count_missing_candles(validation)
        quality_score = self._quality_score(
            bar_count=len(frame),
            validation=validation,
            timezone_consistent=timezone_consistent,
            download_missing=int(download_meta.get("download_missing_candles", 0)),
            download_duplicates=int(download_meta.get("download_duplicate_candles", 0)),
        )

        all_errors = list(validation.errors) + timezone_errors
        start_ts = frame["timestamp"].iloc[0].isoformat() if not frame.empty else None
        end_ts = frame["timestamp"].iloc[-1].isoformat() if not frame.empty else None

        return DatasetQualityReport(
            symbol=symbol,
            timeframe=timeframe_label,
            storage_timeframe=storage_tf,
            fyers_symbol=fyers_symbol,
            bar_count=len(frame),
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            date_range_start=start.isoformat(),
            date_range_end=end.isoformat(),
            download_success=download_success or not frame.empty,
            download_candles=int(download_meta.get("download_candles", len(frame))),
            download_missing_candles=int(download_meta.get("download_missing_candles", 0)),
            download_duplicate_candles=max(
                duplicate_count,
                int(download_meta.get("download_duplicate_candles", 0)),
            ),
            duplicates=max(duplicate_count, validation.duplicates),
            missing_candles=missing_candles,
            invalid_ohlc=validation.invalid_ohlc,
            timezone_consistent=timezone_consistent,
            timezone="Asia/Kolkata",
            gap_count=validation.gap_count,
            validation_valid=validation.is_valid and timezone_consistent,
            data_quality_score=quality_score,
            validation_errors=all_errors,
            validation_warnings=validation.warnings,
            saved_files=list(download_meta.get("saved_files", [])),
            error_message=error_message,
        )

    def run(
        self,
        symbols: tuple[str, ...] | None = None,
        timeframes: tuple[str, ...] | None = None,
        end_date: date | None = None,
    ) -> ExpansionSummaryReport:
        """Download and validate all configured datasets."""
        started = time.perf_counter()
        symbol_list = list(symbols if symbols is not None else SUPPORTED_SYMBOLS)
        timeframe_list = list(timeframes if timeframes is not None else TIMEFRAME_MAP)
        start, end = self._resolve_range(end_date)

        reports: list[DatasetQualityReport] = []
        for symbol in symbol_list:
            for timeframe_label in timeframe_list:
                logger.info("Validating expanded dataset: %s / %s", symbol, timeframe_label)
                reports.append(self.validate_dataset(symbol, timeframe_label, start, end))

        bars_by_symbol: dict[str, int] = {symbol: 0 for symbol in symbol_list}
        bars_by_timeframe: dict[str, int] = {label: 0 for label in timeframe_list}
        successful = 0
        quality_scores: list[float] = []

        for report in reports:
            bars_by_symbol[report.symbol] = bars_by_symbol.get(report.symbol, 0) + report.bar_count
            bars_by_timeframe[report.timeframe] = (
                bars_by_timeframe.get(report.timeframe, 0) + report.bar_count
            )
            if report.bar_count > 0 and report.validation_valid:
                successful += 1
            if report.bar_count > 0:
                quality_scores.append(report.data_quality_score)

        overall_quality = round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else 0.0
        elapsed = time.perf_counter() - started

        return ExpansionSummaryReport(
            expansion_days=self.expansion_days,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            symbols=symbol_list,
            timeframes=timeframe_list,
            total_datasets=len(reports),
            successful_datasets=successful,
            failed_datasets=len(reports) - successful,
            overall_quality_score=overall_quality,
            bars_by_symbol=bars_by_symbol,
            bars_by_timeframe=bars_by_timeframe,
            execution_time_seconds=elapsed,
            datasets=[report.as_dict() for report in reports],
        )


def generate_dataset_summary(
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    expansion_days: int = EXPANSION_DAYS,
    auto_download: bool = True,
    report_path: Path | str | None = None,
    end_date: date | None = None,
) -> ExpansionSummaryReport:
    """Run historical expansion validation and export JSON summary."""
    validator = HistoricalExpansionValidator(
        expansion_days=expansion_days,
        auto_download=auto_download,
    )
    report = validator.run(symbols=symbols, timeframes=timeframes, end_date=end_date)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info(
        "Historical expansion validation completed: datasets=%s quality=%s",
        report.total_datasets,
        report.overall_quality_score,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_dataset_summary()
        print("Historical Expansion Validation Summary")
        print(f"Window: {report.start_date} -> {report.end_date} ({report.expansion_days} days)")
        print(f"Datasets: {report.successful_datasets}/{report.total_datasets} valid")
        print(f"Overall Quality Score: {report.overall_quality_score}")
        print("Bars by Symbol:")
        for symbol, bars in report.bars_by_symbol.items():
            print(f"  - {symbol}: {bars}")
        print("Bars by Timeframe:")
        for timeframe, bars in report.bars_by_timeframe.items():
            print(f"  - {timeframe}: {bars}")
        print("Dataset Quality:")
        for dataset in report.datasets:
            print(
                f"  - {dataset['symbol']} {dataset['timeframe']}: bars={dataset['bar_count']} "
                f"score={dataset['data_quality_score']} valid={dataset['validation_valid']}"
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0 if report.failed_datasets == 0 else 1
    except HistoricalExpansionError as exc:
        logger.error("Historical expansion validation error: %s", exc)
        print(f"Historical expansion validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected historical expansion validation failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
