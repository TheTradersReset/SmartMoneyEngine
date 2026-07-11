"""Tests for historical expansion validation."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.data.loader.data_loader import DataLoaderError
from src.data.validation.dataset_validator import ValidationResult
from src.research.historical_expansion_validator import (
    EXPANSION_DAYS,
    HistoricalExpansionError,
    HistoricalExpansionValidator,
    generate_dataset_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IST = ZoneInfo("Asia/Kolkata")


def _make_candles(count: int, start: datetime | None = None, step_minutes: int = 5) -> pd.DataFrame:
    base = start or datetime(2026, 1, 2, 9, 15, tzinfo=IST)
    rows = []
    for index in range(count):
        ts = base + timedelta(minutes=step_minutes * index)
        price = 100.0 + index * 0.1
        rows.append(
            {
                "timestamp": ts,
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price + 0.1,
                "volume": 1000 + index,
            }
        )
    return pd.DataFrame(rows)


def test_expansion_days_minimum() -> None:
    with pytest.raises(HistoricalExpansionError):
        HistoricalExpansionValidator(expansion_days=0)


def test_storage_timeframe_mapping() -> None:
    assert HistoricalExpansionValidator._storage_timeframe("5M") == "5"
    assert HistoricalExpansionValidator._storage_timeframe("15M") == "15"
    assert HistoricalExpansionValidator._storage_timeframe("1H") == "60"
    with pytest.raises(HistoricalExpansionError):
        HistoricalExpansionValidator._storage_timeframe("4H")


def test_resolve_range_uses_expansion_days() -> None:
    validator = HistoricalExpansionValidator(expansion_days=90)
    end = date(2026, 6, 30)
    start, resolved_end = validator._resolve_range(end)
    assert resolved_end == end
    assert start == end - timedelta(days=90)


def test_validate_timezone_accepts_ist() -> None:
    frame = _make_candles(3)
    ok, errors = HistoricalExpansionValidator._validate_timezone(frame)
    assert ok is True
    assert errors == []


def test_validate_timezone_rejects_naive() -> None:
    frame = _make_candles(3)
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    ok, errors = HistoricalExpansionValidator._validate_timezone(frame)
    assert ok is False
    assert any("timezone-naive" in error for error in errors)


def test_quality_score_penalizes_issues() -> None:
    validation = ValidationResult(
        rows_checked=100,
        duplicates=5,
        missing_timestamps=10,
        gap_count=2,
        invalid_ohlc=3,
        negative_volume=0,
        zero_volume=0,
        is_valid=False,
    )
    score = HistoricalExpansionValidator._quality_score(
        bar_count=100,
        validation=validation,
        timezone_consistent=True,
        download_missing=5,
        download_duplicates=2,
    )
    assert 0.0 < score < 100.0


@patch.object(HistoricalExpansionValidator, "_download_dataset")
@patch.object(HistoricalExpansionValidator, "_load_dataset")
def test_validate_dataset_success(mock_load: MagicMock, mock_download: MagicMock) -> None:
    frame = _make_candles(20)
    mock_download.return_value = (
        True,
        {
            "download_candles": 20,
            "download_missing_candles": 0,
            "download_duplicate_candles": 0,
            "saved_files": ["data/historical/NSE_NIFTY50-INDEX/5/2026.parquet"],
        },
    )
    mock_load.return_value = frame

    validator = HistoricalExpansionValidator(expansion_days=30, auto_download=True)
    report = validator.validate_dataset("NIFTY50", "5M", date(2026, 1, 1), date(2026, 1, 31))

    assert report.bar_count == 20
    assert report.download_success is True
    assert report.timezone_consistent is True
    assert report.data_quality_score > 0
    mock_download.assert_called_once()


@patch.object(HistoricalExpansionValidator, "_download_dataset")
@patch.object(HistoricalExpansionValidator, "_load_dataset")
def test_validate_dataset_load_failure(mock_load: MagicMock, mock_download: MagicMock) -> None:
    mock_download.return_value = (False, {"error_message": "Token expired."})
    mock_load.side_effect = DataLoaderError("No data found.")

    validator = HistoricalExpansionValidator(expansion_days=30, auto_download=True)
    report = validator.validate_dataset("BANKNIFTY", "15M", date(2026, 1, 1), date(2026, 1, 31))

    assert report.bar_count == 0
    assert report.validation_valid is False
    assert report.data_quality_score == 0.0
    assert report.error_message == "Token expired."


@patch.object(HistoricalExpansionValidator, "_download_dataset")
@patch.object(HistoricalExpansionValidator, "_load_dataset")
def test_run_aggregates_summary(mock_load: MagicMock, mock_download: MagicMock) -> None:
    mock_download.return_value = (
        True,
        {
            "download_candles": 10,
            "download_missing_candles": 0,
            "download_duplicate_candles": 0,
            "saved_files": [],
        },
    )
    mock_load.return_value = _make_candles(10)

    validator = HistoricalExpansionValidator(expansion_days=30, auto_download=True)
    report = validator.run(
        symbols=("NIFTY50", "FINNIFTY"),
        timeframes=("5M", "1H"),
        end_date=date(2026, 6, 30),
    )

    assert report.total_datasets == 4
    assert report.expansion_days == 30
    assert report.bars_by_symbol["NIFTY50"] == 20
    assert report.bars_by_symbol["FINNIFTY"] == 20
    assert report.bars_by_timeframe["5M"] == 20
    assert report.bars_by_timeframe["1H"] == 20
    assert report.successful_datasets == 4
    assert report.failed_datasets == 0
    assert report.overall_quality_score > 0


@patch.object(HistoricalExpansionValidator, "_download_dataset")
@patch.object(HistoricalExpansionValidator, "_load_dataset")
def test_generate_dataset_summary_writes_json(
    mock_load: MagicMock,
    mock_download: MagicMock,
    tmp_path: Path,
) -> None:
    mock_download.return_value = (
        True,
        {
            "download_candles": 5,
            "download_missing_candles": 0,
            "download_duplicate_candles": 0,
            "saved_files": [],
        },
    )
    mock_load.return_value = _make_candles(5)

    report_path = tmp_path / "dataset_summary.json"
    report = generate_dataset_summary(
        symbols=("NIFTY50",),
        timeframes=("5M",),
        expansion_days=7,
        auto_download=True,
        report_path=report_path,
        end_date=date(2026, 6, 30),
    )

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["expansion_days"] == 7
    assert payload["symbols"] == ["NIFTY50"]
    assert payload["timeframes"] == ["5M"]
    assert payload["bars_by_symbol"]["NIFTY50"] == 5
    assert len(payload["datasets"]) == 1
    assert report.total_datasets == 1


def test_default_expansion_days_constant() -> None:
    assert EXPANSION_DAYS == 365


@pytest.mark.integration
@patch.object(HistoricalExpansionValidator, "_download_dataset")
def test_real_nifty50_data_if_present(mock_download: MagicMock) -> None:
    """Validate stored NIFTY50 5M data when available (download skipped)."""
    mock_download.return_value = (
        True,
        {
            "download_candles": 0,
            "download_missing_candles": 0,
            "download_duplicate_candles": 0,
            "saved_files": [],
        },
    )
    validator = HistoricalExpansionValidator(expansion_days=365, auto_download=False)
    end = date.today()
    start = end - timedelta(days=365)

    try:
        report = validator.validate_dataset("NIFTY50", "5M", start, end)
    except DataLoaderError:
        pytest.skip("No stored NIFTY50 5M historical data.")

    if report.bar_count == 0:
        pytest.skip("Stored NIFTY50 5M dataset is empty.")

    assert report.start_timestamp is not None
    assert report.end_timestamp is not None
    assert report.data_quality_score >= 0
