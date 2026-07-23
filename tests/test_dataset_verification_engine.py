"""Unit tests for Dataset Verification Engine."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.dataset_verification.health import score_health
from src.dataset_verification.validators import validate_dataset
from src.dataset_verification.engine import DatasetVerificationEngine

IST = ZoneInfo("Asia/Kolkata")


def _bar(ts: datetime, open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0) -> dict:
    return {
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _session_day(day: datetime) -> pd.DataFrame:
    rows = []
    # 75 bars: 09:15 .. 15:25
    for i in range(75):
        minute = 15 + i * 5
        hour = 9 + minute // 60
        minute = minute % 60
        ts = datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST)
        rows.append(_bar(ts, close=100.0 + i * 0.01))
    return pd.DataFrame(rows)


def test_clean_session_day_scores_ready() -> None:
    frame = _session_day(datetime(2026, 3, 10))
    report = validate_dataset(frame, symbol="NSE:NIFTY50-INDEX", resolution="5")
    health = score_health(report)
    assert report["checks"]["duplicate_candles"]["status"] == "PASS"
    assert report["checks"]["ohlc_consistency"]["status"] == "PASS"
    assert report["checks"]["weekend_detection"]["status"] == "PASS"
    assert health["health_score"] >= 95
    assert report["integrity"]["dataset_hash"]
    assert report["integrity"]["dataset_fingerprint"]


def test_ohlc_and_negative_fail() -> None:
    ts = datetime(2026, 3, 10, 9, 15, tzinfo=IST)
    frame = pd.DataFrame(
        [
            _bar(ts, open_=100, high=90, low=95, close=96),  # high < open/low issues
            _bar(datetime(2026, 3, 10, 9, 20, tzinfo=IST), open_=-1, high=1, low=-2, close=0),
            _bar(datetime(2026, 3, 10, 9, 25, tzinfo=IST), volume=-5),
        ],
    )
    report = validate_dataset(frame)
    assert report["checks"]["ohlc_consistency"]["status"] == "FAIL"
    assert report["checks"]["negative_prices"]["status"] == "FAIL"
    assert report["checks"]["negative_volume"]["status"] == "FAIL"
    health = score_health(report)
    assert health["health_score"] < 90
    assert health["band"] == "BLOCK"


def test_weekend_detection() -> None:
    # 2026-03-14 is Saturday
    frame = pd.DataFrame([_bar(datetime(2026, 3, 14, 9, 15, tzinfo=IST))])
    report = validate_dataset(frame)
    assert report["checks"]["weekend_detection"]["status"] == "FAIL"


def test_duplicate_and_gap() -> None:
    a = datetime(2026, 3, 10, 9, 15, tzinfo=IST)
    b = datetime(2026, 3, 10, 9, 25, tzinfo=IST)  # skip 09:20
    frame = pd.DataFrame([_bar(a), _bar(a), _bar(b)])
    report = validate_dataset(frame)
    assert report["checks"]["duplicate_candles"]["status"] == "FAIL"
    assert report["checks"]["unexpected_time_gaps"]["status"] == "FAIL"


def test_engine_exports(tmp_path: Path) -> None:
    frame = _session_day(datetime(2026, 3, 10))
    csv_path = tmp_path / "bars.csv"
    export = frame.copy()
    export["Date"] = export["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    export = export.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    export[["Date", "Open", "High", "Low", "Close", "Volume"]].to_csv(csv_path, index=False)

    out = tmp_path / "out"
    artifacts = DatasetVerificationEngine(output_dir=out).verify_csv(csv_path)
    assert artifacts.report_json.exists()
    assert artifacts.report_csv.exists()
    assert artifacts.report_html.exists()
    assert "health_score" in artifacts.report["health"]
