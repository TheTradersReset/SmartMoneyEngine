"""Tests for the SmartMoneyEngine multi-timeframe engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.signals.multi_timeframe_engine import (
    MultiTimeframeEngine,
    MultiTimeframeEngineError,
    OverallBias,
    generate_multi_timeframe_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
MTF_REPORT = PROJECT_ROOT / "outputs" / "signals" / "multi_timeframe_report.json"


pytestmark = pytest.mark.skipif(
    not PIPELINE_CSV.exists(),
    reason="Real FYERS pipeline CSV is required for multi-timeframe tests.",
)


@pytest.fixture(scope="module")
def mtf_report() -> dict:
    report = generate_multi_timeframe_report(report_path=MTF_REPORT)
    with MTF_REPORT.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["overall_bias"] == report.overall_bias
    return payload


def test_report_contains_all_timeframes(mtf_report: dict) -> None:
    labels = {item["timeframe"] for item in mtf_report["timeframes"]}
    assert labels == {"1D", "4H", "1H", "15M", "5M"}


def test_alignment_score_range(mtf_report: dict) -> None:
    assert 0 <= mtf_report["alignment_score"] <= 100


def test_overall_bias_valid(mtf_report: dict) -> None:
    valid = {item.value for item in OverallBias}
    assert mtf_report["overall_bias"] in valid


def test_per_timeframe_fields(mtf_report: dict) -> None:
    required = {
        "timeframe",
        "trend",
        "structure",
        "bos_status",
        "choch_status",
        "liquidity_status",
        "institutional_bias",
        "bars_analyzed",
    }
    for item in mtf_report["timeframes"]:
        assert required.issubset(item.keys())
        assert item["trend"] in {"Bullish", "Bearish", "Neutral"}
        assert item["structure"] in {"HH-HL", "LH-LL", "Range"}


def test_timeframe_counts_sum(mtf_report: dict) -> None:
    total = (
        mtf_report["bullish_timeframes"]
        + mtf_report["bearish_timeframes"]
        + mtf_report["neutral_timeframes"]
    )
    assert total == 5


def test_alignment_matches_bullish_count(mtf_report: dict) -> None:
    if mtf_report["bullish_timeframes"] > mtf_report["bearish_timeframes"]:
        assert mtf_report["alignment_score"] == mtf_report["bullish_timeframes"] * 20
    elif mtf_report["bearish_timeframes"] > mtf_report["bullish_timeframes"]:
        assert mtf_report["alignment_score"] == mtf_report["bearish_timeframes"] * 20


def test_insufficient_data_raises_error() -> None:
    engine = MultiTimeframeEngine()
    import pandas as pd

    tiny = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="5min", tz="Asia/Kolkata"),
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [100, 100, 100, 100, 100],
        }
    )
    with pytest.raises(MultiTimeframeEngineError):
        engine._analyze_timeframe("5M", tiny)
