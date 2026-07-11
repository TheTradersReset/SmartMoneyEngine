"""Tests for liquidity audit diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.research.liquidity_audit_engine import (
    LiquidityAuditEngine,
    LiquidityAuditError,
    MissedSweepReason,
    generate_liquidity_audit_report,
)
from src.smc.liquidity import LiquiditySide

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _build_frame(rows: list[dict[str, float | None]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in (
        "Equal_High",
        "Equal_Low",
        "Buy_Side_Liquidity",
        "Sell_Side_Liquidity",
        "Buy_Liquidity_Sweep",
        "Sell_Liquidity_Sweep",
        "Liquidity_Strength",
        "Bullish_BOS",
        "Bearish_BOS",
        "Bullish_CHOCH",
        "Bearish_CHOCH",
    ):
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["Date"] = [f"2026-01-02 09:{15 + index * 5:02d}:00+05:30" for index in range(len(frame))]
    return frame


def test_classify_wick_only_miss_for_buy_pool() -> None:
    auditor = LiquidityAuditEngine()
    from src.smc.liquidity import LiquidityPoolRecord

    buy_pool = LiquidityPoolRecord(
        side=LiquiditySide.BUY,
        level=100.0,
        strength=1,
        confirmed_index=1,
        confirmed_position=1,
    )
    reason = auditor._classify_pool_interaction(buy_pool, high=101.0, low=99.5, close=100.5)
    assert reason == MissedSweepReason.WICK_ONLY_NO_CLOSE_BACK


def test_classify_strict_buy_sweep_returns_none() -> None:
    auditor = LiquidityAuditEngine()
    from src.smc.liquidity import LiquidityPoolRecord

    buy_pool = LiquidityPoolRecord(
        side=LiquiditySide.BUY,
        level=100.0,
        strength=1,
        confirmed_index=1,
        confirmed_position=1,
    )
    reason = auditor._classify_pool_interaction(buy_pool, high=101.0, low=99.5, close=99.8)
    assert reason is None


def test_analyze_counts_pools_and_sweeps() -> None:
    rows = [
        {
            "Swing_High": None,
            "Swing_Low": None,
            "High": 99.0,
            "Low": 98.0,
            "Close": 98.5,
        },
        {
            "Swing_High": 100.0,
            "Swing_Low": None,
            "High": 100.0,
            "Low": 99.0,
            "Close": 99.5,
        },
        {
            "Swing_High": None,
            "Swing_Low": None,
            "High": 100.05,
            "Low": 99.0,
            "Close": 99.8,
        },
        {
            "Swing_High": 100.05,
            "Swing_Low": None,
            "High": 100.1,
            "Low": 99.5,
            "Close": 100.0,
        },
        {
            "Swing_High": None,
            "Swing_Low": None,
            "High": 101.0,
            "Low": 99.5,
            "Close": 99.8,
        },
    ]
    frame = _build_frame(rows)
    auditor = LiquidityAuditEngine()
    report = auditor.analyze(frame)

    assert report.total_candles == 5
    assert report.pool_counts["total_liquidity_pools"] >= 1
    assert report.sweep_metrics["total_sweeps"] >= 1
    assert report.frequency_comparison["liquidity_sweeps"] >= 1
    assert report.restrictive_conditions
    assert report.clusters


def test_analyze_rejects_missing_columns() -> None:
    auditor = LiquidityAuditEngine()
    frame = pd.DataFrame({"High": [1.0], "Low": [0.5], "Close": [0.8]})
    with pytest.raises(LiquidityAuditError):
        auditor.analyze(frame)


@patch.object(LiquidityAuditEngine, "_run_detector")
def test_generate_liquidity_audit_report_writes_json(
    mock_run_detector,
    tmp_path: Path,
) -> None:
    from src.models.market_data import MarketData
    from src.smc.liquidity import LiquidityPoolRecord

    frame = _build_frame(
        [
            {
                "Swing_High": 100.0,
                "Swing_Low": None,
                "High": 100.0,
                "Low": 99.0,
                "Close": 99.5,
            }
        ]
    )
    market = MarketData(frame[["High", "Low", "Close"]].copy())
    market.add_column("Buy_Liquidity_Sweep", pd.Series([pd.NA], dtype="Float64"))
    market.add_column("Sell_Liquidity_Sweep", pd.Series([pd.NA], dtype="Float64"))
    mock_run_detector.return_value = (
        market,
        (
            LiquidityPoolRecord(
                side=LiquiditySide.BUY,
                level=100.0,
                strength=1,
                confirmed_index=0,
                confirmed_position=0,
                swept=False,
            ),
        ),
    )

    csv_path = tmp_path / "pipeline.csv"
    report_path = tmp_path / "liquidity_audit_report.json"
    frame.to_csv(csv_path, index=False)

    report = generate_liquidity_audit_report(
        pipeline_csv=csv_path,
        report_path=report_path,
        symbol="NIFTY50",
        timeframe="5",
    )

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "NIFTY50"
    assert "pool_counts" in payload
    assert report.total_candles == 1


@pytest.mark.integration
def test_real_pipeline_liquidity_audit_if_present() -> None:
    pipeline_csv = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
    if not pipeline_csv.exists():
        pytest.skip("Real pipeline CSV not available.")

    auditor = LiquidityAuditEngine(symbol="NIFTY50", timeframe="5")
    report = auditor.run_from_csv(pipeline_csv)

    assert report.total_candles > 1000
    assert report.pool_counts["total_liquidity_pools"] > 0
    assert report.sweep_metrics["total_sweeps"] > 0
