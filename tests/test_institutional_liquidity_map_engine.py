"""Tests for Institutional Liquidity Map Engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.context.institutional_liquidity_map_engine import (
    SWEEP_SCORE_WEIGHTS,
    InstitutionalLiquidityMapEngine,
    InstitutionalLiquidityMapError,
    SweepClassification,
    generate_liquidity_map_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _pipeline_frame(length: int = 120) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(length):
        price = 100.0 + index * 0.15
        timestamp = base + pd.Timedelta(minutes=5 * index)
        is_sweep_bar = index == 80
        rows.append(
            {
                "Date": timestamp.isoformat(),
                "Open": price,
                "High": price + (2.5 if is_sweep_bar else 1.0),
                "Low": price - 0.8,
                "Close": price - 0.5 if is_sweep_bar else price + 0.4,
                "Volume": 150000 if is_sweep_bar else 100000,
                "Swing_High": price + 5 if index % 15 == 0 else pd.NA,
                "Swing_Low": price - 5 if index % 17 == 0 else pd.NA,
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": price if index == 81 else pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": pd.NA,
                "Equal_High": price + 4 if index >= 60 else pd.NA,
                "Equal_Low": price - 4 if index >= 60 else pd.NA,
                "Buy_Side_Liquidity": price + 4 if index >= 60 else pd.NA,
                "Sell_Side_Liquidity": price - 4 if index >= 60 else pd.NA,
                "Buy_Liquidity_Sweep": price + 2.5 if is_sweep_bar else pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
                "Liquidity_Strength": 3 if is_sweep_bar else 1,
            }
        )
    return pd.DataFrame(rows)


def test_sweep_score_weights_total_100() -> None:
    assert sum(SWEEP_SCORE_WEIGHTS.values()) == 100


def test_calendar_levels_attached() -> None:
    engine = InstitutionalLiquidityMapEngine()
    frame = _pipeline_frame()
    enriched = engine._attach_calendar_levels(frame)
    assert "_pdh" in enriched.columns
    assert "_pwh" in enriched.columns
    assert "_pmh" in enriched.columns


def test_evaluate_bar_structure() -> None:
    engine = InstitutionalLiquidityMapEngine()
    frame = _pipeline_frame()
    enriched = engine._attach_calendar_levels(frame)
    candle_map = engine.evaluate_bar(frame, enriched, 80)
    payload = candle_map.as_dict()
    assert "external_liquidity" in payload
    assert "internal_liquidity" in payload
    assert "liquidity_event" in payload
    assert "liquidity_objective" in payload
    assert "sweep_quality_score" in payload
    assert "market_narrative" in payload
    assert candle_map.liquidity_event["event_type"] == "Buy Side Sweep"


def test_buy_side_sweep_classification() -> None:
    engine = InstitutionalLiquidityMapEngine()
    frame = _pipeline_frame()
    enriched = engine._attach_calendar_levels(frame)
    candle_map = engine.evaluate_bar(frame, enriched, 80)
    assert candle_map.liquidity_event["classification"] in {
        SweepClassification.WEAK.value,
        SweepClassification.MEDIUM.value,
        SweepClassification.STRONG.value,
        SweepClassification.INSTITUTIONAL.value,
    }
    assert candle_map.sweep_quality_score > 0


def test_full_evaluate_count() -> None:
    engine = InstitutionalLiquidityMapEngine()
    frame = _pipeline_frame(60)
    evaluations = engine.evaluate(frame)
    assert len(evaluations) == 60


def test_missing_pipeline_raises() -> None:
    with pytest.raises(InstitutionalLiquidityMapError):
        generate_liquidity_map_report(pipeline_csv=Path("missing.csv"))


@pytest.mark.integration
def test_full_report_if_pipeline_exists() -> None:
    pipeline = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
    if not pipeline.exists():
        pytest.skip("Pipeline CSV not available.")

    report = generate_liquidity_map_report(
        pipeline_csv=pipeline,
        report_path=PROJECT_ROOT / "outputs" / "context" / "liquidity_map_report_test.json",
    )
    assert report.total_candles > 0
    assert report.sweep_event_distribution

    test_output = PROJECT_ROOT / "outputs" / "context" / "liquidity_map_report_test.json"
    if test_output.exists():
        payload = json.loads(test_output.read_text(encoding="utf-8"))
        assert payload["total_candles"] == report.total_candles
