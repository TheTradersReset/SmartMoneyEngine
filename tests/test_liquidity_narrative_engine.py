"""Tests for Liquidity Narrative Engine V1."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.context.liquidity_narrative_engine import (
    NARRATIVE_WEIGHTS,
    DisplacementStrength,
    FvgContext,
    LiquidityNarrativeEngine,
    LiquidityNarrativeError,
    MarketIntent,
    generate_liquidity_narrative_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _pipeline_frame(length: int = 80) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(length):
        price = 100.0 + index * 0.1
        timestamp = base + pd.Timedelta(minutes=5 * index)
        rows.append(
            {
                "Date": timestamp.isoformat(),
                "Open": price,
                "High": price + 1.0,
                "Low": price - 0.8,
                "Close": price + 0.4,
                "Volume": 100000,
                "Trend": "BULLISH" if index > 40 else "BEARISH",
                "Trend_Strength": 2,
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bullish_FVG_Bottom": pd.NA,
                "Bearish_FVG_Top": pd.NA,
                "Bearish_FVG_Bottom": pd.NA,
                "Buy_Side_Liquidity": 105.0 if index >= 30 else pd.NA,
                "Sell_Side_Liquidity": 95.0 if index >= 30 else pd.NA,
                "Buy_Liquidity_Sweep": pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
                "Liquidity_Strength": 2,
            }
        )
    return pd.DataFrame(rows)


def test_narrative_weights_total_100() -> None:
    assert sum(NARRATIVE_WEIGHTS.values()) == 100


def test_displacement_strength_strong() -> None:
    row = pd.Series(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 100.0,
            "Close": 100.8,
        }
    )
    assert (
        LiquidityNarrativeEngine._displacement_strength_for_bar(row, "bullish")
        == DisplacementStrength.STRONG
    )


def test_displacement_strength_weak_for_opposing_close() -> None:
    row = pd.Series(
        {
            "Open": 101.0,
            "High": 102.0,
            "Low": 99.0,
            "Close": 99.5,
        }
    )
    assert (
        LiquidityNarrativeEngine._displacement_strength_for_bar(row, "bullish")
        == DisplacementStrength.WEAK
    )


def test_liquidity_event_label() -> None:
    from src.context.liquidity_narrative_engine import LiquidityEventSnapshot

    engine = LiquidityNarrativeEngine()
    snapshot = LiquidityEventSnapshot(
        buy_side_liquidity_taken=True,
        sell_side_liquidity_taken=False,
        latest_buy_sweep_price=105.0,
        latest_sell_sweep_price=None,
        active_buy_side_liquidity=106.0,
        active_sell_side_liquidity=95.0,
    )
    assert engine._liquidity_event_label(snapshot) == "Buy Side Liquidity Taken"


def test_fvg_created_detection() -> None:
    frame = _pipeline_frame(10)
    frame.loc[5, "Bullish_FVG_Top"] = 102.0
    frame.loc[5, "Bullish_FVG_Bottom"] = 101.0
    engine = LiquidityNarrativeEngine()
    context, bias = engine._fvg_context_state(frame, 5, engine._window(frame, 5))
    assert context == FvgContext.CREATED
    assert bias == "bullish"


def test_fvg_reclaimed_detection() -> None:
    frame = _pipeline_frame(10)
    frame.loc[4, "Bullish_FVG_Top"] = 101.0
    frame.loc[4, "Bullish_FVG_Bottom"] = 100.5
    frame.loc[5, "Close"] = 101.5
    engine = LiquidityNarrativeEngine()
    context, bias = engine._fvg_context_state(frame, 5, engine._window(frame, 5))
    assert context == FvgContext.RECLAIMED
    assert bias == "bullish"


def test_market_intent_distribution() -> None:
    frame = _pipeline_frame(60)
    frame.loc[50, "Buy_Liquidity_Sweep"] = 105.0
    frame.loc[50, "Bearish_CHOCH"] = 99.0
    frame.loc[50, "Bearish_BOS"] = 98.5
    frame.loc[50, "Open"] = 100.0
    frame.loc[50, "High"] = 101.0
    frame.loc[50, "Low"] = 98.0
    frame.loc[50, "Close"] = 98.5
    engine = LiquidityNarrativeEngine()
    evaluation = engine.evaluate_bar(frame, 50)
    assert evaluation.market_intent == MarketIntent.DISTRIBUTION.value


def test_narrative_strength_within_bounds() -> None:
    frame = _pipeline_frame(60)
    engine = LiquidityNarrativeEngine()
    evaluation = engine.evaluate_bar(frame, 40)
    assert 0 <= evaluation.narrative_strength_score <= 100


def test_narrative_is_non_empty() -> None:
    frame = _pipeline_frame(60)
    engine = LiquidityNarrativeEngine()
    evaluation = engine.evaluate_bar(frame, 40)
    assert evaluation.narrative
    assert "Market appears" in evaluation.narrative


def test_report_structure() -> None:
    from src.context.liquidity_narrative_engine import LiquidityNarrativeReport

    report = LiquidityNarrativeReport(
        symbol="NIFTY50",
        timeframe="5M",
        source_csv="test.csv",
        total_candles=100,
        average_narrative_strength=55.0,
        score_distribution={},
        intent_distribution={},
        displacement_distribution={},
        fvg_context_distribution={},
        liquidity_event_distribution={},
        top_narrative_examples=[],
        sample_summaries=[],
        execution_time_seconds=1.0,
    )
    payload = report.as_dict()
    assert payload["total_candles"] == 100
    assert "intent_distribution" in payload


def test_missing_pipeline_raises() -> None:
    with pytest.raises(LiquidityNarrativeError):
        generate_liquidity_narrative_report(pipeline_csv=Path("missing.csv"))


@pytest.mark.integration
def test_full_report_if_pipeline_exists() -> None:
    pipeline = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
    if not pipeline.exists():
        pytest.skip("Pipeline CSV not available.")

    frame = pd.read_csv(pipeline, nrows=500)
    engine = LiquidityNarrativeEngine()
    evaluations = engine.evaluate(frame)
    report = engine.build_report(evaluations, pipeline, 1.0)
    assert report.total_candles == len(evaluations)
    assert report.average_narrative_strength >= 0
