"""Tests for Market Intelligence Layer V1."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.context.market_intelligence_engine import (
    INTELLIGENCE_WEIGHTS,
    EmaStructure,
    MarketIntelligenceEngine,
    MarketIntelligenceError,
    MarketLocation,
    MomentumState,
    RsiState,
    SessionState,
    TrendState,
    VolatilityState,
    generate_market_intelligence_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _pipeline_frame(length: int = 250) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(length):
        price = 100.0 + index * 0.15
        timestamp = base + pd.Timedelta(minutes=5 * index)
        rows.append(
            {
                "Date": timestamp.isoformat(),
                "Open": price,
                "High": price + 0.8,
                "Low": price - 0.5,
                "Close": price + 0.2,
                "Volume": 100000 + index * 1000,
                "Swing_High": price + 2.0 if index % 25 == 0 else pd.NA,
                "Swing_Low": price - 2.0 if index % 25 == 12 else pd.NA,
                "Equal_High": pd.NA,
                "Equal_Low": pd.NA,
                "Trend": "BULLISH" if index > 100 else "BEARISH",
                "Trend_Strength": 2,
                "Bullish_OB_Low": pd.NA,
                "Bearish_OB_High": pd.NA,
                "Buy_Side_Liquidity": pd.NA,
                "Sell_Side_Liquidity": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def test_intelligence_weights_total_100() -> None:
    assert sum(INTELLIGENCE_WEIGHTS.values()) == 100


def test_session_state_midday() -> None:
    ts = pd.Timestamp("2026-01-02 12:00:00+05:30")
    assert MarketIntelligenceEngine._session_state(ts) == SessionState.MIDDAY


def test_ema_structure_bull_stack() -> None:
    assert (
        MarketIntelligenceEngine._ema_structure(110.0, 105.0, 100.0)
        == EmaStructure.BULL_STACK
    )


def test_rsi_state_buckets() -> None:
    assert MarketIntelligenceEngine._rsi_state(25.0) == RsiState.OVERSOLD
    assert MarketIntelligenceEngine._rsi_state(55.0) == RsiState.NEUTRAL
    assert MarketIntelligenceEngine._rsi_state(72.0) == RsiState.OVERBOUGHT


def test_momentum_state_strong_bullish() -> None:
    engine = MarketIntelligenceEngine()
    state = engine._momentum_state(rsi=72.0, ema20_slope=1.5, displacement=0.5)
    assert state == MomentumState.STRONG_BULLISH


def test_trend_state_bullish() -> None:
    engine = MarketIntelligenceEngine()
    state = engine._trend_state(
        close=110.0,
        ema20=105.0,
        ema50=100.0,
        ema200=95.0,
        vwap=104.0,
        pipeline_trend=TrendState.BULLISH.value,
        ema_structure=EmaStructure.BULL_STACK,
    )
    assert state == TrendState.BULLISH


def test_market_location_near_support() -> None:
    engine = MarketIntelligenceEngine()
    location = engine._market_location(
        {"close": 100.0, "major_support": 99.8, "major_resistance": 110.0},
        atr=1.0,
    )
    assert location == MarketLocation.NEAR_SUPPORT


def test_intelligence_score_within_bounds() -> None:
    frame = _pipeline_frame(250)
    engine = MarketIntelligenceEngine()
    evaluation = engine.evaluate_bar(engine.enrich(frame), 200)
    assert 0 <= evaluation.intelligence_score <= 100


def test_summary_is_non_empty() -> None:
    frame = _pipeline_frame(250)
    engine = MarketIntelligenceEngine()
    evaluation = engine.evaluate_bar(engine.enrich(frame), 200)
    assert evaluation.summary
    assert len(evaluation.summary.split(".")) >= 3


def test_build_report_structure(tmp_path: Path) -> None:
    frame = _pipeline_frame(120)
    engine = MarketIntelligenceEngine()
    evaluations = engine.evaluate(frame)
    report = engine.build_report(evaluations, tmp_path / "sample.csv", 1.0)
    payload = report.as_dict()
    assert payload["total_candles"] == 120
    assert "score_distribution" in payload
    assert "state_distribution" in payload
    assert len(payload["top_bullish_examples"]) == 10
    assert len(payload["sample_summaries"]) > 0


def test_generate_report_writes_json(tmp_path: Path) -> None:
    frame = _pipeline_frame(120)
    pipeline_csv = tmp_path / "NIFTY50_5m_pipeline.csv"
    report_path = tmp_path / "market_intelligence_report.json"
    frame.to_csv(pipeline_csv, index=False)

    report = generate_market_intelligence_report(
        pipeline_csv=pipeline_csv,
        report_path=report_path,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["total_candles"] == 120
    assert report.average_intelligence_score >= 0


def test_missing_pipeline_raises() -> None:
    with pytest.raises(MarketIntelligenceError):
        generate_market_intelligence_report(
            pipeline_csv=Path("missing_pipeline.csv"),
            report_path=Path("missing_report.json"),
        )


@pytest.mark.integration
def test_real_nifty50_pipeline_if_available() -> None:
    pipeline_csv = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
    if not pipeline_csv.exists():
        pytest.skip("NIFTY50 5M pipeline not available.")

    engine = MarketIntelligenceEngine()
    frame = pd.read_csv(pipeline_csv)
    evaluations = engine.evaluate(frame)
    assert len(evaluations) == len(frame)
