"""Tests for market intelligence validation engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.market_intelligence_validation_engine import (
    HIGH_SCORE_BUCKETS,
    LOW_SCORE_BUCKETS,
    MarketIntelligenceValidationEngine,
    MarketIntelligenceValidationError,
    ValidatedIntelligenceTrade,
    generate_market_intelligence_validation_report,
)


def _trade(
    score: float,
    direction: str = "bullish",
    outcome: str = "Win",
    pnl: float = 10.0,
) -> ValidatedIntelligenceTrade:
    aligned = 100.0 - score if direction == "bearish" else score
    return ValidatedIntelligenceTrade(
        setup_type="Liquidity Grab + FVG Reclaim",
        direction=direction,
        direction_label="BUY" if direction == "bullish" else "SELL",
        timeframe="5M",
        session="Midday",
        trigger_bar=1,
        trigger_timestamp="2026-01-06 12:00:00+05:30",
        outcome=outcome,
        realized_pnl_points=pnl,
        realized_rr=1.0 if pnl > 0 else -1.0,
        intelligence_score=score,
        direction_aligned_score=aligned,
        score_bucket=MarketIntelligenceValidationEngine._score_bucket(score),
        aligned_score_bucket=MarketIntelligenceValidationEngine._score_bucket(aligned),
        has_divergence=False,
        primary_divergence="None",
        trend_state="Bullish",
        momentum_state="Bullish",
    )


def test_score_bucket_boundaries() -> None:
    assert MarketIntelligenceValidationEngine._score_bucket(0.0) == "0-20"
    assert MarketIntelligenceValidationEngine._score_bucket(19.9) == "0-20"
    assert MarketIntelligenceValidationEngine._score_bucket(20.0) == "20-40"
    assert MarketIntelligenceValidationEngine._score_bucket(79.9) == "60-80"
    assert MarketIntelligenceValidationEngine._score_bucket(95.0) == "80-100"


def test_direction_aligned_score_for_sell() -> None:
    aligned = MarketIntelligenceValidationEngine._direction_aligned_score(70.0, "bearish")
    assert aligned == 30.0


def test_metrics_calculation() -> None:
    engine = MarketIntelligenceValidationEngine()
    metrics = engine._metrics([_trade(75.0), _trade(80.0, outcome="Loss", pnl=-5.0)], "test")
    assert metrics.trades == 2
    assert metrics.expectancy == 2.5


def test_low_vs_high_comparison() -> None:
    engine = MarketIntelligenceValidationEngine()
    trades = [
        _trade(70.0, pnl=20.0),
        _trade(85.0, pnl=15.0),
        _trade(15.0, pnl=-10.0),
        _trade(25.0, pnl=-5.0),
    ]
    comparison = engine._low_vs_high(
        trades,
        lambda item: item.aligned_score_bucket,
        LOW_SCORE_BUCKETS,
        HIGH_SCORE_BUCKETS,
        "Aligned",
    )
    assert comparison.high_score["trades"] == 2
    assert comparison.low_score["trades"] == 2
    assert comparison.intelligence_improves_profitability is True


def test_optimal_threshold_respects_min_sample() -> None:
    engine = MarketIntelligenceValidationEngine()
    trades = [
        ValidatedIntelligenceTrade(
            setup_type="Liquidity Grab + FVG Reclaim",
            direction="bullish",
            direction_label="BUY",
            timeframe="5M",
            session="Midday",
            trigger_bar=index,
            trigger_timestamp=f"2026-01-06 12:{index:02d}:00+05:30",
            outcome="Win",
            realized_pnl_points=10.0 - index * 0.1,
            realized_rr=1.0,
            intelligence_score=40.0 + index,
            direction_aligned_score=40.0 + index,
            score_bucket=engine._score_bucket(40.0 + index),
            aligned_score_bucket=engine._score_bucket(40.0 + index),
            has_divergence=False,
            primary_divergence="None",
            trend_state="Bullish",
            momentum_state="Bullish",
        )
        for index in range(40)
    ]
    optimal = engine._find_optimal_threshold(trades, lambda item: item.direction_aligned_score)
    assert optimal["minimum_score_threshold"] is not None
    assert optimal["selected_trades"] >= 30


def test_cross_analysis_min_trades() -> None:
    engine = MarketIntelligenceValidationEngine()
    trades = [_trade(70.0) for _ in range(3)]
    result = engine._cross_analysis(
        trades,
        lambda item: f"{item.aligned_score_bucket} | {item.timeframe}",
    )
    assert result == {}


def test_rank_buckets() -> None:
    engine = MarketIntelligenceValidationEngine()
    buckets = {
        "40-60": {"label": "40-60", "expectancy": 5.0, "trades": 50},
        "60-80": {"label": "60-80", "expectancy": 15.0, "trades": 40},
        "0-20": {"label": "0-20", "expectancy": -5.0, "trades": 30},
    }
    best, worst = engine._rank_buckets(buckets)
    assert best[0]["label"] == "60-80"
    assert worst[0]["label"] == "0-20"


def test_report_structure() -> None:
    from src.research.market_intelligence_validation_engine import (
        MarketIntelligenceValidationReport,
    )

    report = MarketIntelligenceValidationReport(
        symbol="NIFTY50",
        research_window_days=365,
        start_date="2025-07-03",
        end_date="2026-07-03",
        production_stack={},
        total_trades=423,
        by_score_bucket={},
        by_aligned_score_bucket={},
        low_vs_high_comparison={},
        aligned_low_vs_high_comparison={},
        intelligence_improves_profitability=True,
        optimal_threshold={},
        best_score_ranges=[],
        worst_score_ranges=[],
        score_plus_divergence={},
        score_plus_timeframe={},
        score_plus_session={},
        conclusions=[],
        execution_time_seconds=1.0,
    )
    payload = report.as_dict()
    assert payload["total_trades"] == 423
    assert "by_score_bucket" in payload


def test_missing_metadata_raises() -> None:
    with pytest.raises(MarketIntelligenceValidationError):
        generate_market_intelligence_validation_report(
            filter_report_path=Path("missing.json"),
        )


def test_low_and_high_bucket_constants() -> None:
    assert "0-20" in LOW_SCORE_BUCKETS
    assert "80-100" in HIGH_SCORE_BUCKETS


@pytest.mark.integration
def test_full_validation_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parent.parent
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = MarketIntelligenceValidationEngine().run(metadata)
    assert report.total_trades > 0
