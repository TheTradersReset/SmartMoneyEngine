"""Tests for Tier-2 composite edge validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.tier2_composite_edge_validation_research import (
    MIN_PRODUCTION_SIGNALS,
    WINNING_TRAITS,
    Tier2CompositeEdgeValidationError,
    Tier2CompositeEdgeValidationResearch,
    generate_tier2_composite_edge_validation_report,
)
from src.research.tier2_winner_loser_comparison_research import ComparativeTradeRecord


def _sample_record(**overrides: object) -> ComparativeTradeRecord:
    base = ComparativeTradeRecord(
        bos_timestamp="2026-01-02T09:15:00+05:30",
        timeframe="5M",
        direction="bullish",
        realized_pnl_points=100.0,
        realized_rr=2.0,
        win=True,
        displacement_strength="Strong",
        fvg_size_points=25.0,
        fvg_age_bars=10,
        fvg_retest_count=0,
        choch_to_bos_minutes=120.0,
        distance_from_liquidity_points=30.0,
        distance_from_swing_points=15.0,
        session="Midday",
        intelligence_score=40.0,
        narrative_confidence=70,
        regime="Trend Continuation",
        market_location="Near Support",
        rsi=35.0,
        rsi_band="Below 40",
        rsi_divergence="No RSI Divergence",
        trait_tags=(),
    )
    return ComparativeTradeRecord(**{**base.as_dict(), **overrides})


def test_winning_trait_count() -> None:
    assert len(WINNING_TRAITS) == 5


def test_trait_checks_all_true() -> None:
    engine = Tier2CompositeEdgeValidationResearch()
    checks = engine._trait_checks(_sample_record())
    assert all(checks.values())


def test_combination_filter() -> None:
    engine = Tier2CompositeEdgeValidationResearch()
    records = [
        _sample_record(),
        _sample_record(
            bos_timestamp="2026-01-02T10:15:00+05:30",
            session="Opening",
            rsi=55.0,
            market_location="Near Resistance",
            displacement_strength="Medium",
            choch_to_bos_minutes=45.0,
            realized_pnl_points=-50.0,
            realized_rr=-1.0,
            win=False,
        ),
    ]
    filtered = engine._filter_records(records, ("rsi_below_40", "midday_session"))
    assert len(filtered) == 1


def test_metrics_and_minimum_signals() -> None:
    engine = Tier2CompositeEdgeValidationResearch()
    records = [_sample_record(realized_pnl_points=100.0)] * MIN_PRODUCTION_SIGNALS
    metrics = engine._metrics(("midday_session",), records)
    assert metrics.signals == MIN_PRODUCTION_SIGNALS
    assert metrics.meets_minimum_signals is True
    assert metrics.net_points == 100.0 * MIN_PRODUCTION_SIGNALS


def test_evaluate_all_combination_count() -> None:
    engine = Tier2CompositeEdgeValidationResearch()
    records = [_sample_record()] * 25
    evaluated = engine._evaluate_all_combinations(records)
    assert len(evaluated) == 31


def test_missing_metadata_raises() -> None:
    with pytest.raises(Tier2CompositeEdgeValidationError):
        generate_tier2_composite_edge_validation_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = Tier2CompositeEdgeValidationResearch(timeframes=("5M",)).run(metadata)
    assert report.total_signals >= 0
    assert len(report.all_combinations) == 31
