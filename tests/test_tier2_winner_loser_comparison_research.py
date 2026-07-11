"""Tests for Tier-2 winner vs loser comparison research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.tier2_winner_loser_comparison_research import (
    COHORT_FRACTION,
    ComparativeTradeRecord,
    FEATURES_COMPARED,
    Tier2WinnerLoserComparisonError,
    Tier2WinnerLoserComparisonResearch,
    generate_tier2_winner_loser_comparison_report,
)


def _sample_record(pnl: float) -> ComparativeTradeRecord:
    draft = ComparativeTradeRecord(
        bos_timestamp="2026-01-02T09:15:00+05:30",
        timeframe="5M",
        direction="bullish",
        realized_pnl_points=pnl,
        realized_rr=2.0 if pnl > 0 else -1.0,
        win=pnl > 0,
        displacement_strength="Strong",
        fvg_size_points=25.0,
        fvg_age_bars=10,
        fvg_retest_count=0,
        choch_to_bos_minutes=120.0,
        distance_from_liquidity_points=30.0,
        distance_from_swing_points=15.0,
        session="Opening",
        intelligence_score=72.0,
        narrative_confidence=68,
        regime="Trend Continuation",
        market_location="Near Support",
        rsi=55.0,
        rsi_band="50-60",
        rsi_divergence="No RSI Divergence",
        trait_tags=(),
    )
    tags = Tier2WinnerLoserComparisonResearch()._build_trait_tags(draft)
    return ComparativeTradeRecord(**{**draft.as_dict(), "trait_tags": tags})


def test_cohort_fraction() -> None:
    assert COHORT_FRACTION == 0.25


def test_features_compared_count() -> None:
    assert len(FEATURES_COMPARED) == 15


def test_trait_comparisons() -> None:
    engine = Tier2WinnerLoserComparisonResearch()
    top = [_sample_record(200.0)] * 5
    bottom = [_sample_record(-100.0)] * 5
    comparisons = engine._trait_comparisons(top, bottom)
    assert comparisons
    assert all(item.winner_frequency_pct >= 0 for item in comparisons)
    assert all("edge_pct" in item.as_dict() for item in comparisons)


def test_trait_category() -> None:
    assert (
        Tier2WinnerLoserComparisonResearch._trait_category("Regime: Trend Continuation")
        == "Regime Classification"
    )


def test_missing_metadata_raises() -> None:
    with pytest.raises(Tier2WinnerLoserComparisonError):
        generate_tier2_winner_loser_comparison_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = Tier2WinnerLoserComparisonResearch(timeframes=("5M",)).run(metadata)
    assert report.total_signals >= 0
    assert report.cohort_fraction == 0.25
    assert len(report.top_20_winning_traits) <= 20
