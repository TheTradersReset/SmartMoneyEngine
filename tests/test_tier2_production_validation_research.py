"""Tests for Tier-2 production validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.tier2_production_validation_research import (
    Tier2ProductionSignal,
    Tier2ProductionValidationError,
    Tier2ProductionValidationResearch,
    generate_tier2_production_validation_report,
)


def _signal(
    *,
    win: bool = True,
    pnl: float = 50.0,
    htf: bool = True,
    mi: float = 70.0,
    timestamp: str = "2026-01-02T09:15:00+05:30",
) -> Tier2ProductionSignal:
    return Tier2ProductionSignal(
        timeframe="5M",
        direction="bearish",
        bos_bar=50,
        bos_timestamp=timestamp,
        risk_points=10.0,
        realized_pnl_points=pnl,
        realized_rr=pnl / 10.0,
        win=win,
        htf_aligned=htf,
        intelligence_score=mi,
    )


def test_htf_alignment_bullish() -> None:
    engine = Tier2ProductionValidationResearch()
    assert engine._htf_aligned("bullish", "BULLISH", "SIDEWAYS") is True
    assert engine._htf_aligned("bullish", "BULLISH", "BEARISH") is False


def test_variant_filters() -> None:
    engine = Tier2ProductionValidationResearch()
    raw = _signal()
    low_mi = _signal(mi=60.0)
    no_htf = _signal(htf=False)

    assert engine._variant_filter("raw_tier_2")(raw) is True
    assert engine._variant_filter("tier_2_mi_65")(low_mi) is False
    assert engine._variant_filter("tier_2_htf_alignment")(no_htf) is False
    assert engine._variant_filter("tier_2_htf_mi_65")(raw) is True


def test_maximum_drawdown() -> None:
    dd = Tier2ProductionValidationResearch._maximum_drawdown([50.0, -10.0, -20.0, 30.0])
    assert dd == 30.0


def test_variant_metrics() -> None:
    engine = Tier2ProductionValidationResearch()
    metrics = engine._variant_metrics(
        "raw_tier_2",
        [_signal(), _signal(win=False, pnl=-10.0, timestamp="2026-01-02T10:15:00+05:30")],
        research_months=12.0,
    )
    assert metrics.signals == 2
    assert metrics.net_points == 40.0
    assert metrics.win_rate_pct == 50.0


def test_balance_scores() -> None:
    engine = Tier2ProductionValidationResearch()
    from src.research.tier2_production_validation_research import VariantMetrics

    metrics_map = {
        "raw_tier_2": VariantMetrics(
            variant_key="raw_tier_2",
            label="Raw",
            filters=[],
            signals=100,
            signals_per_month=10.0,
            win_rate_pct=40.0,
            profit_factor=2.0,
            expectancy=50.0,
            average_rr=0.8,
            maximum_drawdown_points=500.0,
            net_points=5000.0,
            streak_analysis={},
        ),
        "tier_2_htf_mi_65": VariantMetrics(
            variant_key="tier_2_htf_mi_65",
            label="Filtered",
            filters=["HTF", "MI"],
            signals=30,
            signals_per_month=3.0,
            win_rate_pct=55.0,
            profit_factor=2.5,
            expectancy=80.0,
            average_rr=1.2,
            maximum_drawdown_points=200.0,
            net_points=2400.0,
            streak_analysis={},
        ),
    }
    engine._apply_balance_scores(metrics_map)
    assert metrics_map["tier_2_htf_mi_65"].balance_score > 0


def test_missing_metadata_raises() -> None:
    with pytest.raises(Tier2ProductionValidationError):
        generate_tier2_production_validation_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = Tier2ProductionValidationResearch(timeframes=("5M",)).run(metadata)
    assert report.recommended_production_version in report.variants
