"""Tests for Tier-2 signal ranking validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)
from src.research.tier2_signal_ranking_validation_research import (
    RankedBosCloseSignal,
    Tier2SignalRankingValidationError,
    Tier2SignalRankingValidationResearch,
    generate_tier2_signal_ranking_validation_report,
)


def test_cohort_metrics() -> None:
    engine = Tier2SignalRankingValidationResearch()
    signals = [
        RankedBosCloseSignal(
            bos_timestamp="2026-01-02T09:15:00+05:30",
            timeframe="5M",
            direction="bearish",
            quality_score=80,
            realized_pnl_points=100.0,
            realized_rr=2.0,
            win=True,
            mfe_points=100.0,
            mae_points=10.0,
        ),
        RankedBosCloseSignal(
            bos_timestamp="2026-01-02T10:15:00+05:30",
            timeframe="5M",
            direction="bearish",
            quality_score=20,
            realized_pnl_points=-50.0,
            realized_rr=-1.0,
            win=False,
            mfe_points=5.0,
            mae_points=50.0,
        ),
    ]
    metrics = engine._metrics("test", signals)
    assert metrics.signals == 2
    assert metrics.net_points == 50.0


def test_bucket_for_score() -> None:
    assert InstitutionalQualityValidationResearch._bucket_for_score(75) == "60-80"
    assert InstitutionalQualityValidationResearch._bucket_for_score(100) == "80-100"


def test_missing_metadata_raises() -> None:
    with pytest.raises(Tier2SignalRankingValidationError):
        generate_tier2_signal_ranking_validation_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = Tier2SignalRankingValidationResearch(timeframes=("5M",)).run(metadata)
    assert report.total_signals >= 0
