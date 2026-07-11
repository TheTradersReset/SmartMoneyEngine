"""Tests for institutional quality score validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.institutional_edge_extraction_research import EdgeFeatureRecord
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationError,
    InstitutionalQualityValidationResearch,
    ScoredTier2Signal,
    generate_institutional_quality_validation_report,
)


def _record(**overrides: object) -> EdgeFeatureRecord:
    base = {
        "timeframe": "5M",
        "direction": "bearish",
        "bos_timestamp": "2026-01-02T09:15:00+05:30",
        "realized_pnl_points": 100.0,
        "risk_points": 10.0,
        "win": True,
        "displacement_strength": "Strong",
        "fvg_size_points": 20.0,
        "fvg_freshness_bars": 10,
        "fvg_retests": 1,
        "distance_from_liquidity_pool_points": 30.0,
        "distance_from_swing_points": 10.0,
        "choch_to_bos_minutes": 120.0,
        "bos_to_fvg_reclaim_minutes": 0.0,
        "expansion_speed_points_per_minute": 2.0,
        "expansion_size_points": 200.0,
        "trait_tags": (),
    }
    base.update(overrides)
    return EdgeFeatureRecord(**base)  # type: ignore[arg-type]


def test_perfect_score() -> None:
    engine = InstitutionalQualityValidationResearch()
    score, hits, points = engine.compute_quality_score(_record())
    assert score == 100
    assert all(hits.values())
    assert sum(points.values()) == 100


def test_zero_score() -> None:
    engine = InstitutionalQualityValidationResearch()
    score, hits, _ = engine.compute_quality_score(
        _record(
            displacement_strength="Medium",
            fvg_freshness_bars=2,
            fvg_retests=0,
            distance_from_swing_points=50.0,
            choch_to_bos_minutes=30.0,
        )
    )
    assert score == 0
    assert not any(hits.values())


def test_bucket_for_score() -> None:
    assert InstitutionalQualityValidationResearch._bucket_for_score(0) == "0-20"
    assert InstitutionalQualityValidationResearch._bucket_for_score(79) == "60-80"
    assert InstitutionalQualityValidationResearch._bucket_for_score(100) == "80-100"


def test_cohort_metrics() -> None:
    engine = InstitutionalQualityValidationResearch()
    signals = [
        ScoredTier2Signal(
            bos_timestamp="t1",
            timeframe="5M",
            direction="bearish",
            quality_score=80,
            component_hits={},
            component_points={},
            realized_pnl_points=50.0,
            realized_rr=2.0,
            win=True,
        ),
        ScoredTier2Signal(
            bos_timestamp="t2",
            timeframe="5M",
            direction="bearish",
            quality_score=20,
            component_hits={},
            component_points={},
            realized_pnl_points=-10.0,
            realized_rr=-1.0,
            win=False,
        ),
    ]
    metrics = engine._metrics("test", signals)
    assert metrics.signals == 2
    assert metrics.net_points == 40.0


def test_missing_metadata_raises() -> None:
    with pytest.raises(InstitutionalQualityValidationError):
        generate_institutional_quality_validation_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = InstitutionalQualityValidationResearch(timeframes=("5M",)).run(metadata)
    assert report.total_signals >= 0
