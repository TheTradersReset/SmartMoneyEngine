"""Tests for major level strength research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.major_level_strength_research import (
    LevelStrengthFeatures,
    MajorLevelStrengthError,
    MajorLevelStrengthResearch,
    StrengthCategory,
    generate_major_level_strength_report,
)

def test_strength_classification() -> None:
    engine = MajorLevelStrengthResearch()
    assert engine._classify_strength(25) == StrengthCategory.WEAK.value
    assert engine._classify_strength(35) == StrengthCategory.MODERATE.value
    assert engine._classify_strength(55) == StrengthCategory.STRONG.value
    assert engine._classify_strength(75) == StrengthCategory.INSTITUTIONAL.value


def test_round_number_overlap() -> None:
    engine = MajorLevelStrengthResearch()
    assert engine._round_number_overlap(24000.0)
    assert engine._round_number_overlap(24015.0)
    assert not engine._round_number_overlap(24040.0)


def test_strength_score_range() -> None:
    engine = MajorLevelStrengthResearch()
    features = LevelStrengthFeatures(
        number_of_touches=4,
        days_level_survived=5,
        bars_near_level=10,
        bounce_count=2,
        rejection_count=1,
        liquidity_grabs=2,
        equal_highs_lows_nearby=1,
        previous_day_overlap=True,
        weekly_overlap=True,
        monthly_overlap=False,
        demand_supply_zone_overlap=True,
        round_number_overlap=True,
        gap_interactions=1,
        average_volume_expansion=1.5,
        source_column="Swing_Low",
    )
    score = engine._compute_strength_score(features)
    assert 0 <= score <= 100


def test_build_matrix() -> None:
    engine = MajorLevelStrengthResearch()
    from src.research.major_level_strength_research import LevelStrengthEvent

    events = [
        LevelStrengthEvent(
            timeframe="5M",
            level_price=100.0,
            level_side="support",
            level_source="Swing_Low",
            formation_bar=10,
            event_bar=30,
            formation_timestamp="t0",
            event_timestamp="t1",
            outcome="support_bounce",
            level_strength_score=75.0,
            strength_category=StrengthCategory.STRONG.value,
            features={},
            distance_traveled_after_reaction=50.0,
        ),
        LevelStrengthEvent(
            timeframe="5M",
            level_price=100.0,
            level_side="support",
            level_source="Swing_Low",
            formation_bar=10,
            event_bar=40,
            formation_timestamp="t0",
            event_timestamp="t2",
            outcome="support_break",
            level_strength_score=30.0,
            strength_category=StrengthCategory.WEAK.value,
            features={},
            distance_traveled_after_reaction=80.0,
        ),
    ]
    matrix = engine._build_matrix(events)
    assert matrix[StrengthCategory.STRONG.value]["bounce_probability_pct"] == 100.0
    assert matrix[StrengthCategory.WEAK.value]["breakdown_probability_pct"] == 100.0


def test_missing_metadata_raises() -> None:
    with pytest.raises(MajorLevelStrengthError):
        generate_major_level_strength_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = MajorLevelStrengthResearch(timeframes=("5M",)).run(metadata)
    assert report.total_reaction_events >= 0
    assert report.level_strength_matrix
