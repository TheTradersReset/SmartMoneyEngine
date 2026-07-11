"""Tests for market location validation engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.market_location_validation_engine import (
    MIN_INTELLIGENCE_SCORE,
    LocationTradeRecord,
    MarketLocationValidationEngine,
    MarketLocationValidationError,
    generate_market_location_validation_report,
)
from src.signals.setup_classifier import SetupClassification


def _setup(direction: str = "bullish") -> SetupClassification:
    return SetupClassification(
        setup_type="Liquidity Grab + FVG Reclaim",
        direction=direction,
        confidence=0.9,
        quality_score=90,
        trigger_bar=10,
        trigger_timestamp="2026-01-06 12:00:00+05:30",
        entry=100.0,
        stop_loss=95.0,
        target_1=110.0,
        target_2=115.0,
        reason="test",
    )


def test_distance_bucket_boundaries() -> None:
    assert MarketLocationValidationEngine._distance_bucket(0.1) == "Very Close"
    assert MarketLocationValidationEngine._distance_bucket(0.25) == "Very Close"
    assert MarketLocationValidationEngine._distance_bucket(0.4) == "Close"
    assert MarketLocationValidationEngine._distance_bucket(0.8) == "Medium"
    assert MarketLocationValidationEngine._distance_bucket(2.0) == "Far"


def test_room_to_target_bullish() -> None:
    setup = _setup("bullish")
    levels = {"major_support": 98.0, "major_resistance": 105.0}
    room = MarketLocationValidationEngine._room_to_target(setup, levels)
    assert room == 5.0


def test_room_to_target_bearish() -> None:
    setup = SetupClassification(
        setup_type="Liquidity Grab + FVG Reclaim",
        direction="bearish",
        confidence=0.9,
        quality_score=90,
        trigger_bar=10,
        trigger_timestamp="2026-01-06 12:00:00+05:30",
        entry=100.0,
        stop_loss=105.0,
        target_1=90.0,
        target_2=85.0,
        reason="test",
    )
    levels = {"major_support": 95.0, "major_resistance": 102.0}
    room = MarketLocationValidationEngine._room_to_target(setup, levels)
    assert room == 5.0


def test_reward_risk_potential() -> None:
    assert MarketLocationValidationEngine._reward_risk_potential(_setup()) == 2.0


def test_market_location_label() -> None:
    assert (
        MarketLocationValidationEngine._market_location_label(0.3, 2.0)
        == "Near Support"
    )
    assert (
        MarketLocationValidationEngine._market_location_label(2.0, 0.3)
        == "Near Resistance"
    )


def test_metrics_calculation() -> None:
    engine = MarketLocationValidationEngine()
    trades = [
        LocationTradeRecord(
            setup_type="Liquidity Grab + FVG Reclaim",
            direction="bullish",
            direction_label="BUY",
            timeframe="5M",
            session="Midday",
            trigger_timestamp="2026-01-06 12:00:00+05:30",
            outcome="Win",
            realized_pnl_points=10.0,
            realized_rr=1.0,
            intelligence_score=70.0,
            entry=100.0,
            stop_loss=95.0,
            target_1=110.0,
            distance_to_support=1.0,
            distance_to_resistance=5.0,
            support_distance_atr=0.2,
            resistance_distance_atr=1.0,
            support_distance_bucket="Very Close",
            resistance_distance_bucket="Medium",
            room_to_target=5.0,
            room_to_target_atr=1.0,
            room_to_target_bucket="Medium",
            reward_risk_potential=2.0,
            reward_risk_bucket="2R+",
            market_location="Near Support",
        )
    ]
    metrics = engine._metrics(trades, "test")
    assert metrics.trades == 1
    assert metrics.expectancy == 10.0


def test_optimal_conditions() -> None:
    engine = MarketLocationValidationEngine()
    trades = [
        LocationTradeRecord(
            setup_type="Liquidity Grab + FVG Reclaim",
            direction="bullish",
            direction_label="BUY",
            timeframe="5M",
            session="Midday",
            trigger_timestamp="2026-01-06 12:00:00+05:30",
            outcome="Win",
            realized_pnl_points=10.0,
            realized_rr=1.0,
            intelligence_score=70.0,
            entry=100.0,
            stop_loss=95.0,
            target_1=110.0,
            distance_to_support=1.0,
            distance_to_resistance=5.0,
            support_distance_atr=0.2,
            resistance_distance_atr=1.0,
            support_distance_bucket="Very Close",
            resistance_distance_bucket="Medium",
            room_to_target=5.0,
            room_to_target_atr=1.0,
            room_to_target_bucket="Medium",
            reward_risk_potential=2.0,
            reward_risk_bucket="2R+",
            market_location="Near Support",
        )
        for _ in range(5)
    ]
    optimal = engine._optimal_conditions(trades)
    assert "support_distance" in optimal


def test_report_structure() -> None:
    from src.research.market_location_validation_engine import MarketLocationValidationReport

    report = MarketLocationValidationReport(
        symbol="NIFTY50",
        research_window_days=365,
        start_date="2025-07-03",
        end_date="2026-07-03",
        production_stack={},
        total_trades=100,
        by_support_distance={},
        by_resistance_distance={},
        by_room_to_target={},
        by_reward_risk_potential={},
        by_market_location={},
        optimal_location_conditions={},
        location_avoidance_guidance={},
        best_location_segments=[],
        worst_location_segments=[],
        conclusions=[],
        execution_time_seconds=1.0,
    )
    payload = report.as_dict()
    assert payload["total_trades"] == 100
    assert "by_support_distance" in payload


def test_missing_metadata_raises() -> None:
    with pytest.raises(MarketLocationValidationError):
        generate_market_location_validation_report(
            filter_report_path=Path("missing.json"),
        )


def test_min_intelligence_score_constant() -> None:
    assert MIN_INTELLIGENCE_SCORE == 65


@pytest.mark.integration
def test_full_validation_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parent.parent
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = MarketLocationValidationEngine().run(metadata)
    assert report.total_trades >= 0
