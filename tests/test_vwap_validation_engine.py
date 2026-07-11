"""Tests for VWAP validation engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.vwap_validation_engine import (
    MIN_INTELLIGENCE_SCORE,
    VwapTradeRecord,
    VwapValidationEngine,
    VwapValidationError,
    generate_vwap_validation_report,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-06 09:15", periods=5, freq="5min", tz="Asia/Kolkata"),
            "Open": [100.0, 100.5, 101.0, 100.8, 101.2],
            "High": [101.0, 101.5, 101.5, 101.0, 101.8],
            "Low": [99.5, 100.0, 100.5, 100.2, 100.8],
            "Close": [100.5, 101.0, 101.2, 100.9, 101.5],
            "Volume": [1000, 1200, 1100, 900, 1300],
        }
    )


def _trade(
    vwap_position: str = "Above VWAP",
    aligned: bool = True,
    pnl: float = 10.0,
) -> VwapTradeRecord:
    return VwapTradeRecord(
        setup_type="Liquidity Grab + FVG Reclaim",
        direction="bullish",
        direction_label="BUY",
        timeframe="5M",
        session="Midday",
        trigger_timestamp="2026-01-06 12:00:00+05:30",
        outcome="Win" if pnl > 0 else "Loss",
        realized_pnl_points=pnl,
        realized_rr=1.0 if pnl > 0 else -1.0,
        intelligence_score=70.0,
        vwap_position=vwap_position,
        vwap_reclaim="No Reclaim",
        vwap_rejection="No Rejection",
        distance_from_vwap=0.5,
        distance_from_vwap_atr=0.2,
        distance_from_vwap_bucket="Very Close",
        direction_aligned_vwap=aligned,
        market_location="Mid Range",
    )


def test_vwap_position() -> None:
    assert VwapValidationEngine._vwap_position(101.0, 100.0) == "Above VWAP"
    assert VwapValidationEngine._vwap_position(99.0, 100.0) == "Below VWAP"


def test_vwap_reclaim_bullish() -> None:
    frame = _frame()
    enriched = VwapValidationEngine().filter_engine.context_builder.enrich(frame)
    enriched.loc[2, "Close"] = enriched.loc[2, "_vwap"] + 1.0
    enriched.loc[1, "Close"] = enriched.loc[1, "_vwap"] - 1.0
    assert VwapValidationEngine._vwap_reclaim(enriched, 2, "bullish") == "Bullish Reclaim"


def test_vwap_rejection_from_below() -> None:
    frame = _frame()
    enriched = VwapValidationEngine().filter_engine.context_builder.enrich(frame)
    vwap = float(enriched.loc[3, "_vwap"])
    enriched.loc[2, "Close"] = vwap - 2.0
    enriched.loc[3, "Low"] = vwap - 0.5
    enriched.loc[3, "Close"] = vwap - 1.0
    assert VwapValidationEngine._vwap_rejection(enriched, 3) == "Rejected from Below"


def test_distance_bucket() -> None:
    assert VwapValidationEngine._distance_bucket(0.1) == "Very Close"
    assert VwapValidationEngine._distance_bucket(0.4) == "Close"
    assert VwapValidationEngine._distance_bucket(2.0) == "Far"


def test_direction_aligned_vwap() -> None:
    assert VwapValidationEngine._direction_aligned_vwap("bullish", "Above VWAP") is True
    assert VwapValidationEngine._direction_aligned_vwap("bearish", "Above VWAP") is False


def test_metrics_calculation() -> None:
    engine = VwapValidationEngine()
    metrics = engine._metrics([_trade(pnl=10.0), _trade(pnl=-5.0)], "test")
    assert metrics.trades == 2
    assert metrics.expectancy == 2.5


def test_segment_comparison() -> None:
    engine = VwapValidationEngine()
    trades = [
        _trade(vwap_position="Above VWAP", pnl=20.0),
        _trade(vwap_position="Above VWAP", pnl=15.0),
        _trade(vwap_position="Below VWAP", pnl=-10.0),
        _trade(vwap_position="Below VWAP", pnl=-5.0),
    ]
    comparison = engine._segment_comparison(
        trades,
        lambda item: item.vwap_position,
        frozenset({"Below VWAP"}),
        frozenset({"Above VWAP"}),
        "VWAP Position",
    )
    assert comparison.vwap_improves_profitability is True
    assert comparison.expectancy_delta > 0


def test_optimal_conditions() -> None:
    engine = VwapValidationEngine()
    trades = [_trade(vwap_position="Above VWAP", pnl=10.0) for _ in range(6)]
    trades.extend(_trade(vwap_position="Below VWAP", pnl=-5.0) for _ in range(6))
    optimal = engine._optimal_conditions(trades)
    assert "vwap_position" in optimal
    assert optimal["vwap_position"]["best_bucket"] == "Above VWAP"


def test_report_structure() -> None:
    from src.research.vwap_validation_engine import VwapValidationReport

    report = VwapValidationReport(
        symbol="NIFTY50",
        research_window_days=365,
        start_date="2025-07-03",
        end_date="2026-07-03",
        production_stack={},
        total_trades=100,
        baseline={},
        by_vwap_position={},
        by_vwap_reclaim={},
        by_vwap_rejection={},
        by_distance_from_vwap={},
        by_direction_aligned_vwap={},
        position_comparison={},
        direction_aligned_comparison={},
        vwap_improves_profitability=False,
        recommend_production_filter=False,
        recommended_filter={},
        optimal_vwap_conditions={},
        best_vwap_segments=[],
        worst_vwap_segments=[],
        cross_analysis={},
        conclusions=[],
        execution_time_seconds=1.0,
    )
    payload = report.as_dict()
    assert payload["total_trades"] == 100
    assert "by_vwap_position" in payload


def test_missing_metadata_raises() -> None:
    with pytest.raises(VwapValidationError):
        generate_vwap_validation_report(filter_report_path=Path("missing.json"))


def test_min_intelligence_score_constant() -> None:
    assert MIN_INTELLIGENCE_SCORE == 65


@pytest.mark.integration
def test_full_validation_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parent.parent
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = VwapValidationEngine().run(metadata)
    assert report.total_trades >= 0
