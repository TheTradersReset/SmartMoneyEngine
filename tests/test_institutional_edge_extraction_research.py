"""Tests for institutional edge extraction research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.institutional_edge_extraction_research import (
    EdgeFeatureRecord,
    InstitutionalEdgeExtractionError,
    InstitutionalEdgeExtractionResearch,
    generate_institutional_edge_extraction_report,
)
from src.research.tiered_signal_framework_research import TierSignal


def _frame(length: int = 120) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(length):
        price = 100.0 - index * 0.3
        if index > 45:
            price = 100.0 - (index - 45) * 2.0
        timestamp = base + pd.Timedelta(minutes=5 * index)
        rows.append(
            {
                "Date": timestamp.isoformat(),
                "Open": price + 0.5,
                "High": price + 1.5,
                "Low": price - 1.0,
                "Close": price,
                "Volume": 100000,
                "Trend": "BEARISH",
                "Trend_Strength": 2,
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": 98.0 if index == 50 else pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": 101.0 if index == 46 else pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bullish_FVG_Bottom": pd.NA,
                "Bearish_FVG_Top": 99.5 if 45 <= index <= 50 else pd.NA,
                "Bearish_FVG_Bottom": 98.5 if 45 <= index <= 50 else pd.NA,
                "Buy_Side_Liquidity": 105.0,
                "Sell_Side_Liquidity": pd.NA,
                "Buy_Liquidity_Sweep": pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
                "Liquidity_Strength": 2,
                "Swing_High": 102.0 if index == 50 else pd.NA,
                "Swing_Low": pd.NA,
                "HH": pd.NA,
                "HL": pd.NA,
                "LH": pd.NA,
                "LL": pd.NA,
            }
        )
    rows[50]["Open"] = 99.0
    rows[50]["Close"] = 97.0
    rows[50]["High"] = 99.5
    rows[50]["Low"] = 96.5
    return pd.DataFrame(rows)


def test_fvg_creation_bar() -> None:
    engine = InstitutionalEdgeExtractionResearch()
    frame = _frame()
    bar = engine._find_fvg_creation_bar(frame, 50, "bearish")
    assert bar is not None
    assert bar <= 50


def test_extract_features() -> None:
    engine = InstitutionalEdgeExtractionResearch()
    frame = _frame()
    signal = TierSignal(
        tier="tier_2",
        timeframe="5M",
        direction="bearish",
        bos_bar=50,
        bos_timestamp=str(frame.iloc[50]["Date"]),
        choch_bar=46,
        displacement_bar=50,
    )
    record = engine._extract_features(frame, signal, 100.0, True, 10.0)
    assert record is not None
    assert record.fvg_size_points > 0
    assert len(record.trait_tags) == 10


def test_trait_comparisons() -> None:
    engine = InstitutionalEdgeExtractionResearch()
    top = [
        EdgeFeatureRecord(
            timeframe="5M",
            direction="bearish",
            bos_timestamp="t1",
            realized_pnl_points=200.0,
            risk_points=10.0,
            win=True,
            displacement_strength="Strong",
            fvg_size_points=30.0,
            fvg_freshness_bars=5,
            fvg_retests=0,
            distance_from_liquidity_pool_points=10.0,
            distance_from_swing_points=15.0,
            choch_to_bos_minutes=20.0,
            bos_to_fvg_reclaim_minutes=0.0,
            expansion_speed_points_per_minute=2.0,
            expansion_size_points=250.0,
            trait_tags=("Displacement Strong", "Expansion Size XLarge (400+ pts)"),
        )
    ]
    bottom = [
        EdgeFeatureRecord(
            timeframe="5M",
            direction="bearish",
            bos_timestamp="t2",
            realized_pnl_points=-50.0,
            risk_points=50.0,
            win=False,
            displacement_strength="Medium",
            fvg_size_points=10.0,
            fvg_freshness_bars=25,
            fvg_retests=2,
            distance_from_liquidity_pool_points=80.0,
            distance_from_swing_points=90.0,
            choch_to_bos_minutes=200.0,
            bos_to_fvg_reclaim_minutes=15.0,
            expansion_speed_points_per_minute=0.2,
            expansion_size_points=30.0,
            trait_tags=("Displacement Medium", "FVG Retests 2+"),
        )
    ]
    comparisons = engine._trait_comparisons(top, bottom)
    assert comparisons
    assert any(item.trait == "Displacement Strong" for item in comparisons)


def test_quality_model() -> None:
    engine = InstitutionalEdgeExtractionResearch()
    from src.research.institutional_edge_extraction_research import TraitComparison

    traits = [
        TraitComparison(
            trait="Displacement Strong",
            top_winners_pct=60.0,
            bottom_losers_pct=20.0,
            delta_pct=40.0,
            top_winners_count=60,
            bottom_losers_count=20,
        )
    ]
    model = engine._quality_scoring_model(traits)
    assert model["max_score"] == 100
    assert model["components"]


def test_missing_metadata_raises() -> None:
    with pytest.raises(InstitutionalEdgeExtractionError):
        generate_institutional_edge_extraction_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = InstitutionalEdgeExtractionResearch(timeframes=("5M",)).run(metadata)
    assert report.total_signals >= 0
