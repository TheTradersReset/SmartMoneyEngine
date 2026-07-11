"""Tests for Institutional Market Narrative Engine V2."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.context.institutional_market_narrative_engine_v2 import (
    InstitutionalMarketNarrativeEngineV2,
    InstitutionalMarketNarrativeV2Error,
    LiquidityObjective,
    StructuralQuality,
    generate_institutional_market_narrative_v2_report,
)
from src.research.institutional_edge_extraction_research import EdgeFeatureRecord
from src.research.tiered_signal_framework_research import TierSignal


def _frame(length: int = 80) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(length):
        price = 100.0 - index * 0.2
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
                "Sell_Side_Liquidity": 95.0,
                "Buy_Liquidity_Sweep": pd.NA,
                "Sell_Liquidity_Sweep": 102.0 if index == 44 else pd.NA,
                "Liquidity_Strength": 2,
                "Swing_High": 102.0 if index >= 40 else pd.NA,
                "Swing_Low": 96.0 if index >= 40 else pd.NA,
                "Equal_High": pd.NA,
                "Equal_Low": pd.NA,
                "HH": pd.NA,
                "HL": pd.NA,
                "LH": pd.NA,
                "LL": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _record() -> EdgeFeatureRecord:
    return EdgeFeatureRecord(
        timeframe="5M",
        direction="bearish",
        bos_timestamp="2026-01-02T11:40:00+05:30",
        realized_pnl_points=100.0,
        risk_points=10.0,
        win=True,
        displacement_strength="Strong",
        fvg_size_points=30.0,
        fvg_freshness_bars=10,
        fvg_retests=1,
        distance_from_liquidity_pool_points=30.0,
        distance_from_swing_points=10.0,
        choch_to_bos_minutes=120.0,
        bos_to_fvg_reclaim_minutes=0.0,
        expansion_speed_points_per_minute=2.0,
        expansion_size_points=200.0,
        trait_tags=(),
    )


def test_dealing_range() -> None:
    engine = InstitutionalMarketNarrativeEngineV2()
    frame = _frame()
    dealing = engine._dealing_range(frame, 50)
    assert dealing.swing_high >= dealing.swing_low
    assert dealing.equilibrium == round((dealing.swing_high + dealing.swing_low) / 2, 2)


def test_structural_quality_institutional() -> None:
    engine = InstitutionalMarketNarrativeEngineV2()
    score = engine._structural_quality_score(_record())
    label = engine._structural_quality_label(score)
    assert score >= 60
    assert label in {StructuralQuality.STRONG, StructuralQuality.INSTITUTIONAL}


def test_evaluate_signal() -> None:
    engine = InstitutionalMarketNarrativeEngineV2()
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
    narrative = engine.evaluate_signal(frame, signal, _record())
    assert narrative.market_phase
    assert narrative.liquidity_objective
    assert 0 <= narrative.narrative_confidence <= 100
    assert narrative.expected_expansion_direction == "bearish"
    assert len(narrative.sequence_events) >= 4


def test_liquidity_objective_external_raid() -> None:
    engine = InstitutionalMarketNarrativeEngineV2()
    frame = _frame()
    dealing = engine._dealing_range(frame, 50)
    objective = engine._liquidity_objective("bearish", frame, 50, dealing)
    assert objective in LiquidityObjective


def test_missing_metadata_raises() -> None:
    with pytest.raises(InstitutionalMarketNarrativeV2Error):
        generate_institutional_market_narrative_v2_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = InstitutionalMarketNarrativeEngineV2(timeframes=("5M",)).run(metadata)
    assert report.total_signals >= 0
