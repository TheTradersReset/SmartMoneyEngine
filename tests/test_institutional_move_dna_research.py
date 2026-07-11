"""Tests for institutional move DNA research."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.research.institutional_move_dna_research import (
    MOVE_THRESHOLDS,
    PRE_MOVE_LOOKBACK,
    InstitutionalMoveDnaError,
    InstitutionalMoveDnaResearch,
    generate_institutional_move_dna_report,
)
from src.research.liquidity_move_reconstruction_research import _CheapMoveCandidate


def _pipeline_frame(length: int = 150) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(length):
        price = 100.0 + index * 0.5
        timestamp = base + pd.Timedelta(minutes=5 * index)
        rows.append(
            {
                "Date": timestamp.isoformat(),
                "Open": price,
                "High": price + 1.0,
                "Low": price - 0.8,
                "Close": price + 0.2,
                "Volume": 100000,
                "Swing_High": pd.NA,
                "Swing_Low": pd.NA,
                "Buy_Side_Liquidity": price + 5,
                "Sell_Side_Liquidity": price - 5,
                "Buy_Liquidity_Sweep": pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bullish_FVG_Bottom": pd.NA,
                "Bearish_FVG_Top": pd.NA,
                "Bearish_FVG_Bottom": pd.NA,
                "Bullish_OB_High": pd.NA,
                "Bearish_OB_High": pd.NA,
            }
        )
    frame = pd.DataFrame(rows)
    for index in range(80, length):
        frame.at[index, "High"] = frame.at[79, "Close"] + (index - 79) * 3.0
        frame.at[index, "Low"] = frame.at[index, "High"] - 0.4
        frame.at[index, "Close"] = frame.at[index, "High"] - 0.1
    return frame


def test_move_thresholds_include_500() -> None:
    assert MOVE_THRESHOLDS == (100, 200, 300, 500)
    assert PRE_MOVE_LOOKBACK == 50


def test_build_trait_tags() -> None:
    engine = InstitutionalMoveDnaResearch(symbols=("NIFTY50",))
    measurements = {
        "liquidity_grabs": 2,
        "false_breakouts": 0,
        "false_breakdowns": 0,
        "support_tests": 3,
        "resistance_tests": 0,
        "round_number_interactions": 1,
        "choch_count": 1,
        "bos_count": 1,
        "fvg_created": True,
        "fvg_reclaimed": True,
        "order_block_reactions": 1,
        "hammer_count": 1,
        "shooting_star_count": 0,
        "marubozu_count": 0,
        "gap_up_count": 0,
        "gap_down_count": 0,
        "volume_expansion_ratio": 1.6,
        "average_wick_points": 2.0,
        "rsi_band": "Oversold",
        "bullish_divergence": True,
        "bearish_divergence": False,
        "premium_discount_zone": "Discount Zone",
        "level_strength_category": "Moderate",
        "consolidation_bars": 25,
        "origin_displacement_strength": "Strong",
    }
    tags = engine._build_trait_tags(measurements, "bullish")
    assert "Liquidity Grab x2+" in tags
    assert "FVG Reclaim" in tags
    assert "Bullish Divergence" in tags


def test_analyze_move_produces_dna_record() -> None:
    engine = InstitutionalMoveDnaResearch(symbols=("NIFTY50",))
    frame = _pipeline_frame()
    enriched = engine.context_builder.enrich(frame)
    intel_frame = engine.intelligence_engine.enrich(frame)
    candidate = _CheapMoveCandidate(
        start_bar=70,
        expansion_bar=100,
        direction="bullish",
        magnitude=150.0,
    )
    record = engine._analyze_move("NIFTY50", frame, enriched, intel_frame, candidate, "5M")
    assert record.hit_100_plus
    assert record.dna_pattern
    assert record.measurements["liquidity_grabs"] >= 0


def test_generate_report() -> None:
    report = generate_institutional_move_dna_report()
    assert report.total_moves_analyzed > 0
    assert report.trait_predictive_power
    assert report.top_20_bullish_dna_patterns or report.top_20_bearish_dna_patterns


def test_missing_filter_report_raises(tmp_path: Path) -> None:
    with pytest.raises(InstitutionalMoveDnaError):
        generate_institutional_move_dna_report(
            filter_report_path=tmp_path / "missing.json",
        )
