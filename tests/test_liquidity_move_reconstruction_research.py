"""Tests for liquidity move reconstruction research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.context.liquidity_narrative_engine import FvgContext
from src.context.market_intelligence_engine import RsiState
from src.research.liquidity_move_reconstruction_research import (
    LiquidityMoveReconstructionError,
    LiquidityMoveReconstructionResearch,
    ReconstructedMove,
    generate_liquidity_move_reconstruction_report,
)


def _frame(length: int = 120) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(length):
        price = 100.0 - index * 0.2
        if index > 40:
            price = 100.0 - (index - 40) * 1.5
        timestamp = base + pd.Timedelta(minutes=5 * index)
        rows.append(
            {
                "Date": timestamp.isoformat(),
                "Open": price,
                "High": price + 1.0,
                "Low": price - 1.0,
                "Close": price - 0.2,
                "Volume": 100000,
                "Trend": "BEARISH",
                "Trend_Strength": 2,
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": 100.0 if index == 45 else pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": 101.0 if index == 42 else pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bullish_FVG_Bottom": pd.NA,
                "Bearish_FVG_Top": pd.NA,
                "Bearish_FVG_Bottom": pd.NA,
                "Buy_Side_Liquidity": pd.NA,
                "Sell_Side_Liquidity": pd.NA,
                "Buy_Liquidity_Sweep": 102.0 if index == 40 else pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
                "Liquidity_Strength": 2,
                "Swing_High": pd.NA,
                "Swing_Low": pd.NA,
                "HH": pd.NA,
                "HL": pd.NA,
                "LH": pd.NA,
                "LL": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def test_liquidity_event_labels() -> None:
    assert LiquidityMoveReconstructionResearch._liquidity_event(True, False) == "Buy Side Sweep"
    assert LiquidityMoveReconstructionResearch._liquidity_event(True, True) == "Both Sides"
    assert LiquidityMoveReconstructionResearch._liquidity_event(False, False) == "No Liquidity Sweep"


def test_structure_sequence() -> None:
    sequence = LiquidityMoveReconstructionResearch._structure_sequence(
        False,
        True,
        False,
        True,
    )
    assert sequence == "Bearish CHOCH + Bearish BOS"


def test_fvg_behavior() -> None:
    assert LiquidityMoveReconstructionResearch._fvg_behavior(FvgContext.RECLAIMED, "bearish") == "Reclaimed"
    assert LiquidityMoveReconstructionResearch._fvg_behavior(FvgContext.FAILED, None) == "Failed"


def test_rsi_context() -> None:
    assert LiquidityMoveReconstructionResearch._rsi_context(RsiState.WEAK) == "RSI Weak"
    assert LiquidityMoveReconstructionResearch._rsi_context(RsiState.NEUTRAL) == "RSI Neutral"


def test_build_timing() -> None:
    engine = LiquidityMoveReconstructionResearch()
    timing = engine._build_timing(40, 42, 45, 50, "5M")
    assert timing.sweep_to_choch_bars == 2
    assert timing.choch_to_bos_bars == 3
    assert timing.bos_to_expansion_bars == 5
    assert timing.sweep_to_choch_minutes == 10.0


def test_dedupe_moves() -> None:
    moves = [
        ReconstructedMove(
            timeframe="5M",
            direction="bearish",
            threshold_points=50,
            move_magnitude_points=60.0,
            start_timestamp="a",
            expansion_timestamp="b",
            start_bar=10,
            expansion_bar=20,
            liquidity_event="Buy Side Sweep",
            structure_sequence="Bearish CHOCH + Bearish BOS",
            fvg_behavior="Reclaimed",
            market_location="Near Resistance",
            rsi_context="RSI Neutral",
            intelligence_context="MI 70-79",
            intelligence_score=75.0,
            pre_move_sequence="test",
            timing={},
        ),
        ReconstructedMove(
            timeframe="5M",
            direction="bearish",
            threshold_points=50,
            move_magnitude_points=55.0,
            start_timestamp="c",
            expansion_timestamp="d",
            start_bar=11,
            expansion_bar=22,
            liquidity_event="Buy Side Sweep",
            structure_sequence="Bearish CHOCH + Bearish BOS",
            fvg_behavior="Reclaimed",
            market_location="Near Resistance",
            rsi_context="RSI Neutral",
            intelligence_context="MI 70-79",
            intelligence_score=75.0,
            pre_move_sequence="test",
            timing={},
        ),
    ]
    deduped = LiquidityMoveReconstructionResearch._dedupe_moves(moves)
    assert len(deduped) == 1


def test_detect_moves_for_threshold() -> None:
    engine = LiquidityMoveReconstructionResearch()
    frame = _frame()
    intel = engine.intelligence_engine.enrich(frame)
    highs = frame["High"].astype(float)
    lows = frame["Low"].astype(float)
    cheap = engine._dedupe_cheap_moves(engine._detect_moves_cheap(highs, lows, 50))
    assert cheap
    context = engine._pre_move_context(frame, intel, cheap[0].expansion_bar, "5M")
    assert context[0]


def test_rank_sequences() -> None:
    engine = LiquidityMoveReconstructionResearch()
    move = ReconstructedMove(
        timeframe="5M",
        direction="bearish",
        threshold_points=50,
        move_magnitude_points=60.0,
        start_timestamp="a",
        expansion_timestamp="b",
        start_bar=10,
        expansion_bar=20,
        liquidity_event="Buy Side Sweep",
        structure_sequence="Bearish CHOCH + Bearish BOS",
        fvg_behavior="Reclaimed",
        market_location="Near Resistance",
        rsi_context="RSI Neutral",
        intelligence_context="MI 70-79",
        intelligence_score=75.0,
        pre_move_sequence="seq-a",
        timing={"sweep_to_choch_minutes": 10.0, "choch_to_bos_minutes": 15.0, "bos_to_expansion_minutes": 25.0},
    )
    ranked = engine._rank_sequences([move, move])
    assert ranked[0].count == 2


def test_report_structure() -> None:
    from src.research.liquidity_move_reconstruction_research import LiquidityMoveReconstructionReport

    report = LiquidityMoveReconstructionReport(
        symbol="NIFTY50",
        research_window_days=365,
        start_date="2025-07-03",
        end_date="2026-07-03",
        timeframes_analyzed=["5M"],
        move_thresholds=[50, 100, 150],
        total_moves_detected={"50": 10},
        moves=[],
        average_timing={},
        ranked_sequences_by_threshold={},
        top_move_starting_sequences={},
        conclusions=[],
        execution_time_seconds=1.0,
    )
    payload = report.as_dict()
    assert payload["move_thresholds"] == [50, 100, 150]


def test_missing_metadata_raises() -> None:
    with pytest.raises(LiquidityMoveReconstructionError):
        generate_liquidity_move_reconstruction_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_reconstruction_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parent.parent
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = LiquidityMoveReconstructionResearch(timeframes=("5M",)).run(metadata)
    assert sum(report.total_moves_detected.values()) >= 0
