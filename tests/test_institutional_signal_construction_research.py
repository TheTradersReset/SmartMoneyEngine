"""Tests for institutional signal construction research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.institutional_signal_construction_research import (
    InstitutionalSignalConstructionError,
    InstitutionalSignalConstructionResearch,
    MOVE_THRESHOLDS,
    generate_institutional_signal_construction_report,
)
from src.research.liquidity_move_reconstruction_research import _CheapMoveCandidate


def _pipeline_frame(length: int = 150) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(length):
        price = 100.0 + index * 0.15
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
                "Equal_High": pd.NA,
                "Equal_Low": pd.NA,
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bullish_FVG_Bottom": pd.NA,
                "Bearish_FVG_Top": pd.NA,
                "Bearish_FVG_Bottom": pd.NA,
                "Bullish_OB_High": pd.NA,
                "Bullish_OB_Low": pd.NA,
                "Bearish_OB_High": pd.NA,
                "Bearish_OB_Low": pd.NA,
                "Buy_Side_Liquidity": price + 4,
                "Sell_Side_Liquidity": price - 4,
                "Buy_Liquidity_Sweep": pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
            }
        )
    frame = pd.DataFrame(rows)
    for index in range(80, length):
        frame.at[index, "High"] = frame.at[79, "Close"] + (index - 79) * 2.5
        frame.at[index, "Low"] = frame.at[index, "High"] - 0.4
        frame.at[index, "Close"] = frame.at[index, "High"] - 0.1
    return frame


def test_move_thresholds_include_300() -> None:
    assert 300 in MOVE_THRESHOLDS


def test_analyze_move_produces_narrative() -> None:
    engine = InstitutionalSignalConstructionResearch(symbols=("NIFTY50",))
    frame = _pipeline_frame()
    enriched = engine.context_builder.enrich(frame)
    enriched = engine.liquidity_map_engine._attach_calendar_levels(enriched)
    intel = engine.intelligence_engine.enrich(frame)
    candidate = _CheapMoveCandidate(
        start_bar=60,
        expansion_bar=100,
        direction="bullish",
        magnitude=120.0,
    )
    ctx = engine._analyze_move("NIFTY50", frame, enriched, intel, candidate, "5M")
    assert ctx.narrative
    assert ctx.signal_side == "BUY"
    assert ctx.move_magnitude_points == 120.0


def test_production_candidate_analysis() -> None:
    engine = InstitutionalSignalConstructionResearch(symbols=("NIFTY50",))
    frame = _pipeline_frame()
    enriched = engine.context_builder.enrich(frame)
    enriched = engine.liquidity_map_engine._attach_calendar_levels(enriched)
    intel = engine.intelligence_engine.enrich(frame)
    moves = [
        engine._analyze_move(
            "NIFTY50",
            frame,
            enriched,
            intel,
            _CheapMoveCandidate(60, 100, "bullish", 120.0 + index * 10),
            "5M",
        )
        for index in range(12)
    ]
    power, candidates, scores = engine._production_candidate_analysis(moves)
    assert "BUY" in power
    assert "BUY" in candidates


def test_missing_metadata_raises() -> None:
    with pytest.raises(InstitutionalSignalConstructionError):
        generate_institutional_signal_construction_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = InstitutionalSignalConstructionResearch(
        symbols=("NIFTY50",),
        timeframes=("5M",),
    ).run(metadata)
    assert report.total_moves_analyzed >= 0
    assert report.recommended_production_signal_architecture
