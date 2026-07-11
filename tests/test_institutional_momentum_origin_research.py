"""Tests for institutional momentum origin research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.institutional_momentum_origin_research import (
    InstitutionalMomentumOriginError,
    InstitutionalMomentumOriginResearch,
    MOVE_THRESHOLDS,
    PRE_EXPANSION_LOOKBACK,
    generate_institutional_momentum_origin_report,
)
from src.research.liquidity_move_reconstruction_research import _CheapMoveCandidate


def _pipeline_frame(length: int = 120) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(length):
        price = 100.0 + index * 0.2
        timestamp = base + pd.Timedelta(minutes=5 * index)
        rows.append(
            {
                "Date": timestamp.isoformat(),
                "Open": price,
                "High": price + 1.0,
                "Low": price - 0.8,
                "Close": price + 0.3,
                "Volume": 100000 + index * 1000,
                "Swing_High": price + 5 if index % 15 == 0 else pd.NA,
                "Swing_Low": price - 5 if index % 17 == 0 else pd.NA,
                "Trend": "BULLISH",
                "Trend_Strength": 2,
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bullish_FVG_Bottom": pd.NA,
                "Bearish_FVG_Top": pd.NA,
                "Bearish_FVG_Bottom": pd.NA,
                "Equal_High": pd.NA,
                "Equal_Low": pd.NA,
                "Buy_Side_Liquidity": price + 4,
                "Sell_Side_Liquidity": price - 4,
                "Buy_Liquidity_Sweep": pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
                "Liquidity_Strength": 1,
            }
        )
    frame = pd.DataFrame(rows)
    for index in range(60, length):
        frame.at[index, "High"] = frame.at[59, "Close"] + (index - 59) * 3.0
        frame.at[index, "Low"] = frame.at[index, "High"] - 0.5
        frame.at[index, "Close"] = frame.at[index, "High"] - 0.1
    return frame


def test_move_thresholds() -> None:
    assert MOVE_THRESHOLDS == (50, 100, 150, 200)
    assert PRE_EXPANSION_LOOKBACK == 50


def test_candle_pattern_detection() -> None:
    engine = InstitutionalMomentumOriginResearch()
    frame = _pipeline_frame(80)
    metrics = engine._count_candle_patterns(frame.iloc[10:30])
    assert metrics.average_body_points >= 0
    assert metrics.average_wick_points >= 0


def test_analyze_move() -> None:
    engine = InstitutionalMomentumOriginResearch()
    frame = _pipeline_frame(120)
    candidate = _CheapMoveCandidate(
        start_bar=55,
        expansion_bar=90,
        direction="bullish",
        magnitude=85.0,
    )
    analysis = engine._analyze_move(frame, candidate, "5M", 50)
    assert analysis.threshold_points == 50
    assert analysis.pattern_key
    assert analysis.confirmation_candle["body_pct"] >= 0
    assert analysis.trap_analysis["total_traps"] >= 0


def test_rank_patterns() -> None:
    engine = InstitutionalMomentumOriginResearch()
    frame = _pipeline_frame(120)
    records = [
        engine._analyze_move(
            frame,
            _CheapMoveCandidate(55, 90, "bullish", 85.0),
            "5M",
            50,
        ),
        engine._analyze_move(
            frame,
            _CheapMoveCandidate(56, 91, "bullish", 90.0),
            "5M",
            50,
        ),
    ]
    ranked = engine._rank_patterns(records)
    assert ranked
    assert ranked[0].rank == 1


def test_missing_metadata_raises() -> None:
    with pytest.raises(InstitutionalMomentumOriginError):
        generate_institutional_momentum_origin_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = InstitutionalMomentumOriginResearch(timeframes=("5M",)).run(metadata)
    assert sum(report.total_expansion_moves.values()) >= 0
    assert report.pattern_rankings["most_reliable_expansion_pattern"]
