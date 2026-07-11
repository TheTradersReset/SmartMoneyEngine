"""Tests for liquidity sweep outcome validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.liquidity_sweep_outcome_validation_research import (
    FORWARD_WINDOWS,
    MOVE_THRESHOLDS,
    LiquiditySweepOutcomeValidationError,
    LiquiditySweepOutcomeValidationResearch,
    SweepOutcomeClass,
    generate_liquidity_sweep_outcome_validation_report,
)


def _pipeline_frame(length: int = 120) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(length):
        price = 100.0 + index * 0.2
        timestamp = base + pd.Timedelta(minutes=5 * index)
        is_sweep = index == 40
        rows.append(
            {
                "Date": timestamp.isoformat(),
                "Open": price,
                "High": price + (3.0 if is_sweep else 1.0),
                "Low": price - 0.8,
                "Close": price - 1.0 if is_sweep else price + 0.4,
                "Volume": 150000 if is_sweep else 100000,
                "Swing_High": price + 5 if index % 15 == 0 else pd.NA,
                "Swing_Low": price - 5 if index % 17 == 0 else pd.NA,
                "Trend": "BEARISH" if index >= 40 else "BULLISH",
                "Trend_Strength": 2,
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": price if index == 45 else pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": price if index == 42 else pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bullish_FVG_Bottom": pd.NA,
                "Bearish_FVG_Top": price + 2 if index == 43 else pd.NA,
                "Bearish_FVG_Bottom": price + 1 if index == 43 else pd.NA,
                "Equal_High": price + 4 if index >= 30 else pd.NA,
                "Equal_Low": price - 4 if index >= 30 else pd.NA,
                "Buy_Side_Liquidity": price + 4 if index >= 30 else pd.NA,
                "Sell_Side_Liquidity": price - 4 if index >= 30 else pd.NA,
                "Buy_Liquidity_Sweep": price + 3 if is_sweep else pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
                "Liquidity_Strength": 3 if is_sweep else 1,
            }
        )
    frame = pd.DataFrame(rows)
    for index in range(41, length):
        frame.at[index, "Low"] = frame.at[40, "Close"] - (index - 40) * 2.5
        frame.at[index, "Close"] = frame.at[index, "Low"] + 0.5
        frame.at[index, "High"] = frame.at[index, "Close"] + 0.5
    return frame


def test_move_thresholds() -> None:
    assert MOVE_THRESHOLDS == (50, 100, 150, 200)
    assert FORWARD_WINDOWS == (20, 40, 80)


def test_forward_move_matrix() -> None:
    engine = LiquiditySweepOutcomeValidationResearch()
    frame = _pipeline_frame()
    matrix = engine._forward_move_matrix(frame, 40, "bearish")
    assert matrix.favorable_move_points["80"] >= 50


def test_classify_outcome_failed() -> None:
    engine = LiquiditySweepOutcomeValidationResearch()
    matrix = engine._forward_move_matrix(_pipeline_frame(50), 10, "bullish")
    outcome = engine._classify_outcome(
        "Sell Side Sweep",
        "bullish",
        "BULLISH",
        True,
        matrix,
        False,
        False,
    )
    assert outcome == SweepOutcomeClass.FAILED.value


def test_hit_rate() -> None:
    engine = LiquiditySweepOutcomeValidationResearch()
    frame = _pipeline_frame()
    enriched = engine.liquidity_map_engine._attach_calendar_levels(frame)
    intel = engine.intelligence_engine.enrich(frame)
    record = engine._analyze_sweep(frame, enriched, intel, 40, "Buy Side Sweep", "5M")
    assert record is not None
    rate = engine._hit_rate([record], 50, 80)
    assert rate in {0.0, 100.0}


def test_missing_metadata_raises() -> None:
    with pytest.raises(LiquiditySweepOutcomeValidationError):
        generate_liquidity_sweep_outcome_validation_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = LiquiditySweepOutcomeValidationResearch(timeframes=("5M",)).run(metadata)
    assert report.total_sweeps > 0
    assert report.top_sweep_patterns is not None
