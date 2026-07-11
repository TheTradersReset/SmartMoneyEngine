"""Tests for institutional move validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.context.liquidity_narrative_engine import DisplacementStrength
from src.research.institutional_move_validation_research import (
    ForwardMoveOutcome,
    InstitutionalMoveValidationError,
    InstitutionalMoveValidationResearch,
    generate_institutional_move_validation_report,
)


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
                "Bearish_FVG_Top": 99.5 if index == 49 else pd.NA,
                "Bearish_FVG_Bottom": 98.5 if index == 49 else pd.NA,
                "Buy_Side_Liquidity": pd.NA,
                "Sell_Side_Liquidity": pd.NA,
                "Buy_Liquidity_Sweep": 102.0 if index == 42 else pd.NA,
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
    rows[50]["Open"] = 99.0
    rows[50]["Close"] = 97.0
    rows[50]["High"] = 99.5
    rows[50]["Low"] = 96.5
    return pd.DataFrame(rows)


def test_displacement_detection() -> None:
    engine = InstitutionalMoveValidationResearch()
    frame = _frame()
    found, bar, strength = engine._has_displacement_between(frame, 42, 50, "bearish")
    assert found is True
    assert strength in {DisplacementStrength.MEDIUM.value, DisplacementStrength.STRONG.value}


def test_forward_move_bearish() -> None:
    engine = InstitutionalMoveValidationResearch()
    frame = _frame()
    outcome = engine._forward_move_from_bar(frame, 50, "bearish", "5M", "bos")
    assert outcome is not None
    assert outcome.forward_move_points > 0


def test_cohort_metrics() -> None:
    engine = InstitutionalMoveValidationResearch()
    outcomes = [
        ForwardMoveOutcome(
            timeframe="5M",
            direction="bearish",
            anchor_bar=50,
            anchor_timestamp="t",
            anchor_type="bos",
            forward_move_points=80.0,
            expansion_bar=55,
            bos_to_expansion_bars=5,
            bos_to_expansion_minutes=25.0,
            moved_over_50=True,
            moved_over_100=False,
            moved_over_150=False,
            directional_win=True,
        )
    ]
    metrics = engine._cohort_metrics("test", outcomes)
    assert metrics.occurrences == 1
    assert metrics.moves_over_50 == 1
    assert metrics.win_rate_pct == 100.0


def test_outperforms_logic() -> None:
    engine = InstitutionalMoveValidationResearch()
    from src.research.institutional_move_validation_research import CohortMetrics

    sequence = CohortMetrics(
        label="seq",
        occurrences=10,
        moves_over_50=8,
        moves_over_100=5,
        moves_over_150=2,
        pct_moves_over_50=80.0,
        pct_moves_over_100=50.0,
        pct_moves_over_150=20.0,
        average_move_size=90.0,
        win_rate_pct=85.0,
        win_rate_by_direction={},
        average_bos_to_expansion_minutes=30.0,
    )
    baseline = CohortMetrics(
        label="base",
        occurrences=100,
        moves_over_50=40,
        moves_over_100=20,
        moves_over_150=5,
        pct_moves_over_50=40.0,
        pct_moves_over_100=20.0,
        pct_moves_over_150=5.0,
        average_move_size=50.0,
        win_rate_pct=60.0,
        win_rate_by_direction={},
        average_bos_to_expansion_minutes=45.0,
    )
    assert engine._outperforms(sequence, baseline) is True


def test_report_structure() -> None:
    from src.research.institutional_move_validation_research import (
        InstitutionalMoveValidationReport,
    )

    report = InstitutionalMoveValidationReport(
        symbol="NIFTY50",
        research_window_days=365,
        start_date="2025-07-03",
        end_date="2026-07-03",
        timeframes_analyzed=["5M"],
        institutional_sequence={},
        comparison={},
        sequence_outperforms_components={},
        win_rate_by_direction={},
        conclusions=[],
        execution_time_seconds=1.0,
    )
    payload = report.as_dict()
    assert "institutional_sequence" in payload


def test_missing_metadata_raises() -> None:
    with pytest.raises(InstitutionalMoveValidationError):
        generate_institutional_move_validation_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_validation_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parent.parent
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = InstitutionalMoveValidationResearch(timeframes=("5M",)).run(metadata)
    assert report.institutional_sequence["occurrences"] >= 0
