"""Tests for support/resistance pressure research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.support_resistance_pressure_research import (
    LevelState,
    MajorLevel,
    SupportResistancePressureError,
    SupportResistancePressureResearch,
    generate_support_resistance_pressure_report,
)


def _pipeline_frame(length: int = 120) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(length):
        price = 102.0
        timestamp = base + pd.Timedelta(minutes=5 * index)
        rows.append(
            {
                "Date": timestamp.isoformat(),
                "Open": price,
                "High": price + 0.5,
                "Low": price - 0.5,
                "Close": price,
                "Volume": 100000,
                "Swing_High": pd.NA,
                "Swing_Low": pd.NA,
                "Equal_High": pd.NA,
                "Equal_Low": pd.NA,
                "Buy_Side_Liquidity": price + 4,
                "Sell_Side_Liquidity": 100.0,
                "Buy_Liquidity_Sweep": pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
            }
        )
    frame = pd.DataFrame(rows)
    frame.at[10, "Swing_Low"] = 100.0
    for index in range(25, 30):
        frame.at[index, "Open"] = 100.2
        frame.at[index, "High"] = 100.4
        frame.at[index, "Low"] = 99.8
        frame.at[index, "Close"] = 100.1
    frame.at[30, "Open"] = 99.5
    frame.at[30, "High"] = 99.7
    frame.at[30, "Low"] = 98.5
    frame.at[30, "Close"] = 98.8
    return frame


def test_level_state_classification() -> None:
    engine = SupportResistancePressureResearch()
    assert engine._level_state_from_tests(1, False) == LevelState.FRESH.value
    assert engine._level_state_from_tests(3, False) == LevelState.RETESTED.value
    assert engine._level_state_from_tests(6, False) == LevelState.EXHAUSTED.value
    assert engine._level_state_from_tests(2, True) == LevelState.BROKEN.value


def test_extract_levels() -> None:
    engine = SupportResistancePressureResearch()
    frame = _pipeline_frame()
    levels = engine._extract_levels(frame)
    assert levels
    assert any(level.level_side == "support" for level in levels)


def test_evaluate_support_break() -> None:
    engine = SupportResistancePressureResearch()
    frame = _pipeline_frame()
    level = MajorLevel(
        level_price=100.0,
        level_side="support",
        source_column="Swing_Low",
        formation_bar=10,
        formation_timestamp=str(frame.iloc[10]["Date"]),
    )
    records = engine._evaluate_level(frame, level, "5M")
    assert records
    assert any(record.outcome == "support_break" for record in records)


def test_rank_patterns() -> None:
    engine = SupportResistancePressureResearch()
    frame = _pipeline_frame()
    level = MajorLevel(
        level_price=100.0,
        level_side="support",
        source_column="Swing_Low",
        formation_bar=10,
        formation_timestamp=str(frame.iloc[10]["Date"]),
    )
    records = engine._evaluate_level(frame, level, "5M")
    ranked = engine._rank_patterns(records, "support_break")
    assert ranked
    assert ranked[0].rank == 1


def test_missing_metadata_raises() -> None:
    with pytest.raises(SupportResistancePressureError):
        generate_support_resistance_pressure_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = SupportResistancePressureResearch(timeframes=("5M",)).run(metadata)
    assert len(report.level_records) >= 0
    assert report.outcome_counts
