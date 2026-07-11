"""Tests for sequence entry timing validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.institutional_move_validation_research import SequenceInstance
from src.research.sequence_entry_timing_validation_research import (
    ENTRY_STAGES,
    EnrichedSequence,
    SequenceEntryTimingValidationError,
    SequenceEntryTimingValidationResearch,
    generate_sequence_entry_timing_validation_report,
)


def _sequence() -> EnrichedSequence:
    base = SequenceInstance(
        timeframe="5M",
        direction="bullish",
        sweep_bar=10,
        displacement_bar=15,
        choch_bar=20,
        bos_bar=25,
        sweep_timestamp="2026-01-02T09:15:00+05:30",
        bos_timestamp="2026-01-02T10:30:00+05:30",
        displacement_strength="Strong",
    )
    return EnrichedSequence(sequence=base, fvg_reclaim_bar=26)


def _pipeline_frame(length: int = 120) -> pd.DataFrame:
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
                "Close": price + 0.3,
                "Volume": 100000,
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
    for index in range(30, length):
        frame.at[index, "High"] = frame.at[29, "Close"] + (index - 29) * 2.0
        frame.at[index, "Low"] = frame.at[index, "High"] - 0.5
        frame.at[index, "Close"] = frame.at[index, "High"] - 0.1
    return frame


def test_entry_stage_count() -> None:
    assert len(ENTRY_STAGES) == 5


def test_entry_bar_mapping() -> None:
    engine = SequenceEntryTimingValidationResearch()
    enriched = _sequence()
    assert engine._entry_bar_for_stage(enriched, "bos_confirmation") == 25
    assert engine._entry_bar_for_stage(enriched, "fvg_reclaim") == 26


def test_simulate_stage_trade() -> None:
    engine = SequenceEntryTimingValidationResearch()
    frame = _pipeline_frame()
    outcome = engine._simulate_stage_trade(frame, _sequence(), "bos_confirmation")
    assert outcome is not None
    assert outcome.stage_key == "bos_confirmation"


def test_metrics_for_stage() -> None:
    engine = SequenceEntryTimingValidationResearch()
    frame = _pipeline_frame()
    outcomes = [
        engine._simulate_stage_trade(frame, _sequence(), "bos_confirmation"),
        engine._simulate_stage_trade(frame, _sequence(), "fvg_reclaim"),
    ]
    outcomes = [item for item in outcomes if item is not None]
    metrics = engine._metrics_for_stage("bos_confirmation", outcomes)
    assert metrics.trades == len(outcomes)


def test_missing_metadata_raises() -> None:
    with pytest.raises(SequenceEntryTimingValidationError):
        generate_sequence_entry_timing_validation_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = SequenceEntryTimingValidationResearch(timeframes=("5M",)).run(metadata)
    assert report.total_sequences >= 0
    assert report.recommended_institutional_entry_stage
