"""Tests for trigger entry optimization research."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.research.trigger_entry_optimization_research import (
    DEFAULT_TRIGGER_REPORT_PATH,
    ENTRY_METHODS,
    TriggerEntryOptimizationError,
    TriggerEntryOptimizationResearch,
    generate_trigger_entry_optimization_report,
)


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
                "Close": price + 0.2,
                "Volume": 100000,
                "Buy_Side_Liquidity": price + 5,
                "Sell_Side_Liquidity": price - 5,
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bullish_FVG_Bottom": pd.NA,
                "Bearish_FVG_Top": pd.NA,
                "Bearish_FVG_Bottom": pd.NA,
            }
        )
    frame = pd.DataFrame(rows)
    frame.at[55, "Bullish_BOS"] = "Active"
    frame.at[55, "Bullish_FVG_Top"] = frame.at[55, "High"] + 0.5
    frame.at[55, "Bullish_FVG_Bottom"] = frame.at[55, "Low"] - 0.2
    return frame


def test_entry_methods_count() -> None:
    assert len(ENTRY_METHODS) == 8


def test_resolve_trigger_close() -> None:
    engine = TriggerEntryOptimizationResearch()
    frame = _pipeline_frame()
    resolution = engine._resolve_trigger_close(frame, 50, "bullish")
    assert resolution.triggered
    assert resolution.entry_price == pytest.approx(round(float(frame.iloc[50]["Close"]), 2))


def test_resolve_confirmation_close() -> None:
    engine = TriggerEntryOptimizationResearch()
    frame = _pipeline_frame()
    resolution = engine._resolve_confirmation_close(frame, 50, "bullish")
    assert resolution.triggered
    assert resolution.entry_bar == 51


def test_resolve_bos_after_trigger() -> None:
    engine = TriggerEntryOptimizationResearch()
    frame = _pipeline_frame()
    resolution = engine._resolve_bos(frame, 50, "bullish")
    assert resolution.triggered
    assert resolution.entry_bar == 55


def test_generate_report(tmp_path: Path) -> None:
    if not DEFAULT_TRIGGER_REPORT_PATH.exists():
        pytest.skip("institutional_trigger_validation.json not available")

    out = tmp_path / "trigger_entry_optimization.json"
    report = generate_trigger_entry_optimization_report(report_path=out)
    assert out.exists()
    assert len(report.entry_methods) == 8
    assert report.overall_entry_metrics
    assert report.best_overall_entry


def test_missing_trigger_report_raises(tmp_path: Path) -> None:
    with pytest.raises(TriggerEntryOptimizationError):
        TriggerEntryOptimizationResearch(
            trigger_report_path=tmp_path / "missing.json",
        ).run()
