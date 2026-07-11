"""Tests for Tier-2 entry optimization research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.tier2_entry_optimization_research import (
    Tier2EntryOptimizationError,
    Tier2EntryOptimizationResearch,
    generate_tier2_entry_optimization_report,
)
from src.research.tiered_signal_framework_research import TierSignal


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
                "Bearish_FVG_Top": 99.5 if 45 <= index <= 55 else pd.NA,
                "Bearish_FVG_Bottom": 98.5 if 45 <= index <= 55 else pd.NA,
                "Bullish_OB_High": pd.NA,
                "Bullish_OB_Low": pd.NA,
                "Bearish_OB_High": 99.0 if 52 <= index <= 55 else pd.NA,
                "Bearish_OB_Low": 97.5 if 52 <= index <= 55 else pd.NA,
                "Bullish_OB_Mitigated": pd.NA,
                "Bearish_OB_Mitigated": pd.NA,
                "Buy_Side_Liquidity": 102.0,
                "Sell_Side_Liquidity": 95.0,
                "Buy_Liquidity_Sweep": pd.NA,
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
    rows[50]["Close"] = 97.0
    rows[53]["Low"] = 98.8
    rows[53]["Close"] = 98.9
    return pd.DataFrame(rows)


def test_bos_close_entry() -> None:
    engine = Tier2EntryOptimizationResearch()
    frame = _frame()
    signal = TierSignal(
        tier="tier_2",
        timeframe="5M",
        direction="bearish",
        bos_bar=50,
        bos_timestamp=str(frame.iloc[50]["Date"]),
    )
    trigger = engine._resolve_entry("A_bos_close", frame, signal)
    assert trigger.triggered is True
    assert trigger.entry_bar == 50


def test_fvg_retest_entry() -> None:
    engine = Tier2EntryOptimizationResearch()
    frame = _frame()
    signal = TierSignal(
        tier="tier_2",
        timeframe="5M",
        direction="bearish",
        bos_bar=50,
        bos_timestamp=str(frame.iloc[50]["Date"]),
    )
    outcome = engine.evaluate_method("B_first_fvg_retest", frame, signal)
    assert outcome.method_key == "B_first_fvg_retest"


def test_simulate_from_entry() -> None:
    engine = Tier2EntryOptimizationResearch()
    frame = _frame()
    risk, mfe, mae, pnl, rr, win = engine._simulate_from_entry(frame, 50, 97.0, "bearish")
    assert risk > 0
    assert mfe >= 0
    assert mae >= 0


def test_missing_metadata_raises() -> None:
    with pytest.raises(Tier2EntryOptimizationError):
        generate_tier2_entry_optimization_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = Tier2EntryOptimizationResearch(timeframes=("5M",)).run(metadata)
    assert report.total_signals >= 0
