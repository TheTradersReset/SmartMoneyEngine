"""Tests for tiered signal framework research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.tiered_signal_framework_research import (
    TierSignal,
    TieredSignalFrameworkError,
    TieredSignalFrameworkResearch,
    generate_tiered_signal_framework_report,
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


def test_tier1_detection() -> None:
    engine = TieredSignalFrameworkResearch()
    frame = _frame()
    signals = engine._detect_tier1(frame, "5M")
    assert len(signals) >= 1
    assert signals[0].tier == "tier_1"


def test_tier2_detection_without_sweep() -> None:
    engine = TieredSignalFrameworkResearch()
    frame = _frame()
    frame["Buy_Liquidity_Sweep"] = pd.NA
    signals = engine._detect_tier2(frame, "5M")
    assert len(signals) >= 1


def test_tier3_requires_trend_alignment() -> None:
    engine = TieredSignalFrameworkResearch()
    frame = _frame()
    frame["Trend"] = "BULLISH"
    signals = engine._detect_tier3(frame, "5M")
    assert len(signals) == 0


def test_simulate_outcome() -> None:
    engine = TieredSignalFrameworkResearch()
    frame = _frame()
    signal = TierSignal(
        tier="tier_1",
        timeframe="5M",
        direction="bearish",
        bos_bar=50,
        bos_timestamp=str(frame.iloc[50]["Date"]),
    )
    outcome = engine._simulate_outcome(frame, signal)
    assert outcome is not None
    assert outcome.forward_move_points >= 0


def test_tier_metrics() -> None:
    engine = TieredSignalFrameworkResearch()
    from src.research.tiered_signal_framework_research import TierSignalOutcome

    outcomes = [
        TierSignalOutcome(
            tier="tier_1",
            timeframe="5M",
            direction="bearish",
            bos_bar=50,
            bos_timestamp="t",
            risk_points=10.0,
            forward_move_points=80.0,
            realized_pnl_points=80.0,
            realized_rr=8.0,
            win=True,
            expansion_bar=55,
            time_to_expansion_minutes=25.0,
        )
    ]
    metrics = engine._tier_metrics("tier_1", outcomes, research_months=12.0)
    assert metrics.signals == 1
    assert metrics.win_rate_pct == 100.0
    assert metrics.signals_per_month == round(1 / 12.0, 2)


def test_balance_scores() -> None:
    engine = TieredSignalFrameworkResearch()
    from src.research.tiered_signal_framework_research import TierMetrics

    tiers = {
        "tier_1": TierMetrics(
            tier="tier_1",
            label="Tier 1",
            components=[],
            signals=10,
            signals_per_month=1.0,
            win_rate_pct=80.0,
            profit_factor=2.0,
            expectancy=50.0,
            average_rr=2.5,
            average_move_size=100.0,
            average_time_to_expansion_minutes=60.0,
        ),
        "tier_2": TierMetrics(
            tier="tier_2",
            label="Tier 2",
            components=[],
            signals=30,
            signals_per_month=3.0,
            win_rate_pct=70.0,
            profit_factor=1.8,
            expectancy=40.0,
            average_rr=2.0,
            average_move_size=80.0,
            average_time_to_expansion_minutes=45.0,
        ),
        "tier_3": TierMetrics(
            tier="tier_3",
            label="Tier 3",
            components=[],
            signals=100,
            signals_per_month=10.0,
            win_rate_pct=55.0,
            profit_factor=1.2,
            expectancy=15.0,
            average_rr=1.2,
            average_move_size=50.0,
            average_time_to_expansion_minutes=30.0,
        ),
    }
    scores = engine._balance_scores(tiers)
    assert "tier_1" in scores
    assert scores["tier_1"] > scores["tier_3"]


def test_missing_metadata_raises() -> None:
    with pytest.raises(TieredSignalFrameworkError):
        generate_tiered_signal_framework_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = TieredSignalFrameworkResearch(timeframes=("5M",)).run(metadata)
    assert "tier_1" in report.tiers
