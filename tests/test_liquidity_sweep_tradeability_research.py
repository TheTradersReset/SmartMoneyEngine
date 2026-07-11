"""Tests for liquidity sweep tradeability validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.liquidity_sweep_tradeability_research import (
    LiquiditySweepTradeabilityError,
    LiquiditySweepTradeabilityResearch,
    generate_liquidity_sweep_tradeability_report,
)


def _pipeline_frame(length: int = 100) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(length):
        price = 100.0 + index * 0.2
        timestamp = base + pd.Timedelta(minutes=5 * index)
        is_sweep = index == 30
        is_bos = index == 35
        rows.append(
            {
                "Date": timestamp.isoformat(),
                "Open": price,
                "High": price + (3.0 if is_sweep else 1.0),
                "Low": price - 0.8,
                "Close": price + 0.4,
                "Volume": 150000 if is_sweep else 100000,
                "Swing_High": price + 5 if index % 15 == 0 else pd.NA,
                "Swing_Low": price - 5 if index % 17 == 0 else pd.NA,
                "Trend": "BEARISH" if index >= 30 else "BULLISH",
                "Trend_Strength": 2,
                "Bullish_BOS": price if is_bos else pd.NA,
                "Bearish_BOS": pd.NA,
                "Bullish_CHOCH": price if index == 33 else pd.NA,
                "Bearish_CHOCH": pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bullish_FVG_Bottom": pd.NA,
                "Bearish_FVG_Top": pd.NA,
                "Bearish_FVG_Bottom": pd.NA,
                "Equal_High": price + 4 if index >= 20 else pd.NA,
                "Equal_Low": price - 4 if index >= 20 else pd.NA,
                "Buy_Side_Liquidity": price + 4 if index >= 20 else pd.NA,
                "Sell_Side_Liquidity": price - 4 if index >= 20 else pd.NA,
                "Buy_Liquidity_Sweep": pd.NA,
                "Sell_Liquidity_Sweep": price - 1 if is_sweep else pd.NA,
                "Liquidity_Strength": 3 if is_sweep else 1,
            }
        )
    frame = pd.DataFrame(rows)
    for index in range(36, length):
        frame.at[index, "High"] = frame.at[35, "Close"] + (index - 35) * 3.0
        frame.at[index, "Low"] = frame.at[index, "High"] - 1.0
        frame.at[index, "Close"] = frame.at[index, "High"] - 0.2
    return frame


def test_find_post_sweep_bos() -> None:
    engine = LiquiditySweepTradeabilityResearch()
    frame = _pipeline_frame()
    bos_bar = engine._find_post_sweep_bos(frame, 30, "bullish")
    assert bos_bar == 35


def test_simulate_trade_path() -> None:
    engine = LiquiditySweepTradeabilityResearch()
    frame = _pipeline_frame()
    entry_bar = 35
    entry_price = float(frame.iloc[entry_bar]["Close"])
    stop, risk = engine.construction_engine._structural_stop(frame, entry_bar, entry_price, "bullish")
    opposite = engine.construction_engine._opposite_liquidity_target(
        frame,
        entry_bar,
        entry_price,
        "bullish",
        risk,
    )
    htf = engine.construction_engine._htf_supply_demand_target(
        frame,
        entry_bar,
        entry_price,
        "bullish",
        risk,
    )
    result = engine._simulate_trade_path(
        frame,
        entry_bar,
        entry_price,
        "bullish",
        stop,
        risk,
        opposite,
        htf,
    )
    assert "hit_1r_before_sl" in result
    assert isinstance(result["stopped_out"], bool)


def test_metrics_for_cohort() -> None:
    engine = LiquiditySweepTradeabilityResearch()
    from src.research.liquidity_sweep_tradeability_research import SweepTradeabilityRecord

    record = SweepTradeabilityRecord(
        sweep_timestamp="2026-01-02T09:15:00+05:30",
        entry_timestamp="2026-01-02T10:00:00+05:30",
        timeframe="5M",
        sweep_type="Sell Side Sweep",
        trade_direction="bullish",
        sweep_bar=30,
        entry_bar=35,
        entry_price=100.0,
        stop_price=95.0,
        risk_points=5.0,
        opposite_liquidity_target=110.0,
        htf_supply_demand_target=108.0,
        sweep_quality_classification="Strong",
        displacement_strength="Strong",
        choch_present=True,
        bos_present_before_entry=False,
        fvg_reclaimed=False,
        market_location="Near Support",
        hit_1r_before_sl=True,
        hit_2r_before_sl=True,
        hit_3r_before_sl=False,
        hit_opposite_liquidity_before_sl=True,
        hit_htf_supply_demand_before_sl=True,
        realized_pnl_points=10.0,
        realized_rr=2.0,
        win=True,
        stopped_out=False,
        configuration_key="test",
        configuration_label="test",
    )
    metrics = engine._metrics_for_cohort("test", [record])
    assert metrics.trades == 1
    assert metrics.win_rate_pct == 100.0


def test_missing_metadata_raises() -> None:
    with pytest.raises(LiquiditySweepTradeabilityError):
        generate_liquidity_sweep_tradeability_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = LiquiditySweepTradeabilityResearch(timeframes=("5M",)).run(metadata)
    assert report.tradable_trades > 0
