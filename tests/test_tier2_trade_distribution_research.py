"""Tests for Tier-2 trade distribution research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.tier2_trade_distribution_research import (
    Tier2TradeDetail,
    Tier2TradeDistributionError,
    Tier2TradeDistributionResearch,
    generate_tier2_trade_distribution_report,
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
                "Bearish_FVG_Top": 99.5 if index == 49 else pd.NA,
                "Bearish_FVG_Bottom": 98.5 if index == 49 else pd.NA,
                "Buy_Side_Liquidity": pd.NA,
                "Sell_Side_Liquidity": pd.NA,
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
    rows[50]["Open"] = 99.0
    rows[50]["Close"] = 97.0
    rows[50]["High"] = 99.5
    rows[50]["Low"] = 96.5
    return pd.DataFrame(rows)


def test_bucket_distribution() -> None:
    engine = Tier2TradeDistributionResearch()
    from src.research.tier2_trade_distribution_research import WINNER_BUCKETS

    dist = engine._distribution([25, 75, 125, 225], WINNER_BUCKETS)
    assert dist["0-50"]["count"] == 1
    assert dist["200+"]["count"] == 1


def test_simulate_detailed() -> None:
    engine = Tier2TradeDistributionResearch()
    frame = _frame()
    signal = TierSignal(
        tier="tier_2",
        timeframe="5M",
        direction="bearish",
        bos_bar=50,
        bos_timestamp=str(frame.iloc[50]["Date"]),
    )
    detail = engine._simulate_detailed(frame, signal)
    assert detail is not None
    assert detail.mfe_points >= 0
    assert detail.mae_points >= 0


def test_streak_stats() -> None:
    stats = Tier2TradeDistributionResearch._streak_stats([True, True, False, False, False, True])
    assert stats["max_win_streak"] == 2
    assert stats["max_loss_streak"] == 3


def test_consecutive_loss_probability() -> None:
    wins = [False, False, False, True, False, False, False, False, False]
    prob = Tier2TradeDistributionResearch._consecutive_loss_probability(wins, 3)
    assert prob > 0


def test_equity_curve_and_drawdown() -> None:
    trades = [
        Tier2TradeDetail(
            timeframe="5M",
            direction="bearish",
            bos_timestamp="2026-01-02T09:15:00+05:30",
            risk_points=10.0,
            mfe_points=50.0,
            mae_points=5.0,
            realized_pnl_points=50.0,
            win=True,
            bars_to_target=3,
            bars_to_stop=None,
            minutes_to_target=15.0,
            minutes_to_stop=None,
        ),
        Tier2TradeDetail(
            timeframe="5M",
            direction="bearish",
            bos_timestamp="2026-01-02T10:15:00+05:30",
            risk_points=10.0,
            mfe_points=0.0,
            mae_points=12.0,
            realized_pnl_points=-10.0,
            win=False,
            bars_to_target=None,
            bars_to_stop=2,
            minutes_to_target=None,
            minutes_to_stop=10.0,
        ),
    ]
    curve = Tier2TradeDistributionResearch._equity_curve(trades)
    assert curve[-1]["cumulative_pnl_points"] == 40.0
    worst, best, peak = Tier2TradeDistributionResearch._drawdown_metrics(
        [point["cumulative_pnl_points"] for point in curve]
    )
    assert worst == 10.0
    assert peak == 50.0


def test_monthly_breakdown() -> None:
    engine = Tier2TradeDistributionResearch()
    trades = [
        Tier2TradeDetail(
            timeframe="5M",
            direction="bearish",
            bos_timestamp="2026-01-02T09:15:00+05:30",
            risk_points=10.0,
            mfe_points=50.0,
            mae_points=5.0,
            realized_pnl_points=50.0,
            win=True,
            bars_to_target=3,
            bars_to_stop=None,
            minutes_to_target=15.0,
            minutes_to_stop=None,
        )
    ]
    rows = engine._monthly_breakdown(trades)
    assert rows[0]["month"] == "2026-01"
    assert rows[0]["signals"] == 1


def test_missing_metadata_raises() -> None:
    with pytest.raises(Tier2TradeDistributionError):
        generate_tier2_trade_distribution_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = Tier2TradeDistributionResearch(timeframes=("5M",)).run(metadata)
    assert report.total_signals >= 0
