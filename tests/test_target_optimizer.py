"""Tests for SmartMoneyEngine target optimizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.signals.decision_engine import TradeDecision
from src.signals.target_optimizer import (
    OptimizedTarget,
    TargetOptimizer,
    TargetType,
    generate_target_optimizer_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
V2_REPORT = PROJECT_ROOT / "outputs" / "signals" / "trade_plan_report_v2.json"
OPTIMIZER_REPORT = PROJECT_ROOT / "outputs" / "signals" / "target_optimizer_report.json"

BASE_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Swing_High",
    "Swing_Low",
    "HH",
    "HL",
    "LH",
    "LL",
    "Trend",
    "Trend_Strength",
    "Bullish_BOS",
    "Bearish_BOS",
    "Bullish_CHOCH",
    "Bearish_CHOCH",
    "Bullish_FVG_Top",
    "Bullish_FVG_Bottom",
    "Bearish_FVG_Top",
    "Bearish_FVG_Bottom",
    "Equal_High",
    "Equal_Low",
    "Buy_Side_Liquidity",
    "Sell_Side_Liquidity",
    "Buy_Liquidity_Sweep",
    "Sell_Liquidity_Sweep",
    "Liquidity_Strength",
    "Decision",
    "Confidence",
    "Reason",
]


def _empty_row(date: str, close: float) -> dict[str, Any]:
    row = {column: None for column in BASE_COLUMNS}
    row.update(
        {
            "Date": date,
            "Open": close,
            "High": close + 5,
            "Low": close - 5,
            "Close": close,
            "Volume": 1000,
            "Decision": "WAIT",
            "Confidence": 0.0,
            "Reason": "WAIT",
        }
    )
    return row


def _build_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in BASE_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[BASE_COLUMNS]


def _bullish_mtf() -> dict[str, Any]:
    return {
        "overall_bias": "Strong Bullish",
        "timeframes": [
            {"timeframe": "1D", "trend": "Bullish"},
            {"timeframe": "4H", "trend": "Bullish"},
        ],
    }


@pytest.fixture
def optimizer() -> TargetOptimizer:
    return TargetOptimizer()


def test_discovers_nearest_liquidity_buy(optimizer: TargetOptimizer) -> None:
    rows = [_empty_row(f"2026-01-01 09:{15 + idx * 5:02d}:00+05:30", 100.0) for idx in range(15)]
    rows[8].update(
        {
            "Decision": "BUY",
            "Confidence": 0.7,
            "Reason": "Bullish",
            "Trend_Strength": 3,
            "Liquidity_Strength": 0.6,
            "Bullish_BOS": True,
        }
    )
    for idx in range(8, 15):
        rows[idx]["Buy_Side_Liquidity"] = 110.0 + idx

    frame = _build_frame(rows)
    analysis = optimizer.analyze_signal(
        frame,
        index=8,
        decision="BUY",
        entry=100.0,
        stop_loss=95.0,
        signal_date=rows[8]["Date"],
    )

    assert analysis.selected_targets
    types = {item["target_type"] for item in analysis.selected_targets}
    assert TargetType.NEAREST_LIQUIDITY.value in types or TargetType.SWING_HIGH.value in types
    for target in analysis.selected_targets:
        assert target["target_price"] > 100.0
        assert target["expected_rr"] > 0
        assert 5.0 <= target["target_probability"] <= 95.0


def test_discovers_swing_low_sell(optimizer: TargetOptimizer) -> None:
    rows = [_empty_row(f"2026-01-02 09:{15 + idx * 5:02d}:00+05:30", 200.0) for idx in range(15)]
    rows[8].update(
        {
            "Decision": "SELL",
            "Confidence": 0.65,
            "Reason": "Bearish",
            "Trend_Strength": 2,
            "Bearish_BOS": True,
        }
    )
    for idx in range(8, 15):
        rows[idx]["Sell_Side_Liquidity"] = 190.0 - idx
        rows[idx]["Swing_Low"] = 188.0 - idx

    frame = _build_frame(rows)
    analysis = optimizer.analyze_signal(
        frame,
        index=8,
        decision="SELL",
        entry=200.0,
        stop_loss=205.0,
        signal_date=rows[8]["Date"],
    )

    assert analysis.selected_targets
    assert all(item["target_price"] < 200.0 for item in analysis.selected_targets)
    assert analysis.selected_targets[0]["target_price"] >= analysis.selected_targets[-1]["target_price"]


def test_structure_target_buy(optimizer: TargetOptimizer) -> None:
    rows = [_empty_row(f"2026-01-03 09:{15 + idx * 5:02d}:00+05:30", 150.0) for idx in range(12)]
    rows[6]["HH"] = True
    rows[6]["Swing_High"] = 160.0
    rows[8].update(
        {
            "Decision": "BUY",
            "Confidence": 0.6,
            "Reason": "Structure",
            "Bullish_BOS": True,
            "Swing_High": 162.0,
        }
    )
    rows[9]["Buy_Side_Liquidity"] = 165.0

    frame = _build_frame(rows)
    analysis = optimizer.analyze_signal(
        frame,
        index=8,
        decision="BUY",
        entry=150.0,
        stop_loss=145.0,
        signal_date=rows[8]["Date"],
    )

    path_types = {item["target_type"] for item in analysis.target_path}
    assert TargetType.STRUCTURE_TARGET.value in path_types


def test_htf_alignment_boosts_probability(optimizer: TargetOptimizer) -> None:
    optimizer._mtf_report = _bullish_mtf()
    row = pd.Series(
        {
            "Trend_Strength": 2,
            "Liquidity_Strength": 0.5,
            "Bullish_BOS": True,
            "Confidence": 0.6,
        }
    )
    aligned = optimizer._estimate_probability(
        row,
        TradeDecision.BUY.value,
        TargetType.HTF_LIQUIDITY,
        expected_rr=3.0,
    )
    optimizer._mtf_report = {
        "timeframes": [
            {"timeframe": "1D", "trend": "Bearish"},
            {"timeframe": "4H", "trend": "Bearish"},
        ]
    }
    opposed = optimizer._estimate_probability(
        row,
        TradeDecision.BUY.value,
        TargetType.HTF_LIQUIDITY,
        expected_rr=3.0,
    )
    assert aligned > opposed


def test_optimized_target_fields(optimizer: TargetOptimizer) -> None:
    target = OptimizedTarget(
        target_price=110.0,
        target_type=TargetType.NEAREST_LIQUIDITY.value,
        target_probability=72.5,
        expected_rr=2.0,
        reasoning="test",
    )
    payload = target.as_dict()
    for field in (
        "target_price",
        "target_type",
        "target_probability",
        "expected_rr",
        "reasoning",
    ):
        assert field in payload


def test_selects_three_distinct_targets(optimizer: TargetOptimizer) -> None:
    rows = [_empty_row(f"2026-01-04 09:{15 + idx * 5:02d}:00+05:30", 1000.0) for idx in range(20)]
    rows[10].update({"Decision": "BUY", "Confidence": 0.7, "Reason": "Bullish"})
    for idx, price in enumerate([1010, 1020, 1035, 1050, 1075], start=10):
        rows[idx]["Buy_Side_Liquidity"] = price
        rows[idx]["Liquidity_Strength"] = 0.8

    frame = _build_frame(rows)
    analysis = optimizer.analyze_signal(
        frame,
        index=10,
        decision="BUY",
        entry=1000.0,
        stop_loss=990.0,
        signal_date=rows[10]["Date"],
    )

    assert 1 <= len(analysis.selected_targets) <= 3
    prices = [item["target_price"] for item in analysis.selected_targets]
    assert prices == sorted(prices)


@pytest.mark.skipif(
    not PIPELINE_CSV.exists() or not V2_REPORT.exists(),
    reason="Real FYERS pipeline and V2 trade plans required.",
)
def test_real_data_report_generation() -> None:
    report = generate_target_optimizer_report(report_path=OPTIMIZER_REPORT)

    assert OPTIMIZER_REPORT.exists()
    assert report.total_signals >= 1
    assert report.average_target_probability > 0
    assert report.average_expected_rr > 0
    assert len(report.top_targets) >= 1

    with OPTIMIZER_REPORT.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert "signal_analyses" in payload
    assert "top_targets" in payload
    sample = payload["signal_analyses"][0]
    assert "selected_targets" in sample
    assert "target_path" in sample

    for target in payload["top_targets"][:3]:
        assert "target_probability" in target
        assert "expected_rr" in target
        assert "score" in target


@pytest.mark.skipif(
    not PIPELINE_CSV.exists(),
    reason="Real FYERS pipeline required.",
)
def test_real_targets_have_positive_rr() -> None:
    report = generate_target_optimizer_report(report_path=OPTIMIZER_REPORT)
    for analysis in report.signal_analyses:
        for target in analysis["selected_targets"]:
            assert target["expected_rr"] > 0
