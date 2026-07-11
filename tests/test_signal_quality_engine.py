"""Tests for SmartMoneyEngine signal quality engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.signals.decision_engine import TradeDecision
from src.signals.signal_quality_engine import (
    SignalGrade,
    SignalQualityEngine,
    generate_signal_quality_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
V2_REPORT = PROJECT_ROOT / "outputs" / "signals" / "trade_plan_report_v2.json"
QUALITY_REPORT = PROJECT_ROOT / "outputs" / "signals" / "signal_quality_report.json"

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
    "Bullish_OB_High",
    "Bullish_OB_Low",
    "Bearish_OB_High",
    "Bearish_OB_Low",
    "Bullish_OB_Mitigated",
    "Bearish_OB_Mitigated",
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


def _bearish_mtf() -> dict[str, Any]:
    return {
        "overall_bias": "Strong Bearish",
        "timeframes": [
            {"timeframe": "1D", "trend": "Bearish"},
            {"timeframe": "4H", "trend": "Bearish"},
        ],
    }


def _plan(
    signal_index: int,
    signal_date: str,
    decision: str,
    grade: str = "A+",
    rr_t2: float = 2.0,
) -> dict[str, Any]:
    return {
        "signal_index": signal_index,
        "signal_date": signal_date,
        "decision": decision,
        "trade_grade": grade,
        "risk_reward_t2": rr_t2,
        "trade_validity": True,
    }


@pytest.fixture
def engine() -> SignalQualityEngine:
    return SignalQualityEngine()


def test_a_plus_bullish_signal(engine: SignalQualityEngine) -> None:
    rows = [_empty_row(f"2026-01-01 09:{15 + idx * 5:02d}:00+05:30", 100.0) for idx in range(15)]
    rows[10].update(
        {
            "Decision": "BUY",
            "Trend": "Bullish",
            "Trend_Strength": 3,
            "HH": True,
            "HL": True,
            "Bullish_BOS": True,
            "Bullish_CHOCH": True,
            "Sell_Liquidity_Sweep": True,
            "Bullish_FVG_Top": 102.0,
            "Bullish_FVG_Bottom": 100.5,
        }
    )

    frame = _build_frame(rows)
    engine._mtf_report = _bullish_mtf()
    quality = engine.evaluate_signal(
        frame,
        index=10,
        decision="BUY",
        signal_date=rows[10]["Date"],
        plan=_plan(10, rows[10]["Date"], "BUY"),
    )

    assert quality.quality_score >= 85
    assert quality.grade == SignalGrade.A_PLUS.value
    assert "Trend Bullish" in quality.reasoning
    assert "HTF aligned (1D + 4H)" in quality.reasoning
    assert "Bullish BOS" in quality.reasoning
    assert "Sell-side liquidity sweep" in quality.reasoning
    assert "Fresh FVG" in quality.reasoning


def test_bearish_signal_with_htf_misalignment(engine: SignalQualityEngine) -> None:
    rows = [_empty_row(f"2026-01-02 09:{15 + idx * 5:02d}:00+05:30", 200.0) for idx in range(12)]
    rows[8].update(
        {
            "Decision": "SELL",
            "Trend": "Bearish",
            "Trend_Strength": 2,
            "LH": True,
            "LL": True,
            "Bearish_BOS": True,
            "Bearish_FVG_Top": 199.0,
            "Bearish_FVG_Bottom": 197.5,
        }
    )

    frame = _build_frame(rows)
    engine._mtf_report = _bullish_mtf()
    quality = engine.evaluate_signal(
        frame,
        index=8,
        decision="SELL",
        signal_date=rows[8]["Date"],
        plan=_plan(8, rows[8]["Date"], "SELL", grade="B", rr_t2=2.0),
    )

    assert quality.grade in {SignalGrade.B.value, SignalGrade.C.value, SignalGrade.A.value}
    assert "HTF misaligned" in quality.reasoning
    assert quality.factors["htf_alignment"] == 0.0


def test_weak_signal_rejected(engine: SignalQualityEngine) -> None:
    rows = [_empty_row("2026-01-03 09:15:00+05:30", 150.0)]
    rows[0]["Decision"] = "BUY"
    rows[0]["Trend"] = "Neutral"

    frame = _build_frame(rows)
    engine._mtf_report = {"timeframes": []}
    quality = engine.evaluate_signal(
        frame,
        index=0,
        decision="BUY",
        signal_date=rows[0]["Date"],
        plan={
            "signal_index": 0,
            "trade_validity": False,
            "trade_grade": "Reject",
            "risk_reward_t2": 0.5,
        },
    )

    assert quality.grade == SignalGrade.REJECT.value
    assert quality.quality_score < 85


def test_score_within_bounds(engine: SignalQualityEngine) -> None:
    rows = [_empty_row(f"2026-01-04 09:{15 + idx * 5:02d}:00+05:30", 1000.0) for idx in range(10)]
    rows[5].update({"Decision": "BUY", "Trend": "Bullish", "Trend_Strength": 1})

    frame = _build_frame(rows)
    engine._mtf_report = _bullish_mtf()
    quality = engine.evaluate_signal(
        frame,
        index=5,
        decision="BUY",
        signal_date=rows[5]["Date"],
        plan=_plan(5, rows[5]["Date"], "BUY"),
    )

    assert 0.0 <= quality.quality_score <= 100.0
    assert sum(quality.factors.values()) <= 100.0


def test_grade_assignment_thresholds(engine: SignalQualityEngine) -> None:
    assert engine._assign_grade(90, True) == SignalGrade.A_PLUS
    assert engine._assign_grade(75, True) == SignalGrade.A
    assert engine._assign_grade(60, True) == SignalGrade.B
    assert engine._assign_grade(45, True) == SignalGrade.C
    assert engine._assign_grade(80, False) == SignalGrade.REJECT


def test_evaluate_ranks_top_and_bottom(engine: SignalQualityEngine) -> None:
    rows = [_empty_row(f"2026-01-05 09:{15 + idx * 5:02d}:00+05:30", 100.0) for idx in range(20)]
    rows[5].update({"Decision": "BUY", "Trend": "Bullish", "Trend_Strength": 1})
    rows[10].update(
        {
            "Decision": "BUY",
            "Trend": "Bullish",
            "Trend_Strength": 3,
            "Bullish_BOS": True,
            "Sell_Liquidity_Sweep": True,
            "HH": True,
            "HL": True,
            "Bullish_FVG_Top": 102.0,
            "Bullish_FVG_Bottom": 100.5,
        }
    )

    frame = _build_frame(rows)
    engine._mtf_report = _bullish_mtf()
    report = engine.evaluate(
        frame,
        trade_plans=[
            _plan(5, rows[5]["Date"], "BUY", grade="B"),
            _plan(10, rows[10]["Date"], "BUY", grade="A+"),
        ],
        mtf_report=_bullish_mtf(),
    )

    assert report.total_signals == 2
    assert report.top_signals[0]["quality_score"] >= report.bottom_signals[0]["quality_score"]


@pytest.mark.skipif(
    not PIPELINE_CSV.exists() or not V2_REPORT.exists(),
    reason="Real FYERS pipeline and V2 trade plans required.",
)
def test_real_data_report_generation() -> None:
    report = generate_signal_quality_report(report_path=QUALITY_REPORT)

    assert QUALITY_REPORT.exists()
    assert report.total_signals >= 1
    assert 0.0 <= report.average_score <= 100.0
    assert sum(report.grade_distribution.values()) == report.total_signals
    assert len(report.top_signals) >= 1
    assert len(report.bottom_signals) >= 1

    with QUALITY_REPORT.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    sample = payload["signals"][0]
    for field in ("quality_score", "grade", "factors", "reasoning"):
        assert field in sample

    for factor in (
        "htf_alignment",
        "trend_strength",
        "bos_quality",
        "choch_quality",
        "liquidity_sweep",
        "fresh_fvg",
        "fresh_order_block",
        "structure_quality",
    ):
        assert factor in sample["factors"]
