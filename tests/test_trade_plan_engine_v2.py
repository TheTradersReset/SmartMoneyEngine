"""Tests for SmartMoneyEngine trade plan engine V2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.backtesting.backtest_engine import BacktestEngine
from src.signals.decision_engine import TradeDecision
from src.signals.trade_plan_engine_v2 import (
    TradeGrade,
    TradePlanEngineV2,
    TradePlanEngineV2Error,
    generate_trade_plans_v2,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
MTF_REPORT = PROJECT_ROOT / "outputs" / "signals" / "multi_timeframe_report.json"
V2_REPORT = PROJECT_ROOT / "outputs" / "signals" / "trade_plan_report_v2.json"

BASE_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Swing_High",
    "Swing_Low",
    "Bullish_OB_High",
    "Bullish_OB_Low",
    "Bearish_OB_High",
    "Bearish_OB_Low",
    "Bullish_OB_Mitigated",
    "Bearish_OB_Mitigated",
    "Bullish_FVG_Top",
    "Bullish_FVG_Bottom",
    "Bearish_FVG_Top",
    "Bearish_FVG_Bottom",
    "Buy_Side_Liquidity",
    "Sell_Side_Liquidity",
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


def _build_synthetic_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in BASE_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[BASE_COLUMNS]


def _bullish_mtf() -> dict[str, Any]:
    return {
        "overall_bias": "Strong Bullish",
        "alignment_score": 100,
        "timeframes": [
            {"timeframe": "1D", "trend": "Bullish"},
            {"timeframe": "4H", "trend": "Bullish"},
            {"timeframe": "1H", "trend": "Bullish"},
            {"timeframe": "15M", "trend": "Bullish"},
            {"timeframe": "5M", "trend": "Bullish"},
        ],
    }


def _bearish_mtf() -> dict[str, Any]:
    return {
        "overall_bias": "Strong Bearish",
        "alignment_score": 100,
        "timeframes": [
            {"timeframe": "1D", "trend": "Bearish"},
            {"timeframe": "4H", "trend": "Bearish"},
            {"timeframe": "1H", "trend": "Bearish"},
            {"timeframe": "15M", "trend": "Bearish"},
            {"timeframe": "5M", "trend": "Bearish"},
        ],
    }


@pytest.fixture
def engine() -> TradePlanEngineV2:
    return TradePlanEngineV2(sl_buffer_points=5.0, min_rr_t2=1.5)


def test_bullish_trade_plan(engine: TradePlanEngineV2) -> None:
    rows = [_empty_row(f"2026-01-01 09:{15 + idx * 5:02d}:00+05:30", 100.0) for idx in range(25)]
    signal_idx = 20
    rows[signal_idx].update(
        {
            "Decision": "BUY",
            "Confidence": 0.72,
            "Reason": "Synthetic bullish signal",
            "Close": 100.0,
            "Swing_Low": 95.0,
            "Bullish_OB_Low": 98.0,
            "Bullish_OB_High": 102.0,
            "Liquidity_Strength": 0.8,
        }
    )
    for idx in range(signal_idx, 25):
        rows[idx]["Buy_Side_Liquidity"] = 110.0 + idx
        rows[idx]["Swing_High"] = 112.0 + idx

    frame = _build_synthetic_frame(rows)
    plan = engine.build_plans(frame, mtf_report=_bullish_mtf())[0]

    assert plan.decision == TradeDecision.BUY.value
    assert plan.stop_loss < plan.entry < plan.target_1 < plan.target_2 < plan.target_3
    assert plan.risk_reward_t1 >= 1.0
    assert plan.risk_reward_t2 >= 1.5
    assert plan.risk_reward_t3 >= 2.0
    assert plan.trade_validity is True
    assert plan.trade_grade in {TradeGrade.A_PLUS.value, TradeGrade.A.value, TradeGrade.B.value}


def test_bearish_trade_plan(engine: TradePlanEngineV2) -> None:
    rows = [_empty_row(f"2026-01-02 09:{15 + idx * 5:02d}:00+05:30", 200.0) for idx in range(25)]
    signal_idx = 20
    rows[signal_idx].update(
        {
            "Decision": "SELL",
            "Confidence": 0.68,
            "Reason": "Synthetic bearish signal",
            "Close": 200.0,
            "Swing_High": 205.0,
            "Bearish_OB_Low": 198.0,
            "Bearish_OB_High": 202.0,
            "Liquidity_Strength": 0.75,
        }
    )
    for idx in range(signal_idx, 25):
        rows[idx]["Sell_Side_Liquidity"] = 190.0 - idx
        rows[idx]["Swing_Low"] = 188.0 - idx

    frame = _build_synthetic_frame(rows)
    plan = engine.build_plans(frame, mtf_report=_bearish_mtf())[0]

    assert plan.decision == TradeDecision.SELL.value
    assert plan.target_3 < plan.target_2 < plan.target_1 < plan.entry < plan.stop_loss
    assert plan.risk_reward_t2 >= 1.5
    assert plan.trade_validity is True


def test_invalid_rr_rejected(engine: TradePlanEngineV2) -> None:
    valid, reason = engine._validate_buy_plan(
        entry=100.0,
        stop_loss=98.0,
        target_1=101.0,
        target_2=101.5,
        target_3=110.0,
        rr_t2=0.75,
    )
    assert valid is False
    assert reason is not None
    assert "1.5" in reason

    strict_engine = TradePlanEngineV2(min_rr_t2=3.0)
    rows = [_empty_row(f"2026-01-03 09:{15 + idx * 5:02d}:00+05:30", 1000.0) for idx in range(10)]
    rows[5].update(
        {
            "Decision": "BUY",
            "Confidence": 0.6,
            "Reason": "Strict RR gate",
            "Close": 1000.0,
            "Swing_Low": 999.0,
        }
    )
    plan = strict_engine.build_plans(_build_synthetic_frame(rows), mtf_report=_bullish_mtf())[0]
    assert plan.risk_reward_t2 < 3.0
    assert plan.trade_validity is False
    assert plan.trade_grade == TradeGrade.REJECT.value


def test_invalid_target_hierarchy_rejected(engine: TradePlanEngineV2) -> None:
    plan = engine._validate_buy_plan(
        entry=100.0,
        stop_loss=95.0,
        target_1=110.0,
        target_2=108.0,
        target_3=115.0,
        rr_t2=1.6,
    )
    valid, reason = plan
    assert valid is False
    assert reason is not None
    assert "hierarchy" in reason.lower()


def test_htf_alignment_scoring(engine: TradePlanEngineV2) -> None:
    engine._mtf_report = _bullish_mtf()
    bullish_score, bullish_note = engine._htf_alignment_score(TradeDecision.BUY.value)

    engine._mtf_report = _bearish_mtf()
    misaligned_score, misaligned_note = engine._htf_alignment_score(TradeDecision.BUY.value)

    assert bullish_score > misaligned_score
    assert "aligned" in bullish_note.lower() or "partial" in bullish_note.lower()
    assert "misaligned" in misaligned_note.lower()


def test_grade_assignment(engine: TradePlanEngineV2) -> None:
    assert engine._assign_grade(90, True) == TradeGrade.A_PLUS
    assert engine._assign_grade(75, True) == TradeGrade.A
    assert engine._assign_grade(60, True) == TradeGrade.B
    assert engine._assign_grade(45, True) == TradeGrade.C
    assert engine._assign_grade(80, False) == TradeGrade.REJECT
    assert engine._assign_grade(30, True) == TradeGrade.REJECT


def test_v2_fields_and_backtest_aliases(engine: TradePlanEngineV2) -> None:
    rows = [_empty_row(f"2026-01-04 09:{15 + idx * 5:02d}:00+05:30", 150.0) for idx in range(15)]
    rows[10].update(
        {
            "Decision": "BUY",
            "Confidence": 0.65,
            "Reason": "Alias test",
            "Swing_Low": 145.0,
            "Bullish_FVG_Bottom": 148.0,
            "Bullish_FVG_Top": 152.0,
        }
    )
    for idx in range(10, 15):
        rows[idx]["Buy_Side_Liquidity"] = 165.0 + idx

    plan = engine.build_plans(_build_synthetic_frame(rows), mtf_report=_bullish_mtf())[0]
    payload = plan.as_dict()

    for field in (
        "entry",
        "stop_loss",
        "target_1",
        "target_2",
        "target_3",
        "risk_reward_t1",
        "risk_reward_t2",
        "risk_reward_t3",
        "trade_grade",
        "confidence",
        "reason",
    ):
        assert field in payload

    assert payload["entry_price"] == payload["entry"]
    assert payload["risk_reward_ratio"] == payload["risk_reward_t2"]
    assert payload["confidence_pct"] == payload["confidence"]


def test_backtest_engine_accepts_v2_json(engine: TradePlanEngineV2, tmp_path: Path) -> None:
    rows = [_empty_row(f"2026-01-05 09:{15 + idx * 5:02d}:00+05:30", 23000.0) for idx in range(15)]
    rows[8].update(
        {
            "Decision": "BUY",
            "Confidence": 0.7,
            "Reason": "Backtest compatibility",
            "Swing_Low": 22950.0,
            "Bullish_OB_Low": 22980.0,
            "Bullish_OB_High": 23020.0,
        }
    )
    for idx in range(8, 15):
        rows[idx]["Buy_Side_Liquidity"] = 23100.0 + idx * 20

    plan = engine.build_plans(_build_synthetic_frame(rows), mtf_report=_bullish_mtf())[0]
    report_payload = {
        "symbol": "NIFTY50",
        "timeframe": "5",
        "trade_plans": [plan.as_dict()],
    }
    report_path = tmp_path / "trade_plan_report_v2.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    ohlcv_rows = []
    base_ts = pd.Timestamp("2026-01-05 09:15:00", tz="Asia/Kolkata")
    for idx in range(20):
        ts = base_ts + pd.Timedelta(minutes=5 * idx)
        price = 23000.0 + idx
        ohlcv_rows.append(
            {
                "timestamp": ts,
                "open": price,
                "high": price + 30,
                "low": price - 10,
                "close": price + 5,
                "volume": 1000,
            }
        )
    ohlcv = pd.DataFrame(ohlcv_rows)

    backtest = BacktestEngine(symbol="NIFTY50", timeframe="5")
    result = backtest.run(trade_plan_report=report_path, ohlcv=ohlcv)

    assert result.total_trade_plans == 1
    assert "entry_price" in report_payload["trade_plans"][0]
    assert result.trade_results[0]["entry_hit"] is True


def test_missing_decision_columns_raises_error(engine: TradePlanEngineV2) -> None:
    frame = pd.DataFrame({"Close": [100.0], "Decision": ["BUY"]})
    with pytest.raises(TradePlanEngineV2Error):
        engine.build_plans(frame, mtf_report=_bullish_mtf())


@pytest.mark.skipif(
    not PIPELINE_CSV.exists() or not MTF_REPORT.exists(),
    reason="Real FYERS pipeline and MTF report required.",
)
def test_real_data_v2_improves_rr_distribution() -> None:
    _, report = generate_trade_plans_v2(report_path=V2_REPORT)

    assert V2_REPORT.exists()
    assert report.engine_version == "v2"
    assert report.total_signals >= 1

    valid = [plan for plan in report.trade_plans if plan["trade_validity"]]
    if valid:
        assert report.average_rr_t2 >= 1.5
        for plan in valid:
            assert plan["risk_reward_t2"] >= 1.5
            assert plan["target_1"] != plan["target_2"]

    with V2_REPORT.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    sample = payload["trade_plans"][0]
    assert "target_3" in sample
    assert "trade_grade" in sample
    assert "entry_price" in sample
