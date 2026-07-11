"""Tests for the SmartMoneyEngine trade plan engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.signals.decision_engine import DecisionEngine, TradeDecision, evaluate_pipeline
from src.signals.trade_plan_engine import (
    TradePlanEngine,
    TradePlanEngineError,
    generate_trade_plans,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
TRADE_PLAN_REPORT = PROJECT_ROOT / "outputs" / "signals" / "trade_plan_report.json"


pytestmark = pytest.mark.skipif(
    not PIPELINE_CSV.exists(),
    reason="Real FYERS pipeline CSV is required for trade plan tests.",
)


@pytest.fixture(scope="module")
def evaluated_pipeline() -> pd.DataFrame:
    frame, _ = evaluate_pipeline()
    return frame


@pytest.fixture(scope="module")
def trade_plans_and_report(evaluated_pipeline: pd.DataFrame) -> tuple[list, dict]:
    _, report = generate_trade_plans(report_path=TRADE_PLAN_REPORT)
    with TRADE_PLAN_REPORT.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    engine = TradePlanEngine()
    plans = engine.build_plans(evaluated_pipeline)
    return plans, payload


def test_generates_plans_for_real_signals(trade_plans_and_report: tuple[list, dict]) -> None:
    plans, report = trade_plans_and_report
    assert report["total_signals"] == len(plans)
    assert report["total_signals"] >= 1


def test_trade_plan_report_json(trade_plans_and_report: tuple[list, dict]) -> None:
    _, report = trade_plans_and_report
    assert TRADE_PLAN_REPORT.exists()
    assert report["total_signals"] == report["valid_trade_plans"] + report["invalid_trade_plans"]
    assert "average_rr" in report
    assert "average_confidence" in report
    assert "trade_plans" in report
    assert len(report["trade_plans"]) == report["total_signals"]


def test_plan_fields_present(trade_plans_and_report: tuple[list, dict]) -> None:
    plans, _ = trade_plans_and_report
    required = {
        "entry_price",
        "stop_loss",
        "target_1",
        "target_2",
        "risk_reward_ratio",
        "confidence_pct",
        "trade_validity",
        "reason",
    }
    for plan in plans:
        payload = plan.as_dict()
        for field in required:
            assert field in payload


def test_buy_plan_price_structure(trade_plans_and_report: tuple[list, dict]) -> None:
    plans, _ = trade_plans_and_report
    buys = [plan for plan in plans if plan.decision == TradeDecision.BUY.value]
    if not buys:
        pytest.skip("No BUY plans in current real-data run.")
    for plan in buys:
        assert plan.stop_loss < plan.entry_price
        assert plan.target_1 > plan.entry_price
        assert plan.target_2 >= plan.target_1


def test_sell_plan_price_structure(trade_plans_and_report: tuple[list, dict]) -> None:
    plans, _ = trade_plans_and_report
    sells = [plan for plan in plans if plan.decision == TradeDecision.SELL.value]
    if not sells:
        pytest.skip("No SELL plans in current real-data run.")
    for plan in sells:
        assert plan.stop_loss > plan.entry_price
        assert plan.target_1 < plan.entry_price
        assert plan.target_2 <= plan.target_1


def test_valid_plans_meet_minimum_rr(trade_plans_and_report: tuple[list, dict]) -> None:
    plans, _ = trade_plans_and_report
    valid = [plan for plan in plans if plan.trade_validity]
    if not valid:
        pytest.skip("No valid trade plans in current real-data run.")
    for plan in valid:
        assert plan.risk_reward_ratio >= TradePlanEngine().min_risk_reward


def test_confidence_percent_range(trade_plans_and_report: tuple[list, dict]) -> None:
    plans, _ = trade_plans_and_report
    for plan in plans:
        assert 0.0 <= plan.confidence_pct <= 100.0


def test_missing_decision_columns_raise_error() -> None:
    engine = TradePlanEngine()
    with pytest.raises(TradePlanEngineError):
        engine.build_plans(pd.DataFrame({"Close": [100.0]}))


def test_synthetic_buy_plan_uses_ob_zone() -> None:
    engine = TradePlanEngine()
    frame = pd.DataFrame(
        {
            "Date": ["2026-01-01", "2026-01-02"],
            "Close": [100.0, 101.0],
            "Decision": ["WAIT", "BUY"],
            "Confidence": [0.0, 0.7],
            "Reason": ["", "test"],
            "Bullish_OB_Low": [99.0, pd.NA],
            "Bullish_OB_High": [100.5, pd.NA],
            "Bullish_OB_Mitigated": [False, pd.NA],
            "Bearish_OB_Low": [pd.NA, pd.NA],
            "Bearish_OB_High": [pd.NA, pd.NA],
            "Bearish_OB_Mitigated": [pd.NA, pd.NA],
            "Bullish_FVG_Bottom": [pd.NA, pd.NA],
            "Bullish_FVG_Top": [pd.NA, pd.NA],
            "Bearish_FVG_Bottom": [pd.NA, pd.NA],
            "Bearish_FVG_Top": [pd.NA, pd.NA],
            "Swing_Low": [98.5, pd.NA],
            "Swing_High": [pd.NA, pd.NA],
            "Buy_Side_Liquidity": [102.0, 102.0],
            "Sell_Side_Liquidity": [97.0, 97.0],
        }
    )
    plans = engine.build_plans(frame)
    assert len(plans) == 1
    assert plans[0].entry_price == 100.5
    assert plans[0].stop_loss < plans[0].entry_price
