"""Tests for the SmartMoneyEngine decision layer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.signals.decision_engine import (
    DecisionEngine,
    DecisionEngineError,
    InstitutionalBias,
    MarketBias,
    TradeDecision,
    evaluate_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
REPORT_JSON = PROJECT_ROOT / "outputs" / "signals" / "decision_report.json"


pytestmark = pytest.mark.skipif(
    not PIPELINE_CSV.exists(),
    reason="Real FYERS pipeline CSV is required for decision engine tests.",
)


@pytest.fixture(scope="module")
def evaluated_frame() -> pd.DataFrame:
    """Evaluate the real NIFTY pipeline once for the module."""
    frame, _ = evaluate_pipeline(
        pipeline_csv=PIPELINE_CSV,
        report_path=REPORT_JSON,
    )
    return frame


@pytest.fixture(scope="module")
def decision_report(evaluated_frame: pd.DataFrame) -> dict:
    """Load the generated decision report JSON."""
    assert REPORT_JSON.exists()
    with REPORT_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_evaluates_real_pipeline_rows(evaluated_frame: pd.DataFrame) -> None:
    assert len(evaluated_frame) == 1617


def test_required_output_columns(evaluated_frame: pd.DataFrame) -> None:
    required = [
        "Decision",
        "Market_Bias",
        "Institutional_Bias",
        "Setup_Quality_Score",
        "Confidence",
        "Reason",
    ]
    for column in required:
        assert column in evaluated_frame.columns


def test_decision_values_are_valid(evaluated_frame: pd.DataFrame) -> None:
    valid = {item.value for item in TradeDecision}
    assert set(evaluated_frame["Decision"].unique()).issubset(valid)


def test_market_bias_values_are_valid(evaluated_frame: pd.DataFrame) -> None:
    valid = {item.value for item in MarketBias}
    assert set(evaluated_frame["Market_Bias"].unique()).issubset(valid)


def test_institutional_bias_values_are_valid(evaluated_frame: pd.DataFrame) -> None:
    valid = {item.value for item in InstitutionalBias}
    assert set(evaluated_frame["Institutional_Bias"].unique()).issubset(valid)


def test_setup_quality_score_range(evaluated_frame: pd.DataFrame) -> None:
    scores = evaluated_frame["Setup_Quality_Score"]
    assert scores.min() >= 0
    assert scores.max() <= 100


def test_confidence_range(evaluated_frame: pd.DataFrame) -> None:
    confidence = evaluated_frame["Confidence"]
    assert confidence.min() >= 0.0
    assert confidence.max() <= 1.0


def test_reason_column_not_empty(evaluated_frame: pd.DataFrame) -> None:
    assert evaluated_frame["Reason"].astype(str).str.len().min() > 0


def test_decision_report_json(decision_report: dict) -> None:
    assert decision_report["rows"] == 1617
    assert decision_report["buy_count"] + decision_report["sell_count"] + decision_report["wait_count"] == 1617
    assert "average_setup_quality" in decision_report
    assert "average_confidence" in decision_report
    assert decision_report["symbol"] == "NIFTY50"


def test_buy_rows_have_bullish_institutional_bias(evaluated_frame: pd.DataFrame) -> None:
    buys = evaluated_frame[evaluated_frame["Decision"] == TradeDecision.BUY.value]
    if buys.empty:
        pytest.skip("No BUY decisions in current real-data run.")
    assert buys["Institutional_Bias"].isin(
        [InstitutionalBias.STRONG_BULLISH.value, InstitutionalBias.WEAK_BULLISH.value]
    ).all()


def test_sell_rows_have_bearish_institutional_bias(evaluated_frame: pd.DataFrame) -> None:
    sells = evaluated_frame[evaluated_frame["Decision"] == TradeDecision.SELL.value]
    if sells.empty:
        pytest.skip("No SELL decisions in current real-data run.")
    assert sells["Institutional_Bias"].isin(
        [InstitutionalBias.STRONG_BEARISH.value, InstitutionalBias.WEAK_BEARISH.value]
    ).all()


def test_missing_columns_raise_error() -> None:
    engine = DecisionEngine()
    with pytest.raises(DecisionEngineError):
        engine.evaluate(pd.DataFrame({"Trend": ["BULLISH"]}))


def test_single_row_bullish_bos_scores_high() -> None:
    engine = DecisionEngine()
    row = pd.Series(
        {
            "Trend": "BULLISH",
            "Trend_Strength": 3,
            "Bullish_BOS": 23400.0,
            "Bearish_BOS": pd.NA,
            "Bullish_CHOCH": pd.NA,
            "Bearish_CHOCH": pd.NA,
            "Bullish_FVG_Top": 23410.0,
            "Bearish_FVG_Top": pd.NA,
            "Bullish_OB_High": 23390.0,
            "Bearish_OB_High": pd.NA,
            "Bullish_OB_Mitigated": False,
            "Bearish_OB_Mitigated": pd.NA,
            "Buy_Liquidity_Sweep": pd.NA,
            "Sell_Liquidity_Sweep": 23380.0,
            "Liquidity_Strength": 2,
        }
    )
    result = engine.evaluate_row(row)
    assert result.market_bias == MarketBias.BULLISH
    assert result.setup_quality_score >= 55
    assert result.decision in {TradeDecision.BUY, TradeDecision.WAIT}
