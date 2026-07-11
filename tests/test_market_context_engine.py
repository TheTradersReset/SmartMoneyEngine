"""Tests for the Market Context Engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.context.market_context_engine import (
    SCORE_WEIGHTS,
    ContextGrade,
    MarketContextEngine,
    MarketContextError,
    SessionLabel,
    generate_context_report,
)
from src.signals.decision_engine import DecisionEngine, TradeDecision

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
                "High": price + 0.6,
                "Low": price - 0.4,
                "Close": price + 0.1,
                "Volume": 1000 + index,
                "Swing_High": pd.NA,
                "Swing_Low": price - 1.0 if index % 20 == 0 else pd.NA,
                "Equal_High": pd.NA,
                "Equal_Low": pd.NA,
                "Trend": "BULLISH",
                "Trend_Strength": 2,
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bullish_FVG_Bottom": pd.NA,
                "Bearish_FVG_Top": pd.NA,
                "Bearish_FVG_Bottom": pd.NA,
                "Bullish_OB_High": pd.NA,
                "Bullish_OB_Low": pd.NA,
                "Bearish_OB_High": pd.NA,
                "Bearish_OB_Low": pd.NA,
                "Bullish_OB_Mitigated": pd.NA,
                "Bearish_OB_Mitigated": pd.NA,
                "Buy_Side_Liquidity": pd.NA,
                "Sell_Side_Liquidity": pd.NA,
                "Buy_Liquidity_Sweep": pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
                "Liquidity_Strength": 2,
            }
        )
    return pd.DataFrame(rows)


def test_score_weights_total_100() -> None:
    assert sum(SCORE_WEIGHTS.values()) == 100


def test_session_label_mid_session() -> None:
    ts = pd.Timestamp("2026-01-02 12:00:00+05:30")
    assert MarketContextEngine._session_label(ts) == SessionLabel.MID


def test_session_label_opening_hour() -> None:
    ts = pd.Timestamp("2026-01-02 09:30:00+05:30")
    assert MarketContextEngine._session_label(ts) == SessionLabel.OPENING


def test_assign_grade_thresholds() -> None:
    assert MarketContextEngine._assign_grade(92) == ContextGrade.A_PLUS
    assert MarketContextEngine._assign_grade(84) == ContextGrade.A
    assert MarketContextEngine._assign_grade(75) == ContextGrade.B
    assert MarketContextEngine._assign_grade(62) == ContextGrade.C
    assert MarketContextEngine._assign_grade(40) == ContextGrade.REJECT


def test_compute_atr_series() -> None:
    frame = _pipeline_frame(30)
    atr = MarketContextEngine._compute_atr(frame)
    assert len(atr) == 30
    assert float(atr.iloc[-1]) > 0


def test_market_levels_identify_support() -> None:
    frame = _pipeline_frame(50)
    frame.loc[40, "Swing_Low"] = 105.0
    frame.loc[49, "Close"] = 106.0
    engine = MarketContextEngine()
    levels = engine._market_levels(frame, 49)
    assert levels["major_support"] is not None
    assert levels["distance_to_support"] is not None


def test_evaluate_buy_signal_context() -> None:
    frame = _pipeline_frame(80)
    frame.loc[60, "Sell_Liquidity_Sweep"] = 118.0
    frame.loc[60, "Bullish_BOS"] = 121.0
    evaluated = DecisionEngine().evaluate(frame)
    evaluated.loc[60, "Decision"] = TradeDecision.BUY.value

    engine = MarketContextEngine()
    working = evaluated.reset_index(drop=True)
    htf_lookup = engine._build_htf_trend_lookup(working)
    atr_series = engine._compute_atr(working)
    context = engine.evaluate_signal(working, 60, htf_lookup, atr_series)

    assert context is not None
    assert context.decision == TradeDecision.BUY.value
    assert 0 <= context.context_score <= 100
    assert context.context_grade in {grade.value for grade in ContextGrade}
    assert context.reasoning
    assert "1D" in context.multi_timeframe


def test_run_skips_wait_signals() -> None:
    frame = _pipeline_frame(40)
    evaluated = DecisionEngine().evaluate(frame)
    engine = MarketContextEngine()
    report = engine.run(evaluated)
    assert report.total_signals == int((evaluated["Decision"] != TradeDecision.WAIT.value).sum())


def test_run_rejects_empty_frame() -> None:
    engine = MarketContextEngine()
    with pytest.raises(MarketContextError):
        engine.run(pd.DataFrame())


def test_generate_context_report_writes_json(tmp_path: Path) -> None:
    frame = _pipeline_frame(80)
    frame.loc[60, "Decision"] = TradeDecision.BUY.value
    frame.loc[60, "Market_Bias"] = "Bullish"
    frame.loc[60, "Institutional_Bias"] = "Strong Bullish"
    frame.loc[60, "Setup_Quality_Score"] = 80
    frame.loc[60, "Confidence"] = 0.8
    frame.loc[60, "Reason"] = "Test"
    frame.loc[60, "Bullish_Score"] = 80
    frame.loc[60, "Bearish_Score"] = 10

    csv_path = tmp_path / "pipeline.csv"
    report_path = tmp_path / "context_report.json"
    frame.to_csv(csv_path, index=False)

    report = generate_context_report(
        pipeline_csv=csv_path,
        report_path=report_path,
        symbol="NIFTY50",
        timeframe="5",
    )

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "NIFTY50"
    assert "signals" in payload
    assert payload["total_candles"] == 80
    assert report.total_signals >= 1


@pytest.mark.integration
def test_real_pipeline_context_if_present() -> None:
    pipeline_csv = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
    if not pipeline_csv.exists():
        pytest.skip("Real pipeline CSV not available.")

    engine = MarketContextEngine(symbol="NIFTY50", timeframe="5")
    report = engine.run_from_csv(pipeline_csv)

    assert report.total_candles > 1000
    assert report.total_signals > 0
    assert 0 <= report.average_context_score <= 100
    assert report.signals[0]["context_score"] >= 0
