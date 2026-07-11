"""Tests for trigger-to-trade validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.trigger_trade_validation_research import (
    DEFAULT_TRIGGER_REPORT_PATH,
    TriggerTradeValidationError,
    TriggerTradeValidationResearch,
    TriggerTradeOutcome,
    generate_trigger_trade_validation_report,
)


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
                "High": price + 1.0,
                "Low": price - 0.8,
                "Close": price + 0.2,
                "Volume": 100000,
                "Buy_Side_Liquidity": price + 5,
                "Sell_Side_Liquidity": price - 5,
            }
        )
    return pd.DataFrame(rows)


def test_classify_trigger_production_ready() -> None:
    result = TriggerTradeValidationResearch._classify_trigger(
        trades=60,
        win_rate_pct=45.0,
        expectancy=80.0,
        profit_factor=2.0,
    )
    assert result == "Production Ready"


def test_classify_trigger_reject_negative_exp() -> None:
    result = TriggerTradeValidationResearch._classify_trigger(
        trades=60,
        win_rate_pct=45.0,
        expectancy=-10.0,
        profit_factor=0.8,
    )
    assert result == "Reject"


def test_aggregate_metrics_groups_by_model() -> None:
    engine = TriggerTradeValidationResearch()
    outcomes = [
        TriggerTradeOutcome(
            symbol="NIFTY50",
            timeframe="5M",
            direction="bullish",
            trigger_model="Model A",
            trigger_timestamp="2026-01-02 09:15:00+05:30",
            trigger_bar=10,
            entry_price=100.0,
            stop_price=98.0,
            target_price=104.0,
            risk_points=2.0,
            realized_pnl_points=4.0,
            realized_rr=2.0,
            win=True,
            holding_bars=3,
            is_false_trigger=False,
        ),
        TriggerTradeOutcome(
            symbol="NIFTY50",
            timeframe="5M",
            direction="bullish",
            trigger_model="Model A",
            trigger_timestamp="2026-01-02 10:15:00+05:30",
            trigger_bar=20,
            entry_price=101.0,
            stop_price=99.0,
            target_price=105.0,
            risk_points=2.0,
            realized_pnl_points=-2.0,
            realized_rr=-1.0,
            win=False,
            holding_bars=2,
            is_false_trigger=False,
        ),
    ]
    # Need 5+ for aggregation — duplicate to meet threshold
    outcomes = outcomes * 3
    metrics = engine._aggregate_metrics(outcomes)
    assert len(metrics) == 1
    assert metrics[0].trades == 6
    assert metrics[0].win_rate_pct == pytest.approx(50.0)


def test_simulate_trigger_trade_bullish() -> None:
    engine = TriggerTradeValidationResearch()
    frame = _pipeline_frame()
    record = {
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "direction": "bullish",
        "trigger_model": "Test Model",
        "trigger_timestamp": frame.iloc[50]["Date"],
        "trigger_bar": 50,
        "is_false_trigger": False,
    }
    outcome = engine._simulate_trigger_trade(frame, record)
    assert outcome is not None
    assert outcome.direction == "bullish"
    assert outcome.risk_points >= 1.0


def test_generate_report(tmp_path: Path) -> None:
    if not DEFAULT_TRIGGER_REPORT_PATH.exists():
        pytest.skip("institutional_trigger_validation.json not available")

    out = tmp_path / "trigger_trade_validation.json"
    report = generate_trigger_trade_validation_report(report_path=out)
    assert out.exists()
    assert report.trades_simulated > 0
    assert report.trigger_trade_metrics
    assert report.production_trigger_matrix


def test_missing_trigger_report_raises(tmp_path: Path) -> None:
    with pytest.raises(TriggerTradeValidationError):
        TriggerTradeValidationResearch(
            trigger_report_path=tmp_path / "missing.json",
        ).run()
