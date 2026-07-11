"""Tests for NIFTY50 liquidity direction decision matrix research."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research.nifty50_liquidity_direction_decision_matrix_research import (
    LIQUIDITY_EVENTS,
    MIN_COMBO_SAMPLES,
    Nifty50LiquidityDirectionDecisionMatrixError,
    Nifty50LiquidityDirectionDecisionMatrixResearch,
    generate_nifty50_liquidity_direction_decision_matrix_report,
)


def test_constants() -> None:
    assert "Liquidity Grab" in LIQUIDITY_EVENTS
    assert MIN_COMBO_SAMPLES == 50


def test_optimal_decision() -> None:
    assert Nifty50LiquidityDirectionDecisionMatrixResearch._optimal_decision(120, 20) == "BUY"
    assert Nifty50LiquidityDirectionDecisionMatrixResearch._optimal_decision(20, 120) == "SELL"
    assert Nifty50LiquidityDirectionDecisionMatrixResearch._optimal_decision(10, 5) == "NO TRADE"


def test_combo_key() -> None:
    key = Nifty50LiquidityDirectionDecisionMatrixResearch._combo_key(
        "Liquidity Grab",
        {
            "htf_trend": "Bullish",
            "choch": "Present",
            "bos": "Absent",
            "vwap": "Below",
            "ema_structure": "Bear Stack",
            "rsi": "30-40",
            "volume": "Expanded",
            "confirmation_candle": "Hammer",
            "location": "Near Support",
        },
    )
    assert "Liquidity Grab" in key
    assert "HTF=Bullish" in key


def test_profit_factor() -> None:
    pf = Nifty50LiquidityDirectionDecisionMatrixResearch._profit_factor([100.0, -50.0, 80.0])
    assert pf == 3.6


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2026-03-10", "end_date": "2026-07-08", "research_window_days": 120}',
        encoding="utf-8",
    )
    destination = tmp_path / "nifty50_liquidity_direction_decision_matrix.json"

    class _FakeReport:
        total_liquidity_events = 500
        decision_matrix = [{"combination": "test", "occurrences": 60}]
        most_reliable_formulas = {"most_reliable_buy_formula": {"direction_accuracy_pct": 70}}

        def as_dict(self) -> dict:
            return {"total_liquidity_events": self.total_liquidity_events}

    def _fake_run(self, metadata: dict) -> _FakeReport:
        del metadata
        return _FakeReport()

    monkeypatch.setattr(Nifty50LiquidityDirectionDecisionMatrixResearch, "run", _fake_run)

    report = generate_nifty50_liquidity_direction_decision_matrix_report(
        report_path=destination,
        filter_report_path=filter_report,
    )
    assert report.total_liquidity_events == 500
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(Nifty50LiquidityDirectionDecisionMatrixError):
        generate_nifty50_liquidity_direction_decision_matrix_report(
            filter_report_path=Path("missing.json"),
        )
