"""Tests for NIFTY50 liquidity decision engine research."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research.nifty50_liquidity_decision_engine_research import (
    DEFAULT_SYMBOL,
    LIQUIDITY_EVENTS,
    MIN_MATRIX_SAMPLES,
    RESEARCH_WINDOW_DAYS,
    Nifty50LiquidityDecisionEngineError,
    Nifty50LiquidityDecisionEngineResearch,
    generate_nifty50_liquidity_decision_engine_report,
)


def test_constants() -> None:
    assert DEFAULT_SYMBOL == "NIFTY50"
    assert RESEARCH_WINDOW_DAYS == 120
    assert "Liquidity Grab" in LIQUIDITY_EVENTS
    assert MIN_MATRIX_SAMPLES == 50


def test_distance_bucket() -> None:
    assert Nifty50LiquidityDecisionEngineResearch._distance_bucket(5) == "0-10"
    assert Nifty50LiquidityDecisionEngineResearch._distance_bucket(20) == "10-25"
    assert Nifty50LiquidityDecisionEngineResearch._distance_bucket(40) == "25-50"
    assert Nifty50LiquidityDecisionEngineResearch._distance_bucket(80) == "50+"


def test_test_bucket() -> None:
    assert Nifty50LiquidityDecisionEngineResearch._test_bucket(1) == "1 test"
    assert Nifty50LiquidityDecisionEngineResearch._test_bucket(4) == "4+ tests"


def test_assign_decision() -> None:
    assert Nifty50LiquidityDecisionEngineResearch._assign_decision("Bullish Reversal") == "BUY"
    assert Nifty50LiquidityDecisionEngineResearch._assign_decision("Bearish Continuation") == "SELL"
    assert Nifty50LiquidityDecisionEngineResearch._assign_decision("No Expansion") == "NO TRADE"


def test_classify_outcome() -> None:
    engine = Nifty50LiquidityDecisionEngineResearch()
    assert engine._classify_outcome(120, 20, "bullish", "Failed Breakdown") == "Bullish Reversal"
    assert engine._classify_outcome(20, 120, "bearish", "Failed Breakout") == "Bearish Reversal"
    assert engine._classify_outcome(10, 5, "bullish", "Gap Continuation") == "No Expansion"


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2026-03-10", "end_date": "2026-07-08", "research_window_days": 120}',
        encoding="utf-8",
    )
    destination = tmp_path / "nifty50_liquidity_decision_engine.json"

    class _FakeReport:
        liquidity_events_detected = 100
        decision_matrix = {"top_50_buy_combinations": [{"combination": "Liquidity Grab + Hammer"}]}

        def as_dict(self) -> dict:
            return {"liquidity_events_detected": self.liquidity_events_detected}

    def _fake_run(self, metadata: dict) -> _FakeReport:
        del metadata
        return _FakeReport()

    monkeypatch.setattr(Nifty50LiquidityDecisionEngineResearch, "run", _fake_run)

    report = generate_nifty50_liquidity_decision_engine_report(
        report_path=destination,
        filter_report_path=filter_report,
    )
    assert report.liquidity_events_detected == 100
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(Nifty50LiquidityDecisionEngineError):
        generate_nifty50_liquidity_decision_engine_report(filter_report_path=Path("missing.json"))
