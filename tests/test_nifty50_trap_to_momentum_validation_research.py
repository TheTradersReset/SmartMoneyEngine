"""Tests for NIFTY50 trap-to-momentum validation research."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.research.nifty50_trap_to_momentum_validation_research import (
    ALL_EVENTS,
    DEFAULT_SYMBOL,
    MOVE_THRESHOLDS,
    RESEARCH_WINDOW_DAYS,
    TRAP_EVENTS,
    Nifty50TrapToMomentumValidationError,
    Nifty50TrapToMomentumValidationResearch,
    generate_nifty50_trap_to_momentum_validation_report,
)


def test_constants() -> None:
    assert DEFAULT_SYMBOL == "NIFTY50"
    assert RESEARCH_WINDOW_DAYS == 120
    assert "Liquidity Grab" in TRAP_EVENTS
    assert "CHOCH" in ALL_EVENTS
    assert 500 in MOVE_THRESHOLDS


def test_forward_max_move() -> None:
    highs = pd.Series([100, 105, 130, 125])
    lows = pd.Series([99, 100, 110, 115])
    closes = pd.Series([100, 104, 128, 120])
    magnitude = Nifty50TrapToMomentumValidationResearch._forward_max_move(highs, lows, closes, 0, 3)
    assert magnitude >= 30


def test_probability() -> None:
    assert Nifty50TrapToMomentumValidationResearch._probability([True, False, True]) == 66.67


def test_predictive_score() -> None:
    score = Nifty50TrapToMomentumValidationResearch._predictive_score(
        {
            "probability_100_plus_pct": 50.0,
            "probability_200_plus_pct": 25.0,
            "probability_300_plus_pct": 10.0,
            "probability_500_plus_pct": 5.0,
        },
    )
    assert score > 0


def test_round_number_level() -> None:
    assert Nifty50TrapToMomentumValidationResearch._round_number_level(25123.0) == 25100.0


def test_detect_gap_reversal() -> None:
    engine = Nifty50TrapToMomentumValidationResearch()
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(3):
        price = 25000.0 + index * 10
        rows.append(
            {
                "Date": (base + pd.Timedelta(minutes=5 * index)).isoformat(),
                "Open": price + (5 if index == 1 else 0),
                "High": price + 8,
                "Low": price - 2,
                "Close": price + (1 if index != 1 else -3),
                "Volume": 100000,
                "Swing_High": pd.NA,
                "Swing_Low": pd.NA,
                "Buy_Liquidity_Sweep": pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
                "Equal_High": pd.NA,
                "Equal_Low": pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": pd.NA,
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bearish_FVG_Top": pd.NA,
                "Bullish_OB_High": pd.NA,
                "Bearish_OB_High": pd.NA,
            },
        )
    frame = pd.DataFrame(rows)
    calendar = frame.copy()
    calendar["_pdh"] = pd.NA
    calendar["_pdl"] = pd.NA
    calendar["_pwh"] = pd.NA
    calendar["_pwl"] = pd.NA
    events = engine._detect_events_at_bar(frame, calendar, 1)
    assert "Gap Reversal" in events


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2026-03-10", "end_date": "2026-07-08", "research_window_days": 120}',
        encoding="utf-8",
    )
    destination = tmp_path / "nifty50_trap_to_momentum_validation.json"

    class _FakeReport:
        final_answers = {
            "most_predictive_event": "Liquidity Grab",
            "average_bars_before_move": 12.0,
            "earliest_warning_combination": {"combination": "Gap Reversal + Liquidity Grab"},
        }

        def as_dict(self) -> dict:
            return {"final_answers": self.final_answers}

    def _fake_run(self, metadata: dict) -> _FakeReport:
        del metadata
        return _FakeReport()

    monkeypatch.setattr(Nifty50TrapToMomentumValidationResearch, "run", _fake_run)

    report = generate_nifty50_trap_to_momentum_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
    )
    assert report.final_answers["most_predictive_event"] == "Liquidity Grab"
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(Nifty50TrapToMomentumValidationError):
        generate_nifty50_trap_to_momentum_validation_report(filter_report_path=Path("missing.json"))
