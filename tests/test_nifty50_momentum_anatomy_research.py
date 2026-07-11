"""Tests for NIFTY50 momentum anatomy research."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.research.nifty50_momentum_anatomy_research import (
    ANATOMY_TIMELINE_OFFSETS_MINUTES,
    DEFAULT_SYMBOL,
    MOVE_THRESHOLDS,
    Nifty50MomentumAnatomyError,
    Nifty50MomentumAnatomyResearch,
    RESEARCH_WINDOW_DAYS,
    TIMEFRAMES,
    generate_nifty50_momentum_anatomy_report,
)


def test_constants() -> None:
    assert DEFAULT_SYMBOL == "NIFTY50"
    assert RESEARCH_WINDOW_DAYS == 120
    assert "1D" in TIMEFRAMES
    assert 100 in MOVE_THRESHOLDS
    assert 0 in ANATOMY_TIMELINE_OFFSETS_MINUTES


def test_bar_minutes_before() -> None:
    bar = Nifty50MomentumAnatomyResearch._bar_minutes_before(1000, 60, "5M")
    assert bar == 988


def test_threshold_tiers() -> None:
    tiers = Nifty50MomentumAnatomyResearch._threshold_tiers(245.5)
    assert tiers == [100, 200]


def test_timeline_label() -> None:
    assert Nifty50MomentumAnatomyResearch._timeline_label(0) == "Move Start"
    assert Nifty50MomentumAnatomyResearch._timeline_label(30) == "T-30 minutes"


def test_classify_move_origin_liquidity_grab() -> None:
    origin = Nifty50MomentumAnatomyResearch._classify_move_origin(
        direction="bullish",
        tags=("Liquidity Grab", "BOS"),
        measurements={"liquidity": {"liquidity_grab_count": 1}, "support_resistance": {}},
        reasons={"bos": True},
        flags={},
    )
    assert origin == "Liquidity Grab"


def test_resample_daily() -> None:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(8):
        ts = base + pd.Timedelta(hours=index)
        price = 100.0 + index
        rows.append(
            {
                "Date": ts.isoformat(),
                "Open": price,
                "High": price + 1,
                "Low": price - 1,
                "Close": price + 0.5,
                "Volume": 1000 + index,
            },
        )
    daily = Nifty50MomentumAnatomyResearch._resample_daily(pd.DataFrame(rows))
    assert len(daily) >= 1
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(daily.columns)


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2026-03-10", "end_date": "2026-07-08", "research_window_days": 120}',
        encoding="utf-8",
    )
    destination = tmp_path / "nifty50_momentum_anatomy_120d.json"

    class _FakeReport:
        completed_moves = {"100": [{"date": "2026-04-01", "move_size_points": 120.0}]}
        origin_frequency_ranking = [{"origin_trigger": "Liquidity Grab", "occurrences": 1}]
        final_questions = {
            "supporting_metrics": {"major_move_engine_detection_rate_pct": 42.0},
            "10_biggest_improvement_opportunity": "No Confirmation Candle",
        }

        def as_dict(self) -> dict:
            return {
                "completed_moves": self.completed_moves,
                "final_questions": self.final_questions,
            }

    def _fake_run(self, metadata: dict) -> _FakeReport:
        del metadata
        return _FakeReport()

    monkeypatch.setattr(Nifty50MomentumAnatomyResearch, "run", _fake_run)

    report = generate_nifty50_momentum_anatomy_report(
        report_path=destination,
        filter_report_path=filter_report,
    )
    assert report.final_questions["supporting_metrics"]["major_move_engine_detection_rate_pct"] == 42.0
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(Nifty50MomentumAnatomyError):
        generate_nifty50_momentum_anatomy_report(filter_report_path=Path("missing.json"))
