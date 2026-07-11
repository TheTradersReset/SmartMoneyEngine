"""Tests for SmartMoneyEngine reality check validation research."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research.smartmoneyengine_reality_check_validation_research import (
    MAJOR_TIMELINE_THRESHOLDS,
    MISSED_THRESHOLDS,
    TIMELINE_OFFSETS_MINUTES,
    RealityCheckValidationError,
    SmartMoneyEngineRealityCheckValidationResearch,
    generate_reality_check_validation_report,
)


def test_parse_archetype_key() -> None:
    parsed = SmartMoneyEngineRealityCheckValidationResearch._parse_archetype_key(
        "timeframe=5M | direction=SELL | session=Closing",
    )
    assert parsed["timeframe"] == "5M"
    assert parsed["direction"] == "SELL"


def test_missed_classification() -> None:
    assert (
        SmartMoneyEngineRealityCheckValidationResearch._missed_classification(True, False)
        == "Detected"
    )
    assert (
        SmartMoneyEngineRealityCheckValidationResearch._missed_classification(False, True)
        == "Partially Detected"
    )
    assert (
        SmartMoneyEngineRealityCheckValidationResearch._missed_classification(False, False)
        == "Missed"
    )


def test_bar_minutes_before() -> None:
    bar = SmartMoneyEngineRealityCheckValidationResearch._bar_minutes_before(1000, 30, "5M")
    assert bar == 994


def test_rank_missed_reasons() -> None:
    ranking = SmartMoneyEngineRealityCheckValidationResearch._rank_missed_reasons(
        [
            {"classification": "Missed", "missed_reasons": ["No BOS", "No CHOCH"]},
            {"classification": "Missed", "missed_reasons": ["No BOS"]},
            {"classification": "Detected", "missed_reasons": []},
        ],
    )
    assert ranking[0]["reason"] == "No BOS"
    assert ranking[0]["occurrences"] == 2


def test_constants() -> None:
    assert 200 in MISSED_THRESHOLDS
    assert 200 in MAJOR_TIMELINE_THRESHOLDS
    assert 0 in TIMELINE_OFFSETS_MINUTES


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2025-07-03", "end_date": "2026-07-03", "research_window_days": 365}',
        encoding="utf-8",
    )
    destination = tmp_path / "smartmoneyengine_reality_check_validation.json"

    class _FakeReport:
        overall_statistics = {"total_signals": 25}
        final_production_verdict = {
            "production_readiness_verdict": "NOT READY",
            "pct_200_plus_moves_detected": 12.0,
        }

        def as_dict(self) -> dict:
            return {"overall_statistics": self.overall_statistics}

    def _fake_run(self, metadata: dict) -> _FakeReport:
        del metadata
        return _FakeReport()

    monkeypatch.setattr(SmartMoneyEngineRealityCheckValidationResearch, "run", _fake_run)

    report = generate_reality_check_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
    )
    assert report.overall_statistics["total_signals"] == 25
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(RealityCheckValidationError):
        generate_reality_check_validation_report(filter_report_path=Path("missing.json"))
