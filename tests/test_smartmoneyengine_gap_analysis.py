"""Tests for SmartMoneyEngine V1 gap analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research.smartmoneyengine_gap_analysis import (
    GapAnalysisError,
    SmartMoneyEngineGapAnalysis,
    SPEC_PATH,
    generate_gap_analysis_report,
)


def test_spec_exists() -> None:
    assert SPEC_PATH.exists()


def test_gap_analysis_runs() -> None:
    engine = SmartMoneyEngineGapAnalysis()
    report = engine.run()
    assert report.total_rules_evaluated >= 10
    assert report.classification_summary
    assert report.contradictory_rules


def test_generate_report(tmp_path: Path) -> None:
    out = tmp_path / "gap.json"
    report = generate_gap_analysis_report(report_path=out)
    assert out.exists()
    assert report.overall_verdict


def test_missing_spec_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.research.smartmoneyengine_gap_analysis as mod

    monkeypatch.setattr(mod, "SPEC_PATH", tmp_path / "missing.md")
    with pytest.raises(GapAnalysisError):
        SmartMoneyEngineGapAnalysis().run()
