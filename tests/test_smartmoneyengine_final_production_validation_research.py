"""Tests for SmartMoneyEngine final production validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.smartmoneyengine_final_production_validation_research import (
    MANDATORY_CORE,
    MIN_SAMPLE_SIZE,
    SmartMoneyEngineFinalProductionValidationResearch,
    FinalProductionValidationError,
    generate_final_production_validation_report,
)


def test_mandatory_core() -> None:
    assert MANDATORY_CORE == ("Displacement", "CHOCH", "BOS", "FVG Reclaim")


def test_grade_signal() -> None:
    grade = SmartMoneyEngineFinalProductionValidationResearch._grade_signal(
        {
            "sample_size": 100,
            "win_rate_pct": 62,
            "profit_factor": 2.1,
            "expectancy": 90,
        },
    )
    assert grade == "A+"


def test_grade_rejects_small_sample() -> None:
    grade = SmartMoneyEngineFinalProductionValidationResearch._grade_signal(
        {
            "sample_size": 10,
            "win_rate_pct": 80,
            "profit_factor": 3.0,
            "expectancy": 200,
        },
    )
    assert grade == "D"


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2025-07-03", "end_date": "2026-07-03", "research_window_days": 365}',
        encoding="utf-8",
    )
    destination = tmp_path / "smartmoneyengine_final_production_validation.json"

    class _FakeReport:
        production_readiness_verdict = "READY"
        best_buy_stack = {"stack_label": "Test BUY"}
        best_sell_stack = {"stack_label": "Test SELL"}
        smartmoneyengine_v1_final_production_card = {
            "expected_signals_per_month": {"raw_tier2_capacity": 42},
        }

        def as_dict(self) -> dict:
            return {
                "production_readiness_verdict": self.production_readiness_verdict,
                "smartmoneyengine_v1_final_production_card": self.smartmoneyengine_v1_final_production_card,
            }

    def _fake_run(self: SmartMoneyEngineFinalProductionValidationResearch, metadata: dict) -> _FakeReport:
        del metadata
        return _FakeReport()

    monkeypatch.setattr(SmartMoneyEngineFinalProductionValidationResearch, "run", _fake_run)

    report = generate_final_production_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
    )
    assert report.production_readiness_verdict == "READY"
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(FinalProductionValidationError):
        generate_final_production_validation_report(filter_report_path=Path("missing.json"))


def test_synthesis_with_real_exports_if_present() -> None:
    research_dir = Path("outputs/research")
    candidate = research_dir / "smartmoneyengine_production_candidate.json"
    if not candidate.exists():
        pytest.skip("production candidate export not available")

    engine = SmartMoneyEngineFinalProductionValidationResearch(research_dir=research_dir)
    engine._load_exports()
    buy_pool, sell_pool = engine._collect_candidate_stacks()
    top_buy = engine._rank_stacks(buy_pool, "BUY")
    assert len(top_buy) <= 20
    if top_buy:
        assert top_buy[0]["sample_size"] >= MIN_SAMPLE_SIZE
        assert "mandatory_core" in top_buy[0]
