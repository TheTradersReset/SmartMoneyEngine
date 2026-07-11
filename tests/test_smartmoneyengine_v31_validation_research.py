"""Tests for SmartMoneyEngine V3.1 validation research."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research.smartmoneyengine_v31_validation_research import (
    SmartMoneyEngineV31ValidationError,
    SmartMoneyEngineV31ValidationReport,
    SmartMoneyEngineV31ValidationResearch,
    generate_smartmoneyengine_v31_validation_report,
)


def test_cluster_map_v3_groups_intracluster_refires() -> None:
    research = SmartMoneyEngineV31ValidationResearch()
    signals = [
        {"bar": 10, "timestamp": "2026-01-01 10:00", "mfe_points": 120.0},
        {"bar": 12, "timestamp": "2026-01-01 10:10", "mfe_points": 100.0},
        {"bar": 40, "timestamp": "2026-01-02 10:00", "mfe_points": 80.0},
    ]
    clusters = research._cluster_map_v3(signals)
    assert len(clusters) == 2
    assert len(clusters[1]["signals"]) == 2
    assert signals[1]["delay_bars_vs_cluster_first"] == 2


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text('{"end_date": "2026-07-03"}', encoding="utf-8")
    destination = tmp_path / "smartmoneyengine_v31_validation.json"

    class _FakeReport(SmartMoneyEngineV31ValidationReport):
        pass

    def _fake_run(self, metadata: dict) -> SmartMoneyEngineV31ValidationReport:
        del metadata
        return SmartMoneyEngineV31ValidationReport(
            report_type="SmartMoneyEngine V3.1 Validation",
            engine_versions_compared=["SmartMoneyEngine V3", "SmartMoneyEngine V3.1"],
            symbol="NIFTY50",
            timeframe="5M",
            trading_days_replayed=120,
            replay_start_date="2026-01-01",
            replay_end_date="2026-07-03",
            v3_change_summary={},
            replay_rules={},
            comparison={
                "v3": {"overall_statistics": {"signals_emitted": 10}},
                "v3.1": {"overall_statistics": {"signals_emitted": 7}},
            },
            timing_audit={},
            major_move_entry_comparison=[],
            july_7_8_selloff_analysis={},
            final_questions={"5_is_v31_better_than_v3": {"answer": "PARTIAL"}},
            conclusions=[],
            execution_time_seconds=1.0,
        )

    monkeypatch.setattr(SmartMoneyEngineV31ValidationResearch, "run", _fake_run)

    report = generate_smartmoneyengine_v31_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
    )
    assert report.comparison["v3.1"]["overall_statistics"]["signals_emitted"] == 7
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(SmartMoneyEngineV31ValidationError):
        generate_smartmoneyengine_v31_validation_report(
            filter_report_path=Path("missing.json"),
        )
