"""Tests for research integrity ground truth validation audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.research_integrity_ground_truth_validation_audit_research import (
    ResearchIntegrityGroundTruthValidationAuditError,
    ResearchIntegrityGroundTruthValidationAuditResearch,
    generate_research_integrity_ground_truth_validation_audit_report,
)


def test_integrity_audit_mocked(tmp_path: Path) -> None:
    sources = {
        "extended_trade_level_truth_audit": {
            "replay_windows": [240],
            "available_trading_days": 247,
            "replay_start_date": "2025-07-11",
            "replay_end_date": "2026-07-02",
            "per_signal_details": {"buy_v3": [{"x": 1}] * 3, "sell_v6": [{"x": 1}] * 5},
            "core_metrics_by_window": {"240": {"replay_start_date": "2025-07-11", "replay_end_date": "2026-07-02"}},
        },
        "extended_evidence_validation_real_deployment_audit": {
            "replay_windows": [120, 250, 500],
            "methodology": {"actual_replay": True},
            "window_results": {
                "250": {"buy_v3_only": {"signals_emitted": 261}, "sell_v6_only": {"signals_emitted": 664}},
                "500": {"buy_v3_only": {"signals_emitted": 261}, "sell_v6_only": {"signals_emitted": 664}},
            },
        },
        "buy_v4_sell_v7_design_blueprint_audit": {"report_type": "blueprint"},
        "buy_v4_sell_v7_final_production_validation": {
            "available_trading_days": 121,
            "replay_windows": [240, 250, 500],
            "methodology": {"signal_source": "filtered"},
            "approved_filters": {"buy_v4": ["Liquidity Sweep Failure"], "sell_v7": ["Volatility Collapse"]},
            "core_metrics_by_window": {
                "240": {
                    "buy_v4": {
                        "signals_emitted": 100,
                        "win_rate_pct": 60,
                        "profit_factor": 4.5,
                        "expectancy": 50,
                        "capture_pct": 40,
                        "max_drawdown_points": 200,
                    },
                    "sell_v7": {
                        "signals_emitted": 400,
                        "win_rate_pct": 80,
                        "profit_factor": 7,
                        "expectancy": 60,
                        "capture_pct": 40,
                        "max_drawdown_points": 300,
                    },
                },
            },
            "final_answer": {
                "should_buy_v4_replace_buy_v3": "YES",
                "should_sell_v7_replace_sell_v6": "YES",
            },
            "trade_outcome_distribution": {"240": {}},
            "target_path_analysis": {},
            "signal_timing_reality": {},
        },
        "buy_v3_candidate_validation": {
            "trading_days_replayed": 120,
            "replay_start_date": "2026-01-05",
            "replay_end_date": "2026-07-02",
            "per_signal_details": {"buy_v3": [{"a": 1}]},
        },
        "sell_v6_replay_validation": {
            "trading_days_replayed": 120,
            "replay_start_date": "2026-01-05",
            "replay_end_date": "2026-07-02",
            "per_signal_details": {"sell_v6": [{"a": 1}]},
        },
    }
    research = ResearchIntegrityGroundTruthValidationAuditResearch()
    report = research.run(sources)
    out = tmp_path / "research_integrity_ground_truth_validation_audit.json"
    research.export(report, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["final_answer"]["can_buy_v4_replace_buy_v3"] == "NO"
    assert payload["final_answer"]["can_sell_v7_replace_sell_v6"] == "NO"
    assert payload["definitive_verdict"]["research_complete"] == "NO"
    assert payload["buy_v4_validation_audit"]["method_code"] == "B"
    assert payload["window_replay_status"]["500d"]["derived_only"] is True


def test_generate_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research import research_integrity_ground_truth_validation_audit_research as mod

    monkeypatch.setattr(
        mod,
        "REQUIRED_EXPORTS",
        {"extended_trade_level_truth_audit": tmp_path / "missing.json"},
    )
    with pytest.raises(ResearchIntegrityGroundTruthValidationAuditError):
        generate_research_integrity_ground_truth_validation_audit_report(report_path=tmp_path / "out.json")
