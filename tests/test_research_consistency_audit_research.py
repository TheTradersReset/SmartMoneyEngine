"""Tests for research consistency audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.research_consistency_audit_research import (
    ResearchConsistencyAuditError,
    ResearchConsistencyAuditResearch,
    SOURCE_FILES,
    generate_research_consistency_audit_report,
)


def test_normalize_tokens() -> None:
    tokens = ResearchConsistencyAuditResearch._normalize_tokens(
        "Failed Breakout + Liquidity Grab",
    )
    assert "failed" in tokens
    assert "liquidity" in tokens


def test_overlap_score() -> None:
    score = ResearchConsistencyAuditResearch._overlap_score(
        "Liquidity Grab + Failed Breakdown",
        "Failed Breakdown + Stop Hunt",
    )
    assert score > 0


def test_look_ahead_risk() -> None:
    assert ResearchConsistencyAuditResearch._look_ahead_risk("momentum_anatomy", "BOS") == "HIGH"
    assert ResearchConsistencyAuditResearch._look_ahead_risk("walkforward", "Below VWAP") == "LOW"


def test_audit_pattern_contradiction() -> None:
    engine = ResearchConsistencyAuditResearch()
    sources = {
        "walkforward": {
            "survival_verdict": "DEGRADED",
            "survives_unseen_market_data": False,
            "out_of_sample_buy": {"win_rate_pct": 0, "sample_size": 2, "profit_factor": 0},
            "out_of_sample_sell": {"win_rate_pct": 58, "profit_factor": 1.62},
            "in_sample_buy": {},
            "in_sample_sell": {},
            "performance_degradation": {"buy": {}, "sell": {}},
        },
        "trap_to_momentum": {"trap_event_statistics": [{"event": "Gap Reversal"}]},
        "reality_check": {
            "final_production_verdict": {"pct_200_plus_moves_detected": 87, "production_readiness_verdict": "READY"},
            "overall_statistics": {"profit_factor": 1.49, "expectancy": 52, "signals_per_month": 64},
            "replay_rules": {"no_future_leakage": True},
        },
        "liquidity_decision_engine": {
            "final_questions": {"supporting_metrics": {"engine_detection_rate_200_plus_pct": 1.85}},
        },
    }
    row = {
        "pattern": "Liquidity Grab + Failed Breakdown",
        "source_module": "liquidity_decision_engine",
        "sample_count": 66,
    }
    audit = engine._audit_pattern(row, "BUY", sources)
    assert audit["side"] == "BUY"
    assert audit["look_ahead_bias_risk"] == "MEDIUM"
    assert audit["contradictions"]


def test_generate_report_with_fixtures(tmp_path: Path) -> None:
    research_dir = tmp_path / "research"
    research_dir.mkdir()

    def _write(name: str, payload: dict) -> None:
        (research_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    _write(
        "nifty50_momentum_anatomy_120d.json",
        {
            "final_questions": {"2_most_important_liquidity_events": ["Gap Reversal"]},
            "momentum_blueprint_discovery": {"most_profitable_bullish": [], "most_profitable_bearish": []},
        },
    )
    _write(
        "nifty50_trap_to_momentum_validation.json",
        {
            "final_answers": {
                "2_most_important_liquidity_events": ["Liquidity Grab"],
                "most_predictive_event": "Liquidity Grab",
            },
            "trap_event_statistics": [
                {"event": "Liquidity Grab", "occurrences": 100, "probability_200_plus_pct": 68},
            ],
        },
    )
    _write(
        "nifty50_liquidity_decision_engine.json",
        {
            "decision_matrix": {"top_50_buy_combinations": [], "top_50_sell_combinations": []},
            "final_questions": {
                "2_most_important_liquidity_events": ["Failed Breakdown"],
                "supporting_metrics": {"engine_detection_rate_200_plus_pct": 2},
            },
        },
    )
    _write(
        "smartmoneyengine_reality_check_validation.json",
        {
            "final_production_verdict": {"production_readiness_verdict": "READY", "pct_200_plus_moves_detected": 87},
            "overall_statistics": {"profit_factor": 1.49, "expectancy": 52, "signals_per_month": 64},
            "replay_rules": {"no_future_leakage": True},
        },
    )
    _write(
        "smartmoneyengine_walkforward_validation.json",
        {
            "survival_verdict": "DEGRADED",
            "survives_unseen_market_data": False,
            "out_of_sample_buy": {"win_rate_pct": 0, "profit_factor": 0, "sample_size": 2},
            "out_of_sample_sell": {"win_rate_pct": 58, "profit_factor": 1.62, "expectancy": 135},
            "in_sample_buy": {},
            "in_sample_sell": {},
            "performance_degradation": {"buy": {}, "sell": {}},
        },
    )
    _write(
        "smartmoneyengine_v2_signal_ranking.json",
        {
            "top_10_buy_models": [],
            "top_50_signal_archetypes": [
                {
                    "archetype_key": "direction=SELL | session=Closing",
                    "signal_side": "SELL",
                    "sample_size": 31,
                    "tier": "A",
                    "signal_quality_score": 80,
                    "profit_factor": 5.8,
                    "expectancy": 211,
                },
            ],
        },
    )

    files = {
        "momentum_anatomy": research_dir / "nifty50_momentum_anatomy_120d.json",
        "trap_to_momentum": research_dir / "nifty50_trap_to_momentum_validation.json",
        "liquidity_decision_engine": research_dir / "nifty50_liquidity_decision_engine.json",
        "reality_check": research_dir / "smartmoneyengine_reality_check_validation.json",
        "walkforward": research_dir / "smartmoneyengine_walkforward_validation.json",
        "v2_ranking": research_dir / "smartmoneyengine_v2_signal_ranking.json",
    }
    destination = research_dir / "research_consistency_audit.json"
    report = generate_research_consistency_audit_report(
        report_path=destination,
        source_files=files,
    )
    assert report.confirmed_findings
    assert destination.exists()


def test_missing_source_raises(tmp_path: Path) -> None:
    files = dict(SOURCE_FILES)
    files["momentum_anatomy"] = tmp_path / "missing.json"
    with pytest.raises(ResearchConsistencyAuditError):
        generate_research_consistency_audit_report(source_files=files)
