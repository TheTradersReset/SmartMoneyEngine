"""Tests for production gap closure audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.production_gap_closure_audit_research import (
    BLOCKER_CATEGORIES,
    CAPITAL_TIERS_INR,
    EVIDENCE_COMPONENTS,
    ProductionGapClosureAuditResearch,
    _blockers_audit,
    _capital_tier_metrics,
    _evidence_gap_audit,
    _proven_status,
    _quality_score,
)


def test_constants() -> None:
    assert len(BLOCKER_CATEGORIES) == 6
    assert len(EVIDENCE_COMPONENTS) == 6
    assert "paper" in CAPITAL_TIERS_INR
    assert CAPITAL_TIERS_INR["inr_50k"] == 50_000


def test_proven_status() -> None:
    assert _proven_status(85.0) == "PROVEN"
    assert _proven_status(85.0, has_caveat=True) == "PARTIALLY PROVEN"
    assert _proven_status(55.0) == "PARTIALLY PROVEN"
    assert _proven_status(30.0) == "UNPROVEN"


def test_quality_score() -> None:
    score = _quality_score(replay="HIGH", walk_forward="MEDIUM", live="NONE")
    assert 60 <= score <= 80


def test_blockers_audit_smoke() -> None:
    reality = {
        "evidence_quality": {"current_sample_sizes": {"buy_v3": 116, "sell_v6": 336}},
        "execution_bottleneck_audit": {"primary_bottleneck": "runner"},
    }
    deployment = {
        "engine_validation_reconciliation": {
            "buy_v3": {"win_rate_pct": {"authoritative_for_gates": 72.41}},
            "sell_v6": {"win_rate_pct": {"reconciled": 70.24}},
        },
        "final_answer": {"evidence": {"sell_v6_validate_pf_unthrottled": 1.44, "sell_v6_validate_pf_throttled": 7.08}},
        "deployment_playbook": {
            "sizing_rules": {"buy_sleeve_pct": 35, "sell_sleeve_pct": 65},
            "risk_rules": {"portfolio_daily_loss_limit_points": 593.79},
        },
    }
    regime = {"final_answer": {"throttle_restores_validate_pf_2_plus": True}, "throttle_recommendation": {"sell_v6_regime_throttle": []}}
    live = {"final_answer": {"capture_efficiency_pct": {"paper_combined": 37.66}, "optimal_exit_structures": {"buy_v3": "60/100/Runner"}}}
    result = _blockers_audit(reality=reality, deployment=deployment, regime=regime, live=live, closure={})
    assert len(result["recommendations"]) == 6
    assert result["summary"]["partially_proven_count"] >= 1


def test_evidence_gap_audit_smoke() -> None:
    reality = {
        "production_truth_audit": {
            "evidence_scores": {"buy_v3": 95.2, "sell_v6": 80.0, "regime_throttle": 76.0},
            "aggregate_evidence_score": 84.9,
        },
        "evidence_quality": {"is_120d_sufficient": {"buy_v3_validate_caveat": "n=6"}},
    }
    result = _evidence_gap_audit(
        reality=reality,
        deployment={},
        regime={"throttle_recommendation": {"sell_v6_regime_throttle": [{}]}},
        live={},
        extended={},
    )
    assert len(result["breakdown"]) == 6
    assert result["extended_evidence_status"] == "missing"


def test_capital_tier_paper() -> None:
    row = _capital_tier_metrics(
        tier_key="paper",
        capital_inr=None,
        monthly_points_paper=8528.0,
        monthly_points_real=4037.0,
        max_dd_paper_pts=50.0,
        max_dd_real_pts=914.0,
        execution_risk_score=83.8,
    )
    assert row["readiness"] == "READY"
    assert row["execution_risk_material"] is False


@pytest.fixture
def tmp_research_dir(tmp_path: Path) -> Path:
    research_dir = tmp_path / "outputs" / "research"
    research_dir.mkdir(parents=True)

    def _write(name: str, payload: dict) -> None:
        (research_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    _write(
        "production_reality_audit.json",
        {
            "symbol": "NIFTY50",
            "timeframe": "5M",
            "trading_days_replayed": 120,
            "replay_start_date": "2026-01-05",
            "replay_end_date": "2026-07-02",
            "evidence_quality": {
                "current_sample_sizes": {"buy_v3": 116, "sell_v6": 336},
                "is_120d_sufficient": {"buy_v3_validate_caveat": "n=6", "verdict": "PARTIAL"},
            },
            "execution_bottleneck_audit": {"primary_bottleneck": "runner"},
            "production_truth_audit": {
                "evidence_scores": {
                    "buy_v3": 95.2,
                    "sell_v6": 80.0,
                    "regime_throttle": 76.0,
                    "60_100_runner": 86.0,
                    "fixed_10_stop": 86.0,
                    "structure_stop": 86.0,
                },
                "aggregate_evidence_score": 84.9,
            },
            "production_scores": {
                "production_readiness_score": 72.0,
                "confidence_score": 66.2,
                "production_risk_score": 68.5,
                "evidence_score": 84.9,
                "deployment_tier": "Production Candidate",
                "capture_summary": {"improvement_potential_capture_pct": 2.88},
            },
            "final_answer": {
                "paper_trade_tomorrow": "YES",
                "real_capital_deployment": "NO",
                "should_research_buy_v4": "NO",
                "should_research_sell_v7": "NO",
                "can_expectancy_improve_without_buy_v4_sell_v7": "YES",
                "biggest_uncertainty_before_real_capital": "Live slippage",
                "biggest_opportunity_for_improvement": "Runner trail",
                "rationale": "Test rationale",
            },
        },
    )
    _write(
        "final_production_deployment_audit.json",
        {
            "trading_days_replayed": 120,
            "engine_validation_reconciliation": {
                "buy_v3": {"win_rate_pct": {"authoritative_for_gates": 72.41}},
                "sell_v6": {"win_rate_pct": {"reconciled": 70.24}},
            },
            "deployment_playbook": {
                "paper_trading_checklist": ["Enable BUY_V3"],
                "sizing_rules": {"buy_sleeve_pct": 35, "sell_sleeve_pct": 65, "buy_sizing_mode": "regime_adaptive", "sell_sizing_mode": "regime_adaptive"},
                "risk_rules": {"portfolio_daily_loss_limit_points": 593.79},
                "runner_policy": "33% runner trail",
                "conflict_policy": "NO_TRADE",
            },
            "final_answer": {
                "still_unverified": [
                    "Live slippage and fill quality on NIFTY50 5M",
                    "SELL_V6 validate-window PF stability beyond 40 trading days",
                ],
                "evidence": {"sell_v6_validate_pf_unthrottled": 1.44, "sell_v6_validate_pf_throttled": 7.08},
                "pf_audit_critical_flags": 2,
            },
        },
    )
    _write(
        "regime_detection_audit.json",
        {
            "final_answer": {"throttle_restores_validate_pf_2_plus": True},
            "throttle_recommendation": {
                "sell_v6_regime_throttle": [{"regime": "test", "throttle": "BLOCK", "validate_signal_count": 6}],
            },
        },
    )
    _write(
        "live_trade_management_execution_efficiency_audit.json",
        {
            "final_answer": {
                "optimal_stops": {"buy_v3": "fixed_10", "sell_v6": "fixed_10"},
                "optimal_exit_structures": {"buy_v3": "60/100/Runner", "sell_v6": "60/100/Runner"},
                "expected_monthly_points": {"paper_combined": 8528.72, "real_capital_combined": 4037.76},
                "expected_drawdown_points": {"paper_combined": 50.0, "real_capital_combined": 914.64},
                "capture_efficiency_pct": {"paper_combined": 37.66},
            },
        },
    )
    _write(
        "production_readiness_closure_audit.json",
        {
            "part5_live_execution_risk": {"execution_risk_score": 83.8},
            "part6_research_closure": {
                "primary_bottleneck": "runner",
                "walk_forward_context": {"primary_degradation_engine": "SELL_V6", "top_root_cause": "regime_change"},
            },
            "top_risks": [{"rank": 1, "risk": "Live slippage", "severity": "HIGH"}],
            "top_opportunities": [{"rank": 1, "opportunity": "Runner trail", "impact": "HIGH"}],
        },
    )
    return research_dir


def test_export_synthetic(tmp_research_dir: Path, tmp_path: Path) -> None:
    report_path = tmp_research_dir / "production_gap_closure_audit.json"
    research = ProductionGapClosureAuditResearch(report_path=report_path)

    import src.research.production_gap_closure_audit_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_primary = module.PRIMARY_EXPORTS.copy()
    original_optional = module.OPTIONAL_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = tmp_research_dir
        module.PRIMARY_EXPORTS = {k: tmp_research_dir / Path(v).name for k, v in module.PRIMARY_EXPORTS.items()}
        module.OPTIONAL_EXPORTS = {k: tmp_research_dir / Path(v).name for k, v in module.OPTIONAL_EXPORTS.items()}
        exported = research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.PRIMARY_EXPORTS = original_primary
        module.OPTIONAL_EXPORTS = original_optional

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Production Gap Closure Audit"
    assert payload["methodology"]["synthesis_only"] is True
    assert "blockers_audit" in payload
    assert "evidence_gap_audit" in payload
    assert "capital_deployment_readiness" in payload
    assert "deployment_roadmap" in payload
    assert len(payload["top_10_risks"]) >= 1
    assert len(payload["top_10_opportunities"]) >= 1
    assert payload["definitive_verdict"]["research_complete"] in {"YES", "NO"}
    assert payload["final_answer"]["paper_trading_verdict"] == "YES"
    assert payload["final_answer"]["full_capital_verdict"] == "NO"
    assert len(payload["blockers_audit"]["recommendations"]) == 6
    assert len(payload["evidence_gap_audit"]["breakdown"]) == 6


@pytest.mark.skipif(
    not Path("outputs/research/production_reality_audit.json").exists(),
    reason="Requires completed replay exports",
)
def test_export_real_data() -> None:
    research = ProductionGapClosureAuditResearch()
    path = research.export()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["trading_days_replayed"] >= 120
    assert payload["final_answer"]["paper_trading_verdict"] == "YES"
    assert payload["final_answer"]["full_capital_verdict"] == "NO"
    assert payload["deployment_readiness_matrix"]["evidence_complete_pct"] > 80
