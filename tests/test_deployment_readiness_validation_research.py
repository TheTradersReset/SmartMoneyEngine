"""Tests for deployment readiness validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.deployment_readiness_validation_research import (
    CAPITAL_TIERS_INR,
    EVIDENCE_COMPONENTS,
    MFE_TIERS,
    DeploymentReadinessValidationResearch,
    _capital_tier_row,
    _data_sufficiency,
    _evidence_gap_row,
    _final_answer,
    _proven_status,
)


def test_constants() -> None:
    assert len(MFE_TIERS) == 8
    assert len(EVIDENCE_COMPONENTS) == 8
    assert CAPITAL_TIERS_INR["inr_50k"] == 50_000


def test_proven_status() -> None:
    assert _proven_status(85.0) == "PROVEN"
    assert _proven_status(85.0, has_caveat=True) == "PARTIALLY PROVEN"
    assert _proven_status(30.0) == "MISSING"


def test_capital_tier_row() -> None:
    row = _capital_tier_row(
        tier_key="inr_50k",
        capital_inr=50_000,
        monthly_points_real=4037.76,
        max_dd_real_pts=914.64,
        execution_risk_score=83.8,
    )
    assert row["capital_inr"] == 50_000
    assert row["deployment_verdict"] == "NO"
    assert row["readiness"] == "CONDITIONAL"


def test_data_sufficiency_smoke() -> None:
    reality = {
        "trading_days_replayed": 120,
        "evidence_quality": {
            "current_sample_sizes": {"buy_v3": 116, "sell_v6": 336},
            "is_120d_sufficient": {"verdict": "PARTIAL", "buy_v3_validate_caveat": "n=6"},
            "required_sample_sizes_by_confidence": {
                "80": {"buy_v3_wr": 132, "sell_v6_wr": 138, "combined_min": 135},
            },
        },
    }
    result = _data_sufficiency(reality=reality, closure={}, gap={}, extended={})
    assert result["minimum_buy_trades"] == 132
    assert result["minimum_sell_trades"] == 138
    assert "120d" in result["horizon_requirements"]


def test_evidence_gap_row() -> None:
    row = _evidence_gap_row(
        component="Live Execution",
        status="MISSING",
        evidence_score=None,
        basis="No telemetry",
        expected_impact="5–15% erosion",
        validation_method="Paper trade",
        estimated_days=30,
    )
    assert row["status"] == "MISSING"
    assert row["estimated_validation_time_days"] == 30


def test_final_answer_smoke() -> None:
    reality = {"final_answer": {"paper_trade_tomorrow": "YES", "should_research_buy_v4": "NO", "should_research_sell_v7": "NO"}}
    gap = {"final_answer": {"should_research_stop": "YES", "paper_trading_verdict": "YES"}, "definitive_verdict": {"research_complete_evidence": "test"}}
    capital = {
        "tiers": {
            "inr_50k": {"deployment_verdict": "NO"},
            "inr_1l": {"deployment_verdict": "NO"},
            "inr_2l": {"deployment_verdict": "NO"},
        },
    }
    result = _final_answer(
        reality=reality,
        gap=gap,
        closure={},
        capital=capital,
        evidence_gap={"summary": {"missing": 1, "partially_proven": 5}},
        scores={"production_readiness_score": 72.0, "confidence_score": 66.2, "production_risk_score": 68.5, "evidence_score": 84.9},
    )
    assert result["can_paper_trading_start_now"]["answer"] == "YES"
    assert result["should_research_buy_v4"]["answer"] == "NO"
    assert result["can_inr_50k_deployment_start_now"]["answer"] == "NO"


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
                "is_120d_sufficient": {"verdict": "PARTIAL", "buy_v3_validate_caveat": "n=6"},
                "required_sample_sizes_by_confidence": {
                    "80": {"buy_v3_wr": 132, "sell_v6_wr": 138, "combined_min": 135},
                    "90": {"buy_v3_wr": 217, "sell_v6_wr": 227, "combined_min": 222},
                },
            },
            "trade_outcome_distribution": {
                "buy_v3": {
                    "sample_size": 116,
                    "tiers": {"20": {"count": 112, "pct_of_signals": 96.55, "conditional_probability_pct": 96.55}},
                },
                "sell_v6": {
                    "sample_size": 336,
                    "tiers": {"20": {"count": 324, "pct_of_signals": 96.43, "conditional_probability_pct": 96.43}},
                },
            },
            "target_achievement_matrix": {
                "buy_v3": {"by_tier": {}, "aggregate": {"capture_pct": 38.86}},
                "sell_v6": {"by_tier": {}, "aggregate": {"capture_pct": 38.22}},
            },
            "signal_reality": {
                "buy_v3": {
                    "timing_class_summary": {"Very Early": {"count": 87, "pct": 75.0, "win_rate_pct": 80.0, "expectancy": 100.0}},
                    "predictive_vs_reactive": {"verdict": "PREDICTIVE"},
                },
                "sell_v6": {
                    "timing_class_summary": {"Very Early": {"count": 231, "pct": 68.75, "win_rate_pct": 78.79, "expectancy": 155.79}},
                    "predictive_vs_reactive": {"verdict": "PREDICTIVE"},
                },
            },
            "production_scores": {
                "production_readiness_score": 72.0,
                "confidence_score": 66.2,
                "production_risk_score": 68.5,
                "evidence_score": 84.9,
                "deployment_tier": "Production Candidate",
            },
            "production_truth_audit": {
                "evidence_scores": {"buy_v3": 95.2, "sell_v6": 80.0, "regime_throttle": 76.0},
                "aggregate_evidence_score": 84.9,
            },
            "final_answer": {
                "paper_trade_tomorrow": "YES",
                "real_capital_deployment": "NO",
                "should_research_buy_v4": "NO",
                "should_research_sell_v7": "NO",
                "biggest_uncertainty_before_real_capital": "Live slippage",
                "biggest_opportunity_for_improvement": "Runner trail",
            },
        },
    )
    _write(
        "production_gap_closure_audit.json",
        {
            "trading_days_replayed": 120,
            "production_scores": {"production_readiness_score": 72.0, "confidence_score": 66.2, "production_risk_score": 68.5, "evidence_score": 84.9},
            "deployment_roadmap": {
                "phase_1_paper": {"duration_sessions": 20, "success_criteria": ["Slippage ≤5pt"]},
                "phase_2_small_capital": {"success_criteria": ["40 sessions"]},
                "playbook_checklist": ["Enable BUY_V3"],
            },
            "capital_deployment_readiness": {"max_safe_capital_recommendation_inr": 200_000},
            "evidence_gap_audit": {"breakdown": [], "aggregate_evidence_score": 84.9},
            "final_answer": {"paper_trading_verdict": "YES", "should_research_stop": "YES"},
            "definitive_verdict": {"research_complete": "YES", "research_complete_evidence": "test"},
            "top_10_risks": [{"rank": 1, "risk": "Live slippage", "severity": "HIGH"}],
            "top_10_opportunities": [{"rank": 1, "opportunity": "Runner trail", "impact": "HIGH"}],
            "top_10_unknowns": [{"rank": 1, "unknown": "Partial fills"}],
            "authoritative_deployment_stack": {"sell_engine": {"throttle_required": True}},
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
                "capture_efficiency_pct": {"paper_combined": 37.66, "real_capital_combined": 37.43},
            },
            "signal_timing": {"buy_v3": {}, "sell_v6": {}},
        },
    )
    _write(
        "final_production_deployment_audit.json",
        {
            "deployment_playbook": {
                "paper_trading_checklist": ["Enable BUY_V3", "Enable SELL_V6 throttle"],
                "risk_rules": {"portfolio_daily_loss_limit_points": 593.79},
            },
            "engine_validation_reconciliation": {
                "buy_v3": {"win_rate_pct": {"authoritative_for_gates": 72.41}},
                "sell_v6": {"win_rate_pct": {"reconciled": 70.24}},
            },
            "final_answer": {
                "still_unverified": ["Live slippage and fill quality on NIFTY50 5M"],
                "evidence": {"sell_v6_validate_pf_throttled": 7.08},
            },
        },
    )
    _write(
        "production_readiness_closure_audit.json",
        {
            "part1_evidence_expansion": {
                "confidence_at_horizons": {
                    "120d": {"combined_confidence_pct": 91.5},
                    "250d": {"combined_confidence_pct": 95.0},
                    "500d": {"combined_confidence_pct": 95.0},
                },
            },
            "part5_live_execution_risk": {
                "slippage_viability_threshold_points": 10,
                "execution_risk_score": 83.8,
                "verdict": "LOW",
                "by_slippage_level": {"10": {"combined": {"profit_factor": 20.78, "viable": True}}},
            },
            "part6_research_closure": {
                "primary_bottleneck": "runner",
                "should_research_buy_v4": "NO",
                "should_research_sell_v7": "NO",
                "missing_evidence_for_real_capital": ["Live slippage and fill quality on NIFTY50 5M"],
            },
        },
    )
    return research_dir


def test_export_synthetic(tmp_research_dir: Path, tmp_path: Path) -> None:
    report_path = tmp_research_dir / "deployment_readiness_validation.json"
    research = DeploymentReadinessValidationResearch(report_path=report_path)

    import src.research.deployment_readiness_validation_research as module

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
    assert payload["report_type"] == "Deployment Readiness Validation"
    assert payload["methodology"]["synthesis_only"] is True
    assert payload["methodology"]["no_buy_v4"] is True
    assert "data_sufficiency" in payload
    assert "paper_trading_requirements" in payload
    assert "small_capital_deployment" in payload
    assert "live_execution_risk" in payload
    assert "target_distribution" in payload
    assert "signal_timing_quality" in payload
    assert "production_readiness_gates" in payload
    assert "evidence_gap_analysis" in payload
    assert payload["final_answer"]["can_paper_trading_start_now"]["answer"] == "YES"
    assert payload["final_answer"]["can_inr_50k_deployment_start_now"]["answer"] == "NO"
    assert payload["final_answer"]["should_research_buy_v4"]["answer"] == "NO"
    assert len(payload["top_10_risks"]) >= 1


@pytest.mark.skipif(
    not Path("outputs/research/production_reality_audit.json").exists(),
    reason="Requires completed replay exports",
)
def test_export_real_data() -> None:
    research = DeploymentReadinessValidationResearch()
    path = research.export()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["trading_days_replayed"] >= 120
    assert payload["final_answer"]["can_paper_trading_start_now"]["answer"] == "YES"
    assert payload["final_answer"]["can_inr_50k_deployment_start_now"]["answer"] == "NO"
    assert payload["production_scores"]["evidence_score"] > 80
