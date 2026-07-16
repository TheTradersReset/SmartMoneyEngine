"""Tests for production readiness closure audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.production_readiness_closure_audit_research import (
    RUNNER_STRATEGIES,
    SLIPPAGE_STRESS_LEVELS,
    ProductionReadinessClosureAuditResearch,
    _part1_evidence_expansion,
    _part2_regime_throttle_reality,
    _part5_live_execution_risk,
    _rule_evidence_type,
    _timing_class,
)


def _buy_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-06 09:30:00+05:30",
        "direction": "BUY",
        "entry": 23500.0,
        "stop_loss": 23450.0,
        "target_1": 23550.0,
        "target_2": 23600.0,
        "target_3": 23650.0,
        "bars_before_expansion": 10,
        "points_before_expansion": 12.5,
        "mfe_points": 80.0,
        "mae_points": 20.0,
        "trade_duration_bars": 40,
        "hit_1r": True,
        "hit_2r": True,
        "hit_3r": False,
        "win": True,
        "win_default_r": True,
        "classification": "Real Reversal",
        "realized_pnl_points": 60.0,
        "layers": {
            "layer1": {"events_detected": ["Failed Breakdown"]},
            "layer2": {"htf_trend": "Bullish", "vwap_state": "Reclaimed", "aligned": True},
            "layer5": {"pass": True, "reason_codes": []},
        },
    }
    base.update(overrides)
    return base


def _sell_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-05 10:25:00+05:30",
        "direction": "SELL",
        "entry": 23600.0,
        "stop_loss": 23650.0,
        "target_1": 23550.0,
        "target_2": 23500.0,
        "target_3": 23450.0,
        "bars_before_expansion": 5,
        "mfe_points": 120.0,
        "mae_points": 40.0,
        "trade_duration_bars": 35,
        "hit_1r": True,
        "hit_2r": True,
        "hit_3r": True,
        "win": True,
        "realized_pnl_points": 80.0,
        "classification": "Winner",
        "regime": {
            "trend_regime": "trending",
            "vol_regime": "low_vol",
            "gap_regime": "no_gap",
            "composite": "trending|low_vol|no_gap",
        },
        "layers": {
            "layer1": {"events_detected": ["Failed Breakout"]},
            "layer2": {
                "htf_trend": "Bearish",
                "vwap_state": "Below",
                "vwap_gate_passes": True,
                "aligned": True,
            },
            "layer5": {"pass": True, "reason_codes": []},
        },
    }
    base.update(overrides)
    return base


def test_constants() -> None:
    assert "60_100_runner" in RUNNER_STRATEGIES
    assert SLIPPAGE_STRESS_LEVELS == (0, 2, 5, 10)


def test_timing_class() -> None:
    assert _timing_class(10) == "Very Early"
    assert _timing_class(-2) == "Late"


def test_rule_evidence_type() -> None:
    rule = {"validate_signal_count": 6, "validate_pf": 0.08}
    assert _rule_evidence_type(rule, direction="SELL", replay_regime_count=10, total_signals=20) == "replay_verified"
    assert _rule_evidence_type({"validate_signal_count": 1}, direction="BUY", replay_regime_count=0, total_signals=5) == "partial"


def test_part1_evidence_expansion() -> None:
    result = _part1_evidence_expansion(
        buy_signals=[_buy_signal()],
        sell_signals=[_sell_signal()],
        buy_export={"walk_forward": {"validate": {"buy_v3": {"signals_emitted_count": 6}}}},
        sell_export={},
        reality_audit={"evidence_quality": {"is_120d_sufficient": {"verdict": "PARTIAL"}, "required_sample_sizes_by_confidence": {}}},
        deployment_audit={"engine_validation_reconciliation": {"buy_v3": {"win_rate_pct": {"authoritative_for_gates": 72.0}}, "sell_v6": {"win_rate_pct": {"reconciled": 70.0}}}},
        wf_audit={"final_answer": {"primary_degradation_engine": "SELL_V6"}},
        window_days=120,
    )
    assert "confidence_at_horizons" in result
    assert result["would_larger_samples_change_conclusions"]["sell_signal_quality"] == "NO"


def test_part2_regime_throttle() -> None:
    regime_audit = {
        "throttle_recommendation": {
            "sell_v6_regime_throttle": [
                {
                    "regime": "Strong Trend | High Volatility | Gap Compression | Liquidity Compression",
                    "throttle": "BLOCK",
                    "weight": 0.0,
                    "validate_pf": 0.08,
                    "validate_signal_count": 6,
                },
            ],
            "buy_v3_regime_throttle": [],
        },
        "final_answer": {"baseline_sell_v6_validate_pf": 1.44, "throttled_sell_v6_validate_pf": 7.08},
    }
    result = _part2_regime_throttle_reality(
        regime_audit=regime_audit,
        sell_signals=[_sell_signal()],
        buy_signals=[_buy_signal()],
        window_days=120,
    )
    assert len(result["per_rule_analysis"]) == 1
    assert result["per_rule_analysis"][0]["evidence_type"] == "replay_verified"


def test_part5_slippage_stress() -> None:
    live_audit = {
        "final_answer": {
            "optimal_stops": {"buy_v3": "fixed_10", "sell_v6": "fixed_10"},
            "optimal_exit_structures": {"buy_v3": "60/100/Runner", "sell_v6": "60/100/Runner"},
        },
    }
    result = _part5_live_execution_risk(
        buy_signals=[_buy_signal(), _buy_signal(mfe_points=150.0)],
        sell_signals=[_sell_signal()],
        live_audit=live_audit,
        window_days=120,
    )
    assert "0" in result["by_slippage_level"]
    assert result["execution_risk_score"] >= 0


@pytest.fixture
def tmp_research_dir(tmp_path: Path) -> Path:
    research_dir = tmp_path / "outputs" / "research"
    research_dir.mkdir(parents=True)

    buy_signals = [_buy_signal(), _buy_signal(mfe_points=150.0)]
    sell_signals = [_sell_signal(), _sell_signal(mfe_points=200.0)]

    def _write(name: str, payload: dict) -> None:
        (research_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    _write(
        "buy_v3_candidate_validation.json",
        {
            "symbol": "NIFTY50",
            "timeframe": "5M",
            "trading_days_replayed": 120,
            "replay_start_date": "2026-01-05",
            "replay_end_date": "2026-07-02",
            "walk_forward": {"validate": {"buy_v3": {"signals_emitted_count": 6}}},
            "per_signal_details": {"buy_v3": buy_signals},
        },
    )
    _write(
        "sell_v6_replay_validation.json",
        {
            "symbol": "NIFTY50",
            "timeframe": "5M",
            "trading_days_replayed": 120,
            "per_signal_details": {"sell_v6": sell_signals},
        },
    )
    _write(
        "unified_production_replay_validation.json",
        {"per_signal_details": {"buy_v3": buy_signals}},
    )
    _write(
        "regime_detection_audit.json",
        {
            "throttle_recommendation": {
                "sell_v6_regime_throttle": [
                    {
                        "regime": "Strong Trend | High Volatility | Gap Compression | Liquidity Compression",
                        "throttle": "BLOCK",
                        "weight": 0.0,
                        "validate_pf": 0.08,
                        "validate_signal_count": 6,
                    },
                ],
                "buy_v3_regime_throttle": [],
            },
            "final_answer": {
                "baseline_sell_v6_validate_pf": 1.44,
                "throttled_sell_v6_validate_pf": 7.08,
                "throttle_restores_validate_pf_2_plus": True,
            },
        },
    )
    _write(
        "production_trading_playbook_audit.json",
        {"production_scores": {"production_readiness_score": 82.0}},
    )
    _write(
        "live_trade_management_execution_efficiency_audit.json",
        {
            "capture_leakage": {
                "buy_v3": {"miss_reason_ranking": [{"reason": "timing", "count": 1, "rank": 1}]},
                "sell_v6": {"miss_reason_ranking": [{"reason": "runner", "count": 1, "rank": 1}]},
            },
            "final_answer": {
                "optimal_stops": {"buy_v3": "fixed_10", "sell_v6": "fixed_10"},
                "optimal_exit_structures": {"buy_v3": "60/100/Runner", "sell_v6": "60/100/Runner"},
            },
        },
    )
    _write(
        "production_reality_audit.json",
        {
            "trading_days_replayed": 120,
            "evidence_quality": {
                "is_120d_sufficient": {"verdict": "PARTIAL", "buy_v3": True, "sell_v6": True},
                "required_sample_sizes_by_confidence": {
                    "80": {"buy_v3_wr": 100, "sell_v6_wr": 200, "combined_min": 150},
                },
                "current_confidence_pct": {"combined_estimate": 66.0},
            },
            "execution_bottleneck_audit": {
                "primary_bottleneck": "runner",
                "bottleneck_ranking": [{"bottleneck": "runner", "contribution_pct": 40.0, "rank": 1}],
            },
            "production_scores": {
                "production_readiness_score": 72.0,
                "confidence_score": 66.2,
                "production_risk_score": 68.5,
                "evidence_score": 84.9,
                "deployment_tier": "Production Candidate",
                "capture_summary": {
                    "current_capture_pct": 38.54,
                    "max_achievable_capture_pct": 41.42,
                    "improvement_potential_capture_pct": 2.88,
                },
            },
            "production_truth_audit": {
                "aggregate_evidence_score": 84.9,
                "evidence_scores": {"buy_v3": 85.0, "sell_v6": 90.0},
            },
            "final_answer": {
                "paper_trade_tomorrow": "YES",
                "real_capital_deployment": "NO",
                "should_research_buy_v4": "NO",
                "should_research_sell_v7": "NO",
                "can_expectancy_improve_without_buy_v4_sell_v7": "YES",
                "evidence_score": 84.9,
                "improvement_potential_capture_pct": 2.88,
            },
            "signal_reality": {
                "buy_v3": {"predictive_vs_reactive": {"verdict": "PREDICTIVE"}},
                "sell_v6": {"predictive_vs_reactive": {"verdict": "PREDICTIVE"}},
            },
        },
    )
    _write(
        "final_production_deployment_audit.json",
        {
            "engine_validation_reconciliation": {
                "buy_v3": {"win_rate_pct": {"authoritative_for_gates": 72.0}},
                "sell_v6": {"win_rate_pct": {"reconciled": 70.0}},
            },
            "final_answer": {
                "still_unverified": ["Live slippage test"],
                "paper_trade_tomorrow": "YES",
                "real_capital_deployment": "NO",
            },
        },
    )
    _write(
        "regime_aware_execution_validation.json",
        {
            "loss_root_cause": {"primary_cause": "target", "cause_ranking": [{"cause": "runner", "pct": 40.0}]},
            "final_answer": {
                "highest_impact_remaining_improvement": "Improve runner trail giveback",
                "should_research_buy_v4": "NO",
                "should_research_sell_v7": "NO",
            },
        },
    )
    _write(
        "walk_forward_failure_root_cause_audit.json",
        {
            "final_answer": {
                "primary_degradation_engine": "SELL_V6",
                "top_root_cause": "regime shift",
            },
        },
    )
    return research_dir


def test_export_synthetic(tmp_research_dir: Path, tmp_path: Path) -> None:
    report_path = tmp_research_dir / "production_readiness_closure_audit.json"
    research = ProductionReadinessClosureAuditResearch(report_path=report_path)

    import src.research.production_readiness_closure_audit_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_sources = module.SOURCE_EXPORTS.copy()
    original_refs = module.REFERENCE_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = tmp_research_dir
        module.SOURCE_EXPORTS = {k: tmp_research_dir / Path(v).name for k, v in module.SOURCE_EXPORTS.items()}
        module.REFERENCE_EXPORTS = {k: tmp_research_dir / Path(v).name for k, v in module.REFERENCE_EXPORTS.items()}
        exported = research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.SOURCE_EXPORTS = original_sources
        module.REFERENCE_EXPORTS = original_refs

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Production Readiness Closure Audit"
    assert payload["methodology"]["synthesis_only"] is True
    assert "part1_evidence_expansion" in payload
    assert "part2_regime_throttle_reality" in payload
    assert "part3_runner_optimization" in payload
    assert "part4_trade_lifecycle" in payload
    assert "part5_live_execution_risk" in payload
    assert "part6_research_closure" in payload
    assert len(payload["top_risks"]) == 10
    assert len(payload["top_opportunities"]) == 10
    final = payload["final_answer"]
    assert final["paper_trading_verdict"] in {"YES", "NO", "PARTIAL"}
    assert final["real_capital_verdict"] in {"YES", "NO", "PARTIAL"}
    assert final["should_research_buy_v4"] in {"YES", "NO"}
    assert final["should_research_sell_v7"] in {"YES", "NO"}
    assert final["evidence_score"] > 0


@pytest.mark.skipif(
    not Path("outputs/research/production_reality_audit.json").exists(),
    reason="Requires completed replay exports",
)
def test_export_real_data() -> None:
    research = ProductionReadinessClosureAuditResearch()
    path = research.export()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["part4_trade_lifecycle"]["buy_v3"]["sample_size"] >= 100
    assert payload["final_answer"]["paper_trading_verdict"] == "YES"
    assert payload["final_answer"]["real_capital_verdict"] == "NO"
    assert payload["final_answer"]["should_research_buy_v4"] == "NO"
    assert payload["final_answer"]["should_research_sell_v7"] == "NO"
