"""Tests for production reality audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.production_reality_audit_research import (
    MFE_TIERS,
    RUNNER_STRATEGIES,
    ProductionRealityAuditResearch,
    _execution_bottleneck_audit,
    _mfe_tier_distribution,
    _required_sample_size,
    _signal_reality_analysis,
    _target_achievement_matrix,
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
        "signal_reason_stack": {
            "layer1": ["Failed Breakdown", "Gap Reversal", "Liquidity Grab", "PDL Sweep"],
            "layer2": {"htf_trend": "Bullish", "vwap": "Reclaimed", "location": "Near Support"},
        },
        "layers": {
            "layer1": {
                "events_detected": ["Failed Breakdown", "Gap Reversal", "Liquidity Grab", "PDL Sweep"],
                "formula_events_matched": ["Failed Breakdown", "Gap Reversal", "Liquidity Grab", "PDL Sweep"],
            },
            "layer2": {
                "htf_trend": "Bullish",
                "vwap_state": "Reclaimed",
                "location": "Near Support",
                "aligned": True,
            },
            "layer3": {"confirmation_candle": "Hammer", "volume_bucket": "Normal", "confirmation_optional": True},
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
            "layer1": {"events_detected": ["Failed Breakout"], "primary_event": "Failed Breakout"},
            "layer2": {
                "htf_trend": "Bearish",
                "vwap_state": "Below",
                "vwap_gate_rule": "VWAP Below only",
                "vwap_gate_passes": True,
                "ema_structure": "Bear Context",
                "aligned": True,
            },
            "layer3": {"confirmation_candle": "Evening Star", "volume_bucket": "Normal"},
            "layer5": {"pass": True, "reason_codes": []},
        },
    }
    base.update(overrides)
    return base


def test_constants() -> None:
    assert 40 in MFE_TIERS
    assert "60_100_runner" in RUNNER_STRATEGIES


def test_timing_class() -> None:
    assert _timing_class(10) == "Very Early"
    assert _timing_class(3) == "Early"
    assert _timing_class(0) == "Same"
    assert _timing_class(-2) == "Late"
    assert _timing_class(None) == "No Linked Move"


def test_mfe_tier_distribution() -> None:
    signals = [_buy_signal(mfe_points=100.0), _buy_signal(mfe_points=30.0)]
    result = _mfe_tier_distribution(signals)
    assert result["sample_size"] == 2
    assert result["tiers"]["40"]["count"] == 1
    assert result["tiers"]["20"]["count"] == 2


def test_target_achievement_matrix() -> None:
    structure = RUNNER_STRATEGIES["60_100_runner"]
    result = _target_achievement_matrix(
        [_buy_signal()],
        structure=structure,
        stop_variant="fixed_10",
        window_days=120,
        side="BUY",
    )
    assert result["side"] == "BUY"
    assert "by_tier" in result
    assert result["aggregate"]["max_achievable_points"] > 0


def test_signal_reality_analysis() -> None:
    from src.research.production_edge_enhancement_audit_research import _is_buy_winner

    result = _signal_reality_analysis(
        [_buy_signal(), _buy_signal(bars_before_expansion=-1)],
        side="BUY",
        win_fn=_is_buy_winner,
        window_days=120,
    )
    assert result["timing_class_summary"]["Very Early"]["count"] == 1
    assert result["timing_class_summary"]["Late"]["count"] == 1
    assert "predictive_vs_reactive" in result


def test_required_sample_size() -> None:
    n90 = _required_sample_size(0.7, confidence_pct=90)
    n60 = _required_sample_size(0.7, confidence_pct=60)
    assert n90 > n60
    assert n90 > 100


def test_execution_bottleneck_audit() -> None:
    live_audit = {
        "capture_leakage": {
            "buy_v3": {"miss_reason_ranking": [{"reason": "timing", "count": 10, "rank": 1}]},
            "sell_v6": {"miss_reason_ranking": [{"reason": "runner", "count": 20, "rank": 1}]},
        },
    }
    regime_audit = {
        "loss_root_cause": {
            "cause_ranking": [
                {"cause": "execution", "pct": 40.0, "rank": 1},
                {"cause": "runner", "pct": 30.0, "rank": 2},
            ],
            "primary_cause": "execution",
        },
    }
    result = _execution_bottleneck_audit(
        live_audit=live_audit,
        regime_audit=regime_audit,
        buy_signals=[_buy_signal()],
        sell_signals=[_sell_signal()],
    )
    assert result["primary_bottleneck"] is not None
    assert len(result["bottleneck_ranking"]) == 6


@pytest.fixture
def tmp_research_dir(tmp_path: Path) -> Path:
    research_dir = tmp_path / "outputs" / "research"
    research_dir.mkdir(parents=True)

    buy_signals = [_buy_signal(), _buy_signal(mfe_points=150.0, bars_before_expansion=3)]
    sell_signals = [_sell_signal(), _sell_signal(mfe_points=200.0, bars_before_expansion=8)]

    (research_dir / "buy_v3_candidate_validation.json").write_text(
        json.dumps(
            {
                "symbol": "NIFTY50",
                "timeframe": "5M",
                "trading_days_replayed": 120,
                "replay_start_date": "2026-01-05",
                "replay_end_date": "2026-07-02",
                "walk_forward": {
                    "validate": {
                        "buy_v3": {"signals_emitted_count": 6, "overall_statistics": {"signals_emitted": 6}},
                    },
                },
                "per_signal_details": {"buy_v3": buy_signals},
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "sell_v6_replay_validation.json").write_text(
        json.dumps(
            {
                "symbol": "NIFTY50",
                "timeframe": "5M",
                "trading_days_replayed": 120,
                "replay_start_date": "2026-01-05",
                "replay_end_date": "2026-07-02",
                "comparison_table": {"sell_v6": {"win_rate_pct": 70.24, "profit_factor": 4.09}},
                "per_signal_details": {"sell_v6": sell_signals},
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "final_production_deployment_audit.json").write_text(
        json.dumps(
            {
                "methodology": {"synthesis_only": True},
                "limitations": ["synthetic flag example"],
                "engine_validation_reconciliation": {
                    "buy_v3": {
                        "win_rate_pct": {"authoritative_for_gates": 72.41},
                        "profit_factor": {"reconciled": 4.21},
                    },
                    "sell_v6": {
                        "win_rate_pct": {"reconciled": 70.24},
                        "profit_factor": {"reconciled": 4.09, "validate_unthrottled": 1.44},
                    },
                },
                "production_scores": {
                    "production_readiness_score": 72.0,
                    "confidence_score": 66.2,
                    "production_risk_score": 68.5,
                },
                "final_answer": {
                    "paper_trade_tomorrow": "YES",
                    "real_capital_deployment": "NO",
                    "buy_v3_paper_trading": "YES",
                    "sell_v6_paper_trading_throttled": "YES",
                    "deployment_tier": "Production Candidate",
                },
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "live_trade_management_execution_efficiency_audit.json").write_text(
        json.dumps(
            {
                "methodology": {"synthesis_only": True},
                "capture_leakage": {
                    "buy_v3": {
                        "miss_reason_ranking": [{"reason": "timing", "count": 1, "rank": 1}],
                        "capture_efficiency_pct": 65.0,
                    },
                    "sell_v6": {
                        "miss_reason_ranking": [{"reason": "runner", "count": 1, "rank": 1}],
                        "capture_efficiency_pct": 55.0,
                    },
                },
                "final_answer": {
                    "optimal_stops": {"buy_v3": "fixed_10", "sell_v6": "fixed_10"},
                    "optimal_exit_structures": {"buy_v3": "60/100/Runner", "sell_v6": "60/100/Runner"},
                },
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "regime_aware_execution_validation.json").write_text(
        json.dumps(
            {
                "methodology": {"synthesis_only": True},
                "loss_root_cause": {
                    "cause_ranking": [{"cause": "execution", "pct": 50.0, "rank": 1}],
                    "primary_cause": "execution",
                },
                "final_answer": {
                    "buy_v3_near_optimal": "PARTIAL",
                    "sell_v6_near_optimal": "YES",
                    "regime_aware_execution_improves_pf_expectancy": "PARTIAL",
                },
            },
        ),
        encoding="utf-8",
    )
    return research_dir


def test_export_synthetic(tmp_research_dir: Path, tmp_path: Path) -> None:
    report_path = tmp_research_dir / "production_reality_audit.json"
    research = ProductionRealityAuditResearch(report_path=report_path)

    import src.research.production_reality_audit_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_exports = module.SOURCE_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = tmp_research_dir
        module.SOURCE_EXPORTS = {
            "live_trade_management_execution_efficiency_audit": tmp_research_dir
            / "live_trade_management_execution_efficiency_audit.json",
            "regime_aware_execution_validation": tmp_research_dir / "regime_aware_execution_validation.json",
            "buy_v3_candidate_validation": tmp_research_dir / "buy_v3_candidate_validation.json",
            "sell_v6_replay_validation": tmp_research_dir / "sell_v6_replay_validation.json",
            "final_production_deployment_audit": tmp_research_dir / "final_production_deployment_audit.json",
        }
        exported = research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.SOURCE_EXPORTS = original_exports

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Production Reality Audit"
    assert payload["methodology"]["research_only"] is True
    assert payload["methodology"]["synthesis_only"] is True
    assert "trade_outcome_distribution" in payload
    assert "target_achievement_matrix" in payload
    assert "signal_reality" in payload
    assert "runner_exit_optimization" in payload
    assert "execution_bottleneck_audit" in payload
    assert "evidence_quality" in payload
    assert "production_truth_audit" in payload
    final = payload["final_answer"]
    assert final["can_expectancy_improve_without_buy_v4_sell_v7"] in {"YES", "NO", "PARTIAL"}
    assert final["should_research_buy_v4"] in {"YES", "NO"}
    assert final["should_research_sell_v7"] in {"YES", "NO"}
    assert final["paper_trade_tomorrow"] in {"YES", "NO", "PARTIAL"}
    assert final["real_capital_deployment"] in {"YES", "NO", "PARTIAL"}
    assert "evidence_scores" in final
    assert len(payload["conclusions"]) >= 7


@pytest.mark.skipif(
    not Path("outputs/research/buy_v3_candidate_validation.json").exists(),
    reason="Requires completed replay exports",
)
def test_export_real_data() -> None:
    research = ProductionRealityAuditResearch()
    path = research.export()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["trade_outcome_distribution"]["buy_v3"]["sample_size"] >= 100
    assert payload["final_answer"]["evidence_score"] > 0
