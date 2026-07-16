"""Tests for final production deployment audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.final_production_deployment_audit_research import (
    DEPLOYMENT_STOP_VARIANTS,
    FinalProductionDeploymentAuditResearch,
    _audit_pf_calculations,
    _reconcile_buy_v3_wr,
    _resolve_stop_points_extended,
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
    assert "fixed_15" in DEPLOYMENT_STOP_VARIANTS
    assert "structure_based" in DEPLOYMENT_STOP_VARIANTS


def test_resolve_stop_points_extended_fixed_15() -> None:
    assert _resolve_stop_points_extended(_buy_signal(), "fixed_15", cohort_mae_median=50.0) == 15.0


def test_wr_reconciliation_mismatch() -> None:
    signals = [
        _buy_signal(win=True, classification="Real Reversal"),
        _buy_signal(win=True, classification="Bull Trap"),
        _buy_signal(win=False, classification="Real Reversal"),
        _buy_signal(win=True, classification="Range Failure"),
    ]
    tradeability = {
        "engine_comparison": {"buy_v3": {"win_rate_pct": 75.0}},
        "exit_target_optimization": {"by_target": {"60": {"win_rate_pct": 95.0}}},
    }
    playbook = {"buy_v3_playbook": {"baseline_replay_metrics": {"win_rate_pct": 50.0}}}

    result = _reconcile_buy_v3_wr(
        buy_signals=signals,
        buy_tradeability=tradeability,
        playbook=playbook,
    )
    assert result["headline_mismatch"]["high_wr_pct"] == 75.0
    assert result["headline_mismatch"]["low_wr_pct"] == 50.0
    assert result["authoritative_wr_for_production_gates"] == 75.0
    assert "Real Reversal" in result["reconciliation_verdict"]


def test_pf_audit_flags_critical() -> None:
    flags = _audit_pf_calculations(
        buy_tradeability={
            "combined_engine_simulation": {"combined_metrics": {"profit_factor": 12.38}},
            "tradeability_tier_metrics": {"by_tier": {"200": {"profit_factor": 61.53}}},
            "engine_comparison": {"buy_v3": {"profit_factor": 4.21}},
        },
        sell_v6={
            "comparison_table": {"sell_v6": {"profit_factor": 4.09}},
            "walk_forward": {"train": {"sell_v6": {"profit_factor": 5.21}}},
        },
        wf_audit={
            "walk_forward_comparison": {
                "split": {"validate": {"buy_v3": {"profit_factor": 1235.25, "signals_emitted": 6}}},
            },
            "output_metrics": {"combined_validate_pf": 1.73},
        },
        regime_audit={
            "final_answer": {"baseline_sell_v6_validate_pf": 1.44, "throttled_sell_v6_validate_pf": 7.08},
            "throttle_recommendation": {
                "sell_v6_regime_throttle": [
                    {
                        "regime": "Test Regime",
                        "validate_pf": 2190.0,
                        "validate_signal_count": 8,
                        "throttle": "FULL",
                    },
                ],
            },
        },
        playbook={"capital_curve_proxy": {"profit_factor": 633.7}},
    )
    assert flags["flag_count"] >= 3
    assert flags["critical_count"] >= 1


def test_generate_report(tmp_path: Path) -> None:
    research_dir = tmp_path / "outputs" / "research"
    research_dir.mkdir(parents=True)

    buy_signals = [
        _buy_signal(),
        _buy_signal(win=False, classification="Bull Trap", mfe_points=20.0, realized_pnl_points=-20.0),
    ]
    sell_signals = [_sell_signal(), _sell_signal(win=False, mfe_points=10.0, realized_pnl_points=-30.0)]

    (research_dir / "buy_v3_candidate_validation.json").write_text(
        json.dumps(
            {
                "symbol": "NIFTY50",
                "timeframe": "5M",
                "trading_days_replayed": 120,
                "replay_start_date": "2026-01-05",
                "replay_end_date": "2026-07-02",
                "per_signal_details": {"buy_v3": buy_signals},
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "unified_production_replay_validation.json").write_text(
        json.dumps({"per_signal_details": {"buy_v3": buy_signals}}),
        encoding="utf-8",
    )
    (research_dir / "sell_v6_replay_validation.json").write_text(
        json.dumps(
            {
                "symbol": "NIFTY50",
                "timeframe": "5M",
                "trading_days_replayed": 120,
                "comparison_table": {
                    "sell_v6": {
                        "signals_per_month": 61.6,
                        "win_rate_pct": 70.24,
                        "profit_factor": 4.09,
                        "expectancy": 131.15,
                    },
                    "point_capture": {"sell_v6": {"40": {"capture_rate_pct": 56.23}}},
                },
                "walk_forward": {"train": {"sell_v6": {"profit_factor": 5.21}}},
                "per_signal_details": {"sell_v6": sell_signals},
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "buy_v3_tradeability_production_validation.json").write_text(
        json.dumps(
            {
                "engine_comparison": {
                    "buy_v3": {
                        "signals_per_month": 21.27,
                        "win_rate_pct": 72.41,
                        "profit_factor": 4.21,
                        "expectancy": 158.65,
                    },
                },
                "exit_target_optimization": {"by_target": {"60": {"win_rate_pct": 95.69}}},
                "lead_time_analysis": {"before_expansion_pct": 96.3},
                "tradeability_tier_metrics": {"by_tier": {}},
                "final_answers": {"optimal_target_tier_points": 60},
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "walk_forward_failure_root_cause_audit.json").write_text(
        json.dumps(
            {
                "output_metrics": {
                    "combined_train_pf": 4.77,
                    "combined_validate_pf": 1.73,
                    "walk_forward_stable": False,
                    "production_readiness_score": 62.0,
                    "production_risk_score": 76.0,
                    "confidence_score": 68.0,
                },
                "final_answer": {
                    "timing_summary": {
                        "buy_v3_before_momentum_pct": 89.66,
                        "buy_v3_same_candle_pct": 3.45,
                        "sell_v6_before_momentum_pct": 77.08,
                    },
                    "primary_degradation_engine": "SELL_V6",
                    "combined_paper_trading": "PARTIAL",
                },
                "degradation_classification": {"classification": "Regime-Specific"},
                "walk_forward_comparison": {
                    "split": {"validate": {"buy_v3": {"profit_factor": 1235.25, "signals_emitted": 6}}},
                },
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "regime_detection_audit.json").write_text(
        json.dumps(
            {
                "methodology": {"synthesis_only": True},
                "throttle_recommendation": {
                    "sell_v6_regime_throttle": [
                        {
                            "regime": "Strong Trend | High Volatility | Gap Compression | Liquidity Compression",
                            "throttle": "BLOCK",
                            "validate_pf": 0.08,
                            "validate_signal_count": 6,
                        },
                    ],
                    "buy_v3_regime_throttle": [],
                },
                "final_answer": {
                    "paper_trading_verdict": "YES",
                    "buy_v3_paper_trading": "YES",
                    "sell_v6_paper_trading_unthrottled": "NO",
                    "sell_v6_paper_trading_throttled": "YES",
                    "combined_paper_trading_throttled": "YES",
                    "baseline_sell_v6_validate_pf": 1.44,
                    "throttled_sell_v6_validate_pf": 7.08,
                    "throttle_restores_validate_pf_2_plus": True,
                    "output_metrics": {
                        "production_readiness_score": 82.0,
                        "production_risk_score": 61.0,
                        "confidence_score": 76.0,
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "production_trading_playbook_audit.json").write_text(
        json.dumps(
            {
                "buy_v3_playbook": {
                    "baseline_replay_metrics": {
                        "signals_per_month": 21.27,
                        "win_rate_pct": 56.03,
                        "profit_factor": 4.21,
                    },
                    "per_signal_distribution": {"median_mfe": 200.0, "median_mae": 80.0},
                    "signal_execution_rules": {},
                    "target_rules": {"recommended_structure": "40/80/Runner", "recommended_single_target_points": 60},
                    "stop_rules": {"recommended_variant": "fixed_10"},
                },
                "sell_v6_playbook": {
                    "baseline_replay_metrics": {
                        "signals_per_month": 61.6,
                        "win_rate_pct": 70.24,
                        "profit_factor": 4.09,
                    },
                    "per_signal_distribution": {"median_mfe": 180.0, "median_mae": 70.0},
                    "signal_execution_rules": {},
                    "target_rules": {"recommended_structure": "40/80/Runner"},
                    "stop_rules": {"recommended_variant": "fixed_10"},
                },
                "combined_playbook": {
                    "signal_execution_rules": {"conflict_policy": "NO_TRADE"},
                    "capital_allocation_rules": {"buy_sizing_mode": "regime_adaptive", "sell_sizing_mode": "regime_adaptive"},
                    "regime_rules": {"import_source": "regime_detection_audit.json"},
                    "risk_rules": {
                        "buy": {"risk_per_trade_points": 100.0, "daily_loss_limit_points": 300.0, "daily_profit_lock_points": 900.0, "max_concurrent_positions": 2},
                        "sell": {"risk_per_trade_points": 90.0, "daily_loss_limit_points": 280.0, "daily_profit_lock_points": 400.0, "max_concurrent_positions": 3},
                        "portfolio_daily_loss_limit_points": 580.0,
                    },
                    "target_rules": {"buy_structure": "40/80/Runner", "sell_structure": "40/80/Runner"},
                    "stop_rules": {"buy_variant": "fixed_10", "sell_variant": "fixed_10"},
                },
                "target_structure_comparison": {},
                "position_sizing_comparison": {"buy_v3": {}, "sell_v6": {}},
                "production_scores": {},
                "capital_curve_proxy": {"profit_factor": 633.7},
                "final_answer": {"evidence": {"combined_expected_signals_per_month": 70.77}},
            },
        ),
        encoding="utf-8",
    )

    report_path = research_dir / "final_production_deployment_audit.json"
    research = FinalProductionDeploymentAuditResearch(report_path=report_path)

    import src.research.final_production_deployment_audit_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_exports = module.SOURCE_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = research_dir
        module.SOURCE_EXPORTS = {
            "buy_v3_candidate_validation": research_dir / "buy_v3_candidate_validation.json",
            "buy_v3_tradeability_production_validation": research_dir
            / "buy_v3_tradeability_production_validation.json",
            "sell_v6_replay_validation": research_dir / "sell_v6_replay_validation.json",
            "unified_production_replay_validation": research_dir
            / "unified_production_replay_validation.json",
            "walk_forward_failure_root_cause_audit": research_dir
            / "walk_forward_failure_root_cause_audit.json",
            "regime_detection_audit": research_dir / "regime_detection_audit.json",
            "production_trading_playbook_audit": research_dir / "production_trading_playbook_audit.json",
        }
        exported = research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.SOURCE_EXPORTS = original_exports

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Final Production Deployment Audit"
    assert payload["methodology"]["research_only"] is True
    assert payload["methodology"]["synthesis_only"] is True
    assert "buy_v3_wr_reconciliation" in payload
    assert "pf_audit" in payload
    assert "deployment_playbook" in payload
    assert payload["final_answer"]["paper_trade_tomorrow"] in {"YES", "NO", "PARTIAL"}
    assert payload["final_answer"]["real_capital_deployment"] in {"YES", "NO", "PARTIAL"}
    assert payload["final_answer"]["deployment_tier"] in {"Production Candidate", "Paper Trading Only", "Research Only"}
    assert len(payload["conclusions"]) >= 5


@pytest.mark.skipif(
    not Path("outputs/research/buy_v3_candidate_validation.json").exists(),
    reason="Requires completed replay exports",
)
def test_export_real_data() -> None:
    research = FinalProductionDeploymentAuditResearch()
    path = research.export()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["buy_v3_wr_reconciliation"]["headline_mismatch"]["high_wr_pct"] == pytest.approx(72.41, abs=0.5)
    assert payload["buy_v3_wr_reconciliation"]["headline_mismatch"]["low_wr_pct"] == pytest.approx(56.03, abs=0.5)
