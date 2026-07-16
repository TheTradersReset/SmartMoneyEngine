"""Tests for regime-aware execution validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.regime_aware_execution_validation_research import (
    EXECUTION_STOP_VARIANTS,
    EXIT_STRUCTURES,
    RegimeAwareExecutionValidationResearch,
    _best_stop_exit_for_cohort,
    _entry_precision_analysis,
    _execution_failure_audit,
    _group_signals_by_regime,
    _regime_dimension_counts,
    _regime_performance_row,
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
        "classification": "Real Reversal",
        "realized_pnl_points": 60.0,
        "layers": {
            "layer1": {"events_detected": ["Failed Breakdown"], "formula_events_matched": ["Failed Breakdown"]},
            "layer2": {
                "htf_trend": "Bullish",
                "vwap_state": "Reclaimed",
                "ema_structure": "Bull Stack",
                "aligned": True,
            },
            "layer3": {"confirmation_candle": "Hammer"},
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
        "points_before_expansion": 8.0,
        "mfe_points": 120.0,
        "mae_points": 40.0,
        "trade_duration_bars": 35,
        "hit_1r": True,
        "hit_2r": True,
        "hit_3r": True,
        "win": True,
        "realized_pnl_points": 80.0,
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
                "ema_structure": "Bear Context",
                "aligned": True,
            },
            "layer3": {"confirmation_candle": "Evening Star"},
            "layer5": {"pass": True, "reason_codes": []},
        },
    }
    base.update(overrides)
    return base


def test_constants() -> None:
    assert "fixed_10" in EXECUTION_STOP_VARIANTS
    assert "60/100/Runner" in EXIT_STRUCTURES


def test_regime_dimension_counts() -> None:
    result = _regime_dimension_counts([_buy_signal(), _sell_signal()], direction="BUY")
    assert result["total_signals"] == 2
    assert "by_trend" in result
    assert "by_composite" in result


def test_group_signals_by_regime() -> None:
    grouped = _group_signals_by_regime([_buy_signal(), _sell_signal()], direction="BUY")
    assert len(grouped) >= 1


def test_regime_performance_row() -> None:
    from src.research.production_edge_enhancement_audit_research import _is_buy_winner

    row = _regime_performance_row(
        [_buy_signal()],
        window_days=120,
        win_fn=_is_buy_winner,
        structure=EXIT_STRUCTURES["60/100/Runner"],
        stop_variant="fixed_10",
    )
    assert row["signal_count"] == 1
    assert row["avg_mfe"] == 80.0


def test_execution_failure_audit() -> None:
    from src.research.production_edge_enhancement_audit_research import _is_buy_winner

    result = _execution_failure_audit(
        [_buy_signal(), _buy_signal(mfe_points=15.0, mae_points=25.0, win=False)],
        structure=EXIT_STRUCTURES["40/80/Runner"],
        win_fn=_is_buy_winner,
        window_days=120,
    )
    assert result["best_stop_variant"] in EXECUTION_STOP_VARIANTS
    assert "stop_hit_pct" in result["by_stop_variant"]["fixed_10"]


def test_entry_precision_analysis() -> None:
    result = _entry_precision_analysis([_buy_signal()], direction="BUY")
    assert result["aggregate"]["avg_points_lost_before_entry"] == 12.5
    assert result["timing_class_summary"]["Early"]["count"] == 1


def test_best_stop_exit_for_cohort() -> None:
    result = _best_stop_exit_for_cohort([_sell_signal(), _sell_signal(mfe_points=30.0, mae_points=50.0, win=False)], window_days=120)
    assert result["best_stop_variant"] in EXECUTION_STOP_VARIANTS
    assert result["best_tiered_exit"] in EXIT_STRUCTURES


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
    (research_dir / "sell_v6_replay_validation.json").write_text(
        json.dumps(
            {
                "symbol": "NIFTY50",
                "timeframe": "5M",
                "trading_days_replayed": 120,
                "comparison_table": {"sell_v6": {"signals_per_month": 61.6, "profit_factor": 4.09, "win_rate_pct": 70.0}},
                "per_signal_details": {"sell_v6": sell_signals},
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "regime_detection_audit.json").write_text(
        json.dumps(
            {
                "throttle_recommendation": {
                    "sell_v6_regime_throttle": [
                        {
                            "regime": "Strong Trend | High Volatility | Gap Compression | Liquidity Compression",
                            "throttle": "BLOCK",
                            "validate_pf": 0.08,
                        },
                    ],
                },
                "final_answer": {
                    "paper_trading_verdict": "YES",
                    "sell_v6_validate_pf_throttled": 7.08,
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
    (research_dir / "final_production_deployment_audit.json").write_text(
        json.dumps(
            {
                "deployment_playbook": {
                    "paper_trading_checklist": ["Enable BUY_V3", "Enable throttled SELL_V6"],
                    "risk_rules": {"buy": {"risk_per_trade_points": 50.0}, "sell": {"risk_per_trade_points": 45.0}},
                },
                "production_scores": {
                    "production_readiness_score": 72.0,
                    "confidence_score": 72.0,
                    "production_risk_score": 68.5,
                },
                "final_answer": {
                    "paper_trade_tomorrow": "YES",
                    "real_capital_deployment": "NO",
                    "deployment_tier": "Production Candidate",
                },
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "live_trade_management_execution_efficiency_audit.json").write_text(
        json.dumps(
            {
                "final_answer": {
                    "optimal_stops": {"buy_v3": "fixed_10", "sell_v6": "fixed_10"},
                    "optimal_exit_structures": {"buy_v3": "60/100/Runner", "sell_v6": "60/100/Runner"},
                    "expected_monthly_points": {"paper_combined": 1000.0, "real_capital_combined": 500.0},
                },
            },
        ),
        encoding="utf-8",
    )

    report_path = research_dir / "regime_aware_execution_validation.json"
    research = RegimeAwareExecutionValidationResearch(report_path=report_path)

    import src.research.regime_aware_execution_validation_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_exports = module.SOURCE_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = research_dir
        module.SOURCE_EXPORTS = {
            "live_trade_management_execution_efficiency_audit": research_dir
            / "live_trade_management_execution_efficiency_audit.json",
            "regime_detection_audit": research_dir / "regime_detection_audit.json",
            "final_production_deployment_audit": research_dir / "final_production_deployment_audit.json",
            "buy_v3_candidate_validation": research_dir / "buy_v3_candidate_validation.json",
            "sell_v6_replay_validation": research_dir / "sell_v6_replay_validation.json",
        }
        exported = research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.SOURCE_EXPORTS = original_exports

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Regime Aware Execution Validation"
    assert payload["methodology"]["research_only"] is True
    assert payload["methodology"]["synthesis_only"] is True
    assert "regime_classification" in payload
    assert "per_regime_performance" in payload
    assert "execution_failure_audit" in payload
    assert "entry_precision" in payload
    assert "capture_leakage" in payload
    assert "regime_aware_playbook" in payload
    assert "paper_vs_real_configs" in payload
    assert payload["final_answer"]["regime_aware_execution_improves_pf_expectancy"] in {"YES", "NO", "PARTIAL"}
    assert payload["final_answer"]["buy_v3_near_optimal"] in {"YES", "NO", "PARTIAL"}
    assert payload["final_answer"]["sell_v6_near_optimal"] in {"YES", "NO", "PARTIAL"}
    assert len(payload["conclusions"]) >= 5


@pytest.mark.skipif(
    not Path("outputs/research/buy_v3_candidate_validation.json").exists(),
    reason="Requires completed replay exports",
)
def test_export_real_data() -> None:
    required = [
        "live_trade_management_execution_efficiency_audit.json",
        "regime_detection_audit.json",
        "final_production_deployment_audit.json",
        "buy_v3_candidate_validation.json",
        "sell_v6_replay_validation.json",
    ]
    for name in required:
        if not Path(f"outputs/research/{name}").exists():
            pytest.skip(f"Requires {name}")

    research = RegimeAwareExecutionValidationResearch()
    path = research.export()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["regime_classification"]["buy_v3"]["total_signals"] > 0
    assert payload["regime_classification"]["sell_v6"]["total_signals"] > 0
    assert payload["final_answer"]["highest_impact_remaining_improvement"]
