"""Tests for live trade management execution efficiency audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.live_trade_management_execution_efficiency_audit_research import (
    EXECUTION_STOP_VARIANTS,
    EXIT_STRUCTURES,
    LiveTradeManagementExecutionEfficiencyAuditResearch,
    _capture_leakage_analysis,
    _entry_efficiency_analysis,
    _exit_structure_matrix,
    _mfe_mae_distribution,
    _resolve_stop_extended,
    _stop_quality_matrix,
    _timing_bucket,
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
            "layer2": {"htf_trend": "Bullish", "vwap_state": "Reclaimed", "ema_structure": "Bull Stack", "aligned": True},
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
    assert "fixed_15" in EXECUTION_STOP_VARIANTS
    assert "40/80/Runner" in EXIT_STRUCTURES


def test_resolve_stop_extended_fixed_15() -> None:
    assert _resolve_stop_extended(_buy_signal(), "fixed_15", cohort_mae_median=50.0) == 15.0


def test_mfe_mae_distribution() -> None:
    result = _mfe_mae_distribution([_buy_signal(), _buy_signal(mfe_points=100.0, mae_points=30.0)])
    assert result["sample_size"] == 2
    assert result["avg_mfe"] == 90.0
    assert result["median_mfe"] == 90.0


def test_entry_efficiency_analysis() -> None:
    result = _entry_efficiency_analysis([_buy_signal()], side="BUY")
    assert result["aggregate"]["avg_points_lost_before_entry"] == 12.5
    assert result["aggregate"]["avg_points_captured_after_entry"] == 80.0
    assert 0 < result["aggregate"]["avg_entry_efficiency_pct"] <= 100.0


def test_timing_bucket() -> None:
    signals = [
        _buy_signal(bars_before_expansion=5),
        _buy_signal(bars_before_expansion=0),
        _buy_signal(bars_before_expansion=-1),
    ]
    result = _timing_bucket(signals)
    assert result["before_momentum_pct"] == pytest.approx(33.33, abs=0.1)
    assert result["at_momentum_pct"] == pytest.approx(33.33, abs=0.1)
    assert result["after_momentum_pct"] == pytest.approx(33.33, abs=0.1)


def test_stop_quality_matrix() -> None:
    signals = [_buy_signal(), _buy_signal(mfe_points=15.0, mae_points=25.0, win=False)]
    result = _stop_quality_matrix(signals, structure=EXIT_STRUCTURES["40/80/Runner"], window_days=120)
    assert result["best_stop_variant"] in EXECUTION_STOP_VARIANTS
    assert "expectancy" in result["best_stop_evidence"]


def test_exit_structure_matrix() -> None:
    signals = [_sell_signal(), _sell_signal(mfe_points=30.0, mae_points=50.0, win=False)]
    result = _exit_structure_matrix(signals, stop_variant="fixed_20", window_days=120)
    assert result["best_structure"] in EXIT_STRUCTURES
    assert "capture_efficiency_pct" in result["best_structure_evidence"]


def test_capture_leakage_analysis() -> None:
    result = _capture_leakage_analysis(
        [_buy_signal()],
        structure=EXIT_STRUCTURES["40/80/Runner"],
        stop_variant="fixed_20",
    )
    assert result["max_available_points"] == 80.0
    assert result["capture_efficiency_pct"] > 0
    assert len(result["miss_reason_ranking"]) >= 0


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
                "comparison_table": {"sell_v6": {"signals_per_month": 61.6, "profit_factor": 4.09}},
                "per_signal_details": {"sell_v6": sell_signals},
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "production_trading_playbook_audit.json").write_text(
        json.dumps(
            {
                "buy_v3_playbook": {
                    "target_rules": {"recommended_single_target_points": 60, "recommended_structure": "40/80/Runner"},
                },
                "combined_playbook": {
                    "stop_rules": {"buy_variant": "fixed_10", "sell_variant": "fixed_10"},
                    "target_rules": {"buy_structure": "40/80/Runner", "sell_structure": "40/80/Runner"},
                    "risk_rules": {
                        "buy": {"risk_per_trade_points": 50.0, "daily_loss_limit_points": 150.0},
                        "sell": {"risk_per_trade_points": 45.0, "daily_loss_limit_points": 140.0},
                    },
                },
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
                "trade_management": {
                    "playbook_recommendations": {
                        "buy_stop_variant": "structure_based",
                        "sell_stop_variant": "structure_based",
                    },
                },
                "deployment_playbook": {
                    "paper_trading_checklist": ["Enable BUY_V3", "Enable throttled SELL_V6"],
                },
                "production_scores": {
                    "production_readiness_score": 72.0,
                    "confidence_score": 72.0,
                    "production_risk_score": 68.5,
                },
                "final_answer": {
                    "paper_trade_tomorrow": "YES",
                    "real_capital_deployment": "NO",
                    "deployment_tier": "Paper Trading Only",
                },
            },
        ),
        encoding="utf-8",
    )

    report_path = research_dir / "live_trade_management_execution_efficiency_audit.json"
    research = LiveTradeManagementExecutionEfficiencyAuditResearch(report_path=report_path)

    import src.research.live_trade_management_execution_efficiency_audit_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_exports = module.SOURCE_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = research_dir
        module.SOURCE_EXPORTS = {
            "final_production_deployment_audit": research_dir / "final_production_deployment_audit.json",
            "buy_v3_candidate_validation": research_dir / "buy_v3_candidate_validation.json",
            "sell_v6_replay_validation": research_dir / "sell_v6_replay_validation.json",
            "production_trading_playbook_audit": research_dir / "production_trading_playbook_audit.json",
            "regime_detection_audit": research_dir / "regime_detection_audit.json",
        }
        exported = research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.SOURCE_EXPORTS = original_exports

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Live Trade Management & Execution Efficiency Audit"
    assert payload["methodology"]["research_only"] is True
    assert payload["methodology"]["synthesis_only"] is True
    assert "mfe_mae_summary" in payload
    assert "entry_efficiency" in payload
    assert "stop_quality" in payload
    assert "exit_structures" in payload
    assert "capture_leakage" in payload
    assert "regime_trade_management" in payload
    assert "deployment_playbook" in payload
    assert "deployment_reconciliation" in payload
    assert payload["final_answer"]["paper_trading_config"]["deployment_mode"] == "paper_trading"
    assert payload["final_answer"]["real_capital_config"]["deployment_mode"] == "real_capital"
    assert len(payload["conclusions"]) >= 5


@pytest.mark.skipif(
    not Path("outputs/research/buy_v3_candidate_validation.json").exists(),
    reason="Requires completed replay exports",
)
def test_export_real_data() -> None:
    if not Path("outputs/research/final_production_deployment_audit.json").exists():
        pytest.skip("Requires final_production_deployment_audit.json")
    research = LiveTradeManagementExecutionEfficiencyAuditResearch()
    path = research.export()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mfe_mae_summary"]["buy_v3"]["sample_size"] > 0
    assert payload["mfe_mae_summary"]["sell_v6"]["sample_size"] > 0
    assert payload["final_answer"]["optimal_stops"]["buy_v3"] in EXECUTION_STOP_VARIANTS
