"""Tests for production trading playbook audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.production_trading_playbook_audit_research import (
    BUY_MIN_SIGNALS_PER_MONTH,
    SELL_MIN_SIGNALS_PER_MONTH,
    TARGET_STRUCTURES,
    ProductionTradingPlaybookAuditResearch,
    _extract_entry_rules,
    _signal_distribution_metrics,
    _structure_stop_points,
    _tiered_structure_pnl,
)


def _buy_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-05 09:25:00+05:30",
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
            "layer1": {
                "events_detected": ["Failed Breakout"],
                "primary_event": "Failed Breakout",
            },
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
    assert BUY_MIN_SIGNALS_PER_MONTH == 20.0
    assert SELL_MIN_SIGNALS_PER_MONTH == 60.0
    assert "40/60/100" in TARGET_STRUCTURES


def test_structure_stop_points() -> None:
    assert _structure_stop_points(_buy_signal()) == 50.0


def test_tiered_structure_pnl_runner() -> None:
    pnl, win = _tiered_structure_pnl(
        _sell_signal(mfe_points=150.0),
        TARGET_STRUCTURES["40/80/Runner"],
        stop_pts=30.0,
    )
    assert win is True
    assert pnl > 40.0


def test_tiered_structure_pnl_loss() -> None:
    pnl, win = _tiered_structure_pnl(
        _buy_signal(mfe_points=15.0, mae_points=25.0),
        TARGET_STRUCTURES["40/60/100"],
        stop_pts=20.0,
    )
    assert win is False
    assert pnl < 0


def test_extract_entry_rules() -> None:
    buy_rules = _extract_entry_rules(_buy_signal(), side="BUY")
    sell_rules = _extract_entry_rules(_sell_signal(), side="SELL")
    assert buy_rules["model_id"] == "LDM-BUY-V3"
    assert buy_rules["layer5_gate"]["pass"] is True
    assert sell_rules["layer2_gate"]["vwap_gate_rule"] == "VWAP Below only"


def test_signal_distribution_metrics() -> None:
    metrics = _signal_distribution_metrics([_buy_signal(), _sell_signal()])
    assert metrics["sample_size"] == 2
    assert metrics["average_mfe"] == 100.0
    assert metrics["hit_1r_rate_pct"] == 100.0


def test_generate_report(tmp_path: Path) -> None:
    research_dir = tmp_path / "outputs" / "research"
    research_dir.mkdir(parents=True)

    buy_signals = [_buy_signal(), _buy_signal(mfe_points=30.0, win=False, classification="Bull Trap")]
    sell_signals = [
        _sell_signal(),
        _sell_signal(win=False, mfe_points=15.0, mae_points=90.0, realized_pnl_points=-50.0),
    ]

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
                "per_signal_details": {"sell_v6": sell_signals},
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "unified_production_replay_validation.json").write_text(
        json.dumps({"per_signal_details": {"buy_v3": buy_signals}}),
        encoding="utf-8",
    )
    (research_dir / "regime_detection_audit.json").write_text(
        json.dumps(
            {
                "throttle_recommendation": {
                    "sell_v6_regime_throttle": [
                        {
                            "regime": "Strong Trend | Low Volatility | Gap Compression | Liquidity Compression",
                            "throttle": "FULL",
                            "weight": 1.0,
                        },
                    ],
                    "buy_v3_regime_throttle": [
                        {
                            "regime": "Range | High Volatility | Gap Expansion | Liquidity Expansion",
                            "throttle": "HALF",
                            "weight": 0.5,
                        },
                    ],
                },
                "output_metrics": {
                    "production_readiness_score": 80.0,
                    "production_risk_score": 55.0,
                    "confidence_score": 72.0,
                },
                "final_answer": {
                    "paper_trading_verdict": "YES",
                    "buy_v3_paper_trading": "YES",
                    "sell_v6_paper_trading_unthrottled": "NO",
                    "sell_v6_paper_trading_throttled": "YES",
                    "combined_paper_trading_throttled": "YES",
                    "baseline_sell_v6_validate_pf": 1.44,
                    "throttled_sell_v6_validate_pf": 7.08,
                },
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "buy_v3_tradeability_production_validation.json").write_text(
        json.dumps({"final_answers": {"optimal_target_tier_points": 60}}),
        encoding="utf-8",
    )
    (research_dir / "walk_forward_failure_root_cause_audit.json").write_text("{}", encoding="utf-8")

    report_path = research_dir / "production_trading_playbook_audit.json"
    research = ProductionTradingPlaybookAuditResearch(report_path=report_path)

    import src.research.production_trading_playbook_audit_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_exports = module.SOURCE_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = research_dir
        module.SOURCE_EXPORTS = {
            "buy_v3_candidate_validation": research_dir / "buy_v3_candidate_validation.json",
            "sell_v6_replay_validation": research_dir / "sell_v6_replay_validation.json",
            "regime_detection_audit": research_dir / "regime_detection_audit.json",
            "unified_production_replay_validation": research_dir
            / "unified_production_replay_validation.json",
            "walk_forward_failure_root_cause_audit": research_dir
            / "walk_forward_failure_root_cause_audit.json",
            "buy_v3_tradeability_production_validation": research_dir
            / "buy_v3_tradeability_production_validation.json",
        }
        exported = research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.SOURCE_EXPORTS = original_exports

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Production Trading Playbook Audit"
    assert payload["methodology"]["research_only"] is True
    assert payload["methodology"]["synthesis_only"] is True
    assert "buy_v3_playbook" in payload
    assert "sell_v6_playbook" in payload
    assert "combined_playbook" in payload
    assert "target_structure_comparison" in payload
    assert "stop_loss_optimization" in payload
    assert "position_sizing_comparison" in payload
    assert "regime_deployment" in payload
    assert payload["final_answer"]["paper_trade_tomorrow"] in {"YES", "NO", "PARTIAL"}
    assert len(payload["conclusions"]) >= 5


@pytest.mark.skipif(
    not Path("outputs/research/buy_v3_candidate_validation.json").exists(),
    reason="Requires completed replay exports",
)
def test_generate_report_from_exports() -> None:
    report_path = Path("outputs/research/production_trading_playbook_audit.json")
    research = ProductionTradingPlaybookAuditResearch(report_path=report_path)
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["buy_v3_playbook"]["baseline_replay_metrics"]["sample_size"] >= 100
    assert payload["sell_v6_playbook"]["baseline_replay_metrics"]["sample_size"] >= 300
    assert payload["buy_v3_playbook"]["baseline_replay_metrics"]["signals_per_month"] >= BUY_MIN_SIGNALS_PER_MONTH
    assert payload["sell_v6_playbook"]["baseline_replay_metrics"]["signals_per_month"] >= SELL_MIN_SIGNALS_PER_MONTH
    assert payload["final_answer"]["paper_trade_tomorrow"] in {"YES", "NO", "PARTIAL"}
    assert len(payload["target_structure_comparison"]["buy_v3"]["by_structure"]) == 4
    assert len(payload["stop_loss_optimization"]["sell_v6"]["by_stop_variant"]) == 6
