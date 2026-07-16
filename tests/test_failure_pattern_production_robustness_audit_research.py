"""Tests for failure pattern & production robustness audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.failure_pattern_production_robustness_audit_research import (
    BUY_FAILURE_CLASSES,
    SELL_FAILURE_CLASSES,
    STRUCTURAL_PATTERNS,
    FailurePatternProductionRobustnessAuditError,
    FailurePatternProductionRobustnessAuditResearch,
    _classify_buy_failure,
    _classify_sell_failure,
    _detect_structural_patterns,
    generate_failure_pattern_production_robustness_audit_report,
)


def _buy(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-06 09:30:00+05:30",
        "bar": 100,
        "direction": "BUY",
        "entry": 23500.0,
        "stop_loss": 23490.0,
        "classification": "Bull Trap",
        "realized_pnl_points": -80.0,
        "mfe_points": 35.0,
        "mae_points": 120.0,
        "bars_before_expansion": 4,
        "win": False,
        "layers": {
            "layer1": {"events_detected": ["Failed Breakdown", "Liquidity Grab"]},
            "layer2": {"htf_trend": "Bullish", "vwap_state": "Below", "location": "Near Support"},
        },
        "signal_reason_stack": {"layer1": ["Failed Breakdown"], "layer2": {}},
    }
    base.update(overrides)
    return base


def _sell(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-05 10:25:00+05:30",
        "bar": 90,
        "direction": "SELL",
        "entry": 23600.0,
        "stop_loss": 23610.0,
        "classification": "Bear Trap",
        "realized_pnl_points": -90.0,
        "mfe_points": 45.0,
        "mae_points": 150.0,
        "bars_before_expansion": 3,
        "win": False,
        "layers": {
            "layer1": {"events_detected": ["Failed Breakout"]},
            "layer2": {"htf_trend": "Bearish", "vwap_state": "Below"},
        },
        "signal_reason_stack": {"layer1": ["Failed Breakout"], "layer2": {"htf_trend": "Bearish"}},
    }
    base.update(overrides)
    return base


def test_constants() -> None:
    assert "Bull Trap" in BUY_FAILURE_CLASSES
    assert "Bear Trap" in SELL_FAILURE_CLASSES
    assert "Weak Displacement" in STRUCTURAL_PATTERNS


def test_classify_buy_failure() -> None:
    assert _classify_buy_failure(_buy()) == "Bull Trap"
    assert _classify_buy_failure(_buy(classification="Real Reversal", realized_pnl_points=50)) == "Winner"


def test_classify_sell_failure() -> None:
    assert _classify_sell_failure(_sell()) == "Bear Trap"
    assert _classify_sell_failure(_sell(win=True, realized_pnl_points=80, mfe_points=120, mae_points=20)) == "Winner"


def test_detect_structural_patterns() -> None:
    patterns = _detect_structural_patterns(_buy(), side="BUY")
    assert "Volatility Collapse" in patterns or "Weak Displacement" in patterns or "Liquidity Sweep Failure" in patterns


def test_mocked_full_run(tmp_path: Path) -> None:
    buy = [
        _buy(),
        _buy(bar=101, classification="Real Reversal", realized_pnl_points=90, mfe_points=150, mae_points=15, bars_before_expansion=6),
        _buy(bar=102, classification="No Expansion", realized_pnl_points=-40, mfe_points=10, mae_points=50, bars_before_expansion=2),
        _buy(bar=103, classification="Real Reversal", realized_pnl_points=70, mfe_points=110, mae_points=12, bars_before_expansion=3),
        _buy(bar=104, classification="Range Failure", realized_pnl_points=20, mfe_points=80, mae_points=40, bars_before_expansion=1),
        _buy(bar=105, classification="Real Reversal", realized_pnl_points=100, mfe_points=200, mae_points=10, bars_before_expansion=8),
    ]
    sell = [
        _sell(),
        _sell(bar=91, win=True, realized_pnl_points=100, mfe_points=160, mae_points=20, bars_before_expansion=5),
        _sell(bar=92, win=False, realized_pnl_points=-50, mfe_points=12, mae_points=40, bars_before_expansion=-1),
        _sell(bar=93, win=True, realized_pnl_points=120, mfe_points=180, mae_points=18, bars_before_expansion=4),
        _sell(bar=94, win=False, realized_pnl_points=-70, mfe_points=55, mae_points=130, bars_before_expansion=2),
        _sell(bar=95, win=True, realized_pnl_points=90, mfe_points=140, mae_points=22, bars_before_expansion=7),
    ]
    extended_trade = {
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "per_signal_details": {"buy_v3": buy, "sell_v6": sell},
        "core_metrics_by_window": {
            "240": {
                "trading_days": 240,
                "buy_v3": {"profit_factor": 1.8, "signals_per_month": 20, "max_drawdown_points": 500},
                "sell_v6": {"profit_factor": 2.2, "signals_per_month": 50, "max_drawdown_points": 400},
            },
        },
        "final_answer": {"stop_loss_validation": {"best_stop": "fixed_10"}},
    }
    extended_evidence = {
        "final_answer": {
            "window_profit_factors": {"120d": 4.0, "250d": 2.1, "500d": 2.1},
            "throttled_pf_500d": 5.5,
            "evidence_score": 81.0,
            "production_readiness_score": 72.0,
        },
    }
    research = FailurePatternProductionRobustnessAuditResearch()
    report = research.run(
        {
            "extended_trade_level_truth_audit": extended_trade,
            "extended_evidence_validation_real_deployment_audit": extended_evidence,
            "buy_v3_candidate_validation": {"per_signal_details": {"buy_v3": buy[:4]}},
            "sell_v6_replay_validation": {"per_signal_details": {"sell_v6": sell[:4]}},
            "production_reality_audit": {},
            "production_gap_closure_audit": {},
            "buy_v4_sell_v7_design_justification_audit": {},
            "regime_detection_audit": {},
        },
    )
    out = tmp_path / "failure_pattern_production_robustness_audit.json"
    research.export(report, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Failure Pattern & Production Robustness Audit"
    assert "failure_pattern_root_cause_audit" in payload
    assert "final_answer" in payload
    assert payload["final_answer"]["3_buy_v4_verdict"] in {"YES", "NO"}
    assert payload["final_answer"]["4_sell_v7_verdict"] in {"YES", "NO"}


def test_generate_raises_without_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research import failure_pattern_production_robustness_audit_research as mod

    monkeypatch.setattr(mod, "REQUIRED_EXPORTS", {"extended_trade_level_truth_audit": tmp_path / "missing.json"})
    monkeypatch.setattr(mod, "OPTIONAL_EXPORTS", {})
    with pytest.raises(FailurePatternProductionRobustnessAuditError):
        generate_failure_pattern_production_robustness_audit_report(report_path=tmp_path / "out.json")
