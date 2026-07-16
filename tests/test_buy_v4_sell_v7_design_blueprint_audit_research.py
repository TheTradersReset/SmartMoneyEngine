"""Tests for BUY_V4 / SELL_V7 design blueprint audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.buy_v4_sell_v7_design_blueprint_audit_research import (
    BuyV4SellV7DesignBlueprintAuditError,
    BuyV4SellV7DesignBlueprintAuditResearch,
    generate_buy_v4_sell_v7_design_blueprint_audit_report,
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
        "points_before_expansion": 5.0,
        "trade_duration_bars": 12,
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
        "points_before_expansion": 4.0,
        "trade_duration_bars": 12,
        "win": False,
        "layers": {
            "layer1": {"events_detected": ["Failed Breakout"]},
            "layer2": {"htf_trend": "Bearish", "vwap_state": "Below"},
        },
        "signal_reason_stack": {"layer1": ["Failed Breakout"], "layer2": {"htf_trend": "Bearish"}},
    }
    base.update(overrides)
    return base


def test_mocked_blueprint_export(tmp_path: Path) -> None:
    buy = [
        _buy(),
        _buy(bar=101, classification="Real Reversal", realized_pnl_points=90, mfe_points=150, mae_points=15, bars_before_expansion=6),
        _buy(bar=102, classification="No Expansion", realized_pnl_points=-40, mfe_points=10, mae_points=50),
        _buy(bar=103, classification="Real Reversal", realized_pnl_points=70, mfe_points=110, mae_points=12, bars_before_expansion=3),
        _buy(bar=104, classification="Real Reversal", realized_pnl_points=100, mfe_points=200, mae_points=10, bars_before_expansion=8),
        _buy(bar=105, classification="Range Failure", realized_pnl_points=15, mfe_points=70, mae_points=35),
    ]
    sell = [
        _sell(),
        _sell(bar=91, win=True, realized_pnl_points=100, mfe_points=160, mae_points=20, bars_before_expansion=5),
        _sell(bar=92, win=False, realized_pnl_points=-50, mfe_points=12, mae_points=40, bars_before_expansion=-1),
        _sell(bar=93, win=True, realized_pnl_points=120, mfe_points=180, mae_points=18),
        _sell(bar=94, win=False, realized_pnl_points=-70, mfe_points=55, mae_points=130),
        _sell(bar=95, win=True, realized_pnl_points=90, mfe_points=140, mae_points=22, bars_before_expansion=7),
    ]
    extended_trade = {
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "per_signal_details": {"buy_v3": buy, "sell_v6": sell},
        "core_metrics_by_window": {
            "240": {
                "trading_days": 240,
                "buy_v3": {
                    "profit_factor": 1.8,
                    "signals_per_month": 20,
                    "max_drawdown_points": 500,
                    "capture_efficiency_pct": 37.0,
                },
                "sell_v6": {
                    "profit_factor": 2.2,
                    "signals_per_month": 50,
                    "max_drawdown_points": 400,
                    "capture_efficiency_pct": 38.0,
                },
            },
        },
    }
    extended_evidence = {
        "final_answer": {
            "window_profit_factors": {"120d": 4.0, "250d": 2.1, "500d": 2.1},
            "throttled_pf_500d": 5.5,
            "evidence_score": 81.0,
        },
    }
    failure = {
        "final_answer": {
            "3_buy_v4_verdict": "YES",
            "4_sell_v7_verdict": "YES",
            "10_highest_roi_improvement_remaining": "Regime Detection",
        },
        "production_scores": {"confidence_score": 85.0},
    }
    research = BuyV4SellV7DesignBlueprintAuditResearch()
    report = research.run(
        {
            "extended_trade_level_truth_audit": extended_trade,
            "extended_evidence_validation_real_deployment_audit": extended_evidence,
            "failure_pattern_production_robustness_audit": failure,
            "buy_v3_candidate_validation": {"per_signal_details": {"buy_v3": buy[:4]}},
            "sell_v6_replay_validation": {"per_signal_details": {"sell_v6": sell[:4]}},
            "buy_v4_sell_v7_design_justification_audit": {},
            "regime_detection_audit": {},
        },
    )
    out = tmp_path / "buy_v4_sell_v7_design_blueprint_audit.json"
    research.export(report, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["report_type"] == "BUY_V4 & SELL_V7 Design Blueprint Audit"
    assert payload["final_answer"]["should_buy_v4_replace_buy_v3"] in {"YES", "NO"}
    assert payload["final_answer"]["should_sell_v7_replace_sell_v6"] in {"YES", "NO"}
    assert "research_closure_verdict" in payload
    assert payload["methodology"]["contrast_only_windows"] == [120]


def test_generate_raises_without_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research import buy_v4_sell_v7_design_blueprint_audit_research as mod

    monkeypatch.setattr(
        mod,
        "REQUIRED_EXPORTS",
        {"extended_trade_level_truth_audit": tmp_path / "missing.json"},
    )
    monkeypatch.setattr(mod, "OPTIONAL_EXPORTS", {})
    with pytest.raises(BuyV4SellV7DesignBlueprintAuditError):
        generate_buy_v4_sell_v7_design_blueprint_audit_report(report_path=tmp_path / "out.json")
