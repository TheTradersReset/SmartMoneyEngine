"""Tests for BUY_V4 / SELL_V7 final production validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.buy_v4_sell_v7_final_production_validation_research import (
    BuyV4SellV7FinalProductionValidationError,
    BuyV4SellV7FinalProductionValidationResearch,
    _apply_engine_filters,
    _proportion_z_test,
    generate_buy_v4_sell_v7_final_production_validation_report,
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


def test_proportion_z_test() -> None:
    result = _proportion_z_test(30, 100, 45, 100)
    assert "significant" in result
    assert "p_value" in result


def test_apply_engine_filters_reduces() -> None:
    signals = [_buy(), _buy(bar=101, classification="Real Reversal", realized_pnl_points=80, mfe_points=150, mae_points=10)]
    filtered = _apply_engine_filters(
        signals,
        side="BUY",
        reject_patterns=["Liquidity Sweep Failure", "Volatility Collapse"],
        engine_version="BUY_V4",
    )
    assert len(filtered) <= len(signals)


def test_mocked_validation_export(tmp_path: Path) -> None:
    buy = [
        _buy(),
        _buy(bar=101, classification="Real Reversal", realized_pnl_points=90, mfe_points=150, mae_points=15, bars_before_expansion=6),
        _buy(bar=102, classification="No Expansion", realized_pnl_points=-40, mfe_points=10, mae_points=50),
        _buy(bar=103, classification="Real Reversal", realized_pnl_points=70, mfe_points=110, mae_points=12),
        _buy(bar=104, classification="Real Reversal", realized_pnl_points=100, mfe_points=200, mae_points=10),
        _buy(bar=105, classification="Range Failure", realized_pnl_points=15, mfe_points=70, mae_points=35),
        _buy(bar=106, classification="Bull Trap", realized_pnl_points=-100, mfe_points=30, mae_points=140),
        _buy(bar=107, classification="Real Reversal", realized_pnl_points=85, mfe_points=130, mae_points=18),
    ]
    sell = [
        _sell(),
        _sell(bar=91, win=True, realized_pnl_points=100, mfe_points=160, mae_points=20),
        _sell(bar=92, win=False, realized_pnl_points=-50, mfe_points=12, mae_points=40),
        _sell(bar=93, win=True, realized_pnl_points=120, mfe_points=180, mae_points=18),
        _sell(bar=94, win=False, realized_pnl_points=-70, mfe_points=55, mae_points=130),
        _sell(bar=95, win=True, realized_pnl_points=90, mfe_points=140, mae_points=22),
        _sell(bar=96, win=True, realized_pnl_points=110, mfe_points=170, mae_points=25),
        _sell(bar=97, win=False, realized_pnl_points=-80, mfe_points=40, mae_points=160),
    ]
    sources = {
        "buy_v4_sell_v7_design_blueprint_audit": {
            "buy_v4_design": {"selected_patterns": ["Liquidity Sweep Failure", "Gap Continuation"]},
            "sell_v7_design": {"selected_patterns": ["Liquidity Sweep Failure", "Volatility Collapse"]},
        },
        "extended_trade_level_truth_audit": {
            "symbol": "NIFTY50",
            "timeframe": "5M",
            "per_signal_details": {"buy_v3": buy, "sell_v6": sell},
        },
        "extended_evidence_validation_real_deployment_audit": {
            "final_answer": {
                "window_profit_factors": {"250d": 2.1, "500d": 2.1},
                "throttled_pf_500d": 5.5,
                "evidence_score": 81.0,
            },
        },
        "failure_pattern_production_robustness_audit": {
            "production_scores": {"overfitting_risk_score": 34.0},
        },
        "buy_v3_candidate_validation": {},
        "sell_v6_replay_validation": {},
        "regime_detection_audit": {},
    }
    research = BuyV4SellV7FinalProductionValidationResearch()
    report = research.run(sources)
    out = tmp_path / "buy_v4_sell_v7_final_production_validation.json"
    research.export(report, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["report_type"] == "BUY_V4 & SELL_V7 Final Production Validation"
    assert payload["final_answer"]["should_buy_v4_replace_buy_v3"] in {"YES", "NO"}
    assert payload["final_answer"]["should_sell_v7_replace_sell_v6"] in {"YES", "NO"}
    assert "statistical_significance_validation" in payload


def test_generate_raises_without_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research import buy_v4_sell_v7_final_production_validation_research as mod

    monkeypatch.setattr(
        mod,
        "REQUIRED_EXPORTS",
        {"buy_v4_sell_v7_design_blueprint_audit": tmp_path / "missing.json"},
    )
    monkeypatch.setattr(mod, "OPTIONAL_EXPORTS", {})
    with pytest.raises(BuyV4SellV7FinalProductionValidationError):
        generate_buy_v4_sell_v7_final_production_validation_report(report_path=tmp_path / "out.json")
