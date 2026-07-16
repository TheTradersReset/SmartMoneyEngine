"""Tests for BUY_V4 / SELL_V7 design justification audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.buy_v4_sell_v7_design_justification_audit_research import (
    AUTHORITATIVE_WINDOWS,
    BUY_DESIGN_CLASSES,
    SELL_DESIGN_CLASSES,
    BuyV4SellV7DesignJustificationAuditError,
    BuyV4SellV7DesignJustificationAuditResearch,
    _classify_buy_design,
    _classify_sell_design,
    _confidence_pct,
    _focus_decision,
    _nature_of_class,
    generate_buy_v4_sell_v7_design_justification_audit_report,
)


def _buy_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-06 09:30:00+05:30",
        "bar": 100,
        "direction": "BUY",
        "entry": 23500.0,
        "classification": "Bull Trap",
        "realized_pnl_points": -80.0,
        "mfe_points": 40.0,
        "mae_points": 120.0,
        "win": False,
        "layers": {
            "layer1": {"events_detected": ["Failed Breakdown", "Liquidity Grab"]},
            "layer2": {"htf_trend": "Bullish", "vwap_state": "Reclaimed"},
        },
        "signal_reason_stack": {"layer1": ["Failed Breakdown"], "layer2": {}},
    }
    base.update(overrides)
    return base


def _sell_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-05 10:25:00+05:30",
        "bar": 90,
        "direction": "SELL",
        "entry": 23600.0,
        "classification": "Bear Trap",
        "realized_pnl_points": -90.0,
        "mfe_points": 50.0,
        "mae_points": 150.0,
        "win": False,
        "bars_before_expansion": 2,
        "layers": {
            "layer1": {"events_detected": ["Failed Breakout"]},
            "layer2": {"htf_trend": "Bearish", "vwap_state": "Below"},
        },
        "signal_reason_stack": {"layer1": ["Failed Breakout"], "layer2": {"htf_trend": "Bearish"}},
    }
    base.update(overrides)
    return base


def test_constants() -> None:
    assert 240 in AUTHORITATIVE_WINDOWS
    assert 250 in AUTHORITATIVE_WINDOWS
    assert 500 in AUTHORITATIVE_WINDOWS
    assert "Bull Trap" in BUY_DESIGN_CLASSES
    assert "Bear Trap" in SELL_DESIGN_CLASSES


def test_classify_buy_design_bull_trap() -> None:
    assert _classify_buy_design(_buy_signal()) == "Bull Trap"


def test_classify_buy_design_winner() -> None:
    assert _classify_buy_design(_buy_signal(classification="Real Reversal", realized_pnl_points=60.0)) == "Winner"


def test_classify_sell_design_bear_trap() -> None:
    assert _classify_sell_design(_sell_signal()) == "Bear Trap"


def test_nature_of_class() -> None:
    assert _nature_of_class(frequency_pct=24.0, pf_impact_pct=100.0, label="Bull Trap") == "Structural"
    assert _nature_of_class(frequency_pct=2.0, pf_impact_pct=1.0, label="PDL Failure") == "Sample Noise"
    assert _nature_of_class(frequency_pct=8.0, pf_impact_pct=20.0, label="Gap Failure") == "Regime Specific"


def test_confidence_pct() -> None:
    score = _confidence_pct(sample_size=234, best_pf_improvement=100.0, structural_count=2, longer_window_pf=2.19)
    assert 70.0 <= score <= 95.0


def test_focus_decision_both() -> None:
    focus = _focus_decision(
        {"recommendation": "YES"},
        {"recommendation": "YES"},
    )
    assert focus["focus_code"] == "C"
    assert focus["focus_label"] == "Both"


def test_mocked_run_exports(tmp_path: Path) -> None:
    buy_signals = [
        _buy_signal(),
        _buy_signal(bar=101, classification="Real Reversal", realized_pnl_points=80.0, mfe_points=120.0, mae_points=20.0),
        _buy_signal(bar=102, classification="No Expansion", realized_pnl_points=-70.0, mfe_points=10.0, mae_points=80.0),
        _buy_signal(bar=103, classification="Real Reversal", realized_pnl_points=90.0, mfe_points=140.0, mae_points=15.0),
    ]
    sell_signals = [
        _sell_signal(),
        _sell_signal(bar=91, win=True, realized_pnl_points=100.0, mfe_points=150.0, mae_points=20.0),
        _sell_signal(
            bar=92,
            win=False,
            realized_pnl_points=-60.0,
            mfe_points=15.0,
            mae_points=40.0,
            bars_before_expansion=1,
        ),
        _sell_signal(bar=93, win=True, realized_pnl_points=110.0, mfe_points=160.0, mae_points=25.0),
    ]

    extended_trade = {
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "available_trading_days": 247,
        "per_signal_details": {"buy_v3": buy_signals, "sell_v6": sell_signals},
        "core_metrics_by_window": {
            "240": {
                "buy_v3": {"signals_per_month": 20.0, "max_drawdown_points": 200.0},
                "sell_v6": {"signals_per_month": 50.0, "max_drawdown_points": 300.0},
            },
        },
        "uncaptured_edge": {
            "max_window": {
                "buy_v3": {"additional_available": {"capture_delta_pct": 3.0}},
                "sell_v6": {"additional_available": {"capture_delta_pct": 3.0}},
            },
        },
    }
    extended_evidence = {
        "final_answer": {
            "window_profit_factors": {"250d": 2.19, "500d": 2.19},
            "throttled_pf_500d": 5.79,
        },
    }

    research = BuyV4SellV7DesignJustificationAuditResearch()
    report = research.run(
        {
            "extended_trade_level_truth_audit": extended_trade,
            "extended_evidence_validation_real_deployment_audit": extended_evidence,
            "trade_level_truth_audit": {},
            "buy_v3_candidate_validation": {},
            "sell_v6_replay_validation": {},
        },
    )
    out = tmp_path / "buy_v4_sell_v7_design_justification_audit.json"
    research.export(report, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["report_type"] == "BUY_V4 & SELL_V7 Design Justification Audit"
    assert "part1_buy_v4_justification" in payload
    assert "part2_sell_v7_justification" in payload
    assert payload["final_answer"]["should_future_work_focus_on"] in {"A", "B", "C", "D"}
    assert payload["methodology"]["ignore_120d_for_verdict"] is True


def test_generate_raises_without_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.research import buy_v4_sell_v7_design_justification_audit_research as mod

    monkeypatch.setattr(mod, "REQUIRED_EXPORTS", {"extended_trade_level_truth_audit": tmp_path / "missing.json"})
    monkeypatch.setattr(mod, "OPTIONAL_EXPORTS", {})
    with pytest.raises(BuyV4SellV7DesignJustificationAuditError):
        generate_buy_v4_sell_v7_design_justification_audit_report(report_path=tmp_path / "out.json")
