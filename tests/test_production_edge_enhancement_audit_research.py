"""Tests for production edge enhancement audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.production_edge_enhancement_audit_research import (
    BUY_MIN_SIGNALS_PER_MONTH,
    LOSER_CLASSIFICATIONS,
    SELL_MIN_SIGNALS_PER_MONTH,
    ProductionEdgeEnhancementAuditResearch,
    _classify_sell_signal,
    _extract_sell_conditions,
    _map_buy_audit_classification,
    _timing_label,
    _winner_loser_side_analysis,
)


def _buy_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-05 09:25:00+05:30",
        "bars_before_expansion": 10,
        "points_before_expansion": 12.5,
        "mfe_points": 80.0,
        "mae_points": 20.0,
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
                "events_at_bar": ["Failed Breakdown"],
                "formula_events_matched": ["Failed Breakdown", "Gap Reversal", "Liquidity Grab", "PDL Sweep"],
            },
            "layer2": {
                "htf_trend": "Bullish",
                "vwap_state": "Reclaimed",
                "location": "Near Support",
            },
        },
    }
    base.update(overrides)
    return base


def _sell_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-05 10:25:00+05:30",
        "mfe_points": 120.0,
        "mae_points": 40.0,
        "win": True,
        "realized_pnl_points": 80.0,
        "signal_reason_stack": {
            "layer1": ["Failed Breakout", "Gap Continuation"],
            "layer2": {"htf_trend": "Bearish", "vwap": "Below", "location": "Near Resistance"},
            "layer3": {"confirmation_candle": "Evening Star"},
        },
        "layers": {
            "layer1": {
                "events_detected": ["Failed Breakout", "Gap Continuation"],
                "primary_event": "Failed Breakout",
                "failed_breakout_present": True,
            },
            "layer2": {
                "htf_trend": "Bearish",
                "vwap_state": "Below",
                "ema_structure": "Bear Context",
            },
            "layer3": {
                "confirmation_candle": "Evening Star",
                "volume_bucket": "Normal",
            },
        },
    }
    base.update(overrides)
    return base


def test_constants() -> None:
    assert BUY_MIN_SIGNALS_PER_MONTH == 20.0
    assert SELL_MIN_SIGNALS_PER_MONTH == 60.0
    assert "Bull Trap" in LOSER_CLASSIFICATIONS
    assert "Bear Trap" in LOSER_CLASSIFICATIONS


def test_map_buy_audit_classification() -> None:
    assert _map_buy_audit_classification("Real Reversal") == "Winner"
    assert _map_buy_audit_classification("Bull Trap") == "Bull Trap"
    assert _map_buy_audit_classification("Counter Trend Bounce") == "Trend Exhaustion"


def test_timing_label() -> None:
    assert _timing_label(5) == "Early"
    assert _timing_label(0) == "Same Candle"
    assert _timing_label(-2) == "Delayed"
    assert _timing_label(None) == "No Linked Move"


def test_classify_sell_signal() -> None:
    assert _classify_sell_signal(_sell_signal()) == "Winner"
    assert _classify_sell_signal(_sell_signal(win=False, mfe_points=10.0, mae_points=50.0)) == "No Expansion"
    assert _classify_sell_signal(
        _sell_signal(win=False, mfe_points=80.0, mae_points=150.0, layers={"layer2": {"htf_trend": "Bearish"}}),
    ) == "Bear Trap"


def test_extract_sell_conditions() -> None:
    conditions = _extract_sell_conditions(_sell_signal())
    assert conditions["Failed Breakout"] is True
    assert conditions["HTF Bearish"] is True
    assert conditions["VWAP Below"] is True
    assert conditions["Near Resistance"] is True
    assert conditions["Confirmation Present"] is True


def test_winner_loser_side_analysis() -> None:
    signals = [
        _buy_signal(),
        _buy_signal(classification="Bull Trap", win=False, mfe_points=30.0, realized_pnl_points=-20.0),
    ]
    for signal in signals:
        signal["audit_classification"] = _map_buy_audit_classification(signal["classification"])
        signal["conditions"] = {}
    analysis = _winner_loser_side_analysis(
        signals,
        side="BUY_V3",
        window_days=120,
        is_winner_fn=lambda s: s.get("classification") == "Real Reversal",
        classify_fn=lambda s: s["audit_classification"],
    )
    assert analysis["baseline"]["sample_size"] == 2
    assert analysis["winners"]["sample_size"] == 1
    assert analysis["losers"]["sample_size"] == 1


def test_generate_report(tmp_path: Path) -> None:
    research_dir = tmp_path / "outputs" / "research"
    research_dir.mkdir(parents=True)

    buy_signals = [
        _buy_signal(),
        _buy_signal(classification="Bull Trap", win=False, mfe_points=35.0, realized_pnl_points=-20.0),
        _buy_signal(classification="Range Failure", win=False, mfe_points=45.0, realized_pnl_points=-15.0),
    ]
    sell_signals = [
        _sell_signal(),
        _sell_signal(win=False, mfe_points=15.0, mae_points=90.0, realized_pnl_points=-50.0),
        _sell_signal(win=False, mfe_points=70.0, mae_points=130.0, realized_pnl_points=-40.0),
    ]

    unified_export = {
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "trading_days_replayed": 120,
        "replay_start_date": "2026-01-05",
        "replay_end_date": "2026-07-02",
        "walk_forward": {
            "train_trading_days": 80,
            "validate_trading_days": 40,
            "train_start_date": "2026-01-05",
            "train_end_date": "2026-05-05",
            "validate_start_date": "2026-05-06",
            "validate_end_date": "2026-07-02",
            "stable": True,
        },
        "engine_comparison": {
            "buy_v3_only": {"point_capture_bullish": {"40": {"capture_rate_pct": 16.0}}},
            "sell_v5_only": {"point_capture_bearish": {"40": {"capture_rate_pct": 59.0}}},
        },
        "per_signal_details": {"buy_v3": buy_signals, "sell_v5": sell_signals},
    }
    v3_export = {
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "trading_days_replayed": 120,
        "replay_start_date": "2026-01-05",
        "replay_end_date": "2026-07-02",
        "per_signal_details": {"buy_v3": buy_signals},
    }
    v5_export = {
        "comparison": {
            "v5_candidate": {
                "overall_statistics": {"signals_per_month": 69.0, "profit_factor": 3.3},
                "point_capture": {"40": {"capture_rate_pct": 59.0}},
            },
        },
    }

    (research_dir / "unified_production_replay_validation.json").write_text(
        json.dumps(unified_export),
        encoding="utf-8",
    )
    (research_dir / "buy_v3_candidate_validation.json").write_text(json.dumps(v3_export), encoding="utf-8")
    (research_dir / "smartmoneyengine_v5_candidate_validation.json").write_text(
        json.dumps(v5_export),
        encoding="utf-8",
    )

    report_path = research_dir / "production_edge_enhancement_audit.json"
    research = ProductionEdgeEnhancementAuditResearch(report_path=report_path)

    import src.research.production_edge_enhancement_audit_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_exports = module.SOURCE_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = research_dir
        module.SOURCE_EXPORTS = {
            "buy_v3_candidate_validation": research_dir / "buy_v3_candidate_validation.json",
            "smartmoneyengine_v5_candidate_validation": research_dir
            / "smartmoneyengine_v5_candidate_validation.json",
            "unified_production_replay_validation": research_dir
            / "unified_production_replay_validation.json",
            "buy_v3_signal_quality_audit": research_dir / "buy_v3_signal_quality_audit.json",
            "buy_v3_tradeability_production_validation": research_dir
            / "buy_v3_tradeability_production_validation.json",
            "buy_winner_vs_false_reversal_analysis": research_dir
            / "buy_winner_vs_false_reversal_analysis.json",
            "smartmoneyengine_walkforward_validation": research_dir
            / "smartmoneyengine_walkforward_validation.json",
        }
        exported = research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.SOURCE_EXPORTS = original_exports

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Production Edge Enhancement Audit"
    assert payload["methodology"]["research_only"] is True
    assert payload["methodology"]["synthesis_only"] is True
    assert "buy_v3_winner_loser_analysis" in payload
    assert "sell_v5_winner_loser_analysis" in payload
    assert "condition_rankings" in payload
    assert "proposed_filters" in payload
    assert "timing_analysis" in payload
    assert "walk_forward_impact" in payload
    assert payload["final_answer"]["can_production_engine_improve_further"] in {"YES", "NO", "PARTIAL"}
    assert len(payload["conclusions"]) >= 5


@pytest.mark.skipif(
    not Path("outputs/research/unified_production_replay_validation.json").exists(),
    reason="Requires completed replay exports",
)
def test_generate_report_from_exports() -> None:
    report_path = Path("outputs/research/production_edge_enhancement_audit.json")
    research = ProductionEdgeEnhancementAuditResearch(report_path=report_path)
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["buy_v3_winner_loser_analysis"]["baseline"]["sample_size"] >= 100
    assert payload["sell_v5_winner_loser_analysis"]["baseline"]["sample_size"] >= 300
    assert payload["buy_v3_winner_loser_analysis"]["baseline"]["signals_per_month"] >= BUY_MIN_SIGNALS_PER_MONTH
    assert payload["sell_v5_winner_loser_analysis"]["baseline"]["signals_per_month"] >= SELL_MIN_SIGNALS_PER_MONTH
    assert payload["final_answer"]["can_production_engine_improve_further"] in {"YES", "NO", "PARTIAL"}
    assert len(payload["proposed_filters"]["buy_v3"]["simulations"]) == 8
    assert len(payload["proposed_filters"]["sell_v5"]["simulations"]) == 8
