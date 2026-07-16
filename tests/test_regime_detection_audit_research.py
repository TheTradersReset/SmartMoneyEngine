"""Tests for regime detection and production throttle audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.regime_detection_audit_research import (
    BUY_V3_MODEL_ID,
    SELL_V6_MODEL_ID,
    RegimeDetectionAuditResearch,
    THROTTLE_WEIGHT,
    _optimize_throttle_map,
    _pf_bucket,
    _rank_regimes_by_deterioration,
    classify_signal_regime,
)
from src.research.production_edge_enhancement_audit_research import _timing_label


def _walk_forward() -> dict:
    return {
        "train_trading_days": 80,
        "validate_trading_days": 40,
        "train_start_date": "2026-01-05",
        "train_end_date": "2026-05-05",
        "validate_start_date": "2026-05-06",
        "validate_end_date": "2026-07-02",
    }


def _sell_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-15 11:00:00+05:30",
        "direction": "SELL",
        "bars_before_expansion": 5,
        "mfe_points": 120.0,
        "mae_points": 40.0,
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
            "layer1": {"events_detected": ["Failed Breakout"]},
            "layer2": {
                "htf_trend": "Bearish",
                "ema_structure": "Bear Context",
            },
        },
    }
    base.update(overrides)
    return base


def _buy_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-15 10:00:00+05:30",
        "direction": "BUY",
        "bars_before_expansion": 10,
        "mfe_points": 80.0,
        "mae_points": 20.0,
        "win": True,
        "classification": "Real Reversal",
        "realized_pnl_points": 60.0,
        "layers": {
            "layer1": {"events_detected": ["Failed Breakdown", "Gap Reversal", "PDL Sweep"]},
            "layer2": {
                "htf_trend": "Bullish",
                "ema_structure": "Bull Stack",
                "location": "Near Support",
            },
        },
    }
    base.update(overrides)
    return base


def test_model_constants() -> None:
    assert BUY_V3_MODEL_ID == "LDM-BUY-V3"
    assert SELL_V6_MODEL_ID == "LDM-SELL-V6"
    assert THROTTLE_WEIGHT["BLOCK"] == 0.0


def test_classify_signal_regime_sell() -> None:
    regime = classify_signal_regime(_sell_signal(), direction="SELL")
    assert regime["trend"] == "Strong Trend"
    assert regime["volatility"] == "Low Volatility"
    assert regime["gap"] == "Gap Compression"
    assert "Strong Trend" in regime["composite"]


def test_classify_signal_regime_buy_synthesis() -> None:
    regime = classify_signal_regime(_buy_signal(), direction="BUY")
    assert regime["trend"] in {"Strong Trend", "Weak Trend"}
    assert regime["liquidity"] == "Liquidity Expansion"
    assert regime["export_regime_present"] is False


def test_pf_bucket() -> None:
    assert _pf_bucket(3.5) == "PF>3"
    assert _pf_bucket(2.2) == "PF 2-3"
    assert _pf_bucket(1.5) == "PF<2"
    assert _pf_bucket(0.8) == "PF<1"


def test_rank_regimes_by_deterioration() -> None:
    train = {
        "Strong Trend | Low Volatility | Gap Compression | Liquidity Expansion": {
            "profit_factor": 5.0,
            "expectancy": 150.0,
            "signal_count": 10,
            "pf_bucket": "PF>3",
        }
    }
    validate = {
        "Strong Trend | Low Volatility | Gap Compression | Liquidity Expansion": {
            "profit_factor": 1.2,
            "expectancy": 20.0,
            "signal_count": 8,
            "pf_bucket": "PF<2",
        }
    }
    ranked = _rank_regimes_by_deterioration(train, validate)
    assert ranked[0]["rank"] == 1
    assert ranked[0]["pf_delta"] < 0


def test_optimize_throttle_map_blocks_weak_regime() -> None:
    wf = _walk_forward()
    good = "Strong Trend | Low Volatility | Gap Compression | Liquidity Expansion"
    bad = "Range | High Volatility | Gap Expansion | Liquidity Compression"

    signals = [
        _sell_signal(timestamp="2026-01-15 11:00:00+05:30", realized_pnl_points=100.0),
        _sell_signal(timestamp="2026-02-15 11:00:00+05:30", realized_pnl_points=90.0),
        _sell_signal(timestamp="2026-06-15 11:00:00+05:30", realized_pnl_points=80.0),
        _sell_signal(
            timestamp="2026-06-20 11:00:00+05:30",
            win=False,
            realized_pnl_points=-120.0,
            mfe_points=20.0,
            mae_points=150.0,
            regime={
                "trend_regime": "range",
                "vol_regime": "high_vol",
                "gap_regime": "gap_event",
                "composite": "range|high_vol|gap_event",
            },
            layers={
                "layer1": {"events_detected": ["Gap Reversal", "Failed Breakout"]},
                "layer2": {"htf_trend": "Neutral", "ema_structure": ""},
            },
        ),
    ]

    train_table = {
        good: {"profit_factor": 5.0, "signal_count": 2},
        bad: {"profit_factor": 4.0, "signal_count": 1},
    }
    validate_table = {
        good: {"profit_factor": 2.5, "signal_count": 2},
        bad: {"profit_factor": 0.5, "signal_count": 1},
    }
    throttle_map = _optimize_throttle_map(signals, train_table, validate_table, wf, direction="SELL")
    assert throttle_map[bad] == "BLOCK"


def test_generate_report(tmp_path: Path) -> None:
    research_dir = tmp_path / "outputs" / "research"
    research_dir.mkdir(parents=True)
    wf = _walk_forward()

    buy_signals = [
        _buy_signal(timestamp="2026-01-15 10:00:00+05:30", realized_pnl_points=100.0),
        _buy_signal(
            timestamp="2026-02-15 10:00:00+05:30",
            classification="Bull Trap",
            win=False,
            realized_pnl_points=-30.0,
        ),
        _buy_signal(timestamp="2026-06-15 10:00:00+05:30", realized_pnl_points=50.0),
    ]
    sell_signals = [
        _sell_signal(timestamp="2026-01-20 11:00:00+05:30", realized_pnl_points=90.0),
        _sell_signal(timestamp="2026-02-20 11:00:00+05:30", realized_pnl_points=85.0),
        _sell_signal(timestamp="2026-06-20 11:00:00+05:30", realized_pnl_points=40.0),
        _sell_signal(
            timestamp="2026-06-25 11:00:00+05:30",
            win=False,
            realized_pnl_points=-60.0,
            mfe_points=80.0,
            mae_points=150.0,
            classification="Bear Trap",
        ),
    ]

    unified_export = {
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "trading_days_replayed": 120,
        "replay_start_date": "2026-01-05",
        "replay_end_date": "2026-07-02",
        "walk_forward": wf,
        "per_signal_details": {"buy_v3": buy_signals, "sell_v5": sell_signals},
    }
    sell_v6_export = {
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "trading_days_replayed": 120,
        "walk_forward": {
            **wf,
            "train": {"sell_v6": {"profit_factor": 5.0, "signals_emitted": 2}},
            "validate": {"sell_v6": {"profit_factor": 1.4, "signals_emitted": 2}},
        },
        "per_signal_details": {"sell_v6": sell_signals},
    }
    buy_v3_export = {
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "trading_days_replayed": 120,
        "walk_forward": wf,
        "per_signal_details": {"buy_v3": buy_signals},
    }
    wf_audit = {
        "root_cause_probability": {
            "probabilities_pct": {"regime_change": 48.0, "volatility_shift": 29.0},
            "root_cause_ranking": [
                {"cause": "regime_change", "probability_pct": 48.0},
                {"cause": "volatility_shift", "probability_pct": 29.0},
            ],
            "top_root_cause": "regime_change",
        },
        "output_metrics": {
            "confidence_score": 68.0,
            "production_risk_score": 76.0,
            "production_readiness_score": 62.0,
        },
    }

    (research_dir / "unified_production_replay_validation.json").write_text(
        json.dumps(unified_export),
        encoding="utf-8",
    )
    (research_dir / "sell_v6_replay_validation.json").write_text(json.dumps(sell_v6_export), encoding="utf-8")
    (research_dir / "buy_v3_candidate_validation.json").write_text(json.dumps(buy_v3_export), encoding="utf-8")
    (research_dir / "buy_v3_tradeability_production_validation.json").write_text(json.dumps({}), encoding="utf-8")
    (research_dir / "walk_forward_failure_root_cause_audit.json").write_text(json.dumps(wf_audit), encoding="utf-8")

    report_path = research_dir / "regime_detection_audit.json"
    research = RegimeDetectionAuditResearch(report_path=report_path)

    import src.research.regime_detection_audit_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_exports = module.SOURCE_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = research_dir
        module.SOURCE_EXPORTS = {
            "walk_forward_failure_root_cause_audit": research_dir / "walk_forward_failure_root_cause_audit.json",
            "sell_v6_replay_validation": research_dir / "sell_v6_replay_validation.json",
            "unified_production_replay_validation": research_dir / "unified_production_replay_validation.json",
            "buy_v3_candidate_validation": research_dir / "buy_v3_candidate_validation.json",
            "buy_v3_tradeability_production_validation": research_dir
            / "buy_v3_tradeability_production_validation.json",
        }
        exported = research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.SOURCE_EXPORTS = original_exports

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Regime Detection & Production Throttle Audit"
    assert payload["methodology"]["synthesis_only"] is True
    assert "regime_ranking" in payload
    assert "throttle_recommendation" in payload
    assert "throttled_impact_estimate" in payload
    assert payload["final_answer"]["paper_trading_verdict"] in {"YES", "NO", "PARTIAL"}
    assert "throttle_restores_validate_pf_2_plus" in payload["final_answer"]
    assert len(payload["conclusions"]) >= 5


@pytest.mark.skipif(
    not Path("outputs/research/sell_v6_replay_validation.json").exists(),
    reason="Requires completed replay exports",
)
def test_generate_report_from_exports() -> None:
    report_path = Path("outputs/research/regime_detection_audit.json")
    research = RegimeDetectionAuditResearch(report_path=report_path)
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["engines"] == ["SELL_V6", "BUY_V3", "COMBINED"]
    assert payload["walk_forward_comparison"]["sell_v6"]["degraded"] is True
    assert payload["throttled_impact_estimate"]["sell_v6_validate_baseline"]["profit_factor"] is not None
    assert payload["final_answer"]["paper_trading_verdict"] in {"YES", "NO", "PARTIAL"}
    assert payload["output_metrics"]["confidence_score"] > 0
    assert len(payload["throttle_recommendation"]["sell_v6_regime_throttle"]) > 0
