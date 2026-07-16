"""Tests for walk-forward failure root cause audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.walk_forward_failure_root_cause_audit_research import (
    BUY_V3_MODEL_ID,
    LOSER_CLASSIFICATIONS,
    SELL_V6_MODEL_ID,
    WalkForwardFailureRootCauseAuditResearch,
    _classify_buy_failure,
    _classify_sell_failure,
    _compare_train_validate,
    _split_signals_by_walk_forward,
    _timing_distribution,
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


def _buy_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-15 10:00:00+05:30",
        "bars_before_expansion": 10,
        "mfe_points": 80.0,
        "mae_points": 20.0,
        "win": True,
        "classification": "Real Reversal",
        "realized_pnl_points": 60.0,
    }
    base.update(overrides)
    return base


def _sell_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-15 11:00:00+05:30",
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
    }
    base.update(overrides)
    return base


def test_model_constants() -> None:
    assert BUY_V3_MODEL_ID == "LDM-BUY-V3"
    assert SELL_V6_MODEL_ID == "LDM-SELL-V6"
    assert "Execution Timing Failure" in LOSER_CLASSIFICATIONS


def test_timing_label() -> None:
    assert _timing_label(5) == "Early"
    assert _timing_label(0) == "Same Candle"
    assert _timing_label(-2) == "Delayed"
    assert _timing_label(None) == "No Linked Move"


def test_classify_buy_failure_execution_timing() -> None:
    assert _classify_buy_failure(_buy_signal(classification="Real Reversal")) == "Winner"
    assert (
        _classify_buy_failure(_buy_signal(classification="Bull Trap", bars_before_expansion=-1))
        == "Execution Timing Failure"
    )


def test_classify_sell_failure() -> None:
    assert _classify_sell_failure(_sell_signal()) == "Winner"
    assert _classify_sell_failure(_sell_signal(classification="Late Entry")) == "Execution Timing Failure"


def test_split_signals_by_walk_forward() -> None:
    wf = _walk_forward()
    signals = [
        _buy_signal(timestamp="2026-01-15 10:00:00+05:30"),
        _buy_signal(timestamp="2026-06-15 10:00:00+05:30"),
    ]
    train, validate = _split_signals_by_walk_forward(signals, wf)
    assert len(train) == 1
    assert len(validate) == 1


def test_compare_train_validate_degraded() -> None:
    result = _compare_train_validate(
        {"profit_factor": 4.0, "win_rate_pct": 70.0, "expectancy": 150.0, "signals_per_month": 30.0},
        {"profit_factor": 1.2, "win_rate_pct": 60.0, "expectancy": 20.0, "signals_per_month": 45.0},
    )
    assert result["degraded"] is True
    assert result["degradation_severity"] == "severe"


def test_timing_distribution() -> None:
    signals = [
        _buy_signal(bars_before_expansion=10),
        _buy_signal(bars_before_expansion=0),
        _buy_signal(bars_before_expansion=None),
    ]
    dist = _timing_distribution(signals)
    assert dist["early_count"] == 1
    assert dist["same_candle_count"] == 1
    assert dist["no_linked_move_count"] == 1


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
            mfe_points=30.0,
        ),
        _buy_signal(timestamp="2026-06-15 10:00:00+05:30", realized_pnl_points=50.0),
    ]
    sell_signals = [
        _sell_signal(timestamp="2026-01-20 11:00:00+05:30", realized_pnl_points=90.0),
        _sell_signal(
            timestamp="2026-02-20 11:00:00+05:30",
            win=False,
            realized_pnl_points=-50.0,
            mfe_points=15.0,
            mae_points=90.0,
            classification="No Expansion",
        ),
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
        "walk_forward": {
            **wf,
            "train": {
                "combined": {"profit_factor": 4.0, "win_rate_pct": 70.0, "signals_emitted": 4},
                "buy_v3": {"profit_factor": 3.5, "win_rate_pct": 68.0, "signals_emitted": 2},
                "sell_v5": {"profit_factor": 4.2, "win_rate_pct": 71.0, "signals_emitted": 2},
            },
            "validate": {
                "combined": {"profit_factor": 1.3, "win_rate_pct": 55.0, "signals_emitted": 2},
                "buy_v3": {"profit_factor": 2.0, "win_rate_pct": 100.0, "signals_emitted": 1},
                "sell_v5": {"profit_factor": 1.1, "win_rate_pct": 50.0, "signals_emitted": 1},
            },
            "stable": False,
        },
        "engine_comparison": {
            "buy_v3_only": {"point_capture_bullish": {"40": {"capture_rate_pct": 16.0}}},
        },
        "per_signal_details": {"buy_v3": buy_signals, "sell_v5": sell_signals},
    }
    sell_v6_export = {
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "trading_days_replayed": 120,
        "walk_forward": {
            **wf,
            "train": {"sell_v6": {"profit_factor": 5.0, "win_rate_pct": 72.0, "signals_emitted": 2}},
            "validate": {"sell_v6": {"profit_factor": 1.4, "win_rate_pct": 65.0, "signals_emitted": 2}},
            "v6_improves_validate_pf": True,
        },
        "comparison_table": {
            "point_capture": {"sell_v6": {"40": {"capture_rate_pct": 56.0}}},
        },
        "per_signal_details": {"sell_v6": sell_signals},
    }
    buy_v3_export = {
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "trading_days_replayed": 120,
        "walk_forward": unified_export["walk_forward"],
        "per_signal_details": {"buy_v3": buy_signals},
    }

    (research_dir / "unified_production_replay_validation.json").write_text(
        json.dumps(unified_export),
        encoding="utf-8",
    )
    (research_dir / "sell_v6_replay_validation.json").write_text(json.dumps(sell_v6_export), encoding="utf-8")
    (research_dir / "buy_v3_candidate_validation.json").write_text(json.dumps(buy_v3_export), encoding="utf-8")

    report_path = research_dir / "walk_forward_failure_root_cause_audit.json"
    research = WalkForwardFailureRootCauseAuditResearch(report_path=report_path)

    import src.research.walk_forward_failure_root_cause_audit_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_exports = module.SOURCE_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = research_dir
        module.SOURCE_EXPORTS = {
            "unified_production_replay_validation": research_dir / "unified_production_replay_validation.json",
            "sell_v6_replay_validation": research_dir / "sell_v6_replay_validation.json",
            "buy_v3_candidate_validation": research_dir / "buy_v3_candidate_validation.json",
            "buy_v3_tradeability_production_validation": research_dir
            / "buy_v3_tradeability_production_validation.json",
            "production_edge_enhancement_audit": research_dir / "production_edge_enhancement_audit.json",
            "buy_v3_signal_quality_audit": research_dir / "buy_v3_signal_quality_audit.json",
        }
        exported = research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.SOURCE_EXPORTS = original_exports

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Walk Forward Failure Root Cause Audit"
    assert payload["methodology"]["synthesis_only"] is True
    assert "walk_forward_comparison" in payload
    assert "signal_timing_analysis" in payload
    assert "root_cause_probability" in payload
    assert "output_metrics" in payload
    assert payload["final_answer"]["can_buy_v3_plus_sell_v6_proceed_to_paper_trading"] in {"YES", "NO", "PARTIAL"}
    assert payload["output_metrics"]["confidence_score"] > 0
    assert len(payload["conclusions"]) >= 5


@pytest.mark.skipif(
    not Path("outputs/research/unified_production_replay_validation.json").exists(),
    reason="Requires completed replay exports",
)
def test_generate_report_from_exports() -> None:
    report_path = Path("outputs/research/walk_forward_failure_root_cause_audit.json")
    research = WalkForwardFailureRootCauseAuditResearch(report_path=report_path)
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["engines"] == ["BUY_V3", "SELL_V6", "COMBINED"]
    assert payload["walk_forward_comparison"]["per_engine"]["sell_v6"]["degraded"] is True
    assert payload["engine_degradation"]["primary_degraded_engine"] == "SELL_V6"
    assert payload["final_answer"]["can_buy_v3_plus_sell_v6_proceed_to_paper_trading"] == "PARTIAL"
    assert payload["root_cause_probability"]["top_root_cause"] in {
        "regime_change",
        "sample_size_variance",
        "volatility_shift",
        "overfitting",
        "timing_shift",
        "liquidity_shift",
    }
    assert payload["output_metrics"]["production_readiness_score"] >= 30
    assert payload["output_metrics"]["production_risk_score"] >= 30
