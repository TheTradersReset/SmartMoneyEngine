"""Tests for BUY_V3 signal quality audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.buy_v3_signal_quality_audit_research import (
    ACHIEVEMENT_THRESHOLDS,
    AUDIT_CLASSIFICATIONS,
    BUY_V3_MODEL_ID,
    BuyV3SignalQualityAuditResearch,
    _achievement_counts,
    _audit_classification,
    _build_per_signal_audit,
    _classification_summary,
    _mfe_bucket,
    _signal_performance,
    _timing_label,
)


def _sample_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-05 09:25:00+05:30",
        "move_start_time": "2026-01-05 11:25:00+05:30",
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


def test_constants() -> None:
    assert BUY_V3_MODEL_ID == "LDM-BUY-V3"
    assert "Winner" in AUDIT_CLASSIFICATIONS
    assert 40 in ACHIEVEMENT_THRESHOLDS


def test_audit_classification_mapping() -> None:
    assert _audit_classification("Real Reversal") == "Winner"
    assert _audit_classification("Bull Trap") == "Bull Trap"
    assert _audit_classification("False Reversal") == "Liquidity Failure"


def test_timing_label() -> None:
    assert _timing_label({"bars_before_expansion": 5}) == "Early"
    assert _timing_label({"bars_before_expansion": 0}) == "Same Candle"
    assert _timing_label({"bars_before_expansion": -2}) == "Delayed"
    assert _timing_label({"bars_before_expansion": None}) == "No Linked Move"


def test_mfe_bucket() -> None:
    assert _mfe_bucket(15.0) == "0-20"
    assert _mfe_bucket(35.0) == "20-40"
    assert _mfe_bucket(250.0) == "200+"


def test_build_per_signal_audit() -> None:
    rows = _build_per_signal_audit([_sample_signal(), _sample_signal(classification="Bull Trap", win=False)])
    assert len(rows) == 2
    assert rows[0]["audit_classification"] == "Winner"
    assert rows[1]["audit_classification"] == "Bull Trap"
    assert rows[0]["timing_label"] == "Early"


def test_classification_summary() -> None:
    rows = _build_per_signal_audit(
        [
            _sample_signal(),
            _sample_signal(classification="Bull Trap", win=False),
            _sample_signal(classification="Range Failure", win=False, mfe_points=30.0),
        ],
    )
    summary = _classification_summary(rows)
    assert summary["counts"]["Winner"] == 1
    assert summary["counts"]["Bull Trap"] == 1
    assert summary["total_signals"] == 3


def test_achievement_counts() -> None:
    rows = _build_per_signal_audit(
        [
            _sample_signal(mfe_points=100.0),
            _sample_signal(mfe_points=50.0, classification="Bull Trap", win=False),
        ],
    )
    achievements = _achievement_counts(rows)
    assert achievements["all_signals"]["40_plus"]["count"] == 2
    assert achievements["all_signals"]["100_plus"]["count"] == 1


def test_signal_performance() -> None:
    perf = _signal_performance([_sample_signal(), _sample_signal(classification="Bull Trap", win=False)], window_days=120)
    assert perf["sample_size"] == 2
    assert perf["win_rate_pct"] == 50.0


def test_generate_report(tmp_path: Path) -> None:
    v3_export = {
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "trading_days_replayed": 120,
        "replay_start_date": "2026-01-05",
        "replay_end_date": "2026-07-02",
        "signal_timing": {"buy_v3": {"before_expansion_pct": 90.0}},
        "comparison": {
            "buy_v3": {
                "overall_statistics": {
                    "signals_emitted": 3,
                    "signals_per_month": 22.0,
                    "win_rate_pct": 66.67,
                    "profit_factor": 2.5,
                    "expectancy": 50.0,
                },
                "classification_summary": {"real_reversal_rate_pct": 66.67, "false_reversal_rate_pct": 33.33},
            },
        },
        "ablation_analysis": {
            "contribution_ranking": {"most_quality_contribution": "Liquidity Grab"},
            "variants": {
                "full_buy_v3": {
                    "label": "Full BUY_V3",
                    "removed_condition": None,
                    "overall_statistics": {"signals_per_month": 22.0, "win_rate_pct": 66.67, "profit_factor": 2.5},
                    "false_reversal_rate_pct": 33.33,
                },
            },
        },
        "final_verdict": {"ablation_insights": {"most_quality_contribution": "Liquidity Grab"}},
        "per_signal_details": {
            "buy_v3": [
                _sample_signal(),
                _sample_signal(
                    classification="Bull Trap",
                    win=False,
                    mfe_points=35.0,
                    bars_before_expansion=0,
                    realized_pnl_points=-20.0,
                ),
                _sample_signal(
                    classification="Range Failure",
                    win=False,
                    mfe_points=25.0,
                    bars_before_expansion=-3,
                    realized_pnl_points=-15.0,
                ),
            ],
        },
    }

    research_dir = tmp_path / "outputs" / "research"
    research_dir.mkdir(parents=True)
    (research_dir / "buy_v3_candidate_validation.json").write_text(json.dumps(v3_export), encoding="utf-8")

    report_path = research_dir / "buy_v3_signal_quality_audit.json"
    research = BuyV3SignalQualityAuditResearch(report_path=report_path)

    import src.research.buy_v3_signal_quality_audit_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_exports = module.SOURCE_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = research_dir
        module.SOURCE_EXPORTS = {
            "buy_v3_candidate_validation": research_dir / "buy_v3_candidate_validation.json",
            "buy_v3_tradeability_production_validation": research_dir
            / "buy_v3_tradeability_production_validation.json",
            "buy_winner_vs_false_reversal_analysis": research_dir
            / "buy_winner_vs_false_reversal_analysis.json",
            "buy_v2_candidate_validation": research_dir / "buy_v2_candidate_validation.json",
        }
        exported = research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.SOURCE_EXPORTS = original_exports

    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["model_id"] == "LDM-BUY-V3"
    assert len(payload["per_signal_audit"]) == 3
    assert payload["audit_classification_summary"]["counts"]["Winner"] == 1
    assert "why_buy_v3_fails" in payload
    assert "move_distribution" in payload
    assert "achievement_counts" in payload
    assert "signal_timing" in payload
    assert "winners_vs_failures_condition_comparison" in payload
    assert "single_filter_simulation" in payload
    assert payload["final_answer"]["near_optimal_without_sacrificing_frequency"] in {"YES", "NO", "PARTIAL"}


def test_generate_report_missing_v3_export(tmp_path: Path) -> None:
    research_dir = tmp_path / "outputs" / "research"
    research_dir.mkdir(parents=True)
    report_path = research_dir / "buy_v3_signal_quality_audit.json"
    research = BuyV3SignalQualityAuditResearch(report_path=report_path)

    import src.research.buy_v3_signal_quality_audit_research as module

    original_root = module.PROJECT_ROOT
    original_dir = module.RESEARCH_DIR
    original_exports = module.SOURCE_EXPORTS.copy()
    try:
        module.PROJECT_ROOT = tmp_path
        module.RESEARCH_DIR = research_dir
        module.SOURCE_EXPORTS = {
            "buy_v3_candidate_validation": research_dir / "missing.json",
            "buy_v3_tradeability_production_validation": research_dir
            / "buy_v3_tradeability_production_validation.json",
            "buy_winner_vs_false_reversal_analysis": research_dir
            / "buy_winner_vs_false_reversal_analysis.json",
            "buy_v2_candidate_validation": research_dir / "buy_v2_candidate_validation.json",
        }
        with pytest.raises(Exception):
            research.export()
    finally:
        module.PROJECT_ROOT = original_root
        module.RESEARCH_DIR = original_dir
        module.SOURCE_EXPORTS = original_exports


def test_generate_report_from_real_exports() -> None:
    report_path = Path("outputs/research/buy_v3_signal_quality_audit.json")
    research = BuyV3SignalQualityAuditResearch(report_path=report_path)
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["report_type"] == "BUY_V3 Signal Quality Audit"
    assert payload["methodology"]["synthesis_only"] is True
    assert payload["audit_classification_summary"]["total_signals"] == 116
    assert payload["audit_classification_summary"]["counts"]["Winner"] == 65
    assert len(payload["per_signal_audit"]) == 116
    assert payload["achievement_counts"]["all_signals"]["40_plus"]["count"] >= 1
    assert payload["final_answer"]["near_optimal_without_sacrificing_frequency"] in {"YES", "NO", "PARTIAL"}
