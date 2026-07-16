"""Tests for BUY winner vs false reversal analysis research."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.buy_winner_vs_false_reversal_analysis_research import (
    ANALYSIS_CONDITIONS,
    BUY_V2_COMPONENTS,
    BUY_V2_FORMULA_TEXT,
    FORMULA_TEXT,
    MODEL_ID,
    PRODUCTION_GATES,
    BuyWinnerVsFalseReversalAnalysisResearch,
    _build_false_reversal_cohort,
    _build_winner_cohort,
    _condition_metrics,
    _extract_conditions_from_signal,
    _information_gain,
)


def _sample_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-05 09:25:00+05:30",
        "bar": 100,
        "win": True,
        "classification": "Real Reversal",
        "realized_pnl_points": 120.0,
        "mfe_points": 180.0,
        "signal_reason_stack": {
            "layer1": ["Liquidity Grab", "Failed Breakdown"],
            "layer2": {
                "htf_trend": "Bullish",
                "vwap": "Reclaimed",
                "location": "Near Support",
            },
        },
        "layers": {
            "layer1": {
                "events_detected": ["Liquidity Grab", "Failed Breakdown", "Gap Reversal"],
                "events_at_bar": ["Failed Breakdown"],
                "formula_events_matched": ["Liquidity Grab", "Failed Breakdown"],
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
    assert "Liquidity Grab" in FORMULA_TEXT
    assert BUY_V2_FORMULA_TEXT == "Failed Breakdown + Gap Reversal"
    assert MODEL_ID == "LDM-BUY-V1"
    assert len(ANALYSIS_CONDITIONS) == 10
    assert PRODUCTION_GATES["win_rate_min_pct"] == 65.0
    assert BUY_V2_COMPONENTS == ("Failed Breakdown", "Gap Reversal")


def test_extract_conditions_from_signal() -> None:
    conditions = _extract_conditions_from_signal(_sample_signal())
    assert conditions["Liquidity Grab"] is True
    assert conditions["Failed Breakdown"] is True
    assert conditions["Near Support"] is True
    assert conditions["Gap Reversal"] is True
    assert conditions["HTF Bullish"] is True
    assert conditions["VWAP Reclaim"] is True


def test_information_gain_perfect_separator() -> None:
    ig = _information_gain(
        winner_present=10,
        winner_absent=0,
        false_present=0,
        false_absent=90,
    )
    assert ig > 0.0


def test_condition_metrics() -> None:
    winners = [{"conditions": {"Liquidity Grab": True, "Near Support": True}}]
    false_rows = [{"conditions": {"Liquidity Grab": False, "Near Support": False}}]
    metrics = _condition_metrics("Liquidity Grab", winners, false_rows)
    assert metrics["winner_coverage_pct"] == 100.0
    assert metrics["false_reversal_coverage_pct"] == 0.0
    assert metrics["precision_pct"] == 100.0


def test_generate_report() -> None:
    report_path = Path("outputs/research/buy_winner_vs_false_reversal_analysis.json")
    research = BuyWinnerVsFalseReversalAnalysisResearch(report_path=report_path)
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["report_type"] == "BUY Winner vs False Reversal Analysis"
    assert payload["methodology"]["research_only"] is True
    assert payload["cohort_summary"]["winner_cohort_size"] == 17
    assert payload["cohort_summary"]["false_reversal_cohort_size"] == 947
    assert len(payload["per_condition_metrics"]) == len(ANALYSIS_CONDITIONS)

    first = payload["per_condition_metrics"][0]
    assert "winner_coverage_pct" in first
    assert "false_reversal_coverage_pct" in first
    assert "information_gain" in first
    assert "precision_pct" in first
    assert "recall_pct" in first
    assert "separation_score" in first

    assert "condition_rankings" in payload
    assert len(payload["condition_rankings"]["composite_ranked"]) == len(ANALYSIS_CONDITIONS)
    assert "condition_classification" in payload
    assert "smallest_condition_set" in payload
    assert "buy_v2_filter_simulations" in payload
    assert len(payload["buy_v2_filter_simulations"]) >= 3
    assert "future_leakage_validation" in payload
    assert payload["future_leakage_validation"]["requires_future_confirmation"] == []
    assert "buy_v3_feasibility" in payload
    assert "proposed_buy_v3_stack" in payload["buy_v3_feasibility"]
    assert payload["final_answer"]["overall_verdict"] in {"YES", "NO", "PARTIAL"}
    assert payload["final_recommendation"]["highest_value_condition"] in ANALYSIS_CONDITIONS
    assert len(payload["conclusions"]) >= 4


def test_cohort_builders_match_export_counts() -> None:
    validation_path = Path("outputs/research/buy_v2_candidate_validation.json")
    if not validation_path.exists():
        return
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    winners, _ = _build_winner_cohort(validation)
    false_rows = _build_false_reversal_cohort(validation)
    assert len(winners) == validation["missed_reversal_recovery"]["recovered_by_buy_v2"]
    assert len(false_rows) == validation["missed_reversal_recovery"]["new_false_reversals_buy_v2"]
