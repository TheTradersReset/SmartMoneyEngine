"""Tests for BUY_V1 missed reversal analysis research."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.buy_v1_missed_reversal_analysis_research import (
    ANALYSIS_CONDITIONS,
    BUY_V1_REQUIRED,
    FORMULA_TEXT,
    FREQUENCY_TARGETS,
    MODEL_ID,
    OUTCOME_TIERS,
    PRODUCTION_GATES,
    BuyV1MissedReversalAnalysisResearch,
    _buy_v1_missing_conditions,
)


def test_constants() -> None:
    assert "Liquidity Grab" in FORMULA_TEXT
    assert "Failed Breakdown" in FORMULA_TEXT
    assert "Near Support" in FORMULA_TEXT
    assert MODEL_ID == "LDM-BUY-V1"
    assert len(BUY_V1_REQUIRED) == 3
    assert len(ANALYSIS_CONDITIONS) == 10
    assert OUTCOME_TIERS == (40, 60, 80, 100, 200)
    assert FREQUENCY_TARGETS == (15, 20, 30)
    assert PRODUCTION_GATES["win_rate_min_pct"] == 65.0


def test_buy_v1_missing_conditions() -> None:
    missing = _buy_v1_missing_conditions(
        {
            "Liquidity Grab": False,
            "Failed Breakdown": True,
            "Near Support": False,
        },
    )
    assert missing == ["Liquidity Grab", "Near Support"]
    assert missing[0] == "Liquidity Grab"
    assert missing[-1] == "Near Support"


def test_generate_report(tmp_path: Path) -> None:
    report_path = tmp_path / "buy_v1_missed_reversal_analysis.json"
    research = BuyV1MissedReversalAnalysisResearch(report_path=report_path)
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["report_type"] == "BUY_V1 Missed Reversal Analysis"
    assert payload["model_id"] == "LDM-BUY-V1"
    assert payload["methodology"]["research_only"] is True
    assert payload["missed_reversal_summary"]["missed_real_reversal_count"] >= 1
    assert len(payload["per_missed_reversal"]) == payload["missed_reversal_summary"]["missed_real_reversal_count"]

    first = payload["per_missed_reversal"][0]
    assert "first_missing_buy_v1_condition" in first
    assert "last_missing_buy_v1_condition" in first
    assert "condition_stack_present" in first
    assert "causal_events" in first
    assert "outcome" in first
    assert len(first["conditions"]) == len(ANALYSIS_CONDITIONS)

    assert "buy_v1_blocker_analysis" in payload
    assert payload["buy_v1_blocker_analysis"]["primary_prevention_condition"] is not None
    assert "causal_event_rankings" in payload
    assert len(payload["causal_event_rankings"]["by_frequency"]) >= 1
    assert "recovery_stack_candidates" in payload
    assert len(payload["recovery_stack_candidates"]) >= 5

    best = payload["recovery_stack_candidates"][0]
    assert "recovered_real_reversals_count" in best
    assert "signals_per_month" in best
    assert "win_rate_pct" in best
    assert "passes_production_gates" in best

    assert "essential_optional_bottleneck" in payload
    assert "essential_condition" in payload["essential_optional_bottleneck"]
    assert "optional_condition" in payload["essential_optional_bottleneck"]
    assert "frequency_bottleneck" in payload["essential_optional_bottleneck"]
    assert "best_buy_v2_candidate_stack" in payload["essential_optional_bottleneck"]

    assert "final_answer" in payload
    assert payload["final_answer"]["overall_verdict"] in {"YES", "NO", "PARTIAL"}
    assert payload["final_answer"]["can_recover_47_missed_reversals"] in {"YES", "NO", "PARTIAL"}
    assert "outcome_measurement" in payload
    assert len(payload["outcome_measurement"]["tier_capture_rates_pct"]) == len(OUTCOME_TIERS)
    assert len(payload["conclusions"]) >= 4
