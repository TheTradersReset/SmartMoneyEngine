"""Tests for BUY side frequency expansion analysis research."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.buy_side_frequency_expansion_analysis_research import (
    CONDITION_LABELS,
    FORMULA_TEXT,
    FREQUENCY_TARGETS,
    MODEL_ID,
    PRODUCTION_GATES,
    BuySideFrequencyExpansionAnalysisResearch,
)


def test_constants() -> None:
    assert "Liquidity Grab" in FORMULA_TEXT
    assert "Failed Breakdown" in FORMULA_TEXT
    assert "Near Support" in FORMULA_TEXT
    assert MODEL_ID == "LDM-BUY-V1"
    assert FREQUENCY_TARGETS == (20, 30, 40)
    assert PRODUCTION_GATES["win_rate_min_pct"] == 65.0
    assert len(CONDITION_LABELS) == 11


def test_generate_report(tmp_path: Path) -> None:
    report_path = tmp_path / "buy_side_frequency_expansion_analysis.json"
    research = BuySideFrequencyExpansionAnalysisResearch(report_path=report_path)
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["report_type"] == "BUY Side Frequency Expansion Analysis"
    assert payload["model_id"] == "LDM-BUY-V1"
    assert payload["methodology"]["research_only"] is True
    assert "bullish_move_classification" in payload
    assert payload["bullish_move_classification"]["total_bullish_moves_analyzed"] >= 1
    assert "condition_attribution" in payload
    assert "combination_rankings" in payload
    assert payload["combination_rankings"]["total_combinations_evaluated"] >= 1
    assert "frequency_expansion_candidates" in payload
    assert "mandatory_vs_false_conditions" in payload
    assert "final_answer" in payload
    assert payload["final_answer"]["overall_verdict"] in {"YES", "NO", "PARTIAL"}
    assert payload["final_answer"]["can_buy_reach_20_plus_per_month"] in {"YES", "NO", "PARTIAL"}
    assert "most_valuable_setup" in payload
    assert "production_recommendation" in payload
    assert payload["production_recommendation"]["recommendation"] in {
        "BUY_V1",
        "Expanded setup",
        "Hybrid",
    }
    assert "conclusions" in payload
    assert len(payload["conclusions"]) >= 3

    mandatory = payload["mandatory_vs_false_conditions"]
    assert "A_mandatory_conditions" in mandatory
    assert "B_frequency_increasing_conditions" in mandatory
    assert "C_false_reversal_conditions" in mandatory
    assert "D_real_reversal_vs_dead_cat_separators" in mandatory

    real_attr = payload["condition_attribution"]
    assert real_attr["real_reversal_count"] >= 0
    assert len(real_attr["condition_presence_rates_pct"]) == len(CONDITION_LABELS)
