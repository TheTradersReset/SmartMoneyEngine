"""Smoke tests for BUY entry timing validation research."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.buy_entry_timing_validation_research import (
    BAR_OFFSETS,
    CURRENT_ENTRY_BARS_BEFORE_MOVE,
    FORMULA_TEXT,
    MODEL_ID,
    BuyEntryTimingValidationResearch,
)


def test_formula_and_offsets() -> None:
    assert "Liquidity Grab" in FORMULA_TEXT
    assert "Failed Breakdown" in FORMULA_TEXT
    assert "Near Support" in FORMULA_TEXT
    assert BAR_OFFSETS == (30, 20, 10, 5, 0)
    assert CURRENT_ENTRY_BARS_BEFORE_MOVE == 3


def test_model_id() -> None:
    assert MODEL_ID == "LDM-BUY-V1"


def test_generate_report(tmp_path: Path) -> None:
    report_path = tmp_path / "buy_entry_timing_validation.json"
    research = BuyEntryTimingValidationResearch(report_path=report_path)
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["model_id"] == "LDM-BUY-V1"
    assert payload["report_type"] == "BUY Entry Timing Validation"
    assert len(payload["occurrences_with_timelines"]) >= 1
    assert payload["final_verdict"]["verdict"] in {"YES", "NO", "PARTIAL"}
    assert "methodology" in payload
    assert "source_exports" in payload
    assert "earliest_causal_entry_analysis" in payload
    assert "entry_comparison" in payload
    assert "limitations" in payload

    first = payload["occurrences_with_timelines"][0]
    assert len(first["timeline"]) == len(BAR_OFFSETS)
    for step in first["timeline"]:
        assert "liquidity_grab" in step
        assert "failed_breakdown" in step
        assert "near_support" in step
        assert "formula_complete" in step
        assert "htf_context" in step
        assert "bos_present" in step

    comparison = payload["entry_comparison"]
    assert "current_entry" in comparison
    assert "earliest_causal_entry" in comparison
    assert comparison["current_entry"]["bars_before_move"] == 3
    assert comparison["current_entry"]["win_rate_pct"] >= 0.0

    analysis = payload["earliest_causal_entry_analysis"]
    assert "aggregate_earliest_entry" in analysis
    assert "offset_pass_summary" in analysis
