"""Tests for BUY formula reality verification."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.buy_formula_reality_verification_research import (
    FORMULA_TEXT,
    BuyFormulaRealityVerificationResearch,
)


def test_formula_text() -> None:
    assert "Failed Breakdown" in FORMULA_TEXT
    assert "Gap Reversal" in FORMULA_TEXT
    assert "Near Support" in FORMULA_TEXT


def test_generate_report(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    discovery_path = root / "outputs/research/nifty50_buy_side_reality_discovery.json"
    report_path = tmp_path / "buy_formula_reality_verification.json"
    research = BuyFormulaRealityVerificationResearch(
        discovery_path=discovery_path,
        report_path=report_path,
    )
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["actual_occurrences"] >= 1
    assert payload["final_decision"]["can_buy_formula_survive_reality"] in {"YES", "NO"}
    assert "all_occurrences" in payload
    assert payload["all_occurrences"][0]["causal_validation"]["near_support_at_signal_bar"] is True
