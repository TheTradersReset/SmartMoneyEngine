"""Smoke tests for BUY_V1 production validation research."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.buy_v1_production_validation_research import (
    FORMULA_TEXT,
    MODEL_ID,
    BuyV1ProductionValidationResearch,
)


def test_formula_text() -> None:
    assert "Liquidity Grab" in FORMULA_TEXT
    assert "Failed Breakdown" in FORMULA_TEXT
    assert "Near Support" in FORMULA_TEXT


def test_model_id() -> None:
    assert MODEL_ID == "LDM-BUY-V1"


def test_generate_report(tmp_path: Path) -> None:
    report_path = tmp_path / "buy_v1_production_validation.json"
    research = BuyV1ProductionValidationResearch(report_path=report_path)
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["model_id"] == "LDM-BUY-V1"
    assert len(payload["all_occurrences"]) >= 1
    assert payload["coexistence_verdict"]["verdict"] in {"YES", "NO", "PARTIAL"}
    assert "causal_validation_summary" in payload
    assert "classification_summary" in payload
    assert "performance_metrics" in payload
    assert "sell_v5_comparison" in payload
    assert "production_formula_or_failure_reasons" in payload

    first = payload["all_occurrences"][0]
    assert first["symbol"] == "NIFTY50"
    assert first["timeframe"] == "5M"
    assert first["classification"] in {"Real Reversal", "Dead Cat Bounce", "Range Failure"}
    assert first["causal_validation"]["near_support_at_signal_bar"] is True
    assert first["causal_validation"]["liquidity_grab_in_pre_move_events"] is True
    assert first["causal_validation"]["failed_breakdown_in_pre_move_events"] is True

    metrics = payload["performance_metrics"]
    assert metrics["true_causal_win_rate_pct"] >= 0.0
    assert metrics["signals_per_month"] >= 0.0
    assert metrics["capture_200_plus_pct"] >= 0.0

    v5 = payload["sell_v5_comparison"]["sell_v5"]
    assert v5["model_id"] == "LDM-SELL-V5"
    assert v5["signals_emitted"] is not None
