"""Tests for NIFTY50 BUY-side reality discovery synthesis."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.nifty50_buy_side_reality_discovery_research import (
    CAUSAL_WARNING_EVENTS,
    Nifty50BuySideRealityDiscoveryResearch,
)


def test_causal_event_universe() -> None:
    assert "Gap Reversal" in CAUSAL_WARNING_EVENTS
    assert "BOS" not in CAUSAL_WARNING_EVENTS


def test_generate_report(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report_path = tmp_path / "buy_side_discovery.json"
    research = Nifty50BuySideRealityDiscoveryResearch(report_path=report_path)
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["report_type"] == "NIFTY50 BUY Side Reality Discovery"
    assert "bullish_move_anatomy" in payload
    assert "buy_side_failure_reasons" in payload
    assert payload["buy_side_failure_reasons"]["v3_implementation"]["buy_signals_emitted"] == 0
    assert "50" in payload["move_threshold_analysis"]
