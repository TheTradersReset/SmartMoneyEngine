"""Tests for SmartMoneyEngine signal timing reality audit."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.smartmoneyengine_signal_timing_reality_audit_research import (
    SmartMoneyEngineSignalTimingRealityAuditResearch,
)


def test_generate_timing_audit_report(tmp_path: Path) -> None:
    report_path = tmp_path / "timing_reality_audit.json"
    research = SmartMoneyEngineSignalTimingRealityAuditResearch(report_path=report_path)
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["report_type"] == "SmartMoneyEngine Signal Timing Reality Audit"
    assert payload["timing_audit"]["summary"]["signal_count"] == 43
    assert "filter_impact_audit" in payload
    assert "final_recommendation" in payload
