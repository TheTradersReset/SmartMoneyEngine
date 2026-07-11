"""Tests for NIFTY50 filter ablation study synthesis."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.nifty50_filter_ablation_study_research import (
    MODELS,
    Nifty50FilterAblationStudyResearch,
    _observed_trade_stats,
)


def test_models_defined_a_through_e() -> None:
    assert set(MODELS) == {"A", "B", "C", "D", "E"}
    assert MODELS["E"].get("is_current_v3") is True


def test_observed_trade_stats_empty() -> None:
    stats = _observed_trade_stats([])
    assert stats["signal_count"] == 0
    assert stats["win_rate_pct"] == 0.0


def test_generate_report_from_exports(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    v3 = json.loads(
        (root / "outputs/research/smartmoneyengine_v3_implementation_validation.json").read_text(
            encoding="utf-8"
        )
    )
    timing = json.loads(
        (root / "outputs/research/nifty50_signal_timing_audit.json").read_text(encoding="utf-8")
    )
    v3_path = tmp_path / "v3.json"
    timing_path = tmp_path / "timing.json"
    report_path = tmp_path / "ablation.json"
    v3_path.write_text(json.dumps(v3), encoding="utf-8")
    timing_path.write_text(json.dumps(timing), encoding="utf-8")

    research = Nifty50FilterAblationStudyResearch(
        v3_path=v3_path,
        timing_path=timing_path,
        report_path=report_path,
    )
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["models"]["E"]["signal_count"] == 43
    assert payload["models"]["E"]["metrics_source"] == "observed_v3_replay"
    assert payload["models"]["A"]["metrics_source"] == "synthesis_from_rejection_blocks"
    assert payload["models"]["A"]["signal_count"] > payload["models"]["E"]["signal_count"]
    assert "single_filter_removal_analysis" in payload
