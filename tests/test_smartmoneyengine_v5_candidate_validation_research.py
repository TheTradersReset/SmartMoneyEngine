"""Tests for SmartMoneyEngine V5 Candidate validation research."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research.smartmoneyengine_v4_candidate_validation_research import V4_EMA_BEAR_CONTEXT
from src.research.smartmoneyengine_v5_candidate_validation_research import (
    V5CandidateEngine,
    V5_ALLOWED_VWAP_STATES,
    V5_VWAP_GATE_RULE,
    SmartMoneyEngineV5CandidateValidationError,
    SmartMoneyEngineV5CandidateValidationReport,
    SmartMoneyEngineV5CandidateValidationResearch,
    _v5_vwap_gate_passes,
    generate_smartmoneyengine_v5_candidate_validation_report,
)


def test_v5_vwap_gate_passes_below_and_rejected() -> None:
    assert _v5_vwap_gate_passes("Below") is True
    assert _v5_vwap_gate_passes("Rejected") is True
    assert _v5_vwap_gate_passes("Above") is False
    assert _v5_vwap_gate_passes("Reclaimed") is False
    assert _v5_vwap_gate_passes(None) is False


def test_v5_allowed_vwap_states() -> None:
    assert V5_ALLOWED_VWAP_STATES == frozenset({"Below", "Rejected"})


def test_v5_layer2_accepts_rejected_vwap() -> None:
    engine = V5CandidateEngine()
    layer2 = engine._layer2_directional_filter(
        {
            "htf_trend": "Bearish",
            "vwap": "Rejected",
            "v4_ema_bearish": "True",
            "v4_ema_structure": V4_EMA_BEAR_CONTEXT,
        },
    )
    assert layer2["vwap_gate_passes"] is True
    assert layer2["vwap_gate_rule"] == V5_VWAP_GATE_RULE
    assert layer2["aligned"] is True
    assert layer2["direction"] == "SELL"


def test_v5_layer2_rejects_above_vwap() -> None:
    engine = V5CandidateEngine()
    layer2 = engine._layer2_directional_filter(
        {
            "htf_trend": "Bearish",
            "vwap": "Above",
            "v4_ema_bearish": "True",
            "v4_ema_structure": V4_EMA_BEAR_CONTEXT,
        },
    )
    assert layer2["vwap_gate_passes"] is False
    assert layer2["aligned"] is False


def test_v5_layer5_passes_with_rejected_vwap() -> None:
    engine = V5CandidateEngine()
    layer1 = {
        "active": True,
        "events_detected": ["Failed Breakout"],
        "failed_breakout_present": True,
    }
    layer2 = {
        "aligned": True,
        "htf_trend": "Bearish",
        "vwap_gate_passes": True,
        "vwap_state": "Rejected",
        "ema_structure": V4_EMA_BEAR_CONTEXT,
    }
    layer3 = {"confirmed": True}
    context = {"location": "Near Resistance"}
    result = engine._layer5_no_trade_filters(
        layer1=layer1,
        layer2=layer2,
        layer3=layer3,
        context=context,
        bar=10,
        emitted_bars=set(),
    )
    assert result["pass"] is True
    assert "VWAP_MISMATCH" not in result["reason_codes"]


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text('{"end_date": "2026-07-03"}', encoding="utf-8")
    destination = tmp_path / "smartmoneyengine_v5_candidate_validation.json"

    def _fake_run(self, metadata: dict) -> SmartMoneyEngineV5CandidateValidationReport:
        del metadata
        return SmartMoneyEngineV5CandidateValidationReport(
            report_type="SmartMoneyEngine V5 Candidate Validation",
            engine_versions_compared=["V3", "V4 Candidate", "V5 Candidate"],
            symbol="NIFTY50",
            timeframe="5M",
            trading_days_replayed=120,
            replay_start_date="2026-01-01",
            replay_end_date="2026-07-03",
            v5_change_summary={},
            methodology={},
            replay_rules={},
            comparison={
                "v3": {"overall_statistics": {"signals_emitted": 290}},
                "v4_candidate": {"overall_statistics": {"signals_emitted": 320}},
                "v5_candidate": {"overall_statistics": {"signals_emitted": 350}},
            },
            incremental_vs_v4={"additional_moves_200_plus": 5},
            point_capture={},
            missed_move_recovery={},
            final_questions={"4_is_v5_superior_to_v4": {"answer": "PARTIAL"}},
            conclusions=[],
            execution_time_seconds=1.0,
        )

    monkeypatch.setattr(SmartMoneyEngineV5CandidateValidationResearch, "run", _fake_run)

    report = generate_smartmoneyengine_v5_candidate_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
    )
    assert report.comparison["v5_candidate"]["overall_statistics"]["signals_emitted"] == 350
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(SmartMoneyEngineV5CandidateValidationError):
        generate_smartmoneyengine_v5_candidate_validation_report(
            filter_report_path=Path("missing.json"),
        )
