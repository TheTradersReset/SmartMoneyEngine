"""Tests for SmartMoneyEngine V4 Candidate validation research."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.research.smartmoneyengine_v4_candidate_validation_research import (
    V4CandidateEngine,
    V4_EMA22_RULE,
    V4_EMA_BEAR_CONTEXT,
    V4_EMA_BULL_CONTEXT,
    SmartMoneyEngineV4CandidateValidationError,
    SmartMoneyEngineV4CandidateValidationReport,
    SmartMoneyEngineV4CandidateValidationResearch,
    _attach_ema22,
    _v4_ema_evaluation,
    generate_smartmoneyengine_v4_candidate_validation_report,
)


def test_v4_ema_bearish_rule() -> None:
    enriched = _attach_ema22(
        pd.DataFrame(
            {
                "Close": [100.0, 99.0, 98.0, 97.0, 96.0],
                "_ema_200": [105.0, 104.0, 103.0, 102.0, 101.0],
            }
        )
    )
    result = _v4_ema_evaluation(enriched, 4, close=96.0)
    assert result["rule"] == V4_EMA22_RULE
    assert result["v4_ema_structure"] == V4_EMA_BEAR_CONTEXT
    assert result["v4_ema_bearish"] is True


def test_v4_ema_bull_context() -> None:
    enriched = _attach_ema22(
        pd.DataFrame(
            {
                "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "_ema_200": [95.0, 96.0, 97.0, 98.0, 99.0],
            }
        )
    )
    result = _v4_ema_evaluation(enriched, 4, close=104.0)
    assert result["v4_ema_structure"] == V4_EMA_BULL_CONTEXT
    assert result["v4_ema_bearish"] is False


def test_v4_layer3_confirmation_optional() -> None:
    engine = V4CandidateEngine()
    layer3 = engine._layer3_confirmation(
        {"confirmation_candle": "None", "volume": "Normal"},
    )
    assert layer3["confirmed"] is True
    assert layer3["confirmation_optional"] is True


def test_v4_layer5_no_confirmation_failed_gate() -> None:
    engine = V4CandidateEngine()
    layer1 = {
        "active": True,
        "events_detected": ["Failed Breakout"],
        "failed_breakout_present": True,
    }
    layer2 = {
        "aligned": True,
        "htf_trend": "Bearish",
        "vwap_state": "Below",
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
    assert "CONFIRMATION_FAILED" not in result["reason_codes"]


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text('{"end_date": "2026-07-03"}', encoding="utf-8")
    destination = tmp_path / "smartmoneyengine_v4_candidate_validation.json"

    def _fake_run(self, metadata: dict) -> SmartMoneyEngineV4CandidateValidationReport:
        del metadata
        return SmartMoneyEngineV4CandidateValidationReport(
            report_type="SmartMoneyEngine V4 Candidate Validation",
            engine_versions_compared=["SmartMoneyEngine V3", "SmartMoneyEngine V4 Candidate"],
            symbol="NIFTY50",
            timeframe="5M",
            trading_days_replayed=120,
            replay_start_date="2026-01-01",
            replay_end_date="2026-07-03",
            v4_change_summary={},
            methodology={},
            replay_rules={},
            comparison={
                "v3": {"overall_statistics": {"signals_emitted": 290}},
                "v4_candidate": {"overall_statistics": {"signals_emitted": 320}},
            },
            missed_move_recovery={},
            entry_timing_delta={},
            major_move_capture={},
            final_questions={"4_is_v4_superior_to_v3": {"answer": "PARTIAL"}},
            conclusions=[],
            execution_time_seconds=1.0,
        )

    monkeypatch.setattr(SmartMoneyEngineV4CandidateValidationResearch, "run", _fake_run)

    report = generate_smartmoneyengine_v4_candidate_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
    )
    assert report.comparison["v4_candidate"]["overall_statistics"]["signals_emitted"] == 320
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(SmartMoneyEngineV4CandidateValidationError):
        generate_smartmoneyengine_v4_candidate_validation_report(
            filter_report_path=Path("missing.json"),
        )
