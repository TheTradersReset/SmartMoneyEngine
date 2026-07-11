"""Tests for SmartMoneyEngine realtime replay validation research."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research.smartmoneyengine_realtime_replay_validation_research import (
    MOMENTUM_THRESHOLDS,
    MISSED_MOVE_THRESHOLDS,
    SmartMoneyEngineRealtimeReplayValidationResearch,
    RealtimeReplayValidationError,
    generate_realtime_replay_validation_report,
)


def test_signal_score_buy() -> None:
    engine = SmartMoneyEngineRealtimeReplayValidationResearch()
    score = engine._signal_score(
        "BUY",
        {
            "bos": True,
            "choch": True,
            "liquidity_grab": True,
            "fvg": True,
            "displacement": "Strong",
            "htf_trend": "Bullish",
            "vwap": "Below VWAP",
        },
        {"strong_confirmation": True},
    )
    assert score > 50.0


def test_signal_score_no_trade() -> None:
    engine = SmartMoneyEngineRealtimeReplayValidationResearch()
    assert engine._signal_score("NO_TRADE", {}, {}) == 0.0


def test_window_count() -> None:
    import numpy as np

    engine = SmartMoneyEngineRealtimeReplayValidationResearch()
    cumsum = np.array([0, 1, 1, 2, 2, 3])
    assert engine._window_count(cumsum, 5, 2) == 2


def test_keys_from_labels() -> None:
    keys = SmartMoneyEngineRealtimeReplayValidationResearch._keys_from_labels(
        ["Below VWAP", "EMA20 < EMA50 < EMA200"],
    )
    assert "below_vwap" in keys
    assert "ema_bear_stack" in keys


def test_constants() -> None:
    assert 200 in MOMENTUM_THRESHOLDS
    assert 200 in MISSED_MOVE_THRESHOLDS


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2025-07-03", "end_date": "2026-07-03", "research_window_days": 365}',
        encoding="utf-8",
    )
    v2_export = tmp_path / "smartmoneyengine_v2_frequency_optimization.json"
    v2_export.write_text(
        """{
          "smartmoneyengine_v2_production_card": {
            "buy_rules": {"filter_stack": ["EMA20 < EMA50 < EMA200"]},
            "sell_rules": {"filter_stack": ["Below VWAP"]},
            "no_trade_rules": []
          }
        }""",
        encoding="utf-8",
    )
    destination = tmp_path / "smartmoneyengine_realtime_replay_validation.json"

    class _FakeReport:
        overall_statistics = {"total_signals": 42, "signals_per_month": 5.0, "win_rate_pct": 55.0}
        major_200_plus_move_analysis = {"capture_rate_pct": 12.5}

        def as_dict(self) -> dict:
            return {"overall_statistics": self.overall_statistics}

    def _fake_run(
        self: SmartMoneyEngineRealtimeReplayValidationResearch,
        metadata: dict,
    ) -> _FakeReport:
        del metadata
        return _FakeReport()

    monkeypatch.setattr(SmartMoneyEngineRealtimeReplayValidationResearch, "run", _fake_run)

    report = generate_realtime_replay_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
        v2_optimization_path=v2_export,
    )
    assert report.overall_statistics["total_signals"] == 42
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(RealtimeReplayValidationError):
        generate_realtime_replay_validation_report(filter_report_path=Path("missing.json"))
