"""Tests for SmartMoneyEngine V3 implementation validation research."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.research.smartmoneyengine_v3_implementation_validation_research import (
    LAYER1_EVENTS,
    SmartMoneyEngineV3Engine,
    SmartMoneyEngineV3ImplementationValidationError,
    SmartMoneyEngineV3ImplementationValidationResearch,
    generate_smartmoneyengine_v3_implementation_validation_report,
)


def test_layer1_events_frozen() -> None:
    assert "Failed Breakout" in LAYER1_EVENTS
    assert "Liquidity Grab" in LAYER1_EVENTS
    assert len(LAYER1_EVENTS) == 5


def test_layer5_rejects_without_failed_breakout() -> None:
    engine = SmartMoneyEngineV3Engine()
    layer1 = {"active": True, "events_detected": ["Gap Reversal"], "failed_breakout_present": False}
    layer2 = {"aligned": True, "htf_trend": "Bearish", "vwap_state": "Below", "ema_structure": "Bear Stack"}
    layer3 = {"confirmed": True}
    context = {"location": "Near Support"}
    result = engine._layer5_no_trade_filters(
        layer1=layer1,
        layer2=layer2,
        layer3=layer3,
        context=context,
        bar=10,
        emitted_bars=set(),
    )
    assert result["pass"] is False
    assert "NO_FAILED_BREAKOUT" in result["reason_codes"]


def test_last_trading_days_window() -> None:
    frame = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=40, freq="B")})
    days = SmartMoneyEngineV3ImplementationValidationResearch._last_n_trading_day_set(frame, 10)
    assert len(days) == 10


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text('{"end_date": "2026-07-03"}', encoding="utf-8")
    destination = tmp_path / "smartmoneyengine_v3_implementation_validation.json"

    class _FakeReport:
        overall_statistics = {"signals_emitted": 3, "win_rate_pct": 66.0, "profit_factor": 2.0}
        conclusions = ["ok"]

        def as_dict(self) -> dict:
            return {"overall_statistics": self.overall_statistics}

    def _fake_run(self, metadata: dict) -> _FakeReport:
        del metadata
        return _FakeReport()

    monkeypatch.setattr(SmartMoneyEngineV3ImplementationValidationResearch, "run", _fake_run)

    report = generate_smartmoneyengine_v3_implementation_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
    )
    assert report.overall_statistics["signals_emitted"] == 3
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(SmartMoneyEngineV3ImplementationValidationError):
        generate_smartmoneyengine_v3_implementation_validation_report(
            filter_report_path=Path("missing.json"),
        )
