"""Tests for SmartMoneyEngine walk-forward validation research."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.research.smartmoneyengine_walkforward_validation_research import (
    MANDATORY_CORE,
    TRAIN_FRACTION,
    TEST_FRACTION,
    WalkForwardMetrics,
    WalkForwardSignal,
    WalkForwardValidationError,
    SmartMoneyEngineWalkForwardValidationResearch,
    generate_walkforward_validation_report,
)


def test_mandatory_core() -> None:
    assert MANDATORY_CORE == ("Displacement", "CHOCH", "BOS", "FVG Reclaim")


def test_train_test_fractions() -> None:
    assert TRAIN_FRACTION == 0.70
    assert TEST_FRACTION == 0.30


def test_split_dates() -> None:
    engine = SmartMoneyEngineWalkForwardValidationResearch()
    start, train_end, test_start, end = engine._split_dates(
        {"start_date": "2025-01-01", "end_date": "2025-12-31"},
    )
    assert start == date(2025, 1, 1)
    assert end == date(2025, 12, 31)
    assert train_end > start
    assert test_start > train_end
    assert test_start <= end


def test_no_trade_blocked() -> None:
    blocked = SmartMoneyEngineWalkForwardValidationResearch._no_trade_blocked(
        ("Market Location: Near Resistance", "Displacement: Medium"),
        ["Market Location: Near Resistance"],
    )
    assert blocked is True


def test_aggregate_metrics() -> None:
    engine = SmartMoneyEngineWalkForwardValidationResearch()
    signals = [
        WalkForwardSignal(
            symbol="NIFTY50",
            bos_timestamp="2025-06-01 10:00:00",
            timeframe="5M",
            signal_side="BUY",
            direction="bullish",
            risk_points=20.0,
            realized_pnl_points=40.0,
            realized_rr=2.0,
            win=True,
            hit_1r=True,
            hit_2r=True,
            hit_3r=False,
            trait_tags=(),
            blocked_by_no_trade=False,
        ),
        WalkForwardSignal(
            symbol="NIFTY50",
            bos_timestamp="2025-06-02 10:00:00",
            timeframe="5M",
            signal_side="SELL",
            direction="bearish",
            risk_points=20.0,
            realized_pnl_points=-20.0,
            realized_rr=-1.0,
            win=False,
            hit_1r=False,
            hit_2r=False,
            hit_3r=False,
            trait_tags=(),
            blocked_by_no_trade=False,
        ),
    ]
    metrics = engine._aggregate(signals, "test", None, 30)
    assert metrics.sample_size == 2
    assert metrics.win_rate_pct == 50.0
    assert metrics.hit_1r_rate_pct == 50.0
    assert metrics.hit_2r_rate_pct == 50.0


def test_survival_verdict_survives() -> None:
    engine = SmartMoneyEngineWalkForwardValidationResearch()
    in_metrics = WalkForwardMetrics(
        scope="in_sample",
        signal_side=None,
        sample_size=100,
        signals_per_month=10.0,
        win_rate_pct=55.0,
        profit_factor=2.0,
        expectancy=80.0,
        average_rr=1.2,
        maximum_drawdown_points=100.0,
        hit_1r_rate_pct=70.0,
        hit_2r_rate_pct=50.0,
        hit_3r_rate_pct=30.0,
        net_points=8000.0,
    )
    out_metrics = WalkForwardMetrics(
        scope="out_of_sample",
        signal_side=None,
        sample_size=40,
        signals_per_month=8.0,
        win_rate_pct=45.0,
        profit_factor=1.8,
        expectancy=60.0,
        average_rr=1.0,
        maximum_drawdown_points=80.0,
        hit_1r_rate_pct=65.0,
        hit_2r_rate_pct=45.0,
        hit_3r_rate_pct=25.0,
        net_points=2400.0,
    )
    verdict, survives = engine._survival_verdict(in_metrics, out_metrics)
    assert verdict == "SURVIVES"
    assert survives is True


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2025-07-03", "end_date": "2026-07-03", "research_window_days": 365}',
        encoding="utf-8",
    )
    production_card = tmp_path / "smartmoneyengine_final_production_validation.json"
    production_card.write_text(
        """{
          "smartmoneyengine_v1_final_production_card": {
            "buy_rules": {"filter_stack": ["Strong Confirmation Candle", "EMA20 < EMA50 < EMA200"]},
            "sell_rules": {"filter_stack": ["Below VWAP", "Gap Down"]},
            "no_trade_rules": ["Market Location: Near Resistance"]
          }
        }""",
        encoding="utf-8",
    )
    destination = tmp_path / "smartmoneyengine_walkforward_validation.json"

    class _FakeReport:
        survival_verdict = "SURVIVES"
        survives_unseen_market_data = True
        train_start_date = "2025-07-03"
        train_end_date = "2026-04-03"
        test_start_date = "2026-04-04"
        test_end_date = "2026-07-03"
        in_sample_metrics = {"overall": {"sample_size": 50, "expectancy": 70.0}}
        out_of_sample_metrics = {"overall": {"sample_size": 20, "expectancy": 55.0}}

        def as_dict(self) -> dict:
            return {
                "survival_verdict": self.survival_verdict,
                "survives_unseen_market_data": self.survives_unseen_market_data,
            }

    def _fake_run(
        self: SmartMoneyEngineWalkForwardValidationResearch,
        metadata: dict,
    ) -> _FakeReport:
        del metadata
        return _FakeReport()

    monkeypatch.setattr(SmartMoneyEngineWalkForwardValidationResearch, "run", _fake_run)

    report = generate_walkforward_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
        production_card_path=production_card,
    )
    assert report.survival_verdict == "SURVIVES"
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(WalkForwardValidationError):
        generate_walkforward_validation_report(filter_report_path=Path("missing.json"))


def test_generate_report_missing_production_card(tmp_path: Path) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2025-07-03", "end_date": "2026-07-03"}',
        encoding="utf-8",
    )
    with pytest.raises(WalkForwardValidationError):
        generate_walkforward_validation_report(
            filter_report_path=filter_report,
            production_card_path=tmp_path / "missing_card.json",
        )
