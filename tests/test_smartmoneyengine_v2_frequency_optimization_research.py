"""Tests for SmartMoneyEngine V2 frequency optimization research."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research.smartmoneyengine_v2_frequency_optimization_research import (
    FREQUENCY_TARGETS,
    MIN_EXPECTANCY,
    MIN_PROFIT_FACTOR,
    MIN_WIN_RATE,
    EnrichedTier2Trade,
    SmartMoneyEngineV2FrequencyOptimizationResearch,
    V2FrequencyOptimizationError,
    generate_v2_frequency_optimization_report,
)
from src.research.smartmoneyengine_production_candidate_research import ProductionCandidateTrade


def _sample_trade(side: str = "BUY", pnl: float = 100.0, **flags: bool) -> ProductionCandidateTrade:
    base_flags = {
        "strong_confirmation": False,
        "ema_bear_stack": False,
        "below_vwap": False,
        "gap_down": False,
    }
    base_flags.update(flags)
    return ProductionCandidateTrade(
        bos_timestamp="2025-08-01 10:00:00",
        timeframe="5M",
        direction="bullish" if side == "BUY" else "bearish",
        signal_side=side,
        risk_points=50.0,
        realized_pnl_points=pnl,
        realized_rr=pnl / 50.0,
        win=pnl > 0,
        hit_1r_before_sl=pnl >= 50,
        hit_2r_before_sl=pnl >= 100,
        hit_3r_before_sl=pnl >= 150,
        feature_flags=base_flags,
        feature_tags=(),
    )


def test_quality_thresholds() -> None:
    engine = SmartMoneyEngineV2FrequencyOptimizationResearch()
    assert engine._meets_thresholds(
        {"profit_factor": 1.6, "expectancy": 80.0, "win_rate_pct": 55.0},
    )
    assert not engine._meets_thresholds(
        {"profit_factor": 1.4, "expectancy": 80.0, "win_rate_pct": 55.0},
    )
    assert not engine._meets_thresholds(
        {"profit_factor": 1.6, "expectancy": 70.0, "win_rate_pct": 55.0},
    )


def test_apply_configuration_and_aggregate() -> None:
    engine = SmartMoneyEngineV2FrequencyOptimizationResearch()
    trades = [
        EnrichedTier2Trade(
            symbol="NIFTY50",
            trade=_sample_trade("BUY", 100.0, strong_confirmation=True, ema_bear_stack=True),
            trait_tags=(),
        ),
        EnrichedTier2Trade(
            symbol="NIFTY50",
            trade=_sample_trade("SELL", 120.0, below_vwap=True, gap_down=True),
            trait_tags=(),
        ),
        EnrichedTier2Trade(
            symbol="NIFTY50",
            trade=_sample_trade("BUY", -50.0, strong_confirmation=True),
            trait_tags=("Market Location: Near Resistance",),
        ),
    ]
    selected = engine._apply_configuration(
        trades,
        ("strong_confirmation", "ema_bear_stack"),
        ("below_vwap", "gap_down"),
        ["Market Location: Near Resistance"],
    )
    assert len(selected) == 2
    metrics = engine._aggregate_trades(selected, 365)
    assert metrics["sample_size"] == 2
    assert metrics["expectancy"] == 110.0


def test_frequency_targets() -> None:
    engine = SmartMoneyEngineV2FrequencyOptimizationResearch()
    targets = engine._frequency_targets(35.0)
    assert targets["20+"] is True
    assert targets["30+"] is True
    assert targets["40+"] is False
    assert FREQUENCY_TARGETS == (20, 30, 40)


def test_subsets() -> None:
    engine = SmartMoneyEngineV2FrequencyOptimizationResearch()
    subsets = engine._subsets(["A", "B"])
    assert len(subsets) == 4
    assert [] in subsets
    assert ["A", "B"] in subsets


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
    walkforward = tmp_path / "smartmoneyengine_walkforward_validation.json"
    walkforward.write_text(
        """{
          "survival_verdict": "DEGRADED",
          "in_sample_metrics": {"overall": {"signals_per_month": 3.2, "expectancy": 137.0}},
          "out_of_sample_metrics": {"overall": {"signals_per_month": 5.3, "expectancy": 94.0}}
        }""",
        encoding="utf-8",
    )
    destination = tmp_path / "smartmoneyengine_v2_frequency_optimization.json"

    class _FakeReport:
        v1_vs_v2_comparison = {
            "v1": {"signals_per_month": 3.2},
            "v2": {"signals_per_month": 25.0},
        }
        filter_combination_analysis = [{"label": "test"}]

        def as_dict(self) -> dict:
            return {"v1_vs_v2_comparison": self.v1_vs_v2_comparison}

    def _fake_run(
        self: SmartMoneyEngineV2FrequencyOptimizationResearch,
        metadata: dict,
    ) -> _FakeReport:
        del metadata
        return _FakeReport()

    monkeypatch.setattr(SmartMoneyEngineV2FrequencyOptimizationResearch, "run", _fake_run)

    report = generate_v2_frequency_optimization_report(
        report_path=destination,
        filter_report_path=filter_report,
        production_card_path=production_card,
        walkforward_path=walkforward,
    )
    assert report.v1_vs_v2_comparison["v2"]["signals_per_month"] == 25.0
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(V2FrequencyOptimizationError):
        generate_v2_frequency_optimization_report(filter_report_path=Path("missing.json"))


def test_constants() -> None:
    assert MIN_PROFIT_FACTOR == 1.5
    assert MIN_EXPECTANCY == 75.0
    assert MIN_WIN_RATE == 50.0
