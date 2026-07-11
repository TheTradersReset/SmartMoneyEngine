"""Tests for SmartMoneyEngine archetype walk-forward validation research."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.research.smartmoneyengine_archetype_walkforward_research import (
    MIN_TEST_SAMPLES,
    SURVIVES_MIN_EXPECTANCY,
    SURVIVES_MIN_PF,
    SURVIVES_MIN_WIN_RATE,
    ArchetypeWalkForwardError,
    PeriodMetrics,
    SmartMoneyEngineArchetypeWalkForwardResearch,
    generate_archetype_walkforward_report,
)
from src.research.smartmoneyengine_v2_signal_ranking_research import RankedV2Signal


def _sample_signal(**overrides: object) -> RankedV2Signal:
    base = {
        "symbol": "NIFTY50",
        "bos_timestamp": "2025-08-01 10:00:00",
        "timeframe": "5M",
        "signal_side": "SELL",
        "direction": "bearish",
        "session": "Closing",
        "vwap_state": "Below VWAP",
        "rsi_bucket": "Below 40",
        "ema_structure": "EMA20 < EMA50 < EMA200",
        "choch_bos_timing": "Fast (<30 min)",
        "displacement_strength": "Strong",
        "level_context": "Near Resistance",
        "liquidity_context": "Close (20-50 pts)",
        "confirmation_candle": "Strong Confirmation",
        "risk_points": 50.0,
        "realized_pnl_points": 100.0,
        "realized_rr": 2.0,
        "win": True,
        "hit_1r": True,
        "hit_2r": True,
        "hit_3r": False,
        "mae_points": 20.0,
        "holding_bars": 8,
        "holding_minutes": 40.0,
        "archetype_key": "test",
    }
    base.update(overrides)
    return RankedV2Signal(**base)


def test_parse_archetype_key() -> None:
    key = "timeframe=5M | direction=SELL | session=Closing"
    parsed = SmartMoneyEngineArchetypeWalkForwardResearch._parse_archetype_key(key)
    assert parsed["timeframe"] == "5M"
    assert parsed["direction"] == "SELL"


def test_matches_archetype() -> None:
    criteria = {"timeframe": "5M", "direction": "SELL", "session": "Closing"}
    signal = _sample_signal()
    assert SmartMoneyEngineArchetypeWalkForwardResearch._matches_archetype(signal, criteria)


def test_classify_survives() -> None:
    metrics = PeriodMetrics(
        sample_size=20,
        signals_per_month=2.0,
        win_rate_pct=55.0,
        profit_factor=1.8,
        expectancy=90.0,
        hit_1r_rate_pct=60.0,
        hit_2r_rate_pct=40.0,
        hit_3r_rate_pct=20.0,
        net_points=1800.0,
    )
    assert SmartMoneyEngineArchetypeWalkForwardResearch._classify(metrics) == "SURVIVES"


def test_classify_fails() -> None:
    metrics = PeriodMetrics(
        sample_size=20,
        signals_per_month=2.0,
        win_rate_pct=30.0,
        profit_factor=0.8,
        expectancy=-10.0,
        hit_1r_rate_pct=20.0,
        hit_2r_rate_pct=10.0,
        hit_3r_rate_pct=5.0,
        net_points=-200.0,
    )
    assert SmartMoneyEngineArchetypeWalkForwardResearch._classify(metrics) == "FAILS"


def test_edge_decay() -> None:
    train = PeriodMetrics(10, 1.0, 60.0, 2.0, 100.0, 50.0, 30.0, 10.0, 1000.0)
    test = PeriodMetrics(8, 1.0, 55.0, 1.8, 80.0, 45.0, 25.0, 8.0, 640.0)
    decay = SmartMoneyEngineArchetypeWalkForwardResearch._edge_decay(train, test)
    assert decay == -20.0


def test_robustness_score() -> None:
    train = PeriodMetrics(20, 2.0, 65.0, 2.5, 120.0, 70.0, 50.0, 30.0, 2400.0)
    test = PeriodMetrics(15, 2.0, 58.0, 1.9, 95.0, 65.0, 45.0, 25.0, 1425.0)
    score = SmartMoneyEngineArchetypeWalkForwardResearch._robustness_score(
        train,
        test,
        "DEGRADES",
        -20.0,
    )
    assert 0.0 <= score <= 100.0


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2025-07-03", "end_date": "2026-07-03", "research_window_days": 365}',
        encoding="utf-8",
    )
    ranking = tmp_path / "smartmoneyengine_v2_signal_ranking.json"
    ranking.write_text(
        """{
          "top_50_signal_archetypes": [{
            "archetype_key": "direction=SELL",
            "signal_side": "SELL",
            "signal_quality_score": 80.0,
            "tier": "A"
          }]
        }""",
        encoding="utf-8",
    )
    destination = tmp_path / "smartmoneyengine_archetype_walkforward.json"

    class _FakeReport:
        archetypes_validated = 1
        classification_summary = {"SURVIVES": 1, "DEGRADES": 0, "FAILS": 0}
        production_candidate_list = [{"archetype_key": "direction=SELL"}]
        top_20_robust_sell_archetypes = [{"robustness_score": 85.0}]

        def as_dict(self) -> dict:
            return {"archetypes_validated": self.archetypes_validated}

    def _fake_run(
        self: SmartMoneyEngineArchetypeWalkForwardResearch,
        metadata: dict,
    ) -> _FakeReport:
        del metadata
        return _FakeReport()

    monkeypatch.setattr(SmartMoneyEngineArchetypeWalkForwardResearch, "run", _fake_run)

    report = generate_archetype_walkforward_report(
        report_path=destination,
        filter_report_path=filter_report,
        v2_ranking_path=ranking,
    )
    assert report.archetypes_validated == 1
    assert destination.exists()


def test_constants() -> None:
    assert SURVIVES_MIN_PF == 1.5
    assert SURVIVES_MIN_EXPECTANCY == 75.0
    assert SURVIVES_MIN_WIN_RATE == 50.0
    assert MIN_TEST_SAMPLES == 5
