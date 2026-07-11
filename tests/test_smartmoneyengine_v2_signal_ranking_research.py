"""Tests for SmartMoneyEngine V2 signal ranking research."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research.smartmoneyengine_v2_signal_ranking_research import (
    ARCHETYPE_COMBO_SIZES,
    GROUPING_DIMENSIONS,
    MIN_SAMPLE_SIZE,
    RankedV2Signal,
    SmartMoneyEngineV2SignalRankingResearch,
    V2SignalRankingError,
    generate_v2_signal_ranking_report,
)


def _sample_signal(**overrides: object) -> RankedV2Signal:
    base = {
        "symbol": "NIFTY50",
        "bos_timestamp": "2025-08-01 10:00:00",
        "timeframe": "5M",
        "signal_side": "SELL",
        "direction": "bearish",
        "session": "Morning",
        "vwap_state": "Below VWAP",
        "rsi_bucket": "50-60",
        "ema_structure": "EMA20 < EMA50 < EMA200",
        "choch_bos_timing": "Fast (<30 min)",
        "displacement_strength": "Strong",
        "level_context": "Near Resistance",
        "liquidity_context": "Close (20-50 pts)",
        "confirmation_candle": "Weak",
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


def test_quality_score_rejects_small_sample() -> None:
    engine = SmartMoneyEngineV2SignalRankingResearch()
    score = engine._quality_score(
        {
            "sample_size": 10,
            "win_rate_pct": 80.0,
            "profit_factor": 3.0,
            "expectancy": 200.0,
            "hit_1r_rate_pct": 90.0,
            "hit_2r_rate_pct": 70.0,
            "hit_3r_rate_pct": 50.0,
        },
    )
    assert score == 0.0


def test_quality_score_and_tier() -> None:
    engine = SmartMoneyEngineV2SignalRankingResearch()
    metrics = {
        "sample_size": 50,
        "win_rate_pct": 62.0,
        "profit_factor": 2.2,
        "expectancy": 120.0,
        "hit_1r_rate_pct": 70.0,
        "hit_2r_rate_pct": 50.0,
        "hit_3r_rate_pct": 30.0,
    }
    score = engine._quality_score(metrics)
    assert score >= 60.0
    tier, rejected, reason = engine._tier_for_score(score, 50)
    assert rejected is False
    assert tier in {"A", "B", "C"}


def test_aggregate_signals() -> None:
    engine = SmartMoneyEngineV2SignalRankingResearch()
    signals = [
        _sample_signal(realized_pnl_points=100.0, win=True),
        _sample_signal(realized_pnl_points=-50.0, win=False),
    ]
    metrics = engine._aggregate_signals(
        signals,
        archetype_key="test",
        signal_side="SELL",
        grouping_dimension="direction",
        grouping_value="SELL",
        research_days=365,
    )
    assert metrics.sample_size == 2
    assert metrics.win_rate_pct == 50.0
    assert metrics.average_drawdown_points == 20.0


def test_grouped_analysis() -> None:
    engine = SmartMoneyEngineV2SignalRankingResearch()
    signals = [_sample_signal(session="Morning"), _sample_signal(session="Afternoon")]
    grouped = engine._grouped_analysis(signals, 365)
    assert "session" in grouped
    assert len(grouped["session"]) == 2


def test_constants() -> None:
    assert MIN_SAMPLE_SIZE == 30
    assert len(GROUPING_DIMENSIONS) == 12
    assert ARCHETYPE_COMBO_SIZES == (4, 5, 6)


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
    destination = tmp_path / "smartmoneyengine_v2_signal_ranking.json"

    class _FakeReport:
        total_v2_signals = 100
        top_50_signal_archetypes = [{"signal_quality_score": 85.0}]
        tier_a_archetypes = [{"tier": "A"}]
        tier_b_archetypes = []
        top_10_buy_models = []
        top_10_sell_models = [{"signal_quality_score": 88.0}]

        def as_dict(self) -> dict:
            return {"total_v2_signals": self.total_v2_signals}

    def _fake_run(
        self: SmartMoneyEngineV2SignalRankingResearch,
        metadata: dict,
    ) -> _FakeReport:
        del metadata
        return _FakeReport()

    monkeypatch.setattr(SmartMoneyEngineV2SignalRankingResearch, "run", _fake_run)

    report = generate_v2_signal_ranking_report(
        report_path=destination,
        filter_report_path=filter_report,
        v2_optimization_path=v2_export,
    )
    assert report.total_v2_signals == 100
    assert destination.exists()


def test_generate_report_missing_v2_export() -> None:
    filter_report = Path("nonexistent_filter.json")
    with pytest.raises(V2SignalRankingError):
        generate_v2_signal_ranking_report(
            filter_report_path=filter_report,
            v2_optimization_path=Path("missing_v2.json"),
        )
