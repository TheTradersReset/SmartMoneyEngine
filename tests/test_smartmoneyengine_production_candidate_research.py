"""Tests for SmartMoneyEngine production candidate research."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research.smartmoneyengine_production_candidate_research import (
    FEATURE_DEFINITIONS,
    MIN_EXPECTANCY,
    MIN_PROFIT_FACTOR,
    MIN_SAMPLES,
    ProductionCandidateTrade,
    SmartMoneyEngineProductionCandidateResearch,
    generate_production_candidate_report,
)


def _trade(
    side: str = "BUY",
    win: bool = True,
    pnl: float = 80.0,
    hit_1r: bool = True,
    hit_2r: bool = True,
    hit_3r: bool = False,
    flags: dict[str, bool] | None = None,
) -> ProductionCandidateTrade:
    feature_flags = flags or {"near_support": True, "liquidity_sweep": True}
    return ProductionCandidateTrade(
        bos_timestamp="2026-01-02 09:15:00+05:30",
        timeframe="5M",
        direction="bullish" if side == "BUY" else "bearish",
        signal_side=side,
        risk_points=20.0,
        realized_pnl_points=pnl,
        realized_rr=pnl / 20.0,
        win=win,
        hit_1r_before_sl=hit_1r,
        hit_2r_before_sl=hit_2r,
        hit_3r_before_sl=hit_3r,
        feature_flags=feature_flags,
        feature_tags=tuple(
            FEATURE_DEFINITIONS[name]
            for name, active in sorted(feature_flags.items())
            if active
        ),
    )


def test_feature_definitions_cover_required_traits() -> None:
    required = {
        "near_support",
        "near_resistance",
        "liquidity_sweep",
        "false_breakout",
        "false_breakdown",
        "choch_present",
        "bos_present",
        "fvg_reclaim",
        "order_block_reaction",
        "strong_confirmation",
        "rsi_below_40",
        "rsi_above_60",
        "rsi_divergence",
        "ema_bull_stack",
        "ema_bear_stack",
        "above_vwap",
        "below_vwap",
        "round_number",
        "level_strong",
        "session_morning",
        "gap_up",
        "gap_down",
        "htf_aligned",
    }
    assert required.issubset(set(FEATURE_DEFINITIONS))


def test_matches_requires_all_features() -> None:
    trade = _trade(flags={"near_support": True, "liquidity_sweep": False})
    assert SmartMoneyEngineProductionCandidateResearch._matches(trade, ("near_support",))
    assert not SmartMoneyEngineProductionCandidateResearch._matches(
        trade,
        ("near_support", "liquidity_sweep"),
    )


def test_aggregate_metrics_baseline_label() -> None:
    engine = SmartMoneyEngineProductionCandidateResearch()
    trades = [_trade(side="BUY"), _trade(side="BUY", pnl=60.0)]
    metrics = engine._aggregate_metrics((), trades, research_days=365)
    assert metrics.model_key == "baseline_buy"
    assert "Baseline BUY" in metrics.model_label
    assert metrics.trades == 2
    assert metrics.expectancy == pytest.approx(70.0)


def test_evaluate_combinations_filters_by_thresholds() -> None:
    engine = SmartMoneyEngineProductionCandidateResearch()
    winners = [
        _trade(
            flags={"near_support": True, "liquidity_sweep": True, "rsi_below_40": True},
            pnl=100.0,
        )
        for _ in range(MIN_SAMPLES)
    ]
    losers = [
        _trade(
            flags={"near_support": True, "liquidity_sweep": True, "rsi_below_40": True},
            win=False,
            pnl=-20.0,
            hit_1r=False,
            hit_2r=False,
        )
        for _ in range(10)
    ]
    trades = winners + losers
    eligible, rejected = engine._evaluate_combinations(trades, research_days=365)
    assert eligible
    top = eligible[0]
    assert top.trades >= MIN_SAMPLES
    assert (top.profit_factor or 0) >= MIN_PROFIT_FACTOR
    assert top.expectancy >= MIN_EXPECTANCY


def test_rank_models_assigns_ranks() -> None:
    engine = SmartMoneyEngineProductionCandidateResearch()
    trades_a = [_trade(pnl=120.0, hit_3r=True) for _ in range(MIN_SAMPLES)]
    trades_b = [_trade(pnl=90.0, hit_3r=False) for _ in range(MIN_SAMPLES)]
    model_a = engine._aggregate_metrics(("near_support",), trades_a, 365)
    model_b = engine._aggregate_metrics(("liquidity_sweep",), trades_b, 365)
    ranked = engine._rank_models([model_a, model_b])
    assert len(ranked) == 2
    assert ranked[0].overall_rank == 1
    assert ranked[0].rank_1r >= 1


def test_generate_report(tmp_path: Path) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2025-01-01", "end_date": "2026-01-01", "research_window_days": 365}',
        encoding="utf-8",
    )
    destination = tmp_path / "smartmoneyengine_production_candidate.json"
    report = generate_production_candidate_report(
        report_path=destination,
        filter_report_path=filter_report,
        symbol="NIFTY50",
    )
    assert destination.exists()
    assert report.total_historical_moves >= 0
    assert "recommended_production_signal_engine" in report.as_dict()
    assert report.recommended_production_signal_engine.get("entry")
    assert report.recommended_production_signal_engine.get("stop_loss")
    assert report.recommended_production_signal_engine.get("t1")
    assert report.recommended_production_signal_engine.get("t2")
    assert report.recommended_production_signal_engine.get("t3")
