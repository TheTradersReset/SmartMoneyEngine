"""Tests for unified production replay validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.unified_production_replay_validation_research import (
    BUY_V3_MODEL_ID,
    SELL_V5_MODEL_ID,
    UnifiedProductionReplayValidationResearch,
    _capital_curve_metrics,
    _classify_signals,
    _engine_comparison_three_way,
    _final_answer,
    _human_tradeability,
    _production_readiness_score,
    _reconcile_synthesis,
    _tier_capture_from_signals,
)


def _sample_signal(
    *,
    bar: int,
    direction: str,
    pnl: float,
    win: bool,
    mfe: float = 50.0,
    timestamp: str | None = None,
) -> dict:
    return {
        "bar": bar,
        "timestamp": timestamp or f"2026-01-15 10:{bar % 60:02d}:00",
        "direction": direction,
        "realized_pnl_points": pnl,
        "win": win,
        "mfe_points": mfe,
        "mae_points": 10.0,
    }


def test_model_constants() -> None:
    assert BUY_V3_MODEL_ID == "LDM-BUY-V3"
    assert SELL_V5_MODEL_ID == "LDM-SELL-V5"


def test_tier_capture_from_signals() -> None:
    signals = [
        _sample_signal(bar=1, direction="BUY", pnl=20, win=True, mfe=45),
        _sample_signal(bar=2, direction="SELL", pnl=30, win=True, mfe=120),
    ]
    tiers = _tier_capture_from_signals(signals)
    assert tiers["40"]["signals_hitting_tier"] == 2
    assert tiers["100"]["signals_hitting_tier"] == 1


def test_classify_signals_overlap_and_conflict() -> None:
    buy = [
        _sample_signal(bar=10, direction="BUY", pnl=20, win=True, timestamp="2026-01-15 10:00:00"),
        _sample_signal(bar=20, direction="BUY", pnl=-10, win=False, timestamp="2026-01-16 10:00:00"),
    ]
    sell = [
        _sample_signal(bar=10, direction="SELL", pnl=15, win=True, timestamp="2026-01-15 10:05:00"),
        _sample_signal(bar=30, direction="SELL", pnl=25, win=True, timestamp="2026-01-17 10:00:00"),
    ]
    result = _classify_signals(buy, sell)
    assert result["same_bar_overlap_count"] == 1
    assert result["buy_only_count"] == 1
    assert result["sell_only_count"] == 1
    assert result["same_day_both_engines_count"] == 1
    assert result["conflict_bar_summary"]["count"] == 1


def test_capital_curve_metrics() -> None:
    signals = [
        _sample_signal(bar=1, direction="BUY", pnl=30, win=True),
        _sample_signal(bar=2, direction="SELL", pnl=-10, win=False),
        _sample_signal(bar=3, direction="BUY", pnl=20, win=True),
    ]
    curve = _capital_curve_metrics(signals)
    assert curve["net_points"] == 40.0
    assert curve["max_drawdown_points"] == 10.0
    assert curve["profit_distribution"]["win_count"] == 2


def test_human_tradeability() -> None:
    signals = [
        _sample_signal(bar=1, direction="BUY", pnl=10, win=True, timestamp="2026-01-15 10:00:00"),
        _sample_signal(bar=2, direction="SELL", pnl=10, win=True, timestamp="2026-01-15 11:00:00"),
        _sample_signal(bar=3, direction="BUY", pnl=10, win=True, timestamp="2026-01-16 10:00:00"),
    ]
    result = _human_tradeability(signals)
    assert result["trading_days_with_signals"] == 2
    assert result["avg_signals_per_day"] == 1.5
    assert result["max_signals_per_day"] == 2


def test_engine_comparison_three_way() -> None:
    buy = [_sample_signal(bar=1, direction="BUY", pnl=20, win=True)]
    sell = [_sample_signal(bar=2, direction="SELL", pnl=30, win=True)]
    combined = buy + sell
    result = _engine_comparison_three_way(
        buy,
        sell,
        combined,
        moves=[],
        frame=__import__("pandas").DataFrame({"Date": ["2026-01-15"]}),
        replay_dates=set(),
        trading_days=120,
    )
    assert result["buy_v3_only"]["overall_statistics"]["signals_emitted"] == 1
    assert result["sell_v5_only"]["overall_statistics"]["signals_emitted"] == 1
    assert result["combined"]["overall_statistics"]["signals_emitted"] == 2


def test_reconcile_synthesis_match_and_no_match(tmp_path: Path) -> None:
    synthesis_path = tmp_path / "synthesis.json"
    synthesis_path.write_text(
        json.dumps(
            {
                "combined_engine_simulation": {
                    "simulation_basis": "proxy",
                    "combined_metrics": {
                        "signals_per_month": 100.0,
                        "win_rate_pct": 70.0,
                        "profit_factor": 2.5,
                        "expectancy": 15.0,
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    replay_combined = {
        "overall_statistics": {
            "signals_per_month": 102.0,
            "win_rate_pct": 71.0,
            "profit_factor": 2.6,
            "expectancy": 15.5,
        },
    }
    match = _reconcile_synthesis(replay_combined, synthesis_path)
    assert match["synthesis_match"] == "yes"

    mismatch = _reconcile_synthesis(
        {"overall_statistics": {"signals_per_month": 50.0, "win_rate_pct": 40.0}},
        synthesis_path,
    )
    assert mismatch["synthesis_match"] == "no"


def test_production_readiness_and_final_answer() -> None:
    buy_stats = {
        "win_rate_pct": 70.0,
        "profit_factor": 2.5,
        "signals_per_month": 22.0,
        "expectancy": 12.0,
    }
    sell_stats = {
        "win_rate_pct": 68.0,
        "profit_factor": 3.2,
        "signals_per_month": 70.0,
        "expectancy": 18.0,
    }
    combined_stats = {
        "win_rate_pct": 69.0,
        "profit_factor": 2.8,
        "signals_per_month": 92.0,
        "expectancy": 16.0,
    }
    classification = {"same_bar_conflict_rate_pct": 0.5}
    capital_curve = {"max_drawdown_points": 50.0, "recovery_factor": 2.5}

    readiness = _production_readiness_score(
        buy_stats=buy_stats,
        sell_stats=sell_stats,
        combined_stats=combined_stats,
        classification=classification,
        walk_forward_stable=True,
        capital_curve=capital_curve,
    )
    assert readiness["score"] > 0
    assert readiness["recommendation_tier"] in {
        "Research",
        "Dry Run",
        "Paper Trading",
        "Production Candidate",
    }

    final = _final_answer(
        buy_stats=buy_stats,
        sell_stats=sell_stats,
        combined_stats=combined_stats,
        classification=classification,
        walk_forward_stable=True,
        readiness=readiness,
    )
    assert final["can_operate_as_single_production_engine"] in {"YES", "NO", "PARTIAL"}
    assert "production_readiness_score" in final


def test_generate_report_requires_filter_report(tmp_path: Path) -> None:
    from src.research.unified_production_replay_validation_research import (
        UnifiedProductionReplayValidationError,
        generate_unified_production_replay_validation_report,
    )

    missing = tmp_path / "missing.json"
    with pytest.raises(UnifiedProductionReplayValidationError):
        generate_unified_production_replay_validation_report(filter_report_path=missing)


def test_research_class_instantiation() -> None:
    research = UnifiedProductionReplayValidationResearch()
    assert research.buy_engine.MODEL_ID == BUY_V3_MODEL_ID
    assert research.sell_engine.MODEL_ID == SELL_V5_MODEL_ID
