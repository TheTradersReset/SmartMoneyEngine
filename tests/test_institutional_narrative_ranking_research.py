"""Tests for institutional narrative ranking research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.context.liquidity_narrative_engine import FvgContext
from src.research.rsi_divergence_research_engine import DivergenceType
from src.research.institutional_narrative_ranking_research import (
    MIN_INTELLIGENCE_SCORE,
    InstitutionalNarrativeRankingError,
    InstitutionalNarrativeRankingResearch,
    InstitutionalTradeRecord,
    generate_institutional_narrative_ranking_report,
)


def _trade(
    outcome: str = "Win",
    pnl: float = 60.0,
    narrative: str = "test narrative",
) -> InstitutionalTradeRecord:
    return InstitutionalTradeRecord(
        setup_type="Liquidity Grab + FVG Reclaim",
        direction="bearish",
        direction_label="SELL",
        timeframe="5M",
        session="Midday",
        trigger_timestamp="2026-01-06 12:00:00+05:30",
        outcome=outcome,
        realized_pnl_points=pnl,
        realized_rr=1.0 if pnl > 0 else -1.0,
        liquidity_event="Buy Side Sweep",
        choch_state="Bearish CHOCH",
        bos_state="Bearish BOS",
        fvg_context="Bearish FVG Reclaim",
        intelligence_score=72.0,
        intelligence_bucket="MI 70-79",
        rsi_band="RSI 50-60",
        rsi_divergence="RSI Bearish Divergence",
        market_location="Near Resistance",
        institutional_narrative=narrative,
        narrative_key=narrative,
    )


def test_build_narrative() -> None:
    narrative, key = InstitutionalNarrativeRankingResearch._build_narrative(
        "Buy Side Sweep",
        "Bearish CHOCH",
        "Bearish BOS",
        "Bearish FVG Reclaim",
        "MI 70-79",
        "RSI 50-60",
        "RSI Bearish Divergence",
        "Near Resistance",
    )
    assert "Buy Side Sweep" in narrative
    assert "Bearish CHOCH" in narrative
    assert "Near Resistance" in narrative
    assert narrative == key


def test_intelligence_bucket() -> None:
    assert InstitutionalNarrativeRankingResearch._intelligence_bucket(66.0) == "MI 65-69"
    assert InstitutionalNarrativeRankingResearch._intelligence_bucket(75.0) == "MI 70-79"
    assert InstitutionalNarrativeRankingResearch._intelligence_bucket(85.0) == "MI 80-100"


def test_divergence_label() -> None:
    assert (
        InstitutionalNarrativeRankingResearch._divergence_label(DivergenceType.BEARISH)
        == "RSI Bearish Divergence"
    )
    assert (
        InstitutionalNarrativeRankingResearch._divergence_label(DivergenceType.NONE)
        == "No RSI Divergence"
    )


def test_fvg_label() -> None:
    assert (
        InstitutionalNarrativeRankingResearch._fvg_label(FvgContext.RECLAIMED, "bearish")
        == "Bearish FVG Reclaim"
    )


def test_metrics_with_drawdown() -> None:
    engine = InstitutionalNarrativeRankingResearch()
    metrics = engine._metrics(
        [_trade(pnl=50.0), _trade(outcome="Loss", pnl=-20.0)],
        "test",
    )
    assert metrics.trades == 2
    assert metrics.max_drawdown >= 0.0
    assert metrics.expectancy == 15.0


def test_rank_narratives() -> None:
    engine = InstitutionalNarrativeRankingResearch()
    trades = [
        _trade(narrative="A", pnl=50.0),
        _trade(narrative="A", pnl=40.0),
        _trade(narrative="B", pnl=-10.0, outcome="Loss"),
    ]
    ranked = engine._rank_narratives(trades)
    assert len(ranked) == 2


def test_large_move_analysis() -> None:
    engine = InstitutionalNarrativeRankingResearch()
    trades = [
        _trade(narrative="Big", pnl=120.0),
        _trade(narrative="Big", pnl=110.0),
        _trade(narrative="Small", pnl=30.0),
    ]
    analysis = engine._large_move_analysis(trades)
    assert analysis["100"]["trade_count"] == 2
    assert analysis["100"]["most_common_narrative"] == "Big"


def test_report_structure() -> None:
    from src.research.institutional_narrative_ranking_research import (
        InstitutionalNarrativeRankingReport,
    )

    report = InstitutionalNarrativeRankingReport(
        symbol="NIFTY50",
        research_window_days=365,
        start_date="2025-07-03",
        end_date="2026-07-03",
        production_stack={},
        total_trades=100,
        total_winning_trades=60,
        total_losing_trades=40,
        unique_narratives=25,
        ranked_narratives=[],
        top_20_narratives=[],
        worst_20_narratives=[],
        large_move_analysis={},
        conclusions=[],
        execution_time_seconds=1.0,
    )
    payload = report.as_dict()
    assert payload["unique_narratives"] == 25
    assert "top_20_narratives" in payload


def test_missing_metadata_raises() -> None:
    with pytest.raises(InstitutionalNarrativeRankingError):
        generate_institutional_narrative_ranking_report(
            filter_report_path=Path("missing.json"),
        )


def test_min_intelligence_score_constant() -> None:
    assert MIN_INTELLIGENCE_SCORE == 65


@pytest.mark.integration
def test_full_ranking_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parent.parent
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = InstitutionalNarrativeRankingResearch().run(metadata)
    assert report.total_trades >= 0
