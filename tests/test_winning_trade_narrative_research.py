"""Tests for winning trade narrative research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.context.liquidity_narrative_engine import FvgContext, LiquidityEventSnapshot, StructureShiftSnapshot
from src.context.market_intelligence_engine import RsiState
from src.research.winning_trade_narrative_research import (
    MIN_INTELLIGENCE_SCORE,
    NarrativeTradeRecord,
    WinningTradeNarrativeError,
    WinningTradeNarrativeResearch,
    generate_winning_trade_narrative_report,
)


def _trade(
    outcome: str = "Win",
    pnl: float = 10.0,
    liquidity: str = "Buy Side Sweep",
    structure: str = "Bearish CHOCH + Bearish BOS",
    fvg: str = "Bearish FVG Reclaim",
) -> NarrativeTradeRecord:
    return NarrativeTradeRecord(
        setup_type="Liquidity Grab + FVG Reclaim",
        direction="bearish",
        direction_label="SELL",
        timeframe="5M",
        session="Midday",
        trigger_timestamp="2026-01-06 12:00:00+05:30",
        outcome=outcome,
        realized_pnl_points=pnl,
        realized_rr=1.0 if pnl > 0 else -1.0,
        liquidity_event=liquidity,
        structure_sequence=structure,
        structure_bucket="CHOCH + BOS",
        fvg_state=fvg,
        market_location="Resistance",
        rsi_state="Neutral",
        intelligence_score=72.0,
        narrative_strength_score=85.0,
        core_narrative_sequence=WinningTradeNarrativeResearch._core_sequence(
            liquidity,
            structure,
            fvg,
        ),
        full_narrative_sequence="test",
    )


def test_liquidity_event_labels() -> None:
    both = LiquidityEventSnapshot(True, True, 1.0, 1.0, 1.0, 1.0)
    buy = LiquidityEventSnapshot(True, False, 1.0, None, 1.0, 1.0)
    none = LiquidityEventSnapshot(False, False, None, None, None, None)
    assert WinningTradeNarrativeResearch._liquidity_event_label(both) == "Both"
    assert WinningTradeNarrativeResearch._liquidity_event_label(buy) == "Buy Side Sweep"
    assert WinningTradeNarrativeResearch._liquidity_event_label(none) == "None"


def test_structure_labels() -> None:
    structure = StructureShiftSnapshot(
        bullish_choch=False,
        bearish_choch=True,
        bullish_bos=False,
        bearish_bos=True,
        latest_bullish_choch_price=None,
        latest_bearish_choch_price=99.0,
        latest_bullish_bos_price=None,
        latest_bearish_bos_price=98.0,
    )
    detailed, bucket = WinningTradeNarrativeResearch._structure_labels(structure)
    assert detailed == "Bearish CHOCH + Bearish BOS"
    assert bucket == "CHOCH + BOS"


def test_fvg_state_label() -> None:
    assert (
        WinningTradeNarrativeResearch._fvg_state_label(FvgContext.RECLAIMED, "bearish")
        == "Bearish FVG Reclaim"
    )
    assert WinningTradeNarrativeResearch._fvg_state_label(FvgContext.FAILED, None) == "Failed"


def test_rsi_bucket() -> None:
    assert WinningTradeNarrativeResearch._rsi_bucket(RsiState.WEAK) == "Weak"
    assert WinningTradeNarrativeResearch._rsi_bucket(RsiState.NEUTRAL) == "Neutral"
    assert WinningTradeNarrativeResearch._rsi_bucket(RsiState.STRONG) == "Strong"


def test_core_sequence() -> None:
    sequence = WinningTradeNarrativeResearch._core_sequence(
        "Buy Side Sweep",
        "Bearish CHOCH + Bearish BOS",
        "Bearish FVG Reclaim",
    )
    assert "Buy Side Sweep" in sequence
    assert "Bearish CHOCH" in sequence
    assert "Bearish FVG Reclaim" in sequence


def test_sequence_metrics() -> None:
    engine = WinningTradeNarrativeResearch()
    metrics = engine._sequence_metrics(
        [_trade(pnl=20.0), _trade(outcome="Loss", pnl=-10.0)],
        "test",
        total_winners=1,
    )
    assert metrics.trades == 2
    assert metrics.wins == 1
    assert metrics.win_rate_pct == 50.0
    assert metrics.expectancy == 5.0


def test_large_win_threshold() -> None:
    engine = WinningTradeNarrativeResearch()
    winners = [_trade(pnl=10.0), _trade(pnl=20.0), _trade(pnl=30.0), _trade(pnl=40.0)]
    threshold = engine._large_win_threshold(winners)
    assert threshold >= 30.0


def test_most_common_before_large_wins() -> None:
    engine = WinningTradeNarrativeResearch()
    winners = [
        _trade(pnl=50.0, liquidity="Buy Side Sweep"),
        _trade(pnl=45.0, liquidity="Buy Side Sweep"),
        _trade(pnl=10.0, liquidity="Sell Side Sweep"),
    ]
    result = engine._most_common_before_large_wins(winners, threshold=40.0)
    assert result["large_win_count"] == 2
    assert "Buy Side Sweep" in result["most_common_sequence"]


def test_report_structure() -> None:
    from src.research.winning_trade_narrative_research import WinningTradeNarrativeReport

    report = WinningTradeNarrativeReport(
        symbol="NIFTY50",
        research_window_days=365,
        start_date="2025-07-03",
        end_date="2026-07-03",
        production_stack={},
        total_stack_trades=100,
        total_winning_trades=60,
        total_losing_trades=40,
        large_win_pnl_threshold=25.0,
        by_liquidity_event={},
        by_structure_sequence={},
        by_fvg_state={},
        by_market_location={},
        by_rsi_state={},
        narrative_sequences=[],
        top_20_narrative_sequences=[],
        most_common_before_large_wins={},
        winning_trade_summary={},
        sample_winning_trades=[],
        conclusions=[],
        execution_time_seconds=1.0,
    )
    payload = report.as_dict()
    assert payload["total_winning_trades"] == 60
    assert "top_20_narrative_sequences" in payload


def test_missing_metadata_raises() -> None:
    with pytest.raises(WinningTradeNarrativeError):
        generate_winning_trade_narrative_report(filter_report_path=Path("missing.json"))


def test_min_intelligence_score_constant() -> None:
    assert MIN_INTELLIGENCE_SCORE == 65


@pytest.mark.integration
def test_full_research_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parent.parent
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = WinningTradeNarrativeResearch().run(metadata)
    assert report.total_stack_trades >= 0
