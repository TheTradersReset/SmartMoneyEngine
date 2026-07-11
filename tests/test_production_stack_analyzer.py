"""Tests for production stack analyzer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.research.filter_research_engine import FilterState, FilteredTradeRecord
from src.research.production_stack_analyzer import (
    PRODUCTION_FILTERS,
    PRODUCTION_SETUP,
    ProductionStackAnalyzer,
    StackTrade,
)
from src.signals.setup_classifier import SetupType


def _filtered_trade(
    index: int,
    outcome: str,
    pnl: float,
    timeframe: str = "5M",
    direction: str = "bullish",
    session: str = "Midday",
) -> FilteredTradeRecord:
    base = pd.Timestamp("2026-01-06 09:15:00+05:30")
    return FilteredTradeRecord(
        setup_type=PRODUCTION_SETUP,
        direction=direction,
        timeframe=timeframe,
        trigger_bar=index,
        trigger_timestamp=(base + pd.Timedelta(minutes=5 * index)).isoformat(),
        entry_hit=True,
        outcome=outcome,
        realized_pnl_points=pnl,
        realized_rr=1.0 if pnl > 0 else -1.0,
        filters=FilterState(
            ema_alignment="Mixed",
            vwap_position="Above VWAP",
            rsi_band=PRODUCTION_FILTERS["rsi_band"],
            session=session,
            atr_percentile="Mid (34-66)",
            volume_spike=PRODUCTION_FILTERS["volume_spike"],
        ),
    )


def test_matches_production_stack() -> None:
    trade = _filtered_trade(1, "Win", 10.0)
    assert ProductionStackAnalyzer._matches_production_stack(trade) is True


def test_rejects_non_production_setup() -> None:
    trade = _filtered_trade(1, "Win", 10.0)
    other = FilteredTradeRecord(
        setup_type=SetupType.CONTINUATION_BOS.value,
        direction=trade.direction,
        timeframe=trade.timeframe,
        trigger_bar=trade.trigger_bar,
        trigger_timestamp=trade.trigger_timestamp,
        entry_hit=trade.entry_hit,
        outcome=trade.outcome,
        realized_pnl_points=trade.realized_pnl_points,
        realized_rr=trade.realized_rr,
        filters=trade.filters,
    )
    assert ProductionStackAnalyzer._matches_production_stack(other) is False


def test_direction_label_mapping() -> None:
    assert ProductionStackAnalyzer._direction_label("bullish") == "BUY"
    assert ProductionStackAnalyzer._direction_label("bearish") == "SELL"


def test_segment_metrics() -> None:
    analyzer = ProductionStackAnalyzer()
    trades = [
        StackTrade(
            setup_type=PRODUCTION_SETUP,
            direction="bullish",
            direction_label="BUY",
            timeframe="1H",
            session="Closing",
            day_of_week="Friday",
            trigger_timestamp="2026-01-09 15:00:00+05:30",
            outcome="Win",
            realized_pnl_points=10.0,
            realized_rr=1.0,
        )
    ]
    metrics = analyzer._segment_metrics("timeframe", "1H", trades)
    assert metrics.trades == 1
    assert metrics.win_rate_pct == 100.0
    assert metrics.expectancy == 10.0


def test_group_metrics_by_timeframe() -> None:
    analyzer = ProductionStackAnalyzer()
    trades = [
        analyzer._to_stack_trade(_filtered_trade(1, "Win", 10.0, timeframe="5M")),
        analyzer._to_stack_trade(_filtered_trade(2, "Loss", -5.0, timeframe="1H")),
    ]
    grouped = analyzer._group_metrics(trades, "timeframe", lambda trade: trade.timeframe)
    assert "5M" in grouped
    assert "1H" in grouped


def test_configuration_description() -> None:
    description = ProductionStackAnalyzer._configuration_description(
        {
            "setup": PRODUCTION_SETUP,
            "rsi_band": "50-60",
            "timeframe": "1H",
            "session": "Closing",
            "direction_label": "BUY",
        }
    )
    assert "1H" in description
    assert "Closing Session" in description
    assert "BUY" in description


def test_ranking_pool_includes_multi_dimension_segments() -> None:
    analyzer = ProductionStackAnalyzer()
    trades = [
        analyzer._to_stack_trade(
            _filtered_trade(index, "Win", 10.0, timeframe="1H", session="Closing")
        )
        for index in range(6)
    ]
    segments = analyzer._configuration_segments(trades, min_trades=5)
    full = [segment for segment in segments if len(segment.segment_key) == 4]
    assert full


def test_report_structure() -> None:
    from src.research.production_stack_analyzer import ProductionStackReport

    report = ProductionStackReport(
        symbol="NIFTY50",
        research_window_days=365,
        start_date="2025-07-03",
        end_date="2026-07-03",
        production_stack={"setup": PRODUCTION_SETUP, "filters": PRODUCTION_FILTERS, "total_trades": 423},
        overall_metrics={},
        by_timeframe={},
        by_direction={},
        by_session={},
        by_day_of_week={},
        top_10_profitable_segments=[],
        worst_10_segments=[],
        best_production_configuration={},
        execution_time_seconds=1.0,
    )
    payload = report.as_dict()
    assert payload["production_stack"]["total_trades"] == 423
    assert "by_timeframe" in payload


def test_max_drawdown() -> None:
    drawdown = ProductionStackAnalyzer._max_drawdown([20.0, -10.0, -15.0, 25.0])
    assert drawdown == 25.0


def test_rejects_wrong_rsi_band() -> None:
    trade = _filtered_trade(1, "Win", 10.0)
    wrong = FilteredTradeRecord(
        setup_type=trade.setup_type,
        direction=trade.direction,
        timeframe=trade.timeframe,
        trigger_bar=trade.trigger_bar,
        trigger_timestamp=trade.trigger_timestamp,
        entry_hit=trade.entry_hit,
        outcome=trade.outcome,
        realized_pnl_points=trade.realized_pnl_points,
        realized_rr=trade.realized_rr,
        filters=FilterState(
            ema_alignment="Mixed",
            vwap_position="Above VWAP",
            rsi_band="40-50",
            session="Midday",
            atr_percentile="Mid (34-66)",
            volume_spike="No",
        ),
    )
    assert ProductionStackAnalyzer._matches_production_stack(wrong) is False


@pytest.mark.integration
def test_full_production_stack_analysis_if_pipelines_exist() -> None:
    project_root = Path(__file__).resolve().parent.parent
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    report = ProductionStackAnalyzer().run()
    assert report.production_stack["total_trades"] > 0
    assert report.overall_metrics["trades"] == report.production_stack["total_trades"]
