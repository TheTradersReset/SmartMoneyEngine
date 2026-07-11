"""Tests for RSI divergence research engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.filter_research_engine import FilterState, FilteredTradeRecord
from src.research.rsi_divergence_research_engine import (
    DivergenceType,
    RsiDivergenceDetector,
    RsiDivergenceResearchEngine,
    RsiDivergenceResearchError,
    generate_rsi_divergence_research_report,
)
from src.signals.setup_classifier import SetupType


def _frame(length: int = 80) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-06 09:15:00+05:30")
    for index in range(length):
        price = 100.0 + index * 0.1
        rows.append(
            {
                "Date": (base + pd.Timedelta(minutes=5 * index)).isoformat(),
                "Open": price,
                "High": price + 0.8,
                "Low": price - 0.5,
                "Close": price + 0.1,
                "Volume": 1_000_000,
                "Trend": "BULLISH",
                "Trend_Strength": 2,
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bullish_FVG_Bottom": pd.NA,
                "Bearish_FVG_Top": pd.NA,
                "Bearish_FVG_Bottom": pd.NA,
                "Bullish_OB_High": pd.NA,
                "Bullish_OB_Low": pd.NA,
                "Bearish_OB_High": pd.NA,
                "Bearish_OB_Low": pd.NA,
                "Bullish_OB_Mitigated": pd.NA,
                "Bearish_OB_Mitigated": pd.NA,
                "Buy_Liquidity_Sweep": pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
                "Liquidity_Strength": 2,
                "Swing_High": pd.NA,
                "Swing_Low": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def test_bullish_divergence_detection() -> None:
    frame = _frame(40)
    detector = RsiDivergenceDetector(pivot_strength=2, lookback=30)
    rsi = detector._compute_rsi(frame["Close"].astype(float))

    frame.loc[10, "Low"] = 95.0
    frame.loc[10, "Close"] = 96.0
    frame.loc[20, "Low"] = 92.0
    frame.loc[20, "Close"] = 93.0
    rsi.iloc[10] = 30.0
    rsi.iloc[20] = 35.0

    types = detector.detect(frame, 25, rsi)
    assert DivergenceType.BULLISH in types


def test_bearish_divergence_detection() -> None:
    frame = _frame(40)
    detector = RsiDivergenceDetector(pivot_strength=2, lookback=30)
    rsi = detector._compute_rsi(frame["Close"].astype(float))

    frame.loc[10, "High"] = 105.0
    frame.loc[20, "High"] = 108.0
    rsi.iloc[10] = 70.0
    rsi.iloc[20] = 65.0

    types = detector.detect(frame, 25, rsi)
    assert DivergenceType.BEARISH in types


def test_hidden_bullish_divergence_detection() -> None:
    frame = _frame(40)
    detector = RsiDivergenceDetector(pivot_strength=2, lookback=30)
    rsi = detector._compute_rsi(frame["Close"].astype(float))

    frame.loc[10, "Low"] = 92.0
    frame.loc[20, "Low"] = 94.0
    rsi.iloc[10] = 40.0
    rsi.iloc[20] = 35.0

    types = detector.detect(frame, 25, rsi)
    assert DivergenceType.HIDDEN_BULLISH in types


def test_primary_divergence_prefers_direction() -> None:
    types = [DivergenceType.BEARISH, DivergenceType.BULLISH]
    primary = RsiDivergenceDetector.primary_divergence(types, "bullish")
    assert primary == DivergenceType.BULLISH


def test_metrics_calculation() -> None:
    from src.research.rsi_divergence_research_engine import DivergenceTradeRecord

    engine = RsiDivergenceResearchEngine()
    trades = [
        DivergenceTradeRecord(
            setup_type=SetupType.CONTINUATION_BOS.value,
            direction="bullish",
            direction_label="BUY",
            timeframe="5M",
            session="Midday",
            trigger_bar=1,
            trigger_timestamp="2026-01-06 09:15:00+05:30",
            entry_hit=True,
            outcome="Win",
            realized_pnl_points=10.0,
            realized_rr=1.0,
            divergence_types=("Bullish Divergence",),
            primary_divergence="Bullish Divergence",
            has_divergence=True,
            production_stack=False,
        )
    ]
    metrics = engine._metrics(trades, "test")
    assert metrics.trades == 1
    assert metrics.expectancy == 10.0


def test_with_vs_without_comparison() -> None:
    from src.research.rsi_divergence_research_engine import DivergenceTradeRecord

    engine = RsiDivergenceResearchEngine()
    trades = [
        DivergenceTradeRecord(
            setup_type=SetupType.CONTINUATION_BOS.value,
            direction="bullish",
            direction_label="BUY",
            timeframe="5M",
            session="Midday",
            trigger_bar=1,
            trigger_timestamp="2026-01-06 09:15:00+05:30",
            entry_hit=True,
            outcome="Win",
            realized_pnl_points=10.0,
            realized_rr=1.0,
            divergence_types=("Bullish Divergence",),
            primary_divergence="Bullish Divergence",
            has_divergence=True,
            production_stack=False,
        ),
        DivergenceTradeRecord(
            setup_type=SetupType.CONTINUATION_BOS.value,
            direction="bullish",
            direction_label="BUY",
            timeframe="5M",
            session="Midday",
            trigger_bar=2,
            trigger_timestamp="2026-01-06 09:20:00+05:30",
            entry_hit=True,
            outcome="Loss",
            realized_pnl_points=-5.0,
            realized_rr=-1.0,
            divergence_types=(),
            primary_divergence="None",
            has_divergence=False,
            production_stack=False,
        ),
    ]
    comparison = engine._comparison(trades, "Continuation BOS")
    assert comparison.with_divergence["trades"] == 1
    assert comparison.without_divergence["trades"] == 1
    assert comparison.divergence_improves_expectancy is True


def test_combination_pool_min_trades() -> None:
    from src.research.rsi_divergence_research_engine import DivergenceTradeRecord

    engine = RsiDivergenceResearchEngine()
    trades = [
        DivergenceTradeRecord(
            setup_type=SetupType.CONTINUATION_BOS.value,
            direction="bullish",
            direction_label="BUY",
            timeframe="5M",
            session="Midday",
            trigger_bar=index,
            trigger_timestamp=f"2026-01-06 09:{15 + index}:00+05:30",
            entry_hit=True,
            outcome="Win",
            realized_pnl_points=10.0,
            realized_rr=1.0,
            divergence_types=("Bullish Divergence",),
            primary_divergence="Bullish Divergence",
            has_divergence=True,
            production_stack=False,
        )
        for index in range(4)
    ]
    assert engine._combination_pool(trades) == []


def test_report_structure() -> None:
    from src.research.rsi_divergence_research_engine import RsiDivergenceResearchReport

    report = RsiDivergenceResearchReport(
        symbol="NIFTY50",
        research_window_days=365,
        start_date="2025-07-03",
        end_date="2026-07-03",
        timeframes_analyzed=["5M"],
        setups_analyzed=["Continuation BOS"],
        production_stack={"setup": "Liquidity Grab + FVG Reclaim", "filters": {}, "total_trades": 0},
        total_trades=0,
        setup_comparisons={},
        production_stack_comparison={},
        by_divergence_type={},
        by_timeframe={},
        by_direction={},
        by_session={},
        best_divergence_combinations=[],
        worst_divergence_combinations=[],
        conclusions=[],
        execution_time_seconds=1.0,
    )
    payload = report.as_dict()
    assert "setup_comparisons" in payload
    assert "production_stack_comparison" in payload


def test_missing_metadata_raises() -> None:
    with pytest.raises(RsiDivergenceResearchError):
        generate_rsi_divergence_research_report(
            filter_report_path=Path("missing_filter_report.json"),
        )


def test_max_drawdown() -> None:
    drawdown = RsiDivergenceResearchEngine._max_drawdown([10.0, -5.0, -8.0])
    assert drawdown == 13.0


@pytest.mark.integration
def test_full_research_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parent.parent
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    report = RsiDivergenceResearchEngine().run(json.loads(filter_report.read_text(encoding="utf-8")))
    assert report.total_trades > 0
