"""Tests for setup filter research engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.filter_research_engine import (
    FILTER_DIMENSIONS,
    PROFITABLE_SETUPS,
    FilterContextBuilder,
    FilterResearchEngine,
    FilterState,
    FilteredTradeRecord,
)
from src.signals.setup_classifier import SetupType


def _frame(length: int = 120) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-06 09:15:00+05:30")
    for index in range(length):
        price = 100.0 + index * 0.4
        rows.append(
            {
                "Date": (base + pd.Timedelta(minutes=5 * index)).isoformat(),
                "Open": price,
                "High": price + 1.0,
                "Low": price - 0.6,
                "Close": price + 0.2,
                "Volume": 1_000_000 + index * 10_000,
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


def test_profitable_setups_only() -> None:
    assert SetupType.LIQUIDITY_GRAB_FVG_RECLAIM.value in PROFITABLE_SETUPS
    assert SetupType.CONTINUATION_BOS.value in PROFITABLE_SETUPS
    assert SetupType.CHOCH_FVG.value not in PROFITABLE_SETUPS


def test_session_label_midday() -> None:
    ts = pd.Timestamp("2026-01-06 12:00:00+05:30")
    assert FilterContextBuilder._session_label(ts) == "Midday"


def test_rsi_band_buckets() -> None:
    assert FilterContextBuilder._rsi_band(45.0) == "40-50"
    assert FilterContextBuilder._rsi_band(55.0) == "50-60"
    assert FilterContextBuilder._rsi_band(65.0) == "60-70"


def test_ema_alignment_labels() -> None:
    assert (
        FilterContextBuilder._ema_alignment_label(110.0, 105.0, 100.0)
        == "EMA20 > EMA50 > EMA200"
    )
    assert (
        FilterContextBuilder._ema_alignment_label(90.0, 95.0, 100.0)
        == "EMA20 < EMA50 < EMA200"
    )
    assert FilterContextBuilder._ema_alignment_label(100.0, 95.0, 100.0) == "Mixed"


def test_context_builder_enriches_indicators() -> None:
    frame = _frame(80)
    enriched = FilterContextBuilder().enrich(frame)
    assert "_ema_20" in enriched.columns
    assert "_rsi" in enriched.columns
    assert "_vwap" in enriched.columns
    assert "_atr" in enriched.columns


def test_filter_state_has_all_dimensions() -> None:
    frame = _frame(80)
    builder = FilterContextBuilder()
    enriched = builder.enrich(frame)
    state = builder.filter_state(enriched, 50)
    assert set(state.as_dict()) == set(FILTER_DIMENSIONS)


def test_metrics_for_trades() -> None:
    trades = [
        FilteredTradeRecord(
            setup_type=SetupType.CONTINUATION_BOS.value,
            direction="bullish",
            timeframe="5M",
            trigger_bar=1,
            trigger_timestamp="2026-01-06 09:15:00+05:30",
            entry_hit=True,
            outcome="Win",
            realized_pnl_points=10.0,
            realized_rr=1.0,
            filters=FilterState(
                ema_alignment="EMA20 > EMA50 > EMA200",
                vwap_position="Above VWAP",
                rsi_band="50-60",
                session="Opening",
                atr_percentile="Mid (34-66)",
                volume_spike="No",
            ),
        )
    ]
    engine = FilterResearchEngine(timeframes=("5M",))
    metrics = engine._metrics_for_trades("test", trades, baseline_expectancy=2.0)
    assert metrics.trades == 1
    assert metrics.expectancy == 10.0
    assert metrics.expectancy_improvement == 8.0


def test_run_with_pipeline_paths(tmp_path: Path) -> None:
    frame = _frame(80)
    frame.loc[10, "Sell_Liquidity_Sweep"] = 99.0
    frame.loc[12, "Bullish_BOS"] = 103.0
    csv_path = tmp_path / "NIFTY50_5m_pipeline.csv"
    frame.to_csv(csv_path, index=False)

    engine = FilterResearchEngine(timeframes=("5M",), research_days=30)
    report = engine.run(pipeline_paths={"5M": csv_path})

    assert report.total_trades >= 0
    assert len(report.baseline) == 3
    assert "combined" in report.baseline
    for setup_type in PROFITABLE_SETUPS:
        assert setup_type in report.single_filter_analysis


def test_top_20_combinations_limited_and_ranked(tmp_path: Path) -> None:
    frame = _frame(80)
    frame.loc[10, "Sell_Liquidity_Sweep"] = 99.0
    frame.loc[12, "Bullish_BOS"] = 103.0
    csv_path = tmp_path / "NIFTY50_5m_pipeline.csv"
    frame.to_csv(csv_path, index=False)

    engine = FilterResearchEngine(timeframes=("5M",), research_days=30)
    report = engine.run(pipeline_paths={"5M": csv_path})

    assert len(report.top_20_combinations) <= 20
    if report.top_20_combinations:
        ranks = [item["rank"] for item in report.top_20_combinations]
        assert ranks == list(range(1, len(ranks) + 1))


def test_report_json_export(tmp_path: Path) -> None:
    frame = _frame(80)
    csv_path = tmp_path / "NIFTY50_5m_pipeline.csv"
    report_path = tmp_path / "filter_research_report.json"
    frame.to_csv(csv_path, index=False)

    engine = FilterResearchEngine(timeframes=("5M",), research_days=30)
    report = engine.run(pipeline_paths={"5M": csv_path})
    report_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "top_20_combinations" in payload
    assert "single_filter_analysis" in payload
    assert "baseline" in payload


@pytest.mark.integration
def test_real_pipeline_filter_research_if_available() -> None:
    project_root = Path(__file__).resolve().parent.parent
    pipeline_csv = project_root / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
    if not pipeline_csv.exists():
        pytest.skip("Real 5M pipeline CSV not available.")

    engine = FilterResearchEngine(timeframes=("5M",), research_days=365)
    report = engine.run(pipeline_paths={"5M": pipeline_csv})
    assert report.total_trades > 0
    assert len(report.setups_analyzed) == 2
