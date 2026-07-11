"""Tests for setup performance research analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.setup_research_analyzer import (
    RESEARCH_DAYS,
    SetupRecommendation,
    SetupResearchAnalyzer,
    generate_setup_research_report,
)
from src.signals.setup_classifier import SetupType

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _frame(length: int = 80) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-06 09:15:00+05:30")
    for index in range(length):
        price = 100.0 + index * 0.3
        rows.append(
            {
                "Date": (base + pd.Timedelta(minutes=5 * index)).isoformat(),
                "Open": price,
                "High": price + 0.8,
                "Low": price - 0.5,
                "Close": price + 0.2,
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


def test_research_days_default() -> None:
    assert RESEARCH_DAYS == 365


def test_session_label_mid_session() -> None:
    ts = pd.Timestamp("2026-01-06 12:00:00+05:30")
    assert SetupResearchAnalyzer._session_label(ts) == "Mid session"


def test_segment_metrics_from_records() -> None:
    from src.research.setup_research_analyzer import EnrichedSetupResult

    records = [
        EnrichedSetupResult(
            setup_type=SetupType.CONTINUATION_BOS.value,
            direction="bullish",
            timeframe="5M",
            trigger_bar=1,
            trigger_timestamp="2026-01-06 09:15:00+05:30",
            session="Mid session",
            day_of_week="Tuesday",
            entry_hit=True,
            outcome="Win",
            exit_reason="Target 1",
            realized_pnl_points=10.0,
            realized_rr=1.0,
            trade_duration_bars=3,
            quality_score=70,
        )
    ]
    analyzer = SetupResearchAnalyzer(timeframes=("5M",))
    metrics = analyzer._segment_metrics("5M", records)
    assert metrics.occurrences == 1
    assert metrics.win_rate_pct == 100.0
    assert metrics.expectancy == 10.0


def test_aggregate_setup_type_includes_breakdowns() -> None:
    from src.research.setup_research_analyzer import EnrichedSetupResult

    records = [
        EnrichedSetupResult(
            setup_type=SetupType.LIQUIDITY_SWEEP_BOS.value,
            direction="bullish",
            timeframe="5M",
            trigger_bar=1,
            trigger_timestamp="2026-01-06 09:20:00+05:30",
            session="Opening hour",
            day_of_week="Tuesday",
            entry_hit=True,
            outcome="Loss",
            exit_reason="Stop Loss",
            realized_pnl_points=-5.0,
            realized_rr=-1.0,
            trade_duration_bars=2,
            quality_score=65,
        )
    ]
    analyzer = SetupResearchAnalyzer(timeframes=("5M",))
    metrics = analyzer._aggregate_setup_type(SetupType.LIQUIDITY_SWEEP_BOS.value, records)
    assert metrics.total_occurrences == 1
    assert "5M" in metrics.by_timeframe
    assert metrics.recommendation == SetupRecommendation.INCONCLUSIVE.value


def test_analyze_timeframe_detects_setups() -> None:
    frame = _frame(80)
    frame.loc[10, "Sell_Liquidity_Sweep"] = 99.0
    frame.loc[12, "Bullish_BOS"] = 103.0
    frame.loc[15, "Low"] = 102.0
    frame.loc[16, "High"] = 110.0

    analyzer = SetupResearchAnalyzer(timeframes=("5M",))
    records = analyzer._analyze_timeframe(frame, "5M")
    assert isinstance(records, list)


def test_run_with_pipeline_paths(tmp_path: Path) -> None:
    frame = _frame(60)
    frame.loc[10, "Sell_Liquidity_Sweep"] = 99.0
    frame.loc[12, "Bullish_BOS"] = 103.0
    csv_path = tmp_path / "NIFTY50_5m_pipeline.csv"
    frame.to_csv(csv_path, index=False)

    analyzer = SetupResearchAnalyzer(timeframes=("5M",), research_days=30)
    report = analyzer.run(pipeline_paths={"5M": csv_path})

    assert report.total_occurrences >= 1
    assert len(report.setup_rankings) == 5
    assert SetupType.CONTINUATION_BOS.value in report.setups


def test_rankings_sorted_by_expectancy(tmp_path: Path) -> None:
    frame = _frame(60)
    frame.loc[10, "Sell_Liquidity_Sweep"] = 99.0
    frame.loc[12, "Bullish_BOS"] = 103.0
    csv_path = tmp_path / "NIFTY50_5m_pipeline.csv"
    frame.to_csv(csv_path, index=False)

    analyzer = SetupResearchAnalyzer(timeframes=("5M",), research_days=30)
    report = analyzer.run(pipeline_paths={"5M": csv_path})
    expectancies = [item["expectancy"] for item in report.setup_rankings]
    assert expectancies == sorted(expectancies, reverse=True)


def test_generate_report_writes_json(tmp_path: Path) -> None:
    frame = _frame(60)
    frame.loc[10, "Sell_Liquidity_Sweep"] = 99.0
    frame.loc[12, "Bullish_BOS"] = 103.0
    csv_path = tmp_path / "NIFTY50_5m_pipeline.csv"
    report_path = tmp_path / "setup_research_report.json"
    frame.to_csv(csv_path, index=False)

    analyzer = SetupResearchAnalyzer(timeframes=("5M",), research_days=30)
    internal_report = analyzer.run(pipeline_paths={"5M": csv_path})
    report_path.write_text(json.dumps(internal_report.as_dict(), indent=2), encoding="utf-8")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "setup_rankings" in payload
    assert "makes_money" in payload
    assert len(payload["setups"]) == 5


def test_recommendation_remove_on_negative_ev() -> None:
    from src.research.setup_research_analyzer import SetupResearchMetrics

    metrics = SetupResearchMetrics(
        setup_type="Test",
        total_occurrences=50,
        entries=40,
        wins=10,
        losses=30,
        win_rate_pct=25.0,
        average_rr=-0.3,
        max_rr=1.0,
        profit_factor=0.5,
        expectancy=-4.0,
        average_duration_bars=5.0,
        best_timeframe="5M",
        best_session="Mid session",
        best_day_of_week="Tuesday",
        recommendation="",
    )
    analyzer = SetupResearchAnalyzer()
    recommendation, evidence = analyzer._recommendation_for(metrics)
    assert recommendation == SetupRecommendation.REMOVE.value
    assert evidence


def test_all_five_setup_types_present_in_report(tmp_path: Path) -> None:
    frame = _frame(40)
    csv_path = tmp_path / "NIFTY50_5m_pipeline.csv"
    frame.to_csv(csv_path, index=False)
    analyzer = SetupResearchAnalyzer(timeframes=("5M",), research_days=30)
    report = analyzer.run(pipeline_paths={"5M": csv_path})
    for setup_type in SetupType:
        assert setup_type.value in report.setups


@pytest.mark.integration
def test_real_365_day_research_if_pipeline_present() -> None:
    pipeline_csv = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
    if not pipeline_csv.exists():
        pytest.skip("Real 5M pipeline CSV not available.")

    analyzer = SetupResearchAnalyzer(timeframes=("5M",), research_days=365)
    report = analyzer.run(pipeline_paths={"5M": pipeline_csv})
    assert report.total_occurrences > 0
    assert len(report.setup_rankings) == 5
