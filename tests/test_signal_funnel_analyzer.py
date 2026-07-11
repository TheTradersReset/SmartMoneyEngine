"""Tests for signal funnel diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.research.signal_funnel_analyzer import (
    STAGE_ORDER,
    SignalFunnelAnalyzer,
    SignalFunnelError,
    TrendPath,
    generate_signal_funnel_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _base_row(
    trend: str = "BULLISH",
    bos: float | None = None,
    choch: float | None = None,
    fvg: float | None = None,
    liquidity: float | None = None,
) -> dict:
    return {
        "Date": "2026-01-02 09:15:00+05:30",
        "Open": 100.0,
        "High": 101.0,
        "Low": 99.0,
        "Close": 100.5,
        "Volume": 1000,
        "Trend": trend,
        "Trend_Strength": 2,
        "Bullish_BOS": bos,
        "Bearish_BOS": None,
        "Bullish_CHOCH": choch,
        "Bearish_CHOCH": None,
        "Bullish_FVG_Top": fvg,
        "Bearish_FVG_Top": None,
        "Bullish_OB_High": None,
        "Bearish_OB_High": None,
        "Bullish_OB_Mitigated": None,
        "Bearish_OB_Mitigated": None,
        "Buy_Liquidity_Sweep": None,
        "Sell_Liquidity_Sweep": liquidity,
        "Liquidity_Strength": 2,
    }


def _synthetic_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_stage_flags_cumulative_progression() -> None:
    analyzer = SignalFunnelAnalyzer()
    row = pd.Series(_base_row(bos=101.0, choch=101.0, fvg=101.5, liquidity=99.0))
    flags = analyzer._stage_flags(
        row=row,
        path=TrendPath.BULLISH,
        htf_1d="BULLISH",
        htf_4h="BULLISH",
        decision="BUY",
    )

    assert flags["cumulative"]["trend_qualified"] is True
    assert flags["cumulative"]["bos_qualified"] is True
    assert flags["cumulative"]["choch_qualified"] is True
    assert flags["cumulative"]["fvg_qualified"] is True
    assert flags["cumulative"]["liquidity_qualified"] is True
    assert flags["cumulative"]["htf_aligned"] is True
    assert flags["cumulative"]["decision_buy_sell"] is True


def test_stage_flags_fail_on_neutral_trend() -> None:
    analyzer = SignalFunnelAnalyzer()
    row = pd.Series(_base_row(trend="SIDEWAYS", bos=101.0))
    flags = analyzer._stage_flags(
        row=row,
        path=TrendPath.NEUTRAL,
        htf_1d="BULLISH",
        htf_4h="BULLISH",
        decision="WAIT",
    )

    assert flags["cumulative"]["trend_qualified"] is False
    assert flags["cumulative"]["decision_buy_sell"] is False


def test_htf_alignment_rejects_opposed_higher_timeframe() -> None:
    analyzer = SignalFunnelAnalyzer()
    row = pd.Series(_base_row(bos=101.0, choch=101.0, fvg=101.5, liquidity=99.0))
    flags = analyzer._stage_flags(
        row=row,
        path=TrendPath.BULLISH,
        htf_1d="BEARISH",
        htf_4h="BULLISH",
        decision="BUY",
    )

    assert flags["cumulative"]["liquidity_qualified"] is True
    assert flags["cumulative"]["htf_aligned"] is False


@patch.object(SignalFunnelAnalyzer, "_build_htf_lookup")
def test_analyze_builds_monotonic_funnel(mock_htf: MagicMock) -> None:
    mock_htf.return_value = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-02 09:15:00+05:30",
                    "2026-01-02 09:20:00+05:30",
                    "2026-01-02 09:25:00+05:30",
                ],
                utc=True,
            ).tz_convert("Asia/Kolkata"),
            "HTF_1D_Trend": ["BULLISH", "BULLISH", "BEARISH"],
            "HTF_4H_Trend": ["BULLISH", "BULLISH", "BULLISH"],
        }
    )

    rows = [
        _base_row(bos=101.0, choch=101.0, fvg=101.5, liquidity=99.0),
        _base_row(bos=101.0),
        _base_row(trend="SIDEWAYS"),
    ]
    rows[1]["Date"] = "2026-01-02 09:20:00+05:30"
    rows[2]["Date"] = "2026-01-02 09:25:00+05:30"

    frame = _synthetic_frame(rows)
    analyzer = SignalFunnelAnalyzer()
    report = analyzer.analyze(frame)

    assert report.total_candles == 3
    assert report.final_signals >= 0
    assert len(report.stages) == len(STAGE_ORDER)

    pass_counts = [stage["pass_count"] for stage in report.stages]
    assert pass_counts[0] >= pass_counts[-1]
    assert report.top_bottlenecks
    assert report.most_restrictive_filters
    assert "trend_qualified" in report.signal_loss_pct_by_stage


@patch.object(SignalFunnelAnalyzer, "_build_htf_lookup")
def test_generate_signal_funnel_report_writes_json(
    mock_htf: MagicMock,
    tmp_path: Path,
) -> None:
    mock_htf.return_value = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-02 09:15:00+05:30"], utc=True).tz_convert(
                "Asia/Kolkata"
            ),
            "HTF_1D_Trend": ["BULLISH"],
            "HTF_4H_Trend": ["BULLISH"],
        }
    )

    csv_path = tmp_path / "pipeline.csv"
    report_path = tmp_path / "signal_funnel_report.json"
    frame = _synthetic_frame([_base_row(bos=101.0)])

    from src.signals.decision_engine import DecisionEngine

    evaluated = DecisionEngine().evaluate(frame)
    evaluated.to_csv(csv_path, index=False)

    report = generate_signal_funnel_report(
        pipeline_csv=csv_path,
        report_path=report_path,
        symbol="NIFTY50",
        timeframe="5",
    )

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "NIFTY50"
    assert payload["total_candles"] == 1
    assert len(payload["stages"]) == 7
    assert report.total_candles == 1


def test_analyze_rejects_empty_frame() -> None:
    analyzer = SignalFunnelAnalyzer()
    with pytest.raises(SignalFunnelError):
        analyzer.analyze(pd.DataFrame())


@pytest.mark.integration
def test_real_pipeline_funnel_if_present() -> None:
    pipeline_csv = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
    if not pipeline_csv.exists():
        pytest.skip("Real pipeline CSV not available.")

    analyzer = SignalFunnelAnalyzer(symbol="NIFTY50", timeframe="5")
    report = analyzer.run_from_csv(pipeline_csv)

    assert report.total_candles > 0
    assert report.stages[0]["pass_count"] >= report.stages[-1]["pass_count"]
    assert report.final_signals >= 0
