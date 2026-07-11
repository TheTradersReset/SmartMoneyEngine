"""Tests for Tier-2 exit optimization research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.tier2_exit_optimization_research import (
    Tier2ExitOptimizationError,
    Tier2ExitOptimizationResearch,
    generate_tier2_exit_optimization_report,
)


def _sample_frame(rows: int = 40) -> pd.DataFrame:
    close = [100.0 + index * 0.5 for index in range(rows)]
    return pd.DataFrame(
        {
            "Date": pd.date_range(
                "2026-01-02 09:15",
                periods=rows,
                freq="5min",
                tz="Asia/Kolkata",
            ).astype(str),
            "Open": close,
            "High": [value + 1 for value in close],
            "Low": [value - 1 for value in close],
            "Close": close,
            "Volume": [1000] * rows,
            "Buy_Side_Liquidity": [pd.NA] * rows,
            "Sell_Side_Liquidity": [pd.NA] * rows,
        }
    )


def test_model_a_full_exit_at_1r() -> None:
    engine = Tier2ExitOptimizationResearch()
    frame = _sample_frame()
    frame.loc[12, "High"] = 116.0
    pnl, win = engine._simulate_model_a(
        frame,
        entry_bar=10,
        entry=105.0,
        stop=95.0,
        risk=10.0,
        direction="bullish",
    )
    assert win
    assert pnl == 10.0


def test_model_b_partial_exits() -> None:
    engine = Tier2ExitOptimizationResearch()
    frame = _sample_frame()
    frame.loc[12, "High"] = 116.0
    frame.loc[14, "High"] = 126.0
    pnl, win = engine._simulate_partial_legs(
        frame,
        entry_bar=10,
        entry=105.0,
        stop=95.0,
        risk=10.0,
        direction="bullish",
        legs=[(0.5, 1.0), (0.5, 2.0)],
    )
    assert win
    assert pnl == 15.0


def test_missing_metadata_raises() -> None:
    with pytest.raises(Tier2ExitOptimizationError):
        generate_tier2_exit_optimization_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = Tier2ExitOptimizationResearch(timeframes=("5M",)).run(metadata)
    assert report.total_trades >= 0
    assert len(report.model_metrics) == 5
