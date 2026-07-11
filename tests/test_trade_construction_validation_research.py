"""Tests for trade construction validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.trade_construction_validation_research import (
    TradeConstructionValidationError,
    TradeConstructionValidationResearch,
    generate_trade_construction_validation_report,
)
from src.research.tiered_signal_framework_research import TierSignal


def _sample_frame(rows: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02 09:15", periods=rows, freq="5min", tz="Asia/Kolkata")
    close = [100.0 + index * 0.5 for index in range(rows)]
    return pd.DataFrame(
        {
            "Date": dates.astype(str),
            "Open": close,
            "High": [value + 1 for value in close],
            "Low": [value - 1 for value in close],
            "Close": close,
            "Volume": [1000] * rows,
            "Swing_High": [pd.NA] * rows,
            "Swing_Low": [pd.NA] * rows,
            "Buy_Side_Liquidity": [pd.NA] * rows,
            "Sell_Side_Liquidity": [pd.NA] * rows,
            "Sell_Liquidity_Sweep": [pd.NA] * rows,
            "Buy_Liquidity_Sweep": [pd.NA] * rows,
        }
    )


def test_structural_stop_bullish() -> None:
    engine = TradeConstructionValidationResearch()
    frame = _sample_frame()
    stop, risk = engine._structural_stop(frame, 10, 105.0, "bullish")
    assert stop < 105.0
    assert risk >= 1.0


def test_fixed_r_target() -> None:
    target = TradeConstructionValidationResearch._fixed_r_target(100.0, 10.0, "bullish", 2.0)
    assert target == 120.0


def test_simulate_trade_target_hit() -> None:
    engine = TradeConstructionValidationResearch()
    frame = _sample_frame()
    frame.loc[15, "High"] = 130.0
    pnl, rr, win, holding, stop_hit, target_hit = engine._simulate_trade(
        frame,
        entry_bar=10,
        entry_price=105.0,
        direction="bullish",
        stop=95.0,
        target=125.0,
        risk=10.0,
        timeframe="5M",
    )
    assert target_hit
    assert win
    assert pnl == 20.0


def test_empty_combination_metrics() -> None:
    engine = TradeConstructionValidationResearch()
    metrics = engine._metrics_for_outcomes("A_bos_close", "A_structural_swing", "A_1r", [])
    assert metrics.trades == 0


def test_missing_metadata_raises() -> None:
    with pytest.raises(TradeConstructionValidationError):
        generate_trade_construction_validation_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = TradeConstructionValidationResearch(timeframes=("5M",)).run(metadata)
    assert report.combinations_evaluated == 15
    assert report.total_tier2_signals >= 0
