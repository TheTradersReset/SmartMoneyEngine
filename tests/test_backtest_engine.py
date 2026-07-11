"""Tests for the SmartMoneyEngine backtesting engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.backtesting.backtest_engine import (
    BacktestEngine,
    BacktestEngineError,
    run_backtest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRADE_PLAN_REPORT = PROJECT_ROOT / "outputs" / "signals" / "trade_plan_report.json"
BACKTEST_REPORT = PROJECT_ROOT / "outputs" / "backtesting" / "backtest_report.json"
TRADE_RESULTS_CSV = PROJECT_ROOT / "outputs" / "backtesting" / "trade_results.csv"

pytestmark = pytest.mark.skipif(
    not TRADE_PLAN_REPORT.exists(),
    reason="Real FYERS trade plan report is required for backtest tests.",
)


@pytest.fixture(scope="module")
def backtest_report() -> dict:
    run_backtest()
    with BACKTEST_REPORT.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_exports_created() -> None:
    run_backtest()
    assert BACKTEST_REPORT.exists()
    assert TRADE_RESULTS_CSV.exists()


def test_report_summary_fields(backtest_report: dict) -> None:
    required = {
        "total_trade_plans",
        "total_trades",
        "win_rate_pct",
        "average_rr",
        "profit_factor",
        "maximum_drawdown_points",
        "trade_results",
    }
    for field in required:
        assert field in backtest_report


def test_trade_result_fields(backtest_report: dict) -> None:
    required = {
        "entry_hit",
        "sl_hit",
        "target_1_hit",
        "target_2_hit",
        "trade_duration_bars",
        "mfe_points",
        "mae_points",
        "realized_rr",
        "outcome",
    }
    for trade in backtest_report["trade_results"]:
        for field in required:
            assert field in trade


def test_trade_counts_consistent(backtest_report: dict) -> None:
    results = backtest_report["trade_results"]
    assert backtest_report["total_trade_plans"] == len(results)
    assert backtest_report["total_trades"] == sum(1 for item in results if item["entry_hit"])


def test_missing_report_raises_error(tmp_path: Path) -> None:
    engine = BacktestEngine()
    with pytest.raises(BacktestEngineError):
        engine.run(trade_plan_report=tmp_path / "missing.json")
