"""Tests for SmartMoneyEngine batch backtester."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.backtesting.batch_backtester import (
    MIN_ANALYSIS_DAYS,
    BatchBacktester,
    BatchBacktesterError,
    BatchRunMetrics,
    generate_batch_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_analysis_days_minimum() -> None:
    with pytest.raises(BatchBacktesterError):
        BatchBacktester(analysis_days=30)


def test_artifact_paths() -> None:
    backtester = BatchBacktester(analysis_days=90)
    paths = backtester.artifact_paths("NIFTY50", "5M")
    assert paths["pipeline_csv"].name == "NIFTY50_5m_pipeline.csv"
    assert "batch" in str(paths["run_dir"])


def test_compute_metrics_from_backtest() -> None:
    backtester = BatchBacktester(analysis_days=90)
    payload = {
        "trade_results": [
            {"entry_hit": True, "outcome": "Win", "realized_pnl_points": 50.0, "realized_rr": 2.0},
            {"entry_hit": True, "outcome": "Loss", "realized_pnl_points": -20.0, "realized_rr": -1.0},
        ]
    }
    metrics = backtester._compute_metrics_from_backtest(payload)
    assert metrics["total_trades"] == 2
    assert metrics["win_rate_pct"] == 50.0
    assert metrics["net_points"] == 30.0
    assert metrics["expectancy"] == 15.0
    assert metrics["profit_factor"] == 2.5


def test_compute_grade_statistics() -> None:
    backtester = BatchBacktester(analysis_days=90)
    quality = {
        "signals": [
            {"signal_index": 1, "grade": "A"},
            {"signal_index": 2, "grade": "B"},
        ]
    }
    backtest = {
        "trade_results": [
            {"signal_index": 1, "entry_hit": True, "outcome": "Win", "realized_pnl_points": 40.0},
            {"signal_index": 2, "entry_hit": True, "outcome": "Loss", "realized_pnl_points": -10.0},
        ]
    }
    stats = backtester._compute_grade_statistics(quality, backtest)
    assert stats["A"]["trades"] == 1
    assert stats["A"]["win_rate_pct"] == 100.0
    assert stats["B"]["trades"] == 1
    assert stats["B"]["win_rate_pct"] == 0.0


def test_ranking_and_best_selection() -> None:
    backtester = BatchBacktester(analysis_days=90)
    runs = [
        BatchRunMetrics(
            symbol="NIFTY50",
            timeframe="5M",
            start_date="2026-01-01",
            end_date="2026-04-01",
            analysis_days=90,
            success=True,
            total_trades=3,
            win_rate_pct=66.0,
            average_rr=1.0,
            profit_factor=2.0,
            expectancy=20.0,
            maximum_drawdown_points=10.0,
            net_points=60.0,
            grade_statistics={},
            pipeline_rows=1000,
            buy_signals=2,
            sell_signals=1,
            valid_trade_plans=3,
            average_quality_score=70.0,
        ),
        BatchRunMetrics(
            symbol="BANKNIFTY",
            timeframe="15M",
            start_date="2026-01-01",
            end_date="2026-04-01",
            analysis_days=90,
            success=True,
            total_trades=2,
            win_rate_pct=100.0,
            average_rr=1.5,
            profit_factor=3.0,
            expectancy=35.0,
            maximum_drawdown_points=5.0,
            net_points=70.0,
            grade_statistics={},
            pipeline_rows=800,
            buy_signals=1,
            sell_signals=1,
            valid_trade_plans=2,
            average_quality_score=75.0,
        ),
    ]
    rankings = backtester._rank_runs(runs)
    best_symbol, best_timeframe = backtester._best_symbol_and_timeframe(runs)

    assert rankings["by_expectancy"][0]["symbol"] == "BANKNIFTY"
    assert best_symbol == "BANKNIFTY"
    assert best_timeframe == "15M"


@patch.object(BatchBacktester, "ensure_data")
@patch("src.backtesting.batch_backtester.MarketPipelineRunner")
@patch("src.backtesting.batch_backtester.evaluate_pipeline")
@patch("src.backtesting.batch_backtester.generate_multi_timeframe_report")
@patch("src.backtesting.batch_backtester.generate_trade_plans_v2")
@patch("src.backtesting.batch_backtester.generate_signal_quality_report")
@patch("src.backtesting.batch_backtester.BacktestEngine")
def test_run_single_orchestration(
    mock_backtest_engine: MagicMock,
    mock_quality: MagicMock,
    mock_trade_plan: MagicMock,
    mock_mtf: MagicMock,
    mock_evaluate: MagicMock,
    mock_pipeline: MagicMock,
    mock_ensure: MagicMock,
) -> None:
    backtester = BatchBacktester(symbols=("NIFTY50",), timeframes=("5M",), analysis_days=90)

    pipeline_report = MagicMock()
    pipeline_report.success = True
    pipeline_report.rows = 1500
    pipeline_report.failure_message = None
    mock_pipeline.return_value.run.return_value = pipeline_report

    decision_report = MagicMock()
    decision_report.buy_count = 2
    decision_report.sell_count = 1
    mock_evaluate.return_value = (MagicMock(), decision_report)

    trade_plan_report = MagicMock()
    trade_plan_report.valid_trade_plans = 2
    mock_trade_plan.return_value = ([], trade_plan_report)

    quality_report = MagicMock()
    quality_report.average_score = 72.0
    mock_quality.return_value = quality_report

    backtest_report = MagicMock()
    backtest_report.as_dict.return_value = {
        "trade_results": [
            {
                "signal_index": 10,
                "entry_hit": True,
                "outcome": "Win",
                "realized_pnl_points": 25.0,
                "realized_rr": 1.5,
            }
        ]
    }
    backtest_report.trade_results = backtest_report.as_dict.return_value["trade_results"]
    mock_backtest_engine.return_value.run.return_value = backtest_report

    paths = backtester.artifact_paths("NIFTY50", "5M")
    for value in paths.values():
        if isinstance(value, Path):
            value.parent.mkdir(parents=True, exist_ok=True)
    paths["signal_quality"].write_text(
        json.dumps(
            {
                "signals": [{"signal_index": 10, "grade": "A", "quality_score": 80.0}],
            }
        ),
        encoding="utf-8",
    )

    with patch.object(backtester, "artifact_paths", return_value=paths):
        result = backtester.run_single("NIFTY50", "5M", end_date=date(2026, 4, 1))

    assert result.success is True
    assert result.total_trades == 1
    assert result.win_rate_pct == 100.0
    assert result.expectancy == 25.0
    mock_ensure.assert_called_once()
    mock_pipeline.assert_called_once()
    mock_evaluate.assert_called_once()
    mock_mtf.assert_called_once()
    mock_trade_plan.assert_called_once()
    mock_quality.assert_called_once()
    mock_backtest_engine.assert_called_once()


@patch.object(BatchBacktester, "run_single")
def test_run_all_builds_master_report(mock_run_single: MagicMock) -> None:
    mock_run_single.return_value = BatchRunMetrics(
        symbol="NIFTY50",
        timeframe="5M",
        start_date="2026-01-01",
        end_date="2026-04-01",
        analysis_days=90,
        success=True,
        total_trades=1,
        win_rate_pct=100.0,
        average_rr=1.0,
        profit_factor=2.0,
        expectancy=10.0,
        maximum_drawdown_points=0.0,
        net_points=10.0,
        grade_statistics={
            "A+": {"signals": 1, "trades": 1, "wins": 1, "win_rate_pct": 100.0, "net_points": 10.0, "expectancy": 10.0},
            "A": {"signals": 0, "trades": 0, "wins": 0, "win_rate_pct": 0.0, "net_points": 0.0, "expectancy": 0.0},
            "B": {"signals": 0, "trades": 0, "wins": 0, "win_rate_pct": 0.0, "net_points": 0.0, "expectancy": 0.0},
            "C": {"signals": 0, "trades": 0, "wins": 0, "win_rate_pct": 0.0, "net_points": 0.0, "expectancy": 0.0},
        },
        pipeline_rows=100,
        buy_signals=1,
        sell_signals=0,
        valid_trade_plans=1,
        average_quality_score=85.0,
    )

    backtester = BatchBacktester(
        symbols=("NIFTY50", "BANKNIFTY", "FINNIFTY"),
        timeframes=("5M",),
        analysis_days=90,
    )
    report = backtester.run_all(end_date=date(2026, 4, 1))

    assert report.total_runs == 3
    assert report.analysis_days == MIN_ANALYSIS_DAYS
    assert report.best_symbol == "NIFTY50"
    assert "grade_statistics_aggregate" in report.as_dict()


@patch.object(BatchBacktester, "run_single")
def test_generate_batch_report_export(mock_run_single: MagicMock, tmp_path: Path) -> None:
    mock_run_single.return_value = BatchRunMetrics(
        symbol="NIFTY50",
        timeframe="5M",
        start_date="2026-01-01",
        end_date="2026-04-01",
        analysis_days=90,
        success=True,
        total_trades=5,
        win_rate_pct=80.0,
        average_rr=0.87,
        profit_factor=7.84,
        expectancy=48.21,
        maximum_drawdown_points=35.22,
        net_points=241.06,
        grade_statistics={
            "A+": {"signals": 0, "trades": 0, "wins": 0, "win_rate_pct": 0.0, "net_points": 0.0, "expectancy": 0.0},
            "A": {"signals": 3, "trades": 3, "wins": 2, "win_rate_pct": 66.67, "net_points": 85.48, "expectancy": 28.49},
            "B": {"signals": 1, "trades": 1, "wins": 1, "win_rate_pct": 100.0, "net_points": 57.08, "expectancy": 57.08},
            "C": {"signals": 1, "trades": 1, "wins": 1, "win_rate_pct": 100.0, "net_points": 98.5, "expectancy": 98.5},
        },
        pipeline_rows=1617,
        buy_signals=3,
        sell_signals=2,
        valid_trade_plans=5,
        average_quality_score=70.2,
    )

    report_path = tmp_path / "batch_report.json"
    report = generate_batch_report(
        symbols=("NIFTY50",),
        timeframes=("5M",),
        analysis_days=90,
        auto_download=False,
        report_path=report_path,
    )

    assert report_path.exists()
    assert report.total_runs == 1
    assert report.successful_runs == 1
    assert report.best_symbol == "NIFTY50"
