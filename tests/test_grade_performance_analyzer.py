"""Tests for SmartMoneyEngine grade performance analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.backtesting.grade_performance_analyzer import (
    GradePerformanceAnalyzer,
    GradePerformanceAnalyzerError,
    generate_grade_performance_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGNAL_QUALITY_REPORT = PROJECT_ROOT / "outputs" / "signals" / "signal_quality_report.json"
TRADE_PLAN_V2 = PROJECT_ROOT / "outputs" / "signals" / "trade_plan_report_v2.json"
V2_BACKTEST_REPORT = PROJECT_ROOT / "outputs" / "backtesting" / "v2" / "backtest_report.json"
V2_TRADE_RESULTS = PROJECT_ROOT / "outputs" / "backtesting" / "v2" / "trade_results.csv"
GRADE_REPORT = PROJECT_ROOT / "outputs" / "backtesting" / "grade_performance_report.json"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _quality(signal_index: int, grade: str, score: float = 80.0) -> dict:
    return {
        "signal_index": signal_index,
        "signal_date": f"2026-01-0{signal_index} 09:15:00+05:30",
        "decision": "BUY",
        "quality_score": score,
        "grade": grade,
        "factors": {},
        "reasoning": [],
    }


def _plan(signal_index: int) -> dict:
    return {
        "signal_index": signal_index,
        "signal_date": f"2026-01-0{signal_index} 09:15:00+05:30",
        "decision": "BUY",
        "confidence": 60.0,
        "trade_validity": True,
        "risk_reward_t2": 2.0,
    }


def _backtest_result(
    signal_index: int,
    outcome: str,
    pnl: float,
    rr: float,
    confidence: float = 60.0,
) -> dict:
    return {
        "signal_index": signal_index,
        "signal_date": f"2026-01-0{signal_index} 09:15:00+05:30",
        "decision": "BUY",
        "confidence_pct": confidence,
        "entry_hit": True,
        "outcome": outcome,
        "realized_pnl_points": pnl,
        "realized_rr": rr,
    }


def _trade_csv_row(signal_index: int, outcome: str, pnl: float, rr: float) -> dict:
    return {
        "signal_index": signal_index,
        "signal_date": f"2026-01-0{signal_index} 09:15:00+05:30",
        "decision": "BUY",
        "confidence_pct": 60.0,
        "trade_validity": True,
        "entry_hit": True,
        "outcome": outcome,
        "realized_pnl_points": pnl,
        "realized_rr": rr,
    }


@pytest.fixture
def analyzer() -> GradePerformanceAnalyzer:
    return GradePerformanceAnalyzer()


def test_monotonic_grades_pass_validation(analyzer: GradePerformanceAnalyzer, tmp_path: Path) -> None:
    quality_path = tmp_path / "signal_quality_report.json"
    plan_path = tmp_path / "trade_plan_report_v2.json"
    backtest_path = tmp_path / "backtest_report.json"
    results_path = tmp_path / "trade_results.csv"

    _write_json(
        quality_path,
        {
            "symbol": "NIFTY50",
            "timeframe": "5",
            "signals": [
                _quality(1, "A+", 90.0),
                _quality(2, "A", 75.0),
                _quality(3, "B", 60.0),
                _quality(4, "C", 45.0),
            ],
        },
    )
    _write_json(
        plan_path,
        {"trade_plans": [_plan(i) for i in range(1, 5)]},
    )
    _write_json(
        backtest_path,
        {
            "trade_results": [
                _backtest_result(1, "Win", 100.0, 2.0),
                _backtest_result(2, "Win", 50.0, 1.5),
                _backtest_result(3, "Win", 20.0, 1.0),
                _backtest_result(4, "Loss", -10.0, -1.0),
            ]
        },
    )
    pd.DataFrame(
        [
            _trade_csv_row(1, "Win", 100.0, 2.0),
            _trade_csv_row(2, "Win", 50.0, 1.5),
            _trade_csv_row(3, "Win", 20.0, 1.0),
            _trade_csv_row(4, "Loss", -10.0, -1.0),
        ]
    ).to_csv(results_path, index=False)

    report = analyzer.analyze(
        signal_quality_report=quality_path,
        trade_plan_report=plan_path,
        backtest_report=backtest_path,
        trade_results=results_path,
    )

    assert report.validation_passed is True
    assert report.best_grade == "A+"
    assert report.worst_grade == "C"

    by_grade = {item["grade"]: item for item in report.grade_performance}
    assert by_grade["A+"]["win_rate_pct"] == 100.0
    assert by_grade["C"]["win_rate_pct"] == 0.0


def test_monotonic_grades_fail_validation(analyzer: GradePerformanceAnalyzer, tmp_path: Path) -> None:
    quality_path = tmp_path / "signal_quality_report.json"
    plan_path = tmp_path / "trade_plan_report_v2.json"
    backtest_path = tmp_path / "backtest_report.json"
    results_path = tmp_path / "trade_results.csv"

    _write_json(
        quality_path,
        {
            "symbol": "NIFTY50",
            "timeframe": "5",
            "signals": [_quality(1, "A", 75.0), _quality(2, "C", 45.0)],
        },
    )
    _write_json(plan_path, {"trade_plans": [_plan(1), _plan(2)]})
    _write_json(
        backtest_path,
        {
            "trade_results": [
                _backtest_result(1, "Loss", -20.0, -1.0),
                _backtest_result(2, "Win", 80.0, 2.0),
            ]
        },
    )
    pd.DataFrame(
        [
            _trade_csv_row(1, "Loss", -20.0, -1.0),
            _trade_csv_row(2, "Win", 80.0, 2.0),
        ]
    ).to_csv(results_path, index=False)

    report = analyzer.analyze(
        signal_quality_report=quality_path,
        trade_plan_report=plan_path,
        backtest_report=backtest_path,
        trade_results=results_path,
    )

    assert report.validation_passed is False
    assert any("inversion" in note.lower() for note in report.validation_notes)


def test_per_grade_metrics_present(analyzer: GradePerformanceAnalyzer, tmp_path: Path) -> None:
    quality_path = tmp_path / "signal_quality_report.json"
    plan_path = tmp_path / "trade_plan_report_v2.json"
    backtest_path = tmp_path / "backtest_report.json"
    results_path = tmp_path / "trade_results.csv"

    _write_json(
        quality_path,
        {"symbol": "NIFTY50", "timeframe": "5", "signals": [_quality(1, "B", 58.0)]},
    )
    _write_json(plan_path, {"trade_plans": [_plan(1)]})
    _write_json(
        backtest_path,
        {"trade_results": [_backtest_result(1, "Win", 57.0, 1.0, confidence=64.0)]},
    )
    pd.DataFrame(
        [{"signal_index": 1, "signal_date": "2026-01-01 09:15:00+05:30", "decision": "BUY",
          "confidence_pct": 64.0, "trade_validity": True, "entry_hit": True,
          "outcome": "Win", "realized_pnl_points": 57.0, "realized_rr": 1.0}]
    ).to_csv(results_path, index=False)

    report = analyzer.analyze(
        signal_quality_report=quality_path,
        trade_plan_report=plan_path,
        backtest_report=backtest_path,
        trade_results=results_path,
    )

    bucket = next(item for item in report.grade_performance if item["grade"] == "B")
    for field in (
        "total_trades",
        "win_rate_pct",
        "average_rr",
        "profit_factor",
        "net_points",
        "expectancy",
        "average_confidence",
    ):
        assert field in bucket
    assert bucket["total_trades"] == 1
    assert bucket["average_confidence"] == 64.0


def test_missing_report_raises_error(analyzer: GradePerformanceAnalyzer, tmp_path: Path) -> None:
    with pytest.raises(GradePerformanceAnalyzerError):
        analyzer.analyze(
            signal_quality_report=tmp_path / "missing.json",
            trade_plan_report=tmp_path / "missing_plan.json",
            backtest_report=tmp_path / "missing_backtest.json",
            trade_results=tmp_path / "missing.csv",
        )


@pytest.mark.skipif(
    not SIGNAL_QUALITY_REPORT.exists()
    or not TRADE_PLAN_V2.exists()
    or not V2_BACKTEST_REPORT.exists()
    or not V2_TRADE_RESULTS.exists(),
    reason="Real NIFTY quality/backtest artifacts required.",
)
def test_real_data_grade_performance_report() -> None:
    report = generate_grade_performance_report(
        backtest_report=V2_BACKTEST_REPORT,
        trade_results=V2_TRADE_RESULTS,
        report_path=GRADE_REPORT,
    )

    assert GRADE_REPORT.exists()
    assert report.total_matched_trades >= 1
    assert report.best_grade is not None
    assert report.worst_grade is not None
    assert len(report.grade_performance) == 5

    with GRADE_REPORT.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert "validation_passed" in payload
    assert "validation_notes" in payload
    active = [item for item in payload["grade_performance"] if item["total_trades"] > 0]
    assert len(active) >= 1
