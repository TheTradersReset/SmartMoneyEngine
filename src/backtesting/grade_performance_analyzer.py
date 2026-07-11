"""
SmartMoneyEngine grade performance analyzer.

Validates whether SignalQualityEngine grades predict backtest profitability
by joining quality reports with trade plan and backtest outcomes.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIGNAL_QUALITY_REPORT = PROJECT_ROOT / "outputs" / "signals" / "signal_quality_report.json"
DEFAULT_TRADE_PLAN_V2 = PROJECT_ROOT / "outputs" / "signals" / "trade_plan_report_v2.json"
DEFAULT_BACKTEST_REPORT = PROJECT_ROOT / "outputs" / "backtesting" / "backtest_report.json"
DEFAULT_TRADE_RESULTS = PROJECT_ROOT / "outputs" / "backtesting" / "trade_results.csv"
DEFAULT_V2_BACKTEST_REPORT = PROJECT_ROOT / "outputs" / "backtesting" / "v2" / "backtest_report.json"
DEFAULT_V2_TRADE_RESULTS = PROJECT_ROOT / "outputs" / "backtesting" / "v2" / "trade_results.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "backtesting" / "grade_performance_report.json"

GRADE_ORDER: tuple[str, ...] = ("A+", "A", "B", "C", "Reject")
CLOSED_OUTCOMES = frozenset({"Win", "Loss", "Breakeven"})


class GradePerformanceAnalyzerError(Exception):
    """Raised when grade performance analysis fails."""


@dataclass
class GradePerformance:
    """Backtest performance summary for one signal quality grade."""

    grade: str
    total_trades: int
    win_rate_pct: float
    average_rr: float
    profit_factor: float | None
    net_points: float
    expectancy: float
    average_confidence: float

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable grade performance dictionary."""
        payload = asdict(self)
        if payload["profit_factor"] == float("inf"):
            payload["profit_factor"] = "inf"
        return payload


@dataclass
class GradePerformanceReport:
    """Aggregate grade-vs-performance validation report."""

    symbol: str
    timeframe: str
    source_signal_quality_report: str
    source_trade_plan_report: str
    source_backtest_report: str
    source_trade_results: str
    total_matched_trades: int
    grade_performance: list[dict[str, Any]]
    best_grade: str | None
    worst_grade: str | None
    validation_passed: bool
    validation_notes: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable report dictionary."""
        return asdict(self)


class GradePerformanceAnalyzer:
    """
    Analyze backtest profitability grouped by signal quality grade.

    Parameters
    ----------
    symbol : str, optional
        Symbol label for reporting.
    timeframe : str, optional
        Timeframe label for reporting.
    """

    def __init__(self, symbol: str = "NIFTY50", timeframe: str = "5") -> None:
        self.symbol = symbol
        self.timeframe = timeframe

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise GradePerformanceAnalyzerError(f"JSON report not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _resolve_backtest_paths(
        backtest_report: Path | str | None,
        trade_results: Path | str | None,
    ) -> tuple[Path, Path]:
        report_path = (
            Path(backtest_report)
            if backtest_report is not None
            else DEFAULT_BACKTEST_REPORT
        )
        results_path = (
            Path(trade_results)
            if trade_results is not None
            else DEFAULT_TRADE_RESULTS
        )
        if not report_path.exists() and DEFAULT_V2_BACKTEST_REPORT.exists():
            report_path = DEFAULT_V2_BACKTEST_REPORT
        if not results_path.exists() and DEFAULT_V2_TRADE_RESULTS.exists():
            results_path = DEFAULT_V2_TRADE_RESULTS
        return report_path, results_path

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    @staticmethod
    def _grade_rank(grade: str) -> int:
        try:
            return GRADE_ORDER.index(grade)
        except ValueError:
            return len(GRADE_ORDER)

    def _build_matched_frame(
        self,
        quality_payload: dict[str, Any],
        trade_plan_payload: dict[str, Any],
        backtest_payload: dict[str, Any],
        trade_results_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        quality_rows = {
            int(item["signal_index"]): item
            for item in quality_payload.get("signals", [])
        }
        plan_rows = {
            int(item["signal_index"]): item
            for item in trade_plan_payload.get("trade_plans", [])
        }
        backtest_rows = {
            int(item["signal_index"]): item
            for item in backtest_payload.get("trade_results", [])
        }

        records: list[dict[str, Any]] = []
        for _, row in trade_results_frame.iterrows():
            signal_index = int(row["signal_index"])
            quality = quality_rows.get(signal_index)
            plan = plan_rows.get(signal_index)
            backtest = backtest_rows.get(signal_index)
            if quality is None or backtest is None:
                logger.warning("Skipping unmatched signal index %s", signal_index)
                continue

            records.append(
                {
                    "signal_index": signal_index,
                    "signal_date": str(row.get("signal_date", quality.get("signal_date"))),
                    "decision": str(row.get("decision", quality.get("decision"))),
                    "grade": str(quality.get("grade", "Reject")),
                    "quality_score": float(quality.get("quality_score", 0.0)),
                    "confidence_pct": float(
                        row.get("confidence_pct", plan.get("confidence", 0.0) if plan else 0.0)
                    ),
                    "outcome": str(backtest.get("outcome", row.get("outcome", ""))),
                    "entry_hit": bool(backtest.get("entry_hit", row.get("entry_hit", False))),
                    "realized_pnl_points": float(
                        backtest.get("realized_pnl_points", row.get("realized_pnl_points", 0.0))
                    ),
                    "realized_rr": float(
                        backtest.get("realized_rr", row.get("realized_rr", 0.0))
                    ),
                }
            )

        if not records:
            raise GradePerformanceAnalyzerError("No matched trades across input reports.")

        return pd.DataFrame(records)

    def _compute_grade_performance(self, frame: pd.DataFrame) -> list[GradePerformance]:
        executed = frame[frame["entry_hit"]].copy()
        closed = executed[executed["outcome"].isin(CLOSED_OUTCOMES)].copy()

        results: list[GradePerformance] = []
        for grade in GRADE_ORDER:
            bucket = closed[closed["grade"] == grade]
            if bucket.empty:
                results.append(
                    GradePerformance(
                        grade=grade,
                        total_trades=0,
                        win_rate_pct=0.0,
                        average_rr=0.0,
                        profit_factor=None,
                        net_points=0.0,
                        expectancy=0.0,
                        average_confidence=0.0,
                    )
                )
                continue

            pnls = bucket["realized_pnl_points"].tolist()
            wins = int((bucket["outcome"] == "Win").sum())
            total = len(bucket)
            win_rate = round(wins / total * 100, 2) if total else 0.0
            avg_rr = round(float(bucket["realized_rr"].mean()), 2) if total else 0.0
            avg_conf = round(float(bucket["confidence_pct"].mean()), 2) if total else 0.0

            results.append(
                GradePerformance(
                    grade=grade,
                    total_trades=total,
                    win_rate_pct=win_rate,
                    average_rr=avg_rr,
                    profit_factor=self._profit_factor(pnls),
                    net_points=round(sum(pnls), 2),
                    expectancy=round(sum(pnls) / total, 2) if total else 0.0,
                    average_confidence=avg_conf,
                )
            )
        return results

    @staticmethod
    def _metric_value(value: float | None) -> float:
        if value is None:
            return 0.0
        if value == float("inf"):
            return 9999.0
        return float(value)

    def _rank_grades(
        self,
        grade_performance: list[GradePerformance],
    ) -> tuple[str | None, str | None]:
        active = [item for item in grade_performance if item.total_trades > 0]
        if not active:
            return None, None

        ranked = sorted(
            active,
            key=lambda item: (
                item.expectancy,
                item.win_rate_pct,
                self._metric_value(item.profit_factor),
            ),
            reverse=True,
        )
        return ranked[0].grade, ranked[-1].grade

    def _validate_monotonicity(
        self,
        grade_performance: list[GradePerformance],
    ) -> tuple[bool, list[str]]:
        """Verify higher grades outperform lower grades on key metrics."""
        by_grade = {item.grade: item for item in grade_performance}
        active_grades = [grade for grade in GRADE_ORDER if by_grade[grade].total_trades > 0]
        notes: list[str] = []
        passed = True

        if len(active_grades) < 2:
            notes.append("Insufficient grade buckets for monotonicity validation.")
            return True, notes

        for higher, lower in zip(active_grades, active_grades[1:], strict=False):
            high = by_grade[higher]
            low = by_grade[lower]
            if high.win_rate_pct < low.win_rate_pct:
                passed = False
                notes.append(
                    f"Win rate inversion: {higher} ({high.win_rate_pct}%) "
                    f"< {lower} ({low.win_rate_pct}%)"
                )
            high_pf = self._metric_value(high.profit_factor)
            low_pf = self._metric_value(low.profit_factor)
            if high_pf < low_pf:
                passed = False
                notes.append(
                    f"Profit factor inversion: {higher} ({high.profit_factor}) "
                    f"< {lower} ({low.profit_factor})"
                )
            if high.expectancy < low.expectancy:
                passed = False
                notes.append(
                    f"Expectancy inversion: {higher} ({high.expectancy}) "
                    f"< {lower} ({low.expectancy})"
                )

        if passed:
            notes.append("Higher grades statistically outperformed lower grades.")
        return passed, notes

    def analyze(
        self,
        signal_quality_report: Path | str | None = None,
        trade_plan_report: Path | str | None = None,
        backtest_report: Path | str | None = None,
        trade_results: Path | str | None = None,
    ) -> GradePerformanceReport:
        """
        Join reports and compute per-grade performance metrics.

        Returns
        -------
        GradePerformanceReport
            Aggregate grade performance validation report.
        """
        started = time.perf_counter()

        quality_path = (
            Path(signal_quality_report)
            if signal_quality_report is not None
            else DEFAULT_SIGNAL_QUALITY_REPORT
        )
        plan_path = (
            Path(trade_plan_report)
            if trade_plan_report is not None
            else DEFAULT_TRADE_PLAN_V2
        )
        backtest_path, results_path = self._resolve_backtest_paths(
            backtest_report,
            trade_results,
        )

        quality_payload = self._load_json(quality_path)
        plan_payload = self._load_json(plan_path)
        backtest_payload = self._load_json(backtest_path)

        if not results_path.exists():
            raise GradePerformanceAnalyzerError(f"Trade results CSV not found: {results_path}")
        trade_results_frame = pd.read_csv(results_path)

        matched = self._build_matched_frame(
            quality_payload,
            plan_payload,
            backtest_payload,
            trade_results_frame,
        )
        grade_performance = self._compute_grade_performance(matched)
        best_grade, worst_grade = self._rank_grades(grade_performance)
        validation_passed, validation_notes = self._validate_monotonicity(grade_performance)

        elapsed = time.perf_counter() - started
        return GradePerformanceReport(
            symbol=str(quality_payload.get("symbol", self.symbol)),
            timeframe=str(quality_payload.get("timeframe", self.timeframe)),
            source_signal_quality_report=str(quality_path),
            source_trade_plan_report=str(plan_path),
            source_backtest_report=str(backtest_path),
            source_trade_results=str(results_path),
            total_matched_trades=int(matched["entry_hit"].sum()),
            grade_performance=[item.as_dict() for item in grade_performance],
            best_grade=best_grade,
            worst_grade=worst_grade,
            validation_passed=validation_passed,
            validation_notes=validation_notes,
            execution_time_seconds=elapsed,
        )


def generate_grade_performance_report(
    signal_quality_report: Path | str | None = None,
    trade_plan_report: Path | str | None = None,
    backtest_report: Path | str | None = None,
    trade_results: Path | str | None = None,
    report_path: Path | str | None = None,
) -> GradePerformanceReport:
    """Run grade performance analysis and export JSON report."""
    analyzer = GradePerformanceAnalyzer()
    report = analyzer.analyze(
        signal_quality_report=signal_quality_report,
        trade_plan_report=trade_plan_report,
        backtest_report=backtest_report,
        trade_results=trade_results,
    )

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info(
        "Grade performance analysis completed: matched=%s validation=%s best=%s worst=%s",
        report.total_matched_trades,
        report.validation_passed,
        report.best_grade,
        report.worst_grade,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_grade_performance_report()
        print("Grade Performance Analyzer Summary")
        print(f"Matched Trades: {report.total_matched_trades}")
        print(f"Best Grade: {report.best_grade}")
        print(f"Worst Grade: {report.worst_grade}")
        print(f"Validation Passed: {report.validation_passed}")
        print("Per-Grade Performance:")
        for item in report.grade_performance:
            if item["total_trades"] == 0:
                continue
            pf = item["profit_factor"]
            pf_display = "inf" if pf == "inf" else (pf if pf is not None else "N/A")
            print(
                f"  {item['grade']}: trades={item['total_trades']} "
                f"win_rate={item['win_rate_pct']}% pf={pf_display} "
                f"expectancy={item['expectancy']} net={item['net_points']}"
            )
        for note in report.validation_notes:
            print(f"  - {note}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except GradePerformanceAnalyzerError as exc:
        logger.error("Grade performance analyzer error: %s", exc)
        print(f"Grade performance analyzer error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected grade performance analyzer failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
