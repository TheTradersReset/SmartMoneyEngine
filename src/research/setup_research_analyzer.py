"""
Setup performance research for SmartMoneyEngine.

Analyzes existing institutional setup classifications over expanded
historical data without introducing new signal or decision logic.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.pipeline.market_pipeline import MarketPipelineRunner
from src.signals.setup_classifier import (
    SetupBacktestResult,
    SetupBacktestSimulator,
    SetupClassification,
    SetupClassifier,
    SetupType,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_DIR = PROJECT_ROOT / "outputs" / "pipeline"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "setup_research_report.json"

RESEARCH_DAYS = 365
MIN_SEGMENT_SAMPLES = 5
MIN_RECOMMENDATION_SAMPLES = 30

TIMEFRAME_MAP: dict[str, str] = {
    "5M": "5",
    "15M": "15",
    "1H": "60",
}


class SetupResearchError(Exception):
    """Raised when setup research analysis fails."""


class SetupRecommendation(str, Enum):
    """Research recommendation for a setup category."""

    KEEP = "KEEP"
    REMOVE = "REMOVE"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class EnrichedSetupResult:
    """Setup classification joined with backtest and context metadata."""

    setup_type: str
    direction: str
    timeframe: str
    trigger_bar: int
    trigger_timestamp: str
    session: str
    day_of_week: str
    entry_hit: bool
    outcome: str
    exit_reason: str
    realized_pnl_points: float
    realized_rr: float
    trade_duration_bars: int
    quality_score: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentMetrics:
    """Performance metrics for a setup segment."""

    label: str
    occurrences: int
    entries: int
    wins: int
    losses: int
    win_rate_pct: float
    average_rr: float
    max_rr: float
    profit_factor: float | None
    expectancy: float
    average_duration_bars: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SetupResearchMetrics:
    """Full research metrics for one setup type."""

    setup_type: str
    total_occurrences: int
    entries: int
    wins: int
    losses: int
    win_rate_pct: float
    average_rr: float
    max_rr: float
    profit_factor: float | None
    expectancy: float
    average_duration_bars: float
    best_timeframe: str | None
    best_session: str | None
    best_day_of_week: str | None
    recommendation: str
    statistical_evidence: list[str] = field(default_factory=list)
    by_timeframe: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_session: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_day_of_week: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SetupResearchReport:
    """Aggregate setup research report."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    total_occurrences: int
    setup_rankings: list[dict[str, Any]]
    setups: dict[str, dict[str, Any]]
    makes_money: list[str]
    should_remove: list[str]
    inconclusive: list[str]
    execution_time_seconds: float
    records: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self, include_records: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_records:
            payload.pop("records", None)
        return payload


def _json_safe(value: Any) -> Any:
    """Convert non-standard numeric values for JSON export."""
    if isinstance(value, float) and (value == float("inf") or value == float("-inf")):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


class SetupResearchAnalyzer:
    """
    Research existing setup classifier performance across historical data.

    Parameters
    ----------
    symbol : str, optional
        Symbol to analyze.
    research_days : int, optional
        Calendar days of history.
    timeframes : tuple[str, ...], optional
        Timeframe labels to compare.
    """

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.classifier = SetupClassifier()
        self.simulator = SetupBacktestSimulator()

    @staticmethod
    def _session_label(timestamp: pd.Timestamp) -> str:
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Kolkata")
        else:
            ts = ts.tz_convert("Asia/Kolkata")
        minutes = ts.hour * 60 + ts.minute
        if 9 * 60 + 15 <= minutes < 10 * 60 + 30:
            return "Opening hour"
        if 10 * 60 + 30 <= minutes < 14 * 60 + 30:
            return "Mid session"
        if 14 * 60 + 30 <= minutes <= 15 * 60 + 30:
            return "Closing hour"
        return "Outside session"

    @staticmethod
    def _profit_factor(closed_pnls: list[float]) -> float | None:
        gross_profit = sum(pnl for pnl in closed_pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in closed_pnls if pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    def _pipeline_path(self, timeframe_label: str) -> Path:
        slug = f"{self.symbol}_{timeframe_label.lower()}"
        return DEFAULT_PIPELINE_DIR / f"{slug}_pipeline.csv"

    def _ensure_pipeline(
        self,
        timeframe_label: str,
        start: date,
        end: date,
    ) -> Path:
        """Load existing pipeline CSV or build it from historical data."""
        path = self._pipeline_path(timeframe_label)
        storage_tf = TIMEFRAME_MAP[timeframe_label]
        if path.exists():
            logger.info("Using existing pipeline: %s", path)
            return path

        logger.info("Building pipeline for %s/%s (%s days).", self.symbol, timeframe_label, self.research_days)
        runner = MarketPipelineRunner(
            symbol=self.symbol,
            timeframe=storage_tf,
            start_date=start,
            end_date=end,
            output_csv=path,
        )
        report = runner.run()
        if not report.success or not path.exists():
            raise SetupResearchError(
                f"Failed to build pipeline for {self.symbol}/{timeframe_label}: "
                f"{report.failure_message}"
            )
        return path

    def _analyze_timeframe(
        self,
        frame: pd.DataFrame,
        timeframe_label: str,
    ) -> list[EnrichedSetupResult]:
        working = frame.reset_index(drop=True)
        setups = self.classifier.classify(working)
        results: list[EnrichedSetupResult] = []

        for setup in setups:
            backtest = self.simulator.simulate(working, setup)
            timestamp = pd.to_datetime(setup.trigger_timestamp, errors="coerce")
            session = self._session_label(timestamp) if pd.notna(timestamp) else "Unknown"
            day_of_week = timestamp.day_name() if pd.notna(timestamp) else "Unknown"
            results.append(
                EnrichedSetupResult(
                    setup_type=setup.setup_type,
                    direction=setup.direction,
                    timeframe=timeframe_label,
                    trigger_bar=setup.trigger_bar,
                    trigger_timestamp=setup.trigger_timestamp,
                    session=session,
                    day_of_week=day_of_week,
                    entry_hit=backtest.entry_hit,
                    outcome=backtest.outcome,
                    exit_reason=backtest.exit_reason,
                    realized_pnl_points=backtest.realized_pnl_points,
                    realized_rr=backtest.realized_rr,
                    trade_duration_bars=backtest.trade_duration_bars,
                    quality_score=setup.quality_score,
                )
            )
        return results

    @staticmethod
    def _closed_records(records: list[EnrichedSetupResult]) -> list[EnrichedSetupResult]:
        return [
            record
            for record in records
            if record.entry_hit and record.outcome not in {"Open", "No Entry"}
        ]

    def _segment_metrics(
        self,
        label: str,
        records: list[EnrichedSetupResult],
    ) -> SegmentMetrics:
        entries = sum(1 for record in records if record.entry_hit)
        wins = sum(1 for record in records if record.outcome == "Win")
        losses = sum(1 for record in records if record.outcome == "Loss")
        closed = self._closed_records(records)
        pnls = [record.realized_pnl_points for record in closed]
        rrs = [record.realized_rr for record in closed]
        durations = [record.trade_duration_bars for record in records if record.entry_hit]

        return SegmentMetrics(
            label=label,
            occurrences=len(records),
            entries=entries,
            wins=wins,
            losses=losses,
            win_rate_pct=round((wins / entries) * 100, 2) if entries else 0.0,
            average_rr=round(mean(rrs), 2) if rrs else 0.0,
            max_rr=round(max(rrs), 2) if rrs else 0.0,
            profit_factor=self._profit_factor(pnls),
            expectancy=round(mean(pnls), 2) if entries else 0.0,
            average_duration_bars=round(mean(durations), 2) if durations else 0.0,
        )

    @staticmethod
    def _best_segment(segments: dict[str, SegmentMetrics]) -> str | None:
        eligible = {
            name: metrics
            for name, metrics in segments.items()
            if metrics.entries >= MIN_SEGMENT_SAMPLES
        }
        if not eligible:
            return None
        return max(eligible.items(), key=lambda item: item[1].expectancy)[0]

    def _recommendation_for(self, metrics: SetupResearchMetrics) -> tuple[str, list[str]]:
        evidence: list[str] = []
        evidence.append(f"Sample size: {metrics.total_occurrences} occurrences, {metrics.entries} entries.")

        if metrics.entries < MIN_RECOMMENDATION_SAMPLES:
            evidence.append(
                f"Insufficient entries (< {MIN_RECOMMENDATION_SAMPLES}) for high-confidence removal/keep decision."
            )
            return SetupRecommendation.INCONCLUSIVE.value, evidence

        if metrics.expectancy > 0 and (metrics.profit_factor or 0) > 1.0:
            evidence.append(
                f"Positive expectancy ({metrics.expectancy}) with profit factor {metrics.profit_factor}."
            )
            return SetupRecommendation.KEEP.value, evidence

        if metrics.expectancy <= 0 and (metrics.profit_factor or 0) < 1.0:
            evidence.append(
                f"Negative expectancy ({metrics.expectancy}) with profit factor {metrics.profit_factor}."
            )
            return SetupRecommendation.REMOVE.value, evidence

        evidence.append(
            f"Mixed evidence: expectancy={metrics.expectancy}, profit factor={metrics.profit_factor}."
        )
        return SetupRecommendation.INCONCLUSIVE.value, evidence

    def _aggregate_setup_type(
        self,
        setup_type: str,
        records: list[EnrichedSetupResult],
    ) -> SetupResearchMetrics:
        closed = self._closed_records(records)
        pnls = [record.realized_pnl_points for record in closed]
        rrs = [record.realized_rr for record in closed]
        durations = [record.trade_duration_bars for record in records if record.entry_hit]
        entries = sum(1 for record in records if record.entry_hit)
        wins = sum(1 for record in records if record.outcome == "Win")
        losses = sum(1 for record in records if record.outcome == "Loss")

        by_timeframe = {
            timeframe: self._segment_metrics(timeframe, [record for record in records if record.timeframe == timeframe])
            for timeframe in sorted({record.timeframe for record in records})
        }
        by_session = {
            session: self._segment_metrics(session, [record for record in records if record.session == session])
            for session in sorted({record.session for record in records})
        }
        by_day = {
            day: self._segment_metrics(day, [record for record in records if record.day_of_week == day])
            for day in sorted({record.day_of_week for record in records})
        }

        metrics = SetupResearchMetrics(
            setup_type=setup_type,
            total_occurrences=len(records),
            entries=entries,
            wins=wins,
            losses=losses,
            win_rate_pct=round((wins / entries) * 100, 2) if entries else 0.0,
            average_rr=round(mean(rrs), 2) if rrs else 0.0,
            max_rr=round(max(rrs), 2) if rrs else 0.0,
            profit_factor=self._profit_factor(pnls),
            expectancy=round(sum(pnls) / entries, 2) if entries else 0.0,
            average_duration_bars=round(mean(durations), 2) if durations else 0.0,
            best_timeframe=self._best_segment(by_timeframe),
            best_session=self._best_segment(by_session),
            best_day_of_week=self._best_segment(by_day),
            recommendation=SetupRecommendation.INCONCLUSIVE.value,
            by_timeframe={key: value.as_dict() for key, value in by_timeframe.items()},
            by_session={key: value.as_dict() for key, value in by_session.items()},
            by_day_of_week={key: value.as_dict() for key, value in by_day.items()},
        )
        recommendation, evidence = self._recommendation_for(metrics)
        metrics.recommendation = recommendation
        metrics.statistical_evidence = evidence
        if metrics.best_timeframe:
            tf = by_timeframe[metrics.best_timeframe]
            evidence.append(
                f"Best timeframe: {metrics.best_timeframe} (expectancy {tf.expectancy}, n={tf.entries})."
            )
        if metrics.best_session:
            session = by_session[metrics.best_session]
            evidence.append(
                f"Best session: {metrics.best_session} (expectancy {session.expectancy}, n={session.entries})."
            )
        if metrics.best_day_of_week:
            day = by_day[metrics.best_day_of_week]
            evidence.append(
                f"Best day: {metrics.best_day_of_week} (expectancy {day.expectancy}, n={day.entries})."
            )
        metrics.statistical_evidence = evidence
        return metrics

    def run(
        self,
        end_date: date | None = None,
        pipeline_paths: dict[str, Path] | None = None,
    ) -> SetupResearchReport:
        """Run setup research across configured timeframes."""
        started = time.perf_counter()
        end = end_date if end_date is not None else date.today()
        start = end - timedelta(days=self.research_days)

        all_records: list[EnrichedSetupResult] = []
        for timeframe_label in self.timeframes:
            if pipeline_paths and timeframe_label in pipeline_paths:
                path = pipeline_paths[timeframe_label]
            else:
                path = self._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path)
            logger.info(
                "Analyzing %s setups on %s (%s rows).",
                timeframe_label,
                path.name,
                len(frame),
            )
            all_records.extend(self._analyze_timeframe(frame, timeframe_label))

        grouped: dict[str, list[EnrichedSetupResult]] = defaultdict(list)
        for record in all_records:
            grouped[record.setup_type].append(record)

        setup_metrics: dict[str, SetupResearchMetrics] = {}
        for setup_type in SetupType:
            records = grouped.get(setup_type.value, [])
            setup_metrics[setup_type.value] = self._aggregate_setup_type(setup_type.value, records)

        rankings = sorted(
            [metrics.as_dict() for metrics in setup_metrics.values()],
            key=lambda item: item["expectancy"],
            reverse=True,
        )

        makes_money = [
            metrics.setup_type
            for metrics in setup_metrics.values()
            if metrics.recommendation == SetupRecommendation.KEEP.value
        ]
        should_remove = [
            metrics.setup_type
            for metrics in setup_metrics.values()
            if metrics.recommendation == SetupRecommendation.REMOVE.value
        ]
        inconclusive = [
            metrics.setup_type
            for metrics in setup_metrics.values()
            if metrics.recommendation == SetupRecommendation.INCONCLUSIVE.value
        ]

        return SetupResearchReport(
            symbol=self.symbol,
            research_window_days=self.research_days,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            timeframes_analyzed=list(self.timeframes),
            total_occurrences=len(all_records),
            setup_rankings=rankings,
            setups={key: value.as_dict() for key, value in setup_metrics.items()},
            makes_money=makes_money,
            should_remove=should_remove,
            inconclusive=inconclusive,
            execution_time_seconds=round(time.perf_counter() - started, 3),
            records=[record.as_dict() for record in all_records],
        )


def generate_setup_research_report(
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    research_days: int = RESEARCH_DAYS,
    timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
    end_date: date | None = None,
) -> SetupResearchReport:
    """Run setup research and export JSON report."""
    analyzer = SetupResearchAnalyzer(
        symbol=symbol,
        research_days=research_days,
        timeframes=timeframes,
    )
    report = analyzer.run(end_date=end_date)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Setup research completed: occurrences=%s ranked_setups=%s",
        report.total_occurrences,
        len(report.setup_rankings),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_setup_research_report()
        print("Setup Research Summary")
        print(f"Symbol: {report.symbol} | Window: {report.research_window_days} days")
        print(f"Timeframes: {', '.join(report.timeframes_analyzed)}")
        print(f"Total Occurrences: {report.total_occurrences}")
        print("Rankings by Expectancy:")
        for index, setup in enumerate(report.setup_rankings, start=1):
            print(
                f"  {index}. {setup['setup_type']}: expectancy={setup['expectancy']} "
                f"win_rate={setup['win_rate_pct']}% pf={setup['profit_factor']} "
                f"n={setup['total_occurrences']} -> {setup['recommendation']}"
            )
        print(f"Makes Money: {', '.join(report.makes_money) or 'None'}")
        print(f"Should Remove: {', '.join(report.should_remove) or 'None'}")
        print(f"Inconclusive: {', '.join(report.inconclusive) or 'None'}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except SetupResearchError as exc:
        logger.error("Setup research error: %s", exc)
        print(f"Setup research error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected setup research failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
