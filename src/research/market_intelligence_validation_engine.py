"""
Market Intelligence validation for the production stack.

Validates whether Market Intelligence Score improves profitability of the
validated Liquidity Grab + FVG Reclaim production filter stack.
Research-only; no new signals, setups, or engine modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.market_intelligence_engine import MarketIntelligenceEngine
from src.research.filter_research_engine import (
    RESEARCH_DAYS,
    FilterResearchEngine,
    FilteredTradeRecord,
    _json_safe,
)
from src.research.production_stack_analyzer import PRODUCTION_FILTERS, PRODUCTION_SETUP
from src.research.rsi_divergence_research_engine import RsiDivergenceDetector

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "market_intelligence_validation.json"

SCORE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-20", 0.0, 20.0),
    ("20-40", 20.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60-80", 60.0, 80.0),
    ("80-100", 80.0, 100.01),
)

LOW_SCORE_BUCKETS = frozenset({"0-20", "20-40"})
HIGH_SCORE_BUCKETS = frozenset({"60-80", "80-100"})
MIN_THRESHOLD_SAMPLE_RATIO = 0.25
MIN_THRESHOLD_SAMPLE_ABSOLUTE = 30
MIN_CROSS_SEGMENT_TRADES = 5


class MarketIntelligenceValidationError(Exception):
    """Raised when market intelligence validation fails."""


@dataclass(frozen=True)
class ValidatedIntelligenceTrade:
    """Production-stack trade with intelligence and divergence context."""

    setup_type: str
    direction: str
    direction_label: str
    timeframe: str
    session: str
    trigger_bar: int
    trigger_timestamp: str
    outcome: str
    realized_pnl_points: float
    realized_rr: float
    intelligence_score: float
    direction_aligned_score: float
    score_bucket: str
    aligned_score_bucket: str
    has_divergence: bool
    primary_divergence: str
    trend_state: str
    momentum_state: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentMetrics:
    """Performance metrics for one validation segment."""

    label: str
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    max_drawdown: float
    segment_key: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreComparison:
    """Low versus high intelligence score comparison."""

    low_score: dict[str, Any]
    high_score: dict[str, Any]
    expectancy_delta: float
    profit_factor_delta: float | None
    win_rate_delta: float
    intelligence_improves_profitability: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarketIntelligenceValidationReport:
    """Aggregate market intelligence validation output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    production_stack: dict[str, Any]
    total_trades: int
    by_score_bucket: dict[str, dict[str, Any]]
    by_aligned_score_bucket: dict[str, dict[str, Any]]
    low_vs_high_comparison: dict[str, Any]
    aligned_low_vs_high_comparison: dict[str, Any]
    intelligence_improves_profitability: bool
    optimal_threshold: dict[str, Any]
    best_score_ranges: list[dict[str, Any]]
    worst_score_ranges: list[dict[str, Any]]
    score_plus_divergence: dict[str, dict[str, Any]]
    score_plus_timeframe: dict[str, dict[str, Any]]
    score_plus_session: dict[str, dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MarketIntelligenceValidationEngine:
    """Validate intelligence score impact on production stack trades."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.filter_engine = FilterResearchEngine(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.intelligence_engine = MarketIntelligenceEngine(symbol=symbol)
        self.divergence_detector = RsiDivergenceDetector()

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    @staticmethod
    def _max_drawdown(pnls: list[float]) -> float:
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        return round(max_dd, 2)

    @staticmethod
    def _direction_label(direction: str) -> str:
        return "BUY" if direction == "bullish" else "SELL"

    @staticmethod
    def _is_production_stack(trade: FilteredTradeRecord) -> bool:
        if trade.setup_type != PRODUCTION_SETUP:
            return False
        if not trade.entry_hit:
            return False
        for dimension, value in PRODUCTION_FILTERS.items():
            if getattr(trade.filters, dimension) != value:
                return False
        return True

    @staticmethod
    def _score_bucket(score: float) -> str:
        for label, lower, upper in SCORE_BUCKETS:
            if lower <= score < upper:
                return label
        return "80-100"

    @staticmethod
    def _direction_aligned_score(score: float, direction: str) -> float:
        if direction == "bearish":
            return round(100.0 - score, 2)
        return round(score, 2)

    def _metrics(self, trades: list[ValidatedIntelligenceTrade], label: str) -> SegmentMetrics:
        pnls = [trade.realized_pnl_points for trade in trades]
        rrs = [trade.realized_rr for trade in trades]
        wins = sum(1 for trade in trades if trade.outcome == "Win")
        losses = sum(1 for trade in trades if trade.outcome == "Loss")
        return SegmentMetrics(
            label=label,
            trades=len(trades),
            wins=wins,
            losses=losses,
            win_rate_pct=round((wins / len(trades)) * 100, 2) if trades else 0.0,
            profit_factor=self._profit_factor(pnls),
            expectancy=round(sum(pnls) / len(trades), 2) if trades else 0.0,
            average_rr=round(mean(rrs), 2) if rrs else 0.0,
            max_drawdown=self._max_drawdown(pnls),
        )

    def _collect_trades(self, metadata: dict[str, Any]) -> list[ValidatedIntelligenceTrade]:
        end = (
            date.fromisoformat(metadata["end_date"])
            if metadata.get("end_date")
            else date.today()
        )
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        records: list[ValidatedIntelligenceTrade] = []
        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path)
            enriched = self.intelligence_engine.enrich(frame)
            rsi = self.divergence_detector._compute_rsi(enriched["Close"].astype(float))

            for trade in self.filter_engine._collect_trades(frame, timeframe_label):
                if not self._is_production_stack(trade):
                    continue

                intelligence = self.intelligence_engine.evaluate_bar(enriched, trade.trigger_bar)
                divergence_types = self.divergence_detector.detect(enriched, trade.trigger_bar, rsi)
                primary = self.divergence_detector.primary_divergence(divergence_types, trade.direction)
                aligned = self._direction_aligned_score(
                    intelligence.intelligence_score,
                    trade.direction,
                )

                records.append(
                    ValidatedIntelligenceTrade(
                        setup_type=trade.setup_type,
                        direction=trade.direction,
                        direction_label=self._direction_label(trade.direction),
                        timeframe=trade.timeframe,
                        session=trade.filters.session,
                        trigger_bar=trade.trigger_bar,
                        trigger_timestamp=trade.trigger_timestamp,
                        outcome=trade.outcome,
                        realized_pnl_points=trade.realized_pnl_points,
                        realized_rr=trade.realized_rr,
                        intelligence_score=intelligence.intelligence_score,
                        direction_aligned_score=aligned,
                        score_bucket=self._score_bucket(intelligence.intelligence_score),
                        aligned_score_bucket=self._score_bucket(aligned),
                        has_divergence=bool(divergence_types),
                        primary_divergence=primary.value,
                        trend_state=intelligence.trend_state,
                        momentum_state=intelligence.momentum_state,
                    )
                )
        return records

    def _bucket_metrics(
        self,
        trades: list[ValidatedIntelligenceTrade],
        accessor: Any,
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[ValidatedIntelligenceTrade]] = defaultdict(list)
        for trade in trades:
            grouped[accessor(trade)].append(trade)
        return {
            label: self._metrics(bucket, label).as_dict()
            for label, bucket in sorted(grouped.items())
        }

    def _low_vs_high(
        self,
        trades: list[ValidatedIntelligenceTrade],
        bucket_accessor: Any,
        low_buckets: frozenset[str],
        high_buckets: frozenset[str],
        label: str,
    ) -> ScoreComparison:
        low = [trade for trade in trades if bucket_accessor(trade) in low_buckets]
        high = [trade for trade in trades if bucket_accessor(trade) in high_buckets]
        low_metrics = self._metrics(low, f"{label} Low Score")
        high_metrics = self._metrics(high, f"{label} High Score")
        pf_delta = None
        if low_metrics.profit_factor is not None and high_metrics.profit_factor is not None:
            pf_delta = round(high_metrics.profit_factor - low_metrics.profit_factor, 2)
        improves = high_metrics.expectancy > low_metrics.expectancy and (
            (high_metrics.profit_factor or 0) >= (low_metrics.profit_factor or 0)
        )
        return ScoreComparison(
            low_score=low_metrics.as_dict(),
            high_score=high_metrics.as_dict(),
            expectancy_delta=round(high_metrics.expectancy - low_metrics.expectancy, 2),
            profit_factor_delta=pf_delta,
            win_rate_delta=round(high_metrics.win_rate_pct - low_metrics.win_rate_pct, 2),
            intelligence_improves_profitability=improves,
        )

    def _find_optimal_threshold(
        self,
        trades: list[ValidatedIntelligenceTrade],
        score_accessor: Any,
    ) -> dict[str, Any]:
        min_sample = max(
            MIN_THRESHOLD_SAMPLE_ABSOLUTE,
            int(len(trades) * MIN_THRESHOLD_SAMPLE_RATIO),
        )
        candidates: list[dict[str, Any]] = []
        for threshold in range(0, 101, 5):
            subset = [trade for trade in trades if score_accessor(trade) >= threshold]
            if len(subset) < min_sample:
                continue
            metrics = self._metrics(subset, f"Score >= {threshold}")
            candidates.append(
                {
                    "threshold": threshold,
                    "trades": metrics.trades,
                    "expectancy": metrics.expectancy,
                    "profit_factor": metrics.profit_factor,
                    "win_rate_pct": metrics.win_rate_pct,
                    "metrics": metrics.as_dict(),
                }
            )

        if not candidates:
            return {
                "minimum_score_threshold": None,
                "reason": "No threshold satisfied minimum sample size.",
                "min_sample_required": min_sample,
                "candidates": [],
            }

        best = max(candidates, key=lambda item: (item["expectancy"], item["trades"]))
        return {
            "minimum_score_threshold": best["threshold"],
            "min_sample_required": min_sample,
            "selected_trades": best["trades"],
            "selected_expectancy": best["expectancy"],
            "selected_profit_factor": best["profit_factor"],
            "metrics_at_threshold": best["metrics"],
            "all_candidates": candidates,
        }

    def _cross_analysis(
        self,
        trades: list[ValidatedIntelligenceTrade],
        key_builder: Any,
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[ValidatedIntelligenceTrade]] = defaultdict(list)
        for trade in trades:
            grouped[key_builder(trade)].append(trade)

        results: dict[str, dict[str, Any]] = {}
        for label, bucket in sorted(grouped.items()):
            if len(bucket) < MIN_CROSS_SEGMENT_TRADES:
                continue
            results[label] = self._metrics(bucket, label).as_dict()
        return results

    def _rank_buckets(
        self,
        bucket_metrics: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ranked = sorted(
            bucket_metrics.values(),
            key=lambda item: (item["expectancy"], item["trades"]),
            reverse=True,
        )
        best = ranked[:5]
        worst = list(reversed(ranked[-5:]))
        return best, worst

    def _conclusions(
        self,
        raw_comparison: ScoreComparison,
        aligned_comparison: ScoreComparison,
        optimal: dict[str, Any],
    ) -> list[str]:
        notes: list[str] = []
        notes.append(
            f"Raw intelligence score improves profitability: "
            f"{raw_comparison.intelligence_improves_profitability} "
            f"(high expectancy {raw_comparison.high_score['expectancy']} vs "
            f"low {raw_comparison.low_score['expectancy']}, "
            f"delta {raw_comparison.expectancy_delta})."
        )
        notes.append(
            f"Direction-aligned score comparison: high "
            f"{aligned_comparison.high_score['expectancy']} vs low "
            f"{aligned_comparison.low_score['expectancy']} "
            f"(delta {aligned_comparison.expectancy_delta})."
        )
        threshold = optimal.get("minimum_score_threshold")
        if threshold is not None:
            notes.append(
                f"Optimal raw score threshold >= {threshold} "
                f"({optimal['selected_trades']} trades, "
                f"expectancy {optimal['selected_expectancy']})."
            )
        return notes

    def run(self, metadata: dict[str, Any]) -> MarketIntelligenceValidationReport:
        """Run market intelligence validation on production stack trades."""
        started = time.perf_counter()
        trades = self._collect_trades(metadata)

        by_score_bucket = self._bucket_metrics(trades, lambda item: item.score_bucket)
        by_aligned_bucket = self._bucket_metrics(trades, lambda item: item.aligned_score_bucket)

        raw_comparison = self._low_vs_high(
            trades,
            lambda item: item.score_bucket,
            LOW_SCORE_BUCKETS,
            HIGH_SCORE_BUCKETS,
            "Raw Intelligence",
        )
        aligned_comparison = self._low_vs_high(
            trades,
            lambda item: item.aligned_score_bucket,
            LOW_SCORE_BUCKETS,
            HIGH_SCORE_BUCKETS,
            "Direction-Aligned Intelligence",
        )

        optimal = self._find_optimal_threshold(
            trades,
            lambda item: item.intelligence_score,
        )
        best_ranges, worst_ranges = self._rank_buckets(by_score_bucket)

        score_plus_divergence = self._cross_analysis(
            trades,
            lambda item: (
                f"{item.aligned_score_bucket} | "
                f"divergence={'Yes' if item.has_divergence else 'No'}"
            ),
        )
        score_plus_timeframe = self._cross_analysis(
            trades,
            lambda item: f"{item.aligned_score_bucket} | {item.timeframe}",
        )
        score_plus_session = self._cross_analysis(
            trades,
            lambda item: f"{item.aligned_score_bucket} | {item.session}",
        )

        improves = raw_comparison.intelligence_improves_profitability

        return MarketIntelligenceValidationReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            production_stack={
                "setup": PRODUCTION_SETUP,
                "filters": PRODUCTION_FILTERS,
                "total_trades": len(trades),
            },
            total_trades=len(trades),
            by_score_bucket=by_score_bucket,
            by_aligned_score_bucket=by_aligned_bucket,
            low_vs_high_comparison=raw_comparison.as_dict(),
            aligned_low_vs_high_comparison=aligned_comparison.as_dict(),
            intelligence_improves_profitability=improves,
            optimal_threshold=optimal,
            best_score_ranges=best_ranges,
            worst_score_ranges=worst_ranges,
            score_plus_divergence=score_plus_divergence,
            score_plus_timeframe=score_plus_timeframe,
            score_plus_session=score_plus_session,
            conclusions=self._conclusions(raw_comparison, aligned_comparison, optimal),
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_market_intelligence_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> MarketIntelligenceValidationReport:
    """Run validation and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise MarketIntelligenceValidationError(
            f"Filter research report not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = MarketIntelligenceValidationEngine(
        symbol=metadata.get("symbol", "NIFTY50"),
        research_days=metadata.get("research_window_days", RESEARCH_DAYS),
        timeframes=tuple(metadata.get("timeframes_analyzed", ("5M", "15M", "1H"))),
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Market intelligence validation completed: trades=%s improves=%s threshold=%s",
        report.total_trades,
        report.intelligence_improves_profitability,
        report.optimal_threshold.get("minimum_score_threshold"),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_market_intelligence_validation_report()
        print("Market Intelligence Validation Summary")
        print(f"Production Stack Trades: {report.total_trades}")
        print(f"Intelligence Improves Profitability: {report.intelligence_improves_profitability}")
        threshold = report.optimal_threshold.get("minimum_score_threshold")
        if threshold is not None:
            print(
                f"Optimal Raw Score Threshold: >= {threshold} "
                f"({report.optimal_threshold['selected_trades']} trades, "
                f"exp {report.optimal_threshold['selected_expectancy']})"
            )
        raw = report.low_vs_high_comparison
        print(
            f"Raw High Score Expectancy: {raw['high_score']['expectancy']} | "
            f"Low: {raw['low_score']['expectancy']}"
        )
        for note in report.conclusions:
            print(note)
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except MarketIntelligenceValidationError as exc:
        logger.error("Market intelligence validation error: %s", exc)
        print(f"Market intelligence validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected market intelligence validation failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
