"""
Production stack segment analysis for SmartMoneyEngine.

Analyzes the validated Liquidity Grab + FVG Reclaim production filter stack
across timeframe, direction, session, and day-of-week segments.
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import pandas as pd

from src.research.filter_research_engine import (
    RESEARCH_DAYS,
    FilterResearchEngine,
    FilteredTradeRecord,
    _json_safe,
)
from src.signals.setup_classifier import SetupType

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBUST_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "robust_filter_report.json"
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "production_stack_analysis.json"

PRODUCTION_SETUP = SetupType.LIQUIDITY_GRAB_FVG_RECLAIM.value
PRODUCTION_FILTERS = {
    "rsi_band": "50-60",
    "volume_spike": "No",
}
MIN_CONFIG_TRADES = 15
BEST_CONFIG_DIMENSIONS = frozenset({"timeframe", "direction_label", "session"})
SEGMENT_RANKING_MIN_TRADES = 5


class ProductionStackAnalyzerError(Exception):
    """Raised when production stack analysis fails."""


@dataclass(frozen=True)
class StackTrade:
    """One production-stack trade with segment attributes."""

    setup_type: str
    direction: str
    direction_label: str
    timeframe: str
    session: str
    day_of_week: str
    trigger_timestamp: str
    outcome: str
    realized_pnl_points: float
    realized_rr: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentMetrics:
    """Performance metrics for one segment."""

    dimension: str
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
class ProductionStackReport:
    """Production stack segment analysis report."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    production_stack: dict[str, Any]
    overall_metrics: dict[str, Any]
    by_timeframe: dict[str, dict[str, Any]]
    by_direction: dict[str, dict[str, Any]]
    by_session: dict[str, dict[str, Any]]
    by_day_of_week: dict[str, dict[str, Any]]
    top_10_profitable_segments: list[dict[str, Any]]
    worst_10_segments: list[dict[str, Any]]
    best_production_configuration: dict[str, Any]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProductionStackAnalyzer:
    """
    Analyze validated production filter stack performance by segment.

    Parameters
    ----------
    robust_report_path : Path | str, optional
        Path to robust filter validation report.
    filter_report_path : Path | str, optional
        Path to filter research report for date range metadata.
    """

    def __init__(
        self,
        robust_report_path: Path | str = DEFAULT_ROBUST_REPORT_PATH,
        filter_report_path: Path | str = DEFAULT_FILTER_REPORT_PATH,
    ) -> None:
        self.robust_report_path = Path(robust_report_path)
        self.filter_report_path = Path(filter_report_path)

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
    def _day_of_week(timestamp: str) -> str:
        parsed = pd.to_datetime(timestamp, errors="coerce")
        if pd.isna(parsed):
            return "Unknown"
        return parsed.day_name()

    @staticmethod
    def _matches_production_stack(trade: FilteredTradeRecord) -> bool:
        if trade.setup_type != PRODUCTION_SETUP:
            return False
        if not trade.entry_hit:
            return False
        for dimension, value in PRODUCTION_FILTERS.items():
            if getattr(trade.filters, dimension) != value:
                return False
        return True

    def _to_stack_trade(self, trade: FilteredTradeRecord) -> StackTrade:
        return StackTrade(
            setup_type=trade.setup_type,
            direction=trade.direction,
            direction_label=self._direction_label(trade.direction),
            timeframe=trade.timeframe,
            session=trade.filters.session,
            day_of_week=self._day_of_week(trade.trigger_timestamp),
            trigger_timestamp=trade.trigger_timestamp,
            outcome=trade.outcome,
            realized_pnl_points=trade.realized_pnl_points,
            realized_rr=trade.realized_rr,
        )

    def _segment_metrics(
        self,
        dimension: str,
        label: str,
        trades: list[StackTrade],
        segment_key: dict[str, str] | None = None,
    ) -> SegmentMetrics:
        pnls = [trade.realized_pnl_points for trade in trades]
        rrs = [trade.realized_rr for trade in trades]
        wins = sum(1 for trade in trades if trade.outcome == "Win")
        losses = sum(1 for trade in trades if trade.outcome == "Loss")
        return SegmentMetrics(
            dimension=dimension,
            label=label,
            trades=len(trades),
            wins=wins,
            losses=losses,
            win_rate_pct=round((wins / len(trades)) * 100, 2) if trades else 0.0,
            profit_factor=self._profit_factor(pnls),
            expectancy=round(sum(pnls) / len(trades), 2) if trades else 0.0,
            average_rr=round(mean(rrs), 2) if rrs else 0.0,
            max_drawdown=self._max_drawdown(pnls),
            segment_key=segment_key or {},
        )

    def _group_metrics(
        self,
        trades: list[StackTrade],
        dimension: str,
        accessor: Callable[[StackTrade], str],
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[StackTrade]] = {}
        for trade in trades:
            grouped.setdefault(accessor(trade), []).append(trade)

        return {
            label: self._segment_metrics(
                dimension=dimension,
                label=label,
                trades=grouped[label],
                segment_key={dimension: label},
            ).as_dict()
            for label in sorted(grouped)
        }

    def _configuration_segments(
        self,
        trades: list[StackTrade],
        min_trades: int,
    ) -> list[SegmentMetrics]:
        dimensions = ("timeframe", "direction_label", "session", "day_of_week")
        segments: list[SegmentMetrics] = []

        for size in range(1, len(dimensions) + 1):
            for subset in itertools.combinations(dimensions, size):
                grouped: dict[tuple[tuple[str, str], ...], list[StackTrade]] = {}
                for trade in trades:
                    key = tuple((name, getattr(trade, name)) for name in subset)
                    grouped.setdefault(key, []).append(trade)

                for key, bucket in grouped.items():
                    if len(bucket) < min_trades:
                        continue
                    segment_key = dict(key)
                    label = " | ".join(f"{name}={value}" for name, value in key)
                    segments.append(
                        self._segment_metrics(
                            dimension="+".join(subset),
                            label=label,
                            trades=bucket,
                            segment_key=segment_key,
                        )
                    )
        return segments

    def _load_metadata(self) -> dict[str, Any]:
        if self.filter_report_path.exists():
            with self.filter_report_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        if self.robust_report_path.exists():
            with self.robust_report_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        raise ProductionStackAnalyzerError(
            "Neither filter research nor robust filter report found."
        )

    def _collect_production_trades(self, metadata: dict[str, Any]) -> list[StackTrade]:
        symbol = metadata.get("symbol", "NIFTY50")
        research_days = metadata.get("research_window_days", RESEARCH_DAYS)
        timeframes = tuple(metadata.get("timeframes_analyzed", ("5M", "15M", "1H")))
        end = (
            date.fromisoformat(metadata["end_date"])
            if metadata.get("end_date")
            else date.today()
        )
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=research_days)
        )

        engine = FilterResearchEngine(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        stack_trades: list[StackTrade] = []
        for timeframe_label in timeframes:
            path = engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path)
            logger.info(
                "Analyzing production stack trades from %s (%s rows).",
                path.name,
                len(frame),
            )
            for trade in engine._collect_trades(frame, timeframe_label):
                if self._matches_production_stack(trade):
                    stack_trades.append(self._to_stack_trade(trade))
        return stack_trades

    def run(self) -> ProductionStackReport:
        """Run production stack segment analysis."""
        started = time.perf_counter()
        metadata = self._load_metadata()
        trades = self._collect_production_trades(metadata)

        overall = self._segment_metrics(
            dimension="overall",
            label="Production Stack",
            trades=trades,
        ).as_dict()

        ranking_pool = self._configuration_segments(trades, SEGMENT_RANKING_MIN_TRADES)
        ranked = sorted(ranking_pool, key=lambda item: item.expectancy, reverse=True)
        top_10 = [segment.as_dict() for segment in ranked[:10]]
        worst_10 = [segment.as_dict() for segment in ranked[-10:][::-1]]

        preferred_configs = [
            segment
            for segment in ranking_pool
            if set(segment.segment_key) == BEST_CONFIG_DIMENSIONS
            and segment.trades >= MIN_CONFIG_TRADES
        ]
        preferred_configs.sort(key=lambda item: (item.expectancy, item.trades), reverse=True)

        full_configs = [
            segment
            for segment in ranking_pool
            if len(segment.segment_key) == 4 and segment.trades >= MIN_CONFIG_TRADES
        ]
        full_configs.sort(key=lambda item: (item.expectancy, item.trades), reverse=True)

        best_config = (
            preferred_configs[0]
            if preferred_configs
            else (full_configs[0] if full_configs else (ranked[0] if ranked else None))
        )

        best_production_configuration: dict[str, Any]
        if best_config is None:
            best_production_configuration = {
                "configuration": {},
                "metrics": {},
                "description": "No qualifying configuration found.",
            }
        else:
            configuration = {
                "setup": PRODUCTION_SETUP,
                **PRODUCTION_FILTERS,
                **best_config.segment_key,
            }
            best_production_configuration = {
                "configuration": configuration,
                "metrics": best_config.as_dict(),
                "description": self._configuration_description(configuration),
            }

        return ProductionStackReport(
            symbol=metadata.get("symbol", "NIFTY50"),
            research_window_days=metadata.get("research_window_days", RESEARCH_DAYS),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            production_stack={
                "setup": PRODUCTION_SETUP,
                "filters": PRODUCTION_FILTERS,
                "total_trades": len(trades),
            },
            overall_metrics=overall,
            by_timeframe=self._group_metrics(trades, "timeframe", lambda trade: trade.timeframe),
            by_direction=self._group_metrics(
                trades, "direction", lambda trade: trade.direction_label
            ),
            by_session=self._group_metrics(trades, "session", lambda trade: trade.session),
            by_day_of_week=self._group_metrics(
                trades, "day_of_week", lambda trade: trade.day_of_week
            ),
            top_10_profitable_segments=top_10,
            worst_10_segments=worst_10,
            best_production_configuration=best_production_configuration,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )

    @staticmethod
    def _configuration_description(configuration: dict[str, str]) -> str:
        parts = [
            configuration.get("setup", PRODUCTION_SETUP),
            f"RSI {configuration.get('rsi_band', '50-60')}",
            "No Volume Spike",
        ]
        if configuration.get("timeframe"):
            parts.append(configuration["timeframe"])
        if configuration.get("session"):
            parts.append(f"{configuration['session']} Session")
        if configuration.get("direction_label"):
            parts.append(configuration["direction_label"])
        if configuration.get("day_of_week"):
            parts.append(configuration["day_of_week"])
        return " + ".join(parts)


def generate_production_stack_analysis(
    report_path: Path | str | None = None,
    robust_report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> ProductionStackReport:
    """Run production stack analysis and export JSON report."""
    analyzer = ProductionStackAnalyzer(
        robust_report_path=robust_report_path or DEFAULT_ROBUST_REPORT_PATH,
        filter_report_path=filter_report_path or DEFAULT_FILTER_REPORT_PATH,
    )
    report = analyzer.run()

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Production stack analysis completed: trades=%s",
        report.production_stack["total_trades"],
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_production_stack_analysis()
        print("Production Stack Analysis")
        print(f"Setup: {report.production_stack['setup']}")
        print(f"Filters: {report.production_stack['filters']}")
        print(f"Total Trades: {report.production_stack['total_trades']}")
        print(f"Overall Expectancy: {report.overall_metrics['expectancy']}")
        print(f"Overall PF: {report.overall_metrics['profit_factor']}")
        print("Best Production Configuration:")
        print(f"  {report.best_production_configuration.get('description', 'N/A')}")
        best_metrics = report.best_production_configuration.get("metrics", {})
        if best_metrics:
            print(
                f"  Trades={best_metrics.get('trades')} "
                f"WR={best_metrics.get('win_rate_pct')}% "
                f"Exp={best_metrics.get('expectancy')}"
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except ProductionStackAnalyzerError as exc:
        logger.error("Production stack analysis error: %s", exc)
        print(f"Production stack analysis error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected production stack analysis failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
