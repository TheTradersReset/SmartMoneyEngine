"""
Market location validation for the production stack.

Determines whether distance to support/resistance improves profitability
for the validated Liquidity Grab + FVG Reclaim stack with intelligence filter.
Research-only; no new signals or setup changes.
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
from src.signals.setup_classifier import SetupClassification, SetupClassifier

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "market_location_validation.json"

MIN_INTELLIGENCE_SCORE = 65
MIN_SEGMENT_TRADES = 5

LOCATION_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("Very Close", 0.0, 0.25),
    ("Close", 0.25, 0.5),
    ("Medium", 0.5, 1.0),
    ("Far", 1.0, float("inf")),
)


class MarketLocationValidationError(Exception):
    """Raised when market location validation fails."""


@dataclass(frozen=True)
class LocationTradeRecord:
    """Production-stack trade with location and reward context."""

    setup_type: str
    direction: str
    direction_label: str
    timeframe: str
    session: str
    trigger_timestamp: str
    outcome: str
    realized_pnl_points: float
    realized_rr: float
    intelligence_score: float
    entry: float
    stop_loss: float
    target_1: float
    distance_to_support: float | None
    distance_to_resistance: float | None
    support_distance_atr: float | None
    resistance_distance_atr: float | None
    support_distance_bucket: str
    resistance_distance_bucket: str
    room_to_target: float | None
    room_to_target_atr: float | None
    room_to_target_bucket: str
    reward_risk_potential: float | None
    reward_risk_bucket: str
    market_location: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentMetrics:
    """Performance metrics for one location segment."""

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
class MarketLocationValidationReport:
    """Aggregate market location validation output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    production_stack: dict[str, Any]
    total_trades: int
    by_support_distance: dict[str, dict[str, Any]]
    by_resistance_distance: dict[str, dict[str, Any]]
    by_room_to_target: dict[str, dict[str, Any]]
    by_reward_risk_potential: dict[str, dict[str, Any]]
    by_market_location: dict[str, dict[str, Any]]
    optimal_location_conditions: dict[str, Any]
    location_avoidance_guidance: dict[str, Any]
    best_location_segments: list[dict[str, Any]]
    worst_location_segments: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MarketLocationValidationEngine:
    """Validate location distance impact on filtered production stack trades."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
        min_intelligence_score: float = MIN_INTELLIGENCE_SCORE,
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.min_intelligence_score = min_intelligence_score
        self.filter_engine = FilterResearchEngine(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.classifier = SetupClassifier()
        self.intelligence_engine = MarketIntelligenceEngine(symbol=symbol)

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
    def _atr_ratio(distance: float | None, atr: float) -> float | None:
        if distance is None or atr <= 0:
            return None
        return round(distance / atr, 3)

    @staticmethod
    def _distance_bucket(ratio: float | None) -> str:
        if ratio is None:
            return "Unknown"
        if ratio <= 0.25:
            return "Very Close"
        if ratio <= 0.5:
            return "Close"
        if ratio <= 1.0:
            return "Medium"
        return "Far"

    @staticmethod
    def _room_to_target(
        setup: SetupClassification,
        levels: dict[str, float | None],
    ) -> float | None:
        entry = setup.entry
        support = levels.get("major_support")
        resistance = levels.get("major_resistance")

        if setup.direction == "bullish":
            target_room = setup.target_1 - entry
            if resistance is not None and resistance > entry:
                target_room = min(target_room, resistance - entry)
            return round(max(target_room, 0.0), 2)

        target_room = entry - setup.target_1
        if support is not None and support < entry:
            target_room = min(target_room, entry - support)
        return round(max(target_room, 0.0), 2)

    @staticmethod
    def _reward_risk_potential(setup: SetupClassification) -> float | None:
        risk = abs(setup.entry - setup.stop_loss)
        reward = abs(setup.target_1 - setup.entry)
        if risk <= 0:
            return None
        return round(reward / risk, 2)

    @staticmethod
    def _market_location_label(
        support_ratio: float | None,
        resistance_ratio: float | None,
    ) -> str:
        near_support = support_ratio is not None and support_ratio <= 0.5
        near_resistance = resistance_ratio is not None and resistance_ratio <= 0.5
        if near_support and not near_resistance:
            return "Near Support"
        if near_resistance and not near_support:
            return "Near Resistance"
        if near_support and near_resistance:
            if (support_ratio or 0) <= (resistance_ratio or 0):
                return "Near Support"
            return "Near Resistance"
        return "Mid Range"

    def _collect_trades(self, metadata: dict[str, Any]) -> list[LocationTradeRecord]:
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

        records: list[LocationTradeRecord] = []
        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path)
            filter_enriched = self.filter_engine.context_builder.enrich(frame)
            intel_enriched = self.intelligence_engine.enrich(frame)
            atr_series = intel_enriched["_atr"]

            setup_by_key: dict[tuple[int, str, str], SetupClassification] = {
                (setup.trigger_bar, setup.setup_type, setup.direction): setup
                for setup in self.classifier.classify(filter_enriched)
            }

            for trade in self.filter_engine._collect_trades(frame, timeframe_label):
                if not self._is_production_stack(trade):
                    continue

                intelligence = self.intelligence_engine.evaluate_bar(
                    intel_enriched,
                    trade.trigger_bar,
                )
                if intelligence.intelligence_score < self.min_intelligence_score:
                    continue

                setup = setup_by_key.get(
                    (trade.trigger_bar, trade.setup_type, trade.direction),
                )
                if setup is None:
                    continue

                levels = self.intelligence_engine._market_levels(
                    intel_enriched,
                    trade.trigger_bar,
                )
                atr = (
                    float(atr_series.iloc[trade.trigger_bar])
                    if pd.notna(atr_series.iloc[trade.trigger_bar])
                    else 1.0
                )
                close = levels["close"]
                support = levels["major_support"]
                resistance = levels["major_resistance"]

                distance_support = round(close - support, 2) if support is not None else None
                distance_resistance = (
                    round(resistance - close, 2) if resistance is not None else None
                )
                support_ratio = self._atr_ratio(
                    abs(distance_support) if distance_support is not None else None,
                    atr,
                )
                resistance_ratio = self._atr_ratio(
                    abs(distance_resistance) if distance_resistance is not None else None,
                    atr,
                )
                room = self._room_to_target(setup, levels)
                room_ratio = self._atr_ratio(room, atr)
                rr_potential = self._reward_risk_potential(setup)

                records.append(
                    LocationTradeRecord(
                        setup_type=trade.setup_type,
                        direction=trade.direction,
                        direction_label=self._direction_label(trade.direction),
                        timeframe=trade.timeframe,
                        session=trade.filters.session,
                        trigger_timestamp=trade.trigger_timestamp,
                        outcome=trade.outcome,
                        realized_pnl_points=trade.realized_pnl_points,
                        realized_rr=trade.realized_rr,
                        intelligence_score=intelligence.intelligence_score,
                        entry=setup.entry,
                        stop_loss=setup.stop_loss,
                        target_1=setup.target_1,
                        distance_to_support=distance_support,
                        distance_to_resistance=distance_resistance,
                        support_distance_atr=support_ratio,
                        resistance_distance_atr=resistance_ratio,
                        support_distance_bucket=self._distance_bucket(support_ratio),
                        resistance_distance_bucket=self._distance_bucket(resistance_ratio),
                        room_to_target=room,
                        room_to_target_atr=room_ratio,
                        room_to_target_bucket=self._distance_bucket(room_ratio),
                        reward_risk_potential=rr_potential,
                        reward_risk_bucket=self._reward_risk_bucket(rr_potential),
                        market_location=self._market_location_label(
                            support_ratio,
                            resistance_ratio,
                        ),
                    )
                )
        return records

    def _metrics(self, trades: list[LocationTradeRecord], label: str) -> SegmentMetrics:
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

    def _bucket_metrics(
        self,
        trades: list[LocationTradeRecord],
        accessor: Any,
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[LocationTradeRecord]] = defaultdict(list)
        for trade in trades:
            grouped[accessor(trade)].append(trade)
        bucket_order = ["Very Close", "Close", "Medium", "Far", "Unknown"]
        results: dict[str, dict[str, Any]] = {}
        for label in bucket_order:
            if label in grouped:
                results[label] = self._metrics(grouped[label], label).as_dict()
        for label, bucket in grouped.items():
            if label not in results:
                results[label] = self._metrics(bucket, label).as_dict()
        return results

    @staticmethod
    def _reward_risk_bucket(rr: float | None) -> str:
        if rr is None:
            return "Unknown"
        if rr < 1.0:
            return "Below 1R"
        if rr < 1.5:
            return "1R-1.5R"
        if rr < 2.0:
            return "1.5R-2R"
        return "2R+"

    def _optimal_conditions(
        self,
        trades: list[LocationTradeRecord],
    ) -> dict[str, Any]:
        pools = {
            "support_distance": self._bucket_metrics(trades, lambda item: item.support_distance_bucket),
            "resistance_distance": self._bucket_metrics(trades, lambda item: item.resistance_distance_bucket),
            "room_to_target": self._bucket_metrics(trades, lambda item: item.room_to_target_bucket),
            "reward_risk_potential": self._bucket_metrics(trades, lambda item: item.reward_risk_bucket),
            "market_location": self._bucket_metrics(trades, lambda item: item.market_location),
        }

        optimal: dict[str, Any] = {}
        for dimension, buckets in pools.items():
            eligible = {
                label: metrics
                for label, metrics in buckets.items()
                if metrics["trades"] >= MIN_SEGMENT_TRADES and label != "Unknown"
            }
            if not eligible:
                continue
            best = max(eligible.items(), key=lambda item: (item[1]["expectancy"], item[1]["trades"]))
            optimal[dimension] = {
                "best_bucket": best[0],
                "metrics": best[1],
            }
        return optimal

    def _avoidance_guidance(
        self,
        by_location: dict[str, dict[str, Any]],
        by_support: dict[str, dict[str, Any]],
        by_resistance: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        guidance: dict[str, Any] = {}

        if "Near Support" in by_location and "Near Resistance" in by_location:
            support_exp = by_location["Near Support"]["expectancy"]
            resistance_exp = by_location["Near Resistance"]["expectancy"]
            guidance["avoid_near_support"] = support_exp < resistance_exp
            guidance["avoid_near_resistance"] = resistance_exp < support_exp
            guidance["near_support_expectancy"] = support_exp
            guidance["near_resistance_expectancy"] = resistance_exp

        if "Very Close" in by_support:
            guidance["very_close_to_support_expectancy"] = by_support["Very Close"]["expectancy"]
        if "Very Close" in by_resistance:
            guidance["very_close_to_resistance_expectancy"] = by_resistance["Very Close"]["expectancy"]

        if guidance.get("avoid_near_resistance") is True:
            guidance["recommendation"] = "Avoid entries very close to resistance."
        elif guidance.get("avoid_near_support") is True:
            guidance["recommendation"] = "Avoid entries very close to support."
        else:
            guidance["recommendation"] = "No strong avoidance rule; location edge is bucket-specific."

        return guidance

    def _rank_segments(
        self,
        trades: list[LocationTradeRecord],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        segments: list[SegmentMetrics] = []
        builders = (
            ("support_distance", lambda item: item.support_distance_bucket),
            ("resistance_distance", lambda item: item.resistance_distance_bucket),
            ("room_to_target", lambda item: item.room_to_target_bucket),
            ("reward_risk", lambda item: item.reward_risk_bucket),
            ("market_location", lambda item: item.market_location),
        )
        for dimension, accessor in builders:
            grouped: dict[str, list[LocationTradeRecord]] = defaultdict(list)
            for trade in trades:
                grouped[accessor(trade)].append(trade)
            for label, bucket in grouped.items():
                if len(bucket) < MIN_SEGMENT_TRADES or label == "Unknown":
                    continue
                metrics = self._metrics(bucket, f"{dimension}={label}")
                metrics.segment_key = {dimension: label}
                segments.append(metrics)

        ranked = sorted(segments, key=lambda item: item.expectancy, reverse=True)
        return [item.as_dict() for item in ranked[:10]], [item.as_dict() for item in ranked[-10:][::-1]]

    def _conclusions(
        self,
        optimal: dict[str, Any],
        guidance: dict[str, Any],
    ) -> list[str]:
        notes: list[str] = []
        for dimension, payload in optimal.items():
            notes.append(
                f"Best {dimension}: {payload['best_bucket']} "
                f"(expectancy {payload['metrics']['expectancy']}, "
                f"n={payload['metrics']['trades']})."
            )
        if "recommendation" in guidance:
            notes.append(guidance["recommendation"])
        return notes

    def run(self, metadata: dict[str, Any]) -> MarketLocationValidationReport:
        """Run market location validation."""
        started = time.perf_counter()
        trades = self._collect_trades(metadata)

        by_support = self._bucket_metrics(trades, lambda item: item.support_distance_bucket)
        by_resistance = self._bucket_metrics(trades, lambda item: item.resistance_distance_bucket)
        by_room = self._bucket_metrics(trades, lambda item: item.room_to_target_bucket)
        by_rr = self._bucket_metrics(trades, lambda item: item.reward_risk_bucket)
        by_location = self._bucket_metrics(trades, lambda item: item.market_location)

        optimal = self._optimal_conditions(trades)
        guidance = self._avoidance_guidance(by_location, by_support, by_resistance)
        best, worst = self._rank_segments(trades)

        return MarketLocationValidationReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            production_stack={
                "setup": PRODUCTION_SETUP,
                "filters": {
                    **PRODUCTION_FILTERS,
                    "min_intelligence_score": self.min_intelligence_score,
                },
                "total_trades": len(trades),
            },
            total_trades=len(trades),
            by_support_distance=by_support,
            by_resistance_distance=by_resistance,
            by_room_to_target=by_room,
            by_reward_risk_potential=by_rr,
            by_market_location=by_location,
            optimal_location_conditions=optimal,
            location_avoidance_guidance=guidance,
            best_location_segments=best,
            worst_location_segments=worst,
            conclusions=self._conclusions(optimal, guidance),
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_market_location_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> MarketLocationValidationReport:
    """Run location validation and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise MarketLocationValidationError(
            f"Filter research report not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = MarketLocationValidationEngine(
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
        "Market location validation completed: trades=%s",
        report.total_trades,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_market_location_validation_report()
        print("Market Location Validation Summary")
        print(f"Filtered Production Stack Trades: {report.total_trades}")
        print(f"Filters: {report.production_stack['filters']}")
        for note in report.conclusions:
            print(note)
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except MarketLocationValidationError as exc:
        logger.error("Market location validation error: %s", exc)
        print(f"Market location validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected market location validation failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
