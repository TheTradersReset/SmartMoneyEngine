"""
VWAP validation for the current production stack.

Validates whether VWAP position, reclaim/rejection, and distance improve
profitability for the filtered Liquidity Grab + FVG Reclaim stack.
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
from src.signals.setup_classifier import SetupClassifier

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "vwap_validation_report.json"

MIN_INTELLIGENCE_SCORE = 65
MIN_SEGMENT_TRADES = 5
MIN_FILTER_SAMPLE_RATIO = 0.20
MIN_FILTER_SAMPLE_ABSOLUTE = 15
MIN_FILTER_EXPECTANCY_DELTA = 3.0
EXCLUDED_PRODUCTION_FILTER_VALUES = frozenset({"Not Aligned", "Unknown"})

DISTANCE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("Very Close", 0.0, 0.25),
    ("Close", 0.25, 0.5),
    ("Medium", 0.5, 1.0),
    ("Far", 1.0, float("inf")),
)


class VwapValidationError(Exception):
    """Raised when VWAP validation fails."""


@dataclass(frozen=True)
class VwapTradeRecord:
    """Production-stack trade with VWAP context."""

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
    vwap_position: str
    vwap_reclaim: str
    vwap_rejection: str
    distance_from_vwap: float | None
    distance_from_vwap_atr: float | None
    distance_from_vwap_bucket: str
    direction_aligned_vwap: bool
    market_location: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentMetrics:
    """Performance metrics for one VWAP segment."""

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
class VwapComparison:
    """Low versus high VWAP segment comparison."""

    low_segment: dict[str, Any]
    high_segment: dict[str, Any]
    expectancy_delta: float
    profit_factor_delta: float | None
    win_rate_delta: float
    vwap_improves_profitability: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VwapValidationReport:
    """Aggregate VWAP validation output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    production_stack: dict[str, Any]
    total_trades: int
    baseline: dict[str, Any]
    by_vwap_position: dict[str, dict[str, Any]]
    by_vwap_reclaim: dict[str, dict[str, Any]]
    by_vwap_rejection: dict[str, dict[str, Any]]
    by_distance_from_vwap: dict[str, dict[str, Any]]
    by_direction_aligned_vwap: dict[str, dict[str, Any]]
    position_comparison: dict[str, Any]
    direction_aligned_comparison: dict[str, Any]
    vwap_improves_profitability: bool
    recommend_production_filter: bool
    recommended_filter: dict[str, Any]
    optimal_vwap_conditions: dict[str, Any]
    best_vwap_segments: list[dict[str, Any]]
    worst_vwap_segments: list[dict[str, Any]]
    cross_analysis: dict[str, dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class VwapValidationEngine:
    """Validate VWAP impact on the filtered production stack."""

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
    def _safe_float(value: Any) -> float | None:
        if pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _atr_ratio(distance: float | None, atr: float) -> float | None:
        if distance is None or atr <= 0:
            return None
        return round(abs(distance) / atr, 3)

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
    def _vwap_position(close: float, vwap: float | None) -> str:
        if vwap is None:
            return "Unknown"
        return "Above VWAP" if close >= vwap else "Below VWAP"

    @staticmethod
    def _vwap_reclaim(
        enriched: pd.DataFrame,
        index: int,
        direction: str,
    ) -> str:
        if index < 1:
            return "No Reclaim"

        prev = enriched.iloc[index - 1]
        curr = enriched.iloc[index]
        prev_close = VwapValidationEngine._safe_float(prev["Close"])
        curr_close = VwapValidationEngine._safe_float(curr["Close"])
        prev_vwap = VwapValidationEngine._safe_float(prev["_vwap"])
        curr_vwap = VwapValidationEngine._safe_float(curr["_vwap"])
        if None in (prev_close, curr_close, prev_vwap, curr_vwap):
            return "No Reclaim"

        crossed_up = prev_close < prev_vwap and curr_close >= curr_vwap
        crossed_down = prev_close >= prev_vwap and curr_close < curr_vwap
        if not crossed_up and not crossed_down:
            return "No Reclaim"
        if crossed_up and not crossed_down:
            return "Bullish Reclaim"
        if crossed_down and not crossed_up:
            return "Bearish Reclaim"
        return "No Reclaim"

    @staticmethod
    def _vwap_rejection(
        enriched: pd.DataFrame,
        index: int,
    ) -> str:
        if index < 1:
            return "No Rejection"

        prev = enriched.iloc[index - 1]
        curr = enriched.iloc[index]
        prev_close = VwapValidationEngine._safe_float(prev["Close"])
        curr_close = VwapValidationEngine._safe_float(curr["Close"])
        curr_low = VwapValidationEngine._safe_float(curr["Low"])
        curr_high = VwapValidationEngine._safe_float(curr["High"])
        curr_vwap = VwapValidationEngine._safe_float(curr["_vwap"])
        if None in (prev_close, curr_close, curr_low, curr_high, curr_vwap):
            return "No Rejection"

        was_below = prev_close < curr_vwap
        was_above = prev_close >= curr_vwap
        still_below = curr_close < curr_vwap
        still_above = curr_close >= curr_vwap

        if was_below and still_below and curr_low <= curr_vwap:
            return "Rejected from Below"
        if was_above and still_above and curr_high >= curr_vwap:
            return "Rejected from Above"
        return "No Rejection"

    @staticmethod
    def _direction_aligned_vwap(direction: str, vwap_position: str) -> bool:
        if vwap_position == "Unknown":
            return False
        if direction == "bullish":
            return vwap_position == "Above VWAP"
        return vwap_position == "Below VWAP"

    def _is_near_support(
        self,
        intel_enriched: pd.DataFrame,
        trigger_bar: int,
    ) -> bool:
        atr_series = intel_enriched["_atr"]
        levels = self.intelligence_engine._market_levels(intel_enriched, trigger_bar)
        atr = (
            float(atr_series.iloc[trigger_bar])
            if pd.notna(atr_series.iloc[trigger_bar])
            else 1.0
        )
        close = levels["close"]
        support = levels["major_support"]
        resistance = levels["major_resistance"]
        support_ratio = self._atr_ratio(
            abs(close - support) if support is not None else None,
            atr,
        )
        resistance_ratio = self._atr_ratio(
            abs(resistance - close) if resistance is not None else None,
            atr,
        )
        return self._market_location_label(support_ratio, resistance_ratio) == "Near Support"

    def _metrics(self, trades: list[VwapTradeRecord], label: str) -> SegmentMetrics:
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
        trades: list[VwapTradeRecord],
        accessor: Any,
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[VwapTradeRecord]] = defaultdict(list)
        for trade in trades:
            grouped[accessor(trade)].append(trade)
        bucket_order = [
            "Above VWAP",
            "Below VWAP",
            "Bullish Reclaim",
            "Bearish Reclaim",
            "No Reclaim",
            "Rejected from Below",
            "Rejected from Above",
            "No Rejection",
            "Very Close",
            "Close",
            "Medium",
            "Far",
            "Aligned",
            "Not Aligned",
            "Unknown",
        ]
        results: dict[str, dict[str, Any]] = {}
        for label in bucket_order:
            if label in grouped:
                results[label] = self._metrics(grouped[label], label).as_dict()
        for label, bucket in grouped.items():
            if label not in results:
                results[label] = self._metrics(bucket, label).as_dict()
        return results

    def _collect_trades(self, metadata: dict[str, Any]) -> list[VwapTradeRecord]:
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

        records: list[VwapTradeRecord] = []
        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path)
            filter_enriched = self.filter_engine.context_builder.enrich(frame)
            intel_enriched = self.intelligence_engine.enrich(frame)
            atr_series = filter_enriched["_atr"]

            for trade in self.filter_engine._collect_trades(frame, timeframe_label):
                if not self._is_production_stack(trade):
                    continue

                intelligence = self.intelligence_engine.evaluate_bar(
                    intel_enriched,
                    trade.trigger_bar,
                )
                if intelligence.intelligence_score < self.min_intelligence_score:
                    continue
                if self._is_near_support(intel_enriched, trade.trigger_bar):
                    continue

                row = filter_enriched.iloc[trade.trigger_bar]
                close = self._safe_float(row["Close"]) or 0.0
                vwap = self._safe_float(row["_vwap"])
                atr = (
                    float(atr_series.iloc[trade.trigger_bar])
                    if pd.notna(atr_series.iloc[trade.trigger_bar])
                    else 1.0
                )
                distance = round(close - vwap, 2) if vwap is not None else None
                distance_atr = self._atr_ratio(distance, atr)
                vwap_position = self._vwap_position(close, vwap)
                vwap_reclaim = self._vwap_reclaim(
                    filter_enriched,
                    trade.trigger_bar,
                    trade.direction,
                )
                vwap_rejection = self._vwap_rejection(filter_enriched, trade.trigger_bar)
                aligned = self._direction_aligned_vwap(trade.direction, vwap_position)

                levels = self.intelligence_engine._market_levels(
                    intel_enriched,
                    trade.trigger_bar,
                )
                support = levels["major_support"]
                resistance = levels["major_resistance"]
                support_ratio = self._atr_ratio(
                    abs(close - support) if support is not None else None,
                    atr,
                )
                resistance_ratio = self._atr_ratio(
                    abs(resistance - close) if resistance is not None else None,
                    atr,
                )

                records.append(
                    VwapTradeRecord(
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
                        vwap_position=vwap_position,
                        vwap_reclaim=vwap_reclaim,
                        vwap_rejection=vwap_rejection,
                        distance_from_vwap=distance,
                        distance_from_vwap_atr=distance_atr,
                        distance_from_vwap_bucket=self._distance_bucket(distance_atr),
                        direction_aligned_vwap=aligned,
                        market_location=self._market_location_label(
                            support_ratio,
                            resistance_ratio,
                        ),
                    )
                )
        return records

    def _segment_comparison(
        self,
        trades: list[VwapTradeRecord],
        accessor: Any,
        low_labels: frozenset[str],
        high_labels: frozenset[str],
        label: str,
    ) -> VwapComparison:
        low = [trade for trade in trades if accessor(trade) in low_labels]
        high = [trade for trade in trades if accessor(trade) in high_labels]
        low_metrics = self._metrics(low, f"{label} Low")
        high_metrics = self._metrics(high, f"{label} High")
        pf_delta = None
        if low_metrics.profit_factor is not None and high_metrics.profit_factor is not None:
            pf_delta = round(high_metrics.profit_factor - low_metrics.profit_factor, 2)
        improves = high_metrics.expectancy > low_metrics.expectancy and (
            (high_metrics.profit_factor or 0) >= (low_metrics.profit_factor or 0)
        )
        return VwapComparison(
            low_segment=low_metrics.as_dict(),
            high_segment=high_metrics.as_dict(),
            expectancy_delta=round(high_metrics.expectancy - low_metrics.expectancy, 2),
            profit_factor_delta=pf_delta,
            win_rate_delta=round(high_metrics.win_rate_pct - low_metrics.win_rate_pct, 2),
            vwap_improves_profitability=improves,
        )

    def _optimal_conditions(
        self,
        trades: list[VwapTradeRecord],
    ) -> dict[str, Any]:
        pools = {
            "vwap_position": self._bucket_metrics(trades, lambda item: item.vwap_position),
            "vwap_reclaim": self._bucket_metrics(trades, lambda item: item.vwap_reclaim),
            "vwap_rejection": self._bucket_metrics(trades, lambda item: item.vwap_rejection),
            "distance_from_vwap": self._bucket_metrics(
                trades,
                lambda item: item.distance_from_vwap_bucket,
            ),
            "direction_aligned_vwap": self._bucket_metrics(
                trades,
                lambda item: "Aligned" if item.direction_aligned_vwap else "Not Aligned",
            ),
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

    def _rank_segments(
        self,
        trades: list[VwapTradeRecord],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        segments: list[SegmentMetrics] = []
        builders = (
            ("vwap_position", lambda item: item.vwap_position),
            ("vwap_reclaim", lambda item: item.vwap_reclaim),
            ("vwap_rejection", lambda item: item.vwap_rejection),
            ("distance_from_vwap", lambda item: item.distance_from_vwap_bucket),
            ("direction_aligned_vwap", lambda item: (
                "Aligned" if item.direction_aligned_vwap else "Not Aligned"
            )),
        )
        for dimension, accessor in builders:
            grouped: dict[str, list[VwapTradeRecord]] = defaultdict(list)
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

    def _cross_analysis(
        self,
        trades: list[VwapTradeRecord],
        key_builder: Any,
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[VwapTradeRecord]] = defaultdict(list)
        for trade in trades:
            grouped[key_builder(trade)].append(trade)
        results: dict[str, dict[str, Any]] = {}
        for label, bucket in sorted(grouped.items()):
            if len(bucket) < MIN_SEGMENT_TRADES:
                continue
            results[label] = self._metrics(bucket, label).as_dict()
        return results

    def _recommend_filter(
        self,
        trades: list[VwapTradeRecord],
        baseline: SegmentMetrics,
        optimal: dict[str, Any],
    ) -> dict[str, Any]:
        min_sample = max(
            MIN_FILTER_SAMPLE_ABSOLUTE,
            int(len(trades) * MIN_FILTER_SAMPLE_RATIO),
        )
        candidates: list[dict[str, Any]] = []

        filter_builders: tuple[tuple[str, Any], ...] = (
            ("vwap_position", lambda item: item.vwap_position),
            ("vwap_reclaim", lambda item: item.vwap_reclaim),
            ("vwap_rejection", lambda item: item.vwap_rejection),
            ("distance_from_vwap", lambda item: item.distance_from_vwap_bucket),
            ("direction_aligned_vwap", lambda item: (
                "Aligned" if item.direction_aligned_vwap else "Not Aligned"
            )),
        )

        for dimension, accessor in filter_builders:
            dimension_optimal = optimal.get(dimension)
            if not dimension_optimal:
                continue
            best_bucket = dimension_optimal["best_bucket"]
            if best_bucket in EXCLUDED_PRODUCTION_FILTER_VALUES:
                continue
            subset = [trade for trade in trades if accessor(trade) == best_bucket]
            if len(subset) < min_sample:
                continue
            metrics = self._metrics(subset, best_bucket)
            expectancy_delta = round(metrics.expectancy - baseline.expectancy, 2)
            if expectancy_delta < MIN_FILTER_EXPECTANCY_DELTA:
                continue
            baseline_pf = baseline.profit_factor or 0.0
            if (metrics.profit_factor or 0.0) < baseline_pf:
                continue
            candidates.append(
                {
                    "dimension": dimension,
                    "filter_value": best_bucket,
                    "trades": metrics.trades,
                    "expectancy": metrics.expectancy,
                    "profit_factor": metrics.profit_factor,
                    "expectancy_delta_vs_baseline": expectancy_delta,
                    "metrics": metrics.as_dict(),
                }
            )

        if not candidates:
            return {
                "filter": None,
                "reason": "No VWAP filter improved expectancy with sufficient sample size.",
                "min_sample_required": min_sample,
                "candidates": [],
            }

        best = max(
            candidates,
            key=lambda item: (item["expectancy_delta_vs_baseline"], item["trades"]),
        )
        return {
            "filter": {
                "dimension": best["dimension"],
                "value": best["filter_value"],
            },
            "min_sample_required": min_sample,
            "selected_trades": best["trades"],
            "selected_expectancy": best["expectancy"],
            "selected_profit_factor": best["profit_factor"],
            "expectancy_delta_vs_baseline": best["expectancy_delta_vs_baseline"],
            "metrics_at_filter": best["metrics"],
            "all_candidates": candidates,
        }

    def _conclusions(
        self,
        baseline: SegmentMetrics,
        position_comparison: VwapComparison,
        aligned_comparison: VwapComparison,
        optimal: dict[str, Any],
        recommended: dict[str, Any],
        recommend_filter: bool,
    ) -> list[str]:
        notes: list[str] = []
        notes.append(
            f"Baseline stack expectancy: {baseline.expectancy} "
            f"(PF {baseline.profit_factor}, n={baseline.trades})."
        )
        notes.append(
            f"VWAP position comparison improves profitability: "
            f"{position_comparison.vwap_improves_profitability} "
            f"(delta {position_comparison.expectancy_delta})."
        )
        notes.append(
            f"Direction-aligned VWAP improves profitability: "
            f"{aligned_comparison.vwap_improves_profitability} "
            f"(delta {aligned_comparison.expectancy_delta})."
        )
        for dimension, payload in optimal.items():
            notes.append(
                f"Best {dimension}: {payload['best_bucket']} "
                f"(expectancy {payload['metrics']['expectancy']}, "
                f"n={payload['metrics']['trades']})."
            )
        if recommend_filter and recommended.get("filter"):
            filt = recommended["filter"]
            notes.append(
                f"Recommend production VWAP filter: {filt['dimension']}={filt['value']} "
                f"(expectancy {recommended['selected_expectancy']}, "
                f"n={recommended['selected_trades']})."
            )
        else:
            notes.append(
                recommended.get("reason", "VWAP should not become a production filter yet.")
            )
        return notes

    def _vwap_improves_overall(
        self,
        baseline: SegmentMetrics,
        optimal: dict[str, Any],
    ) -> bool:
        baseline_pf = baseline.profit_factor or 0.0
        for payload in optimal.values():
            metrics = payload["metrics"]
            if metrics["trades"] < MIN_SEGMENT_TRADES:
                continue
            if payload["best_bucket"] in EXCLUDED_PRODUCTION_FILTER_VALUES:
                continue
            expectancy_delta = metrics["expectancy"] - baseline.expectancy
            if expectancy_delta < MIN_FILTER_EXPECTANCY_DELTA:
                continue
            if (metrics["profit_factor"] or 0.0) >= baseline_pf:
                return True
        return False

    def run(self, metadata: dict[str, Any]) -> VwapValidationReport:
        """Run VWAP validation on the production stack."""
        started = time.perf_counter()
        trades = self._collect_trades(metadata)
        baseline = self._metrics(trades, "Baseline")

        by_position = self._bucket_metrics(trades, lambda item: item.vwap_position)
        by_reclaim = self._bucket_metrics(trades, lambda item: item.vwap_reclaim)
        by_rejection = self._bucket_metrics(trades, lambda item: item.vwap_rejection)
        by_distance = self._bucket_metrics(trades, lambda item: item.distance_from_vwap_bucket)
        by_aligned = self._bucket_metrics(
            trades,
            lambda item: "Aligned" if item.direction_aligned_vwap else "Not Aligned",
        )

        position_comparison = self._segment_comparison(
            trades,
            lambda item: item.vwap_position,
            frozenset({"Below VWAP"}),
            frozenset({"Above VWAP"}),
            "VWAP Position",
        )
        aligned_comparison = self._segment_comparison(
            trades,
            lambda item: "Aligned" if item.direction_aligned_vwap else "Not Aligned",
            frozenset({"Not Aligned"}),
            frozenset({"Aligned"}),
            "Direction-Aligned VWAP",
        )

        optimal = self._optimal_conditions(trades)
        recommended = self._recommend_filter(trades, baseline, optimal)
        best, worst = self._rank_segments(trades)

        cross = {
            "vwap_position_by_direction": self._cross_analysis(
                trades,
                lambda item: f"{item.vwap_position} | {item.direction_label}",
            ),
            "vwap_position_by_session": self._cross_analysis(
                trades,
                lambda item: f"{item.vwap_position} | {item.session}",
            ),
            "vwap_reclaim_by_direction": self._cross_analysis(
                trades,
                lambda item: f"{item.vwap_reclaim} | {item.direction_label}",
            ),
            "distance_by_direction": self._cross_analysis(
                trades,
                lambda item: f"{item.distance_from_vwap_bucket} | {item.direction_label}",
            ),
        }

        improves = self._vwap_improves_overall(baseline, optimal)
        recommend_filter = recommended.get("filter") is not None and improves

        return VwapValidationReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            production_stack={
                "setup": PRODUCTION_SETUP,
                "filters": {
                    **PRODUCTION_FILTERS,
                    "min_intelligence_score": self.min_intelligence_score,
                    "avoid_near_support": True,
                },
                "total_trades": len(trades),
            },
            total_trades=len(trades),
            baseline=baseline.as_dict(),
            by_vwap_position=by_position,
            by_vwap_reclaim=by_reclaim,
            by_vwap_rejection=by_rejection,
            by_distance_from_vwap=by_distance,
            by_direction_aligned_vwap=by_aligned,
            position_comparison=position_comparison.as_dict(),
            direction_aligned_comparison=aligned_comparison.as_dict(),
            vwap_improves_profitability=improves,
            recommend_production_filter=recommend_filter,
            recommended_filter=recommended,
            optimal_vwap_conditions=optimal,
            best_vwap_segments=best,
            worst_vwap_segments=worst,
            cross_analysis=cross,
            conclusions=self._conclusions(
                baseline,
                position_comparison,
                aligned_comparison,
                optimal,
                recommended,
                recommend_filter,
            ),
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_vwap_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> VwapValidationReport:
    """Run VWAP validation and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise VwapValidationError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = VwapValidationEngine(
        symbol=metadata.get("symbol", "NIFTY50"),
        research_days=metadata.get("research_window_days", RESEARCH_DAYS),
        timeframes=tuple(metadata.get("timeframes_analyzed", ("5M", "15M", "1H"))),
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info("VWAP validation completed: trades=%s", report.total_trades)
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_vwap_validation_report()
        print("VWAP Validation Summary")
        print(f"Filtered Production Stack Trades: {report.total_trades}")
        print(f"Filters: {report.production_stack['filters']}")
        print(f"VWAP improves profitability: {report.vwap_improves_profitability}")
        print(f"Recommend production filter: {report.recommend_production_filter}")
        for note in report.conclusions:
            print(note)
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except VwapValidationError as exc:
        logger.error("VWAP validation error: %s", exc)
        print(f"VWAP validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected VWAP validation failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
