"""
Liquidity sweep outcome validation research for SmartMoneyEngine.

Validates which detected liquidity sweeps create real forward moves and which
contextual characteristics predict success. Research-only; no signals or
production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.context.liquidity_narrative_engine import (
    DisplacementStrength,
    FvgContext,
    LiquidityNarrativeEngine,
)
from src.context.market_intelligence_engine import MarketIntelligenceEngine
from src.research.filter_research_engine import RESEARCH_DAYS, FilterResearchEngine, _json_safe
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "research" / "liquidity_sweep_outcome_validation.json"
)

MOVE_THRESHOLDS = (50, 100, 150, 200)
FORWARD_WINDOWS = (20, 40, 80)
STRUCTURE_LOOKBACK_BARS = 20
TOP_PATTERN_COUNT = 20
MIN_PATTERN_SAMPLES = 3


class LiquiditySweepOutcomeValidationError(Exception):
    """Raised when liquidity sweep outcome validation fails."""


class SweepOutcomeClass(str, Enum):
    """Outcome classification for one liquidity sweep."""

    FAILED = "Failed Sweep"
    SUCCESSFUL = "Successful Sweep"
    TREND_REVERSAL = "Trend Reversal Sweep"
    TREND_CONTINUATION = "Trend Continuation Sweep"
    FALSE_BREAKOUT = "False Breakout Sweep"


@dataclass(frozen=True)
class ForwardMoveMatrix:
    """Reachability of move thresholds within forward windows."""

    favorable_move_points: dict[str, float]
    adverse_move_points: dict[str, float]
    threshold_hits: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SweepOutcomeRecord:
    """One detected sweep with context and forward outcome."""

    sweep_timestamp: str
    timeframe: str
    sweep_type: str
    sweep_bar: int
    expected_direction: str
    sweep_quality_score: float
    sweep_classification: str
    displacement_strength: str
    choch_present: bool
    bos_present: bool
    fvg_created: bool
    fvg_reclaimed: bool
    market_location: str
    price_zone: str
    external_liquidity_distance: float | None
    internal_liquidity_distance: float | None
    close_back_into_range: bool
    outcome_classification: str
    max_favorable_move_80: float
    max_adverse_move_80: float
    forward_move_matrix: dict[str, Any]
    pattern_tags: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatternSuccessMetrics:
    """Success rate for one sweep pattern."""

    pattern: str
    sweep_count: int
    success_50_pct: float
    success_100_pct: float
    success_150_pct: float
    success_200_pct: float
    average_favorable_move_80: float
    rank_score: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiquiditySweepOutcomeValidationReport:
    """Full liquidity sweep outcome validation output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    move_thresholds_points: list[int]
    forward_windows_bars: list[int]
    total_sweeps: int
    buy_side_sweeps: int
    sell_side_sweeps: int
    outcome_classification_distribution: dict[str, int]
    forward_reachability_summary: dict[str, dict[str, Any]]
    characteristic_success_analysis: dict[str, dict[str, Any]]
    moves_over_50_characteristics: dict[str, dict[str, Any]]
    moves_over_100_characteristics: dict[str, dict[str, Any]]
    moves_over_150_characteristics: dict[str, dict[str, Any]]
    top_sweep_patterns: list[dict[str, Any]]
    sweep_records: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiquiditySweepOutcomeValidationResearch:
    """Validate liquidity sweep forward outcomes and predictive characteristics."""

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
        self.liquidity_map_engine = InstitutionalLiquidityMapEngine(symbol=symbol)
        self.narrative_engine = LiquidityNarrativeEngine(symbol=symbol)
        self.intelligence_engine = MarketIntelligenceEngine(symbol=symbol)

    @staticmethod
    def _is_active(value: Any) -> bool:
        return DecisionEngine._is_active(value)

    @staticmethod
    def _normalize_trend(value: Any) -> str:
        return str(value).strip().upper()

    def _expected_direction(self, sweep_type: str) -> str:
        if sweep_type == "Buy Side Sweep":
            return "bearish"
        return "bullish"

    def _forward_move_matrix(
        self,
        frame: pd.DataFrame,
        sweep_bar: int,
        direction: str,
    ) -> ForwardMoveMatrix:
        close = float(frame.iloc[sweep_bar]["Close"])
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)

        favorable_by_window: dict[str, float] = {}
        adverse_by_window: dict[str, float] = {}
        threshold_hits: dict[str, bool] = {}

        for window in FORWARD_WINDOWS:
            end = min(len(frame) - 1, sweep_bar + window)
            if sweep_bar >= end:
                favorable = 0.0
                adverse = 0.0
            elif direction == "bullish":
                favorable = round(float(highs.iloc[sweep_bar + 1 : end + 1].max()) - close, 2)
                adverse = round(close - float(lows.iloc[sweep_bar + 1 : end + 1].min()), 2)
            else:
                favorable = round(close - float(lows.iloc[sweep_bar + 1 : end + 1].min()), 2)
                adverse = round(float(highs.iloc[sweep_bar + 1 : end + 1].max()) - close, 2)

            favorable = max(favorable, 0.0)
            adverse = max(adverse, 0.0)
            favorable_by_window[str(window)] = favorable
            adverse_by_window[str(window)] = adverse

            for threshold in MOVE_THRESHOLDS:
                key = f"{threshold}pts_{window}bars"
                threshold_hits[key] = favorable >= threshold

        return ForwardMoveMatrix(
            favorable_move_points=favorable_by_window,
            adverse_move_points=adverse_by_window,
            threshold_hits=threshold_hits,
        )

    def _structure_in_window(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
        direction: str,
    ) -> tuple[bool, bool]:
        choch_column = "Bullish_CHOCH" if direction == "bullish" else "Bearish_CHOCH"
        bos_column = "Bullish_BOS" if direction == "bullish" else "Bearish_BOS"
        choch = any(
            self._is_active(frame.iloc[index].get(choch_column))
            for index in range(start_bar, end_bar + 1)
        )
        bos = any(
            self._is_active(frame.iloc[index].get(bos_column))
            for index in range(start_bar, end_bar + 1)
        )
        return choch, bos

    def _fvg_flags_in_window(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
        direction: str,
    ) -> tuple[bool, bool]:
        created = False
        reclaimed = False
        for index in range(start_bar, end_bar + 1):
            window = self.narrative_engine._window(frame, index)
            fvg_context, fvg_bias = self.narrative_engine._fvg_context_state(frame, index, window)
            if fvg_context == FvgContext.CREATED and (
                (direction == "bullish" and fvg_bias == "bullish")
                or (direction == "bearish" and fvg_bias == "bearish")
            ):
                created = True
            if fvg_context == FvgContext.RECLAIMED and (
                (direction == "bullish" and fvg_bias == "bullish")
                or (direction == "bearish" and fvg_bias == "bearish")
            ):
                reclaimed = True
        return created, reclaimed

    def _internal_liquidity_distance(
        self,
        close: float,
        internal: dict[str, Any],
        direction: str,
    ) -> float | None:
        if direction == "bullish":
            target = internal.get("active_buy_side_pool") or internal.get("range_high")
        else:
            target = internal.get("active_sell_side_pool") or internal.get("range_low")
        if target is None:
            return None
        return round(abs(close - float(target)), 2)

    def _classify_outcome(
        self,
        sweep_type: str,
        expected_direction: str,
        trend: str,
        close_back: bool,
        matrix: ForwardMoveMatrix,
        choch_present: bool,
        bos_present: bool,
    ) -> str:
        favorable_80 = matrix.favorable_move_points.get("80", 0.0)
        adverse_80 = matrix.adverse_move_points.get("80", 0.0)

        if not close_back or (adverse_80 >= 50 and favorable_80 < 50):
            return SweepOutcomeClass.FALSE_BREAKOUT.value

        if favorable_80 < MOVE_THRESHOLDS[0]:
            return SweepOutcomeClass.FAILED.value

        trend_bullish = trend == "BULLISH"
        trend_bearish = trend == "BEARISH"
        expected_bullish = expected_direction == "bullish"
        trend_opposed = (expected_bullish and trend_bearish) or (
            not expected_bullish and trend_bullish
        )
        trend_aligned = (expected_bullish and trend_bullish) or (
            not expected_bullish and trend_bearish
        )

        if choch_present and trend_opposed:
            return SweepOutcomeClass.TREND_REVERSAL.value
        if bos_present and trend_aligned:
            return SweepOutcomeClass.TREND_CONTINUATION.value
        if favorable_80 >= MOVE_THRESHOLDS[0]:
            return SweepOutcomeClass.SUCCESSFUL.value
        return SweepOutcomeClass.FAILED.value

    @staticmethod
    def _quality_bucket(score: float) -> str:
        if score >= 80:
            return "Quality 80+"
        if score >= 60:
            return "Quality 60-79"
        if score >= 40:
            return "Quality 40-59"
        return "Quality Below 40"

    def _build_pattern_tags(self, record: SweepOutcomeRecord) -> tuple[str, ...]:
        return (
            f"Sweep: {record.sweep_type}",
            f"Classification: {record.sweep_classification}",
            f"Outcome: {record.outcome_classification}",
            f"Displacement: {record.displacement_strength}",
            f"Quality: {self._quality_bucket(record.sweep_quality_score)}",
            f"CHOCH: {'Yes' if record.choch_present else 'No'}",
            f"BOS: {'Yes' if record.bos_present else 'No'}",
            f"FVG Created: {'Yes' if record.fvg_created else 'No'}",
            f"FVG Reclaimed: {'Yes' if record.fvg_reclaimed else 'No'}",
            f"Location: {record.market_location}",
            f"Zone: {record.price_zone}",
            f"Close Back: {'Yes' if record.close_back_into_range else 'No'}",
        )

    def _analyze_sweep(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        intel_frame: pd.DataFrame,
        sweep_bar: int,
        sweep_type: str,
        timeframe_label: str,
    ) -> SweepOutcomeRecord | None:
        if sweep_bar >= len(frame) - 1:
            return None

        expected_direction = self._expected_direction(sweep_type)
        candle_map = self.liquidity_map_engine.evaluate_bar(frame, enriched, sweep_bar)
        event = candle_map.liquidity_event
        internal = candle_map.internal_liquidity
        external = candle_map.external_liquidity
        close = float(frame.iloc[sweep_bar]["Close"])
        trend = self._normalize_trend(frame.iloc[sweep_bar].get("Trend", "SIDEWAYS"))

        end_bar = min(len(frame) - 1, sweep_bar + FORWARD_WINDOWS[-1])
        choch_present, bos_present = self._structure_in_window(
            frame,
            sweep_bar,
            end_bar,
            expected_direction,
        )
        fvg_created, fvg_reclaimed = self._fvg_flags_in_window(
            frame,
            sweep_bar,
            end_bar,
            expected_direction,
        )

        intelligence = self.intelligence_engine.evaluate_bar(intel_frame, sweep_bar)
        matrix = self._forward_move_matrix(frame, sweep_bar, expected_direction)
        close_back = bool(event.get("close_back_into_range", False))
        outcome_class = self._classify_outcome(
            sweep_type,
            expected_direction,
            trend,
            close_back,
            matrix,
            choch_present,
            bos_present,
        )

        draft = SweepOutcomeRecord(
            sweep_timestamp=str(frame.iloc[sweep_bar]["Date"]),
            timeframe=timeframe_label,
            sweep_type=sweep_type,
            sweep_bar=sweep_bar,
            expected_direction=expected_direction,
            sweep_quality_score=candle_map.sweep_quality_score,
            sweep_classification=str(event.get("classification", "No Sweep")),
            displacement_strength=str(event.get("displacement_after_sweep", "None")),
            choch_present=choch_present,
            bos_present=bos_present,
            fvg_created=fvg_created,
            fvg_reclaimed=fvg_reclaimed,
            market_location=intelligence.market_location,
            price_zone=str(internal.get("price_zone", "Unknown")),
            external_liquidity_distance=external.get("distance_to_nearest_external"),
            internal_liquidity_distance=self._internal_liquidity_distance(
                close,
                internal,
                expected_direction,
            ),
            close_back_into_range=close_back,
            outcome_classification=outcome_class,
            max_favorable_move_80=matrix.favorable_move_points.get("80", 0.0),
            max_adverse_move_80=matrix.adverse_move_points.get("80", 0.0),
            forward_move_matrix=matrix.as_dict(),
            pattern_tags=(),
        )
        tags = self._build_pattern_tags(draft)
        return SweepOutcomeRecord(**{**draft.as_dict(), "pattern_tags": tags})

    def _collect_sweeps(self, metadata: dict[str, Any]) -> list[SweepOutcomeRecord]:
        from datetime import timedelta

        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        records: list[SweepOutcomeRecord] = []
        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)
            enriched = self.liquidity_map_engine._attach_calendar_levels(frame)
            intel_frame = self.intelligence_engine.enrich(frame)

            for index in range(len(frame)):
                row = frame.iloc[index]
                if self._is_active(row.get("Buy_Liquidity_Sweep")):
                    record = self._analyze_sweep(
                        frame,
                        enriched,
                        intel_frame,
                        index,
                        "Buy Side Sweep",
                        timeframe_label,
                    )
                    if record is not None:
                        records.append(record)
                if self._is_active(row.get("Sell_Liquidity_Sweep")):
                    record = self._analyze_sweep(
                        frame,
                        enriched,
                        intel_frame,
                        index,
                        "Sell Side Sweep",
                        timeframe_label,
                    )
                    if record is not None:
                        records.append(record)

        records.sort(key=lambda item: pd.Timestamp(item.sweep_timestamp))
        return records

    @staticmethod
    def _hit_rate(records: list[SweepOutcomeRecord], threshold: int, window: int) -> float:
        if not records:
            return 0.0
        key = f"{threshold}pts_{window}bars"
        hits = sum(1 for record in records if record.forward_move_matrix["threshold_hits"].get(key))
        return round(hits / len(records) * 100, 2)

    def _forward_reachability_summary(
        self,
        records: list[SweepOutcomeRecord],
    ) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for threshold in MOVE_THRESHOLDS:
            threshold_key = f"{threshold}_points"
            summary[threshold_key] = {
                str(window): {
                    "hit_count": sum(
                        1
                        for record in records
                        if record.forward_move_matrix["threshold_hits"].get(
                            f"{threshold}pts_{window}bars",
                        )
                    ),
                    "hit_rate_pct": self._hit_rate(records, threshold, window),
                }
                for window in FORWARD_WINDOWS
            }
        return summary

    @staticmethod
    def _success_for_threshold(
        record: SweepOutcomeRecord,
        threshold: int,
        window: int = 80,
    ) -> bool:
        return bool(
            record.forward_move_matrix["threshold_hits"].get(f"{threshold}pts_{window}bars"),
        )

    def _characteristic_analysis(
        self,
        records: list[SweepOutcomeRecord],
        threshold: int,
    ) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[SweepOutcomeRecord]] = {
            "all_sweeps": records,
            "buy_side_sweep": [r for r in records if r.sweep_type == "Buy Side Sweep"],
            "sell_side_sweep": [r for r in records if r.sweep_type == "Sell Side Sweep"],
            "quality_80_plus": [r for r in records if r.sweep_quality_score >= 80],
            "quality_60_79": [r for r in records if 60 <= r.sweep_quality_score < 80],
            "quality_below_60": [r for r in records if r.sweep_quality_score < 60],
            "displacement_strong": [
                r for r in records if r.displacement_strength == DisplacementStrength.STRONG.value
            ],
            "displacement_medium_plus": [
                r
                for r in records
                if r.displacement_strength
                in {DisplacementStrength.STRONG.value, DisplacementStrength.MEDIUM.value}
            ],
            "choch_present": [r for r in records if r.choch_present],
            "bos_present": [r for r in records if r.bos_present],
            "fvg_created": [r for r in records if r.fvg_created],
            "fvg_reclaimed": [r for r in records if r.fvg_reclaimed],
            "close_back_yes": [r for r in records if r.close_back_into_range],
            "premium_zone": [r for r in records if r.price_zone == "Premium Zone"],
            "discount_zone": [r for r in records if r.price_zone == "Discount Zone"],
            "near_support": [r for r in records if r.market_location == "Near Support"],
            "near_resistance": [r for r in records if r.market_location == "Near Resistance"],
            "external_near_20": [
                r
                for r in records
                if r.external_liquidity_distance is not None and r.external_liquidity_distance <= 20
            ],
            "internal_near_20": [
                r
                for r in records
                if r.internal_liquidity_distance is not None and r.internal_liquidity_distance <= 20
            ],
            "institutional_classification": [
                r for r in records if r.sweep_classification == "Institutional Sweep"
            ],
            "strong_classification": [
                r for r in records if r.sweep_classification == "Strong Sweep"
            ],
        }

        analysis: dict[str, dict[str, Any]] = {}
        for label, group in groups.items():
            if not group:
                continue
            successes = sum(1 for record in group if self._success_for_threshold(record, threshold))
            analysis[label] = {
                "sweep_count": len(group),
                "success_count": successes,
                "success_rate_pct": round(successes / len(group) * 100, 2),
                "average_favorable_move_80": round(
                    mean(record.max_favorable_move_80 for record in group),
                    2,
                ),
            }
        return analysis

    def _pattern_metrics(self, records: list[SweepOutcomeRecord]) -> list[PatternSuccessMetrics]:
        pattern_groups: dict[str, list[SweepOutcomeRecord]] = defaultdict(list)
        for record in records:
            for tag in record.pattern_tags:
                if tag.startswith(("Sweep:", "Quality:", "Displacement:", "CHOCH:", "BOS:", "Zone:", "Location:")):
                    pattern_groups[tag].append(record)

        metrics: list[PatternSuccessMetrics] = []
        for pattern, group in pattern_groups.items():
            if len(group) < MIN_PATTERN_SAMPLES:
                continue
            success_50 = self._hit_rate(group, 50, 80)
            success_100 = self._hit_rate(group, 100, 80)
            success_150 = self._hit_rate(group, 150, 80)
            success_200 = self._hit_rate(group, 200, 80)
            avg_move = round(mean(record.max_favorable_move_80 for record in group), 2)
            rank_score = round(
                success_50 * 0.35 + success_100 * 0.30 + success_150 * 0.20 + success_200 * 0.15,
                2,
            )
            metrics.append(
                PatternSuccessMetrics(
                    pattern=pattern,
                    sweep_count=len(group),
                    success_50_pct=success_50,
                    success_100_pct=success_100,
                    success_150_pct=success_150,
                    success_200_pct=success_200,
                    average_favorable_move_80=avg_move,
                    rank_score=rank_score,
                )
            )

        ranked = sorted(metrics, key=lambda item: (item.rank_score, item.success_100_pct), reverse=True)
        for index, item in enumerate(ranked, start=1):
            item.rank = index
        return ranked

    def run(self, metadata: dict[str, Any]) -> LiquiditySweepOutcomeValidationReport:
        """Run liquidity sweep outcome validation research."""
        started = time.perf_counter()
        records = self._collect_sweeps(metadata)
        if not records:
            raise LiquiditySweepOutcomeValidationError("No liquidity sweeps detected.")

        outcome_distribution = Counter(record.outcome_classification for record in records)
        pattern_metrics = self._pattern_metrics(records)
        top_patterns = pattern_metrics[:TOP_PATTERN_COUNT]

        char_50 = self._characteristic_analysis(records, 50)
        char_100 = self._characteristic_analysis(records, 100)
        char_150 = self._characteristic_analysis(records, 150)

        best_pattern = top_patterns[0] if top_patterns else None
        successful = [r for r in records if r.outcome_classification != SweepOutcomeClass.FAILED.value]

        conclusions = [
            f"Validated {len(records)} liquidity sweeps across {len(self.timeframes)} timeframes.",
            (
                f"Buy-side: {sum(1 for r in records if r.sweep_type == 'Buy Side Sweep')}; "
                f"sell-side: {sum(1 for r in records if r.sweep_type == 'Sell Side Sweep')}."
            ),
            (
                f"50+ point moves within 80 bars: "
                f"{self._hit_rate(records, 50, 80)}% | "
                f"100+: {self._hit_rate(records, 100, 80)}% | "
                f"150+: {self._hit_rate(records, 150, 80)}%."
            ),
        ]
        if best_pattern:
            conclusions.append(
                f"Top sweep pattern: {best_pattern.pattern} "
                f"(50+ success {best_pattern.success_50_pct}%, n={best_pattern.sweep_count})."
            )
        if char_100:
            leader = max(
                ((label, data) for label, data in char_100.items() if data["sweep_count"] >= MIN_PATTERN_SAMPLES),
                key=lambda item: item[1]["success_rate_pct"],
                default=None,
            )
            if leader:
                conclusions.append(
                    f"Best 100+ move characteristic: {leader[0]} "
                    f"({leader[1]['success_rate_pct']}% success, n={leader[1]['sweep_count']})."
                )
        conclusions.append(
            f"Non-failed sweeps: {len(successful)} ({round(len(successful)/len(records)*100, 1)}%)."
        )

        return LiquiditySweepOutcomeValidationReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            move_thresholds_points=list(MOVE_THRESHOLDS),
            forward_windows_bars=list(FORWARD_WINDOWS),
            total_sweeps=len(records),
            buy_side_sweeps=sum(1 for record in records if record.sweep_type == "Buy Side Sweep"),
            sell_side_sweeps=sum(1 for record in records if record.sweep_type == "Sell Side Sweep"),
            outcome_classification_distribution=dict(sorted(outcome_distribution.items())),
            forward_reachability_summary=self._forward_reachability_summary(records),
            characteristic_success_analysis=char_50,
            moves_over_50_characteristics=char_50,
            moves_over_100_characteristics=char_100,
            moves_over_150_characteristics=char_150,
            top_sweep_patterns=[item.as_dict() for item in top_patterns],
            sweep_records=[record.as_dict() for record in records],
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_liquidity_sweep_outcome_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> LiquiditySweepOutcomeValidationReport:
    """Run liquidity sweep outcome validation and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise LiquiditySweepOutcomeValidationError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = LiquiditySweepOutcomeValidationResearch(
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
        "Liquidity sweep outcome validation completed: sweeps=%s",
        report.total_sweeps,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_liquidity_sweep_outcome_validation_report()
        print("Liquidity Sweep Outcome Validation Summary")
        print(f"Total sweeps: {report.total_sweeps}")
        print("Outcome distribution:")
        for label, count in report.outcome_classification_distribution.items():
            print(f"  {label}: {count}")
        print("50+ reachability (80 bars):")
        reach = report.forward_reachability_summary.get("50_points", {})
        for window, data in reach.items():
            print(f"  {window} bars: {data['hit_rate_pct']}%")
        if report.top_sweep_patterns:
            top = report.top_sweep_patterns[0]
            print(f"Top pattern: {top['pattern']} (50+={top['success_50_pct']}%)")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except LiquiditySweepOutcomeValidationError as exc:
        logger.error("Liquidity sweep outcome validation error: %s", exc)
        print(f"Liquidity sweep outcome validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected liquidity sweep outcome validation failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
