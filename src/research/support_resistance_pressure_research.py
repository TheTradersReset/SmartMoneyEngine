"""
Support / Resistance Pressure research for SmartMoneyEngine.

Determines how the market behaves before major support or resistance breaks,
bounces, and rejections. Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.liquidity_narrative_engine import DisplacementStrength, LiquidityNarrativeEngine
from src.research.filter_research_engine import RESEARCH_DAYS, FilterResearchEngine, _json_safe
from src.research.liquidity_move_reconstruction_research import FORWARD_BARS
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "support_resistance_pressure.json"

LEVEL_TOUCH_ATR_RATIO = 0.5
BREAK_BUFFER_ATR_RATIO = 0.15
MIN_BOUNCE_POINTS = 30
LEVEL_CLUSTER_POINTS = 12.0
LEVEL_MAX_AGE_BARS = 400
BOUNCE_LOOKAHEAD_BARS = 25
TOP_PATTERN_COUNT = 20
MIN_PATTERN_SAMPLES = 3
STRONG_BODY_RATIO = 0.65
GAP_MIN_POINTS = 0.5
WICK_REJECTION_RATIO = 0.55

LEVEL_SOURCE_COLUMNS: dict[str, str] = {
    "Swing_High": "resistance",
    "Swing_Low": "support",
    "Equal_High": "resistance",
    "Equal_Low": "support",
    "Buy_Side_Liquidity": "resistance",
    "Sell_Side_Liquidity": "support",
}


class SupportResistancePressureError(Exception):
    """Raised when support/resistance pressure research fails."""


class LevelState(str, Enum):
    """Lifecycle state for a major support/resistance level."""

    FRESH = "Fresh"
    RETESTED = "Retested"
    EXHAUSTED = "Exhausted"
    BROKEN = "Broken"


class LevelOutcome(str, Enum):
    """Terminal interaction outcome for one level."""

    SUPPORT_BREAK = "support_break"
    RESISTANCE_BREAK = "resistance_break"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_REJECTION = "resistance_rejection"


@dataclass(frozen=True)
class MajorLevel:
    """One major support or resistance reference level."""

    level_price: float
    level_side: str
    source_column: str
    formation_bar: int
    formation_timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LevelPressureRecord:
    """Pressure metrics and outcome for one level interaction cycle."""

    timeframe: str
    level_price: float
    level_side: str
    level_source: str
    level_state: str
    outcome: str
    formation_bar: int
    event_bar: int
    formation_timestamp: str
    event_timestamp: str
    number_of_tests: int
    bars_near_level: int
    failed_breakouts: int
    failed_breakdowns: int
    liquidity_grabs: int
    wick_rejection_count: int
    strong_body_candle_count: int
    gap_up_near_resistance: int
    gap_down_near_support: int
    fake_breaks: int
    stop_hunts: int
    distance_traveled_after_event: float
    confirmation_candle: dict[str, Any] | None
    pattern_key: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatternRankMetrics:
    """Ranked pattern frequency for one outcome category."""

    pattern: str
    outcome: str
    sample_count: int
    frequency_pct: float
    average_tests_before_event: float
    average_bars_near_level: float
    average_fake_breaks: float
    average_stop_hunts: float
    average_distance_traveled: float
    reliability_score: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SupportResistancePressureReport:
    """Aggregate support/resistance pressure research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    level_classification_summary: dict[str, int]
    outcome_counts: dict[str, int]
    aggregate_level_metrics: dict[str, dict[str, Any]]
    top_20_support_break_patterns: list[dict[str, Any]]
    top_20_resistance_break_patterns: list[dict[str, Any]]
    top_20_support_bounce_patterns: list[dict[str, Any]]
    top_20_resistance_rejection_patterns: list[dict[str, Any]]
    level_records: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SupportResistancePressureResearch:
    """Analyze pressure behavior around major support and resistance levels."""

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

    @staticmethod
    def _is_active(value: Any) -> bool:
        return DecisionEngine._is_active(value)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _atr(frame: pd.DataFrame, index: int, period: int = 14) -> float:
        start = max(0, index - period)
        window = frame.iloc[start : index + 1]
        if len(window) < 2:
            return 1.0
        highs = window["High"].astype(float)
        lows = window["Low"].astype(float)
        closes = window["Close"].astype(float)
        tr_values = []
        for offset in range(1, len(window)):
            high = float(highs.iloc[offset])
            low = float(lows.iloc[offset])
            prev_close = float(closes.iloc[offset - 1])
            tr_values.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        return max(mean(tr_values), 1.0) if tr_values else 1.0

    @staticmethod
    def _candle_parts(row: pd.Series) -> dict[str, float]:
        open_price = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        body = abs(close - open_price)
        candle_range = max(high - low, 0.01)
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        return {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "body": body,
            "range": candle_range,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "body_pct": body / candle_range,
            "bullish": close > open_price,
            "bearish": close < open_price,
        }

    @staticmethod
    def _level_state_from_tests(tests: int, broken: bool) -> str:
        if broken:
            return LevelState.BROKEN.value
        if tests <= 1:
            return LevelState.FRESH.value
        if tests <= 4:
            return LevelState.RETESTED.value
        return LevelState.EXHAUSTED.value

    def _confirmation_candle(
        self,
        frame: pd.DataFrame,
        bar: int,
        level_side: str,
        broke: bool,
    ) -> dict[str, Any]:
        row = frame.iloc[bar]
        parts = self._candle_parts(row)
        direction = "bearish" if level_side == "support" and broke else "bullish"
        if level_side == "resistance" and not broke:
            direction = "bearish"
        if level_side == "support" and not broke:
            direction = "bullish"

        displacement = LiquidityNarrativeEngine._displacement_strength_for_bar(row, direction)
        volume = self._to_float(row.get("Volume")) or 0.0
        vol_start = max(0, bar - 20)
        avg_volume = mean(
            self._to_float(frame.iloc[index].get("Volume")) or 0.0
            for index in range(vol_start, bar)
        ) if bar > vol_start else volume

        return {
            "timestamp": str(row["Date"]),
            "body_pct": round(parts["body_pct"] * 100, 2),
            "wick_pct": round((parts["upper_wick"] + parts["lower_wick"]) / parts["range"] * 100, 2),
            "volume_expansion_ratio": round(volume / avg_volume, 2) if avg_volume > 0 else 1.0,
            "close_strength_pct": round(
                (parts["close"] - parts["low"]) / parts["range"] * 100,
                2,
            ),
            "displacement_strength": displacement.value,
            "strong_body": parts["body_pct"] >= STRONG_BODY_RATIO,
        }

    def _forward_distance(
        self,
        frame: pd.DataFrame,
        event_bar: int,
        direction: str,
    ) -> float:
        end = min(len(frame) - 1, event_bar + FORWARD_BARS)
        if event_bar >= end:
            return 0.0
        origin = float(frame.iloc[event_bar]["Close"])
        if direction == "bullish":
            move = float(frame.iloc[event_bar + 1 : end + 1]["High"].astype(float).max()) - origin
        else:
            move = origin - float(frame.iloc[event_bar + 1 : end + 1]["Low"].astype(float).min())
        return round(max(move, 0.0), 2)

    def _cluster_levels(self, levels: list[MajorLevel]) -> list[MajorLevel]:
        if not levels:
            return []
        ranked = sorted(levels, key=lambda item: item.formation_bar)
        kept: list[MajorLevel] = []
        for level in ranked:
            merged = False
            for index, existing in enumerate(kept):
                if (
                    existing.level_side == level.level_side
                    and abs(existing.level_price - level.level_price) <= LEVEL_CLUSTER_POINTS
                ):
                    if level.formation_bar < existing.formation_bar:
                        kept[index] = level
                    merged = True
                    break
            if not merged:
                kept.append(level)
        return kept

    def _extract_levels(self, frame: pd.DataFrame) -> list[MajorLevel]:
        levels: list[MajorLevel] = []
        for index in range(len(frame)):
            row = frame.iloc[index]
            timestamp = str(row["Date"])
            for column, side in LEVEL_SOURCE_COLUMNS.items():
                price = self._to_float(row.get(column))
                if price is not None and self._is_active(row.get(column)):
                    levels.append(
                        MajorLevel(
                            level_price=round(price, 2),
                            level_side=side,
                            source_column=column,
                            formation_bar=index,
                            formation_timestamp=timestamp,
                        ),
                    )
        return self._cluster_levels(levels)

    def _build_pattern_key(
        self,
        record: LevelPressureRecord,
    ) -> str:
        confirm = record.confirmation_candle or {}
        confirm_label = confirm.get("displacement_strength", "None")
        if confirm.get("strong_body"):
            confirm_label = f"Strong-{confirm_label}"

        return (
            f"{record.level_state} | Source:{record.level_source} | "
            f"Tests:{record.number_of_tests} | Fake:{record.fake_breaks} | "
            f"Hunts:{record.stop_hunts} | Wicks:{record.wick_rejection_count} | "
            f"Liq:{record.liquidity_grabs} | Confirm:{confirm_label}"
        )

    def _evaluate_level(
        self,
        frame: pd.DataFrame,
        level: MajorLevel,
        timeframe_label: str,
    ) -> list[LevelPressureRecord]:
        records: list[LevelPressureRecord] = []
        start = level.formation_bar + 1
        end = min(len(frame) - 1, level.formation_bar + LEVEL_MAX_AGE_BARS)

        tests = 0
        bars_near = 0
        failed_breakouts = 0
        failed_breakdowns = 0
        liquidity_grabs = 0
        wick_rejections = 0
        strong_bodies = 0
        gap_ups = 0
        gap_downs = 0
        fake_breaks = 0
        stop_hunts = 0
        terminal = False
        was_near = False

        for index in range(start, end + 1):
            if terminal:
                break

            row = frame.iloc[index]
            atr = self._atr(frame, index)
            touch_band = atr * LEVEL_TOUCH_ATR_RATIO
            break_buffer = atr * BREAK_BUFFER_ATR_RATIO
            parts = self._candle_parts(row)
            close = parts["close"]
            high = parts["high"]
            low = parts["low"]
            near_level = abs(close - level.level_price) <= touch_band or (
                low - touch_band <= level.level_price <= high + touch_band
            )

            if index > 0:
                prev_close = float(frame.iloc[index - 1]["Close"])
                open_price = parts["open"]
                gap = open_price - prev_close
                if abs(gap) >= GAP_MIN_POINTS and near_level:
                    if gap > 0 and level.level_side == "resistance":
                        gap_ups += 1
                    if gap < 0 and level.level_side == "support":
                        gap_downs += 1

            if near_level:
                bars_near += 1
                if not was_near:
                    tests += 1
                if parts["body_pct"] >= STRONG_BODY_RATIO:
                    strong_bodies += 1

                if level.level_side == "resistance":
                    if high > level.level_price + break_buffer and close <= level.level_price:
                        failed_breakouts += 1
                        fake_breaks += 1
                    upper_wick = parts["upper_wick"]
                    if upper_wick / parts["range"] >= WICK_REJECTION_RATIO and high >= level.level_price:
                        wick_rejections += 1
                else:
                    if low < level.level_price - break_buffer and close >= level.level_price:
                        failed_breakdowns += 1
                        fake_breaks += 1
                    lower_wick = parts["lower_wick"]
                    if lower_wick / parts["range"] >= WICK_REJECTION_RATIO and low <= level.level_price:
                        wick_rejections += 1

                if self._is_active(row.get("Buy_Liquidity_Sweep")) or self._is_active(
                    row.get("Sell_Liquidity_Sweep"),
                ):
                    liquidity_grabs += 1
                    if level.level_side == "resistance" and high > level.level_price and close < level.level_price:
                        stop_hunts += 1
                    if level.level_side == "support" and low < level.level_price and close > level.level_price:
                        stop_hunts += 1

            was_near = near_level

            broke_support = level.level_side == "support" and close < level.level_price - break_buffer
            broke_resistance = level.level_side == "resistance" and close > level.level_price + break_buffer

            if broke_support and tests >= 1:
                outcome = LevelOutcome.SUPPORT_BREAK.value
                direction = "bearish"
                confirm = self._confirmation_candle(frame, index, level.level_side, broke=True)
                state = self._level_state_from_tests(tests, broken=False)
                record = LevelPressureRecord(
                    timeframe=timeframe_label,
                    level_price=level.level_price,
                    level_side=level.level_side,
                    level_source=level.source_column,
                    level_state=state,
                    outcome=outcome,
                    formation_bar=level.formation_bar,
                    event_bar=index,
                    formation_timestamp=level.formation_timestamp,
                    event_timestamp=str(row["Date"]),
                    number_of_tests=tests,
                    bars_near_level=bars_near,
                    failed_breakouts=failed_breakouts,
                    failed_breakdowns=failed_breakdowns,
                    liquidity_grabs=liquidity_grabs,
                    wick_rejection_count=wick_rejections,
                    strong_body_candle_count=strong_bodies,
                    gap_up_near_resistance=gap_ups,
                    gap_down_near_support=gap_downs,
                    fake_breaks=fake_breaks,
                    stop_hunts=stop_hunts,
                    distance_traveled_after_event=self._forward_distance(frame, index, direction),
                    confirmation_candle=confirm,
                    pattern_key="",
                )
                record = LevelPressureRecord(
                    **{**record.as_dict(), "pattern_key": self._build_pattern_key(record)},
                )
                records.append(record)
                terminal = True
                continue

            if broke_resistance and tests >= 1:
                outcome = LevelOutcome.RESISTANCE_BREAK.value
                direction = "bullish"
                confirm = self._confirmation_candle(frame, index, level.level_side, broke=True)
                state = self._level_state_from_tests(tests, broken=False)
                record = LevelPressureRecord(
                    timeframe=timeframe_label,
                    level_price=level.level_price,
                    level_side=level.level_side,
                    level_source=level.source_column,
                    level_state=state,
                    outcome=outcome,
                    formation_bar=level.formation_bar,
                    event_bar=index,
                    formation_timestamp=level.formation_timestamp,
                    event_timestamp=str(row["Date"]),
                    number_of_tests=tests,
                    bars_near_level=bars_near,
                    failed_breakouts=failed_breakouts,
                    failed_breakdowns=failed_breakdowns,
                    liquidity_grabs=liquidity_grabs,
                    wick_rejection_count=wick_rejections,
                    strong_body_candle_count=strong_bodies,
                    gap_up_near_resistance=gap_ups,
                    gap_down_near_support=gap_downs,
                    fake_breaks=fake_breaks,
                    stop_hunts=stop_hunts,
                    distance_traveled_after_event=self._forward_distance(frame, index, direction),
                    confirmation_candle=confirm,
                    pattern_key="",
                )
                record = LevelPressureRecord(
                    **{**record.as_dict(), "pattern_key": self._build_pattern_key(record)},
                )
                records.append(record)
                terminal = True
                continue

            if tests >= 2 and near_level and index + BOUNCE_LOOKAHEAD_BARS < len(frame):
                lookahead_end = min(len(frame) - 1, index + BOUNCE_LOOKAHEAD_BARS)
                future = frame.iloc[index + 1 : lookahead_end + 1]
                if level.level_side == "support":
                    bounce = (
                        float(future["High"].astype(float).max()) - close >= MIN_BOUNCE_POINTS
                        and float(future["Close"].astype(float).min()) >= level.level_price - break_buffer
                    )
                    if bounce:
                        confirm = self._confirmation_candle(frame, index, level.level_side, broke=False)
                        state = self._level_state_from_tests(tests, broken=False)
                        record = LevelPressureRecord(
                            timeframe=timeframe_label,
                            level_price=level.level_price,
                            level_side=level.level_side,
                            level_source=level.source_column,
                            level_state=state,
                            outcome=LevelOutcome.SUPPORT_BOUNCE.value,
                            formation_bar=level.formation_bar,
                            event_bar=index,
                            formation_timestamp=level.formation_timestamp,
                            event_timestamp=str(row["Date"]),
                            number_of_tests=tests,
                            bars_near_level=bars_near,
                            failed_breakouts=failed_breakouts,
                            failed_breakdowns=failed_breakdowns,
                            liquidity_grabs=liquidity_grabs,
                            wick_rejection_count=wick_rejections,
                            strong_body_candle_count=strong_bodies,
                            gap_up_near_resistance=gap_ups,
                            gap_down_near_support=gap_downs,
                            fake_breaks=fake_breaks,
                            stop_hunts=stop_hunts,
                            distance_traveled_after_event=self._forward_distance(frame, index, "bullish"),
                            confirmation_candle=confirm,
                            pattern_key="",
                        )
                        record = LevelPressureRecord(
                            **{**record.as_dict(), "pattern_key": self._build_pattern_key(record)},
                        )
                        records.append(record)
                        terminal = True
                        continue

                if level.level_side == "resistance":
                    rejection = (
                        close - float(future["Low"].astype(float).min()) >= MIN_BOUNCE_POINTS
                        and float(future["Close"].astype(float).max()) <= level.level_price + break_buffer
                    )
                    if rejection:
                        confirm = self._confirmation_candle(frame, index, level.level_side, broke=False)
                        state = self._level_state_from_tests(tests, broken=False)
                        record = LevelPressureRecord(
                            timeframe=timeframe_label,
                            level_price=level.level_price,
                            level_side=level.level_side,
                            level_source=level.source_column,
                            level_state=state,
                            outcome=LevelOutcome.RESISTANCE_REJECTION.value,
                            formation_bar=level.formation_bar,
                            event_bar=index,
                            formation_timestamp=level.formation_timestamp,
                            event_timestamp=str(row["Date"]),
                            number_of_tests=tests,
                            bars_near_level=bars_near,
                            failed_breakouts=failed_breakouts,
                            failed_breakdowns=failed_breakdowns,
                            liquidity_grabs=liquidity_grabs,
                            wick_rejection_count=wick_rejections,
                            strong_body_candle_count=strong_bodies,
                            gap_up_near_resistance=gap_ups,
                            gap_down_near_support=gap_downs,
                            fake_breaks=fake_breaks,
                            stop_hunts=stop_hunts,
                            distance_traveled_after_event=self._forward_distance(frame, index, "bearish"),
                            confirmation_candle=confirm,
                            pattern_key="",
                        )
                        record = LevelPressureRecord(
                            **{**record.as_dict(), "pattern_key": self._build_pattern_key(record)},
                        )
                        records.append(record)
                        terminal = True

        return records

    def _collect_records(self, metadata: dict[str, Any]) -> list[LevelPressureRecord]:
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        records: list[LevelPressureRecord] = []
        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)
            levels = self._extract_levels(frame)
            logger.info(
                "S/R pressure: %s levels=%s",
                timeframe_label,
                len(levels),
            )

            seen_events: set[tuple[int, float, str]] = set()
            for level in levels:
                for record in self._evaluate_level(frame, level, timeframe_label):
                    event_key = (record.event_bar, record.level_price, record.outcome)
                    if event_key in seen_events:
                        continue
                    seen_events.add(event_key)
                    records.append(record)

        return records

    @staticmethod
    def _aggregate_metrics(records: list[LevelPressureRecord]) -> dict[str, Any]:
        if not records:
            return {}

        def avg(values: list[float]) -> float:
            return round(mean(values), 2) if values else 0.0

        return {
            "sample_count": len(records),
            "average_tests": avg([float(item.number_of_tests) for item in records]),
            "average_bars_near_level": avg([float(item.bars_near_level) for item in records]),
            "average_failed_breakouts": avg([float(item.failed_breakouts) for item in records]),
            "average_failed_breakdowns": avg([float(item.failed_breakdowns) for item in records]),
            "average_liquidity_grabs": avg([float(item.liquidity_grabs) for item in records]),
            "average_wick_rejections": avg([float(item.wick_rejection_count) for item in records]),
            "average_strong_body_candles": avg([float(item.strong_body_candle_count) for item in records]),
            "average_gap_ups_near_resistance": avg([float(item.gap_up_near_resistance) for item in records]),
            "average_gap_downs_near_support": avg([float(item.gap_down_near_support) for item in records]),
            "average_fake_breaks": avg([float(item.fake_breaks) for item in records]),
            "average_stop_hunts": avg([float(item.stop_hunts) for item in records]),
            "average_distance_traveled": avg([item.distance_traveled_after_event for item in records]),
        }

    def _rank_patterns(
        self,
        records: list[LevelPressureRecord],
        outcome: str,
    ) -> list[PatternRankMetrics]:
        bucket = [record for record in records if record.outcome == outcome]
        if not bucket:
            return []

        grouped: dict[str, list[LevelPressureRecord]] = defaultdict(list)
        for record in bucket:
            grouped[record.pattern_key].append(record)

        total = len(bucket)
        metrics: list[PatternRankMetrics] = []
        for pattern, items in grouped.items():
            avg_tests = mean(item.number_of_tests for item in items)
            avg_bars = mean(item.bars_near_level for item in items)
            avg_fake = mean(item.fake_breaks for item in items)
            avg_hunts = mean(item.stop_hunts for item in items)
            avg_distance = mean(item.distance_traveled_after_event for item in items)
            reliability = (len(items) / total) * (avg_distance / 50.0 if avg_distance else 1.0) * 100
            metrics.append(
                PatternRankMetrics(
                    pattern=pattern,
                    outcome=outcome,
                    sample_count=len(items),
                    frequency_pct=round((len(items) / total) * 100, 2),
                    average_tests_before_event=round(avg_tests, 2),
                    average_bars_near_level=round(avg_bars, 2),
                    average_fake_breaks=round(avg_fake, 2),
                    average_stop_hunts=round(avg_hunts, 2),
                    average_distance_traveled=round(avg_distance, 2),
                    reliability_score=round(reliability, 4),
                ),
            )

        ranked = sorted(
            metrics,
            key=lambda item: (item.sample_count, item.reliability_score),
            reverse=True,
        )
        for index, item in enumerate(ranked, start=1):
            item.rank = index
        return ranked

    def run(self, metadata: dict[str, Any]) -> SupportResistancePressureReport:
        started = time.perf_counter()
        records = self._collect_records(metadata)

        outcome_counts = Counter({outcome.value: 0 for outcome in LevelOutcome})
        for record in records:
            outcome_counts[record.outcome] += 1

        state_counts: Counter[str] = Counter(
            {state.value: 0 for state in LevelState if state != LevelState.BROKEN},
        )
        for record in records:
            state_counts[record.level_state] += 1

        level_classification_summary = dict(state_counts)
        level_classification_summary[LevelState.BROKEN.value] = (
            outcome_counts[LevelOutcome.SUPPORT_BREAK.value]
            + outcome_counts[LevelOutcome.RESISTANCE_BREAK.value]
        )

        aggregate_by_outcome = {
            outcome.value: self._aggregate_metrics([r for r in records if r.outcome == outcome.value])
            for outcome in LevelOutcome
        }

        top_support_break = [
            item.as_dict()
            for item in self._rank_patterns(records, LevelOutcome.SUPPORT_BREAK.value)[:TOP_PATTERN_COUNT]
            if item.sample_count >= MIN_PATTERN_SAMPLES
        ]
        top_resistance_break = [
            item.as_dict()
            for item in self._rank_patterns(records, LevelOutcome.RESISTANCE_BREAK.value)[:TOP_PATTERN_COUNT]
            if item.sample_count >= MIN_PATTERN_SAMPLES
        ]
        top_support_bounce = [
            item.as_dict()
            for item in self._rank_patterns(records, LevelOutcome.SUPPORT_BOUNCE.value)[:TOP_PATTERN_COUNT]
            if item.sample_count >= MIN_PATTERN_SAMPLES
        ]
        top_resistance_rejection = [
            item.as_dict()
            for item in self._rank_patterns(records, LevelOutcome.RESISTANCE_REJECTION.value)[:TOP_PATTERN_COUNT]
            if item.sample_count >= MIN_PATTERN_SAMPLES
        ]

        for ranked, target in (
            (self._rank_patterns(records, LevelOutcome.SUPPORT_BREAK.value), top_support_break),
            (self._rank_patterns(records, LevelOutcome.RESISTANCE_BREAK.value), top_resistance_break),
            (self._rank_patterns(records, LevelOutcome.SUPPORT_BOUNCE.value), top_support_bounce),
            (self._rank_patterns(records, LevelOutcome.RESISTANCE_REJECTION.value), top_resistance_rejection),
        ):
            if len(target) < TOP_PATTERN_COUNT:
                target.clear()
                target.extend(item.as_dict() for item in ranked[:TOP_PATTERN_COUNT])

        conclusions = [
            f"Analyzed {len(records)} major support/resistance level events.",
            f"Support breaks: {outcome_counts[LevelOutcome.SUPPORT_BREAK.value]} | "
            f"Resistance breaks: {outcome_counts[LevelOutcome.RESISTANCE_BREAK.value]} | "
            f"Support bounces: {outcome_counts[LevelOutcome.SUPPORT_BOUNCE.value]} | "
            f"Resistance rejections: {outcome_counts[LevelOutcome.RESISTANCE_REJECTION.value]}.",
            f"Most common pre-break state: {state_counts.most_common(1)[0][0] if state_counts else 'N/A'}.",
        ]
        if top_support_break:
            conclusions.append(f"Top support-break pattern: {top_support_break[0]['pattern']}.")
        if top_resistance_break:
            conclusions.append(f"Top resistance-break pattern: {top_resistance_break[0]['pattern']}.")

        return SupportResistancePressureReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            level_classification_summary=level_classification_summary,
            outcome_counts=dict(outcome_counts),
            aggregate_level_metrics=aggregate_by_outcome,
            top_20_support_break_patterns=top_support_break,
            top_20_resistance_break_patterns=top_resistance_break,
            top_20_support_bounce_patterns=top_support_bounce,
            top_20_resistance_rejection_patterns=top_resistance_rejection,
            level_records=[record.as_dict() for record in records],
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_support_resistance_pressure_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> SupportResistancePressureReport:
    """Run support/resistance pressure research and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise SupportResistancePressureError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = SupportResistancePressureResearch(
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
        "Support/resistance pressure research completed: events=%s",
        len(report.level_records),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_support_resistance_pressure_report()
        print("Support Resistance Pressure Research Summary")
        print(f"Total events: {len(report.level_records)}")
        print("Outcome counts:", report.outcome_counts)
        print("Level states:", report.level_classification_summary)
        print("Top support-break pattern:")
        if report.top_20_support_break_patterns:
            print(f"  {report.top_20_support_break_patterns[0]['pattern']}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except SupportResistancePressureError as exc:
        logger.error("Support/resistance pressure research error: %s", exc)
        print(f"Support/resistance pressure research error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected support/resistance pressure research failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
