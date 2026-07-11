"""
Institutional Trigger Validation research for SmartMoneyEngine.

Identifies the exact trigger that starts real momentum from major directional
moves. Research-only; no trades or production modifications.
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

from src.context.liquidity_narrative_engine import LiquidityNarrativeEngine
from src.research.filter_research_engine import (
    RESEARCH_DAYS,
    FilterContextBuilder,
    FilterResearchEngine,
    _json_safe,
)
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    TIMEFRAME_MINUTES,
    _CheapMoveCandidate,
)
from src.research.major_level_strength_research import (
    LevelStrengthFeatures,
    MajorLevelStrengthResearch,
)
from src.research.support_resistance_pressure_research import SupportResistancePressureResearch
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "institutional_trigger_validation.json"

MOVE_THRESHOLDS = (100, 150, 200, 300)
PRE_TRIGGER_LOOKBACK = 50
VOLUME_LOOKBACK = 20
ATR_LOOKBACK = 14
LEVEL_TOUCH_ATR_RATIO = 0.5
CONSOLIDATION_ATR_RATIO = 1.5
WICK_SWEEP_PCT = 60.0
MAX_MOVES_PER_TIMEFRAME = 300
MIN_MODEL_SAMPLES = 5
TOP_MODEL_COUNT = 20
FALSE_TRIGGER_MAX_MAGNITUDE = 150
SIGNATURE_ARROW = " + "

DEFAULT_SYMBOLS = ("NIFTY50", "BANKNIFTY", "FINNIFTY")
TIMEFRAMES = ("5M", "15M", "1H")


class InstitutionalTriggerValidationError(Exception):
    """Raised when institutional trigger validation research fails."""


class MarketState(str, Enum):
    CONSOLIDATION = "Consolidation"
    RANGE_COMPRESSION = "Range Compression"
    RANGE_EXPANSION = "Range Expansion"
    TREND_CONTINUATION = "Trend Continuation"
    REVERSAL = "Reversal"


@dataclass(frozen=True)
class TriggerValidationRecord:
    symbol: str
    timeframe: str
    direction: str
    signal_side: str
    move_magnitude_points: float
    move_threshold_tier: int
    origin_bar: int
    expansion_bar: int
    trigger_bar: int
    trigger_timestamp: str
    level_context: dict[str, Any]
    trigger_candle: dict[str, Any]
    trigger_patterns: dict[str, bool]
    market_state: dict[str, Any]
    timing: dict[str, Any]
    trigger_model: str
    is_false_trigger: bool
    hit_100_plus: bool
    hit_200_plus: bool
    hit_300_plus: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TriggerMatrixEntry:
    trigger_model: str
    direction: str
    sample_count: int
    probability_100_plus_pct: float
    probability_200_plus_pct: float
    probability_300_plus_pct: float
    average_move_magnitude: float
    average_bars_to_expansion: float
    false_trigger_rate_pct: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TriggerModelRank:
    trigger_model: str
    direction: str
    sample_count: int
    frequency_pct: float
    probability_100_plus_pct: float
    probability_200_plus_pct: float
    probability_300_plus_pct: float
    average_move_magnitude: float
    false_trigger_rate_pct: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InstitutionalTriggerValidationReport:
    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    move_thresholds_points: list[int]
    total_moves_analyzed: int
    moves_by_symbol: dict[str, int]
    moves_by_threshold: dict[str, int]
    institutional_trigger_matrix: list[dict[str, Any]]
    top_20_bullish_trigger_models: list[dict[str, Any]]
    top_20_bearish_trigger_models: list[dict[str, Any]]
    top_20_false_trigger_models: list[dict[str, Any]]
    aggregate_trigger_profiles: dict[str, dict[str, Any]]
    trigger_records: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstitutionalTriggerValidationResearch:
    """Validate institutional triggers that start real momentum."""

    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = TIMEFRAMES,
    ) -> None:
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.research_days = research_days
        self.timeframes = timeframes
        self.move_engine = LiquidityMoveReconstructionResearch(research_days=research_days)
        self.level_strength_engine = MajorLevelStrengthResearch(research_days=research_days)
        self.pressure_engine = SupportResistancePressureResearch(research_days=research_days)
        self.context_builder = FilterContextBuilder()

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
    def _minutes_per_bar(timeframe_label: str) -> int:
        return TIMEFRAME_MINUTES.get(timeframe_label.upper(), 5)

    @staticmethod
    def _candle_parts(row: pd.Series) -> dict[str, Any]:
        open_price = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        body = abs(close - open_price)
        candle_range = max(high - low, 0.01)
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        wick_total = max(upper_wick + lower_wick, 0.01)
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
            "upper_wick_pct": upper_wick / candle_range,
            "lower_wick_pct": lower_wick / candle_range,
            "wick_pct": wick_total / candle_range,
            "close_location_pct": (close - low) / candle_range,
            "bullish": close > open_price,
            "bearish": close < open_price,
        }

    def _filter_engine(self, symbol: str) -> FilterResearchEngine:
        return FilterResearchEngine(symbol=symbol, research_days=self.research_days, timeframes=self.timeframes)

    def _detect_moves(self, frame: pd.DataFrame) -> list[_CheapMoveCandidate]:
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        candidates = self.move_engine._detect_moves_cheap(highs, lows, MOVE_THRESHOLDS[0])
        return self.move_engine._dedupe_cheap_moves(candidates)

    def _market_levels(self, frame: pd.DataFrame, index: int) -> dict[str, Any]:
        start = max(0, index - 200)
        window = frame.iloc[start : index + 1]
        close = float(frame.iloc[index]["Close"])

        def collect(column: str) -> list[float]:
            values: list[float] = []
            if column not in window.columns:
                return values
            for value in window[column]:
                parsed = self._to_float(value)
                if parsed is not None and self._is_active(value):
                    values.append(parsed)
            return values

        supports = (
            collect("Swing_Low")
            + collect("Equal_Low")
            + collect("Bullish_OB_Low")
            + collect("Sell_Side_Liquidity")
        )
        resistances = (
            collect("Swing_High")
            + collect("Equal_High")
            + collect("Bearish_OB_High")
            + collect("Buy_Side_Liquidity")
        )
        major_support = max([level for level in supports if level <= close], default=None)
        major_resistance = min([level for level in resistances if level >= close], default=None)
        if major_support is None and supports:
            major_support = min(supports)
        if major_resistance is None and resistances:
            major_resistance = max(resistances)

        nearest_level: float | None = None
        level_type = None
        if direction_support := major_support:
            dist_s = abs(close - direction_support)
            nearest_level = direction_support
            level_type = "support"
        if major_resistance is not None:
            dist_r = abs(close - major_resistance)
            if nearest_level is None or dist_r < abs(close - nearest_level):
                nearest_level = major_resistance
                level_type = "resistance"

        return {
            "major_support": major_support,
            "major_resistance": major_resistance,
            "nearest_level": nearest_level,
            "nearest_level_type": level_type,
        }

    def _level_context(
        self,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        trigger_bar: int,
        direction: str,
    ) -> dict[str, Any]:
        levels = self._market_levels(frame, trigger_bar)
        support = levels["major_support"]
        resistance = levels["major_resistance"]
        target = support if direction == "bullish" else resistance
        close = float(frame.iloc[trigger_bar]["Close"])
        atr = self.pressure_engine._atr(frame, trigger_bar)
        touch_band = atr * LEVEL_TOUCH_ATR_RATIO

        tests = bars_near = 0
        failed_breakouts = failed_breakdowns = 0
        liquidity_grabs = 0
        was_near = False
        start = max(0, trigger_bar - PRE_TRIGGER_LOOKBACK)

        for index in range(start, trigger_bar + 1):
            row = frame.iloc[index]
            high = float(row["High"])
            low = float(row["Low"])
            c = float(row["Close"])

            if self._is_active(row.get("Buy_Liquidity_Sweep")) or self._is_active(row.get("Sell_Liquidity_Sweep")):
                liquidity_grabs += 1

            swing_high = float(frame.iloc[max(0, index - 20) : index + 1]["High"].astype(float).max())
            swing_low = float(frame.iloc[max(0, index - 20) : index + 1]["Low"].astype(float).min())
            if high > swing_high and c <= swing_high:
                failed_breakouts += 1
            if low < swing_low and c >= swing_low:
                failed_breakdowns += 1

            if target is None:
                continue
            near = abs(c - target) <= touch_band or (low - touch_band <= target <= high + touch_band)
            if near:
                bars_near += 1
                if not was_near:
                    tests += 1
            was_near = near

        features = LevelStrengthFeatures(
            number_of_touches=tests,
            days_level_survived=0,
            bars_near_level=bars_near,
            bounce_count=0,
            rejection_count=0,
            liquidity_grabs=liquidity_grabs,
            equal_highs_lows_nearby=0,
            previous_day_overlap=False,
            weekly_overlap=False,
            monthly_overlap=False,
            demand_supply_zone_overlap=False,
            round_number_overlap=self.level_strength_engine._round_number_overlap(close),
            gap_interactions=0,
            average_volume_expansion=1.0,
            source_column="Swing_Low" if direction == "bullish" else "Swing_High",
        )
        pd_ov, wk_ov, mo_ov = self.level_strength_engine._calendar_overlaps(enriched, trigger_bar, close)
        features = LevelStrengthFeatures(
            **{
                **features.as_dict(),
                "previous_day_overlap": pd_ov,
                "weekly_overlap": wk_ov,
                "monthly_overlap": mo_ov,
                "demand_supply_zone_overlap": self.level_strength_engine._demand_supply_overlap(
                    frame,
                    trigger_bar,
                    close,
                ),
            },
        )
        score = self.level_strength_engine._compute_strength_score(features)
        category = self.level_strength_engine._classify_strength(score)

        return {
            "nearest_support": support,
            "nearest_resistance": resistance,
            "last_major_level": target,
            "last_major_level_type": "support" if direction == "bullish" else "resistance",
            "level_strength_score": score,
            "level_strength_category": category,
            "number_of_tests": tests,
            "failed_breakouts": failed_breakouts,
            "failed_breakdowns": failed_breakdowns,
            "liquidity_grabs": liquidity_grabs,
            "bars_near_level": bars_near,
            "time_near_level_bars": bars_near,
        }

    def _trigger_candle(
        self,
        frame: pd.DataFrame,
        trigger_bar: int,
        direction: str,
    ) -> dict[str, Any]:
        row = frame.iloc[trigger_bar]
        parts = self._candle_parts(row)
        atr = self.pressure_engine._atr(frame, trigger_bar)

        atr_start = max(0, trigger_bar - ATR_LOOKBACK * 2)
        prior_atrs = [self.pressure_engine._atr(frame, index) for index in range(atr_start, trigger_bar)]
        avg_prior_atr = mean(prior_atrs) if prior_atrs else atr
        atr_expansion = round(atr / avg_prior_atr, 2) if avg_prior_atr > 0 else 1.0

        volume = self._to_float(row.get("Volume")) or 0.0
        vol_start = max(0, trigger_bar - VOLUME_LOOKBACK)
        avg_volume = mean(
            self._to_float(frame.iloc[index].get("Volume")) or 0.0
            for index in range(vol_start, trigger_bar)
        ) if trigger_bar > vol_start else volume
        volume_expansion = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0

        prev_close = float(frame.iloc[trigger_bar - 1]["Close"]) if trigger_bar > 0 else parts["open"]
        gap = parts["open"] - prev_close
        displacement = LiquidityNarrativeEngine._displacement_strength_for_bar(row, direction)

        return {
            "body_pct": round(parts["body_pct"] * 100, 2),
            "upper_wick_pct": round(parts["upper_wick_pct"] * 100, 2),
            "lower_wick_pct": round(parts["lower_wick_pct"] * 100, 2),
            "close_location_pct": round(parts["close_location_pct"] * 100, 2),
            "volume_expansion_ratio": volume_expansion,
            "atr_expansion_ratio": atr_expansion,
            "gap_up": gap >= 0.5,
            "gap_down": gap <= -0.5,
            "gap_size_points": round(abs(gap), 2),
            "displacement_strength": displacement.value,
            "lower_wick_sweep_gt_60pct": parts["lower_wick_pct"] * 100 >= WICK_SWEEP_PCT,
            "upper_wick_sweep_gt_60pct": parts["upper_wick_pct"] * 100 >= WICK_SWEEP_PCT,
        }

    def _detect_trigger_patterns(self, frame: pd.DataFrame, trigger_bar: int) -> dict[str, bool]:
        window_start = max(0, trigger_bar - 2)
        window = frame.iloc[window_start : trigger_bar + 1]
        patterns = {
            "hammer": False,
            "shooting_star": False,
            "bullish_engulfing": False,
            "bearish_engulfing": False,
            "bullish_marubozu": False,
            "bearish_marubozu": False,
            "inside_bar": False,
            "outside_bar": False,
            "bullish_harami": False,
            "bearish_harami": False,
            "morning_star": False,
            "evening_star": False,
        }

        parts_list = [self._candle_parts(window.iloc[index]) for index in range(len(window))]
        if not parts_list:
            return patterns

        curr = parts_list[-1]
        if curr["body_pct"] >= 0.85:
            if curr["bullish"]:
                patterns["bullish_marubozu"] = True
            elif curr["bearish"]:
                patterns["bearish_marubozu"] = True

        if curr["body"] > 0:
            if curr["lower_wick"] >= 2 * curr["body"] and curr["upper_wick"] <= 0.25 * curr["body"]:
                patterns["hammer"] = True
            if curr["upper_wick"] >= 2 * curr["body"] and curr["lower_wick"] <= 0.25 * curr["body"]:
                patterns["shooting_star"] = True

        if len(parts_list) >= 2:
            prev = parts_list[-2]
            if prev["bearish"] and curr["bullish"]:
                if curr["open"] >= prev["close"] and curr["close"] <= prev["open"]:
                    patterns["bullish_harami"] = True
                if curr["close"] > prev["open"] and curr["open"] < prev["close"]:
                    patterns["bullish_engulfing"] = True
            if prev["bullish"] and curr["bearish"]:
                if curr["open"] <= prev["close"] and curr["close"] >= prev["open"]:
                    patterns["bearish_harami"] = True
                if curr["close"] < prev["open"] and curr["open"] > prev["close"]:
                    patterns["bearish_engulfing"] = True
            if curr["high"] <= prev["high"] and curr["low"] >= prev["low"]:
                patterns["inside_bar"] = True
            if curr["high"] >= prev["high"] and curr["low"] <= prev["low"]:
                patterns["outside_bar"] = True

        if len(parts_list) >= 3:
            first, middle, third = parts_list[-3], parts_list[-2], parts_list[-1]
            midpoint = (first["open"] + first["close"]) / 2
            if (
                first["bearish"]
                and middle["body"] <= first["body"] * 0.35
                and third["bullish"]
                and third["close"] > midpoint
            ):
                patterns["morning_star"] = True
            if (
                first["bullish"]
                and middle["body"] <= first["body"] * 0.35
                and third["bearish"]
                and third["close"] < midpoint
            ):
                patterns["evening_star"] = True

        return patterns

    def _market_state(
        self,
        frame: pd.DataFrame,
        trigger_bar: int,
        direction: str,
        level: dict[str, Any],
    ) -> dict[str, Any]:
        start = max(0, trigger_bar - PRE_TRIGGER_LOOKBACK)
        window = frame.iloc[start : trigger_bar + 1]
        atr = self.pressure_engine._atr(frame, trigger_bar)

        range_high = float(window["High"].astype(float).max())
        range_low = float(window["Low"].astype(float).min())
        range_width = range_high - range_low

        recent = frame.iloc[max(start, trigger_bar - 10) : trigger_bar + 1]
        prior = frame.iloc[max(start, trigger_bar - 20) : max(start, trigger_bar - 10)]
        recent_width = (
            float(recent["High"].astype(float).max()) - float(recent["Low"].astype(float).min())
            if len(recent)
            else range_width
        )
        prior_width = (
            float(prior["High"].astype(float).max()) - float(prior["Low"].astype(float).min())
            if len(prior)
            else recent_width
        )

        consolidation_bars = 0
        for index in range(start, trigger_bar + 1):
            local_start = max(start, index - 5)
            local = frame.iloc[local_start : index + 1]
            local_width = float(local["High"].astype(float).max()) - float(local["Low"].astype(float).min())
            if local_width <= atr * CONSOLIDATION_ATR_RATIO:
                consolidation_bars += 1

        choch = bos = False
        row = frame.iloc[trigger_bar]
        if direction == "bullish":
            choch = self._is_active(row.get("Bullish_CHOCH"))
            bos = self._is_active(row.get("Bullish_BOS"))
        else:
            choch = self._is_active(row.get("Bearish_CHOCH"))
            bos = self._is_active(row.get("Bearish_BOS"))

        states: list[str] = []
        if consolidation_bars >= 10:
            states.append(MarketState.CONSOLIDATION.value)
        if prior_width > 0 and recent_width < prior_width * 0.75:
            states.append(MarketState.RANGE_COMPRESSION.value)
        if prior_width > 0 and recent_width > prior_width * 1.25:
            states.append(MarketState.RANGE_EXPANSION.value)
        if bos and not choch:
            states.append(MarketState.TREND_CONTINUATION.value)
        if choch or level["failed_breakouts"] >= 1 or level["failed_breakdowns"] >= 1:
            states.append(MarketState.REVERSAL.value)
        if not states:
            states.append(MarketState.CONSOLIDATION.value)

        return {
            "states": states,
            "primary_state": states[0],
            "consolidation_bars": consolidation_bars,
            "range_width_points": round(range_width, 2),
            "recent_range_width_points": round(recent_width, 2),
            "prior_range_width_points": round(prior_width, 2),
            "choch_present": choch,
            "bos_present": bos,
        }

    def _build_trigger_model(
        self,
        level: dict[str, Any],
        candle: dict[str, Any],
        patterns: dict[str, bool],
        market: dict[str, Any],
        direction: str,
    ) -> str:
        parts: list[str] = []
        tests = level["number_of_tests"]
        if tests >= 1:
            parts.append(f"Level Retest x{tests}")
        if level["failed_breakouts"] >= 1:
            parts.append("Failed Breakout")
        if level["failed_breakdowns"] >= 1:
            parts.append("Failed Breakdown")
        if level["liquidity_grabs"] >= 1:
            parts.append(f"Liquidity Grab x{level['liquidity_grabs']}")
        if candle["lower_wick_sweep_gt_60pct"] and direction == "bullish":
            parts.append("Lower Wick Sweep >60%")
        if candle["upper_wick_sweep_gt_60pct"] and direction == "bearish":
            parts.append("Upper Wick Sweep >60%")
        if candle["volume_expansion_ratio"] >= 1.5:
            parts.append("Volume Expansion")
        if candle["displacement_strength"] in {"Strong", "Medium"}:
            parts.append(f"{candle['displacement_strength']} Displacement")

        pattern_labels = {
            "hammer": "Hammer",
            "shooting_star": "Shooting Star",
            "bullish_engulfing": "Bullish Engulfing",
            "bearish_engulfing": "Bearish Engulfing",
            "bullish_marubozu": "Bullish Marubozu",
            "bearish_marubozu": "Bearish Marubozu",
            "inside_bar": "Inside Bar",
            "outside_bar": "Outside Bar",
            "bullish_harami": "Bullish Harami",
            "bearish_harami": "Bearish Harami",
            "morning_star": "Morning Star",
            "evening_star": "Evening Star",
        }
        for key, label in pattern_labels.items():
            if patterns.get(key):
                parts.append(label)

        parts.append(f"Level:{level['level_strength_category']}")
        parts.append(market["primary_state"])
        return SIGNATURE_ARROW.join(parts) if parts else "No Trigger Context"

    def _analyze_move(
        self,
        symbol: str,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        candidate: _CheapMoveCandidate,
        timeframe_label: str,
    ) -> TriggerValidationRecord:
        origin_bar = candidate.start_bar
        expansion_bar = candidate.expansion_bar
        trigger_bar = max(0, expansion_bar - 1)
        direction = candidate.direction
        signal_side = "BUY" if direction == "bullish" else "SELL"
        magnitude = candidate.magnitude

        level = self._level_context(frame, enriched, trigger_bar, direction)
        candle = self._trigger_candle(frame, trigger_bar, direction)
        patterns = self._detect_trigger_patterns(frame, trigger_bar)
        market = self._market_state(frame, trigger_bar, direction, level)
        trigger_model = self._build_trigger_model(level, candle, patterns, market, direction)

        minutes = self._minutes_per_bar(timeframe_label)
        bars_to_expansion = expansion_bar - trigger_bar
        timing = {
            "bars_trigger_to_expansion": bars_to_expansion,
            "minutes_trigger_to_expansion": round(bars_to_expansion * minutes, 1),
        }

        tier = MOVE_THRESHOLDS[0]
        for threshold in MOVE_THRESHOLDS:
            if magnitude >= threshold:
                tier = threshold

        is_false = magnitude < FALSE_TRIGGER_MAX_MAGNITUDE

        return TriggerValidationRecord(
            symbol=symbol,
            timeframe=timeframe_label,
            direction=direction,
            signal_side=signal_side,
            move_magnitude_points=magnitude,
            move_threshold_tier=tier,
            origin_bar=origin_bar,
            expansion_bar=expansion_bar,
            trigger_bar=trigger_bar,
            trigger_timestamp=str(frame.iloc[trigger_bar]["Date"]),
            level_context=level,
            trigger_candle=candle,
            trigger_patterns=patterns,
            market_state=market,
            timing=timing,
            trigger_model=trigger_model,
            is_false_trigger=is_false,
            hit_100_plus=magnitude >= 100,
            hit_200_plus=magnitude >= 200,
            hit_300_plus=magnitude >= 300,
        )

    def _collect_records(self, metadata: dict[str, Any]) -> list[TriggerValidationRecord]:
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        records: list[TriggerValidationRecord] = []
        seen: set[tuple[str, str, int, str]] = set()

        for symbol in self.symbols:
            filter_engine = self._filter_engine(symbol)
            for timeframe_label in self.timeframes:
                path = filter_engine._pipeline_path(timeframe_label)
                if not path.exists():
                    try:
                        path = filter_engine._ensure_pipeline(timeframe_label, start, end)
                    except Exception as exc:
                        logger.warning("Skipping %s/%s pipeline: %s", symbol, timeframe_label, exc)
                        continue

                frame = pd.read_csv(path).reset_index(drop=True)
                if len(frame) < FORWARD_BARS + PRE_TRIGGER_LOOKBACK:
                    continue

                enriched = self.context_builder.enrich(frame)
                candidates = self._detect_moves(frame)
                if len(candidates) > MAX_MOVES_PER_TIMEFRAME:
                    candidates = sorted(candidates, key=lambda item: item.magnitude, reverse=True)[
                        :MAX_MOVES_PER_TIMEFRAME
                    ]
                logger.info(
                    "Trigger validation: %s/%s moves=%s",
                    symbol,
                    timeframe_label,
                    len(candidates),
                )

                for candidate in candidates:
                    if candidate.start_bar < PRE_TRIGGER_LOOKBACK:
                        continue
                    key = (symbol, timeframe_label, candidate.expansion_bar, candidate.direction)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(
                        self._analyze_move(symbol, frame, enriched, candidate, timeframe_label),
                    )

        return records

    def _matrix_from_records(
        self,
        records: list[TriggerValidationRecord],
    ) -> list[TriggerMatrixEntry]:
        groups: dict[tuple[str, str], list[TriggerValidationRecord]] = defaultdict(list)
        for record in records:
            groups[(record.trigger_model, record.direction)].append(record)

        entries: list[TriggerMatrixEntry] = []
        for (model, direction), bucket in groups.items():
            if len(bucket) < MIN_MODEL_SAMPLES:
                continue
            total = len(bucket)
            entries.append(
                TriggerMatrixEntry(
                    trigger_model=model,
                    direction=direction,
                    sample_count=total,
                    probability_100_plus_pct=round(
                        sum(1 for item in bucket if item.hit_100_plus) / total * 100,
                        2,
                    ),
                    probability_200_plus_pct=round(
                        sum(1 for item in bucket if item.hit_200_plus) / total * 100,
                        2,
                    ),
                    probability_300_plus_pct=round(
                        sum(1 for item in bucket if item.hit_300_plus) / total * 100,
                        2,
                    ),
                    average_move_magnitude=round(mean(item.move_magnitude_points for item in bucket), 2),
                    average_bars_to_expansion=round(
                        mean(item.timing["bars_trigger_to_expansion"] for item in bucket),
                        2,
                    ),
                    false_trigger_rate_pct=round(
                        sum(1 for item in bucket if item.is_false_trigger) / total * 100,
                        2,
                    ),
                ),
            )

        entries.sort(
            key=lambda item: (
                item.probability_100_plus_pct,
                item.probability_200_plus_pct,
                item.sample_count,
            ),
            reverse=True,
        )
        for index, entry in enumerate(entries, start=1):
            entry.rank = index
        return entries

    def _top_models(
        self,
        records: list[TriggerValidationRecord],
        direction: str,
        *,
        false_only: bool = False,
    ) -> list[TriggerModelRank]:
        if false_only:
            bucket = [record for record in records if record.is_false_trigger]
        elif direction == "bullish":
            bucket = [record for record in records if record.direction == "bullish"]
        else:
            bucket = [record for record in records if record.direction == "bearish"]

        groups: dict[str, list[TriggerValidationRecord]] = defaultdict(list)
        for record in bucket:
            groups[record.trigger_model].append(record)

        total = len(bucket) or 1
        ranked: list[TriggerModelRank] = []
        for model, group in groups.items():
            if len(group) < (2 if false_only else MIN_MODEL_SAMPLES):
                continue
            count = len(group)
            ranked.append(
                TriggerModelRank(
                    trigger_model=model,
                    direction=direction if not false_only else group[0].direction,
                    sample_count=count,
                    frequency_pct=round(count / total * 100, 2),
                    probability_100_plus_pct=round(
                        sum(1 for item in group if item.hit_100_plus) / count * 100,
                        2,
                    ),
                    probability_200_plus_pct=round(
                        sum(1 for item in group if item.hit_200_plus) / count * 100,
                        2,
                    ),
                    probability_300_plus_pct=round(
                        sum(1 for item in group if item.hit_300_plus) / count * 100,
                        2,
                    ),
                    average_move_magnitude=round(mean(item.move_magnitude_points for item in group), 2),
                    false_trigger_rate_pct=round(
                        sum(1 for item in group if item.is_false_trigger) / count * 100,
                        2,
                    ),
                ),
            )

        if false_only:
            ranked.sort(key=lambda item: (item.sample_count, item.false_trigger_rate_pct), reverse=True)
        else:
            ranked.sort(
                key=lambda item: (item.probability_100_plus_pct, item.sample_count, item.average_move_magnitude),
                reverse=True,
            )
        for index, item in enumerate(ranked[:TOP_MODEL_COUNT], start=1):
            item.rank = index
        return ranked[:TOP_MODEL_COUNT]

    def _aggregate_profiles(
        self,
        records: list[TriggerValidationRecord],
    ) -> dict[str, dict[str, Any]]:
        profiles: dict[str, dict[str, Any]] = {}
        for side, direction in (("BUY", "bullish"), ("SELL", "bearish")):
            bucket = [record for record in records if record.direction == direction]
            if not bucket:
                continue
            profiles[side] = {
                "sample_count": len(bucket),
                "average_tests": round(mean(record.level_context["number_of_tests"] for record in bucket), 2),
                "average_failed_breakouts": round(
                    mean(record.level_context["failed_breakouts"] for record in bucket),
                    2,
                ),
                "average_failed_breakdowns": round(
                    mean(record.level_context["failed_breakdowns"] for record in bucket),
                    2,
                ),
                "average_liquidity_grabs": round(
                    mean(record.level_context["liquidity_grabs"] for record in bucket),
                    2,
                ),
                "average_bars_near_level": round(
                    mean(record.level_context["bars_near_level"] for record in bucket),
                    2,
                ),
                "average_body_pct": round(mean(record.trigger_candle["body_pct"] for record in bucket), 2),
                "average_volume_expansion": round(
                    mean(record.trigger_candle["volume_expansion_ratio"] for record in bucket),
                    2,
                ),
                "average_bars_to_expansion": round(
                    mean(record.timing["bars_trigger_to_expansion"] for record in bucket),
                    2,
                ),
                "false_trigger_rate_pct": round(
                    sum(1 for record in bucket if record.is_false_trigger) / len(bucket) * 100,
                    2,
                ),
            }
        return profiles

    def run(self, metadata: dict[str, Any]) -> InstitutionalTriggerValidationReport:
        started = time.perf_counter()
        records = self._collect_records(metadata)
        matrix = self._matrix_from_records(records)
        bullish = self._top_models(records, "bullish")
        bearish = self._top_models(records, "bearish")
        false_models = self._top_models(records, "bullish", false_only=True)

        moves_by_symbol = Counter(record.symbol for record in records)
        moves_by_threshold = Counter(str(record.move_threshold_tier) for record in records)

        top_matrix = matrix[0] if matrix else None
        conclusions = [
            f"Analyzed {len(records)} major directional triggers across {self.symbols}.",
            f"Institutional trigger matrix entries: {len(matrix)}.",
            (
                f"Top trigger: {top_matrix.trigger_model[:100]} = "
                f"{top_matrix.probability_100_plus_pct}% probability of 100+ move"
                if top_matrix
                else "No trigger matrix entries met minimum sample threshold."
            ),
            f"Top bullish model: {bullish[0].trigger_model[:100] if bullish else 'N/A'}.",
            f"Top bearish model: {bearish[0].trigger_model[:100] if bearish else 'N/A'}.",
            f"Top false trigger: {false_models[0].trigger_model[:100] if false_models else 'N/A'}.",
        ]

        return InstitutionalTriggerValidationReport(
            symbols_analyzed=list(self.symbols),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            move_thresholds_points=list(MOVE_THRESHOLDS),
            total_moves_analyzed=len(records),
            moves_by_symbol=dict(moves_by_symbol),
            moves_by_threshold=dict(moves_by_threshold),
            institutional_trigger_matrix=[entry.as_dict() for entry in matrix[:TOP_MODEL_COUNT * 2]],
            top_20_bullish_trigger_models=[item.as_dict() for item in bullish],
            top_20_bearish_trigger_models=[item.as_dict() for item in bearish],
            top_20_false_trigger_models=[item.as_dict() for item in false_models],
            aggregate_trigger_profiles=self._aggregate_profiles(records),
            trigger_records=[record.as_dict() for record in records],
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_institutional_trigger_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> InstitutionalTriggerValidationReport:
    """Run institutional trigger validation research and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise InstitutionalTriggerValidationError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = InstitutionalTriggerValidationResearch(
        symbols=symbols,
        research_days=metadata.get("research_window_days", RESEARCH_DAYS),
        timeframes=tuple(metadata.get("timeframes_analyzed", TIMEFRAMES)),
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Institutional trigger validation completed: moves=%s",
        report.total_moves_analyzed,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_institutional_trigger_validation_report()
        print("Institutional Trigger Validation Research Summary")
        print(f"Moves analyzed: {report.total_moves_analyzed}")
        print(f"Symbols: {report.symbols_analyzed}")
        if report.institutional_trigger_matrix:
            top = report.institutional_trigger_matrix[0]
            print(
                f"Top trigger probability: {top['probability_100_plus_pct']}% for 100+ "
                f"({top['sample_count']} samples)",
            )
        if report.top_20_bullish_trigger_models:
            print(f"Top bullish: {report.top_20_bullish_trigger_models[0]['trigger_model'][:120]}...")
        if report.top_20_bearish_trigger_models:
            print(f"Top bearish: {report.top_20_bearish_trigger_models[0]['trigger_model'][:120]}...")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except InstitutionalTriggerValidationError as exc:
        logger.error("Institutional trigger validation error: %s", exc)
        print(f"Institutional trigger validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected institutional trigger validation error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
