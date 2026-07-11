"""
Institutional Momentum Origin research for SmartMoneyEngine.

Identifies what the market looked like before real expansion moves started.
Research-only; no trades, signals, or production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.context.liquidity_narrative_engine import DisplacementStrength, LiquidityNarrativeEngine
from src.context.market_intelligence_engine import MarketIntelligenceEngine
from src.research.filter_research_engine import RESEARCH_DAYS, FilterResearchEngine, _json_safe
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    MIN_MOVE_SEPARATION_BARS,
    _CheapMoveCandidate,
)
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTER_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "filter_research_report.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "research" / "institutional_momentum_origin.json"

MOVE_THRESHOLDS = (50, 100, 150, 200)
PRE_EXPANSION_LOOKBACK = 50
TOP_PATTERN_COUNT = 20
MIN_PATTERN_SAMPLES = 3
LEVEL_TOUCH_ATR_RATIO = 0.5
CONSOLIDATION_ATR_RATIO = 1.5
VOLUME_LOOKBACK = 20
LOCATION_LOOKBACK = 200


class InstitutionalMomentumOriginError(Exception):
    """Raised when institutional momentum origin research fails."""


@dataclass(frozen=True)
class CandleStructureMetrics:
    """Aggregate candle structure in the pre-expansion window."""

    largest_wick_points: float
    average_wick_points: float
    largest_body_points: float
    average_body_points: float
    average_close_location_pct: float
    bullish_marubozu_count: int
    bearish_marubozu_count: int
    hammer_count: int
    shooting_star_count: int
    bullish_harami_count: int
    bearish_harami_count: int
    morning_star_count: int
    evening_star_count: int
    bullish_engulfing_count: int
    bearish_engulfing_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreExpansionAnalysis:
    """Full pre-expansion context for one detected move."""

    timeframe: str
    threshold_points: int
    direction: str
    origin_bar: int
    expansion_bar: int
    origin_timestamp: str
    expansion_timestamp: str
    move_magnitude_points: float
    liquidity_grabs: int
    buy_side_grabs: int
    sell_side_grabs: int
    failed_breakouts: int
    failed_breakdowns: int
    liquidity_taken_before_expansion: int
    support_resistance_tests: int
    bars_near_level: int
    distance_from_nearest_level: float | None
    break_success_rate_pct: float
    false_break_count: int
    candle_structure: dict[str, Any]
    confirmation_candle: dict[str, Any]
    trap_analysis: dict[str, Any]
    market_structure: dict[str, Any]
    gap_analysis: dict[str, Any]
    pattern_key: str
    pattern_category: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatternRankMetrics:
    """Frequency and reliability for one momentum-origin pattern."""

    pattern: str
    category: str
    sample_count: int
    frequency_pct: float
    average_move_magnitude: float
    average_threshold_points: float
    reliability_score: float
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InstitutionalMomentumOriginReport:
    """Aggregate institutional momentum origin research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    timeframes_analyzed: list[str]
    move_thresholds_points: list[int]
    pre_expansion_lookback_bars: int
    total_expansion_moves: dict[str, int]
    aggregate_pre_expansion_profile: dict[str, dict[str, Any]]
    top_20_momentum_origin_patterns: list[dict[str, Any]]
    pattern_rankings: dict[str, str]
    ranked_patterns_by_category: dict[str, list[dict[str, Any]]]
    expansion_records: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstitutionalMomentumOriginResearch:
    """Analyze market conditions before institutional expansion moves."""

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
        self.move_engine = LiquidityMoveReconstructionResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.narrative_engine = LiquidityNarrativeEngine(symbol=symbol)
        self.intelligence_engine = MarketIntelligenceEngine(symbol=symbol)

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

    def _collect_levels(self, window: pd.DataFrame, column: str) -> list[float]:
        values: list[float] = []
        if column not in window.columns:
            return values
        for value in window[column]:
            parsed = self._to_float(value)
            if parsed is not None and self._is_active(value):
                values.append(parsed)
        return values

    def _market_levels(self, frame: pd.DataFrame, index: int) -> dict[str, Any]:
        start = max(0, index - LOCATION_LOOKBACK)
        window = frame.iloc[start : index + 1]
        close = self._to_float(frame.iloc[index]["Close"]) or 0.0

        supports = (
            self._collect_levels(window, "Swing_Low")
            + self._collect_levels(window, "Equal_Low")
            + self._collect_levels(window, "Bullish_OB_Low")
            + self._collect_levels(window, "Sell_Side_Liquidity")
        )
        resistances = (
            self._collect_levels(window, "Swing_High")
            + self._collect_levels(window, "Equal_High")
            + self._collect_levels(window, "Bearish_OB_High")
            + self._collect_levels(window, "Buy_Side_Liquidity")
        )

        major_support = max([level for level in supports if level <= close], default=None)
        major_resistance = min([level for level in resistances if level >= close], default=None)
        if major_support is None and supports:
            major_support = min(supports)
        if major_resistance is None and resistances:
            major_resistance = max(resistances)

        nearest_level: float | None = None
        distances: list[float] = []
        if major_support is not None:
            distances.append(abs(close - major_support))
        if major_resistance is not None:
            distances.append(abs(close - major_resistance))
        if distances:
            nearest_level = min(distances)

        return {
            "major_support": major_support,
            "major_resistance": major_resistance,
            "distance_from_nearest_level": round(nearest_level, 2) if nearest_level is not None else None,
        }

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
            "body_pct": round(body / candle_range, 4),
            "wick_pct": round((upper_wick + lower_wick) / candle_range, 4),
            "close_location_pct": round((close - low) / candle_range, 4),
            "bullish": close > open_price,
            "bearish": close < open_price,
        }

    def _count_candle_patterns(self, window: pd.DataFrame) -> CandleStructureMetrics:
        wicks: list[float] = []
        bodies: list[float] = []
        close_locations: list[float] = []
        bullish_marubozu = bearish_marubozu = 0
        hammers = shooting_stars = 0
        bull_harami = bear_harami = 0
        morning_stars = evening_stars = 0
        bull_engulf = bear_engulf = 0

        for index in range(len(window)):
            parts = self._candle_parts(window.iloc[index])
            wicks.append(parts["upper_wick"] + parts["lower_wick"])
            bodies.append(parts["body"])
            close_locations.append(parts["close_location_pct"])

            if parts["body_pct"] >= 0.85:
                if parts["bullish"]:
                    bullish_marubozu += 1
                elif parts["bearish"]:
                    bearish_marubozu += 1

            if parts["body"] > 0:
                if parts["lower_wick"] >= 2 * parts["body"] and parts["upper_wick"] <= 0.25 * parts["body"]:
                    hammers += 1
                if parts["upper_wick"] >= 2 * parts["body"] and parts["lower_wick"] <= 0.25 * parts["body"]:
                    shooting_stars += 1

            if index >= 1:
                prev = self._candle_parts(window.iloc[index - 1])
                curr = parts
                if prev["bearish"] and curr["bullish"]:
                    if curr["open"] >= prev["close"] and curr["close"] <= prev["open"]:
                        bull_harami += 1
                    if curr["close"] > prev["open"] and curr["open"] < prev["close"]:
                        bull_engulf += 1
                if prev["bullish"] and curr["bearish"]:
                    if curr["open"] <= prev["close"] and curr["close"] >= prev["open"]:
                        bear_harami += 1
                    if curr["close"] < prev["open"] and curr["open"] > prev["close"]:
                        bear_engulf += 1

            if index >= 2:
                first = self._candle_parts(window.iloc[index - 2])
                middle = self._candle_parts(window.iloc[index - 1])
                third = parts
                midpoint = (first["open"] + first["close"]) / 2
                if (
                    first["bearish"]
                    and middle["body"] <= first["body"] * 0.35
                    and third["bullish"]
                    and third["close"] > midpoint
                ):
                    morning_stars += 1
                if (
                    first["bullish"]
                    and middle["body"] <= first["body"] * 0.35
                    and third["bearish"]
                    and third["close"] < midpoint
                ):
                    evening_stars += 1

        return CandleStructureMetrics(
            largest_wick_points=round(max(wicks), 2) if wicks else 0.0,
            average_wick_points=round(mean(wicks), 2) if wicks else 0.0,
            largest_body_points=round(max(bodies), 2) if bodies else 0.0,
            average_body_points=round(mean(bodies), 2) if bodies else 0.0,
            average_close_location_pct=round(mean(close_locations) * 100, 2) if close_locations else 50.0,
            bullish_marubozu_count=bullish_marubozu,
            bearish_marubozu_count=bearish_marubozu,
            hammer_count=hammers,
            shooting_star_count=shooting_stars,
            bullish_harami_count=bull_harami,
            bearish_harami_count=bear_harami,
            morning_star_count=morning_stars,
            evening_star_count=evening_stars,
            bullish_engulfing_count=bull_engulf,
            bearish_engulfing_count=bear_engulf,
        )

    def _liquidity_behavior(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
        levels: dict[str, Any],
        atr: float,
    ) -> dict[str, int]:
        buy_grabs = sell_grabs = 0
        failed_breakouts = failed_breakdowns = 0
        resistance = levels.get("major_resistance")
        support = levels.get("major_support")

        for index in range(start_bar, end_bar + 1):
            row = frame.iloc[index]
            if self._is_active(row.get("Buy_Liquidity_Sweep")):
                buy_grabs += 1
            if self._is_active(row.get("Sell_Liquidity_Sweep")):
                sell_grabs += 1

            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])

            if resistance is not None and high > resistance + atr * 0.1:
                if close < resistance:
                    failed_breakouts += 1
            if support is not None and low < support - atr * 0.1:
                if close > support:
                    failed_breakdowns += 1

        return {
            "liquidity_grabs": buy_grabs + sell_grabs,
            "buy_side_grabs": buy_grabs,
            "sell_side_grabs": sell_grabs,
            "failed_breakouts": failed_breakouts,
            "failed_breakdowns": failed_breakdowns,
            "liquidity_taken_before_expansion": buy_grabs + sell_grabs,
        }

    def _support_resistance_behavior(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
        expansion_bar: int,
        direction: str,
        levels: dict[str, Any],
        atr: float,
    ) -> dict[str, Any]:
        support = levels.get("major_support")
        resistance = levels.get("major_resistance")
        touch_threshold = atr * LEVEL_TOUCH_ATR_RATIO
        tests = bars_near = 0
        false_breaks = 0
        successful_breaks = 0

        for index in range(start_bar, end_bar + 1):
            close = float(frame.iloc[index]["Close"])
            high = float(frame.iloc[index]["High"])
            low = float(frame.iloc[index]["Low"])

            for level in (support, resistance):
                if level is None:
                    continue
                distance = abs(close - level)
                if distance <= touch_threshold:
                    tests += 1
                    bars_near += 1

            if resistance is not None:
                if high > resistance and close > resistance:
                    successful_breaks += 1
                elif high > resistance and close <= resistance:
                    false_breaks += 1
            if support is not None:
                if low < support and close < support:
                    successful_breaks += 1
                elif low < support and close >= support:
                    false_breaks += 1

        break_attempts = successful_breaks + false_breaks
        break_success_rate = (
            round((successful_breaks / break_attempts) * 100, 2) if break_attempts else 0.0
        )

        return {
            "support_resistance_tests": tests,
            "bars_near_level": bars_near,
            "distance_from_nearest_level": levels.get("distance_from_nearest_level"),
            "break_success_rate_pct": break_success_rate,
            "false_break_count": false_breaks,
        }

    def _confirmation_candle(
        self,
        frame: pd.DataFrame,
        expansion_bar: int,
        direction: str,
    ) -> dict[str, Any]:
        confirm_bar = max(0, expansion_bar - 1)
        row = frame.iloc[confirm_bar]
        parts = self._candle_parts(row)
        atr = self._atr(frame, confirm_bar)

        volume = self._to_float(row.get("Volume")) or 0.0
        vol_start = max(0, confirm_bar - VOLUME_LOOKBACK)
        avg_volume = mean(
            self._to_float(frame.iloc[index].get("Volume")) or 0.0
            for index in range(vol_start, confirm_bar)
        ) if confirm_bar > vol_start else volume
        volume_expansion = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0

        displacement = LiquidityNarrativeEngine._displacement_strength_for_bar(row, direction)
        close_strength = parts["close_location_pct"] * 100

        follow_through = False
        if expansion_bar < len(frame):
            next_row = frame.iloc[expansion_bar]
            next_close = float(next_row["Close"])
            confirm_close = parts["close"]
            if direction == "bullish":
                follow_through = next_close > confirm_close
            else:
                follow_through = next_close < confirm_close

        return {
            "confirmation_bar": confirm_bar,
            "confirmation_timestamp": str(row["Date"]),
            "body_pct": round(parts["body_pct"] * 100, 2),
            "wick_pct": round(parts["wick_pct"] * 100, 2),
            "volume_expansion_ratio": volume_expansion,
            "close_strength_pct": round(close_strength, 2),
            "displacement_strength": displacement.value,
            "follow_through_success": follow_through,
        }

    def _trap_analysis(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
    ) -> dict[str, Any]:
        false_upside: list[float] = []
        false_downside: list[float] = []

        for index in range(start_bar + 1, end_bar + 1):
            window = frame.iloc[max(0, index - 20) : index]
            prior_high = float(window["High"].astype(float).max()) if len(window) else float(frame.iloc[index]["High"])
            prior_low = float(window["Low"].astype(float).min()) if len(window) else float(frame.iloc[index]["Low"])
            high = float(frame.iloc[index]["High"])
            low = float(frame.iloc[index]["Low"])
            close = float(frame.iloc[index]["Close"])

            if high > prior_high and close < prior_high:
                false_upside.append(round(high - prior_high, 2))
            if low < prior_low and close > prior_low:
                false_downside.append(round(prior_low - low, 2))

        trap_sizes = false_upside + [abs(value) for value in false_downside]
        return {
            "false_upside_breaks": len(false_upside),
            "false_downside_breaks": len(false_downside),
            "total_traps": len(false_upside) + len(false_downside),
            "average_trap_size_points": round(mean(trap_sizes), 2) if trap_sizes else 0.0,
            "maximum_trap_size_points": round(max(trap_sizes), 2) if trap_sizes else 0.0,
        }

    def _market_structure(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
        origin_bar: int,
    ) -> dict[str, Any]:
        window = frame.iloc[start_bar : end_bar + 1]
        atr = self._atr(frame, end_bar)
        range_high = float(window["High"].astype(float).max())
        range_low = float(window["Low"].astype(float).min())
        range_width = round(range_high - range_low, 2)

        consolidation_bars = 0
        for index in range(start_bar, end_bar + 1):
            local_start = max(start_bar, index - 5)
            local = frame.iloc[local_start : index + 1]
            local_width = float(local["High"].astype(float).max()) - float(local["Low"].astype(float).min())
            if local_width <= atr * CONSOLIDATION_ATR_RATIO:
                consolidation_bars += 1

        choch_count = bos_count = sweep_count = 0
        for index in range(start_bar, end_bar + 1):
            row = frame.iloc[index]
            if self._is_active(row.get("Bullish_CHOCH")) or self._is_active(row.get("Bearish_CHOCH")):
                choch_count += 1
            if self._is_active(row.get("Bullish_BOS")) or self._is_active(row.get("Bearish_BOS")):
                bos_count += 1
            if self._is_active(row.get("Buy_Liquidity_Sweep")) or self._is_active(
                row.get("Sell_Liquidity_Sweep"),
            ):
                sweep_count += 1

        return {
            "consolidation_duration_bars": consolidation_bars,
            "range_width_points": range_width,
            "choch_count": choch_count,
            "bos_count": bos_count,
            "liquidity_sweep_count": sweep_count,
            "origin_to_expansion_bars": (end_bar + 1) - origin_bar if end_bar >= origin_bar else 0,
        }

    def _gap_analysis(
        self,
        frame: pd.DataFrame,
        start_bar: int,
        end_bar: int,
        expansion_bar: int,
        direction: str,
        levels: dict[str, Any],
    ) -> dict[str, Any]:
        gap_ups = gap_downs = 0
        gap_sizes: list[float] = []
        gap_success = 0
        gap_total = 0

        for index in range(max(start_bar + 1, 1), end_bar + 1):
            prev_close = float(frame.iloc[index - 1]["Close"])
            open_price = float(frame.iloc[index]["Open"])
            gap = round(open_price - prev_close, 2)
            if gap > 0.5:
                gap_ups += 1
                gap_sizes.append(gap)
            elif gap < -0.5:
                gap_downs += 1
                gap_sizes.append(abs(gap))

            if abs(gap) > 0.5:
                gap_total += 1
                forward_end = min(len(frame) - 1, index + 10)
                if direction == "bullish":
                    move = float(frame.iloc[index : forward_end + 1]["High"].astype(float).max()) - open_price
                else:
                    move = open_price - float(frame.iloc[index : forward_end + 1]["Low"].astype(float).min())
                if move >= 20:
                    gap_success += 1

        nearest_distance = levels.get("distance_from_nearest_level")
        return {
            "gap_up_count": gap_ups,
            "gap_down_count": gap_downs,
            "average_gap_size_points": round(mean(gap_sizes), 2) if gap_sizes else 0.0,
            "maximum_gap_size_points": round(max(gap_sizes), 2) if gap_sizes else 0.0,
            "distance_from_major_level_at_expansion": nearest_distance,
            "gap_success_rate_pct": round((gap_success / gap_total) * 100, 2) if gap_total else 0.0,
        }

    def _classify_pattern_category(
        self,
        direction: str,
        liquidity: dict[str, int],
        structure: dict[str, Any],
        sr: dict[str, Any],
        levels: dict[str, Any],
        frame: pd.DataFrame,
        expansion_bar: int,
    ) -> str:
        close = float(frame.iloc[expansion_bar]["Close"])
        support = levels.get("major_support")
        resistance = levels.get("major_resistance")
        atr = self._atr(frame, expansion_bar)

        near_support = support is not None and abs(close - support) <= atr
        near_resistance = resistance is not None and abs(resistance - close) <= atr

        if direction == "bullish" and near_support:
            return "support_bounce"
        if direction == "bearish" and near_resistance:
            return "support_bounce"

        if structure["bos_count"] >= 1 and structure["consolidation_duration_bars"] >= 10:
            return "breakout"

        if (
            structure["choch_count"] >= 1
            and liquidity["liquidity_grabs"] >= 1
            and (liquidity["failed_breakouts"] + liquidity["failed_breakdowns"]) >= 1
        ):
            return "reversal"

        return "expansion"

    def _build_pattern_key(
        self,
        direction: str,
        liquidity: dict[str, int],
        structure: dict[str, Any],
        candle: CandleStructureMetrics,
        confirmation: dict[str, Any],
        trap: dict[str, Any],
        category: str,
    ) -> str:
        sweep_label = "No Sweep"
        if liquidity["sell_side_grabs"] and liquidity["buy_side_grabs"]:
            sweep_label = "Both Sweeps"
        elif liquidity["sell_side_grabs"]:
            sweep_label = "Sell-Side Grab"
        elif liquidity["buy_side_grabs"]:
            sweep_label = "Buy-Side Grab"

        consolidation = "Tight" if structure["consolidation_duration_bars"] >= 20 else "Loose"
        trap_label = "High Trap" if trap["total_traps"] >= 3 else "Low Trap"
        confirm_label = confirmation["displacement_strength"]

        dominant_candle = "None"
        candle_counts = {
            "Hammer": candle.hammer_count,
            "Engulfing": candle.bullish_engulfing_count + candle.bearish_engulfing_count,
            "Marubozu": candle.bullish_marubozu_count + candle.bearish_marubozu_count,
            "Star": candle.morning_star_count + candle.evening_star_count,
        }
        if candle_counts:
            dominant_candle = max(candle_counts, key=candle_counts.get)

        return (
            f"{category.title()} | {direction.title()} | Sweep:{sweep_label} | "
            f"Consolidation:{consolidation} | Trap:{trap_label} | "
            f"Confirm:{confirm_label} | Candle:{dominant_candle} | "
            f"CHOCH:{structure['choch_count']} BOS:{structure['bos_count']}"
        )

    def _analyze_move(
        self,
        frame: pd.DataFrame,
        candidate: _CheapMoveCandidate,
        timeframe_label: str,
        threshold: int,
    ) -> PreExpansionAnalysis:
        expansion_bar = candidate.expansion_bar
        origin_bar = candidate.start_bar
        start_bar = max(0, expansion_bar - PRE_EXPANSION_LOOKBACK)
        end_bar = max(start_bar, expansion_bar - 1)

        levels = self._market_levels(frame, expansion_bar)
        atr = self._atr(frame, expansion_bar)
        liquidity = self._liquidity_behavior(frame, start_bar, end_bar, levels, atr)
        sr = self._support_resistance_behavior(
            frame,
            start_bar,
            end_bar,
            expansion_bar,
            candidate.direction,
            levels,
            atr,
        )
        candle_window = frame.iloc[start_bar : end_bar + 1]
        candle = self._count_candle_patterns(candle_window)
        confirmation = self._confirmation_candle(frame, expansion_bar, candidate.direction)
        trap = self._trap_analysis(frame, start_bar, end_bar)
        structure = self._market_structure(frame, start_bar, end_bar, origin_bar)
        gaps = self._gap_analysis(frame, start_bar, end_bar, expansion_bar, candidate.direction, levels)
        category = self._classify_pattern_category(
            candidate.direction,
            liquidity,
            structure,
            sr,
            levels,
            frame,
            expansion_bar,
        )
        pattern_key = self._build_pattern_key(
            candidate.direction,
            liquidity,
            structure,
            candle,
            confirmation,
            trap,
            category,
        )

        return PreExpansionAnalysis(
            timeframe=timeframe_label,
            threshold_points=threshold,
            direction=candidate.direction,
            origin_bar=origin_bar,
            expansion_bar=expansion_bar,
            origin_timestamp=str(frame.iloc[origin_bar]["Date"]),
            expansion_timestamp=str(frame.iloc[expansion_bar]["Date"]),
            move_magnitude_points=candidate.magnitude,
            liquidity_grabs=liquidity["liquidity_grabs"],
            buy_side_grabs=liquidity["buy_side_grabs"],
            sell_side_grabs=liquidity["sell_side_grabs"],
            failed_breakouts=liquidity["failed_breakouts"],
            failed_breakdowns=liquidity["failed_breakdowns"],
            liquidity_taken_before_expansion=liquidity["liquidity_taken_before_expansion"],
            support_resistance_tests=sr["support_resistance_tests"],
            bars_near_level=sr["bars_near_level"],
            distance_from_nearest_level=sr["distance_from_nearest_level"],
            break_success_rate_pct=sr["break_success_rate_pct"],
            false_break_count=sr["false_break_count"],
            candle_structure=candle.as_dict(),
            confirmation_candle=confirmation,
            trap_analysis=trap,
            market_structure=structure,
            gap_analysis=gaps,
            pattern_key=pattern_key,
            pattern_category=category,
        )

    def _detect_moves_for_threshold(
        self,
        frame: pd.DataFrame,
        timeframe_label: str,
        threshold: int,
    ) -> list[_CheapMoveCandidate]:
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        candidates = self.move_engine._detect_moves_cheap(highs, lows, threshold)
        return self.move_engine._dedupe_cheap_moves(candidates)

    @staticmethod
    def _aggregate_profile(records: list[PreExpansionAnalysis]) -> dict[str, Any]:
        if not records:
            return {}

        def avg(values: list[float]) -> float:
            return round(mean(values), 2) if values else 0.0

        return {
            "sample_count": len(records),
            "liquidity_behavior": {
                "average_liquidity_grabs": avg([float(item.liquidity_grabs) for item in records]),
                "average_buy_side_grabs": avg([float(item.buy_side_grabs) for item in records]),
                "average_sell_side_grabs": avg([float(item.sell_side_grabs) for item in records]),
                "average_failed_breakouts": avg([float(item.failed_breakouts) for item in records]),
                "average_failed_breakdowns": avg([float(item.failed_breakdowns) for item in records]),
            },
            "support_resistance_behavior": {
                "average_tests": avg([float(item.support_resistance_tests) for item in records]),
                "average_bars_near_level": avg([float(item.bars_near_level) for item in records]),
                "average_distance_from_level": avg(
                    [
                        float(item.distance_from_nearest_level)
                        for item in records
                        if item.distance_from_nearest_level is not None
                    ],
                ),
                "average_break_success_rate_pct": avg([item.break_success_rate_pct for item in records]),
                "average_false_break_count": avg([float(item.false_break_count) for item in records]),
            },
            "candle_structure": {
                "average_largest_wick": avg([item.candle_structure["largest_wick_points"] for item in records]),
                "average_wick_size": avg([item.candle_structure["average_wick_points"] for item in records]),
                "average_largest_body": avg([item.candle_structure["largest_body_points"] for item in records]),
                "average_body_size": avg([item.candle_structure["average_body_points"] for item in records]),
                "average_close_location_pct": avg(
                    [item.candle_structure["average_close_location_pct"] for item in records],
                ),
            },
            "confirmation_candle": {
                "average_body_pct": avg([item.confirmation_candle["body_pct"] for item in records]),
                "average_wick_pct": avg([item.confirmation_candle["wick_pct"] for item in records]),
                "average_volume_expansion": avg(
                    [item.confirmation_candle["volume_expansion_ratio"] for item in records],
                ),
                "average_close_strength_pct": avg(
                    [item.confirmation_candle["close_strength_pct"] for item in records],
                ),
                "follow_through_success_rate_pct": round(
                    sum(1 for item in records if item.confirmation_candle["follow_through_success"])
                    / len(records)
                    * 100,
                    2,
                ),
            },
            "trap_analysis": {
                "average_false_upside_breaks": avg(
                    [float(item.trap_analysis["false_upside_breaks"]) for item in records],
                ),
                "average_false_downside_breaks": avg(
                    [float(item.trap_analysis["false_downside_breaks"]) for item in records],
                ),
                "average_trap_size": avg([item.trap_analysis["average_trap_size_points"] for item in records]),
                "average_max_trap_size": avg([item.trap_analysis["maximum_trap_size_points"] for item in records]),
            },
            "market_structure": {
                "average_consolidation_duration_bars": avg(
                    [float(item.market_structure["consolidation_duration_bars"]) for item in records],
                ),
                "average_range_width_points": avg(
                    [item.market_structure["range_width_points"] for item in records],
                ),
                "average_choch_count": avg([float(item.market_structure["choch_count"]) for item in records]),
                "average_bos_count": avg([float(item.market_structure["bos_count"]) for item in records]),
                "average_sweep_count": avg(
                    [float(item.market_structure["liquidity_sweep_count"]) for item in records],
                ),
            },
            "gap_analysis": {
                "average_gap_up_count": avg([float(item.gap_analysis["gap_up_count"]) for item in records]),
                "average_gap_down_count": avg([float(item.gap_analysis["gap_down_count"]) for item in records]),
                "average_gap_size": avg([item.gap_analysis["average_gap_size_points"] for item in records]),
                "average_gap_success_rate_pct": avg([item.gap_analysis["gap_success_rate_pct"] for item in records]),
            },
        }

    def _rank_patterns(
        self,
        records: list[PreExpansionAnalysis],
    ) -> list[PatternRankMetrics]:
        grouped: dict[str, list[PreExpansionAnalysis]] = defaultdict(list)
        for record in records:
            grouped[record.pattern_key].append(record)

        total = len(records)
        metrics: list[PatternRankMetrics] = []
        for pattern, bucket in grouped.items():
            avg_magnitude = mean(item.move_magnitude_points for item in bucket)
            avg_threshold = mean(float(item.threshold_points) for item in bucket)
            reliability = (
                (len(bucket) / total)
                * (avg_magnitude / avg_threshold if avg_threshold else 1.0)
                * 100
            )
            metrics.append(
                PatternRankMetrics(
                    pattern=pattern,
                    category=bucket[0].pattern_category,
                    sample_count=len(bucket),
                    frequency_pct=round((len(bucket) / total) * 100, 2) if total else 0.0,
                    average_move_magnitude=round(avg_magnitude, 2),
                    average_threshold_points=round(avg_threshold, 2),
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

    @staticmethod
    def _best_in_category(
        ranked: list[PatternRankMetrics],
        category: str,
        min_samples: int = MIN_PATTERN_SAMPLES,
    ) -> str:
        filtered = [
            item
            for item in ranked
            if item.category == category and item.sample_count >= min_samples
        ]
        if not filtered:
            fallback = [item for item in ranked if item.category == category]
            filtered = fallback or ranked
        best = max(filtered, key=lambda item: (item.reliability_score, item.sample_count))
        return best.pattern

    def _collect_records(self, metadata: dict[str, Any]) -> list[PreExpansionAnalysis]:
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        records: list[PreExpansionAnalysis] = []
        analysis_cache: dict[tuple[str, int, str], PreExpansionAnalysis] = {}

        for timeframe_label in self.timeframes:
            path = self.filter_engine._ensure_pipeline(timeframe_label, start, end)
            frame = pd.read_csv(path).reset_index(drop=True)
            candidates = self._detect_moves_for_threshold(
                frame,
                timeframe_label,
                MOVE_THRESHOLDS[0],
            )
            logger.info(
                "Momentum origin: %s candidates=%s",
                timeframe_label,
                len(candidates),
            )

            for candidate in candidates:
                if candidate.expansion_bar < PRE_EXPANSION_LOOKBACK:
                    continue

                cache_key = (timeframe_label, candidate.expansion_bar, candidate.direction)
                if cache_key not in analysis_cache:
                    analysis_cache[cache_key] = self._analyze_move(
                        frame,
                        candidate,
                        timeframe_label,
                        MOVE_THRESHOLDS[0],
                    )

                base = analysis_cache[cache_key]
                for threshold in MOVE_THRESHOLDS:
                    if candidate.magnitude >= threshold:
                        records.append(
                            replace(
                                base,
                                threshold_points=threshold,
                                move_magnitude_points=candidate.magnitude,
                            ),
                        )

        return records

    def run(self, metadata: dict[str, Any]) -> InstitutionalMomentumOriginReport:
        started = time.perf_counter()
        records = self._collect_records(metadata)

        by_threshold: dict[str, list[PreExpansionAnalysis]] = defaultdict(list)
        for record in records:
            by_threshold[str(record.threshold_points)].append(record)

        total_moves = {key: len(bucket) for key, bucket in by_threshold.items()}
        aggregate_profile = {
            threshold: self._aggregate_profile(bucket)
            for threshold, bucket in by_threshold.items()
        }

        ranked_patterns = self._rank_patterns(records)
        top_patterns = [
            item.as_dict()
            for item in ranked_patterns[:TOP_PATTERN_COUNT]
            if item.sample_count >= MIN_PATTERN_SAMPLES
        ]
        if len(top_patterns) < TOP_PATTERN_COUNT:
            top_patterns = [item.as_dict() for item in ranked_patterns[:TOP_PATTERN_COUNT]]

        category_rankings = {
            "most_reliable_expansion_pattern": self._best_in_category(ranked_patterns, "expansion"),
            "most_reliable_reversal_pattern": self._best_in_category(ranked_patterns, "reversal"),
            "most_reliable_breakout_pattern": self._best_in_category(ranked_patterns, "breakout"),
            "most_reliable_support_bounce_pattern": self._best_in_category(
                ranked_patterns,
                "support_bounce",
            ),
        }

        ranked_by_category: dict[str, list[dict[str, Any]]] = {}
        for category in ("expansion", "reversal", "breakout", "support_bounce"):
            bucket = [
                item.as_dict()
                for item in ranked_patterns
                if item.category == category and item.sample_count >= MIN_PATTERN_SAMPLES
            ]
            ranked_by_category[category] = bucket[:10]

        conclusions = [
            f"Analyzed {len(records)} expansion moves across thresholds {list(MOVE_THRESHOLDS)}.",
            f"Most common origin pattern: {top_patterns[0]['pattern'] if top_patterns else 'N/A'}.",
            f"Best expansion pattern: {category_rankings['most_reliable_expansion_pattern']}.",
            f"Best reversal pattern: {category_rankings['most_reliable_reversal_pattern']}.",
            f"Best breakout pattern: {category_rankings['most_reliable_breakout_pattern']}.",
            f"Best support bounce pattern: {category_rankings['most_reliable_support_bounce_pattern']}.",
        ]

        return InstitutionalMomentumOriginReport(
            symbol=metadata.get("symbol", self.symbol),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            timeframes_analyzed=list(self.timeframes),
            move_thresholds_points=list(MOVE_THRESHOLDS),
            pre_expansion_lookback_bars=PRE_EXPANSION_LOOKBACK,
            total_expansion_moves=total_moves,
            aggregate_pre_expansion_profile=aggregate_profile,
            top_20_momentum_origin_patterns=top_patterns,
            pattern_rankings=category_rankings,
            ranked_patterns_by_category=ranked_by_category,
            expansion_records=[item.as_dict() for item in records],
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_institutional_momentum_origin_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> InstitutionalMomentumOriginReport:
    """Run institutional momentum origin research and export JSON report."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise InstitutionalMomentumOriginError(
            f"Filter research report not found: {metadata_path}",
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = InstitutionalMomentumOriginResearch(
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
        "Institutional momentum origin research completed: moves=%s",
        sum(report.total_expansion_moves.values()),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_institutional_momentum_origin_report()
        print("Institutional Momentum Origin Research Summary")
        print(f"Total moves: {sum(report.total_expansion_moves.values())}")
        print("Moves by threshold:", report.total_expansion_moves)
        print("Top 5 patterns:")
        for item in report.top_20_momentum_origin_patterns[:5]:
            print(f"  [{item['sample_count']}] {item['pattern']}")
        print("Rankings:")
        for key, value in report.pattern_rankings.items():
            print(f"  {key}: {value}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except InstitutionalMomentumOriginError as exc:
        logger.error("Institutional momentum origin research error: %s", exc)
        print(f"Institutional momentum origin research error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected institutional momentum origin research failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
